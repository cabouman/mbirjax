"""Sweep the CONE forward projection tiling knobs (the parallel forward sweep's sibling).

Cone forward still runs entirely on inherited defaults (fwd_pixel_batch 2048, fwd_view_batch
128) and is now the largest projector cell (29.2 s at 1024^3 n=1 on H100).  Unlike parallel,
cone forward is NOT slice-banded: at n=1 it is one monolithic driver call (pixel scan at
fwd_pixel_batch); at n>=2 the gather path host-tiles pixels at fwd_pixel_batch (communication
granularity) around the same driver.  Both read tiles.fwd_pixel_batch, so the sweep is pure
TilePolicy overrides -- no library changes.

Each configuration runs in its OWN SUBPROCESS (clean peak memory, OOM-safe; large pixel
batches may genuinely OOM at 1024^3 -- that is data, not failure).  Timing is model-level
``sparse_forward_project`` on the full ROR-masked grid, input pre-sharded.  A float64 output
sum guards values against the default config.

Run:  python plans/experiments/projector_kernels/cone_fwd_tile_sweep.py   (edit constants below)
"""
import json
import os
import subprocess
import sys

# ── Sweep grid (edit here) ────────────────────────────────────────────────────
SINO_SHAPES = [(512, 448, 384), (1024, 1008, 992)]
# Stage 1 -- pixel-batch sweep at the default view batch.
STAGE1_PIXEL_BATCHES = [2048, 4096, 8192, 16384]
STAGE1_COUNTS = [1, 2, 4]
# Stage 2 -- view-batch spot checks at the promising pixel batches.
STAGE2_VIEW_BATCHES = [256]
STAGE2_PIXEL_BATCHES = [2048, 8192]
STAGE2_COUNTS = [1, 2]
WARMUP = 1
TRIALS_SMALL, TRIALS_1024 = 3, 1


def worker(cfg):
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.94')
    import time
    import numpy as np
    import mbirjax
    import jax

    sino_shape = tuple(cfg['sino_shape'])
    n = cfg['n_devices']
    out = dict(cfg)
    try:
        num_det_channels = sino_shape[2]
        angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)
        model = mbirjax.ConeBeamModel(sino_shape, angles,
                                      source_detector_dist=4.0 * num_det_channels,
                                      source_iso_dist=2.0 * num_det_channels)
        model.configure_devices(n)
        repl = {}
        if cfg.get('pixel_batch', 'default') != 'default':
            repl['fwd_pixel_batch'] = int(cfg['pixel_batch'])
        if cfg.get('view_batch', 'default') != 'default':
            repl['fwd_view_batch'] = int(cfg['view_batch'])
        if repl:
            model.tiles = model.tiles._replace(**repl)
        recon_shape = model.get_params('recon_shape')
        idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=model.get_params('use_ror_mask'))
        rng = np.random.default_rng(0)
        cylinders = model._shard_recon(
            rng.random((len(idx), recon_shape[2]), dtype=np.float32))
        jax.block_until_ready(cylinders)

        trials = TRIALS_1024 if sino_shape[0] >= 1024 else TRIALS_SMALL
        for _ in range(WARMUP):
            jax.block_until_ready(model.sparse_forward_project(cylinders, idx))
        times = []
        for _ in range(trials):
            t0 = time.perf_counter()
            result = jax.block_until_ready(model.sparse_forward_project(cylinders, idx))
            times.append(time.perf_counter() - t0)
        peak = max(int((d.memory_stats() or {}).get('peak_bytes_in_use', 0))
                   for d in model.sino_placement.devices)
        out.update(min_ms=1e3 * min(times), mem_mb=peak / 2 ** 20,
                   fp_sum=float(np.asarray(result, dtype=np.float64).sum()), status='ok')
    except Exception as e:   # noqa: BLE001
        msg = str(e)
        out.update(status='OOM' if 'RESOURCE_EXHAUSTED' in msg else 'ERROR', error=msg[:200])
    print('RESULT|' + json.dumps(out), flush=True)


def main():
    configs = []
    for sino_shape in SINO_SHAPES:
        for n in STAGE1_COUNTS:
            for pb in STAGE1_PIXEL_BATCHES:
                configs.append(dict(sino_shape=sino_shape, n_devices=n, pixel_batch=pb,
                                    view_batch='default', stage=1))
        for n in STAGE2_COUNTS:
            for vb in STAGE2_VIEW_BATCHES:
                for pb in STAGE2_PIXEL_BATCHES:
                    configs.append(dict(sino_shape=sino_shape, n_devices=n, pixel_batch=pb,
                                        view_batch=vb, stage=2))

    results = []
    for i, cfg in enumerate(configs):
        label = (f"{cfg['sino_shape'][0]}^3 n={cfg['n_devices']} P={cfg['pixel_batch']} "
                 f"vb={cfg['view_batch']}")
        print(f"[{i + 1}/{len(configs)}] {label}", flush=True)
        env = dict(os.environ, MBIRJAX_SWEEP_CFG=json.dumps(cfg))
        proc = subprocess.run([sys.executable, os.path.abspath(__file__)], env=env,
                              capture_output=True, text=True)
        line = next((l for l in proc.stdout.splitlines() if l.startswith('RESULT|')), None)
        if line is None:
            print(f"    NO RESULT (rc={proc.returncode}): {proc.stderr[-200:]}", flush=True)
            continue
        r = json.loads(line.split('|', 1)[1])
        results.append(r)
        if r['status'] == 'ok':
            print(f"    {r['min_ms']:9.1f} ms   {r['mem_mb']:9.1f} MB", flush=True)
        else:
            print(f"    {r['status']}: {r.get('error', '')[:120]}", flush=True)

    def key(r):
        return (tuple(r['sino_shape']), r['n_devices'])

    baseline = {key(r): r for r in results
                if r['status'] == 'ok' and r['pixel_batch'] == 2048
                and r['view_batch'] == 'default'}
    print('\n===== SUMMARY (model-level cone sparse_forward_project) =====')
    print(f"{'size':>7} {'n':>2} {'P':>6} {'vb':>8} {'min_ms':>10} {'mem_MB':>9} "
          f"{'vs_default':>10}  value_check")
    for r in results:
        if r['status'] != 'ok':
            print(f"{r['sino_shape'][0]:>6}^3 {r['n_devices']:>2} {r['pixel_batch']:>6} "
                  f"{str(r['view_batch']):>8} {r['status']:>10}")
            continue
        base = baseline.get(key(r))
        speed = f"{base['min_ms'] / r['min_ms']:9.2f}x" if base else '        ?'
        vchk = ''
        if base and base is not r:
            rel = abs(r['fp_sum'] - base['fp_sum']) / max(abs(base['fp_sum']), 1e-30)
            vchk = f"sum_rel={rel:.1e}" + ('  MISMATCH!' if rel > 1e-5 else '')
        print(f"{r['sino_shape'][0]:>6}^3 {r['n_devices']:>2} {r['pixel_batch']:>6} "
              f"{str(r['view_batch']):>8} {r['min_ms']:>10.1f} {r['mem_mb']:>9.1f} "
              f"{speed}  {vchk}")


if __name__ == '__main__':
    if 'MBIRJAX_SWEEP_CFG' in os.environ:
        worker(json.loads(os.environ['MBIRJAX_SWEEP_CFG']))
    else:
        main()
