"""Sweep the parallel-beam BACK projection tiling knobs (the forward sweep's sibling).

Back is the long pole now (H100: 18.3 s vs forward's 8.2 s at 1024^3 n=1).  Its knobs have
never been swept; all are TilePolicy fields, so the sweep needs no library changes:

  * n=1 (GPU monolithic short-circuit): ``back_view_batch`` sets BOTH the host view-transfer
    slice and the driver's view vmap width; ``back_pixel_batch`` sets the driver's pixel
    concatenation tile (and 100x it, the host pixel-transfer slice).
  * n>=2 (banded reduce-scatter): ``back_slice_band`` overrides the memory-driven band
    formula ('shard' = one band per slice-shard); ``back_pixel_batch`` as above; the view
    batch already single-vmaps the shard by policy.

Each configuration runs in its OWN SUBPROCESS (clean peak memory, OOM-safe).  Timing is
model-level ``sparse_back_project`` on the full ROR-masked grid with a seeded random
sinogram, pre-sharded.  A float64 output sum guards values against the default config.

Run:  python plans/experiments/projector_kernels/back_tile_sweep.py    (edit constants below)
"""
import json
import os
import subprocess
import sys

# ── Sweep grid (edit here) ────────────────────────────────────────────────────
SINO_SHAPES = [(513, 449, 385), (1024, 1008, 992)]
# Stage 1 -- n=1 (monolithic path): view batch x pixel batch.
STAGE1_VIEW_BATCHES = ['default', 64, 256, 512]
STAGE1_PIXEL_BATCHES = [2048, 8192]
# Stage 2 -- n in {2, 4} (banded path): slice band x pixel batch.
STAGE2_BANDS = ['default', 128, 256, 'shard']
STAGE2_PIXEL_BATCHES = [2048, 8192]
STAGE2_COUNTS = [2, 4]
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
        angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)
        model = mbirjax.ParallelBeamModel(sino_shape, angles)
        model.configure_devices(n)
        repl = {}
        if cfg.get('view_batch', 'default') != 'default':
            repl['back_view_batch'] = int(cfg['view_batch'])
        if cfg.get('pixel_batch', 'default') != 'default':
            repl['back_pixel_batch'] = int(cfg['pixel_batch'])
        if repl:
            model.tiles = model.tiles._replace(**repl)
        if cfg.get('band', 'default') == 'shard':
            model.back_project_slice_band = 10 ** 9          # clipped to slices_per_dev
        elif cfg.get('band', 'default') != 'default':
            model.back_project_slice_band = int(cfg['band'])
        recon_shape = model.get_params('recon_shape')
        idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=model.get_params('use_ror_mask'))
        rng = np.random.default_rng(0)
        sino = model._shard_sinogram(
            np.asarray(rng.random(sino_shape, dtype=np.float32)))
        jax.block_until_ready(sino)

        trials = TRIALS_1024 if sino_shape[0] >= 1024 else TRIALS_SMALL
        for _ in range(WARMUP):
            jax.block_until_ready(model.sparse_back_project(sino, idx))
        times = []
        for _ in range(trials):
            t0 = time.perf_counter()
            result = jax.block_until_ready(model.sparse_back_project(sino, idx))
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
        for vb in STAGE1_VIEW_BATCHES:
            for pb in STAGE1_PIXEL_BATCHES:
                configs.append(dict(sino_shape=sino_shape, n_devices=1, view_batch=vb,
                                    pixel_batch=pb, band='default', stage=1))
        for n in STAGE2_COUNTS:
            for band in STAGE2_BANDS:
                for pb in STAGE2_PIXEL_BATCHES:
                    configs.append(dict(sino_shape=sino_shape, n_devices=n, view_batch='default',
                                        pixel_batch=pb, band=band, stage=2))

    results = []
    for i, cfg in enumerate(configs):
        label = (f"{cfg['sino_shape'][0]}^3 n={cfg['n_devices']} vb={cfg['view_batch']} "
                 f"band={cfg['band']} P={cfg['pixel_batch']}")
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

    def is_default(r):
        return (r['view_batch'] == 'default' and r['band'] == 'default'
                and r['pixel_batch'] == 2048)

    baseline = {key(r): r for r in results if r['status'] == 'ok' and is_default(r)}
    print('\n===== SUMMARY (model-level sparse_back_project) =====')
    print(f"{'size':>7} {'n':>2} {'vb':>8} {'band':>8} {'P':>6} {'min_ms':>10} {'mem_MB':>9} "
          f"{'vs_default':>10}  value_check")
    for r in results:
        if r['status'] != 'ok':
            print(f"{r['sino_shape'][0]:>6}^3 {r['n_devices']:>2} {str(r['view_batch']):>8} "
                  f"{str(r['band']):>8} {r['pixel_batch']:>6} {r['status']:>10}")
            continue
        base = baseline.get(key(r))
        speed = f"{base['min_ms'] / r['min_ms']:9.2f}x" if base else '        ?'
        vchk = ''
        if base and base is not r:
            rel = abs(r['fp_sum'] - base['fp_sum']) / max(abs(base['fp_sum']), 1e-30)
            vchk = f"sum_rel={rel:.1e}" + ('  MISMATCH!' if rel > 1e-5 else '')
        print(f"{r['sino_shape'][0]:>6}^3 {r['n_devices']:>2} {str(r['view_batch']):>8} "
              f"{str(r['band']):>8} {r['pixel_batch']:>6} {r['min_ms']:>10.1f} "
              f"{r['mem_mb']:>9.1f} {speed}  {vchk}")


if __name__ == '__main__':
    if 'MBIRJAX_SWEEP_CFG' in os.environ:
        worker(json.loads(os.environ['MBIRJAX_SWEEP_CFG']))
    else:
        main()
