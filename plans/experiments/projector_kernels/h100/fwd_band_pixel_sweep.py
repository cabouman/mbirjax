"""Sweep the parallel-beam FORWARD projection band length and pixel batch size.

Two follow-ups from the Phase A kernel work (see fwd_back_findings.md):

  * BAND LENGTH: forward bands inherit back's memory-driven sizing, but forward's per-band
    transient is small, and every band repeats the kernel's per-call fixed costs (geometry
    compute + the sorted reduce's sort).  The existing ``forward_project_slice_band`` hook
    overrides the band length with no code change ('shard' = one band per slice-shard).
  * PIXEL BATCH: the sorted reduce's sort is ~constant-latency per call (a 6144-key sort
    underutilizes an H100), and fewer pixel-scan steps mean less traffic through the
    (V, B, C) scan carry.  ``pixel_batch_size_for_vmap`` is late-bound, so it is also a
    plain attribute set.

Each configuration runs in its OWN SUBPROCESS (fresh jax, clean peak_bytes_in_use, an OOM
cannot contaminate later configs).  Timing is MODEL-LEVEL (``model.sparse_forward_project``
on the full ROR-masked pixel grid, input pre-sharded), matching what the nightly measures.
A float64 output sum per config is compared against the default-config sum for the same
(size, n) as a value guard (relative; reordering-level differences only).

Run:  python plans/experiments/projector_kernels/fwd_band_pixel_sweep.py    (edit constants below)
"""
import json
import os
import subprocess
import sys

# ── Sweep grid (edit here) ────────────────────────────────────────────────────
SINO_SHAPES = [(513, 449, 385), (1024, 1008, 992)]
DEVICE_COUNTS = [1, 2, 4]
# Stage 1 -- band sweep at the default pixel batch (2048).
STAGE1_BANDS = ['default', 128, 256, 'shard']
# Stage 2 -- pixel-batch sweep at default vs whole-shard bands, n=1 and 2.
STAGE2_PIXEL_BATCHES = [4096, 8192, 16384]
STAGE2_BANDS = ['default', 'shard']
STAGE2_COUNTS = [1, 2]
WARMUP = 1
TRIALS_SMALL, TRIALS_1024 = 3, 1


def worker(cfg):
    """One configuration: build, time model-level forward, print a RESULT line."""
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.94')
    import time
    import numpy as np
    import mbirjax
    import jax
    import jax.numpy as jnp

    sino_shape = tuple(cfg['sino_shape'])
    n = cfg['n_devices']
    out = dict(cfg)
    try:
        angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)
        model = mbirjax.ParallelBeamModel(sino_shape, angles)
        model.configure_devices(n)
        # Overrides via the tile policy (post-TilePolicy libraries) with a fallback to the old
        # attribute for pre-policy libraries.  NOTE: on post-policy libraries, 'default' now
        # means the library's OWN policy (on GPU: band 256 / pixel 8192 / sorted reduce).
        if hasattr(model, 'tiles') and model.tiles is not None:
            model.tiles = model.tiles._replace(fwd_pixel_batch=int(cfg['pixel_batch']))
        else:
            model.pixel_batch_size_for_vmap = int(cfg['pixel_batch'])
        if cfg['band'] == 'shard':
            model.forward_project_slice_band = 10 ** 9      # clipped to slices_per_dev
        elif cfg['band'] != 'default':
            model.forward_project_slice_band = int(cfg['band'])
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
                   for d in model.sino_placement.devices)   # memory_stats is None on CPU
        out.update(min_ms=1e3 * min(times), mem_mb=peak / 2 ** 20,
                   fp_sum=float(np.asarray(result, dtype=np.float64).sum()), status='ok')
    except Exception as e:   # noqa: BLE001 -- classify and report, never crash the sweep
        msg = str(e)
        out.update(status='OOM' if 'RESOURCE_EXHAUSTED' in msg else 'ERROR', error=msg[:200])
    print('RESULT|' + json.dumps(out), flush=True)


def main():
    configs = []
    for sino_shape in SINO_SHAPES:
        for n in DEVICE_COUNTS:
            for band in STAGE1_BANDS:
                configs.append(dict(sino_shape=sino_shape, n_devices=n, band=band,
                                    pixel_batch=2048, stage=1))
        for n in STAGE2_COUNTS:
            for band in STAGE2_BANDS:
                for pb in STAGE2_PIXEL_BATCHES:
                    configs.append(dict(sino_shape=sino_shape, n_devices=n, band=band,
                                        pixel_batch=pb, stage=2))

    results = []
    for i, cfg in enumerate(configs):
        label = (f"{cfg['sino_shape'][0]}^3 n={cfg['n_devices']} band={cfg['band']} "
                 f"P={cfg['pixel_batch']}")
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

    # ── Summary table + value guard against the default config per (size, n) ──
    def key(r):
        return (tuple(r['sino_shape']), r['n_devices'])

    baseline = {key(r): r for r in results
                if r['status'] == 'ok' and r['band'] == 'default' and r['pixel_batch'] == 2048}
    print('\n===== SUMMARY (model-level sparse_forward_project) =====')
    print(f"{'size':>7} {'n':>2} {'band':>8} {'P':>6} {'min_ms':>10} {'mem_MB':>9} "
          f"{'vs_default':>10}  value_check")
    for r in results:
        if r['status'] != 'ok':
            print(f"{r['sino_shape'][0]:>6}^3 {r['n_devices']:>2} {str(r['band']):>8} "
                  f"{r['pixel_batch']:>6} {r['status']:>10}")
            continue
        base = baseline.get(key(r))
        speed = f"{base['min_ms'] / r['min_ms']:9.2f}x" if base else '        ?'
        vchk = ''
        if base and base is not r:
            rel = abs(r['fp_sum'] - base['fp_sum']) / max(abs(base['fp_sum']), 1e-30)
            vchk = f"sum_rel={rel:.1e}" + ('  MISMATCH!' if rel > 1e-5 else '')
        print(f"{r['sino_shape'][0]:>6}^3 {r['n_devices']:>2} {str(r['band']):>8} "
              f"{r['pixel_batch']:>6} {r['min_ms']:>10.1f} {r['mem_mb']:>9.1f} {speed}  {vchk}")


if __name__ == '__main__':
    if 'MBIRJAX_SWEEP_CFG' in os.environ:
        worker(json.loads(os.environ['MBIRJAX_SWEEP_CFG']))
    else:
        main()
