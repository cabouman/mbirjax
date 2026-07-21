"""Increment-4 gate: parallel multi-device banded FORWARD adoption, model-level A/B (2026-07-13).

The fwd guard sweep's band-256 column (plans/projector_kernels/fwd_guard_sweep.md)
measured the pallas forward driver 1.7-3.8x at exactly the banded-forward width.  This
gates the INTEGRATED path (TilePolicy fwd_pallas_band + the per-owner dispatch in
ParallelBeamModel._forward_project_band_to_view_shard): model-level walls, values
flag-on vs kill-switch, and per-device peak memory, at n=2 and n=4 on a 4-GPU node
(1024^3 cell).  The forward cells drive the REAL user-facing path
(model.sparse_forward_project, which shards a plain recon cylinder internally and, at
n>=2, streams the banded per-owner forward -> the seam -> pallas).  VCD cells time a
short recon end-to-end (both the fwd AND back bands are pallas in the on cell, so the
VCD number is the combined recon speedup, not an isolated fwd measurement -- the par_fwd
cells isolate the forward op).

Cells: par_fwd_{n2,n4}_{off,on}, par_vcd_n2_{off,on}; each an isolated subprocess
(CUDA_VISIBLE_DEVICES pins n; MBIRJAX_DISABLE_PALLAS=1 is the off config).  The off
cell of each pair saves its result to scratch; the on cell loads and gates at
rel-max <= 1e-5 (single-shot forward) / 1e-4 (iterated VCD).
"""
import json
import os
import subprocess
import sys
import time

# ── Config ────────────────────────────────────────────────────────────────────
SINO_SHAPE = (1024, 1024, 1024)
WARMUP = 1
TRIALS = 3
VCD_ITERS = 4
OUT_DIR = '/scratch/gautschi/buzzard/w2_inc4_ab'
CELLS = []
for n in (2, 4):
    for cfg in ('off', 'on'):
        CELLS.append(dict(name=f'par_fwd_n{n}_{cfg}', n=n, op='fwd', on=cfg == 'on'))
for cfg in ('off', 'on'):
    CELLS.append(dict(name=f'par_vcd_n2_{cfg}', n=2, op='vcd', on=cfg == 'on'))

if os.environ.get('W2I4_SMOKE') == '1':
    SINO_SHAPE = (32, 16, 24)
    VCD_ITERS = 1
    OUT_DIR = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'w2_inc4_smoke')
    CELLS = [c for c in CELLS if c['n'] <= int(os.environ.get('W2I4_SMOKE_MAX_N', '1'))]


def worker(cfg):
    import numpy as np
    import jax
    import jax.numpy as jnp
    import mbirjax

    t0 = time.perf_counter()

    def note(msg):
        print(f'[{time.perf_counter() - t0:8.2f}s] {msg}', flush=True)

    views, rows, channels = SINO_SHAPE
    angles = np.linspace(-np.pi / 2, np.pi / 2, views, endpoint=False)
    model = mbirjax.ParallelBeamModel(SINO_SHAPE, angles)
    note(f'summary={model.device_summary}')
    recon_shape = model.get_params('recon_shape')
    rng = np.random.default_rng(0)

    if cfg['op'] == 'fwd':
        # The real user-facing forward at n>=2: sparse_forward_project shards a plain
        # recon cylinder internally, then streams the banded per-owner forward ->
        # _forward_project_band_to_view_shard (pallas when fwd_pallas_band, XLA otherwise).
        idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=True)
        recon = jnp.asarray(rng.random((len(idx), recon_shape[2]), dtype=np.float32))
        jax.block_until_ready(recon)
        for _ in range(WARMUP):
            jax.block_until_ready(model.sparse_forward_project(recon, idx))
        ts = []
        for _ in range(TRIALS):
            t = time.perf_counter()
            r = jax.block_until_ready(model.sparse_forward_project(recon, idx))
            ts.append(time.perf_counter() - t)
        result, tol = np.asarray(r), 1e-5
    else:
        sino = jnp.asarray(rng.random(SINO_SHAPE, dtype=np.float32))
        jax.block_until_ready(sino)
        model.set_params(verbose=0)
        # Identical partition subsets in the off and on cells (the load-bearing VCD
        # reproducibility gotcha: partitions draw from the numpy global RNG, which
        # seeds from OS entropy per process -- unseeded off/on cells would compare
        # DIFFERENT recon trajectories and fail the value gate spuriously).
        np.random.seed(0)
        t = time.perf_counter()
        out, _ = model.recon(sino, max_iterations=VCD_ITERS)
        ts = [time.perf_counter() - t]
        result, tol = np.asarray(out), 1e-4
    note(f'RESULT wall={sorted(ts)[len(ts) // 2]:.3f}s trials={["%.3f" % x for x in ts]}')

    ref_path = os.path.join(OUT_DIR, f"{cfg['name'].rsplit('_', 1)[0]}_ref.npy")
    if not cfg['on']:
        np.save(ref_path, result)
        note('RESULT ref=saved')
    else:
        ref = np.load(ref_path)
        rel = float(np.max(np.abs(result - ref)) / np.max(np.abs(ref)))
        note(f'RESULT rel={rel:.2e} {"PASS" if rel < tol else "FAIL"} (tol {tol:g})')
    # Per-DEVICE peak: get_memory_stats appends a trailing 'CPU' entry whose
    # peak_bytes_in_use is the process RSS (memory_stats.py), which at n=4 (small GPU
    # shards, but the host holds the full gathered sinogram + recon input) can dwarf the
    # real GPU peak -- filter to GPU entries so the reported number is the device peak.
    stats = mbirjax.get_memory_stats(print_results=False)
    gpu_peaks = [s['peak_bytes_in_use'] for s in stats if s['id'].startswith('GPU')]
    peak = max(gpu_peaks) / 2**30 if gpu_peaks else 0.0
    note(f'RESULT peak_gb={peak:.2f}')


def orchestrator():
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in os.listdir(OUT_DIR):
        if f.endswith('_ref.npy'):
            os.remove(os.path.join(OUT_DIR, f))
    summary = []
    for cfg in CELLS:
        env = dict(os.environ, W2I4_CELL=json.dumps(cfg),
                   CUDA_VISIBLE_DEVICES=','.join(str(i) for i in range(cfg['n'])))
        if not cfg['on']:
            env['MBIRJAX_DISABLE_PALLAS'] = '1'
        log_path = os.path.join(OUT_DIR, f"{cfg['name']}.log")
        with open(log_path, 'w') as log:
            rc = subprocess.run([sys.executable, os.path.abspath(__file__)],
                                env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        line = f"[{cfg['name']}] rc={rc}"
        with open(log_path) as log:
            for row in log:
                if 'RESULT ' in row or 'summary=' in row:
                    line += '\n    ' + row.strip()
        print(line, flush=True)
        summary.append(line)
    print('===== inc4 A/B summary =====', flush=True)
    print('\n'.join(summary), flush=True)
    print('=== w2_inc4_ab done ===', flush=True)


if __name__ == '__main__':
    worker(json.loads(os.environ['W2I4_CELL'])) if os.environ.get('W2I4_CELL') \
        else orchestrator()
