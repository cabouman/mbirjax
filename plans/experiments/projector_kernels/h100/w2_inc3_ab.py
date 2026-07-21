"""Increment-3 gate: parallel multi-device band adoption, model-level A/B (2026-07-13).

The per-owner band probe (w2_band_ab.py, job 13505309) measured the shipped
register-tile kernel 8.9x on one owner's band work.  This gates the INTEGRATED path
(TilePolicy back_pallas_band + the per-owner dispatch in
ParallelBeamModel._back_project_view_shard_to_band): model-level walls, values
flag-on vs kill-switch, and per-device peak memory, at n=2 and n=4 on a 4-GPU node
(1024^3 cell).  VCD cells time a short recon end-to-end.

Cells: par_back_{n2,n4}_{off,on}, par_vcd_n2_{off,on}; each an isolated subprocess
(CUDA_VISIBLE_DEVICES pins n; MBIRJAX_DISABLE_PALLAS=1 is the off config).  The off
cell of each pair saves its result to scratch; the on cell loads and gates at
rel-max <= 1e-5 (back) / 1e-4 (iterated VCD).
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
OUT_DIR = '/scratch/gautschi/buzzard/w2_inc3_ab'
CELLS = []
for n in (2, 4):
    for cfg in ('off', 'on'):
        CELLS.append(dict(name=f'par_back_n{n}_{cfg}', n=n, op='back', on=cfg == 'on'))
for cfg in ('off', 'on'):
    CELLS.append(dict(name=f'par_vcd_n2_{cfg}', n=2, op='vcd', on=cfg == 'on'))

if os.environ.get('W2I3_SMOKE') == '1':
    SINO_SHAPE = (32, 16, 24)
    VCD_ITERS = 1
    OUT_DIR = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'w2_inc3_smoke')
    CELLS = [c for c in CELLS if c['n'] <= int(os.environ.get('W2I3_SMOKE_MAX_N', '1'))]


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
    sino = jnp.asarray(rng.random(SINO_SHAPE, dtype=np.float32))
    jax.block_until_ready(sino)

    if cfg['op'] == 'back':
        idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=True)
        for _ in range(WARMUP):
            jax.block_until_ready(model.sparse_back_project(sino, idx))
        ts = []
        for _ in range(TRIALS):
            t = time.perf_counter()
            r = jax.block_until_ready(model.sparse_back_project(sino, idx))
            ts.append(time.perf_counter() - t)
        result, tol = np.asarray(r), 1e-5
    else:
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
    stats = mbirjax.get_memory_stats(print_results=False)
    # Per-DEVICE peak only: the trailing 'CPU' entry is process RSS, which can exceed
    # small GPU peaks and masquerade as the device number (inc4 review finding).
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
        env = dict(os.environ, W2I3_CELL=json.dumps(cfg),
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
    print('===== inc3 A/B summary =====', flush=True)
    print('\n'.join(summary), flush=True)
    print('=== w2_inc3_ab done ===', flush=True)


if __name__ == '__main__':
    worker(json.loads(os.environ['W2I3_CELL'])) if os.environ.get('W2I3_CELL') \
        else orchestrator()
