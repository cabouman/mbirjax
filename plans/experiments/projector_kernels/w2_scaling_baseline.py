"""Wave-2 baseline: the multi-device scaling curve on the CURRENT branch (2026-07-13).

The (c) target is multi-GPU TIME scaling (cone back historically ANTI-scales at n=2;
June ncu blamed the band kernel's internal transpose).  All of that data predates the
campaign kernels and the pallas paths, so wave 2 opens by re-measuring the curve the
band-kernel work is justified by (plan section E3b/E4: "the scaling curve IS the
deliverable").

Cells (isolated subprocesses on a 4-GPU node; device count pinned per cell via
CUDA_VISIBLE_DEVICES so each cell sees exactly n GPUs):
  cone     x n in {1,2,4} x {back_full, vcd}   -- the (c) case, no pallas anywhere
  parallel x n in {1,2,4} x {back_full}        -- band path at n>=2; pallas at n=1
  parallel n=1 back_full with MBIRJAX_DISABLE_PALLAS=1 -- the XLA n=1 reference, so
      the scaling table can be read against both shipped and pre-pallas baselines.

Shapes: the 1024^3 cell (1024 views x 1024 rows x 1024 channels).  back_full = full
sinogram -> full masked pixel grid, median of TRIALS after WARMUP.  vcd = a short
recon (VCD_ITERS iterations, default partition sequence) -- wall only, values not
compared across n (device-count independence is covered by the test suite).
Outputs: per-cell logs + a summary table on stdout; big arrays are never saved.
"""
import os
import subprocess
import sys
import time

# ── Run parameters (edit here; no CLI args) ───────────────────────────────────
SINO_SHAPE = (1024, 1024, 1024)
WARMUP = 1
TRIALS = 3
VCD_ITERS = 4
OUT_DIR = '/scratch/gautschi/buzzard/w2_scaling'
CELLS = [
    # (name, geometry, n_gpus, op, extra_env)
    ('cone_back_n1', 'cone', 1, 'back_full', {}),
    ('cone_back_n2', 'cone', 2, 'back_full', {}),
    ('cone_back_n4', 'cone', 4, 'back_full', {}),
    ('cone_vcd_n1',  'cone', 1, 'vcd', {}),
    ('cone_vcd_n2',  'cone', 2, 'vcd', {}),
    ('cone_vcd_n4',  'cone', 4, 'vcd', {}),
    ('par_back_n1',  'parallel', 1, 'back_full', {}),
    ('par_back_n1_xla', 'parallel', 1, 'back_full', {'MBIRJAX_DISABLE_PALLAS': '1'}),
    ('par_back_n2',  'parallel', 2, 'back_full', {}),
    ('par_back_n4',  'parallel', 4, 'back_full', {}),
]

if os.environ.get('W2_SMOKE') == '1':
    SINO_SHAPE = (32, 16, 24)
    VCD_ITERS = 1
    OUT_DIR = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'w2_scaling_smoke')
    CELLS = [c for c in CELLS if c[2] <= int(os.environ.get('W2_SMOKE_MAX_N', '1'))]


def worker():
    t0 = time.perf_counter()

    def note(msg):
        print(f'[{time.perf_counter() - t0:8.2f}s] {msg}', flush=True)

    import jax
    import jax.numpy as jnp
    import numpy as np
    import mbirjax

    geometry = os.environ['W2_GEOM']
    op = os.environ['W2_OP']
    views, rows, channels = SINO_SHAPE
    angles = np.linspace(-np.pi / 2, np.pi / 2, views, endpoint=False)
    if geometry == 'cone':
        model = mbirjax.ConeBeamModel(SINO_SHAPE, angles,
                                      source_detector_dist=4 * channels,
                                      source_iso_dist=4 * channels)
    else:
        model = mbirjax.ParallelBeamModel(SINO_SHAPE, angles)
    note(f'devices={jax.devices()}')
    note(f'summary={model.device_summary}')
    recon_shape = model.get_params('recon_shape')

    rng = np.random.default_rng(0)
    sino = jnp.asarray(rng.random(SINO_SHAPE, dtype=np.float32))
    jax.block_until_ready(sino)

    if op == 'back_full':
        idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=True)
        for _ in range(WARMUP):
            jax.block_until_ready(model.sparse_back_project(sino, idx))
        ts = []
        for _ in range(TRIALS):
            t = time.perf_counter()
            jax.block_until_ready(model.sparse_back_project(sino, idx))
            ts.append(time.perf_counter() - t)
        note(f'RESULT back_full wall={sorted(ts)[len(ts) // 2]:.3f}s '
             f'(trials {["%.3f" % x for x in ts]})')
    else:                                                    # vcd
        model.set_params(verbose=0)
        t = time.perf_counter()
        out, _ = model.recon(sino, max_iterations=VCD_ITERS)
        jax.block_until_ready(out)
        note(f'RESULT vcd wall={time.perf_counter() - t:.3f}s ({VCD_ITERS} iters, '
             f'incl. compile)')
    stats = mbirjax.get_memory_stats(print_results=False)
    peak = max(s['peak_bytes_in_use'] for s in stats) / 2**30 if stats else 0.0
    note(f'RESULT peak_gb={peak:.2f}')


def orchestrator():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = []
    for name, geom, n, op, extra in CELLS:
        env = dict(os.environ, W2_CELL='1', W2_GEOM=geom, W2_OP=op,
                   CUDA_VISIBLE_DEVICES=','.join(str(i) for i in range(n)), **extra)
        log_path = os.path.join(OUT_DIR, f'{name}.log')
        t = time.perf_counter()
        with open(log_path, 'w') as log:
            rc = subprocess.run([sys.executable, os.path.abspath(__file__)],
                                env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        wall = time.perf_counter() - t
        line = f'[{name}] rc={rc} cell_wall={wall:.1f}s'
        with open(log_path) as log:
            for row in log:
                if 'RESULT ' in row or 'summary=' in row:
                    line += '\n    ' + row.strip()
        print(line, flush=True)
        summary.append(line)
    print('===== w2 scaling summary =====', flush=True)
    print('\n'.join(summary), flush=True)
    print('=== w2_scaling done ===', flush=True)


if __name__ == '__main__':
    worker() if os.environ.get('W2_CELL') else orchestrator()
