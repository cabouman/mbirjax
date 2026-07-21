"""Increment-5 gate: the CONE fused-vfan back kernel, integrated, model-level A/B.

E5 spiked 9.4x on the per-owner band sweep (2.97 vs 27.8 s).  This gates the
INTEGRATED path (cone back_pallas at n=1 through the geometry hook; back_pallas_band
at n>=2 through the cone band override): model-level walls, values flag-on vs
kill-switch, per-GPU peaks, at n=1/2/4 (1024^3 cone cell).  The seeded VCD cell is
the empirical trajectory check backing the Hessian-gate decision (a): rel <= 1e-4
after 4 iterations with BOTH gradient and Hessian through the fused kernel.
Scaling anchors (w2_scaling_baseline): cone back n=1 18.9 s / n=2 27.5 (ANTI) /
n=4 14.4; cone VCD n=2 276 s.

Cells: cone_back_{n2,n4}_{off,on}, cone_vcd_n2_{off,on}; each an isolated subprocess
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
OUT_DIR = '/scratch/gautschi/buzzard/w2_inc5_ab'
CELLS = []
for n in (1, 2, 4):
    for cfg in ('off', 'on'):
        CELLS.append(dict(name=f'cone_back_n{n}_{cfg}', n=n, op='back', on=cfg == 'on'))
for cfg in ('off', 'on'):
    CELLS.append(dict(name=f'cone_hess_n1_{cfg}', n=1, op='hess', on=cfg == 'on'))
for cfg in ('off', 'on'):
    CELLS.append(dict(name=f'cone_vcd_n2_{cfg}', n=2, op='vcd', on=cfg == 'on'))

if os.environ.get('W2I5_SMOKE') == '1':
    SINO_SHAPE = (32, 16, 24)
    VCD_ITERS = 1
    OUT_DIR = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'w2_inc5_smoke')
    CELLS = [c for c in CELLS if c['n'] <= int(os.environ.get('W2I5_SMOKE_MAX_N', '1'))]


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
    model = mbirjax.ConeBeamModel(SINO_SHAPE, angles,
                                  source_detector_dist=4 * SINO_SHAPE[2],
                                  source_iso_dist=4 * SINO_SHAPE[2])
    note(f'summary={model.device_summary}')
    recon_shape = model.get_params('recon_shape')
    rng = np.random.default_rng(0)
    sino = jnp.asarray(rng.random(SINO_SHAPE, dtype=np.float32))
    jax.block_until_ready(sino)

    if cfg['op'] in ('back', 'hess'):
        cp = 2 if cfg['op'] == 'hess' else 1
        idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=True)
        for _ in range(WARMUP):
            jax.block_until_ready(model.sparse_back_project(sino, idx, coeff_power=cp))
        ts = []
        for _ in range(TRIALS):
            t = time.perf_counter()
            r = jax.block_until_ready(model.sparse_back_project(sino, idx,
                                                                coeff_power=cp))
            ts.append(time.perf_counter() - t)
        # Hessian tolerance 1e-4 (decision (a): the in-kernel affine-m rounding
        # sequence differs from XLA's; squared weights do not cancel it).
        result, tol = np.asarray(r), 1e-5 if cp == 1 else 1e-4
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
        env = dict(os.environ, W2I5_CELL=json.dumps(cfg),
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
    print('===== inc5 A/B summary =====', flush=True)
    print('\n'.join(summary), flush=True)
    print('=== w2_inc5_ab done ===', flush=True)


if __name__ == '__main__':
    worker(json.loads(os.environ['W2I5_CELL'])) if os.environ.get('W2I5_CELL') \
        else orchestrator()
