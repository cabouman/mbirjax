"""Demo-level sanity (2026-07-13): run the ``demo_1_shepp_logan`` workflow at 1024^3 for
parallel AND cone, flag-on vs ``MBIRJAX_DISABLE_PALLAS=1``, on ONE GPU -- exercising the
n=1 single-device pallas paths (parallel back+fwd, cone back) through the REAL user-facing
``recon()``, which the multi-device gate harnesses do not cover.

Gate: final recons compared at rel-max <= 1e-3 (a loose, iterated, real-workflow gate).
Both configs seed ``np.random`` before ``generate_demo_data`` (identical phantom/sinogram)
AND again before ``recon`` (identical VCD partitions -- the reproducibility gotcha), and
fix ``max_iterations=15`` / ``stop_threshold_change_pct=0`` so off and on run the SAME 15
iterations (apples-to-apples; early stopping could otherwise halt the two paths at
different iterations).  Reports walls, per-GPU peak, and the ``get_compute_config`` device
line (which must show the expected ``(pallas: ...)`` tokens: parallel back+fwd, cone back).

Also reports an INTERIOR-cropped rel as a diagnostic: cone's outermost-radial / axial-edge
voxels are ill-conditioned (the documented inc5 divergence), so a rel-max miss there that
is localized to the edge (interior rel small) is the SAME expected story, not a defect --
report it, do not chase it.  Each config runs in an isolated subprocess (fresh JAX =>
honest peak; CUDA_VISIBLE_DEVICES=0).
"""
import json
import os
import subprocess
import sys
import time

NUM = 1024
MAX_ITERS = 15
OUT_DIR = '/scratch/gautschi/buzzard/soak_demo_sanity'
CELLS = [dict(name=f'{mt}_{cfg}', model_type=mt, on=cfg == 'on')
         for mt in ('parallel', 'cone') for cfg in ('off', 'on')]

if os.environ.get('DEMO_SMOKE') == '1':
    NUM = 64
    MAX_ITERS = 3
    OUT_DIR = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'soak_demo_smoke')


def worker(cfg):
    import numpy as np
    import mbirjax as mj

    t0 = time.perf_counter()

    def note(msg):
        print(f'[{time.perf_counter() - t0:8.2f}s] {msg}', flush=True)

    # Identical phantom + sinogram across off/on (generate_demo_data draws from the
    # numpy global RNG for the phantom).
    np.random.seed(0)
    phantom, sinogram, params = mj.generate_demo_data(
        object_type='shepp-logan', model_type=cfg['model_type'],
        num_views=NUM, num_det_rows=NUM, num_det_channels=NUM, target_max_attenuation=None)
    angles = params['angles']
    if cfg['model_type'] == 'cone':
        model = mj.ConeBeamModel(sinogram.shape, angles,
                                 source_detector_dist=params['source_detector_dist'],
                                 source_iso_dist=params['source_iso_dist'])
    else:
        model = mj.ParallelBeamModel(sinogram.shape, angles)
    weights = mj.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')
    model.set_params(sharpness=1.0, verbose=0)
    note(f'summary={model.device_summary}')     # carries the (pallas: ...) token
    model.get_compute_config(print_results=True)  # full policy + WHY pallas is off, if off

    # Identical VCD partitions across off/on (partitions draw from the numpy global RNG).
    np.random.seed(0)
    t = time.perf_counter()
    recon, _ = model.recon(sinogram, weights=weights, max_iterations=MAX_ITERS,
                           stop_threshold_change_pct=0.0, print_logs=False)
    note(f'RESULT wall={time.perf_counter() - t:.3f}s (max_iterations={MAX_ITERS})')
    result = np.asarray(recon)

    ref_path = os.path.join(OUT_DIR, f"{cfg['model_type']}_ref.npy")
    if not cfg['on']:
        np.save(ref_path, result)
        note('RESULT ref=saved')
    else:
        ref = np.load(ref_path)
        denom = float(np.max(np.abs(ref)))
        rel = float(np.max(np.abs(result - ref)) / denom)
        # Interior crop: drop the outer 12% radius (rows/cols) and the outer 6% of slices
        # -- the ill-conditioned edge voxels the inc5 divergence localized to.
        r, c, s = result.shape
        mr, mc, ms = int(r * 0.12), int(c * 0.12), max(1, int(s * 0.06))
        core = (slice(mr, r - mr), slice(mc, c - mc), slice(ms, s - ms))
        rel_in = float(np.max(np.abs(result[core] - ref[core])) / denom)
        note(f'RESULT rel_max={rel:.2e} {"PASS" if rel <= 1e-3 else "FAIL"} (tol 1e-3) '
             f'rel_interior={rel_in:.2e}')
    stats = mj.get_memory_stats(print_results=False)
    gpu_peaks = [x['peak_bytes_in_use'] for x in stats if x['id'].startswith('GPU')]
    note(f'RESULT peak_gb={max(gpu_peaks) / 2**30 if gpu_peaks else 0.0:.2f}')


def orchestrator():
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in os.listdir(OUT_DIR):
        if f.endswith('_ref.npy'):
            os.remove(os.path.join(OUT_DIR, f))
    summary = []
    for cfg in CELLS:
        env = dict(os.environ, DEMO_CELL=json.dumps(cfg), CUDA_VISIBLE_DEVICES='0')
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
    print('===== demo sanity summary =====', flush=True)
    print('\n'.join(summary), flush=True)
    print('=== soak_demo_sanity done ===', flush=True)


if __name__ == '__main__':
    worker(json.loads(os.environ['DEMO_CELL'])) if os.environ.get('DEMO_CELL') \
        else orchestrator()
