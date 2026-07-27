"""E1: does the streaky-minimizer verdict transfer to the real scan?

The downsampled BGA baseline (registry settings, sharpness 1.5 / snr_db 35),
seeds 1-2, extended to 60 iterations.  Per-iteration scoring vs the
CONSERVATIVE reference (sharpness 0 / snr_db 30, 60 iterations — note: a
different objective's solution, so vs-reference distance plays the role the
ground truth played on the synthetic: distance from a low-regularization
solution).  Snapshots at every iteration feed the run-vs-run (two-seed)
curves — the reconcile test: if the two seeds' fields converge to each other
while severity vs the conservative reference plateaus, the late growth is the
sharp-settings minimizer's own content, as on the synthetic.

Also directly measures severity vs iteration out to 60 on the real scan — the
input the max_iterations 15 -> 25-50 release question needs.

Run on gautschi:  python -u e1_longtail_bga.py
"""

import json
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'driver'))

import mbirjax as mj  # mbirjax must be imported before jax (sets XLA env vars)
import jax.numpy as jnp

from segmented_driver import run_segmented, compute_targets   # noqa: E402
import a2_bga                                                 # noqa: E402
import metrics                                                # noqa: E402
import run_io                                                 # noqa: E402

# ---------------------------------------------------------------- configuration
CENTER_S, CENTER_DB = 1.5, 35.0
ITERATIONS = 60
SEEDS = (1, 2)
IMAGE_ITS = (0, 4, 9, 14, 29, 44, 59)
OUTPUT_ROOT = '/scratch/gautschi/buzzard/sharpness_schedule/e1_longtail'
REFERENCE_PATH = ('/scratch/gautschi/buzzard/sharpness_schedule/a2_bga/'
                  'reference_recon.npy')
# -------------------------------------------------------------------------------


def main():
    t0 = time.time()
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    model, sinogram, weights = a2_bga.load_case()
    model.set_params(sharpness=CENTER_S, snr_db=CENTER_DB)
    reference = np.load(REFERENCE_PATH)
    mask = metrics.interior_mask(reference.shape)
    targets = compute_targets(model, sinogram, weights)
    w_dev = jnp.asarray(weights)
    print(f'e1 longtail: recon {reference.shape}, {ITERATIONS} iterations, '
          f'seeds {SEEDS}', flush=True)

    run_dirs = []
    for seed in SEEDS:
        run_dir = os.path.join(OUTPUT_ROOT, 'baseline', f'seed{seed}')
        run_dirs.append(run_dir)
        if run_io.run_is_complete(run_dir):
            print(f'[seed{seed}] already complete; skipped', flush=True)
            continue
        print(f'=== baseline seed{seed}, 60 iterations ===', flush=True)
        hook = run_io.make_hook(
            model, reference, mask, run_dir, targets=targets,
            weights_device=w_dev, z_step=1,
            snapshot_iterations=tuple(range(ITERATIONS)),
            prior_loss=True, image_iterations=IMAGE_ITS,
            real_sino_size=math.prod(model.get_params('sinogram_shape')))
        rec = run_segmented(model, sinogram, weights=weights,
                            max_iterations=ITERATIONS, seed=seed,
                            per_iteration_hook=hook)
        run_io.save_run(run_dir, rec, dict(
            experiment='e1_longtail', variant='baseline', seed=seed,
            iterations=ITERATIONS, sharpness=CENTER_S, snr_db=CENTER_DB))
        run_io.save_run_images(run_dir, reference,
                               label=f'e1 baseline seed{seed} (60 it)')
        print(f'[seed{seed}] done ({(time.time() - t0) / 60:.1f} min)',
              flush=True)

    ts = run_io.two_seed_curves(run_dirs, mask, z_step=1)
    with open(os.path.join(OUTPUT_ROOT, 'two_seed.json'), 'w') as f:
        json.dump(ts, f, indent=1)
    n = run_io.two_seed_powers(run_dirs, mask,
                               os.path.join(OUTPUT_ROOT, 'two_seed_powers.npz'))
    print(f'two-seed curves + {n} per-bin spectra written', flush=True)

    # Reconcile summary: the two 60-it states vs each other and vs the
    # conservative reference.
    fa = np.load(os.path.join(run_dirs[0], 'final_recon.npy'))
    fb = np.load(os.path.join(run_dirs[1], 'final_recon.npy'))
    diff = fa - fb
    out = dict(
        seed_pair_rel_max=float(np.max(np.abs(diff)) / np.max(np.abs(fa))),
        seed_pair_rms=float(np.sqrt(np.mean(diff ** 2))),
        run_rms=float(np.sqrt(np.mean(fa ** 2))))
    with open(os.path.join(OUTPUT_ROOT, 'reconcile.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(f'reconcile: {out}', flush=True)
    print(f'e1 longtail complete in {(time.time() - t0) / 60:.1f} min',
          flush=True)


if __name__ == '__main__':
    main()
