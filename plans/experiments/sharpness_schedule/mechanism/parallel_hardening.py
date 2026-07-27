"""Parallel-beam hardening probe: do the streaks form without a cone?

The hardened synthetic showed lateral truncation is the switch and the cone
angle a strong amplifier.  Parallel beam removes the cone entirely — slices
are coupled only through the prior — and has no per-slice DC damping at all
(the shipped damping is cone-specific).  If the truncated parallel case shows
the same z-organized streak deposit, the mechanism needs no cone; if it stays
near the contained null, the cone's slice coupling is a necessary ingredient.

Two configurations (contained / laterally truncated slab), seeds 1-2, the
standard 17-iteration protocol at registry settings (sharpness 1.5, snr_db 35),
ball layer at 0.35; scored per iteration against the ground truth
(run_io.make_hook), plus final two-seed spectra.

Run on gautschi:  python -u parallel_hardening.py
"""

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'driver'))

import mbirjax as mj  # mbirjax must be imported before jax (sets XLA env vars)
import jax.numpy as jnp

from segmented_driver import run_segmented, compute_targets   # noqa: E402
from phantom import ball_grid_phantom                         # noqa: E402
from noise import add_transmission_noise                      # noqa: E402
import synthetic_hardening as sh                              # noqa: E402
import metrics                                                # noqa: E402
import run_io                                                 # noqa: E402

# ---------------------------------------------------------------- configuration
ITERATIONS = 17
SEEDS = (1, 2)
BALL_LAYER_Z_FRAC = 0.35
IMAGE_ITS = (0, 2, 5, 9, 14, 16)
CONFIGS = (('par_contained', False), ('par_trunc', True))
OUTPUT_ROOT = '/scratch/gautschi/buzzard/sharpness_schedule/parallel_hardening'
# -------------------------------------------------------------------------------


def build_parallel_case(truncated):
    """Parallel-beam mirror of synthetic_hardening.build_probe_case: same
    detector, phantom family, noise, and rescale; angles over pi."""
    sinogram_shape = (sh.NUM_VIEWS, sh.SIZE, sh.SIZE)
    angles = jnp.linspace(0, np.pi, sh.NUM_VIEWS, endpoint=False)
    model = mj.ParallelBeamModel(sinogram_shape, angles)
    model.set_params(verbose=0)

    if not truncated:
        gt = ball_grid_phantom(model.get_params('recon_shape'),
                               ball_layer_z_frac=BALL_LAYER_Z_FRAC)
        sino = np.asarray(model.forward_project(gt))
        gt_crop = gt
    else:
        model_big = mj.ParallelBeamModel(sinogram_shape, angles)
        model_big.set_params(verbose=0)
        model_big.scale_recon_shape(sh.TRUNC_GRID_SCALE, sh.TRUNC_GRID_SCALE)
        big_shape = model_big.get_params('recon_shape')
        gt_big = ball_grid_phantom(big_shape, slab_xy_frac=sh.TRUNC_SLAB_FRAC,
                                   ball_layer_z_frac=BALL_LAYER_Z_FRAC)
        sino = np.asarray(model_big.forward_project(gt_big))
        small_shape = model.get_params('recon_shape')
        r0 = (big_shape[0] - small_shape[0]) // 2
        c0 = (big_shape[1] - small_shape[1]) // 2
        z0 = (big_shape[2] - small_shape[2]) // 2
        gt_crop = gt_big[r0:r0 + small_shape[0], c0:c0 + small_shape[1],
                         z0:z0 + small_shape[2]]

    scale = sh.TARGET_MAX_SINO / float(sino.max())
    gt_crop = (gt_crop * scale).astype(np.float32)
    sino = (sino * scale).astype(np.float32)
    sino_noisy, weights = add_transmission_noise(sino, i0=sh.I0,
                                                 noise_seed=sh.NOISE_SEED)
    return model, gt_crop, sino_noisy, weights


def main():
    t0 = time.time()
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    results = {}
    for name, truncated in CONFIGS:
        model, gt, sino, weights = build_parallel_case(truncated)
        model.set_params(sharpness=sh.CENTER_S, snr_db=sh.CENTER_DB)
        print(f'=== {name}: recon {model.get_params("recon_shape")}, '
              f'truncated={truncated} ===', flush=True)
        targets = compute_targets(model, sino, weights)
        mask = metrics.interior_mask(gt.shape)
        weights_dev = jnp.asarray(weights)
        run_dirs, finals = [], {}
        for seed in SEEDS:
            run_dir = os.path.join(OUTPUT_ROOT, name, f'seed{seed}')
            run_dirs.append(run_dir)
            if not run_io.run_is_complete(run_dir):
                print(f'--- {name} seed{seed} ---', flush=True)
                hook = run_io.make_hook(
                    model, gt, mask, run_dir, targets=targets,
                    weights_device=weights_dev, z_step=1,
                    snapshot_iterations=tuple(range(ITERATIONS)),
                    prior_loss=True, image_iterations=IMAGE_ITS,
                    real_sino_size=int(np.prod(sino.shape)))
                rec = run_segmented(model, sino, weights=weights,
                                    max_iterations=ITERATIONS, seed=seed,
                                    per_iteration_hook=hook)
                run_io.save_run(run_dir, rec, dict(
                    experiment='parallel_hardening', config=name, seed=seed,
                    iterations=ITERATIONS, truncated=truncated,
                    ball_layer_z_frac=BALL_LAYER_Z_FRAC,
                    sharpness=sh.CENTER_S, snr_db=sh.CENTER_DB))
            finals[seed] = np.load(os.path.join(run_dir, 'final_recon.npy'))

        freqs, p2 = metrics.two_seed_spectrum(finals[1], finals[2], mask=mask)
        ts = metrics.zcoherence_summary(freqs, p2)
        rec1 = np.load(os.path.join(run_dirs[0], 'records.npz'),
                       allow_pickle=True)
        results[name] = dict(
            two_seed_S_low_final=float(ts['S_low']),
            two_seed_Rz_final=float(ts['Rz']),
            vs_gt_S_low_final=float(rec1['S_low'][-1]),
            vs_gt_Rz_final=float(rec1['Rz'][-1]))
        print(f'[{name}] two-seed S_low@{ITERATIONS - 1}='
              f'{ts["S_low"]:.4g} Rz={ts["Rz"]:.1f} | vs-GT S_low='
              f'{results[name]["vs_gt_S_low_final"]:.4g} '
              f'Rz={results[name]["vs_gt_Rz_final"]:.1f} '
              f'({(time.time() - t0) / 60:.1f} min)', flush=True)
        with open(os.path.join(OUTPUT_ROOT, 'parallel_results.json'), 'w') as f:
            json.dump(results, f, indent=1)
    print('cone 17-it confirm for comparison: two-seed S_low@16 = 3.1e-5, '
          'Rz ~ 900 (truncated + 11.3 deg, default damping)', flush=True)
    print(f'parallel hardening complete in {(time.time() - t0) / 60:.1f} min',
          flush=True)


if __name__ == '__main__':
    main()
