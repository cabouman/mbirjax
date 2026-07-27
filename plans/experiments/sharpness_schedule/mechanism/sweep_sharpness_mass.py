"""Sharpness sweep of the truncation-deposit partition (interior-flash test).

Hypothesis (Greg): the streak field is an interior version of the truncation
flash — the unexplained out-of-view attenuation is deposited inside the
reconstruction, and the prior's strength sets HOW it is distributed: a strong
prior squeezes it onto the boundary ring, a weak prior lets it spread inward
as fine-scale texture.  Predictions: ring share falls / interior share and
fine-scale fraction rise with sharpness, while the TOTAL deposit (set by the
data mismatch, not the prior) stays roughly constant.

Runs: hardened synthetic (truncated + 11.3 deg cone, ball layer 0.35, default
damping), sharpness in {0, 0.75, 2.25, 3} at snr_db 35, seed 1, 60 iterations
(the plateau regime).  The sharpness-1.5 point is the e2_longtail fdk_init run
(same configuration; its pair also bounds seed wiggle).  The mass ledger is
printed per run and saved to sweep_summary.json.

Run on gautschi:  python -u sweep_sharpness_mass.py
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
from scipy import ndimage

from segmented_driver import run_segmented, compute_targets   # noqa: E402
import synthetic_hardening as sh                              # noqa: E402
import metrics                                                # noqa: E402
import run_io                                                 # noqa: E402

# ---------------------------------------------------------------- configuration
SHARPNESS_VALUES = (0.0, 0.75, 2.25, 3.0)   # 1.5 = the e2_longtail fdk_init run
SNR_DB = 35.0
ITERATIONS = 60
SEED = 1
BALL_LAYER_Z_FRAC = 0.35
SDD_MULT = 2.5
IMAGE_ITS = (0, 5, 14, 30, 59)
OUTPUT_ROOT = '/scratch/gautschi/buzzard/sharpness_schedule/sweep_mass'
# -------------------------------------------------------------------------------


def mass_ledger(final, gt):
    """Ring/interior partition of the deposited mass + fine-scale fraction."""
    err = final - gt
    shape = gt.shape
    ror = np.asarray(mj.get_2d_ror_mask(shape)).astype(bool)
    interior = np.asarray(mj.get_2d_ror_mask(
        shape, crop_radius_fraction=0.05)).astype(bool)
    ring = ror & ~interior
    gt_mass = float(gt[ror, :].sum())
    out = {}
    for name, m2 in (('interior', interior), ('ring', ring), ('total', ror)):
        e = err[m2, :]
        out[name + '_mass_frac'] = float(e.sum() / gt_mass)
        out[name + '_mean'] = float(e.mean())
    mid = err[:, :, shape[2] // 2]
    smooth = ndimage.gaussian_filter(mid, 4.0)
    rms = float(np.sqrt(np.mean(mid[interior] ** 2)))
    fine = float(np.sqrt(np.mean((mid - smooth)[interior] ** 2)))
    out['mid_rms'] = rms
    out['mid_fine_rms'] = fine
    out['fine_fraction'] = float(fine ** 2 / max(rms ** 2, 1e-30))
    return out


def main():
    t0 = time.time()
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    model, gt, sino, weights, half_angle = sh.build_probe_case(
        SDD_MULT, True, ball_layer_z_frac=BALL_LAYER_Z_FRAC)
    mask = metrics.interior_mask(gt.shape)
    weights_dev = jnp.asarray(weights)
    summary = {}
    for s in SHARPNESS_VALUES:
        name = f's{s:g}'
        run_dir = os.path.join(OUTPUT_ROOT, name, f'seed{SEED}')
        model.set_params(sharpness=s, snr_db=SNR_DB)
        targets = compute_targets(model, sino, weights)
        if not run_io.run_is_complete(run_dir):
            print(f'=== sharpness {s} (seed {SEED}) ===', flush=True)
            hook = run_io.make_hook(
                model, gt, mask, run_dir, targets=targets,
                weights_device=weights_dev, z_step=1,
                snapshot_iterations=(0, 5, 14, 30, 59),
                prior_loss=True, image_iterations=IMAGE_ITS,
                real_sino_size=int(np.prod(sino.shape)))
            rec = run_segmented(model, sino, weights=weights,
                                max_iterations=ITERATIONS, seed=SEED,
                                per_iteration_hook=hook)
            run_io.save_run(run_dir, rec, dict(
                experiment='sweep_sharpness_mass', sharpness=s, snr_db=SNR_DB,
                seed=SEED, iterations=ITERATIONS, sdd_mult=SDD_MULT,
                truncated=True, ball_layer_z_frac=BALL_LAYER_Z_FRAC))
        final = np.load(os.path.join(run_dir, 'final_recon.npy'))
        led = mass_ledger(final, gt)
        rec_npz = np.load(os.path.join(run_dir, 'records.npz'),
                          allow_pickle=True)
        led['S_low_final'] = float(rec_npz['S_low'][-1])
        summary[name] = led
        print(f'[{name}] ledger: total {led["total_mass_frac"]:+.3f} '
              f'ring {led["ring_mass_frac"]:+.3f} '
              f'interior {led["interior_mass_frac"]:+.3f} '
              f'fine_frac {led["fine_fraction"]:.3f} '
              f'S_low {led["S_low_final"]:.4g} '
              f'({(time.time() - t0) / 60:.1f} min)', flush=True)
        with open(os.path.join(OUTPUT_ROOT, 'sweep_summary.json'), 'w') as f:
            json.dump(summary, f, indent=1)
    print(f'sweep complete in {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
