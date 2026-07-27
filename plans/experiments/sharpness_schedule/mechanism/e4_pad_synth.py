"""Lateral padding on the hardened synthetic (the E4 mechanism arm).

Reconstruct the truncated + 11.3 deg cone case on a laterally enlarged grid
(scale_recon_shape 1.5 — the object now fits, so the unexplained attenuation
has a legitimate home).  If the interior-flash picture is right, BOTH the ring
and interior deposits collapse and the vs-truth streak severity drops toward
the contained null's.

Scoring shares a ruler with the unpadded long-tail runs: the gathered volume
is cropped to the central standard grid and scored vs the ground truth crop
with the standard interior mask (run_io.make_crop_hook).  The deposit ledger
is also computed on the FULL padded grid vs the enlarged ground truth, to
distinguish 'deposit gone' from 'deposit moved outside the crop'.

Two seeds, 60 iterations, default damping, registry settings (shp 1.5, snr 35).

Run on gautschi:  python -u e4_pad_synth.py
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

from segmented_driver import run_segmented                    # noqa: E402
from phantom import ball_grid_phantom                         # noqa: E402
import synthetic_hardening as sh                              # noqa: E402
import metrics                                                # noqa: E402
import run_io                                                 # noqa: E402

# ---------------------------------------------------------------- configuration
PAD_SCALE = 1.5              # matches the generation grid: the slab fits fully
ITERATIONS = 60
SEEDS = (1, 2)
BALL_LAYER_Z_FRAC = 0.35
SDD_MULT = 2.5
IMAGE_ITS = (0, 5, 14, 30, 59)
SNAPSHOT_ITS = tuple(range(0, 60, 1))
OUTPUT_ROOT = '/scratch/gautschi/buzzard/sharpness_schedule/e4_pad_synth'
# -------------------------------------------------------------------------------


def main():
    t0 = time.time()
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    # Data + standard-grid ground truth from the shared builder.
    _, gt_crop, sino, weights, half_angle = sh.build_probe_case(
        SDD_MULT, True, ball_layer_z_frac=BALL_LAYER_Z_FRAC)

    # Padded model: same geometry, laterally enlarged recon grid.
    sino_shape = (sh.NUM_VIEWS, sh.SIZE, sh.SIZE)
    angles = jnp.linspace(0, 2 * np.pi, sh.NUM_VIEWS, endpoint=False)
    sdd = SDD_MULT * sh.SIZE
    model = mj.ConeBeamModel(sino_shape, angles, source_detector_dist=sdd,
                             source_iso_dist=sdd / 2.0)
    model.set_params(verbose=0)
    model.scale_recon_shape(PAD_SCALE, PAD_SCALE)
    model.set_params(sharpness=sh.CENTER_S, snr_db=sh.CENTER_DB)
    pad_shape = model.get_params('recon_shape')

    # Enlarged ground truth on the padded grid, at the SAME intensity scale as
    # gt_crop (build_probe_case scale = gt_crop.max() / raw ball value).
    gt_big_raw = ball_grid_phantom(pad_shape, slab_xy_frac=sh.TRUNC_SLAB_FRAC,
                                   ball_layer_z_frac=BALL_LAYER_Z_FRAC)
    scale = float(gt_crop.max()) / float(gt_big_raw.max())
    gt_big = (gt_big_raw * scale).astype(np.float32)
    r0 = (pad_shape[0] - gt_crop.shape[0]) // 2
    c0 = (pad_shape[1] - gt_crop.shape[1]) // 2
    crop_rc = (r0, c0, gt_crop.shape[0], gt_crop.shape[1])
    assert np.allclose(gt_big[r0:r0 + gt_crop.shape[0],
                              c0:c0 + gt_crop.shape[1], :], gt_crop, atol=1e-6)
    print(f'e4 pad synth: pad grid {pad_shape} (scale {PAD_SCALE}), '
          f'crop at {crop_rc}, half-angle {half_angle:.1f} deg', flush=True)

    mask = metrics.interior_mask(gt_crop.shape)
    run_dirs = []
    for seed in SEEDS:
        run_dir = os.path.join(OUTPUT_ROOT, f'seed{seed}')
        run_dirs.append(run_dir)
        if run_io.run_is_complete(run_dir):
            print(f'[seed{seed}] already complete; skipped', flush=True)
            continue
        print(f'=== padded run seed{seed} ===', flush=True)
        hook = run_io.make_crop_hook(
            model, gt_crop, mask, run_dir, crop_rc=crop_rc, z_step=1,
            snapshot_iterations=SNAPSHOT_ITS, image_iterations=IMAGE_ITS,
            label=f'padded x{PAD_SCALE} seed{seed}')
        rec = run_segmented(model, sino, weights=weights,
                            max_iterations=ITERATIONS, seed=seed,
                            per_iteration_hook=hook)
        full_final = rec['final_recon']
        np.save(os.path.join(run_dir, 'final_recon_full.npy'),
                full_final.astype(np.float32))
        rec['final_recon'] = full_final[r0:r0 + gt_crop.shape[0],
                                        c0:c0 + gt_crop.shape[1], :]
        run_io.save_run(run_dir, rec, dict(
            experiment='e4_pad_synth', pad_scale=PAD_SCALE, seed=seed,
            iterations=ITERATIONS, sdd_mult=SDD_MULT, truncated=True,
            ball_layer_z_frac=BALL_LAYER_Z_FRAC,
            sharpness=sh.CENTER_S, snr_db=sh.CENTER_DB))
        print(f'[seed{seed}] done ({(time.time() - t0) / 60:.1f} min)',
              flush=True)

    ts = run_io.two_seed_curves(run_dirs, mask, z_step=1)
    with open(os.path.join(OUTPUT_ROOT, 'two_seed.json'), 'w') as f:
        json.dump(ts, f, indent=1)
    n = run_io.two_seed_powers(run_dirs, mask,
                               os.path.join(OUTPUT_ROOT, 'two_seed_powers.npz'))
    print(f'two-seed curves + {n} per-bin spectra written', flush=True)

    # Deposit ledger on the FULL padded grid (vs enlarged GT) and on the crop.
    from sweep_sharpness_mass import mass_ledger
    ledger = {}
    for seed in SEEDS:
        run_dir = os.path.join(OUTPUT_ROOT, f'seed{seed}')
        full = np.load(os.path.join(run_dir, 'final_recon_full.npy'))
        crop = np.load(os.path.join(run_dir, 'final_recon.npy'))
        ledger[f'seed{seed}_full'] = mass_ledger(full, gt_big)
        ledger[f'seed{seed}_crop'] = mass_ledger(crop, gt_crop)
        print(f'[seed{seed}] full-grid ledger: '
              f'total {ledger[f"seed{seed}_full"]["total_mass_frac"]:+.3f} '
              f'ring {ledger[f"seed{seed}_full"]["ring_mass_frac"]:+.3f} '
              f'interior {ledger[f"seed{seed}_full"]["interior_mass_frac"]:+.3f}'
              f' | crop total {ledger[f"seed{seed}_crop"]["total_mass_frac"]:+.3f}',
              flush=True)
    with open(os.path.join(OUTPUT_ROOT, 'ledger.json'), 'w') as f:
        json.dump(ledger, f, indent=1)
    print(f'e4 pad synth complete in {(time.time() - t0) / 60:.1f} min',
          flush=True)


if __name__ == '__main__':
    main()
