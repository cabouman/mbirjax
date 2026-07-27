"""Synthetic hardening probes: can the ball-grid case be made to show center-slice
noise and streaking?  (Greg, 2026-07-25.)

Levers, per the center-slice campaign's gating and the flash mechanism:
  - CONE ANGLE: shorter source distances at fixed magnification (half-angle
    ~7.1 deg in the Phase A case; ~11-14 deg here), with the slice count growing
    naturally through the axial padding.
  - LATERAL TRUNCATION: forward-project an ENLARGED ground truth phantom (a wider
    slab on a scale_recon_shape(1.5) grid) onto the same 256-channel detector, then
    reconstruct on the standard grid -- the object genuinely leaves the field of
    view, the flash mechanism of the real scan.
  - DAMPING on/off: the shipped per-slice damping suppresses exactly the
    center-slice mode; probes run both.

Each probe: 6 iterations (the default sequence reaches two 128-subset iterations --
the center-slice mode is fine-subset-excited), seeds {1, 2}, scored on
  - two-seed S_low (streaking, partition-driven),
  - a center-slice indicator: per-slice in-plane high-pass error energy P(z),
    summarized as center-band mean / median (the cs mode peaks at the volume
    center),
  - per-iteration (x,z) error images for the eye.

Run on gautschi:  python -u synthetic_hardening.py
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
from phantom import ball_grid_phantom                         # noqa: E402
from noise import add_transmission_noise                      # noqa: E402
import metrics                                                # noqa: E402
import run_io                                                 # noqa: E402

# ---------------------------------------------------------------- configuration
SIZE = 256
NUM_VIEWS = 100
I0 = 1.0e4
TARGET_MAX_SINO = 5.0
NOISE_SEED = 7
ITERATIONS = 6                 # reaches two 128-subset iterations
SEEDS = (1, 2)
CENTER_S, CENTER_DB = 1.5, 35.0
TRUNC_GRID_SCALE = 1.5         # enlarged-phantom grid for the truncated variants
TRUNC_SLAB_FRAC = 0.72         # slab width 0.72*1.5 = 1.08x the standard FoV

# (name, source_detector_dist multiple of SIZE, truncated?, damping on?)
PROBES = [
    ('base_nodamp',   4.0, False, False),   # Phase A geometry, damping off
    ('cone11_damp',   2.5, False, True),    # ~11.3 deg half-angle
    ('cone11_nodamp', 2.5, False, False),
    ('trunc_damp',    4.0, True,  True),    # truncation at the Phase A cone angle
    ('trunc_nodamp',  4.0, True,  False),
    ('cone11_trunc_nodamp', 2.5, True, False),
]

OUTPUT_ROOT = '/scratch/gautschi/buzzard/sharpness_schedule/hardening'
# -------------------------------------------------------------------------------


def build_probe_case(sdd_mult, truncated, ball_layer_z_frac=0.5):
    """Model + ground truth (cropped to the recon grid) + noisy sino + weights."""
    sinogram_shape = (NUM_VIEWS, SIZE, SIZE)
    angles = jnp.linspace(0, 2 * np.pi, NUM_VIEWS, endpoint=False)
    sdd = sdd_mult * SIZE
    model = mj.ConeBeamModel(sinogram_shape, angles, source_detector_dist=sdd,
                             source_iso_dist=sdd / 2.0)
    model.set_params(verbose=0)

    if not truncated:
        gt = ball_grid_phantom(model.get_params('recon_shape'),
                               ball_layer_z_frac=ball_layer_z_frac)
        sino = np.asarray(model.forward_project(gt))
        gt_crop = gt
    else:
        # Enlarged grid holds the wide slab; the 256-channel detector truncates it.
        model_big = mj.ConeBeamModel(sinogram_shape, angles,
                                     source_detector_dist=sdd,
                                     source_iso_dist=sdd / 2.0)
        model_big.set_params(verbose=0)
        model_big.scale_recon_shape(TRUNC_GRID_SCALE, TRUNC_GRID_SCALE)
        big_shape = model_big.get_params('recon_shape')
        gt_big = ball_grid_phantom(big_shape, slab_xy_frac=TRUNC_SLAB_FRAC,
                                   ball_layer_z_frac=ball_layer_z_frac)
        sino = np.asarray(model_big.forward_project(gt_big))
        small_shape = model.get_params('recon_shape')
        r0 = (big_shape[0] - small_shape[0]) // 2
        c0 = (big_shape[1] - small_shape[1]) // 2
        z0 = (big_shape[2] - small_shape[2]) // 2
        gt_crop = gt_big[r0:r0 + small_shape[0], c0:c0 + small_shape[1],
                         z0:z0 + small_shape[2]]

    scale = TARGET_MAX_SINO / float(sino.max())
    gt_crop = (gt_crop * scale).astype(np.float32)
    sino = (sino * scale).astype(np.float32)
    sino_noisy, weights = add_transmission_noise(sino, i0=I0,
                                                 noise_seed=NOISE_SEED)
    half_angle = np.degrees(np.arctan(SIZE / 2.0 / sdd))
    return model, gt_crop, sino_noisy, weights, half_angle


def slice_profile(err, mask, hp_sigma=2.0):
    """Per-slice in-plane high-pass energy P(z) and the center-band summary."""
    prof = np.array([np.mean(metrics.highpass2d(err[:, :, k], hp_sigma)[mask] ** 2)
                     for k in range(err.shape[2])])
    zc = err.shape[2] // 2
    center = float(prof[zc - 2:zc + 3].mean())
    return prof, center / float(np.median(prof))


def main():
    t0 = time.time()
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    results = {}
    for name, sdd_mult, truncated, damping in PROBES:
        model, gt, sino, weights, half_angle = build_probe_case(sdd_mult, truncated)
        recon_shape = model.get_params('recon_shape')
        print(f'=== {name}: half-angle {half_angle:.1f} deg, recon {recon_shape}, '
              f'truncated={truncated}, damping={damping} ===', flush=True)
        if not damping:
            model._dc_damping = None
        model.set_params(sharpness=CENTER_S, snr_db=CENTER_DB)
        mask = metrics.interior_mask(gt.shape)
        finals, profiles = {}, {}
        for seed in SEEDS:
            run_dir = os.path.join(OUTPUT_ROOT, name, f'seed{seed}')
            img_dir = os.path.join(run_dir, 'images')
            os.makedirs(img_dir, exist_ok=True)

            def hook(i, recon_device, ckpt, seg_record, _dirs=(run_dir, img_dir)):
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                vol = np.asarray(model._gather_recon(recon_device))
                err = vol - gt
                del vol
                sc = metrics.streak_score(err, mask=mask)
                freqs, power = metrics.axial_power_spectrum(err, mask=mask)
                v2 = metrics.zcoherence_summary(freqs, power)
                prof, center_ratio = slice_profile(err, mask)
                xz_err = err[err.shape[0] // 2, :, :].T
                emax = float(np.percentile(np.abs(xz_err), 99.5))
                fig, ax = plt.subplots(figsize=(6.0, 4.0))
                im = ax.imshow(xz_err, vmin=-emax, vmax=emax, cmap='seismic',
                               aspect='equal')
                ax.set_title(f'{name} seed{seed} it{i} (x,z) error', fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                fig.colorbar(im, ax=ax, shrink=0.8)
                fig.tight_layout()
                fig.savefig(os.path.join(_dirs[1], f'err_xz_it{i}.png'), dpi=110)
                plt.close(fig)
                print(f'    it {i}: S_low={v2["S_low"]:.4g} Rz={v2["Rz"]:.1f} '
                      f'center_ratio={center_ratio:.2f} '
                      f'alpha={seg_record["alpha"]:.3f}', flush=True)
                del err
                return dict(S_low=v2['S_low'], Rz=v2['Rz'],
                            center_ratio=center_ratio,
                            profile=prof.astype(np.float32))

            print(f'  seed {seed}:', flush=True)
            rec = run_segmented(model, sino, weights=weights,
                                max_iterations=ITERATIONS, seed=seed,
                                per_iteration_hook=hook)
            finals[seed] = rec['final_recon']
            profiles[seed] = [h['profile'] for h in rec['hook']]
            np.savez_compressed(os.path.join(run_dir, 'probe_records.npz'),
                                S_low=[h['S_low'] for h in rec['hook']],
                                Rz=[h['Rz'] for h in rec['hook']],
                                center_ratio=[h['center_ratio']
                                              for h in rec['hook']],
                                profiles=np.stack([h['profile']
                                                   for h in rec['hook']]))

        freqs, p2 = metrics.two_seed_spectrum(finals[1], finals[2], mask=mask)
        ts = metrics.zcoherence_summary(freqs, p2)
        cr_final = float(np.mean([slice_profile(
            np.asarray(finals[s]) - gt, mask)[1] for s in SEEDS]))
        results[name] = dict(half_angle_deg=half_angle,
                             recon_shape=[int(v) for v in recon_shape],
                             two_seed_S_low_final=float(ts['S_low']),
                             two_seed_Rz_final=float(ts['Rz']),
                             center_ratio_final=cr_final)
        print(f'  [{name}] two-seed S_low@{ITERATIONS - 1}={ts["S_low"]:.4g} '
              f'Rz={ts["Rz"]:.1f} center_ratio={cr_final:.2f} '
              f'({(time.time() - t0) / 60:.1f} min)', flush=True)
        with open(os.path.join(OUTPUT_ROOT, 'hardening_results.json'), 'w') as f:
            json.dump(results, f, indent=1)
    print(f'hardening probes complete in {(time.time() - t0) / 60:.1f} min',
          flush=True)


if __name__ == '__main__':
    main()
