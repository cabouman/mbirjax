"""E2 long tail: is the hardened synthetic's growing z-constant error component
iteration-dynamics drift, or descent toward a streaky minimizer?

Two variants x two seeds, 60 iterations, hardened geometry (truncated + 11.3 deg
cone, ball layer at 0.35), default damping, registry settings (sharpness 1.5,
snr_db 35):

  fdk_init   -- the production init path (direct recon + optimal error-sinogram
                scaling), i.e. the configuration every prior run used, extended
                to a long tail.
  truth_init -- init = the ground truth phantom cropped to the usual ROR
                cylinder, with the SAME optimal scalar applied that the FDK
                init receives internally: alpha = argmin_a ||y - a A x0||_W^2
                (vcd_recon skips that scaling for an explicit init, so it is
                applied here before the driver call).

The objective is convex, so both variants share one limit point.  If the
truth-started run GROWS the streak field, the minimizer itself is streaky
(objective-side remedies).  If instead the FDK-started run's field drains to
the truth-started level, the field is dynamics drift (update-side remedies).

Per-iteration scoring vs the ground truth via run_io.make_hook (P(f_z) with
per-bin powers, v1 S/control, objective terms at target sigmas, preconditioner
share), snapshots at every iteration, then disk-based two-seed passes: the
scalar curves (run_io.two_seed_curves) and per-bin two-seed spectra.

Run on gautschi:  python -u e2_longtail.py
"""

import glob
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
import synthetic_hardening as sh                              # noqa: E402
import metrics                                                # noqa: E402
import run_io                                                 # noqa: E402

# ---------------------------------------------------------------- configuration
ITERATIONS = 60
SEEDS = (1, 2)
BALL_LAYER_Z_FRAC = 0.35
SDD_MULT = 2.5                    # 11.3 deg half-angle; truncated (confirm config)
IMAGE_ITS = (0, 2, 5, 9, 14, 20, 30, 45, 59)
OUTPUT_ROOT = '/scratch/gautschi/buzzard/sharpness_schedule/e2_longtail'
# -------------------------------------------------------------------------------


def two_seed_powers(run_dirs, mask, out_path):
    """Per-iteration two-seed P(f_z) vectors (per-bin, unlike two_seed_curves'
    scalars) from the on-disk snapshot pairs of exactly two runs."""
    assert len(run_dirs) == 2
    snaps_a = sorted(glob.glob(os.path.join(run_dirs[0], 'snapshots', 'it_*.npy')))
    its, powers, freqs = [], [], None
    for pa in snaps_a:
        pb = os.path.join(run_dirs[1], 'snapshots', os.path.basename(pa))
        if not os.path.exists(pb):
            continue
        va, vb = np.load(pa), np.load(pb)
        freqs, p2 = metrics.two_seed_spectrum(va, vb, mask=mask)
        del va, vb
        its.append(int(os.path.basename(pa)[3:6]))
        powers.append(p2.astype(np.float32))
    np.savez_compressed(out_path, iterations=np.asarray(its),
                        freqs=freqs.astype(np.float32),
                        powers=np.stack(powers))
    return len(its)


def main():
    t0 = time.time()
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    model, gt, sino, weights, half_angle = sh.build_probe_case(
        SDD_MULT, True, ball_layer_z_frac=BALL_LAYER_Z_FRAC)
    model.set_params(sharpness=sh.CENTER_S, snr_db=sh.CENTER_DB)
    recon_shape = model.get_params('recon_shape')
    print(f'e2 longtail: half-angle {half_angle:.1f} deg, recon {recon_shape}',
          flush=True)

    targets = compute_targets(model, sino, weights)
    mask = metrics.interior_mask(gt.shape)
    weights_dev = jnp.asarray(weights)

    # Truth init: GT cropped to the usual ROR cylinder, then the same optimal
    # scalar the internal FDK init receives: alpha = <W A x0, y> / <W A x0, A x0>
    # (host f64 for the reductions; the sinogram is small).
    ror2d = np.asarray(mj.get_2d_ror_mask(recon_shape))
    x0 = (gt * ror2d[:, :, None]).astype(np.float32)
    ax = np.asarray(model.forward_project(x0), dtype=np.float64)
    w64 = np.asarray(weights, dtype=np.float64)
    s64 = np.asarray(sino, dtype=np.float64)
    alpha = float(np.sum(w64 * ax * s64) / np.sum(w64 * ax * ax))
    truth_init = (alpha * x0).astype(np.float32)
    es0 = float(np.sqrt(np.mean((s64 - alpha * ax) ** 2)))
    print(f'truth init: alpha={alpha:.6f}, es_rmse at init={es0:.6f}', flush=True)
    with open(os.path.join(OUTPUT_ROOT, 'truth_init_scale.json'), 'w') as f:
        json.dump(dict(alpha=alpha, es_rmse_init=es0,
                       half_angle_deg=half_angle), f, indent=1)

    for variant, init in (('fdk_init', None), ('truth_init', truth_init)):
        run_dirs = []
        for seed in SEEDS:
            run_dir = os.path.join(OUTPUT_ROOT, variant, f'seed{seed}')
            run_dirs.append(run_dir)
            if run_io.run_is_complete(run_dir):
                print(f'[{variant} seed{seed}] already complete; skipped',
                      flush=True)
                continue
            print(f'=== {variant} seed{seed} ===', flush=True)
            hook = run_io.make_hook(
                model, gt, mask, run_dir, targets=targets,
                weights_device=weights_dev, z_step=1,
                snapshot_iterations=tuple(range(ITERATIONS)),
                prior_loss=True, image_iterations=IMAGE_ITS,
                real_sino_size=int(np.prod(sino.shape)))
            rec = run_segmented(model, sino, weights=weights,
                                max_iterations=ITERATIONS, seed=seed,
                                per_iteration_hook=hook, init_recon=init)
            run_io.save_run(run_dir, rec, dict(
                experiment='e2_longtail', variant=variant, seed=seed,
                iterations=ITERATIONS, sdd_mult=SDD_MULT, truncated=True,
                ball_layer_z_frac=BALL_LAYER_Z_FRAC, damping='default',
                sharpness=sh.CENTER_S, snr_db=sh.CENTER_DB,
                init='fdk' if init is None else f'truth*{alpha:.6f}'))
            print(f'[{variant} seed{seed}] done ({(time.time() - t0) / 60:.1f} '
                  f'min)', flush=True)
        ts = run_io.two_seed_curves(run_dirs, mask, z_step=1)
        with open(os.path.join(OUTPUT_ROOT, variant, 'two_seed.json'), 'w') as f:
            json.dump(ts, f, indent=1)
        n = two_seed_powers(run_dirs, mask,
                            os.path.join(OUTPUT_ROOT, variant,
                                         'two_seed_powers.npz'))
        print(f'[{variant}] two-seed curves + {n} per-bin spectra written '
              f'({(time.time() - t0) / 60:.1f} min)', flush=True)

    print(f'e2 longtail complete in {(time.time() - t0) / 60:.1f} min',
          flush=True)


if __name__ == '__main__':
    main()
