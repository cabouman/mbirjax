"""Calibrated beam-hardening run: real-scan severity, denser ball grid.

The severity sweep bracketed the real padded scan's residual dip (-0.014)
between s = 0 and s = 0.5; this run uses s = 0.2 with a denser ball lattice
(smaller pitch and radius -> many more ball pairs, closer to the real BGA's
grid) for visual parity with the real padded reconstruction.  Cases:

  cal_contained  -- hardening as the only pathology, dense grid.
  cal_truncpad   -- truncated + padded recon (crop-scored), dense grid: the
                    configuration that mirrors the real padded scan.

Seeds 1-2, 17-iteration protocol, registry settings.  Instruments identical
to hardening_bh.py (transfer curves vs true and MAR coordinates, signed
ledger, residual sinograms, per-iteration images).

Run on gautschi:  python -u hardening_calibrated.py
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
from mbirjax.preprocess import mar

from segmented_driver import run_segmented, compute_targets   # noqa: E402
import synthetic_hardening as sh                              # noqa: E402
import metrics                                                # noqa: E402
import run_io                                                 # noqa: E402
import hardening_bh as hb                                     # noqa: E402
from sweep_sharpness_mass import mass_ledger                  # noqa: E402

# ---------------------------------------------------------------- configuration
SEVERITY = 0.2
DENSE_PHANTOM = dict(ball_pitch_frac=0.085, ball_radius_frac=0.028)
ITERATIONS = 17
SEEDS = (1, 2)
IMAGE_ITS = (0, 2, 5, 14, 16)
OUTPUT_ROOT = '/scratch/gautschi/buzzard/sharpness_schedule/hardening_cal'
# -------------------------------------------------------------------------------


def main():
    t0 = time.time()
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    summary = {}
    for geometry in ('contained', 'truncpad'):
        name = f'cal_{geometry}'
        model, gt, sino, weights, crop_rc, t_metals, dr = hb.build_case(
            geometry, SEVERITY, phantom_kwargs=DENSE_PHANTOM)
        model.set_params(sharpness=sh.CENTER_S, snr_db=sh.CENTER_DB)
        mask = metrics.interior_mask(gt.shape)
        print(f'=== {name}: recon {model.get_params("recon_shape")}, '
              f's={SEVERITY}, rel deficit p99 {dr["rel_deficit_p99"]:.3f} ===',
              flush=True)
        case_sum = dict(severity=SEVERITY, rel_deficit_p99=dr['rel_deficit_p99'],
                        per_seed={})
        for seed in SEEDS:
            run_dir = os.path.join(OUTPUT_ROOT, name, f'seed{seed}')
            if not run_io.run_is_complete(run_dir):
                if crop_rc is None:
                    targets = compute_targets(model, sino, weights)
                    hook = run_io.make_hook(
                        model, gt, mask, run_dir, targets=targets,
                        weights_device=jnp.asarray(weights), z_step=1,
                        snapshot_iterations=(0, 5, 14, 16), prior_loss=True,
                        image_iterations=IMAGE_ITS,
                        real_sino_size=int(np.prod(sino.shape)))
                else:
                    hook = run_io.make_crop_hook(
                        model, gt, mask, run_dir, crop_rc=crop_rc, z_step=1,
                        snapshot_iterations=(0, 5, 14, 16),
                        image_iterations=IMAGE_ITS, label=name)
                rec = run_segmented(model, sino, weights=weights,
                                    max_iterations=ITERATIONS, seed=seed,
                                    per_iteration_hook=hook)
                full = rec['final_recon']
                if crop_rc is not None:
                    np.save(os.path.join(run_dir, 'final_recon_full.npy'),
                            full.astype(np.float32))
                    r0, c0, nr, nc = crop_rc
                    rec['final_recon'] = full[r0:r0 + nr, c0:c0 + nc, :]
                run_io.save_run(run_dir, rec, dict(
                    experiment='hardening_calibrated', case=name,
                    geometry=geometry, severity=SEVERITY, seed=seed,
                    iterations=ITERATIONS, phantom=DENSE_PHANTOM,
                    w_pe_metals=list(hb.W_PE_METALS),
                    ball_values=list(hb.BALL_VALUES)))

            final_name = ('final_recon_full.npy' if crop_rc is not None
                          else 'final_recon.npy')
            final = np.load(os.path.join(run_dir, final_name))
            e = sino - np.asarray(model.forward_project(final))
            np.save(os.path.join(run_dir, 'final_residual_sino.npy'),
                    e.astype(np.float32))
            inst = dict(residual_rms=float(np.sqrt(np.mean(e ** 2))))
            for k, tm in enumerate(t_metals):
                inst[f'true_m{k}'] = hb.bin_curve(e, tm)
            try:
                _, mar_metals = mar._est_plastic_metal_sinos_from_recon(
                    final, num_metal=len(hb.W_PE_METALS), ct_model=model)
                for k, ms in enumerate(mar_metals):
                    inst[f'mar_m{k}'] = hb.bin_curve(e, np.asarray(ms))
            except Exception as exc:                       # noqa: BLE001
                inst['mar_error'] = repr(exc)
            crop_final = np.load(os.path.join(run_dir, 'final_recon.npy'))
            inst['ledger'] = mass_ledger(crop_final, gt)
            rec_npz = np.load(os.path.join(run_dir, 'records.npz'),
                              allow_pickle=True)
            inst['S_low_final'] = float(rec_npz['S_low'][-1])
            case_sum['per_seed'][f'seed{seed}'] = inst
            tm0 = inst['true_m0']['bins']
            print(f'  [{name} seed{seed}] res rms={inst["residual_rms"]:.5f} '
                  f'S_low={inst["S_low_final"]:.4g} '
                  f'top-m0 mean={tm0[-1]["mean"] if tm0 else float("nan"):+.5f} '
                  f'ledger total {inst["ledger"]["total_mass_frac"]:+.3f} '
                  f'({(time.time() - t0) / 60:.1f} min)', flush=True)
        summary[name] = case_sum
        with open(os.path.join(OUTPUT_ROOT, 'cal_summary.json'), 'w') as f:
            json.dump(summary, f, indent=1)
    print(f'hardening_calibrated complete in {(time.time() - t0) / 60:.1f} min',
          flush=True)


if __name__ == '__main__':
    main()
