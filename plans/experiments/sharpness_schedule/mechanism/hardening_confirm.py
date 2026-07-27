"""Hardened-synthetic confirmation at the standard protocol (follow-up to
synthetic_hardening.py; feeds the findings Appendix B update).

The probe verdict: lateral truncation is the switch (~20x two-seed S_low), the
larger cone angle compounds it (~264x total at 11.3 deg + truncation + no damping).
This confirmation runs the hardened geometry (11.3 deg + truncation) at 17
iterations, damping default AND off, seeds {1, 2} -- with the ball layer moved
OFF-CENTER (ball_layer_z_frac = 0.35) so the center-slice indicator is no longer
confounded by object structure at mid-height.

Run on gautschi:  python -u hardening_confirm.py
"""

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'driver'))

import mbirjax as mj  # mbirjax must be imported before jax (sets XLA env vars)

from segmented_driver import run_segmented                    # noqa: E402
import synthetic_hardening as sh                              # noqa: E402
import metrics                                                # noqa: E402

# ---------------------------------------------------------------- configuration
ITERATIONS = 17
SEEDS = (1, 2)
BALL_LAYER_Z_FRAC = 0.35
CONFIGS = [                       # (name, sdd multiple, truncated, damping on)
    ('hard_damp', 2.5, True, True),
    ('hard_nodamp', 2.5, True, False),
]
OUTPUT_ROOT = '/scratch/gautschi/buzzard/sharpness_schedule/hardening_confirm'
# -------------------------------------------------------------------------------


def main():
    t0 = time.time()
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    results = {}
    for name, sdd_mult, truncated, damping in CONFIGS:
        model, gt, sino, weights, half_angle = sh.build_probe_case(
            sdd_mult, truncated, ball_layer_z_frac=BALL_LAYER_Z_FRAC)
        print(f'=== {name}: half-angle {half_angle:.1f} deg, '
              f'recon {model.get_params("recon_shape")}, damping={damping} ===',
              flush=True)
        if not damping:
            model._dc_damping = None
        model.set_params(sharpness=sh.CENTER_S, snr_db=sh.CENTER_DB)
        mask = metrics.interior_mask(gt.shape)
        finals, series = {}, {}
        for seed in SEEDS:
            run_dir = os.path.join(OUTPUT_ROOT, name, f'seed{seed}')
            img_dir = os.path.join(run_dir, 'images')
            os.makedirs(img_dir, exist_ok=True)

            def hook(i, recon_device, ckpt, seg_record, _img_dir=img_dir):
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                vol = np.asarray(model._gather_recon(recon_device))
                err = vol - gt
                del vol
                freqs, power = metrics.axial_power_spectrum(err, mask=mask)
                v2 = metrics.zcoherence_summary(freqs, power)
                prof, center_ratio = sh.slice_profile(err, mask)
                if i in (0, 2, 5, 9, 14, 16):
                    xz_err = err[err.shape[0] // 2, :, :].T
                    emax = float(np.percentile(np.abs(xz_err), 99.5))
                    fig, ax = plt.subplots(figsize=(6.0, 4.0))
                    im = ax.imshow(xz_err, vmin=-emax, vmax=emax, cmap='seismic',
                                   aspect='equal')
                    ax.set_title(f'{name} seed{seed} it{i} (x,z) error',
                                 fontsize=9)
                    ax.set_xticks([]); ax.set_yticks([])
                    fig.colorbar(im, ax=ax, shrink=0.8)
                    fig.tight_layout()
                    fig.savefig(os.path.join(_img_dir, f'err_xz_it{i:02d}.png'),
                                dpi=110)
                    plt.close(fig)
                print(f'    it {i:2d}: S_low={v2["S_low"]:.4g} '
                      f'Rz={v2["Rz"]:.1f} center_ratio={center_ratio:.2f} '
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
            series[seed] = dict(
                S_low=[float(h['S_low']) for h in rec['hook']],
                Rz=[float(h['Rz']) for h in rec['hook']],
                center_ratio=[float(h['center_ratio']) for h in rec['hook']])
            np.savez_compressed(
                os.path.join(run_dir, 'confirm_records.npz'),
                profiles=np.stack([h['profile'] for h in rec['hook']]),
                **{k: np.asarray(v) for k, v in series[seed].items()})
            np.save(os.path.join(run_dir, 'final_recon.npy'),
                    rec['final_recon'].astype(np.float32))

        freqs, p2 = metrics.two_seed_spectrum(finals[1], finals[2], mask=mask)
        ts = metrics.zcoherence_summary(freqs, p2)
        results[name] = dict(half_angle_deg=half_angle,
                             per_seed=series,
                             two_seed_S_low_final=float(ts['S_low']),
                             two_seed_Rz_final=float(ts['Rz']))
        print(f'  [{name}] two-seed S_low@16={ts["S_low"]:.4g} Rz={ts["Rz"]:.1f} '
              f'({(time.time() - t0) / 60:.1f} min)', flush=True)
        with open(os.path.join(OUTPUT_ROOT, 'confirm_results.json'), 'w') as f:
            json.dump(results, f, indent=1)
    print(f'hardening confirm complete in {(time.time() - t0) / 60:.1f} min',
          flush=True)


if __name__ == '__main__':
    main()
