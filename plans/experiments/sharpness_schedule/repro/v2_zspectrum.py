"""Metric v2 pass over the EXISTING Phase A volumes (no new reconstructions).

Computes the axial power spectrum P(f_z) of the in-plane high-passed error
(metrics.axial_power_spectrum) on the saved snapshots/finals of the A1 synthetic and
A2 BGA runs, in both the reference-based and two-seed forms, and writes figures + a
digest.  The calibration question this answers (plan: metric v2): does the v2
low-band severity S_low rank the damping-off variant WORSE than center -- matching
the visual panels -- where v1's z-constant S did not?

Runs next to the data (compute node via sbatch -- no login-node compute).
Run:  python -u v2_zspectrum.py    (config constants below; no CLI args)
"""

import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'driver'))

import metrics  # noqa: E402

# ---------------------------------------------------------------- configuration
_SCRATCH = '/scratch/gautschi/buzzard/sharpness_schedule'
ROOTS = {
    'synthetic case': dict(path=os.path.join(_SCRATCH, 'a1'),
                           reference='gt_phantom.npy',
                           variants=('center', 'center_damp_off', 'sharp3.0',
                                     'center_noise_off')),
    'downsampled scan': dict(path=os.path.join(_SCRATCH, 'a2_bga'),
                             reference='reference_recon.npy',
                             variants=('center', 'center_damp_off', 'sharp3.0')),
}
LOW_CUT, HIGH_CUT = 0.05, 0.25     # cycles/slice-sample band edges (see metrics)
Z_STEP = 1

# Figure legend labels: settings, not internal variant names.
DISPLAY = {'center': 'shp=1.5, snr=35',
           'center_damp_off': 'shp=1.5, snr=35, no damp',
           'sharp3.0': 'shp=3.0, snr=35',
           'center_noise_off': 'shp=1.5, snr=35, no noise'}
# -------------------------------------------------------------------------------


def seed_dirs(root_path, variant):
    return sorted(glob.glob(os.path.join(root_path, variant, 'seed*')))


def snapshot_iterations(run_dir):
    return sorted(int(os.path.basename(p)[3:6]) for p in
                  glob.glob(os.path.join(run_dir, 'snapshots', 'it_*.npy')))


def analyze_root(tag, cfg):
    root = cfg['path']
    if not os.path.isdir(root):
        print(f'[{tag}] missing, skipped', flush=True)
        return
    out_dir = os.path.join(root, 'analysis')
    os.makedirs(out_dir, exist_ok=True)
    reference = np.load(os.path.join(root, cfg['reference']))
    mask = metrics.interior_mask(reference.shape)
    digest = {}

    fig_s, axes_s = plt.subplots(1, 2, figsize=(11.5, 4.3))
    fig_f, ax_f = plt.subplots(figsize=(5.2, 3.5))

    for variant in cfg['variants']:
        dirs = seed_dirs(root, variant)
        if not dirs:
            continue
        print(f'[{tag}] {variant} ({len(dirs)} seeds)', flush=True)

        # Reference-based spectrum of the FINAL recon, seed-mean.
        powers = []
        for d in dirs:
            vol = np.load(os.path.join(d, 'final_recon.npy'))
            freqs, p = metrics.axial_power_spectrum(vol - reference, mask=mask,
                                                    z_step=Z_STEP)
            powers.append(p)
        p_ref = np.mean(powers, axis=0)
        ref_sum = metrics.zcoherence_summary(freqs, p_ref, LOW_CUT, HIGH_CUT)

        # Two-seed spectrum at the final and at each common snapshot iteration.
        a, b = dirs[0], dirs[1]
        freqs2, p2_final = metrics.two_seed_spectrum(
            np.load(os.path.join(a, 'final_recon.npy')),
            np.load(os.path.join(b, 'final_recon.npy')), mask=mask, z_step=Z_STEP)
        ts_final = metrics.zcoherence_summary(freqs2, p2_final, LOW_CUT, HIGH_CUT)
        its = sorted(set(snapshot_iterations(a)) & set(snapshot_iterations(b)))
        s_low_curve = []
        for i in its:
            fa = os.path.join(a, 'snapshots', f'it_{i:03d}.npy')
            fb = os.path.join(b, 'snapshots', f'it_{i:03d}.npy')
            fr, p2 = metrics.two_seed_spectrum(np.load(fa), np.load(fb),
                                               mask=mask, z_step=Z_STEP)
            s_low_curve.append(metrics.zcoherence_summary(fr, p2, LOW_CUT,
                                                          HIGH_CUT)['S_low'])
        digest[variant] = dict(reference_final=ref_sum, two_seed_final=ts_final,
                               two_seed_iterations=[int(i) for i in its],
                               two_seed_S_low_curve=[float(v) for v in s_low_curve])

        label = DISPLAY.get(variant, variant)
        axes_s[0].plot(freqs2, p2_final, label=label)
        axes_s[1].plot(freqs, p_ref, label=label)
        ax_f.plot(its, s_low_curve, 'o-', label=label)

    for ax, ttl in zip(axes_s, ('two-seed (primary)', 'vs reference')):
        ax.axvspan(0, LOW_CUT, color='#2563eb', alpha=0.08)
        ax.set_xlabel('axial frequency f_z (cycles/slice)')
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        ax.set_title(f'final-iteration axial spectrum, {ttl}')
        ax.legend(fontsize=8)
    axes_s[0].set_ylabel('mean power (z-uncorrelated error = flat)')
    fig_s.suptitle(f'{tag}: axial power spectra P(f_z), final iteration of 15 '
                   f'(shaded = S_low band)')
    fig_s.tight_layout()
    fig_s.savefig(os.path.join(out_dir, 'v2_spectra.png'), dpi=150)
    plt.close(fig_s)

    ax_f.set_xlabel('iteration')
    ax_f.set_ylabel('two-seed $S_{low}$')
    ax_f.set_yscale('log')
    from matplotlib.ticker import NullFormatter
    ax_f.yaxis.set_minor_formatter(NullFormatter())
    ax_f.grid(alpha=0.3)
    ax_f.legend(fontsize=8)
    ax_f.set_title(f'two-seed $S_{{low}}$ across iterations ({tag})', fontsize=11)
    fig_f.tight_layout()
    fig_f.savefig(os.path.join(out_dir, 'v2_formation.png'), dpi=150)
    plt.close(fig_f)

    with open(os.path.join(out_dir, 'v2_digest.json'), 'w') as f:
        json.dump(digest, f, indent=1)

    # Calibration verdict for the damping pair.
    if 'center' in digest and 'center_damp_off' in digest:
        c = digest['center']['two_seed_final']
        d = digest['center_damp_off']['two_seed_final']
        print(f'[{tag}] VERDICT two-seed final: center S_low={c["S_low"]:.4g} '
              f'Rz={c["Rz"]:.2f} | damp_off S_low={d["S_low"]:.4g} '
              f'Rz={d["Rz"]:.2f} | damp_off/center S_low ratio = '
              f'{d["S_low"] / c["S_low"]:.2f}', flush=True)


def main():
    for tag, cfg in ROOTS.items():
        analyze_root(tag, cfg)
    print('v2 pass complete', flush=True)


if __name__ == '__main__':
    main()
