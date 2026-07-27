"""Figures for the beam-hardening page.

Inputs: the real-data probe json + the synthetic run outputs fetched to the
session scratchpad (regenerable from
/scratch/gautschi/buzzard/sharpness_schedule/{bh_probe, hardening_bh,
hardening_cal}/).

Outputs (into findings/figures/):
  bh_transfer_real.png   -- real-scan residual transfer curves, padded vs
                            unpadded, metal coordinate.
  bh_error_panels.png    -- copied synthetic attribution panels.
  bh_severity.png        -- ledger and residual dip vs the severity dial.
  bh_calibrated.png      -- calibrated synthetic vs the real padded scan
                            (built once the calibrated run's finals are local).
"""

import json
import os
import shutil

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRATCHPAD = ('/private/tmp/claude-501/-Users-gbuzzard-Documents-PyCharm-'
               'Projects-Research-mbirjax/53ed36fd-35fd-4d87-b010-bae240ee9094/'
               'scratchpad')
FIG_DIR = os.path.join(_HERE, 'figures')

# From the severity sweep's job log (jobs 14232792; seed 1 = seed 2 to 3 digits).
SEVERITY_TABLE = {
    0.0: dict(dip=+0.0014, ledger=+0.000, s_low=6.1e-6),
    0.5: dict(dip=-0.0373, ledger=-0.079, s_low=9.2e-6),
    1.0: dict(dip=-0.0306, ledger=-0.170, s_low=1.28e-5),
}
REAL_DIP = -0.0136   # padded real scan, top-m bin


def transfer_real():
    probe = json.load(open(os.path.join(_SCRATCHPAD, 'bhprobe', 'bh_probe.json')))
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.3))
    for ax, name, ttl in ((axes[0], 'padded', 'padded 1.502\N{MULTIPLICATION SIGN} (it 59)'),
                          (axes[1], 'unpadded', 'unpadded (it 59)')):
        case = probe[name]['m']
        x = [r['coord_mean'] for r in case['bins']]
        mu = [r['mean'] for r in case['bins']]
        sd = [r['std'] for r in case['bins']]
        ax.axhline(0, color='#888', lw=0.8)
        ax.errorbar(x, mu, yerr=sd, fmt='o-', ms=3, lw=1.3, color='#1d4ed8',
                    ecolor='#b9c9f2', capsize=2)
        ax.set_xlabel('metal sinogram coordinate m (MAR segmentation)')
        ax.set_ylabel('residual  y \N{MINUS SIGN} A x\N{COMBINING CIRCUMFLEX ACCENT}')
        ax.set_title(ttl, fontsize=11)
        ax.set_ylim(-0.16, 0.16)
        ax.grid(alpha=0.3)
    axes[0].annotate('hardening sign:\nmetal-heavy rays dip', xy=(1.7, -0.014),
                     xytext=(0.9, -0.10), fontsize=9,
                     arrowprops=dict(arrowstyle='->', color='#444'))
    fig.suptitle('Real BGA ds3: residual transfer curves vs the metal coordinate '
                 '(shp=1.5, snr=35, seed 1)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.join(FIG_DIR, 'bh_transfer_real.png')
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print('wrote', out)


def severity_figure():
    s = sorted(SEVERITY_TABLE)
    dip = [SEVERITY_TABLE[v]['dip'] for v in s]
    led = [SEVERITY_TABLE[v]['ledger'] for v in s]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.0))
    axes[0].plot(s, dip, 'o-', color='#1d4ed8', lw=1.4)
    axes[0].axhline(REAL_DIP, color='#a4232e', ls='--', lw=1.2,
                    label=f'real padded scan ({REAL_DIP:+.3f})')
    axes[0].set_xlabel('severity dial s')
    axes[0].set_ylabel('residual top-metal-bin mean')
    axes[0].set_title('the residual channel saturates:\nhigher severity deposits '
                      'more, leaves less residual', fontsize=10)
    axes[0].legend(fontsize=9)
    axes[1].plot(s, led, 's-', color='#374151', lw=1.4)
    axes[1].set_xlabel('severity dial s')
    axes[1].set_ylabel('deposited mass, fraction of true mass')
    axes[1].set_title('the signed ledger: hardening REMOVES mass\n'
                      '(truncation deposited +32%)', fontsize=10)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle('Severity sweep (contained geometry, 17 iterations, seed 1)',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = os.path.join(FIG_DIR, 'bh_severity.png')
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print('wrote', out)


def calibrated_figure():
    """Calibrated synthetic (dense grid, s=0.2) next to the real padded scan."""
    cal_path = os.path.join(_SCRATCHPAD, 'cal', 'cal_truncpad_final.npy')
    gt_path = os.path.join(_SCRATCHPAD, 'cal', 'cal_truncpad_gt.npy')
    if not (os.path.exists(cal_path) and os.path.exists(gt_path)):
        print('calibrated finals not local yet; skipping bh_calibrated.png')
        return
    cal = np.load(cal_path)
    gt = np.load(gt_path)
    real = np.load(os.path.join(_SCRATCHPAD, 'e4long', 'finals',
                                'seed1_final_crop.npy'))
    zball = int(round((gt.shape[2] - 1) * 0.35))
    yc = cal.shape[0] // 2
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 9.6))
    err = cal - gt
    lim = float(np.percentile(np.abs(err[:, :, zball]), 99.0))
    im = axes[0, 0].imshow(err[:, :, zball], vmin=-lim, vmax=lim,
                           cmap='seismic', aspect='equal')
    axes[0, 0].set_title('calibrated synthetic: axial error at the ball layer\n'
                         '(s=0.2, dense grid, it 16)', fontsize=10)
    fig.colorbar(im, ax=axes[0, 0], shrink=0.8)
    im = axes[1, 0].imshow(err[yc, :, :].T, vmin=-lim, vmax=lim,
                           cmap='seismic', aspect='equal')
    axes[1, 0].set_title('calibrated synthetic: (x,z) error', fontsize=10)
    fig.colorbar(im, ax=axes[1, 0], shrink=0.8)
    vmax = float(np.percentile(cal, 99.9))
    im = axes[0, 1].imshow(cal[:, :, zball], vmin=0, vmax=vmax, cmap='gray',
                           aspect='equal')
    axes[0, 1].set_title('calibrated synthetic: reconstruction, ball layer',
                         fontsize=10)
    fig.colorbar(im, ax=axes[0, 1], shrink=0.8)
    rmax = float(np.percentile(real, 99.8))
    ryc = real.shape[0] // 2
    im = axes[1, 1].imshow(real[ryc, :, :].T, vmin=0, vmax=rmax, cmap='gray',
                           aspect='equal')
    axes[1, 1].set_title('real padded BGA, (x,z) mid-plane (it 59) \N{EM DASH} '
                         'the family being modeled', fontsize=10)
    fig.colorbar(im, ax=axes[1, 1], shrink=0.8)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('The calibrated case (severity 0.2, dense ball grid) beside the '
                 'real padded scan (shp=1.5, snr=35)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(FIG_DIR, 'bh_calibrated.png')
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print('wrote', out)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    transfer_real()
    severity_figure()
    shutil.copyfile(os.path.join(_SCRATCHPAD, 'bh', 'bh_error_panels.png'),
                    os.path.join(FIG_DIR, 'bh_error_panels.png'))
    print('copied bh_error_panels.png')
    calibrated_figure()


if __name__ == '__main__':
    main()
