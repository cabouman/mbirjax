"""Figures for the converged-streaks page (built from the long-tail two-init
run's volumes, fetched to the session scratchpad; regenerable from
/scratch/gautschi/buzzard/sharpness_schedule/e2_longtail/ + the phantom
builder in mechanism/synthetic_hardening.py).

Outputs (into findings/figures/):
  converged_error_panels.png — final error vs ground truth: axial mid-slice,
      (x,z) mid-plane, and the deposit ledger (ring/interior/total bars).
  converged_convergence.png  — copied 6-panel convergence figure.
"""

import os
import shutil

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRATCHPAD = ('/private/tmp/claude-501/-Users-gbuzzard-Documents-PyCharm-'
               'Projects-Research-mbirjax/53ed36fd-35fd-4d87-b010-bae240ee9094/'
               'scratchpad')
FIG_DIR = os.path.join(_HERE, 'figures')

import sys
sys.path.insert(0, os.path.join(_HERE, '..', 'driver'))
import mbirjax as mj


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    gt = np.load(os.path.join(_SCRATCHPAD, 'hardening_confirm', 'gt_crop.npy'))
    final = np.load(os.path.join(_SCRATCHPAD, 'e2', 'fdk_init', 'seed1',
                                 'final_recon.npy'))
    err = final - gt
    shape = gt.shape
    ror = np.asarray(mj.get_2d_ror_mask(shape)).astype(bool)
    interior = np.asarray(mj.get_2d_ror_mask(
        shape, crop_radius_fraction=0.05)).astype(bool)
    ring = ror & ~interior
    gt_mass = float(gt[ror, :].sum())
    masses = {name: float(err[m2, :].sum() / gt_mass * 100)
              for name, m2 in (('outer ring\n(5% annulus)', ring),
                               ('interior\n(eroded ROR)', interior),
                               ('total\n(full ROR)', ror))}

    err = err * ror[:, :, None]   # blank the never-updated outside-ROR corners
    zc = shape[2] // 2
    yc = shape[0] // 2
    lim = 0.03
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4),
                             gridspec_kw=dict(width_ratios=(1, 1, 0.75)))
    im0 = axes[0].imshow(err[:, :, zc], vmin=-lim, vmax=lim, cmap='seismic',
                         aspect='equal')
    axes[0].set_title('error vs ground truth, axial mid-slice\n'
                      '(iteration 59, shp=1.5, snr=35)', fontsize=10)
    im1 = axes[1].imshow(err[yc, :, :].T, vmin=-lim, vmax=lim, cmap='seismic',
                         aspect='equal')
    axes[1].set_title('error vs ground truth, (x,z) mid-plane\n'
                      '(ball layer at 0.35 of the height)', fontsize=10)
    for ax, im in ((axes[0], im0), (axes[1], im1)):
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.82)

    names = list(masses)
    vals = [masses[n] for n in names]
    bars = axes[2].bar(names, vals, color=('#b45309', '#1d4ed8', '#374151'),
                       width=0.62)
    axes[2].set_ylabel('added mass, % of ground-truth mass')
    axes[2].set_title('where the deposited mass sits', fontsize=10)
    axes[2].grid(axis='y', alpha=0.3)
    for b, v in zip(bars, vals):
        axes[2].text(b.get_x() + b.get_width() / 2, v + 0.5, f'+{v:.1f}%',
                     ha='center', fontsize=9)
    axes[2].set_ylim(0, max(vals) * 1.2)
    fig.suptitle('The converged solution\N{RIGHT SINGLE QUOTATION MARK}s error field '
                 '(FDK-init run, seed 1; window \N{PLUS-MINUS SIGN}0.03 '
                 '\N{ALMOST EQUAL TO} \N{PLUS-MINUS SIGN}3\N{MULTIPLICATION SIGN} slab value; '
                 'the boundary flash saturates)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = os.path.join(FIG_DIR, 'converged_error_panels.png')
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print('wrote', out)

    src = os.path.join(_SCRATCHPAD, 'e2', 'e2_convergence.png')
    dst = os.path.join(FIG_DIR, 'converged_convergence.png')
    shutil.copyfile(src, dst)
    print('copied', dst)


if __name__ == '__main__':
    main()
