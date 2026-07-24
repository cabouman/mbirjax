"""Charts for slice_parity_findings.html (the parity story page).

Runs LOCALLY from the committed metric arrays in results/{r1,r1_z62,r1_lilly_ds4}/
(cropped_lognrmse trajectories + per-slice errs).  Builds two figures:

  parity_trend.png   — NRMSE (percent) of C0 (default) vs C1 (parity-all) at 30
                       iterations, across the three datasets x sharpness {1.0, 2.5},
                       with the percent error reduction annotated: parity's gap over
                       the default grows with sharpness and problem size.
  parity_profile.png — per-slice interior NRMSE, C0 vs C1, on ds4 s2.5 @30 iterations:
                       parity's gain is distributed across the interior, not a hotspot.

The profile needs per-slice reference norms (results/r1_lilly_ds4/
ds4_s2.5_refslicenorms.npy), regenerated on gautschi from the depot reference:
    ref = np.load('/depot/bouman/data/mbirjax_metrics/slice_parity/refs/'
                  'lilly_ds4_ref_sharp2.5.npy')
    np.save('ds4_s2.5_refslicenorms.npy',
            np.linalg.norm(ref, axis=(0, 1)).astype('float32'))

Also prints the error-reduction table used in the HTML.  Figures are written to
OUT_DIR (a scratch/temp dir) and published to depot; NOT committed (no PNGs in repo).

Run:  ~/miniforge3/envs/mbirjax/bin/python plans/experiments/slice_parity/parity_findings_figs.py OUT_DIR
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
DATASETS = [('r1', 'lilly_ds8'), ('r1_z62', 'z62'), ('r1_lilly_ds4', 'lilly_ds4')]
SHARP = [1.0, 2.5]
BUDGETS = [15, 30]                       # iteration counts to report (1-based)
C_LOW = '#2563eb'                        # sharpness 1.0 (blue)
C_HIGH = '#ea580c'                       # sharpness 2.5 (orange)

# Interior axial crop (10% each end) and window for the per-slice profile.
AXIAL_CROP = 0.10


def load_traj(case_dir, name, s):
    d = np.load(os.path.join(RES, case_dir, f'{name}_sharp{s}.npz'))
    return d['cropped_lognrmse'], d['errs']       # (30,), (30, nz)


def reduction_pct(c0_log, c1_log, it):
    """Percent error reduction of C1 vs C0 at 1-based iteration `it`."""
    return (1 - 10 ** (c1_log[it - 1] - c0_log[it - 1])) * 100


def nrmse_pct(logv, it):
    return 10 ** logv[it - 1] * 100


def build_trend(out_png):
    # NRMSE (percent) of default vs parity at 30 iterations, per (dataset, sharpness).
    labels, nr_def, nr_par, colors = [], [], [], []
    print('=== NRMSE percent (default / parity) and reduction ===')
    print(f'{"dataset":11s} {"s":4s}  {"15 def":>7s} {"15 par":>7s}  '
          f'{"30 def":>7s} {"30 par":>7s}  {"red15":>6s} {"red30":>6s}')
    for case_dir, nice in DATASETS:
        for s in SHARP:
            c0, _ = load_traj(case_dir, 'C0_default', s)
            c1, _ = load_traj(case_dir, 'C1_parityall', s)
            labels.append(f'{nice}\ns{s}')
            nr_def.append(nrmse_pct(c0, 30)); nr_par.append(nrmse_pct(c1, 30))
            colors.append(C_LOW if s == 1.0 else C_HIGH)
            print(f'{nice:11s} {s:<4}  {nrmse_pct(c0,15):6.2f}% {nrmse_pct(c1,15):6.2f}%  '
                  f'{nrmse_pct(c0,30):6.2f}% {nrmse_pct(c1,30):6.2f}%  '
                  f'{reduction_pct(c0,c1,15):5.1f}% {reduction_pct(c0,c1,30):5.1f}%')

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 5))
    w = 0.38
    ax.bar(x - w / 2, nr_def, w, color='#9ca3af', edgecolor='white', label='default')
    ax.bar(x + w / 2, nr_par, w, color=colors, edgecolor='white', label='parity-all')
    for xi, (dv, pv) in enumerate(zip(nr_def, nr_par)):
        ax.annotate(f'{dv:.1f}%', (xi - w / 2, dv), ha='center', va='bottom', fontsize=8)
        ax.annotate(f'{pv:.1f}%', (xi + w / 2, pv), ha='center', va='bottom', fontsize=8)
        red = (1 - pv / dv) * 100
        ax.annotate(f'−{red:.0f}%', (xi, max(dv, pv)), ha='center', va='bottom',
                    fontsize=8.5, fontweight='bold', color='#166534',
                    xytext=(0, 12), textcoords='offset points')
    ax.set_ylabel('NRMSE vs reference (%)')
    ax.set_title('NRMSE at 30 iterations: parity’s gap over the default grows with '
                 'sharpness and problem size', fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.grid(axis='y', color='#ddd', lw=0.7)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(nr_def) * 1.18)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    from matplotlib.patches import Patch
    leg = [Patch(facecolor='#9ca3af', label='default'),
           Patch(facecolor=C_LOW, label='parity-all, sharpness 1.0'),
           Patch(facecolor=C_HIGH, label='parity-all, sharpness 2.5')]
    ax.legend(handles=leg, frameon=False, fontsize=9, loc='upper left')
    ax.text(0.995, 0.97, 'green = parity’s error reduction vs default',
            transform=ax.transAxes, ha='right', va='top', fontsize=8.5, color='#166534')
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print('built', out_png)


def build_profile(out_png):
    # Per-slice NRMSE = ||recon_z - ref_z|| / ||ref_z||.  The saved errs are full-slice
    # ||recon_z - ref_z||; ref_z norms are computed once (same full-slice convention).
    _, c0_errs = load_traj('r1_lilly_ds4', 'C0_default', 2.5)
    _, c1_errs = load_traj('r1_lilly_ds4', 'C1_parityall', 2.5)
    refnorm = np.load(os.path.join(RES, 'r1_lilly_ds4', 'ds4_s2.5_refslicenorms.npy'))
    nz = c0_errs.shape[1]
    z0, z1 = int(np.ceil(nz * AXIAL_CROP)), nz - int(np.ceil(nz * AXIAL_CROP))
    z = np.arange(z0, z1)
    nr0 = c0_errs[29, z0:z1] / refnorm[z0:z1] * 100
    nr1 = c1_errs[29, z0:z1] / refnorm[z0:z1] * 100
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(z, nr0, color='#555', lw=1.3, label='default')
    ax.plot(z, nr1, color=C_HIGH, lw=1.3, label='parity-all')
    ax.set_xlabel('slice z (interior; 10% excluded at each axial end)')
    ax.set_ylabel('per-slice NRMSE (%)')
    ax.set_title('lilly_ds4, sharpness 2.5, 30 iterations: parity lowers NRMSE across '
                 'the whole interior', fontsize=12)
    ax.grid(color='#eee', lw=0.7); ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    med0, med1 = np.median(nr0), np.median(nr1)
    ax.legend(frameon=False, fontsize=10, loc='upper center')
    ax.text(0.99, 0.95, f'interior median NRMSE: default {med0:.1f}%  vs  '
            f'parity {med1:.1f}%  ({(1 - med1 / med0) * 100:.0f}% lower)',
            transform=ax.transAxes, ha='right', va='top', fontsize=9, color='#333')
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print('built', out_png)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else '/tmp/parity_figs'
    os.makedirs(out_dir, exist_ok=True)
    build_trend(os.path.join(out_dir, 'parity_trend.png'))
    build_profile(os.path.join(out_dir, 'parity_profile.png'))
    print('figures in', out_dir)


if __name__ == '__main__':
    main()
