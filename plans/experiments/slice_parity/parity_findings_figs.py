"""Charts for slice_parity_findings.html (the parity story page).

Runs LOCALLY from the committed metric arrays in results/{r1,r1_z62,r1_lilly_ds4}/
(cropped_lognrmse trajectories + per-slice errs).  Builds two figures:

  parity_trend.png   — C1 (parity-all) error reduction vs C0 (default) at the same
                       iteration count, across the three datasets x sharpness {1.0, 2.5}
                       at the 15- and 30-iteration budgets: the "grows with sharpness,
                       iterations, and problem size" hero chart.
  parity_profile.png — per-slice interior error, C0 vs C1, on ds4 s2.5 @30 iterations:
                       parity's gain is distributed across the interior, not a hotspot.

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


def build_trend(out_png):
    # rows: (dataset, sharpness); value: reduction at 15 and 30 iters.
    labels, red15, red30, colors = [], [], [], []
    print('=== C1 (parity-all) error reduction vs C0, percent ===')
    for case_dir, nice in DATASETS:
        for s in SHARP:
            c0, _ = load_traj(case_dir, 'C0_default', s)
            c1, _ = load_traj(case_dir, 'C1_parityall', s)
            r15, r30 = reduction_pct(c0, c1, 15), reduction_pct(c0, c1, 30)
            labels.append(f'{nice}\ns{s}')
            red15.append(r15); red30.append(r30)
            colors.append(C_LOW if s == 1.0 else C_HIGH)
            print(f'  {nice:10s} s{s}:  15 it {r15:5.1f}%   30 it {r30:5.1f}%')

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 5))
    w = 0.38
    b15 = ax.bar(x - w / 2, red15, w, color=colors, alpha=0.55,
                 edgecolor='white', label='15 iterations')
    b30 = ax.bar(x + w / 2, red30, w, color=colors, alpha=1.0,
                 edgecolor='white', label='30 iterations')
    for bars in (b15, b30):
        for b in bars:
            ax.annotate(f'{b.get_height():.0f}%',
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('error reduction vs default (%)')
    ax.set_title('Parity-all reduces error most at high sharpness, more iterations, '
                 'and larger problems', fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(0, color='#888', lw=0.8)
    ax.grid(axis='y', color='#ddd', lw=0.7)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    # legend: shade = iterations; color = sharpness (annotate separately)
    from matplotlib.patches import Patch
    leg = [Patch(facecolor='#888', alpha=0.55, label='15 iterations'),
           Patch(facecolor='#888', alpha=1.0, label='30 iterations'),
           Patch(facecolor=C_LOW, label='sharpness 1.0'),
           Patch(facecolor=C_HIGH, label='sharpness 2.5')]
    ax.legend(handles=leg, ncol=2, frameon=False, fontsize=9, loc='upper left')
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print('built', out_png)


def build_profile(out_png):
    c0_log, c0_errs = load_traj('r1_lilly_ds4', 'C0_default', 2.5)
    c1_log, c1_errs = load_traj('r1_lilly_ds4', 'C1_parityall', 2.5)
    nz = c0_errs.shape[1]
    z0, z1 = int(np.ceil(nz * AXIAL_CROP)), nz - int(np.ceil(nz * AXIAL_CROP))
    z = np.arange(z0, z1)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(z, c0_errs[29, z0:z1], color='#555', lw=1.3, label='default')
    ax.plot(z, c1_errs[29, z0:z1], color=C_HIGH, lw=1.3, label='parity-all')
    ax.set_xlabel('slice z (interior; 10% excluded at each axial end)')
    ax.set_ylabel('per-slice error  ||recon - reference||')
    ax.set_title('lilly_ds4, sharpness 2.5, 30 iterations: parity lowers error across '
                 'the whole interior', fontsize=12)
    ax.grid(color='#eee', lw=0.7); ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    med0 = np.median(c0_errs[29, z0:z1]); med1 = np.median(c1_errs[29, z0:z1])
    ax.legend(frameon=False, fontsize=10, loc='upper center')
    ax.text(0.99, 0.95, f'interior median: default {med0:.3f}  vs  parity {med1:.3f}  '
            f'({(1 - med1 / med0) * 100:.0f}% lower)',
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
