"""Full-resolution formation/persistence figure for findings section 3c.

Needs only the two small summary JSONs (no volumes), so it can run anywhere:
  - a3_fullres/sweep_summary.json  (full-res per-iteration S_low + two-seed points)
  - a2_bga/analysis/v2_digest.json (downsampled two-seed S_low curve, for contrast)

Run:  python fig_a3_local.py [after placing the JSONs per the paths below]
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRATCH = '/scratch/gautschi/buzzard/sharpness_schedule'
_SCRATCHPAD = ('/private/tmp/claude-501/-Users-gbuzzard-Documents-PyCharm-Projects-'
               'Research-mbirjax/53ed36fd-35fd-4d87-b010-bae240ee9094/scratchpad')

# First existing path wins (cluster canonical; local session copies as fallback).
A3_SUMMARY_PATHS = (os.path.join(_SCRATCH, 'a3_fullres', 'sweep_summary.json'),
                    os.path.join(_SCRATCHPAD, 'a3_summary.json'))
A2_DIGEST_PATHS = (os.path.join(_SCRATCH, 'a2_bga', 'analysis', 'v2_digest.json'),
                   os.path.join(_SCRATCHPAD, 'a2_v2_digest.json'))
OUT_PATH = os.path.join(_HERE, 'figures', 'a3_fullres_formation.png')


def _load(paths):
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    raise FileNotFoundError(paths)


def main():
    a3 = _load(A3_SUMMARY_PATHS)['variants']
    a2 = _load(A2_DIGEST_PATHS)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    common_ylim = (1.2e-3, 4.5e-2)   # both panels on the same vertical scale

    from matplotlib.ticker import NullFormatter
    ax = axes[0]
    for name, label, color in (
            ('center', 'full res, shp=1.5, snr=35', '#1d4ed8'),
            ('center_damp_off', 'full res, shp=1.5, snr=35, no damp', '#d97706')):
        ts = a3[name]['two_seed']
        ax.plot(ts['iterations'], [p['S2_low'] for p in ts['points']], 'o-',
                color=color, label=label)
    c2 = a2['center']
    ax.plot(c2['two_seed_iterations'], c2['two_seed_S_low_curve'], 's--',
            color='#166534', alpha=0.8, label='downsampled, shp=1.5, snr=35')
    ax.set_xlabel('iteration')
    ax.set_ylabel('$S_{low}$')
    ax.set_yscale('log')
    ax.set_ylim(*common_ylim)
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title('partition-driven severity (two-seed)', fontsize=11)

    ax = axes[1]
    for name, label, color in (
            ('center', 'full res, shp=1.5, snr=35', '#1d4ed8'),
            ('center_damp_off', 'full res, shp=1.5, snr=35, no damp', '#d97706')):
        s_low = np.mean([ps['S_low'] for ps in a3[name]['per_seed'].values()], axis=0)
        ax.plot(np.arange(len(s_low)), s_low, 'o-', ms=3, color=color, label=label)
    ax.set_xlabel('iteration')
    ax.set_yscale('log')
    ax.set_ylim(*common_ylim)
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title('total error vs the reference (seed-mean)', fontsize=11)

    fig.suptitle('Full resolution: larger and slower-healing (15 iterations; '
                 'reference: shp=0, snr=30, 60 iterations)', fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print('wrote', OUT_PATH)


if __name__ == '__main__':
    main()
