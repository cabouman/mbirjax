"""Follow-up-wave figures for the converged-streaks page.

Inputs: the sweep summary (sweep_mass/sweep_summary.json values inlined below
from the job log — regenerable from scratch), the long-pair records fetched to
the session scratchpad, and the padded-vs-unpadded BGA comparison built there.

Outputs (into findings/figures/):
  sweep_partition.png — deposit partition + expression scale vs sharpness.
  e1_longpair.png     — real-scan long pair: severity vs conservative ref and
                        two-seed decay over 60 iterations.
  pad_vs_unpad_bga.png — copied from the scratchpad build.
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

# Sweep ledger (job 14231880 + the two-start run's sharpness-1.5 point).
SWEEP = {
    0.0:  dict(total=31.5, ring=15.3, interior=16.2, fine=0.518, s_low=1.14e-4),
    0.75: dict(total=31.8, ring=16.2, interior=15.6, fine=0.736, s_low=3.04e-4),
    1.5:  dict(total=32.1, ring=17.3, interior=14.8, fine=0.855, s_low=6.24e-4),
    2.25: dict(total=32.4, ring=18.3, interior=14.1, fine=0.917, s_low=1.14e-3),
    3.0:  dict(total=32.6, ring=19.0, interior=13.6, fine=0.946, s_low=1.84e-3),
}


def sweep_figure():
    s = sorted(SWEEP)
    get = lambda k: [SWEEP[v][k] for v in s]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    ax = axes[0]
    ax.plot(s, get('total'), 'o-', color='#374151', label='total deposit')
    ax.plot(s, get('ring'), 's-', color='#b45309', label='outer ring (5%)')
    ax.plot(s, get('interior'), 'd-', color='#1d4ed8', label='interior')
    ax.set_xlabel('sharpness (snr_db 35)')
    ax.set_ylabel('added mass, % of ground-truth mass')
    ax.set_title('the deposit is conserved; its split barely moves', fontsize=11)
    ax.set_ylim(0, 36)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax = axes[1]
    ax.plot(s, get('fine'), 'o-', color='#0a7a3d',
            label='interior fine-scale fraction')
    ax.set_xlabel('sharpness (snr_db 35)')
    ax.set_ylabel('fine-scale fraction of interior deposit', color='#0a7a3d')
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.semilogy(s, get('s_low'), '^--', color='#a4232e',
                 label='streak severity S$_{low}$ vs truth')
    ax2.set_ylabel('S$_{low}$ vs ground truth (log)', color='#a4232e')
    ax.set_title('what moves is the expression scale — and severity tracks it',
                 fontsize=11)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], fontsize=9,
              loc='center right')
    fig.suptitle('Sharpness sweep of the truncation deposit '
                 '(hardened synthetic, 60 iterations, seed 1)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.join(FIG_DIR, 'sweep_partition.png')
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print('wrote', out)


def e1_figure():
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    for base, color, lab in (('e1', '#1f77b4', 'unpadded'),
                             ('e4long', '#0a7a3d', 'padded (even-delta 1.502)')):
        for seed, ls in ((1, '-'), (2, '--')):
            r = np.load(os.path.join(_SCRATCHPAD, base, f'seed{seed}',
                                     'records.npz'), allow_pickle=True)
            its = np.arange(len(r['S_low']))
            axes[0].semilogy(its, r['S_low'], ls, color=color,
                             label=(lab if seed == 1 else None), lw=1.4)
        ts = json.load(open(os.path.join(_SCRATCHPAD, base, 'two_seed.json')))
        pair = ts['pairs'][0]
        axes[1].semilogy(pair['iterations'],
                         [p['S2_low'] for p in pair['points']], 'o-',
                         color=color, lw=1.4, ms=3, label=lab)
    axes[0].axvline(14, color='#888', lw=0.8, ls=':')
    axes[0].text(14.5, 0.09, 'default budget', fontsize=8, color='#666')
    axes[0].set_title('severity vs the conservative reference:\n'
                      'unpadded grows \N{MULTIPLICATION SIGN}10 from its '
                      'minimum; padded stays flat (\N{PLUS-MINUS SIGN}5%)',
                      fontsize=11)
    axes[0].set_ylabel('S$_{low}$ vs conservative reference (log)')
    axes[1].set_title('the seed-dependent transient dies in both \N{EM DASH} '
                      '3\N{MULTIPLICATION SIGN} lower padded', fontsize=11)
    axes[1].set_ylabel('two-seed S$_{low}$ (log)')
    for ax in axes:
        ax.set_xlabel('iteration')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle('The real scan, 60-iteration pairs (downsampled BGA baseline, '
                 'shp=1.5, snr=35): unpadded vs laterally padded', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = os.path.join(FIG_DIR, 'e1_longpair.png')
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print('wrote', out)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    sweep_figure()
    e1_figure()
    src = os.path.join(_SCRATCHPAD, 'e4bga', 'pad_vs_unpad_bga.png')
    dst = os.path.join(FIG_DIR, 'pad_vs_unpad_bga.png')
    shutil.copyfile(src, dst)
    print('copied', dst)


if __name__ == '__main__':
    main()
