"""Phase B analysis: figures + numeric digests for the results page.

Reads the b1/b2 summaries and run records, writes PNGs and digest JSONs into
<root>/analysis for each phase-B root that exists.  Figure conventions follow the
findings page: settings-style legends, quiet log axes, shared intensity windows for
image comparisons, iteration counts in titles.

Run on gautschi (compute node):  python -u b_analysis.py
"""

import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'driver'))

import metrics   # noqa: E402

# ---------------------------------------------------------------- configuration
_SCRATCH = '/scratch/gautschi/buzzard/sharpness_schedule'
B1_ROOT = os.path.join(_SCRATCH, 'b1')
B2_ROOT = os.path.join(_SCRATCH, 'b2')
A2_REF = os.path.join(_SCRATCH, 'a2_bga', 'reference_recon.npy')
A3_REF = os.path.join(_SCRATCH, 'a3_fullres', 'reference_recon.npy')
GATE_IT = 14
FALLBACK_IT = 16
VARIANT_ORDER = ('baseline', 'D2', 'D4', 'S2', 'S4', 'J2', 'J4')
COLORS = {'baseline': 'k', 'D2': '#1d4ed8', 'D4': '#60a5fa', 'S2': '#166534',
          'S4': '#4ade80', 'J2': '#b45309', 'J4': '#fbbf24'}
# -------------------------------------------------------------------------------


def pair_series(vsum, key='S2_low'):
    """{(runA, runB): (iterations, values incl. final at its position)} per pair."""
    out = {}
    for pair in vsum.get('two_seed', {}).get('pairs', []):
        its = list(pair['iterations'])
        vals = [p[key] for p in pair['points']]
        out[tuple(pair['runs'])] = (its, vals, pair['final'][key])
    return out


def value_at(its, vals, it):
    return vals[its.index(it)] if it in its else float('nan')


def variant_stats(vsum, base_stats=None):
    """C1/C2-relevant numbers for one variant."""
    pairs = pair_series(vsum)
    s14 = [value_at(its, vals, GATE_IT) for its, vals, _ in pairs.values()]
    peaks = [max(vals) for _, vals, _ in pairs.values()]
    per_seed = vsum['per_seed']
    obj = {s: (np.asarray(ps['data_term_target'])
               + np.nan_to_num(np.asarray(ps.get('prior_target',
                                                 [np.nan] * len(ps['es_rmse'])))))
           for s, ps in per_seed.items()}
    es = {s: np.asarray(ps['es_rmse']) for s, ps in per_seed.items()}
    nr = {s: np.asarray(ps['nrmse']) for s, ps in per_seed.items()}
    st = dict(
        s2low_14_pairs=s14, s2low_14_mean=float(np.mean(s14)) if s14 else None,
        peak_pairs=peaks, peak_mean=float(np.mean(peaks)) if peaks else None,
        obj_14_mean=float(np.mean([o[GATE_IT] for o in obj.values()])),
        obj_16_mean=float(np.mean([o[FALLBACK_IT] for o in obj.values()
                                   if len(o) > FALLBACK_IT])),
        es_14_mean=float(np.mean([e[GATE_IT] for e in es.values()])),
        es_16_mean=float(np.mean([e[FALLBACK_IT] for e in es.values()
                                  if len(e) > FALLBACK_IT])),
        nrmse_14_mean=float(np.mean([v[GATE_IT] for v in nr.values()])))
    if base_stats:
        st['s2low_ratio_pairs'] = [v / base_stats['s2low_14_mean'] for v in s14]
        st['peak_ratio_mean'] = st['peak_mean'] / base_stats['peak_mean']
        st['obj_gap_14'] = (st['obj_14_mean'] - base_stats['obj_14_mean']) \
            / abs(base_stats['obj_14_mean'])
        st['obj_gap_16v14'] = (st['obj_16_mean'] - base_stats['obj_14_mean']) \
            / abs(base_stats['obj_14_mean'])
        st['es_gap_14'] = (st['es_14_mean'] - base_stats['es_14_mean']) \
            / base_stats['es_14_mean']
        st['es_gap_16v14'] = (st['es_16_mean'] - base_stats['es_14_mean']) \
            / base_stats['es_14_mean']
        st['C1_final'] = bool(s14) and all(
            v <= 0.5 * base_stats['s2low_14_mean'] for v in s14)
        st['C1_peak'] = st['peak_mean'] <= 0.7 * base_stats['peak_mean']
        st['C2_at_14'] = (st['obj_gap_14'] <= 0.005
                          and st['es_gap_14'] <= 0.005)
        st['C2_at_16'] = (st['obj_gap_16v14'] <= 0.005
                          and st['es_gap_16v14'] <= 0.005)
    return st


def footprint_e0(case_dir, variant):
    """Seed-mean interior E(0) at iteration 0 for the diagnostic."""
    vals = []
    for d in sorted(glob.glob(os.path.join(case_dir, variant, 'seed*'))):
        with open(os.path.join(d, 'config.json')) as f:
            seq = json.load(f)['seq']
        rec = np.load(os.path.join(d, 'records.npz'), allow_pickle=True)
        smap = rec['streak_maps'][0]
        interior = metrics.interior_mask(smap.shape)
        e = metrics.footprint_enrichment(smap, rec[f'partition_entry{seq[0]}'],
                                         rec['perms'][0], mask=interior)
        vals.append(float(e[0]))
    return float(np.mean(vals)) if vals else float('nan')


def fig_trajectories(summary, out_dir, title, fname):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for name in VARIANT_ORDER:
        if name not in summary['variants']:
            continue
        pairs = pair_series(summary['variants'][name])
        if not pairs:
            continue
        allv = np.array([vals for _, vals, _ in pairs.values()])
        its = next(iter(pairs.values()))[0]
        mean = allv.mean(axis=0)
        lw, z = (2.2, 5) if name == 'baseline' else (1.4, 3)
        ax.plot(its, mean, 'o-', color=COLORS.get(name, 'gray'), lw=lw, ms=4,
                zorder=z, label=name)
    ax.axvline(GATE_IT, color='#999', lw=0.8, ls=':')
    ax.set_xlabel('iteration')
    ax.set_ylabel('two-seed $S_{low}$ (pair mean)')
    ax.set_yscale('log')
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)


def xz(vol):
    return np.asarray(vol)[vol.shape[0] // 2, :, :].T


def fig_compare_images(dir_a, dir_b, labels, reference, out_path, suptitle,
                       two_seed_dirs=None):
    """Matched-window comparisons: vs-reference error (top pair) and, when
    two_seed_dirs is given ((a1,a2),(b1,b2)), the two-seed fields (bottom pair)."""
    err_a = xz(np.load(os.path.join(dir_a, 'final_recon.npy')) - reference)
    err_b = xz(np.load(os.path.join(dir_b, 'final_recon.npy')) - reference)
    rows = 2 if two_seed_dirs else 1
    fig, axes = plt.subplots(rows, 2, figsize=(11.5, 4.2 * rows),
                             constrained_layout=True, squeeze=False)
    emax = float(np.percentile(np.abs(np.stack([err_a, err_b])), 99.5))
    for ax, img, ttl in ((axes[0][0], err_a, f'{labels[0]}: error vs reference'),
                         (axes[0][1], err_b, f'{labels[1]}: error vs reference')):
        im = ax.imshow(img, vmin=-emax, vmax=emax, cmap='seismic', aspect='equal')
        ax.set_title(ttl, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=list(axes[0]), shrink=0.85)
    if two_seed_dirs:
        (a1, a2), (b1, b2) = two_seed_dirs
        d_a = xz((np.load(os.path.join(a1, 'final_recon.npy'))
                  - np.load(os.path.join(a2, 'final_recon.npy'))) / np.sqrt(2))
        d_b = xz((np.load(os.path.join(b1, 'final_recon.npy'))
                  - np.load(os.path.join(b2, 'final_recon.npy'))) / np.sqrt(2))
        dmax = float(np.percentile(np.abs(np.stack([d_a, d_b])), 99.5))
        for ax, img, ttl in ((axes[1][0], d_a, f'{labels[0]}: two-seed field'),
                             (axes[1][1], d_b, f'{labels[1]}: two-seed field')):
            im = ax.imshow(img, vmin=-dmax, vmax=dmax, cmap='seismic',
                           aspect='equal')
            ax.set_title(ttl, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=list(axes[1]), shrink=0.85)
    fig.suptitle(suptitle, fontsize=11)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def fig_longtail(summary, out_dir, b0):
    lt = summary.get('longtail')
    if not lt:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    base = lt['baseline']['per_seed']
    win_name = [k for k in lt if k != 'baseline'][0]
    win = lt[win_name]['per_seed']
    es_b = np.mean([ps['es_rmse'] for ps in base.values()], axis=0)
    es_w = np.mean([ps['es_rmse'] for ps in win.values()], axis=0)
    n = min(len(es_b), len(es_w))
    gap = (es_w[:n] - es_b[:n]) / es_b[:n]
    ax.plot(np.arange(n), 100 * gap, 'o-', color='#1d4ed8', ms=3,
            label=f'es_rmse gap, {win_name} vs baseline')
    spread = 100 * b0['downsampled']['es_rmse_seed_spread']
    ax.axhspan(-spread, spread, color='#166534', alpha=0.12,
               label='B0 seed spread')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('iteration')
    ax.set_ylabel('relative gap (%)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title('Long tail: the schedule\'s residual gap closes (downsampled, '
                 '40 iterations)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'longtail_merge.png'), dpi=150)
    plt.close(fig)


def analyze_b1():
    if not os.path.exists(os.path.join(B1_ROOT, 'b1_summary.json')):
        print('[b1] no summary; skipped')
        return
    with open(os.path.join(B1_ROOT, 'b1_summary.json')) as f:
        summary = json.load(f)
    out_dir = os.path.join(B1_ROOT, 'analysis')
    os.makedirs(out_dir, exist_ok=True)
    b0 = {}
    b0_path = os.path.join(B1_ROOT, 'b0_calibration.json')
    if os.path.exists(b0_path):
        with open(b0_path) as f:
            b0 = json.load(f)

    base = variant_stats(summary['variants']['baseline'])
    digest = {'baseline': base}
    for name in VARIANT_ORDER[1:]:
        if name in summary['variants']:
            digest[name] = variant_stats(summary['variants'][name], base)
    for name in digest:
        if name in summary['variants']:
            digest[name]['E0_it0'] = footprint_e0(os.path.join(B1_ROOT, 'bga'),
                                                  name)
    with open(os.path.join(out_dir, 'b1_digest.json'), 'w') as f:
        json.dump(digest, f, indent=1)
    print('[b1] digest:', flush=True)
    for name, st in digest.items():
        line = (f"  {name:9s} S2low@14={st['s2low_14_mean']:.4g} "
                f"peak={st['peak_mean']:.4g} es@14={st['es_14_mean']:.6g} "
                f"obj@14={st['obj_14_mean']:.6g} E0={st.get('E0_it0', float('nan')):.2f}")
        if 'C1_final' in st:
            line += (f" | C1f={st['C1_final']} C1p={st['C1_peak']} "
                     f"C2@14={st['C2_at_14']} C2@16={st['C2_at_16']} "
                     f"(objgap {st['obj_gap_14']*100:+.3f}%, "
                     f"esgap {st['es_gap_14']*100:+.3f}%)")
        print(line, flush=True)

    fig_trajectories(summary, out_dir,
                     'Schedule variants vs baseline (downsampled, 17 iterations)',
                     'b1_trajectories.png')
    fig_longtail(summary, out_dir, b0) if b0 else None
    print('[b1] figures written to', out_dir, flush=True)


def analyze_b2():
    if not os.path.exists(os.path.join(B2_ROOT, 'b2_summary.json')):
        print('[b2] no summary; skipped')
        return
    with open(os.path.join(B2_ROOT, 'b2_summary.json')) as f:
        summary = json.load(f)
    out_dir = os.path.join(B2_ROOT, 'analysis')
    os.makedirs(out_dir, exist_ok=True)
    base = variant_stats(summary['variants']['baseline'])
    digest = {'baseline': base}
    for name in summary['variants']:
        if name != 'baseline':
            digest[name] = variant_stats(summary['variants'][name], base)
    with open(os.path.join(out_dir, 'b2_digest.json'), 'w') as f:
        json.dump(digest, f, indent=1)
    print('[b2] digest:', flush=True)
    for name, st in digest.items():
        line = (f"  {name:9s} S2low@14={st['s2low_14_mean']:.4g} "
                f"peak={st['peak_mean']:.4g} es@14={st['es_14_mean']:.6g}")
        if 'C1_final' in st:
            line += (f" | C1f={st['C1_final']} C1p={st['C1_peak']} "
                     f"esgap@14={st['es_gap_14']*100:+.3f}%")
        print(line, flush=True)
    fig_trajectories(summary, out_dir,
                     'Full-resolution confirmation (17 iterations)',
                     'b2_trajectories.png')
    print('[b2] figures written to', out_dir, flush=True)


def image_comparisons():
    """Winner-vs-baseline matched-window images at both scales (run after the
    winner is known; b1 snapshots/finals must still exist)."""
    for root, ref_path, tag in ((B1_ROOT, A2_REF, 'b1'), (B2_ROOT, A3_REF, 'b2')):
        spath = os.path.join(root, f'{tag}_summary.json')
        if not os.path.exists(spath):
            continue
        with open(spath) as f:
            summary = json.load(f)
        names = [n for n in summary['variants'] if n != 'baseline']
        if not names:
            continue
        win = summary.get('config', {}).get('winner')
        # No winner (the b1 outcome): compare the least-harmful and most-harmful
        # variants against baseline for the results page.
        compare = [win] if win in summary['variants'] else \
            [n for n in ('D4', 'S4') if n in summary['variants']] or names[:1]
        case_dir = os.path.join(root, 'bga') if tag == 'b1' else root
        b_dirs = sorted(glob.glob(os.path.join(case_dir, 'baseline', 'seed*')))
        reference = np.load(ref_path)
        for name in compare:
            w_dirs = sorted(glob.glob(os.path.join(case_dir, name, 'seed*')))
            if len(b_dirs) < 2 or len(w_dirs) < 2:
                continue
            fig_compare_images(
                b_dirs[0], w_dirs[0], ('baseline', name), reference,
                os.path.join(root, 'analysis', f'{tag}_{name}_compare.png'),
                f'{"Downsampled" if tag == "b1" else "Full resolution"}: baseline '
                f'vs {name}, matched windows (final iteration; seed 1; two-seed '
                f'from seeds 1−2)',
                two_seed_dirs=((b_dirs[0], b_dirs[1]), (w_dirs[0], w_dirs[1])))
            print(f'[{tag}] comparison written: baseline vs {name}', flush=True)


def main():
    analyze_b1()
    analyze_b2()
    image_comparisons()
    print('b analysis complete', flush=True)


if __name__ == '__main__':
    main()
