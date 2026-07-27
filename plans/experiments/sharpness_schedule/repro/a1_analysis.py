"""A1 analysis: turn the sweep outputs into figures + a compact digest.

Runs NEXT TO THE DATA (gautschi; no GPU needed) and writes PNG figures + a
metrics digest into <sweep>/analysis/, small enough to rsync back for the findings
page.  Reads only the sweep outputs (sweep_summary.json, per-run records.npz,
snapshot volumes, gt_phantom.npy).

Figures:
  1. severity_axes.png    - final two-seed S2 (primary) + reference S vs sharpness
                            (at fixed snr_db) and vs snr_db (at fixed sharpness),
                            with the z-incoherent controls.
  2. collapse_diagonal.png- S2 along the balance-matched diagonal (plan pred. 3:
                            flat = P-weak; decreasing = saturation participates).
  3. formation_curves.png - per-iteration reference S and snapshot-iteration S2 for
                            the key variants (formation + decay; plan pred. 1),
                            including the long-tail variant.
  4. footprint.png        - enrichment E(rank) for iterations 0-2 (plan pred. 2),
                            averaged over seeds, for key variants.
  5. panels_<variant>.png - mid-axial + mid-(x,z) slices of the final recon and the
                            error volume for visual validation of the metric.

Run:  python -u a1_analysis.py   (config constants below; no CLI args)
"""

import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'driver'))
import metrics  # noqa: E402

# ---------------------------------------------------------------- configuration
# Every sweep root that exists is analyzed (a1 synthetic, a2 BGA, mechanism
# probes); each uses whichever reference volume its sweep saved (the ground truth
# phantom, or the converged reference on real data).
_SCRATCH = '/scratch/gautschi/buzzard/sharpness_schedule'
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_ROOTS = (os.path.join(_SCRATCH, 'a1'),
                    os.path.join(_SCRATCH, 'a2_bga'),
                    os.path.join(_SCRATCH, 'mechanism'),
                    os.path.join(_HERE, 'output', 'a1_smoke'))
REFERENCE_FILES = ('gt_phantom.npy', 'reference_recon.npy')
PANEL_VARIANTS = ('center', 'center_damp_off', 'center_noise_off',
                  'sharp3.0', 'sharp0.0', 'coarse_late', 'q2_control')
FOOTPRINT_VARIANTS = ('center', 'center_damp_off', 'sharp3.0', 'coarse_late')
FORMATION_VARIANTS = ('center', 'center_damp_off', 'sharp3.0', 'sharp0.0',
                      'snr45', 'center_long', 'coarse_late', 'q2_control')

# Module state set per root by main() (the fig functions read these).
SWEEP_ROOT = None
OUT_DIR = None
# -------------------------------------------------------------------------------


def load_summary():
    for name in ('sweep_summary.json', 'probes_summary.json'):
        path = os.path.join(SWEEP_ROOT, name)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError(f'no summary json in {SWEEP_ROOT}')


def run_dirs(variant):
    return sorted(glob.glob(os.path.join(SWEEP_ROOT, variant, 'seed*')))


def final_scores(vsum):
    """(mean reference S, mean control, two-seed S2, two-seed control) at the final
    iteration for one variant summary."""
    s_final = np.mean([ps['S'][-1] for ps in vsum['per_seed'].values()])
    c_final = np.mean([ps['control'][-1] for ps in vsum['per_seed'].values()])
    ts = vsum.get('two_seed', {})
    return s_final, c_final, ts.get('final_S2', np.nan), ts.get('final_control2', np.nan)


def fig_severity_axes(summary):
    variants = summary['variants']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    def series(names, xvals, ax, xlabel):
        rows = [(x, *final_scores(variants[n])) for n, x in zip(names, xvals)
                if n in variants]
        if not rows:
            return
        x, S, C, S2, C2 = map(np.asarray, zip(*rows))
        ax.plot(x, S2, 'o-', label='two-seed S2 (primary)')
        ax.plot(x, C2, 'o--', alpha=0.6, label='two-seed control')
        ax.plot(x, S, 's-', label='reference S')
        ax.plot(x, C, 's--', alpha=0.6, label='reference control')
        ax.set_xlabel(xlabel)
        ax.set_yscale('log')
        ax.grid(alpha=0.3)

    sharp_names = ['sharp0.0', 'sharp1.0', 'center', 'sharp2.0', 'sharp3.0']
    sharp_x = [0.0, 1.0, 1.5, 2.0, 3.0]
    series(sharp_names, sharp_x, axes[0], 'sharpness  (snr_db = 35)')
    snr_names = ['snr25', 'snr30', 'center', 'snr40', 'snr45']
    snr_x = [25, 30, 35, 40, 45]
    series(snr_names, snr_x, axes[1], 'snr_db  (sharpness = 1.5)')
    axes[0].set_ylabel('final streak energy')
    axes[0].legend(fontsize=8)
    fig.suptitle('Synthetic severity vs regularization (final iteration of 15; '
                 'two-seed from seeds 1-2)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'severity_axes.png'), dpi=150)
    plt.close(fig)


def fig_collapse(summary):
    variants = summary['variants']
    names = ['diag_t-1', 'center', 'diag_t+1', 'diag_t+2']
    tvals = [-1.0, 0.0, 1.0, 2.0]
    rows = [(t, *final_scores(variants[n])) for n, t in zip(names, tvals)
            if n in variants]
    if not rows:
        return
    t, S, C, S2, C2 = map(np.asarray, zip(*rows))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(t, S2, 'o-', label='two-seed S2 (primary)')
    ax.plot(t, C2, 'o--', alpha=0.6, label='two-seed control')
    ax.plot(t, S, 's-', alpha=0.8, label='reference S')
    ax.set_xlabel('t along balance diagonal (sharpness = 1.5 + t, '
                  'snr_db = 35 - 6.02 t)')
    ax.set_ylabel('final streak energy')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title('Balance-collapse test (flat = P-weak; decreasing = saturation)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'collapse_diagonal.png'), dpi=150)
    plt.close(fig)


def fig_formation(summary):
    variants = summary['variants']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name in FORMATION_VARIANTS:
        if name not in variants:
            continue
        vsum = variants[name]
        s_mean = np.mean([ps['S'] for ps in vsum['per_seed'].values()], axis=0)
        axes[0].plot(np.arange(len(s_mean)), s_mean, '-', label=name)
        ts = vsum.get('two_seed')
        if ts:
            axes[1].plot(ts['iterations'], ts['S2'], 'o-', label=name)
    for ax, ttl in zip(axes, ('reference S(i), seed-mean', 'two-seed S2 at snapshots')):
        ax.set_xlabel('iteration')
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        ax.set_title(ttl)
        ax.legend(fontsize=7)
    axes[0].set_ylabel('streak energy')
    fig.suptitle('Formation / decay')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'formation_curves.png'), dpi=150)
    plt.close(fig)


def footprint_curves(run_dir, iterations=(0, 1, 2)):
    """Enrichment E(rank) per iteration for one run, from its records.npz.

    Computed over the eroded interior: the bright band near the support boundary
    carries a large, rank-independent share of the map energy and dilutes E(r)."""
    with open(os.path.join(run_dir, 'config.json')) as f:
        seq = json.load(f)['seq']
    rec = np.load(os.path.join(run_dir, 'records.npz'), allow_pickle=True)
    interior = None
    out = {}
    for i in iterations:
        if i >= len(seq):
            continue
        part = rec[f'partition_entry{seq[i]}']
        smap = rec['streak_maps'][i]
        if interior is None:
            interior = metrics.interior_mask(smap.shape)
        out[i] = metrics.footprint_enrichment(smap, part, rec['perms'][i],
                                              mask=interior)
    return out


def fig_footprint():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    for ax, it in zip(axes, (0, 1, 2)):
        for name in FOOTPRINT_VARIANTS:
            curves = []
            for rd in run_dirs(name):
                c = footprint_curves(rd, iterations=(it,))
                if it in c:
                    curves.append(c[it])
            if curves:
                display = {'center': 'shp=1.5, snr=35',
                           'center_damp_off': 'shp=1.5, snr=35, no damp',
                           'sharp3.0': 'shp=3.0, snr=35',
                           'coarse_late': 'coarse-late probe'}.get(name, name)
                mean_curve = np.mean(curves, axis=0)
                ax.plot(np.arange(len(mean_curve)), mean_curve, 'o-', ms=3,
                        label=f'{display} (n={len(curves)})')
        ax.axhline(1.0, color='k', lw=0.8, alpha=0.5)
        ax.set_title(f'iteration {it}')
        ax.set_xlabel('update rank r')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('enrichment E(r)')
    axes[0].legend(fontsize=7)
    fig.suptitle('Streak-footprint enrichment by subset update rank '
                 '(seed-mean, eroded interior; 15-iteration runs)', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'footprint.png'), dpi=150)
    plt.close(fig)


def fig_panels(gt_phantom):
    for name in PANEL_VARIANTS:
        dirs = run_dirs(name)
        if not dirs:
            continue
        with open(os.path.join(dirs[0], 'config.json')) as f:
            vcfg = json.load(f).get('variant', {})
        settings = (f"sharpness {vcfg.get('sharpness', '?')}, "
                    f"snr_db {vcfg.get('snr_db', '?')}, "
                    f"{vcfg.get('max_iterations', 15)} iterations")
        vol = np.load(os.path.join(dirs[0], 'final_recon.npy'))
        err = vol - gt_phantom
        zc = vol.shape[2] // 2
        yc = vol.shape[0] // 2
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 9))
        v = np.percentile(gt_phantom, 99.9)
        e = max(np.percentile(np.abs(err), 99.9), 1e-8)
        for ax, img, ttl, kw in (
                (axes[0, 0], vol[:, :, zc], f'{name} final, axial mid-slice',
                 dict(vmin=0, vmax=v, cmap='gray')),
                (axes[0, 1], err[:, :, zc], 'error, axial',
                 dict(vmin=-e, vmax=e, cmap='seismic')),
                (axes[1, 0], vol[yc, :, :].T, 'final, (x,z) mid-plane',
                 dict(vmin=0, vmax=v, cmap='gray')),
                (axes[1, 1], err[yc, :, :].T, 'error, (x,z)',
                 dict(vmin=-e, vmax=e, cmap='seismic'))):
            im = ax.imshow(img, **kw)
            ax.set_title(ttl, fontsize=9)
            ax.axis('off')
            fig.colorbar(im, ax=ax, shrink=0.7)
        fig.suptitle(f'{name}  ({settings})', fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f'panels_{name}.png'), dpi=140)
        plt.close(fig)


def analyze_root(root):
    global SWEEP_ROOT, OUT_DIR
    SWEEP_ROOT = root
    OUT_DIR = os.path.join(root, 'analysis')
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = load_summary()
    print(f'analysis of {root} ({len(summary["variants"])} variants)', flush=True)

    is_a1 = 'sharp0.0' in summary['variants']   # the axes/diagonal exist only in A1
    if is_a1:
        fig_severity_axes(summary)
        fig_collapse(summary)
    fig_formation(summary)
    fig_footprint()
    reference = None
    for name in REFERENCE_FILES:
        path = os.path.join(root, name)
        if os.path.exists(path):
            reference = np.load(path)
            break
    if reference is not None:
        fig_panels(reference)

    # Compact digest: final scores per variant (for the findings page tables).
    digest = {}
    for name, vsum in summary['variants'].items():
        S, C, S2, C2 = final_scores(vsum)
        digest[name] = dict(sharpness=vsum.get('sharpness'),
                            snr_db=vsum.get('snr_db'),
                            noise=vsum.get('noise'), damping=vsum.get('damping'),
                            final_reference_S=float(S), final_control=float(C),
                            final_two_seed_S2=None if np.isnan(S2) else float(S2),
                            final_two_seed_control=None if np.isnan(C2) else float(C2))
    with open(os.path.join(OUT_DIR, 'digest.json'), 'w') as f:
        json.dump(digest, f, indent=1)
    print('  figures:', sorted(os.listdir(OUT_DIR)), flush=True)


def main():
    for root in _CANDIDATE_ROOTS:
        if os.path.isdir(root) and (
                os.path.exists(os.path.join(root, 'sweep_summary.json'))
                or os.path.exists(os.path.join(root, 'probes_summary.json'))):
            analyze_root(root)


if __name__ == '__main__':
    main()
