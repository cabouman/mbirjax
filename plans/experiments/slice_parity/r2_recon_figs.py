"""Build the default-vs-skip-0 head-to-head slice figures from the volumes that
r2_recon_capture.py staged (run on gautschi AFTER the capture job; CPU-only).

For each (dataset, sharpness, iteration): a figure with two view rows (axial
mid-slice, coronal mid-cut) and columns [150-it reference (if present) | default |
skip 0 | difference].  The grayscale window is FIXED PER DATASET: the 5th-95th
percentile of that dataset's sharpness-1.0 reference volume; the figure title shows
the window and the full data range.  The difference panel is symmetric with its limit
shown both absolutely and as a percent of the 95% value.  Also builds the
center-slice (fan-plane) zoom for lilly_ds4 s2.0.

Outputs PNGs to OUT_DIR only.  The narrative page is hand-maintained at
plans/slice_parity/r2_recon_compare.html and published (with these figures) at
/depot/bouman/www/mbirjax/skip_0_results/ — figures are NOT committed to the repo
(Greg 2026-07-12: no PNGs in the repo; they live on depot).

Run:  python -u plans/experiments/slice_parity/r2_recon_figs.py   (login node is fine)
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
STAGE = '/scratch/gautschi/buzzard/parity_recons'
REF_DIRS = {'lilly_ds8': '/home/buzzard/parity_lilly',
            'z62': '/scratch/gautschi/buzzard/parity_z62',
            'lilly_ds4': '/scratch/gautschi/buzzard/parity_lilly_ds4'}
OUT_DIR = os.path.join(STAGE, 'figs')
CASES = ['lilly_ds8', 'z62', 'lilly_ds4']
SHARPNESS_LIST = [1.0, 2.0]
FIG_ITERS = [15, 20]
ARMS = ['D0_default', 'D1_g2start']
ARM_LABELS = {'D0_default': 'default [0,2,4,6,7]',
              'D1_g2start': 'skip 0 [2,4,6,7]'}
WINDOW_PCT = (5, 95)                # fixed per-dataset window percentiles


def dataset_window(case):
    """Per-dataset display window from the sharpness-1.0 reference volume."""
    ref = np.load(os.path.join(REF_DIRS[case], 'ref_sharp1.0.npy'))
    vmin, vmax = np.percentile(ref, WINDOW_PCT)
    return float(vmin), float(vmax), float(ref.min()), float(ref.max())


def mid_slices(vol):
    """(axial mid-slice, coronal mid-cut) as 2-D arrays with z vertical for coronal."""
    axial = vol[:, :, vol.shape[2] // 2]
    coronal = vol[:, vol.shape[1] // 2, :].T      # (slices, rows): z runs vertically
    return {'axial (z mid)': axial, 'coronal (y mid)': coronal}


def build_fig(case, s, it, window, out_png):
    vmin, vmax, dmin, dmax = window
    vols = {a: np.load(os.path.join(STAGE, f'{case}_{a}_s{s}_it{it}.npy'))
            for a in ARMS}
    ref_path = os.path.join(REF_DIRS[case], f'ref_sharp{s}.npy')
    ref = np.load(ref_path) if os.path.exists(ref_path) else None

    cols = ([('reference (150 it)', ref)] if ref is not None else []) + \
           [(ARM_LABELS[a], vols[a]) for a in ARMS] + [('diff', None)]
    views = mid_slices(vols[ARMS[0]]).keys()
    fig, axes = plt.subplots(2, len(cols), figsize=(3.2 * len(cols), 6.6))
    for i, view in enumerate(views):
        d0v = mid_slices(vols[ARMS[0]])[view]
        d1v = mid_slices(vols[ARMS[1]])[view]
        for j, (label, vol) in enumerate(cols):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            if label == 'diff':
                diff = d1v - d0v
                lim = np.percentile(np.abs(diff), 99.9) or 1e-6
                ax.imshow(diff, cmap='coolwarm', vmin=-lim, vmax=lim)
                ax.set_title(f'skip 0 − default '
                             f'(±{lim:.2g} = {lim / vmax * 100:.1f}% of the 95% value)',
                             fontsize=8)
            else:
                ax.imshow(mid_slices(vol)[view], cmap='gray', vmin=vmin, vmax=vmax)
                ax.set_title(label if i == 0 else '', fontsize=9)
            if j == 0:
                ax.set_ylabel(view, fontsize=9)
    fig.suptitle(f'{case}  sharpness {s}  —  {it} iterations   |   '
                 f'window [5%, 95%] = [{vmin:.3g}, {vmax:.3g}],  '
                 f'data range [{dmin:.3g}, {dmax:.3g}]', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def build_zoom(out_png):
    """Center-slice (fan-plane) zoom for lilly_ds4 s2.0 at 15 iterations."""
    d0 = np.load(f'{STAGE}/lilly_ds4_D0_default_s2.0_it15.npy')
    d1 = np.load(f'{STAGE}/lilly_ds4_D1_g2start_s2.0_it15.npy')
    n0, n1, nz = d0.shape
    ii = np.arange(n0)[:, None] - (n0 - 1) / 2
    jj = np.arange(n1)[None, :] - (n1 - 1) / 2
    disk = np.sqrt(ii ** 2 + jj ** 2) < 0.85 * (min(n0, n1) / 2)
    zc = 330
    sl0, sl1 = d0[:, :, zc], d1[:, :, zc]
    vmin, vmax = np.percentile(sl0[disk], [25, 90])   # deliberately tight window
    zs = np.arange(280, 380)
    ax2 = {}
    for name, v in (('default', d0), ('skip 0', d1)):
        ax2[name] = [np.sqrt(np.mean(
            ((v[:, :, z] - 0.5 * (v[:, :, z - 1] + v[:, :, z + 1]))[disk]) ** 2)) * 1e3
            for z in zs]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, (label, sl) in zip(axes[:2], [('default [0,2,4,6,7]', sl0),
                                          ('skip 0 [2,4,6,7]', sl1)]):
        ax.imshow(sl, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'{label}  z={zc}', fontsize=10)
    diff = sl1 - sl0
    lim = np.percentile(np.abs(diff), 99.9)
    axes[2].imshow(diff, cmap='coolwarm', vmin=-lim, vmax=lim)
    axes[2].set_xticks([]); axes[2].set_yticks([])
    axes[2].set_title(f'skip 0 − default (±{lim:.2g} = {lim / vmax * 100:.1f}% '
                      f'of the 95% value)', fontsize=9)
    axes[3].plot(zs, ax2['default'], label='default', lw=1.2)
    axes[3].plot(zs, ax2['skip 0'], label='skip 0', lw=1.2, ls='--')
    axes[3].axvline(zc, color='gray', lw=0.7, ls=':')
    axes[3].set_xlabel('slice z')
    axes[3].set_ylabel('axial 2nd-diff RMS x1e3 (in-disk)')
    axes[3].set_title('slice-decoupling noise vs z (15 iterations)', fontsize=10)
    axes[3].legend(fontsize=9)
    fig.suptitle(f'lilly_ds4 s2.0, 15 iterations — center-slice (fan-plane) zoom, '
                 f'tight window [{vmin:.3g}, {vmax:.3g}]', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for case in CASES:
        window = dataset_window(case)
        print(f'{case}: window [5%,95%] = [{window[0]:.4g}, {window[1]:.4g}], '
              f'range [{window[2]:.4g}, {window[3]:.4g}]', flush=True)
        for s in SHARPNESS_LIST:
            for it in FIG_ITERS:
                name = f'{case}_s{s}_it{it}.png'
                build_fig(case, s, it, window, os.path.join(OUT_DIR, name))
                print(f'built {name}', flush=True)
    build_zoom(os.path.join(OUT_DIR, 'lilly_ds4_s2.0_it15_centerzoom.png'))
    print('built lilly_ds4_s2.0_it15_centerzoom.png', flush=True)
    print(f'figures in {OUT_DIR}', flush=True)


if __name__ == '__main__':
    main()
