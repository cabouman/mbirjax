"""Build the default-vs-skip-0 head-to-head slice figures from the volumes that
r2_recon_capture.py staged (run on gautschi AFTER the capture job; CPU-only).

For each (dataset, sharpness, iteration): two view groups (axial mid-slice, coronal
mid-cut).  Each group is a recon row [reference | default | skip 0] over a
difference row [ (blank) | default − reference | skip 0 − reference ] — so each
schedule's difference from the 150-iteration reference sits directly under that
schedule's reconstruction.  Grayscale window is FIXED PER DATASET (1st-99th percentile
of the sharpness-1.0 reference); the signed difference images share ONE symmetric color
scale PER DATASET (a colorbar is drawn on each figure).  Cases with no reference
(lilly_ds4 s2.0) show recons only.  Also builds the center-slice (fan-plane) zoom for
lilly_ds4 s2.0 (layout unchanged, colorbar added).

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
DIFF_LABELS = {'D0_default': 'default − reference',
               'D1_g2start': 'skip 0 − reference'}
WINDOW_PCT = (1, 99)                # fixed per-dataset window percentiles
DIFF_PCT = 99                       # per-dataset diff scale = this pct of whole-vol |diff|
AXIAL, CORONAL = 'axial (z mid)', 'coronal (y mid)'


def dataset_window(case):
    """Per-dataset grayscale window from the sharpness-1.0 reference volume."""
    ref = np.load(os.path.join(REF_DIRS[case], 'ref_sharp1.0.npy'))
    vmin, vmax = np.percentile(ref, WINDOW_PCT)
    return float(vmin), float(vmax), float(ref.min()), float(ref.max())


def mid_slices(vol):
    """(axial mid-slice, coronal mid-cut) as 2-D arrays with z vertical for coronal.

    Returns COPIES, not views — load_mids keeps these across all cases, and a view
    would pin the whole (up to ~670 MB) source volume in memory, OOM-killing the run
    on the case-to-case transition.
    """
    return {AXIAL: vol[:, :, vol.shape[2] // 2].copy(),
            CORONAL: vol[:, vol.shape[1] // 2, :].T.copy()}   # (slices, rows): z vert


def load_case_data(case):
    """Load every needed volume ONCE, keeping only its 2-D mid-slices (NFS-friendly),
    and derive one per-dataset diff color limit.

    Returns recon[(s, it, arm)] -> {view: 2-D}, ref[s] -> {view: 2-D} or None, and
    difflim = the symmetric limit for the (recon − reference) images: the 99th
    percentile of |recon − reference| over the WHOLE volume (max over ref-having
    combos).  Whole-volume rather than the axial mid-slice, because the central axial
    slice can be anomalously well converged (e.g. ds4 z=333 is ~35x below the volume
    error) and would set an unrepresentatively tiny scale.  The 99th percentile clips
    the extreme axial-end flash so the interior stays visible.
    """
    recon, ref, diffvals = {}, {}, []
    for s in SHARPNESS_LIST:
        p = os.path.join(REF_DIRS[case], f'ref_sharp{s}.npy')
        refvol = np.load(p) if os.path.exists(p) else None
        ref[s] = mid_slices(refvol) if refvol is not None else None
        for it in FIG_ITERS:
            for arm in ARMS:
                vol = np.load(os.path.join(STAGE, f'{case}_{arm}_s{s}_it{it}.npy'))
                recon[(s, it, arm)] = mid_slices(vol)
                if refvol is not None:
                    diffvals.append(np.percentile(np.abs(vol - refvol), DIFF_PCT))
    return recon, ref, (max(diffvals) if diffvals else 1e-6)


def _coronal_ratio(mids):
    a, c = mids[AXIAL].shape, mids[CORONAL].shape
    return c[0] / a[0]                       # tall coronal -> >1


def build_fig(case, s, it, window, difflim, recon, ref, out_png):
    vmin, vmax, dmin, dmax = window
    d0 = recon[(s, it, 'D0_default')]
    d1 = recon[(s, it, 'D1_g2start')]
    rmids = ref[s]
    cr = _coronal_ratio(d0)

    if rmids is None:                        # no reference -> recons only
        fig, axes = plt.subplots(2, 2, figsize=(6.6, 3.4 * (1 + cr)),
                                 gridspec_kw={'height_ratios': [1, cr]})
        for i, view in enumerate((AXIAL, CORONAL)):
            for j, (arm, mids) in enumerate((('D0_default', d0), ('D1_g2start', d1))):
                ax = axes[i, j]; ax.set_xticks([]); ax.set_yticks([])
                ax.imshow(mids[view], cmap='gray', vmin=vmin, vmax=vmax)
                if i == 0:
                    ax.set_title(ARM_LABELS[arm], fontsize=9)
                if j == 0:
                    ax.set_ylabel(view, fontsize=9)
        fig.suptitle(f'{case}  sharpness {s}  —  {it} iterations  (no reference: '
                     f'reconstructions only)\nwindow [1%, 99%] = [{vmin:.3g}, '
                     f'{vmax:.3g}],  data range [{dmin:.3g}, {dmax:.3g}]', fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out_png, dpi=110); plt.close(fig)
        return

    # With a reference: recon row over diff row, for each of the two views.
    fig, axes = plt.subplots(4, 3, figsize=(9.6, 4.7 * (1 + cr)),
                             gridspec_kw={'height_ratios': [1, 1, cr, cr]})
    recon_cols = [('reference (150 it)', rmids), (ARM_LABELS['D0_default'], d0),
                  (ARM_LABELS['D1_g2start'], d1)]
    diff_im = None
    for vi, view in enumerate((AXIAL, CORONAL)):
        rr, rd = vi * 2, vi * 2 + 1          # recon row, diff row
        for c, (label, mids) in enumerate(recon_cols):
            ax = axes[rr, c]; ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(mids[view], cmap='gray', vmin=vmin, vmax=vmax)
            if rr == 0:
                ax.set_title(label, fontsize=9)
            if c == 0:
                ax.set_ylabel(view, fontsize=9)
        axes[rd, 0].axis('off')              # nothing under the reference
        for c, (arm, mids) in enumerate((('D0_default', d0), ('D1_g2start', d1)),
                                        start=1):
            ax = axes[rd, c]; ax.set_xticks([]); ax.set_yticks([])
            diff_im = ax.imshow(mids[view] - rmids[view], cmap='coolwarm',
                                vmin=-difflim, vmax=difflim)
            ax.set_title(DIFF_LABELS[arm], fontsize=9)
    cb = fig.colorbar(diff_im, ax=axes.ravel().tolist(), location='right',
                      shrink=0.45, aspect=40, pad=0.02)
    cb.set_label(f'recon − reference   (±{difflim:.3g} = ±{difflim / vmax * 100:.0f}% '
                 f'of the 99% value; axial-end flash zones may clip)', fontsize=9)
    fig.suptitle(f'{case}  sharpness {s}  —  {it} iterations   |   '
                 f'window [1%, 99%] = [{vmin:.3g}, {vmax:.3g}],  '
                 f'data range [{dmin:.3g}, {dmax:.3g}]', fontsize=11)
    fig.savefig(out_png, dpi=110, bbox_inches='tight'); plt.close(fig)


def build_zoom(out_png):
    """Center-slice (fan-plane) zoom for lilly_ds4 s2.0 at 15 iterations (layout
    unchanged from before; a colorbar is added to the difference panel)."""
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
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.2))
    for ax, (label, sl) in zip(axes[:2], [('default [0,2,4,6,7]', sl0),
                                          ('skip 0 [2,4,6,7]', sl1)]):
        ax.imshow(sl, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'{label}  z={zc}', fontsize=10)
    diff = sl1 - sl0
    lim = np.percentile(np.abs(diff), 99.9)
    ds_vmax = dataset_window('lilly_ds4')[1]
    im = axes[2].imshow(diff, cmap='coolwarm', vmin=-lim, vmax=lim)
    axes[2].set_xticks([]); axes[2].set_yticks([])
    axes[2].set_title(f'skip 0 − default (±{lim / ds_vmax * 100:.1f}% '
                      f'of the 99% value)', fontsize=9)
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
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
    fig.savefig(out_png, dpi=110); plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for case in CASES:
        window = dataset_window(case)
        recon, ref, difflim = load_case_data(case)
        print(f'{case}: window [1%,99%] = [{window[0]:.4g}, {window[1]:.4g}], '
              f'range [{window[2]:.4g}, {window[3]:.4g}], diff scale ±{difflim:.4g}',
              flush=True)
        for s in SHARPNESS_LIST:
            for it in FIG_ITERS:
                name = f'{case}_s{s}_it{it}.png'
                build_fig(case, s, it, window, difflim, recon, ref,
                          os.path.join(OUT_DIR, name))
                print(f'built {name}', flush=True)
    build_zoom(os.path.join(OUT_DIR, 'lilly_ds4_s2.0_it15_centerzoom.png'))
    print('built lilly_ds4_s2.0_it15_centerzoom.png', flush=True)
    print(f'figures in {OUT_DIR}', flush=True)


if __name__ == '__main__':
    main()
