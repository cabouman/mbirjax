"""Render the p3b BGA axial-A/B figures (see p3b_bga_axial_check.py).  Streaming reads
(row blocks / single planes) so it runs on a login node; PNGs land in PAD_DIR.

The question this scan answers: BGA is truncated BOTH laterally (untreated in both
variants) and axially -- does the axial extension ALONE improve the prominent
center-slice noise and the slow convergence?  So the headline panels are x-y CENTER
slices (same physical slice in both variants) plus the convergence curves; the x-z view
and z-profile carry the axial-end story as in p3a.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

# ---------------- run parameters (edit here; no CLI args) ----------------
PAD_DIR = '/depot/bouman/data/mbirjax_metrics/padding'
TAG = 'bga_normal_v2x_d2x'
ITERS = (15, 50)             # 15 = experiment_zeiss.py's default cap; 50 = long run
BLOCK = 64


def _dset(f):
    return f['recon'] if 'recon' in f else f[list(f.keys())[0]]


def main():
    paths = {(v, it): os.path.join(PAD_DIR, f'{TAG}_{v}_iter{it}.h5')
             for v in ('new', 'old') for it in ITERS}
    with h5py.File(paths[('new', ITERS[-1])], 'r') as f:
        n_rows, n_cols, n_new = _dset(f).shape
    with h5py.File(paths[('old', ITERS[-1])], 'r') as f:
        n_old = _dset(f).shape[2]
    n_ext = (n_new - n_old) // 2
    print(f'shapes: new slices {n_new}, old {n_old}, extension per end = {n_ext}')

    # Grayscale window from the old iter-50 mid-slice region (robust percentile).
    with h5py.File(paths[('old', ITERS[-1])], 'r') as f:
        mid_old = _dset(f)[:, :, n_old // 2]
    vmax = float(np.percentile(mid_old, 99.5))
    print(f'window: [0, {vmax:.4g}] (old mid-slice p99.5)')

    # ---------------- Figure 1 (headline): x-y center slices, old vs new ----------------
    # Same PHYSICAL slice: old k <-> new k + n_ext.  One row per iteration snapshot.
    k_centers = (n_old // 2 - 20, n_old // 2, n_old // 2 + 20)
    for it in ITERS:
        fig, axes = plt.subplots(2, len(k_centers), figsize=(5 * len(k_centers), 10))
        with h5py.File(paths[('old', it)], 'r') as fo, h5py.File(paths[('new', it)], 'r') as fn:
            do, dn = _dset(fo), _dset(fn)
            for j, k in enumerate(k_centers):
                axes[0, j].imshow(do[:, :, k], cmap='gray', vmin=0, vmax=vmax)
                axes[0, j].set_title(f'old, slice {k}', fontsize=10)
                axes[1, j].imshow(dn[:, :, k + n_ext], cmap='gray', vmin=0, vmax=vmax)
                axes[1, j].set_title(f'new, same physical slice', fontsize=10)
        for ax in axes.flat:
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f'{TAG} iter {it}: center x-y slices, old (top) vs new (bottom)',
                     fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(PAD_DIR, f'p3b_{TAG}_xy_center_iter{it}.png'), dpi=110)
        plt.close(fig)

    # ---------------- Figure 2: x-z at y mid + zoomed ends (iter 50) ----------------
    it = ITERS[-1]
    with h5py.File(paths[('old', it)], 'r') as fo, h5py.File(paths[('new', it)], 'r') as fn:
        plane_old = _dset(fo)[n_rows // 2]
        plane_new = _dset(fn)[n_rows // 2]
    zoom = min(4 * n_ext, n_old // 2)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), gridspec_kw={'width_ratios': [3, 1, 1]})
    panels = [
        ('old, full x-z', plane_old.T, None),
        ('old, bottom end', plane_old[:, :zoom].T, None),
        ('old, top end', plane_old[:, -zoom:].T, None),
        ('new, full x-z', plane_new.T, (n_ext, n_new - n_ext)),
        ('new, bottom end', plane_new[:, :zoom + n_ext].T, (n_ext,)),
        ('new, top end', plane_new[:, -(zoom + n_ext):].T, (zoom,)),
    ]
    for ax, (title, img, marks) in zip(axes.flat, panels):
        ax.imshow(img, cmap='gray', vmin=0, vmax=vmax, aspect='auto', origin='lower')
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        for m in (marks or ()):
            ax.axhline(m, color='tab:red', lw=0.8, ls='--')
    fig.suptitle(f'{TAG} iter {it}: old ({n_old} slices) vs new ({n_new}; red dashes = '
                 f'old-slab boundary)', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(PAD_DIR, f'p3b_{TAG}_xz_iter{it}.png'), dpi=110)
    plt.close(fig)

    # ---------------- Streamed stats: z-profiles + per-slice diff + capsule ----------------
    rr, cc = np.ogrid[:n_rows, :n_cols]
    disk = ((rr - n_rows / 2)**2 + (cc - n_cols / 2)**2) <= (0.35 * n_rows)**2
    ss_diff = np.zeros(n_old, dtype=np.float64)
    ss_old = np.zeros(n_old, dtype=np.float64)
    abs_old = np.zeros(n_old, dtype=np.float64)          # disk-masked mean |value| profiles
    abs_new = np.zeros(n_new, dtype=np.float64)
    disk_count = float(disk.sum())
    with h5py.File(paths[('old', it)], 'r') as fo, h5py.File(paths[('new', it)], 'r') as fn:
        do, dn = _dset(fo), _dset(fn)
        for i0 in range(0, n_rows, BLOCK):
            i1 = min(i0 + BLOCK, n_rows)
            a = dn[i0:i1]
            b = do[i0:i1]
            d = a[:, :, n_ext:n_ext + n_old] - b
            ss_diff += (d.astype(np.float64)**2).sum(axis=(0, 1))
            ss_old += (b.astype(np.float64)**2).sum(axis=(0, 1))
            m = disk[i0:i1]
            abs_old += np.abs(b[m]).sum(axis=0)
            abs_new += np.abs(a[m]).sum(axis=0)
    n_pix = n_rows * n_cols
    q = n_old // 4
    body_rms = float(np.sqrt(ss_old[q:-q].sum() / (n_pix * (n_old - 2 * q))))
    print(f'body RMS (old, middle half) = {body_rms:.4g}')
    print(f'interior RMS(new-old)/bodyRMS = '
          f'{float(np.sqrt(ss_diff[q:-q].sum() / (n_pix * (n_old - 2 * q)))) / body_rms:.4f}')
    for label, sl in (('bottom', slice(0, 8)), ('top', slice(-8, None))):
        print(f'{label}-end 8-slice RMS(new-old)/bodyRMS = '
              f'{float(np.sqrt(ss_diff[sl].mean() / n_pix)) / body_rms:.4f}')
    # Center-slice noise proxy: RMS of (new-old) over the middle 40 slices vs body.
    c0 = n_old // 2 - 20
    print(f'center-40-slice RMS(new-old)/bodyRMS = '
          f'{float(np.sqrt(ss_diff[c0:c0 + 40].mean() / n_pix)) / body_rms:.4f}')

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(np.arange(n_new), abs_new / disk_count, label='new (extended)', lw=1.2)
    axes[0].plot(np.arange(n_old) + n_ext, abs_old / disk_count, label='old', lw=1.2)
    for m in (n_ext, n_new - n_ext):
        axes[0].axvline(m, color='tab:red', lw=0.8, ls='--')
    axes[0].set_xlabel('slice (new-volume index)')
    axes[0].set_ylabel('interior-disk mean |value|')
    axes[0].set_title(f'axial profile, iter {it}')
    axes[0].legend()
    axes[1].semilogy(np.sqrt(ss_diff / n_pix) / body_rms)
    axes[1].set_xlabel('shared-slab slice')
    axes[1].set_ylabel('RMS(new-old) / body RMS')
    axes[1].set_title('per-slice relative difference')
    fig.tight_layout()
    fig.savefig(os.path.join(PAD_DIR, f'p3b_{TAG}_zprofile_diff.png'), dpi=110)
    plt.close(fig)

    # ---------------- Convergence ----------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in ('old', 'new'):
        with open(os.path.join(PAD_DIR, f'{TAG}_{variant}_log.json')) as f:
            rows = json.load(f)['rows']
        ax.semilogy([r['iteration'] for r in rows], [r['change_pct'] for r in rows],
                    label=variant, lw=1.4)
    ax.axhline(0.2, color='gray', lw=0.8, ls=':', label='0.2% stop')
    ax.set_xlabel('iteration'); ax.set_ylabel('change %')
    ax.set_title(f'{TAG}: convergence metric, old vs new slab (axial-only A/B)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PAD_DIR, f'p3b_{TAG}_convergence.png'), dpi=110)
    plt.close(fig)
    print('figures written to', PAD_DIR)


if __name__ == '__main__':
    main()
