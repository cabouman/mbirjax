"""Streaming completion of the p3a round-2 analysis (the 1024-class volumes do not fit
comfortably in login-node memory, so everything here reads h5 in row blocks; peak ~1 GB).

Produces the two figures the full-volume script could not (shared-slab difference and
convergence), prints the quantitative capsule, and runs the PROVENANCE CROSS-CHECK:
this experiment's old-variant iter50 vs the pre-existing partition-sequence reference
recon (recons/<TAG>_default_iter50.h5, prerelease-era, same sequence/seed machinery) --
expected to agree at the ~1e-3-class full-pipeline level (lessons.md sec. 2).
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
RECONS_DIR = '/depot/bouman/data/mbirjax_metrics/recons'
TAG = 'sic_v3x_d2x_nv534_nch1024'
ITER = 50
BLOCK = 64                       # rows per streamed block


def _dset(f):
    return f['recon'] if 'recon' in f else f[list(f.keys())[0]]


def main():
    new_path = os.path.join(PAD_DIR, f'{TAG}_new_iter{ITER}.h5')
    old_path = os.path.join(PAD_DIR, f'{TAG}_old_iter{ITER}.h5')
    ref_path = os.path.join(RECONS_DIR, f'{TAG}_default_iter{ITER}.h5')

    with h5py.File(new_path, 'r') as fn, h5py.File(old_path, 'r') as fo:
        dn, do = _dset(fn), _dset(fo)
        n_rows, n_cols, n_new = dn.shape
        n_old = do.shape[2]
        n_ext = (n_new - n_old) // 2
        print(f'shapes: new {dn.shape}, old {do.shape}, extension per end = {n_ext}')

        # Streamed accumulators: per-slice sum of squared (new-old) on the shared slab,
        # sum of squares of old (for the body scale), extension-region |value| sums.
        ss_diff = np.zeros(n_old, dtype=np.float64)
        ss_old = np.zeros(n_old, dtype=np.float64)
        ext_abs_bot = ext_abs_top = 0.0
        for i0 in range(0, n_rows, BLOCK):
            i1 = min(i0 + BLOCK, n_rows)
            a = dn[i0:i1]                       # (block, cols, n_new), contiguous read
            b = do[i0:i1]
            d = a[:, :, n_ext:n_ext + n_old] - b
            ss_diff += (d.astype(np.float64)**2).sum(axis=(0, 1))
            ss_old += (b.astype(np.float64)**2).sum(axis=(0, 1))
            ext_abs_bot += float(np.abs(a[:, :, :n_ext]).sum())
            ext_abs_top += float(np.abs(a[:, :, -n_ext:]).sum())
        plane_new = dn[n_rows // 2]             # y-mid planes for the difference image
        plane_old = do[n_rows // 2]

    n_pix = n_rows * n_cols
    rms_slice = np.sqrt(ss_diff / n_pix)
    q = n_old // 4
    body_rms = float(np.sqrt(ss_old[q:-q].sum() / (n_pix * (n_old - 2 * q))))
    interior_rel = float(np.sqrt(ss_diff[q:-q].sum() / (n_pix * (n_old - 2 * q)))) / body_rms
    bot_rel = float(np.sqrt(ss_diff[:8].mean() / n_pix)) / body_rms
    top_rel = float(np.sqrt(ss_diff[-8:].mean() / n_pix)) / body_rms
    print(f'body RMS (old, middle half) = {body_rms:.4g}')
    print(f'interior (middle-half slab) RMS(new-old)/bodyRMS = {interior_rel:.4f}')
    print(f'bottom-end 8-slice RMS(new-old)/bodyRMS = {bot_rel:.4f}')
    print(f'top-end 8-slice RMS(new-old)/bodyRMS = {top_rel:.4f}')
    print(f'extension-region mean |value|: bottom {ext_abs_bot / (n_pix * n_ext):.4g}, '
          f'top {ext_abs_top / (n_pix * n_ext):.4g}')

    # ---------------- Figure: shared-slab difference ----------------
    diff_plane = plane_new[:, n_ext:n_ext + n_old] - plane_old
    vmax_body = 0.21                            # matches the volume figures' window
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 2]})
    axes[0].semilogy(rms_slice / body_rms)
    axes[0].set_xlabel('shared-slab slice'); axes[0].set_ylabel('RMS(new-old) / body RMS')
    axes[0].set_title('per-slice relative difference')
    im = axes[1].imshow(diff_plane.T, cmap='coolwarm', vmin=-0.1 * vmax_body,
                        vmax=0.1 * vmax_body, aspect='auto', origin='lower')
    axes[1].set_title('difference, x-z at y mid (window +-10% of body window)')
    axes[1].set_xticks([]); axes[1].set_yticks([])
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(PAD_DIR, f'p3a_{TAG}_shared_diff.png'), dpi=110)
    plt.close(fig)

    # ---------------- Figure: convergence ----------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in ('old', 'new'):
        with open(os.path.join(PAD_DIR, f'{TAG}_{variant}_log.json')) as f:
            rows = json.load(f)['rows']
        ax.semilogy([r['iteration'] for r in rows], [r['change_pct'] for r in rows],
                    label=variant, lw=1.4)
    ax.axhline(0.2, color='gray', lw=0.8, ls=':', label='0.2% stop')
    ax.set_xlabel('iteration'); ax.set_ylabel('change %')
    ax.set_title(f'{TAG}: convergence metric, old vs new slab')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PAD_DIR, f'p3a_{TAG}_convergence.png'), dpi=110)
    plt.close(fig)

    # ---------------- Provenance cross-check vs the reference recon ----------------
    if os.path.exists(ref_path):
        max_abs = 0.0
        max_ref = 0.0
        with h5py.File(old_path, 'r') as fo, h5py.File(ref_path, 'r') as fr:
            do, dr = _dset(fo), _dset(fr)
            if do.shape != dr.shape:
                print(f'cross-check SKIPPED: shape mismatch old {do.shape} vs ref {dr.shape}')
            else:
                for i0 in range(0, do.shape[0], BLOCK):
                    i1 = min(i0 + BLOCK, do.shape[0])
                    a, b = do[i0:i1], dr[i0:i1]
                    max_abs = max(max_abs, float(np.abs(a - b).max()))
                    max_ref = max(max_ref, float(np.abs(b).max()))
                print(f'cross-check vs {os.path.basename(ref_path)}: '
                      f'rel max diff = {max_abs / max_ref:.3e} '
                      f'(max|old-ref| {max_abs:.4g} / max|ref| {max_ref:.4g})')
    else:
        print(f'cross-check reference not found: {ref_path}')
    print('done; figures written to', PAD_DIR)


if __name__ == '__main__':
    main()
