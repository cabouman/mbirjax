"""Seam analysis for the Lilly split reproduction: compare recon_split vs recon_full.

Runs where the volumes live (cluster scratch); saves compact PNGs + an npz with the seam
slab so only small files need to move.  All knobs below (no CLI args).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = '/scratch/gautschi/buzzard/flash_lilly'
SEAM_HALF_WINDOW = 16            # slices around the split to keep/show
INTERIOR_RADIUS_FRAC = 0.85

if __name__ == '__main__':
    split_vol = np.load(f'{OUT_DIR}/recon_split.npy')
    full_vol = np.load(f'{OUT_DIR}/recon_full.npy')
    rows, cols, slices = split_vol.shape
    print(f'shapes: split {split_vol.shape}, full {full_vol.shape}', flush=True)
    split_index = int(np.round((slices - 1) / 2.0))   # recon_slice_offset is 0 for this scan

    # Interior disk mask (avoid the FoV edge; this scan may also have its own ring)
    i = np.arange(rows, dtype=np.float32)[:, None] - (rows - 1) / 2.0
    j = np.arange(cols, dtype=np.float32)[None, :] - (cols - 1) / 2.0
    interior2d = np.sqrt(i ** 2 + j ** 2) < INTERIOR_RADIUS_FRAC * (min(rows, cols) / 2.0)

    lo, hi = split_index - SEAM_HALF_WINDOW, split_index + SEAM_HALF_WINDOW + 1
    center_col = cols // 2

    # --- x-z sections (center column), full z then seam zoom, plus the difference ---
    xz_split = split_vol[:, center_col, :]
    xz_full = full_vol[:, center_col, :]
    body = full_vol[:, :, lo:hi][interior2d]
    vmin, vmax = np.percentile(body, [1, 99.5])
    dspan = 0.15 * (vmax - vmin)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, img, title in [(axes[0], xz_split[:, lo:hi].T, 'split (568f6b7, no taper)'),
                           (axes[1], xz_full[:, lo:hi].T, 'unsplit reference')]:
        ax.imshow(img, vmin=vmin, vmax=vmax, cmap='gray', origin='lower', aspect='auto')
        ax.set_title(title)
        ax.axhline(split_index - lo, color='red', lw=0.6, ls='--')
        ax.set_ylabel('slice (z)'); ax.set_xlabel('x')
    im = axes[2].imshow((xz_split - xz_full)[:, lo:hi].T, vmin=-dspan, vmax=dspan,
                        cmap='coolwarm', origin='lower', aspect='auto')
    axes[2].set_title('split - unsplit')
    axes[2].axhline(split_index - lo, color='k', lw=0.6, ls='--')
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.suptitle(f'Lilly D01788: x-z near the seam (split at slice {split_index})')
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/lilly_seam_xz.png', dpi=140)
    plt.close(fig)

    # --- z-profiles: interior-disk mean and std per slice ---
    prof_split = split_vol[interior2d].mean(axis=0)
    prof_full = full_vol[interior2d].mean(axis=0)
    diff_rms = np.sqrt(((split_vol - full_vol)[interior2d] ** 2).mean(axis=0))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    z = np.arange(slices)
    axes[0].plot(z[lo:hi], prof_full[lo:hi], 'k--', label='unsplit')
    axes[0].plot(z[lo:hi], prof_split[lo:hi], label='split')
    axes[0].axvline(split_index, color='red', lw=0.6, ls='--')
    axes[0].set_title('interior-disk mean per slice'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].semilogy(z, diff_rms)
    axes[1].axvline(split_index, color='red', lw=0.6, ls='--')
    axes[1].set_title('RMS(split - unsplit) per slice, full z'); axes[1].grid(alpha=0.3)
    for ax in axes:
        ax.set_xlabel('slice (z)')
    fig.suptitle('Lilly D01788: seam profiles')
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/lilly_seam_profiles.png', dpi=140)
    plt.close(fig)

    # --- transaxial slices at the split and one kept slice each side ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for col_ax, k in zip(range(3), [split_index - 3, split_index, split_index + 3]):
        axes[0, col_ax].imshow(split_vol[:, :, k], vmin=vmin, vmax=vmax, cmap='gray')
        axes[0, col_ax].set_title(f'split, slice {k}')
        im = axes[1, col_ax].imshow(split_vol[:, :, k] - full_vol[:, :, k],
                                    vmin=-dspan, vmax=dspan, cmap='coolwarm')
        axes[1, col_ax].set_title(f'split - unsplit, slice {k}')
        fig.colorbar(im, ax=axes[1, col_ax], fraction=0.046)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Lilly D01788: transaxial slices near the seam')
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/lilly_seam_transaxial.png', dpi=140)
    plt.close(fig)

    np.savez_compressed(f'{OUT_DIR}/lilly_seam_slab.npz',
                        split_slab=split_vol[:, :, lo:hi], full_slab=full_vol[:, :, lo:hi],
                        prof_split=prof_split, prof_full=prof_full, diff_rms=diff_rms,
                        split_index=split_index, lo=lo, hi=hi)
    print(f'seam RMS at split slice: {diff_rms[split_index]:.3e}; '
          f'away-from-seam median: {np.median(diff_rms):.3e}', flush=True)
    print('analysis done', flush=True)
