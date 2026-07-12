"""Focused probe for the BGA center-slice NOISE question (p3b): per-slice in-plane
high-pass noise per VARIANT (not old-vs-new difference -- a direct 'how noisy is this
slice' index), plus tight-window zoomed center-slice crops for visual judgment.

Noise index per slice: std of (slice - 7x7 box blur) over the interior disk EXCLUDING
bright structure (values above 0.3 * window), i.e. the background speckle Greg described.
766-class volumes are small enough to load whole (~1.6 GB each; two live at a time).
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
from scipy.ndimage import uniform_filter

# ---------------- run parameters (edit here; no CLI args) ----------------
PAD_DIR = '/depot/bouman/data/mbirjax_metrics/padding'
TAG = 'bga_normal_v2x_d2x'
ITERS = (15, 50)
VMAX = 1.389                 # the p3b window (old mid-slice p99.5)
STRUCT_FRAC = 0.3            # exclude voxels above this fraction of VMAX from the noise index
CROP = (np.s_[380:700], np.s_[150:620])   # lower-interior region with the speckle
CROP_VMAX = 0.15 * VMAX      # tight window for the zoom panels


def _load(variant, it):
    with h5py.File(os.path.join(PAD_DIR, f'{TAG}_{variant}_iter{it}.h5'), 'r') as f:
        d = f['recon'] if 'recon' in f else f[list(f.keys())[0]]
        return d[()]


def noise_profile(vol):
    n_rows, n_cols, n_slices = vol.shape
    rr, cc = np.ogrid[:n_rows, :n_cols]
    disk = ((rr - n_rows / 2)**2 + (cc - n_cols / 2)**2) <= (0.42 * n_rows)**2
    out = np.zeros(n_slices)
    for k in range(n_slices):
        img = vol[:, :, k]
        hp = img - uniform_filter(img, size=7)
        mask = disk & (img < STRUCT_FRAC * VMAX)
        out[k] = float(hp[mask].std()) if mask.any() else 0.0
    return out


def main():
    profiles = {}
    n_ext = None
    for it in ITERS:
        new = _load('new', it)
        old = _load('old', it)
        n_ext = (new.shape[2] - old.shape[2]) // 2
        profiles[('new', it)] = noise_profile(new)
        profiles[('old', it)] = noise_profile(old)

        # Tight-window zoomed center-slice crops (old vs new at the same physical slice).
        k_old = old.shape[2] // 2
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].imshow(old[:, :, k_old][CROP], cmap='gray', vmin=0, vmax=CROP_VMAX)
        axes[0].set_title(f'old, center slice {k_old} (window 0..{CROP_VMAX:.2f})')
        axes[1].imshow(new[:, :, k_old + n_ext][CROP], cmap='gray', vmin=0, vmax=CROP_VMAX)
        axes[1].set_title('new, same physical slice')
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f'{TAG} iter {it}: center-slice interior crop, tight window', fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(PAD_DIR, f'p3b_{TAG}_center_zoom_iter{it}.png'), dpi=110)
        plt.close(fig)
        del new, old

    # Noise profiles (physical z alignment: old slice k <-> new slice k + n_ext).
    fig, axes = plt.subplots(1, len(ITERS), figsize=(8 * len(ITERS), 5), squeeze=False)
    for j, it in enumerate(ITERS):
        ax = axes[0, j]
        pn, po = profiles[('new', it)], profiles[('old', it)]
        ax.plot(np.arange(len(pn)), pn, label='new (extended)', lw=1.2)
        ax.plot(np.arange(len(po)) + n_ext, po, label='old', lw=1.2)
        for m in (n_ext, len(pn) - n_ext):
            ax.axvline(m, color='tab:red', lw=0.8, ls='--')
        ax.set_xlabel('slice (new-volume index)')
        ax.set_ylabel('in-plane high-pass noise (std)')
        ax.set_title(f'iter {it}')
        ax.legend()
    fig.suptitle(f'{TAG}: per-slice background noise index, old vs new', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(PAD_DIR, f'p3b_{TAG}_noise_profile.png'), dpi=110)
    plt.close(fig)

    # Capsule: center-40 mean noise per variant/iter.
    for it in ITERS:
        pn, po = profiles[('new', it)], profiles[('old', it)]
        c_old = len(po) // 2
        old40 = float(po[c_old - 20:c_old + 20].mean())
        new40 = float(pn[c_old - 20 + n_ext:c_old + 20 + n_ext].mean())
        print(f'iter {it}: center-40 noise old {old40:.5f} vs new {new40:.5f} '
              f'(old/new = {old40 / new40:.2f}x)')
    print('figures written to', PAD_DIR)


if __name__ == '__main__':
    main()
