"""Analysis for the p3e lateral+axial padding comparison (BGA three-way + Lilly pair).

Streamed reads throughout (the 1149-grid BGA volume is ~4 GB).  Produces:
  p3e_bga_convergence.png   -- change%% per iteration: no-pad, axial, axial+lat1.5
  p3e_bga_radial.png        -- center-slab radial profiles (physical radius): the
                               RoR-edge ring at the old FoV boundary vs the padded run,
                               INCLUDING the padded run's own boundary (the knee check:
                               a ring at the new boundary = still under cover)
  p3e_bga_noise_profile.png -- per-slice background noise index over the SAME physical
                               interior disk, axial vs axial+lat1.5 (aligned slices)
  p3e_bga_xy_center.png     -- center x-y slices, all three variants (shared window)
  p3e_lilly_edge.png        -- Lilly: x-profiles through the right-edge region + panels,
                               p3c ref vs lat1.25 (mild one-sided case)
Prints a capsule of the numbers for the record.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

PAD = '/depot/bouman/data/mbirjax_metrics/padding'
BLOCK = 64


def _dset(f):
    return f['recon'] if 'recon' in f else f[list(f.keys())[0]]


def radial_profile_h5(path, slab, n_bins=200, r_max=None):
    """Mean value vs radius (voxel units) over the given central slice slab, streamed."""
    with h5py.File(path, 'r') as f:
        d = _dset(f)
        n_rows, n_cols, _ = d.shape
        rr = np.arange(n_rows) - (n_rows - 1) / 2.0
        cc = np.arange(n_cols) - (n_cols - 1) / 2.0
        r_max = r_max or float(np.hypot(rr[-1], cc[-1]))
        sums = np.zeros(n_bins)
        counts = np.zeros(n_bins)
        for i0 in range(0, n_rows, BLOCK):
            i1 = min(i0 + BLOCK, n_rows)
            block = d[i0:i1, :, slab].mean(axis=2)          # (block, cols) slab-mean
            r = np.hypot(rr[i0:i1][:, None], cc[None, :])
            idx = np.minimum((r / r_max * n_bins).astype(int), n_bins - 1)
            sums += np.bincount(idx.ravel(), weights=block.ravel(), minlength=n_bins)
            counts += np.bincount(idx.ravel(), minlength=n_bins)
    centers = (np.arange(n_bins) + 0.5) * r_max / n_bins
    return centers, sums / np.maximum(counts, 1)


def noise_profile_h5(path, disk_radius_vox, struct_thresh):
    """Per-slice in-plane high-pass noise (std) over a disk of the given PHYSICAL radius
    (voxel units of this grid), excluding bright structure; streamed by slice blocks in z
    via row blocks + per-slice accumulation of (sum, sumsq, count) of the high-pass."""
    from scipy.ndimage import uniform_filter
    with h5py.File(path, 'r') as f:
        d = _dset(f)
        n_rows, n_cols, n_slices = d.shape
        rr = np.arange(n_rows) - (n_rows - 1) / 2.0
        cc = np.arange(n_cols) - (n_cols - 1) / 2.0
        disk = (rr[:, None]**2 + cc[None, :]**2) <= disk_radius_vox**2
        out = np.zeros(n_slices)
        # Per-slice filtering needs whole planes; read z in chunks of 32 slices.
        for k0 in range(0, n_slices, 32):
            k1 = min(k0 + 32, n_slices)
            chunk = d[:, :, k0:k1]
            for j in range(k1 - k0):
                img = chunk[:, :, j]
                hp = img - uniform_filter(img, size=7)
                mask = disk & (img < struct_thresh)
                out[k0 + j] = float(hp[mask].std()) if mask.any() else 0.0
    return out


def main():
    # ---------------- BGA: convergence overlay ----------------
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for label, path in [('no pad (old default)', f'{PAD}/bga_normal_v2x_d2x_old_log.json'),
                        ('axial (new default)', f'{PAD}/bga_normal_v2x_d2x_new_log.json'),
                        ('axial + lateral 1.5x', f'{PAD}/p3e_bga_lat150_log.json')]:
        with open(path) as f:
            rows = json.load(f)['rows']
        ax.semilogy([r['iteration'] for r in rows], [r['change_pct'] for r in rows],
                    label=label, lw=1.4)
    ax.axhline(0.2, color='gray', lw=0.8, ls=':', label='0.2% stop')
    ax.set_xlabel('iteration'); ax.set_ylabel('change %')
    ax.set_title('BGA: convergence, no-pad vs axial vs axial+lateral')
    ax.legend()
    fig.tight_layout(); fig.savefig(f'{PAD}/p3e_bga_convergence.png', dpi=110); plt.close(fig)

    # ---------------- BGA: radial profiles (physical radius, voxel units shared) ----------
    # Center slab: 40 slices about the volume middle; grids share delta_voxel so voxel
    # radius IS physical.  Old-FoV boundary at r = 383; lat1.5 boundary at r = 574.5.
    prof = {}
    with h5py.File(f'{PAD}/bga_normal_v2x_d2x_new_iter50.h5', 'r') as f:
        n_ax = _dset(f).shape[2]
    slab_ax = np.s_[n_ax // 2 - 20: n_ax // 2 + 20]
    r_ax, prof['axial (766 grid)'] = radial_profile_h5(
        f'{PAD}/bga_normal_v2x_d2x_new_iter50.h5', slab_ax)
    with h5py.File(f'{PAD}/p3e_bga_lat150_iter50.h5', 'r') as f:
        n_lat = _dset(f).shape[2]
    slab_lat = np.s_[n_lat // 2 - 20: n_lat // 2 + 20]
    r_lat, prof['axial + lateral 1.5x (1149 grid)'] = radial_profile_h5(
        f'{PAD}/p3e_bga_lat150_iter50.h5', slab_lat)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(r_ax, prof['axial (766 grid)'], label='axial (766 grid)', lw=1.3)
    ax.plot(r_lat, prof['axial + lateral 1.5x (1149 grid)'],
            label='axial + lateral 1.5x (1149 grid)', lw=1.3)
    ax.axvline(383, color='tab:blue', lw=0.9, ls='--', label='old FoV boundary (r=383)')
    ax.axvline(574.5, color='tab:orange', lw=0.9, ls='--', label='1.5x boundary (r=574.5)')
    ax.set_xlabel('radius (voxels, shared physical scale)')
    ax.set_ylabel('center-slab mean value')
    ax.set_title('BGA iter 50: radial profile -- ring at each grid boundary?')
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(f'{PAD}/p3e_bga_radial.png', dpi=110); plt.close(fig)

    # Knee capsule: ring amplitude just inside each boundary vs the local background.
    def ring_stat(r, p, boundary):
        near = p[(r > boundary - 25) & (r < boundary)]
        inner = p[(r > boundary - 120) & (r < boundary - 60)]
        return float(near.max()), float(np.median(inner))
    ring_old, bg_old = ring_stat(r_ax, prof['axial (766 grid)'], 383)
    ring_new, bg_new = ring_stat(r_lat, prof['axial + lateral 1.5x (1149 grid)'], 574.5)
    at_old_in_lat = prof['axial + lateral 1.5x (1149 grid)'][(r_lat > 358) & (r_lat < 408)]
    print(f'BGA ring at OLD boundary (axial run):     peak {ring_old:.4f} vs inner median {bg_old:.4f}')
    print(f'BGA ring at NEW boundary (lat1.5 run):    peak {ring_new:.4f} vs inner median {bg_new:.4f}')
    print(f'BGA lat1.5 profile across the old boundary region: '
          f'max {float(at_old_in_lat.max()):.4f} (ring gone there if ~interior level)')

    # ---------------- BGA: noise profiles over the SAME physical disk ----------------
    vmax = 1.389
    noise_ax = noise_profile_h5(f'{PAD}/bga_normal_v2x_d2x_new_iter50.h5',
                                disk_radius_vox=0.42 * 766, struct_thresh=0.3 * vmax)
    noise_lat = noise_profile_h5(f'{PAD}/p3e_bga_lat150_iter50.h5',
                                 disk_radius_vox=0.42 * 766, struct_thresh=0.3 * vmax)
    off = (n_lat - n_ax) // 2
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(n_lat), noise_lat, label='axial + lateral 1.5x', lw=1.2)
    ax.plot(np.arange(n_ax) + off, noise_ax, label='axial (new default)', lw=1.2)
    ax.set_xlabel('slice (lat-grid index)'); ax.set_ylabel('in-plane high-pass noise (std)')
    ax.set_title('BGA iter 50: background noise per slice, same physical interior disk')
    ax.legend()
    fig.tight_layout(); fig.savefig(f'{PAD}/p3e_bga_noise_profile.png', dpi=110); plt.close(fig)
    c = n_ax // 2
    print(f'BGA center-40 noise: axial {noise_ax[c-20:c+20].mean():.5f} vs '
          f'axial+lat1.5 {noise_lat[c-20+off:c+20+off].mean():.5f}')
    interior = np.r_[noise_ax[60:c-30], noise_ax[c+30:-60]]
    interior_lat = np.r_[noise_lat[60+off:c-30+off], noise_lat[c+30+off:n_ax-60+off]]
    print(f'BGA off-center noise medians: axial {np.median(interior):.5f} vs '
          f'axial+lat1.5 {np.median(interior_lat):.5f}')

    # ---------------- BGA: center x-y panels ----------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    panels = [('no pad', f'{PAD}/bga_normal_v2x_d2x_old_iter50.h5'),
              ('axial (new default)', f'{PAD}/bga_normal_v2x_d2x_new_iter50.h5'),
              ('axial + lateral 1.5x', f'{PAD}/p3e_bga_lat150_iter50.h5')]
    for ax_, (label, path) in zip(axes, panels):
        with h5py.File(path, 'r') as f:
            d = _dset(f)
            img = d[:, :, d.shape[2] // 2]
        ax_.imshow(img, cmap='gray', vmin=0, vmax=vmax)
        ax_.set_title(f'{label}  ({img.shape[0]}²)', fontsize=11)
        ax_.set_xticks([]); ax_.set_yticks([])
    fig.suptitle('BGA iter 50: center slice (each on its own grid; same window)', fontsize=13)
    fig.tight_layout(); fig.savefig(f'{PAD}/p3e_bga_xy_center.png', dpi=110); plt.close(fig)

    # ---------------- Lilly: right-edge profile + panels ----------------
    ref = np.load(f'{PAD}/p3c_lilly_ds4_ref.npy')            # (374, 374, 667)
    lat = np.load(f'{PAD}/p3e_lilly_ds4_lat125_iter15.npy')  # (467, 467, 667)
    k = ref.shape[2] // 2
    slab = np.s_[k - 15: k + 15]
    # x-profile: mean over the central y band and the slab, vs column index mapped to
    # physical x (grid centers differ by (467-373)/2 - .5 = half-voxel parity; profile use).
    yb_ref = np.s_[ref.shape[0]//2 - 40: ref.shape[0]//2 + 40]
    yb_lat = np.s_[lat.shape[0]//2 - 40: lat.shape[0]//2 + 40]
    x_ref = np.arange(ref.shape[1]) - (ref.shape[1] - 1) / 2.0
    x_lat = np.arange(lat.shape[1]) - (lat.shape[1] - 1) / 2.0
    p_ref = ref[yb_ref, :, slab].mean(axis=(0, 2))
    p_lat = lat[yb_lat, :, slab].mean(axis=(0, 2))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [3, 2]})
    axes[0].plot(x_lat, p_lat, label='axial + lateral 1.25x (467 grid)', lw=1.2)
    axes[0].plot(x_ref, p_ref, label='axial only (374 grid)', lw=1.2)
    axes[0].axvline(x_ref[-1], color='tab:orange', lw=0.9, ls='--', label='old right FoV edge')
    axes[0].set_xlabel('x (voxels from center)'); axes[0].set_ylabel('band mean value')
    axes[0].set_title('Lilly iter 15: x-profile through the truncated (right) side')
    axes[0].legend(fontsize=9)
    with np.errstate(all='ignore'):
        img = lat[:, :, k]
    axes[1].imshow(img, cmap='gray', vmin=0, vmax=float(np.percentile(ref[:, :, k], 99.8)))
    axes[1].set_title('lat 1.25x, center slice (full grid)')
    axes[1].set_xticks([]); axes[1].set_yticks([])
    fig.tight_layout(); fig.savefig(f'{PAD}/p3e_lilly_edge.png', dpi=110); plt.close(fig)
    edge_ring_ref = float(p_ref[-8:].max())
    same_region_lat = float(p_lat[(x_lat >= x_ref[-8]) & (x_lat <= x_ref[-1] + 1)].max())
    print(f'Lilly right-edge band max: axial-only {edge_ring_ref:.4f} vs '
          f'lat1.25 at the same physical x {same_region_lat:.4f} '
          f'(interior median {float(np.median(p_ref[100:250])):.4f})')
    print('figures written to', PAD)


if __name__ == '__main__':
    main()
