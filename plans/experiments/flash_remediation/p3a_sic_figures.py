"""Render the step-A SiC validation figures from the p3a volumes (see
p3a_sic_axial_check.py).  Reads the h5 volumes in PAD_DIR, writes PNGs there.
Light enough for a login node (loads two ~0.7 GB volumes; matplotlib Agg).

Figures:
  p3a_<tag>_xz_iter50.png     -- x-z cross-sections (y mid): old vs new, full + both ends
  p3a_<tag>_zprofile.png      -- interior-disk mean |value| vs physical z, old vs new
  p3a_<tag>_shared_diff.png   -- per-slice RMS(new-old) on the shared slab + difference image
  p3a_<tag>_convergence.png   -- change%% per iteration, old vs new
Prints a quantitative capsule (interior agreement, end-slice behavior) for the record.
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
TAG = 'sic_v3x_d2x_nv534_nch1024'   # round 2; round 1 was sic_v4x_d4x_nv401_nch512
ITER = 50


def load_recon(variant, it=ITER):
    path = os.path.join(PAD_DIR, f'{TAG}_{variant}_iter{it}.h5')
    with h5py.File(path, 'r') as f:
        # save_recon_hdf5 stores the volume under 'recon' (slice_viewer convention)
        name = 'recon' if 'recon' in f else list(f.keys())[0]
        vol = f[name][()]
    return vol


def main():
    new = load_recon('new')
    old = load_recon('old')
    n_ext = (new.shape[2] - old.shape[2]) // 2          # extension slices per end (symmetric here)
    print(f'shapes: new {new.shape}, old {old.shape}, extension per end = {n_ext}')

    # Shared-slab views: old slice k <-> new slice k + n_ext (offsets both 0, symmetric ext).
    new_shared = new[:, :, n_ext:n_ext + old.shape[2]]

    # Grayscale window from the old volume's interior (robust percentiles).
    interior = old[old.shape[0]//4:-old.shape[0]//4,
                   old.shape[1]//4:-old.shape[1]//4,
                   old.shape[2]//4:-old.shape[2]//4]
    vmin, vmax = 0.0, float(np.percentile(interior, 99.5))
    print(f'window: [0, {vmax:.4g}] (interior p99.5)')

    ymid = new.shape[0] // 2
    zoom = 4 * n_ext if n_ext > 0 else 64               # end-region depth to show

    # ---------------- Figure 1: x-z cross sections ----------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 9),
                             gridspec_kw={'width_ratios': [3, 1, 1]})
    panels = [
        ('old, full x-z', old[ymid].T, None),
        ('old, bottom end', old[ymid, :, :zoom].T, None),
        ('old, top end', old[ymid, :, -zoom:].T, None),
        ('new, full x-z', new[ymid].T, (n_ext, new.shape[2] - n_ext)),
        ('new, bottom end', new[ymid, :, :zoom + n_ext].T, (n_ext,)),
        ('new, top end', new[ymid, :, -(zoom + n_ext):].T, (zoom,)),
    ]
    for ax, (title, img, marks) in zip(axes.flat, panels):
        ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto', origin='lower')
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        for m in (marks or ()):
            ax.axhline(m, color='tab:red', lw=0.8, ls='--')
    fig.suptitle(f'{TAG} iter {ITER}: old ({old.shape[2]} slices) vs new ({new.shape[2]}; '
                 f'red dashes = old-slab boundary)', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(PAD_DIR, f'p3a_{TAG}_xz_iter{ITER}.png'), dpi=110)
    plt.close(fig)

    # ---------------- Figure 2: interior-disk z-profile ----------------
    rr, cc = np.ogrid[:old.shape[0], :old.shape[1]]
    r2 = (rr - old.shape[0]/2)**2 + (cc - old.shape[1]/2)**2
    disk = r2 <= (0.35 * old.shape[0])**2
    prof_old = np.array([float(np.abs(old[:, :, k][disk]).mean())
                         for k in range(old.shape[2])])
    prof_new = np.array([float(np.abs(new[:, :, k][disk]).mean())
                         for k in range(new.shape[2])])
    z_old = np.arange(old.shape[2]) + n_ext             # place old on the new slice axis
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(np.arange(new.shape[2]), prof_new, label='new (extended)', lw=1.2)
    ax.plot(z_old, prof_old, label='old', lw=1.2)
    for m in (n_ext, new.shape[2] - n_ext):
        ax.axvline(m, color='tab:red', lw=0.8, ls='--')
    ax.set_xlabel('slice (new-volume index; red dashes = old-slab boundary)')
    ax.set_ylabel('interior-disk mean |value|')
    ax.set_title(f'{TAG} iter {ITER}: axial profile, old vs new')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PAD_DIR, f'p3a_{TAG}_zprofile.png'), dpi=110)
    plt.close(fig)

    # ---------------- Figure 3: shared-slab difference ----------------
    diff = new_shared - old
    rms_slice = np.sqrt((diff**2).mean(axis=(0, 1)))
    body_rms = float(np.sqrt((old[..., old.shape[2]//4:-old.shape[2]//4]**2).mean()))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5),
                             gridspec_kw={'width_ratios': [1, 2]})
    axes[0].semilogy(rms_slice / body_rms)
    axes[0].set_xlabel('shared-slab slice'); axes[0].set_ylabel('RMS(new-old) / body RMS')
    axes[0].set_title('per-slice relative difference')
    im = axes[1].imshow(diff[ymid].T, cmap='coolwarm',
                        vmin=-0.1*vmax, vmax=0.1*vmax, aspect='auto', origin='lower')
    axes[1].set_title('difference, x-z at y mid (window +-10% of body window)')
    axes[1].set_xticks([]); axes[1].set_yticks([])
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(PAD_DIR, f'p3a_{TAG}_shared_diff.png'), dpi=110)
    plt.close(fig)

    # ---------------- Figure 4: convergence ----------------
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

    # ---------------- Quantitative capsule ----------------
    q = old.shape[2] // 4
    interior_rel = float(np.sqrt((diff[:, :, q:-q]**2).mean())) / body_rms
    print(f'interior (middle-half slab) RMS(new-old)/bodyRMS = {interior_rel:.4f}')
    for label, sl in (('bottom', slice(0, 8)), ('top', slice(-8, None))):
        rel = float(np.sqrt((diff[:, :, sl]**2).mean())) / body_rms
        print(f'{label}-end 8-slice RMS(new-old)/bodyRMS = {rel:.4f}')
    print(f'extension-region mean |value|: bottom {np.abs(new[:, :, :n_ext]).mean():.4g}, '
          f'top {np.abs(new[:, :, -n_ext:]).mean():.4g}, body scale {body_rms:.4g}')
    print('figures written to', PAD_DIR)


if __name__ == '__main__':
    main()
