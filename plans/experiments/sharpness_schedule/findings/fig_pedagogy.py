"""Pedagogical figures for the findings page, section 2 (the instruments).

Built from the A2 downsampled-BGA outputs (real data reads better than synthetic
for these illustrations):

  two_seed_illustration.png  -- seed-A recon, seed-B recon, and their difference
                                (x,z) mid-planes: the object cancels, the
                                partition-driven artifact field remains.
  pfz_illustration.png       -- top: three idealized cylinder error profiles and
                                their axial spectra (what P(f_z) distinguishes);
                                bottom: the real two-seed field (in-plane
                                high-passed, (x,z) view) and its measured P(f_z).
  footprint_illustration.png -- iteration-0 update-order map, the same run's
                                streak map, and the enrichment E(r).

Run on gautschi next to the data (sbatch; no login-node compute):
  python -u fig_pedagogy.py
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'driver'))

import metrics  # noqa: E402

# ---------------------------------------------------------------- configuration
A2_ROOT = '/scratch/gautschi/buzzard/sharpness_schedule/a2_bga'
A1_ROOT = '/scratch/gautschi/buzzard/sharpness_schedule/a1'
RUN_A = os.path.join(A2_ROOT, 'center', 'seed1')
RUN_B = os.path.join(A2_ROOT, 'center', 'seed2')
RUN_DAMP = os.path.join(A2_ROOT, 'center_damp_off', 'seed1')
RUN_DAMP2 = os.path.join(A2_ROOT, 'center_damp_off', 'seed2')
REFERENCE = os.path.join(A2_ROOT, 'reference_recon.npy')
BGA_PATH = '/depot/bouman/data/Zeiss/purdue_BGA/17U1-250TC-Normal_Tomo_No_HART.txrm'
OUT_DIR = os.path.join(A2_ROOT, 'analysis')
# -------------------------------------------------------------------------------


def xz(vol):
    """(x, z) mid-plane with z as the vertical axis."""
    return np.asarray(vol)[vol.shape[0] // 2, :, :].T


def fig_two_seed(vol_a, vol_b):
    d = (vol_a - vol_b) / np.sqrt(2.0)
    vmax = float(np.percentile(vol_a, 99.9))
    dmax = float(np.percentile(np.abs(d), 99.8))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, img, ttl, kw in (
            (axes[0], xz(vol_a), 'reconstruction, seed A', dict(vmin=0, vmax=vmax, cmap='gray')),
            (axes[1], xz(vol_b), 'reconstruction, seed B', dict(vmin=0, vmax=vmax, cmap='gray')),
            (axes[2], xz(d), 'difference d = (A − B)/√2\n(~25× tighter color scale)',
             dict(vmin=-dmax, vmax=dmax, cmap='seismic'))):
        # Voxels are cubic (isotropic delta_voxel), so equal aspect is the true shape.
        im = ax.imshow(img, **kw, aspect='equal')
        ax.set_title(ttl, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.85)
    axes[0].set_ylabel('z (cylinder axis)')
    fig.suptitle('Two reconstructions differing only in the partition seed, and their difference\n'
                 '((x,z) mid-plane, real BGA scan; sharpness 1.5, snr_db 35, 15 iterations)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'two_seed_illustration.png'), dpi=140)
    plt.close(fig)
    return d


def fig_pfz(d, mask):
    # Idealized cylinder profiles and their spectra (same windowed-power recipe as
    # the metric, so shapes are directly comparable).
    n_z = 200
    z = np.arange(n_z)
    rng = np.random.default_rng(3)
    profiles = [
        ('constant along z', 0.8 * np.ones(n_z)),
        ('partial extent', 1.1 * np.exp(-0.5 * ((z - 70) / 18.0) ** 2)),
        ('uncorrelated in z', 0.8 * rng.standard_normal(n_z)),
    ]
    window = np.hanning(n_z)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    ax = axes[0, 0]
    for k, (label, p) in enumerate(profiles):
        ax.plot(z, p + 3.2 * (2 - k), label=label)
    ax.set_xlabel('z (slice index)')
    ax.set_yticks([])
    ax.set_title('three idealized cylinder error profiles (offset for display)')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    freqs_c = np.fft.rfftfreq(n_z, d=1.0)
    for label, p in profiles:
        power = np.abs(np.fft.rfft(p * window)) ** 2 / (window ** 2).sum()
        ax.plot(freqs_c, np.maximum(power, 1e-6), label=label)
    ax.axvspan(0, 0.05, color='#2563eb', alpha=0.10)
    ax.set_yscale('log')
    ax.set_xlabel('axial frequency $f_z$ (cycles/slice)')
    ax.set_title('their axial power spectra (shaded: the $S_{low}$ band)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # Real field: the in-plane high-passed two-seed difference, and its P(f_z).
    hp = np.empty_like(d, dtype=np.float32)
    for k in range(d.shape[2]):
        hp[:, :, k] = metrics.highpass2d(d[:, :, k])
    ax = axes[1, 0]
    hmax = float(np.percentile(np.abs(hp), 99.5))
    im = ax.imshow(xz(hp), vmin=-hmax, vmax=hmax, cmap='seismic', aspect='equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('z (cylinder axis)')
    ax.set_title('the measured field: in-plane high-pass of d ((x,z) view)\n'
                 'vertical striping = error organized along cylinders\n'
                 '(center variant: sharpness 1.5, snr_db 35, iteration 14 of 15)')
    plt.colorbar(im, ax=ax, shrink=0.85)

    ax = axes[1, 1]
    freqs, power = metrics.axial_power_spectrum(d, mask=mask)
    ax.plot(freqs, power, color='#1d4ed8')
    ax.axvspan(0, 0.05, color='#2563eb', alpha=0.10)
    high = power[freqs >= 0.25].mean()
    ax.axhline(high, color='#666', lw=0.9, ls='--')
    ax.annotate('cylinder-coherent excess', xy=(0.02, power[freqs <= 0.05].mean()),
                xytext=(0.13, power[freqs <= 0.05].mean() * 0.5),
                arrowprops=dict(arrowstyle='->', color='#333'), fontsize=9)
    ax.annotate('z-uncorrelated floor', xy=(0.4, high), xytext=(0.28, high * 6),
                arrowprops=dict(arrowstyle='->', color='#333'), fontsize=9)
    ax.set_yscale('log')
    ax.set_xlabel('axial frequency $f_z$ (cycles/slice)')
    ax.set_title('its measured $P(f_z)$')
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'pfz_illustration.png'), dpi=140)
    plt.close(fig)


def fig_footprint():
    with open(os.path.join(RUN_A, 'config.json')) as f:
        seq = json.load(f)['seq']
    rec = np.load(os.path.join(RUN_A, 'records.npz'), allow_pickle=True)
    part = rec[f'partition_entry{seq[0]}']
    perm = rec['perms'][0]
    smap = rec['streak_maps'][0]
    rows, cols = smap.shape

    rank_map = np.full(rows * cols, np.nan)
    for r in range(len(perm)):
        rank_map[part[perm[r]]] = r
    rank_map = rank_map.reshape(rows, cols)

    # E(r) over the eroded interior: the band near the support boundary carries a
    # large, rank-independent share of the map energy and dilutes the enrichment.
    interior = metrics.interior_mask(smap.shape)
    enrichment = metrics.footprint_enrichment(smap, part, perm, mask=interior)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    cmap = ListedColormap(plt.get_cmap('viridis')(np.linspace(0, 0.95, len(perm))))
    im = axes[0].imshow(rank_map, cmap=cmap)
    axes[0].set_title('iteration 0: cylinders colored by their\nsubset update rank r')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    cb = fig.colorbar(im, ax=axes[0], shrink=0.85, ticks=range(len(perm)))
    cb.set_label('update rank r')

    # Log scale: the linear map is dominated by the bright band near the support
    # boundary; log reveals the interior structure (ball row, package outline).
    pos = smap[smap > 0]
    vmin = float(np.percentile(pos, 5))
    vmax = float(np.percentile(pos, 99.9))
    from matplotlib.colors import LogNorm
    im = axes[1].imshow(np.clip(smap, vmin, None), norm=LogNorm(vmin=vmin, vmax=vmax),
                        cmap='magma')
    axes[1].set_title('the same run\'s streak map m(x,y)\nafter iteration 0 (log scale)')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im, ax=axes[1], shrink=0.85)

    axes[2].bar(range(len(perm)), enrichment, color='#2563eb')
    axes[2].axhline(1.0, color='k', lw=0.9)
    axes[2].set_xlabel('update rank r')
    axes[2].set_ylabel('enrichment E(r), interior')
    axes[2].set_title('streak energy per rank over the eroded\ninterior, relative to the mean')
    axes[2].grid(alpha=0.25, axis='y')

    fig.suptitle('The footprint probe on the real scan '
                 '(downsampled scan, default settings: sharpness 1.5, snr_db 35; seed 1)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'footprint_illustration.png'), dpi=140)
    plt.close(fig)


def fig_damp_compare():
    """Side-by-side (x,z) mid-plane error vs the reference, default settings vs
    damping disabled, on ONE shared intensity window -- the visual companion to the
    8.8x two-seed ranking (findings section 4a)."""
    reference = np.load(REFERENCE)
    err_c = xz(np.load(os.path.join(RUN_A, 'final_recon.npy')) - reference)
    err_d = xz(np.load(os.path.join(RUN_DAMP, 'final_recon.npy')) - reference)
    emax = float(np.percentile(np.abs(np.stack([err_c, err_d])), 99.5))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    for ax, img, ttl in ((axes[0], err_c, 'shp=1.5, snr=35'),
                         (axes[1], err_d, 'shp=1.5, snr=35, no damp')):
        im = ax.imshow(img, vmin=-emax, vmax=emax, cmap='seismic', aspect='equal')
        ax.set_title(ttl, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_ylabel('z (cylinder axis)')
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85,
                 label='error vs reference')
    fig.suptitle('Error vs the converged reference, (x,z) mid-plane, same intensity '
                 'window (final iteration of 15; seed 1)')
    fig.savefig(os.path.join(OUT_DIR, 'damp_compare_illustration.png'), dpi=140)
    plt.close(fig)


def fig_two_seed_damp_compare():
    """Two-seed difference fields for the default settings vs damping-off, on ONE
    shared window: the seed-dependent component alone -- the visual counterpart of
    the 8.8x two-seed S_low ratio (findings section 4a)."""
    d_c = xz((np.load(os.path.join(RUN_A, 'final_recon.npy'))
              - np.load(os.path.join(RUN_B, 'final_recon.npy'))) / np.sqrt(2.0))
    d_d = xz((np.load(os.path.join(RUN_DAMP, 'final_recon.npy'))
              - np.load(os.path.join(RUN_DAMP2, 'final_recon.npy'))) / np.sqrt(2.0))
    dmax = float(np.percentile(np.abs(np.stack([d_c, d_d])), 99.5))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    for ax, img, ttl in ((axes[0], d_c, 'shp=1.5, snr=35'),
                         (axes[1], d_d, 'shp=1.5, snr=35, no damp')):
        im = ax.imshow(img, vmin=-dmax, vmax=dmax, cmap='seismic', aspect='equal')
        ax.set_title(ttl, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_ylabel('z (cylinder axis)')
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85,
                 label='two-seed difference d')
    fig.suptitle('Two-seed difference fields, (x,z) mid-plane, same intensity window '
                 '(final iteration of 15; seeds 1−2)')
    fig.savefig(os.path.join(OUT_DIR, 'two_seed_damp_compare.png'), dpi=140)
    plt.close(fig)


def fig_synth_vs_real():
    """The synthetic control next to the real scan it imitates: one projection view
    of each sinogram (top) and a ball-grid reconstruction slice of each (bottom) --
    the likeness figure for the findings appendix on the synthetic null."""
    import mbirjax.preprocess as mjp
    syn_sino = np.load(os.path.join(A1_ROOT, 'sinogram_noisy.npy'))
    syn_rec = np.load(os.path.join(A1_ROOT, 'center', 'seed1', 'final_recon.npy'))
    real_sino, _ = mjp.zeiss.get_sino_and_model(
        BGA_PATH, downsample_factor=(3, 3), subsample_view_factor=5)
    real_sino = np.asarray(real_sino)
    real_rec = np.load(REFERENCE)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.6), constrained_layout=True)
    panels = (
        (axes[0, 0], syn_sino[0], 'synthetic: one projection view'),
        (axes[0, 1], real_sino[0], 'real scan: one projection view'),
        (axes[1, 0], syn_rec[:, :, syn_rec.shape[2] // 2],
         'synthetic: reconstruction, axial mid-slice\n(the ball layer)'),
        (axes[1, 1], xz(real_rec),
         'real scan: reconstruction, (x,z) mid-plane\n(the ball grid; converged reference)'),
    )
    for ax, img, ttl in panels:
        vmax = float(np.percentile(img, 99.9))
        ax.imshow(img, vmin=0, vmax=vmax, cmap='gray', aspect='equal')
        ax.set_title(ttl, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle('The synthetic control next to the real scan '
                 '(projection data above, reconstructions below; independent gray '
                 'windows)')
    fig.savefig(os.path.join(OUT_DIR, 'synth_vs_real.png'), dpi=140)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    vol_a = np.load(os.path.join(RUN_A, 'final_recon.npy'))
    vol_b = np.load(os.path.join(RUN_B, 'final_recon.npy'))
    mask = metrics.interior_mask(vol_a.shape)
    d = fig_two_seed(vol_a, vol_b)
    del vol_b
    fig_pfz(d, mask)
    del d, vol_a
    fig_footprint()
    fig_damp_compare()
    fig_two_seed_damp_compare()
    fig_synth_vs_real()
    print('pedagogy figures written to', OUT_DIR, flush=True)


if __name__ == '__main__':
    main()
