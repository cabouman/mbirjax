"""Figures for the Phase 2c page rewrite: the Lilly seam stripes and the variants.

Two kinds of figure, built from the saved Lilly recon volumes (see lilly_split_repro.py /
lilly_split_ablations.py / lilly_ds8_check.py / lilly_ds8_deep.py for how those were made):

  exhibit   x-z sections near the seam (no-taper split | unsplit reference | difference)
            with the recon intensity WINDOWED around the body value so the stripes are
            clearly visible (the full range is dominated by bright inclusions, which
            previously left the body nearly black).
  variants  one row per split variant: the windowed recon and the difference against the
            MATCHING-iteration unsplit reference, each labeled with what the variant is
            and its peak per-slice seam RMS.

Run where the volumes live (any node; plotting only, no GPU).  All knobs below (no CLI
args).  Outputs land next to the volumes; copy into plans/experiments/flash_remediation/
figures/ and re-embed via embed_report_figures.py.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VOL_DIR = '/scratch/gautschi/buzzard/flash_lilly'
OUT_DIR = VOL_DIR

# Display windows, set from the measured slab statistics (same at 4x and 8x): body
# material 0.013-0.035 (median 0.02), dense pins 0.17-0.2, stripe amplitude up to 0.04.
# The recon window puts the body across the grayscale (pins saturate white); the
# difference window is ~the 99th percentile of |split - unsplit| so stripes render at
# full contrast.
RECON_WINDOW = (0.0, 0.05)
DIFF_WINDOW = 0.025     # difference panels show +/- this
SEAM_HALF_WIDTH = 16    # slices shown on each side of the split
INTERIOR_RADIUS_FRAC = 0.85

# Each figure set: the unsplit reference(s) and the variants to show, with display labels.
# 'ref' names the reference volume for the difference; iteration-matched per variant.
FIGSETS = {
    'p2c_lilly_stripes_xz': dict(
        kind='exhibit', split_vol='recon_split', ref_vol='recon_full',
        split_label='no-taper split (h_recon = 5), 15 iterations',
        ref_label='unsplit reference, 15 iterations',
        title='Lilly D01788 at 4x: the seam stripes'),
    'p2c_lilly_variants': dict(
        kind='variants', title='Lilly D01788 at 4x: split variants near the seam',
        rows=[
            ('recon_split', 'recon_full',
             'no taper, h_recon = 5, 15 iters (the pre-taper shipped design)'),
            ('recon_taper_15', 'recon_full',
             'sine taper, h_recon = 5, 15 iters (the shipped fix)'),
            ('recon_no_taper_deep_15', 'recon_full',
             'no taper, h_recon = 12 (geometry-derived), 15 iters'),
            ('recon_no_taper_60', 'recon_ref60',
             'no taper, h_recon = 5, 60 iters (structural: does not decay)'),
            ('recon_no_taper_deep_60', 'recon_ref60',
             'no taper, h_recon = 12, 60 iters (decays: normal convergence)'),
        ]),
    'p2c_lilly_variants_ds8': dict(
        kind='variants', title='Lilly D01788 at 8x: split variants near the seam',
        rows=[
            ('recon_no_taper_15_ds8', 'recon_ref_15_ds8',
             'no taper, h_recon = 5, 15 iters'),
            ('recon_taper_15_ds8', 'recon_ref_15_ds8',
             'sine taper, h_recon = 5, 15 iters (the shipped fix -- insufficient here)'),
            ('recon_deep9_15_ds8', 'recon_ref_15_ds8',
             'no taper, h_recon = 9 (the geometry formula), 15 iters'),
            ('recon_deep12_15_ds8', 'recon_ref_15_ds8',
             'no taper, h_recon = 12, 15 iters'),
        ]),
}


def load(name):
    path = os.path.join(VOL_DIR, name + '.npy')
    return np.load(path) if os.path.exists(path) else None


def split_index_of(shape):
    """The iso/split slice.  recon_slice_offset in mm over delta_slice; both scale with
    downsampling so the ratio is fixed: offset 0.4228 mm, delta 0.10837 mm at 4x
    (3.9 slices), 0.2167 mm at 8x (1.95 slices).  Derive from shape: 4x has 471 slices."""
    slices = shape[2]
    offset_slices = 3.9 if slices > 300 else 1.95
    return int(np.round((slices - 1) / 2.0 - offset_slices))


def seam_rms_peak(vol, ref, split_index, half_width):
    shape = ref.shape
    i = np.arange(shape[0], dtype=np.float32)[:, None] - (shape[0] - 1) / 2.0
    j = np.arange(shape[1], dtype=np.float32)[None, :] - (shape[1] - 1) / 2.0
    disk = np.sqrt(i ** 2 + j ** 2) < INTERIOR_RADIUS_FRAC * (min(shape[:2]) / 2.0)
    rms = np.sqrt(np.mean((vol - ref)[disk] ** 2, axis=0))
    lo, hi = max(0, split_index - half_width), min(shape[2], split_index + half_width + 1)
    return float(rms[lo:hi].max())


def xz(vol, z_lo, z_hi):
    return vol[:, vol.shape[1] // 2, z_lo:z_hi].T  # (z, x), z upward via origin='lower'


def make_figset(name, spec):
    if spec['kind'] == 'exhibit':
        rows = [(spec['split_vol'], spec['ref_vol'], spec['split_label'])]
    else:
        rows = spec['rows']
    ref0 = load(rows[0][1])
    if ref0 is None:
        print(f'{name}: missing {rows[0][1]}, skipped')
        return
    split_index = split_index_of(ref0.shape)
    z_lo, z_hi = split_index - SEAM_HALF_WIDTH, split_index + SEAM_HALF_WIDTH + 1
    vmin, vmax = RECON_WINDOW
    dmax = DIFF_WINDOW

    if spec['kind'] == 'exhibit':
        vol, ref = load(spec['split_vol']), load(spec['ref_vol'])
        panels = [(xz(vol, z_lo, z_hi), spec['split_label'], 'gray', vmin, vmax),
                  (xz(ref, z_lo, z_hi), spec['ref_label'], 'gray', vmin, vmax),
                  (xz(vol - ref, z_lo, z_hi), 'difference (split - unsplit)',
                   'coolwarm', -dmax, dmax)]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
        for ax, (img, label, cmap, lo_, hi_) in zip(axes, panels):
            im = ax.imshow(img, cmap=cmap, vmin=lo_, vmax=hi_, origin='lower',
                           aspect='auto', extent=(0, img.shape[1], z_lo, z_hi))
            ax.axhline(split_index, color='r' if cmap == 'gray' else 'k',
                       ls='--', lw=0.8)
            ax.set_title(label, fontsize=10)
            ax.set_xlabel('x')
            ax.set_ylabel('slice (z)')
            fig.colorbar(im, ax=ax, shrink=0.85)
        fig.suptitle(f"{spec['title']} (split at slice {split_index}; recon window "
                     f"{RECON_WINDOW[0]}-{RECON_WINDOW[1]}, difference +/-{DIFF_WINDOW})", fontsize=12)
    else:
        n = len(rows)
        fig, axes = plt.subplots(n, 2, figsize=(13, 2.1 * n + 0.8),
                                 constrained_layout=True)
        axes = np.atleast_2d(axes)
        for r, (vol_name, ref_name, label) in enumerate(rows):
            vol, ref = load(vol_name), load(ref_name)
            if vol is None or ref is None:
                axes[r, 0].set_visible(False)
                axes[r, 1].set_visible(False)
                print(f'{name}: missing {vol_name} or {ref_name}, row skipped')
                continue
            peak = seam_rms_peak(vol, ref, split_index, SEAM_HALF_WIDTH)
            im0 = axes[r, 0].imshow(xz(vol, z_lo, z_hi), cmap='gray', vmin=vmin,
                                    vmax=vmax, origin='lower', aspect='auto')
            im1 = axes[r, 1].imshow(xz(vol - ref, z_lo, z_hi), cmap='coolwarm',
                                    vmin=-dmax, vmax=dmax, origin='lower', aspect='auto')
            for c in (0, 1):
                axes[r, c].axhline(SEAM_HALF_WIDTH, color='r' if c == 0 else 'k',
                                   ls='--', lw=0.8)
                axes[r, c].set_yticks([])
                axes[r, c].set_xticks([])
            axes[r, 0].set_title(f'{label}\nrecon', fontsize=9, loc='left')
            axes[r, 1].set_title(f'difference vs matching unsplit; peak seam RMS '
                                 f'{peak:.1e}\n(dashed = split)', fontsize=9, loc='left')
            if r == n - 1:
                fig.colorbar(im0, ax=axes[r, 0], shrink=0.9, location='bottom')
                fig.colorbar(im1, ax=axes[r, 1], shrink=0.9, location='bottom')
        fig.suptitle(f"{spec['title']} (x-z at mid-y, split slice {split_index} "
                     f"+/- {SEAM_HALF_WIDTH}; recon window "
                     f"{RECON_WINDOW[0]}-{RECON_WINDOW[1]}, difference +/-{DIFF_WINDOW})", fontsize=12)
    out = os.path.join(OUT_DIR, name + '.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    for name, spec in FIGSETS.items():
        make_figset(name, spec)
    print('done: lilly_variant_figures')
