"""Shared machinery for the Phase 1 flash-characterization repros (plans/flash_remediation).

The reproduction pattern used by both scripts (lateral_truncation_repro.py /
z_truncation_repro.py):

1. Build a "truth" model that shares the real (small) detector but has an ENLARGED recon
   grid, and build the phantom on that enlarged grid so it extends outside the small
   model's field of view.  Forward-projecting the enlarged grid through the small detector
   IS the physical truncated measurement -- no sinogram cropping games needed.
2. Reconstruct with a default-shape model (what a user gets), snapshotting every iteration.
3. Compare against the center crop of the truth phantom, with the error split into an
   interior region and the flash regions (radial ring / end slices), so "interior
   convergence" and "flash buildup" are measured separately.

Both grids are centered (recon voxel x = delta_voxel * (j - (N-1)/2)), so as long as the
big-minus-small size difference is EVEN in every dimension, the small grid is exactly a
center crop of the big one and truth comparisons are alignment-exact.
"""

import numpy as np
import mbirjax as mj  # noqa: F401 -- must precede anything that touches jax
import matplotlib
matplotlib.use('Agg')  # headless: scripts save PNGs, never open windows
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Geometry / model construction
# ---------------------------------------------------------------------------

def make_cone_models(sinogram_shape, angles, source_detector_dist, source_iso_dist,
                     lateral_margin, slice_margin, sharpness=0.0):
    """Build the (recon_model, truth_model) pair for one repro case.

    recon_model keeps its automatic recon shape (the default a user gets: recon width ==
    detector FoV at iso, zero margin).  truth_model shares the identical detector and scan
    geometry but gets an enlarged recon grid: +lateral_margin voxels on EACH side of rows
    and cols, +slice_margin on EACH side in z (per-side margins keep the size difference
    even, so the center crop aligns exactly).
    """
    recon_model = mj.ConeBeamModel(sinogram_shape, angles,
                                   source_detector_dist=source_detector_dist,
                                   source_iso_dist=source_iso_dist)
    recon_model.set_params(sharpness=sharpness, verbose=0)

    truth_model = mj.ConeBeamModel(sinogram_shape, angles,
                                   source_detector_dist=source_detector_dist,
                                   source_iso_dist=source_iso_dist)
    truth_model.set_params(verbose=0)

    small_shape = recon_model.get_params('recon_shape')
    big_shape = (small_shape[0] + 2 * lateral_margin,
                 small_shape[1] + 2 * lateral_margin,
                 small_shape[2] + 2 * slice_margin)
    # Setting recon_shape directly preserves delta_voxel (already auto-set identically on
    # both models), so the big grid has the same voxel pitch and the same center.
    truth_model.set_params(recon_shape=big_shape)
    return recon_model, truth_model


def make_padded_model(recon_model, pad_scale_lateral=1.0, pad_scale_slices=1.0):
    """Build the padded-support variant: same geometry, recon support enlarged via
    scale_recon_shape (the existing utility for objects projecting outside the detector).

    scale_recon_shape multiplies each dimension by the scale and truncates to int, which
    can make (padded - default) ODD; a one-voxel parity fix keeps the padded grid's center
    crop exactly aligned with the default grid for truth comparisons.
    """
    padded_model = mj.copy_ct_model(recon_model)
    padded_model.set_params(verbose=0)
    small_shape = padded_model.get_params('recon_shape')
    padded_model.scale_recon_shape(row_scale=pad_scale_lateral, col_scale=pad_scale_lateral,
                                   slice_scale=pad_scale_slices)
    padded_shape = padded_model.get_params('recon_shape')
    fixed = tuple(p + ((p - s) % 2) for p, s in zip(padded_shape, small_shape))
    if fixed != tuple(padded_shape):
        padded_model.set_params(recon_shape=fixed)
    return padded_model


def center_crop(big_volume, small_shape):
    """Center-crop a (rows, cols, slices) volume to small_shape; sizes must differ evenly."""
    slices = []
    for big_n, small_n in zip(big_volume.shape, small_shape):
        diff = big_n - small_n
        if diff < 0 or diff % 2:
            raise ValueError(f'cannot center-crop {big_volume.shape} to {small_shape}: '
                             f'size differences must be non-negative and even')
        slices.append(slice(diff // 2, diff // 2 + small_n))
    return big_volume[tuple(slices)]


# ---------------------------------------------------------------------------
# Phantom
# ---------------------------------------------------------------------------

def build_phantom(big_shape, small_shape, delta_voxel, radius_frac, z_lo_frac, z_hi_frac,
                  target_line_integral=2.0, laminate_period=0):
    """Build the truth phantom on the big grid, sized relative to the SMALL model's FoV.

    The main body is a cylinder whose radius is radius_frac times the small FoV radius
    (radius_frac > 1 -> lateral truncation) and whose z extent is [z_lo_frac, z_hi_frac]
    in units of the small slab's half-height, measured from the slab center (so the slab
    itself is [-1, +1]; z_hi_frac > 1 -> the phantom sticks out the top, the SiC-like
    one-sided axial case).

    The cylinder's attenuation is set so a center ray's line integral is
    target_line_integral (keeps the synthetic data in a physically sane range).  Interior
    detail for judging visual quality: three spheres of varied contrast/size, and (when
    laminate_period > 0) alternating-contrast horizontal layers through the whole body --
    z structure that makes axial ringing visible, loosely like the SiC laminate.

    Returns a float32 numpy volume of big_shape.
    """
    big_rows, big_cols, big_slices = big_shape
    small_rows, small_cols, small_slices = small_shape

    # Voxel-index coordinates, centered like the recon grids.
    i = np.arange(big_rows, dtype=np.float32)[:, None, None] - (big_rows - 1) / 2.0
    j = np.arange(big_cols, dtype=np.float32)[None, :, None] - (big_cols - 1) / 2.0
    k = np.arange(big_slices, dtype=np.float32)[None, None, :] - (big_slices - 1) / 2.0

    fov_radius = small_cols / 2.0             # small-model FoV radius, in voxels
    half_slab = small_slices / 2.0            # small-model slab half-height, in voxels
    r = np.sqrt(i ** 2 + j ** 2)
    z = k / half_slab                         # slab = [-1, 1]

    body_radius = radius_frac * fov_radius
    body = (r <= body_radius) & (z >= z_lo_frac) & (z <= z_hi_frac)
    body_value = target_line_integral / (2.0 * body_radius * delta_voxel)

    phantom = np.zeros(big_shape, dtype=np.float32)
    phantom[body] = body_value

    # Laminate layers: alternate +/-20% in bands of laminate_period voxels in z.
    if laminate_period > 0:
        bands = (np.arange(big_slices) // laminate_period) % 2
        layer = np.where(bands, 1.2, 0.8).astype(np.float32)[None, None, :]
        phantom *= np.where(body, layer, 1.0)

    # Interior spheres (positions relative to the small FoV, mid-body in z): one bright,
    # one dark, one small-and-subtle -- enough structure to judge interior visual quality.
    z_mid_vox = 0.5 * (max(z_lo_frac, -1.0) + min(z_hi_frac, 1.0)) * half_slab
    spheres = [  # (row_frac, col_frac, z_off_vox, radius_frac_of_fov, relative_contrast)
        (0.40, 0.00, 0.0, 0.15, 1.6),
        (-0.20, -0.35, 0.0, 0.15, 0.4),
        (0.00, 0.45, 0.0, 0.07, 1.2),
    ]
    for row_f, col_f, z_off, rad_f, contrast in spheres:
        ci, cj, ck = row_f * fov_radius, col_f * fov_radius, z_mid_vox + z_off
        rad = rad_f * fov_radius
        inside = ((i - ci) ** 2 + (j - cj) ** 2 + (k - ck) ** 2) <= rad ** 2
        phantom[inside & body] = body_value * contrast

    return phantom


# ---------------------------------------------------------------------------
# Weight tapers and axial geometry (Phase 2)
# ---------------------------------------------------------------------------

def make_row_taper_weights(sinogram_shape, k_first=0, k_last=0):
    """Unit weights with a quarter-sine ramp over the first/last k detector rows.

    Matches the split_sino_recon precedent: sin((pi/2) * linspace(0, 1, k, endpoint=False)),
    so the weight is exactly 0 at the extreme row and rises toward the interior.  k_first
    tapers rows [0:k_first]; k_last tapers rows [-k_last:] (ramp reversed).  Row index 0 vs
    -1 orientation is the caller's responsibility (taper the side that is truncated).
    """
    num_views, num_rows, num_channels = sinogram_shape
    weights = np.ones(sinogram_shape, dtype=np.float32)
    if k_first > 0:
        ramp = np.sin((np.pi / 2) * np.linspace(0, 1, k_first, endpoint=False)).astype(np.float32)
        weights[:, :k_first, :] *= ramp[None, :, None]
    if k_last > 0:
        ramp = np.sin((np.pi / 2) * np.linspace(0, 1, k_last, endpoint=False)).astype(np.float32)
        weights[:, num_rows - k_last:, :] *= ramp[None, ::-1, None]
    return weights


def make_channel_taper_weights(sinogram_shape, k_each_side):
    """Unit weights with a quarter-sine ramp over the first and last k detector CHANNELS.

    The radial analog of make_row_taper_weights: lateral truncation is two-sided (outside
    material sweeps across both detector edges as the gantry rotates), so both edges taper.
    """
    num_views, num_rows, num_channels = sinogram_shape
    weights = np.ones(sinogram_shape, dtype=np.float32)
    if k_each_side > 0:
        ramp = np.sin((np.pi / 2) * np.linspace(0, 1, k_each_side,
                                                endpoint=False)).astype(np.float32)
        weights[:, :, :k_each_side] *= ramp[None, None, :]
        weights[:, :, num_channels - k_each_side:] *= ramp[None, None, ::-1]
    return weights


def cone_axial_geometry(model, psf_margin=2):
    """Axial (z) truncation geometry for a circular-orbit cone-beam model.

    With the source in the iso plane (zero helical shift), a ray to a detector row at
    physical height h reaches z = h * t / SDD at distance t from the source, and t inside
    the FoV cylinder is at most SID + R.  Two consequences, both returned here:

    - max_visible_z = h_max * (SID + R) / SDD: NO measured ray reaches |z| beyond this, so
      material further out never projects -- full z-padding never needs to exceed it, i.e.
      the padding cost in z is geometry-bounded (scale <= (SID + R) / SID, plus a psf margin).
    - taper_rows: the number of edge rows whose rays exit the recon slab inside the FoV
      (h > half_slab * SDD / (SID + R)) -- the principled row-taper width, plus psf_margin.

    Assumes centered geometry (recon_slice_offset == 0, det_row_offset == 0), as in these
    synthetic repros.  All lengths in the model's physical (ALU) units.
    """
    if float(model.get_params('recon_slice_offset')) != 0.0 or \
            float(model.get_params('det_row_offset')) != 0.0:
        raise ValueError('cone_axial_geometry assumes centered geometry (zero offsets).')
    sdd = float(model.get_params('source_detector_dist'))
    sid = float(model.get_params('source_iso_dist'))
    delta_det_row = float(model.get_params('delta_det_row'))
    num_det_rows = model.get_params('sinogram_shape')[1]
    recon_shape = model.get_params('recon_shape')
    delta_voxel = float(model.get_params('delta_voxel'))
    delta_slice = float(model.get_params('voxel_slice_aspect')) * delta_voxel

    half_slab = recon_shape[2] * delta_slice / 2.0
    fov_radius = min(recon_shape[0], recon_shape[1]) * delta_voxel / 2.0
    h_max = num_det_rows * delta_det_row / 2.0

    max_visible_z = h_max * (sid + fov_radius) / sdd
    full_pad_scale = (max_visible_z + psf_margin * delta_slice) / half_slab

    h_star = half_slab * sdd / (sid + fov_radius)  # rows above this exit the slab in-FoV
    exit_rows = max(0.0, (h_max - h_star) / delta_det_row)
    taper_rows = int(np.ceil(exit_rows)) + psf_margin if exit_rows > 0 else 0

    return {'half_slab': half_slab, 'fov_radius': fov_radius, 'max_visible_z': max_visible_z,
            'full_pad_scale': full_pad_scale, 'taper_rows': taper_rows}


def add_transmission_noise(sinogram, i0=1e4, seed=1):
    """Add photon-count (transmission) noise and return matching statistical weights.

    Model: expected counts I = i0 * exp(-sino); noisy counts I + sqrt(I) * N(0,1) (the
    Gaussian approximation to Poisson -- fine at these count levels); noisy sinogram
    = -log(counts / i0), with counts floored at 1 so the log stays finite.  The returned
    weights are mbirjax's standard 'transmission' inverse-variance weights computed from
    the NOISY sinogram, exactly as a user would build them.

    Seeded via numpy's Generator (does not touch the global np.random state that the VCD
    partition draws depend on).  Returns (noisy_sinogram, weights), both float32.
    """
    rng = np.random.default_rng(seed)
    counts = i0 * np.exp(-np.asarray(sinogram, dtype=np.float64))
    counts = counts + np.sqrt(counts) * rng.standard_normal(sinogram.shape)
    counts = np.maximum(counts, 1.0)
    noisy = (-np.log(counts / i0)).astype(np.float32)
    weights = np.asarray(mj.gen_weights(noisy, 'transmission'), dtype=np.float32)
    return noisy, weights


# ---------------------------------------------------------------------------
# Regions and metrics
# ---------------------------------------------------------------------------

def make_masks(small_shape, interior_radius_frac=0.85, end_slice_margin=4):
    """Region masks on the small grid, as a dict of boolean volumes:

    - 'interior':  radial interior (r < frac * R), central slices -- the part users care about.
    - 'ring':      the RoR edge annulus (frac * R <= r, inside the RoR mask), central slices
                   -- where the radial flash lives.
    - 'end_top' / 'end_bot': radial interior of the last/first end_slice_margin slices --
                   where the axial flash lives (kept separate: one-sided cases differ).
    - 'ror':       the full RoR cylinder (normalization region).
    """
    rows, cols, slices = small_shape
    ror2d = np.asarray(mj.get_2d_ror_mask(small_shape)).astype(bool)
    i = np.arange(rows, dtype=np.float32)[:, None] - (rows - 1) / 2.0
    j = np.arange(cols, dtype=np.float32)[None, :] - (cols - 1) / 2.0
    r = np.sqrt(i ** 2 + j ** 2)
    interior2d = (r < interior_radius_frac * (min(rows, cols) / 2.0)) & ror2d
    ring2d = ror2d & ~interior2d

    central = np.zeros(slices, dtype=bool)
    central[end_slice_margin:slices - end_slice_margin] = True
    top = np.zeros(slices, dtype=bool)
    top[slices - end_slice_margin:] = True
    bot = np.zeros(slices, dtype=bool)
    bot[:end_slice_margin] = True

    return {'interior': interior2d[:, :, None] & central[None, None, :],
            'ring': ring2d[:, :, None] & central[None, None, :],
            'end_top': interior2d[:, :, None] & top[None, None, :],
            'end_bot': interior2d[:, :, None] & bot[None, None, :],
            'ror': ror2d[:, :, None] & np.ones(slices, dtype=bool)[None, None, :]}


def region_metrics(recon, truth, masks, normalizer):
    """Per-region NRMSE (normalized by the fixed truth RMS) and signed mean excess."""
    err = recon - truth
    out = {}
    for name, mask in masks.items():
        if name == 'ror':
            continue
        out[f'nrmse_{name}'] = float(np.sqrt(np.mean(err[mask] ** 2)) / normalizer)
        out[f'excess_{name}'] = float(np.mean(err[mask]))
    return out


def run_tracked_recon(model, sinogram, truth_small, masks, num_iterations,
                      snapshot_iters=(), label='', weights=None):
    """Run recon one iteration at a time, computing per-region metrics after each.

    Uses the restart pattern (first_iteration=j, init_recon=previous) with np.random
    re-seeded before every call, so the partition draws match a single continuous run
    (lessons.md section 2: VCD partitions come from global np.random).  If the model's
    recon grid is larger than truth_small (the padded variant), each snapshot is center-cropped
    to the ground-truth phantom's grid before metrics, so all variants are compared on identical voxels.
    Optional weights (e.g. a row taper) are passed to every recon call; the auto
    regularization then derives sigma_y from the weighted sinogram, as it would for a user.

    Returns (metrics, snapshots): metrics is a dict of per-iteration lists (regions from
    region_metrics, plus 'change_pct', the mean |delta| / mean |recon| in percent -- a
    proxy for the stop metric); snapshots maps iteration -> cropped recon (the final
    iteration is always included).
    """
    normalizer = float(np.sqrt(np.mean(truth_small[masks['ror']] ** 2)))
    metrics = {}
    snapshots = {}
    init = None
    prev = None
    for it in range(num_iterations):
        np.random.seed(0)
        recon, _ = model.recon(sinogram, weights=weights, init_recon=init,
                               first_iteration=it, max_iterations=it + 1,
                               stop_threshold_change_pct=1e-9, print_logs=False)
        recon = np.asarray(recon)
        init = recon
        cropped = center_crop(recon, truth_small.shape)

        step = region_metrics(cropped, truth_small, masks, normalizer)
        denom = float(np.mean(np.abs(cropped))) + 1e-12
        step['change_pct'] = (100.0 * float(np.mean(np.abs(cropped - prev))) / denom
                              if prev is not None else np.nan)
        prev = cropped
        for key, value in step.items():
            metrics.setdefault(key, []).append(value)

        if it in snapshot_iters or it == num_iterations - 1:
            snapshots[it] = cropped
        if label:
            print(f'  [{label}] iter {it:3d}  interior {step["nrmse_interior"]:.4f}  '
                  f'ring {step["nrmse_ring"]:.4f}  end_top {step["nrmse_end_top"]:.4f}  '
                  f'change% {step["change_pct"]:.3f}', flush=True)
    return metrics, snapshots


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_convergence(metrics_by_variant, keys, title, path):
    """One subplot per metric key, one line per variant, iterations on x."""
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4), squeeze=False)
    for ax, key in zip(axes[0], keys):
        for variant, metrics in metrics_by_variant.items():
            ax.plot(metrics[key], label=variant)
        ax.set_xlabel('iteration')
        ax.set_title(key)
        if key.startswith('nrmse') or key == 'change_pct':
            ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_slice_montage(truth, recons_by_variant, axis, index, title, path, window=None,
                       region_mask=None, region_label='', reference_label='truth'):
    """Truth | per-variant recon | per-variant difference, along one slice axis.

    window: (vmin, vmax) for truth/recon panels; the difference panels use a symmetric
    window at 25% of the truth panel's span so subtle artifacts show.

    Variant dict keys are display labels and may be MULTI-LINE (e.g. a second line carrying
    a metric value); the difference row reuses only the first line.  region_mask (3D bool,
    truth-shaped) outlines its bounding box on the truth panel with a dashed red rectangle
    -- use it to show WHERE a reported metric is measured; region_label is appended to the
    truth panel's title.
    """
    def take(volume):
        section = np.take(volume, index, axis=axis)
        # Put z on the vertical axis for axial (x-z) sections so "top" reads as up.
        return section.T if axis in (0, 1) else section

    truth_slice = take(truth)
    # For transposed (x-z) sections, origin='lower' makes increasing slice index render
    # upward, so "top" in titles matches the picture.
    origin = 'lower' if axis in (0, 1) else 'upper'
    if window is None:
        window = (0.0, float(truth_slice.max()) * 1.1 + 1e-6)
    variants = list(recons_by_variant.items())
    fig, axes = plt.subplots(2, len(variants) + 1, figsize=(4 * (len(variants) + 1), 8),
                             squeeze=False)
    axes[0, 0].imshow(truth_slice, vmin=window[0], vmax=window[1], cmap='gray',
                      origin=origin)
    truth_title = reference_label
    if region_mask is not None:
        # Outline the metric region on the truth panel as a CONTOUR (same section/take
        # transform as the volumes, so it lands in imshow coordinates).  A contour follows
        # any region shape -- a rectangle for the end-slab regions, two circles for the
        # ring annulus -- where a bounding box would mislead.
        # No origin arg: contour uses array-index coordinates, which already align with
        # the imshow axes (imshow's origin setting flips the axis, not the data).
        mask_section = take(region_mask.astype(np.float32))
        axes[0, 0].contour(mask_section, levels=[0.5], colors='red',
                           linestyles='--', linewidths=1.5)
        if region_label:
            truth_title = f'{reference_label}\n({region_label})'
    axes[0, 0].set_title(truth_title)
    axes[1, 0].axis('off')
    diff_span = 0.25 * (window[1] - window[0])
    for col, (variant, recon) in enumerate(variants, start=1):
        recon_slice = take(recon)
        axes[0, col].imshow(recon_slice, vmin=window[0], vmax=window[1], cmap='gray',
                            origin=origin)
        axes[0, col].set_title(variant)
        short = variant.split('\n')[0].replace('recon: ', '')
        im = axes[1, col].imshow(recon_slice - truth_slice, vmin=-diff_span,
                                 vmax=diff_span, cmap='coolwarm', origin=origin)
        axes[1, col].set_title(f'{short} - {reference_label}')
        fig.colorbar(im, ax=axes[1, col], fraction=0.046)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_z_profile(truth, recons_by_variant, masks, title, path, xlim=None, truth_label='truth'):
    """Mean over the radial-interior disk, per slice: the instrument for axial flash/ringing.

    xlim: optional (lo, hi) slice range to zoom the plot (e.g. the last slices, where the
    variants actually differ).
    """
    interior2d = masks['interior'].any(axis=2)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    z_axis = np.arange(truth.shape[2])
    ax.plot(z_axis, truth[interior2d].mean(axis=0), 'k--', label=truth_label)
    for variant, recon in recons_by_variant.items():
        ax.plot(z_axis, recon[interior2d].mean(axis=0), label=variant)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel('slice index (z)')
    ax.set_ylabel('mean over interior disk')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_radial_profile(truth, recons_by_variant, small_shape, end_slice_margin, title, path):
    """Angle- and central-slice-averaged value vs radius: the instrument for the radial ring."""
    rows, cols, slices = small_shape
    i = np.arange(rows, dtype=np.float32)[:, None] - (rows - 1) / 2.0
    j = np.arange(cols, dtype=np.float32)[None, :] - (cols - 1) / 2.0
    r = np.sqrt(i ** 2 + j ** 2)
    bins = np.arange(0, min(rows, cols) / 2.0 + 1)
    which = np.digitize(r.ravel(), bins)
    central = slice(end_slice_margin, slices - end_slice_margin)

    def profile(volume):
        flat = volume[:, :, central].mean(axis=2).ravel()
        return np.array([flat[which == b].mean() if np.any(which == b) else np.nan
                         for b in range(1, len(bins))])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    centers = 0.5 * (bins[:-1] + bins[1:])
    ax.plot(centers, profile(truth), 'k--', label='truth')
    for variant, recon in recons_by_variant.items():
        ax.plot(centers, profile(recon), label=variant)
    ax.axvline(min(rows, cols) / 2.0, color='gray', lw=0.8, label='RoR boundary')
    ax.set_xlabel('radius (voxels)')
    ax.set_ylabel('mean over angle and central slices')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
