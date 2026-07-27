"""Streak metrics for the sharpness/snr_db study (plan: "Streak metric").

Streaks are z-coherent, in-plane high-frequency structure: a voxel cylinder that took
spurious steps differs from its in-plane neighbors in (nearly) the same way across
many slices.  All functions operate on host numpy volumes shaped (rows, cols, slices)
and return plain floats/arrays.

Metric roles (see the plan):
  - two_seed_score: PRIMARY, reference-free discriminator.  Partition-draw-driven
    streaks decorrelate between two runs that differ only in seed; true structure and
    systematic edge rendering cancel in the difference.
  - streak_score: secondary, per-run reference-based score (vs the ground truth
    phantom, or -- relative curves only -- vs a converged reference on real data).
  - streak_map + footprint_enrichment: the subset-footprint attribution probe
    (per-run maps, because subset membership is per-run).
"""

import numpy as np
from scipy import fft as sfft
from scipy import ndimage

import mbirjax as mj

# In-plane Gaussian high-pass scale (voxels): small enough to keep few-voxel streak
# spots, large enough to reject the smooth error background.
DEFAULT_HP_SIGMA = 2.0
# Central fraction of slices scored (avoids cone edge-slice effects).
DEFAULT_INTERIOR_SLICE_FRAC = 0.7
# ROR-radius fraction removed for scoring (keeps the mask off the ROR boundary ring).
DEFAULT_ERODE_FRAC = 0.05


def interior_mask(recon_shape, erode_frac=DEFAULT_ERODE_FRAC):
    """Boolean (rows, cols) mask: the ROR ellipse shrunk by erode_frac of its radius."""
    return mj.get_2d_ror_mask(recon_shape, crop_radius_fraction=erode_frac)


def highpass2d(img, hp_sigma=DEFAULT_HP_SIGMA):
    """img minus its Gaussian smoothing -- the in-plane high-pass component.
    Computed in float64 for filter accuracy; returns float64."""
    img = np.asarray(img, dtype=np.float64)
    return img - ndimage.gaussian_filter(img, hp_sigma)


def zcoherent_split(err_vol, interior_slice_frac=DEFAULT_INTERIOR_SLICE_FRAC,
                    z_step=1):
    """Split an error volume into its z-coherent map and the per-slice residual.

    Returns (c, resid): c(x, y) = mean over the central slices (the voxel-cylinder-
    aligned component), resid = the central slices minus c (the z-incoherent part).
    z_step subsamples the central slices first (cost control at full resolution; the
    statistics are unchanged for fields varying slowly vs z_step).
    """
    ns = err_vol.shape[2]
    lo = int(round(ns * (1.0 - interior_slice_frac) / 2.0))
    hi = ns - lo
    # float32 volume work (f64 doubled host memory at full res -- see
    # axial_power_spectrum); downstream scoring accumulates in f64.
    core = np.asarray(err_vol[:, :, lo:hi:z_step], dtype=np.float32)
    c = core.mean(axis=2, dtype=np.float64).astype(np.float32)
    return c, core - c[:, :, None]


def streak_score(err_vol, mask=None, hp_sigma=DEFAULT_HP_SIGMA,
                 interior_slice_frac=DEFAULT_INTERIOR_SLICE_FRAC, z_step=1):
    """S = mean squared in-plane high-pass of the z-coherent error over the interior.

    Returns dict(S=..., control=...): control is the matched z-INcoherent energy
    (mean over central slices of the per-slice high-pass of the residual).  S well
    above control indicates cylinder-aligned (streak-like) error specifically, not
    generic high-frequency noise.
    """
    err_vol = np.asarray(err_vol)
    if mask is None:
        mask = interior_mask(err_vol.shape)
    c, resid = zcoherent_split(err_vol, interior_slice_frac, z_step)
    S = float(np.mean(highpass2d(c, hp_sigma)[mask] ** 2))
    control = float(np.mean([np.mean(highpass2d(resid[:, :, k], hp_sigma)[mask] ** 2)
                             for k in range(resid.shape[2])]))
    return dict(S=S, control=control)


def two_seed_score(recon_a, recon_b, **kwargs):
    """PRIMARY reference-free discriminator: streak_score of (a - b)/sqrt(2).

    The two recons differ only in partition seed; seed-independent content cancels
    and two independent per-run streak fields add in variance, so the 1/sqrt(2) puts
    the score on a per-run scale (comparable to streak_score magnitudes).
    """
    d = (np.asarray(recon_a, dtype=np.float64)
         - np.asarray(recon_b, dtype=np.float64)) / np.sqrt(2.0)
    return streak_score(d, **kwargs)


def streak_map(err_vol, hp_sigma=DEFAULT_HP_SIGMA,
               interior_slice_frac=DEFAULT_INTERIOR_SLICE_FRAC, z_step=1):
    """|in-plane high-pass of the z-coherent error| as a (rows, cols) map -- the
    footprint probe's input.  Per-run (subset membership is per-run), so this uses a
    reference-based error volume, never the two-seed difference."""
    c, _ = zcoherent_split(err_vol, interior_slice_frac, z_step)
    return np.abs(highpass2d(c, hp_sigma))


# ─────────────────────────────────────────────────────────────────────────────
# Metric v2: the axial (z) power spectrum of the in-plane high-passed error.
#
# v1's split is the two ENDS of a spectrum: a z-CONSTANT cylinder error lives at
# axial frequency 0 (v1's S), per-slice speckle is flat across axial frequencies
# (v1's control).  The A2 damping-off finding showed real cylinder artifacts can
# VARY along z (partial-extent streaks, slice-varying common-mode kicks), which
# puts their energy at LOW but nonzero axial frequency -- invisible to v1's S,
# lumped into its control.  The spectrum resolves the whole range.
# ─────────────────────────────────────────────────────────────────────────────

def axial_power_spectrum(err_vol, mask=None, hp_sigma=DEFAULT_HP_SIGMA,
                         interior_slice_frac=DEFAULT_INTERIOR_SLICE_FRAC, z_step=1):
    """Mean axial power spectrum P(f_z) of the in-plane high-passed error.

    Steps: (1) per-slice in-plane Gaussian high-pass (keeps few-voxel in-plane
    detail, drops smooth background); (2) Hann window along z over the interior
    slices (limits leakage from the volume ends); (3) rFFT along z per (x, y);
    (4) average |.|^2 over the masked pixels, normalized by the window power so a
    unit-variance white (z-incoherent) field gives a flat spectrum of 1.

    z_step subsamples slices before the transform (cost control at full res); the
    spectrum then reaches Nyquist/z_step, ample for the low-f_z excess this metric
    targets (aliased speckle stays flat, so the baseline is unaffected).

    Returns:
        (freqs, P): frequencies in cycles/slice-sample and the mean power spectrum.
    """
    err_vol = np.asarray(err_vol)
    if mask is None:
        mask = interior_mask(err_vol.shape)
    ns = err_vol.shape[2]
    lo = int(round(ns * (1.0 - interior_slice_frac) / 2.0))
    hi = ns - lo
    # Volume-sized work in float32 (per-slice high-pass math still runs in f64
    # inside highpass2d): at full resolution the f64 core + windowed copy + c128
    # rFFT tripled host memory and OOM-killed the job -- f32 halves every term and
    # is far below the metric's discrimination scale.  The windowing is in place.
    core = err_vol[:, :, lo:hi:z_step]
    hp = np.empty(core.shape, dtype=np.float32)
    for k in range(hp.shape[2]):
        hp[:, :, k] = highpass2d(core[:, :, k], hp_sigma)
    n_z = hp.shape[2]
    window = np.hanning(n_z).astype(np.float32)
    hp *= window[None, None, :]
    # scipy.fft preserves single precision (numpy.fft would upcast to complex128).
    spec = sfft.rfft(hp, axis=2)
    del hp
    power = (np.abs(spec) ** 2)[mask].mean(axis=0, dtype=np.float64) \
        / (window.astype(np.float64) ** 2).sum()
    freqs = np.fft.rfftfreq(n_z, d=1.0)
    return freqs, power


def zcoherence_summary(freqs, power, low_cut=0.05, high_cut=0.25):
    """Collapse an axial spectrum into (S_low, S_high, Rz).

    S_low = mean power at f_z <= low_cut cycles/sample (including f_z = 0) -- the
    cylinder-coherent band, capturing z-constant AND slowly-z-varying streaks.
    S_high = mean power at f_z >= high_cut -- the speckle baseline.
    Rz = S_low / S_high -- ~1 for pure speckle, >> 1 when cylinder-coherent
    structure dominates.  Report S_low (absolute severity) alongside Rz
    (coherence character); a remedy should reduce S_low.
    """
    low = float(power[freqs <= low_cut].mean())
    high = float(power[freqs >= high_cut].mean())
    return dict(S_low=low, S_high=high, Rz=low / max(high, 1e-300))


def two_seed_spectrum(recon_a, recon_b, **kwargs):
    """Axial spectrum of the seed-difference field (a - b)/sqrt(2) -- the
    reference-free v2 primary, on the same per-run scale as axial_power_spectrum."""
    d = (np.asarray(recon_a, dtype=np.float64)
         - np.asarray(recon_b, dtype=np.float64)) / np.sqrt(2.0)
    return axial_power_spectrum(d, **kwargs)


def footprint_enrichment(map2d, partition_host, perm, mask=None):
    """Enrichment E(r) of a streak map over the rank-r-updated subset's pixels.

    Args:
        map2d: (rows, cols) nonnegative streak map (streak_map output).
        partition_host: (num_subsets, subset_size) flat indices into the flattened
            (rows, cols) grid -- one host partition from records['partitions_host'].
        perm: the update-order permutation for the iteration (records['perm'][i]):
            subset partition_host[perm[r]] was updated r-th.
        mask: optional (rows, cols) boolean restricting the comparison (default:
            all partition pixels).  Pass interior_mask(...) to exclude the bright
            band near the reconstruction-support boundary, whose large,
            update-order-independent energy otherwise DILUTES the enrichment
            (measured on the BGA scan: ~60% of map energy in the outer 6% of the
            radius; interior E(0) ~ 1.4 vs ~1.3 unmasked at iteration 0).

    Returns:
        (num_subsets,) array: mean map value over the rank-r subset (within mask)
        divided by the mean over all masked subset pixels.  E(0) >> 1 means the
        first-updated subset carries disproportionate streak energy.  Note:
        gen_pixel_partition replicates a few pixels to equalize subset sizes --
        negligible here.
    """
    flat = np.asarray(map2d, dtype=np.float64).ravel()
    partition_host = np.asarray(partition_host)
    if mask is None:
        keep = None
        base = float(flat[partition_host.ravel()].mean())
    else:
        keep = np.asarray(mask, dtype=bool).ravel()
        all_idx = partition_host.ravel()
        base = float(flat[all_idx[keep[all_idx]]].mean())
    enrichment = np.empty(partition_host.shape[0])
    for r in range(partition_host.shape[0]):
        idx = partition_host[perm[r]]
        if keep is not None:
            idx = idx[keep[idx]]
        enrichment[r] = float(flat[idx].mean()) / base
    return enrichment
