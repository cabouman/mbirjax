from collections import namedtuple
import numpy as np
import jax
import jax.numpy as jnp
import mbirjax
from functools import partial


# The ProjectorParams namedtuple TYPE is defined ONCE here (module level), not rebuilt per
# instance.  jax registers namedtuples as pytrees and keys the jit static-argument cache on the
# pytree treedef -- which includes the namedtuple CLASS -- so rebuilding the class per call (as
# namedtuple() inside a function does) would give each instance a distinct pytree type and defeat
# the shared module-level projector jit cache.  geometry_params gets the same treatment at its
# source (ParameterHandler.make_geometry_params).
#
# ``sort_by_channel`` (int: 1 = use the sorted segment-sum channel reduction in the FORWARD
# kernels, 0 = scatter-add; default 0) and ``back_stacked_gather`` (int: 1 = the BACK kernel
# gathers all psf taps in one stacked (psf_width * num_pixels, num_rows) gather followed by a
# reshape-sum over taps, 0 = the per-tap gather + FMA (fused multiply-add) loop; default 0)
# are the kernel-algorithm
# flags DECIDED by the model's tile policy
# (TomographyModel._select_tile_policy: GPU layouts whose slice bands are wide enough to
# amortize the sort) and baked in here by create_projectors.  The default keeps
# externally-constructed instances (tests, experiments) on the portable scatter path.  It is an
# INT, not a bool/str, because ProjectorParams is a pytree: some helpers take it TRACED (e.g.
# cone's compute_y_mag_for_pixel), where every leaf must be a valid jax type -- an unused int
# leaf traces harmlessly.  In the kernels projector_params is STATIC, so the flag is a concrete
# Python int there and the kernel's branch on it is trace-time.
ProjectorParams = namedtuple('ProjectorParams', ['sinogram_shape', 'recon_shape', 'geometry_params',
                                                 'sort_by_channel', 'back_stacked_gather'],
                             defaults=(0, 0))


# ──────────────────────────────────────────────────────────────────────────────
# Platform-split channel reduction for the forward-projection kernels
#
# Forward projection scatters each pixel's contribution into its detector channel.  Many
# pixels hit the same channel, and on GPU those colliding scatter-adds serialize (each is an
# ATOMIC memory update), making the duplicate-index scatter the dominant kernel cost.  The
# best formulation is platform-OPPOSITE (measured; the full study is
# plans/projector_kernels/fwd_back_findings.md):
#   * CPU: the plain per-tap scatter-add loop.  Sort-based alternatives are several times
#     WORSE there, so the CPU path is the original formulation, unchanged.
#   * GPU: the "SORTED channel reduction" (the term used throughout the geometry policies):
#     sort the (tap, pixel) contributions by channel, then a sorted segment-sum --
#     collision-free, 2-3x faster than the atomic scatter, same values.
# channel_scatter_reduce below implements both; ProjectorParams.sort_by_channel selects.
#
# The three constants below guard WHEN the sorted form actually wins.  Each encodes a
# measured crossover; the named scripts in plans/experiments/projector_kernels/ reproduce them.
#
# The sort's cost is per-call and nearly independent of the reduction's COLUMN count, while
# the scatter's grows with it -- so narrow slice bands favor the scatter.  End-to-end
# anchors: sorted loses at band length 24, wins at 63; 48 sits between them.
SORTED_CHANNEL_REDUCE_MIN_COLS = 48
# The sorted form also loses at WIDE psf (point-spread function: how many detector channels
# one voxel projects onto): full-kernel speedup 1.27x at psf_width 3, 1.02x at 5, 0.85x at 7
# (translation_fwd_psf_ab.py).  Radius 2 (width 5, the measured neutral point) is the
# inclusive cap; geometries whose psf can widen (translation at large cone angle, multiaxis
# at large elevation) gate their policy on this.
SORTED_CHANNEL_REDUCE_MAX_PSF_RADIUS = 2
# The sorted form's win comes from eliminating scatter collisions, so the controlling
# variable is the mean collisions per channel: psf_width * num_pixels / num_det_channels.
# Sorted wins at ratio >= 6; at ratio ~2 (wide detectors -- translation's real ~3000-channel
# TCT shapes) it is 4.5-6.5x SLOWER, a cliff rather than a taper
# (pixel_count_crossover_ab.py).  4 splits the measured bracket.  Parallel/cone policies
# predate this constant and their measured configurations all sit at ratio >= 6; add the
# guard there if very wide detectors with modest pixel batches become a real configuration.
SORTED_CHANNEL_REDUCE_MIN_COLLISION_RATIO = 4
# ──────────────────────────────────────────────────────────────────────────────
def channel_scatter_reduce(n, A, values, num_out, use_sorted=0):
    """Reduce weighted per-pixel rows into channel bins: out[c, :] = sum_{k,p: n[k,p]==c} A[k,p] * values[p, :].

    The shared reduction behind the forward kernels' channel scatter.  ``use_sorted`` picks the
    algorithm (see the note above; decided by the model's tile policy); both produce the same
    values up to float32 summation order.

    Contract: callers must pass ``n`` CLIPPED to [0, num_out-1] with ``A`` zeroed wherever the
    unclipped index was out of range (the kernels' existing range mask) -- both implementations
    then handle boundaries identically, with no reliance on scatter drop semantics.

    Args:
        n (int array, (psf_width, num_pixels)): channel index per psf tap per pixel,
            pre-clipped (psf_width = 2 * psf_radius + 1 taps).
        A (array, (psf_width, num_pixels)): weight per tap per pixel, zero where out of range.
        values (array, (num_pixels, num_cols)): the rows to be weighted and binned.
        num_out (int): number of output channels (static).
        use_sorted (int/bool): truthy = sorted segment-sum, else scatter-add.  Must be CONCRETE
            at trace time (from the static ProjectorParams.sort_by_channel).

    Returns:
        array of shape (num_out, num_cols).
    """
    if use_sorted:
        return _channel_reduce_sort_segsum(n, A, values, num_out)
    return _channel_reduce_scatter_add(n, A, values, num_out)


def _channel_reduce_scatter_add(n, A, values, num_out):
    """Unrolled per-tap scatter-add -- the original formulation (best on CPU)."""
    out = jnp.zeros((num_out, values.shape[1]))
    for k in range(n.shape[0]):        # unrolled over the psf_width taps, as the original loop
        out = out.at[n[k], :].add(A[k].reshape(-1, 1) * values)
    return out


def _channel_reduce_sort_segsum(n, A, values, num_out):
    """Sort contributions by channel, then a SORTED segment-sum (best on GPU: no colliding
    atomic adds).

    ``lax.sort_key_val`` returns the sorted keys and the permutation TOGETHER, and the sorted
    keys themselves are used as the segment ids -- so the ids are consistent with the sort by
    construction.  (Deliberate: an argsort-then-regather formulation would let the known
    round-inside-jit XLA hazard -- see plans/bugs_and_artifacts/jax rounding bug/ --
    produce ids inconsistent with the order, which indices_are_sorted=True would silently
    mis-reduce.)
    """
    num_pixels = values.shape[0]
    flat_n = n.reshape(-1)          # (psf_width * num_pixels,), row-major: tap-major blocks
    sorted_n, order = jax.lax.sort_key_val(flat_n, jnp.arange(flat_n.shape[0]))
    # Pixel p of tap block k sits at flat index k * num_pixels + p, so each sorted entry's
    # values row is order % num_pixels; gathering A and values in sorted order avoids
    # materializing a second (psf_width * num_pixels, num_cols) transient.
    updates = A.reshape(-1)[order][:, None] * values[order % num_pixels]
    return jax.ops.segment_sum(updates, sorted_n, num_segments=num_out, indices_are_sorted=True)


# ──────────────────────────────────────────────────────────────────────────────
# Concrete integer scatter centers (the horizontal-fan rounding-bug fix)
#
# The horizontal fans' integer channel centers, round(n_p), are computed in THIS separate
# small jit -- called eagerly by the public projector wrappers -- and passed into the
# projector programs as concrete arrays.  Reason: computing the round INSIDE those programs
# is the precondition of a known XLA miscompilation (round of an in-jit continuous
# projection coordinate feeding a scatter/gather inside a vmap -> (lax.map) -> scatter
# chain; see plans/bugs_and_artifacts/jax rounding bug/, esp. phase_d_design.md).
# Two consequences worth knowing: parallel beam's compiled forward/back programs contain no
# round at all, and forward/back consume ONE shared center value per (view, pixel), so the
# pair stays adjoint even at rounding ties.  The vertical fans' per-slice rounds remain
# in-jit -- a (views x pixels x slices) precompute is not materializable -- a documented,
# monitored risk (phase_d_design.md section 7).
#
# channel_coord_fn is the geometry's compute_channel_coordinate: the SAME float coordinate
# chain the kernels use for their weights, so centers and weights cannot disagree.
# pixels_major picks the output layout: False -> (views, pixels) for the back/band drivers
# (which batch views on the leading axis), True -> (pixels, views) for the forward driver
# (which batches pixels on the leading axis).
#
# N_PC_SINGLE_CALL_MAX_BYTES: above this size the public wrappers CHUNK the view axis (by
# the op's view batch) instead of materializing the full (views, pixels) int32 array as one
# driver input, bounding the resident centers array (~0.5 GB at a 1024^3 full grid).  The
# threshold only chooses where the view loop lives, never values.
N_PC_SINGLE_CALL_MAX_BYTES = 256 * 2**20


@partial(jax.jit, static_argnames=['channel_coord_fn', 'projector_params', 'pixels_major'])
def _jit_compute_scatter_centers(view_params_array, pixel_indices, channel_coord_fn,
                                 projector_params, pixels_major=False, owned_view_indices=()):
    """int32 channel centers round(n_p) for every (owned view, pixel); see the note above.

    owned_view_indices restricts to a view subset IN-JIT (a traced gather), exactly as the
    projector drivers do.  The wrappers must never gather view params EAGERLY: one eager
    array op per projector call costs ~1 ms of host time, which dominates workloads made of
    many small calls (this was a measured +35% on a VCD reconstruction).
    """
    if len(owned_view_indices) > 0:
        view_params_array = view_params_array[jnp.asarray(owned_view_indices)]

    def one_view(single_view_params):
        n_p = channel_coord_fn(pixel_indices, single_view_params, projector_params)
        return jnp.round(n_p).astype(jnp.int32)
    return jax.vmap(one_view, out_axes=1 if pixels_major else 0)(view_params_array)


# Donated eager accumulator for the chunked back paths: acc's buffer is reused, so the
# accumulation holds two (num_pixels, num_slices) arrays, not three.
_accumulate_donated = jax.jit(lambda acc, x: acc + x, donate_argnums=0)


# ──────────────────────────────────────────────────────────────────────────────
# Shared horizontal-fan kernels
#
# Every geometry's horizontal fan applies the same trapezoid rule; the geometry enters ONLY
# through (n_p, n_p_center, W_p_c) -- the continuous projected channel coordinate, its
# rounded center, and the projected voxel width in channel units -- plus a per-geometry
# weight scale.  These two helpers hold the tap loop and its platform-conditional
# alternatives ONCE instead of once per geometry:
#   * horizontal_fan_project (forward): scatter weighted pixel rows into channels; the
#     reduction is chosen by the tile policy's sort_by_channel flag (channel_scatter_reduce).
#   * horizontal_fan_back (adjoint): gather each pixel's weighted channel rows; per-tap loop
#     or one stacked gather (back_stacked_gather -- a GPU win only where no vertical fan
#     hides the gather; measured a composition no-op for cone/multiaxis/translation).
#
# Weight rule (identical across geometries): tap n = n_p_center + offset receives
#     A = weight_scale * clip((W_p_c + 1) / 2 - |n_p - n|, 0, min(1, W_p_c))
# -- the trapezoid overlap of the projected voxel with detector cell n -- zeroed outside the
# detector.  weight_scale is a scalar or per-pixel (num_pixels,) array (e.g. in-plane voxel
# area / footprint length); precomputing it outside the tap loop is value-identical to the
# historical in-loop expressions except translation, where dvr * L / cos became
# (dvr / cos) * L -- a deliberate, accepted reassociation at the ~1 ULP (unit in the last
# place, i.e. last-bit) level.
# ──────────────────────────────────────────────────────────────────────────────
def horizontal_fan_project(n_p, n_p_center, W_p_c, weight_scale, values, num_channels,
                           psf_radius, use_sorted=0):
    """Forward horizontal fan: bin weighted per-pixel rows into their detector channels.

    Args:
        n_p (array, (num_pixels,)): continuous projected channel coordinate per pixel.
        n_p_center (int array, (num_pixels,)): rounded center channel per pixel.
        W_p_c (scalar or (num_pixels,)): projected voxel width in channel units.
        weight_scale (scalar or (num_pixels,)): geometry weight applied to the trapezoid term.
        values (array, (num_pixels, num_cols)): the rows to weight and bin (voxel cylinders,
            or a vertical fan's output; num_cols = slices or detector rows).
        num_channels (int, static): number of detector channels.
        psf_radius (int, static): tap radius (psf_width = 2 * psf_radius + 1 taps).
        use_sorted (int/bool): truthy = the sorted segment-sum reduction (GPU), else the
            scatter-add loop.  Concrete at trace time (ProjectorParams.sort_by_channel).

    Returns:
        (num_channels, num_cols) CHANNEL-MAJOR partial view.  Channel-major so the scatter
        writes CONTIGUOUS rows (stride 1) rather than columns of stride num_channels (a
        power-of-2 column stride aliases the CPU cache); callers transpose on return (one
        cheap pass, fused by XLA).
    """
    L_max = jnp.minimum(1.0, W_p_c)
    if use_sorted:
        # Stack the taps; per the reduce contract, n is CLIPPED into range with the weights
        # zeroed where the unclipped tap was outside the detector.
        n_offsets = jnp.arange(start=-psf_radius, stop=psf_radius + 1)
        n = n_p_center[None, :] + n_offsets[:, None]              # (psf_width, num_pixels)
        abs_delta_p_c_n = jnp.abs(n_p - n)
        L_p_c_n = jnp.clip((W_p_c + 1.0) / 2.0 - abs_delta_p_c_n, 0.0, L_max)
        A_chan_n = weight_scale * L_p_c_n
        A_chan_n = A_chan_n * ((n >= 0) & (n < num_channels))
        n = jnp.clip(n, 0, num_channels - 1)
        return channel_scatter_reduce(n, A_chan_n, values, num_channels, use_sorted=True)

    # The per-tap scatter-add loop (the historical formulation; best on CPU).  Out-of-range
    # taps scatter with ZERO weight and a raw index -- jax drops out-of-bounds scatter
    # indices, so no clip is needed here.
    sinogram_view_T = jnp.zeros((num_channels, values.shape[1]))
    for n_offset in jnp.arange(start=-psf_radius, stop=psf_radius + 1):
        n = n_p_center + n_offset
        abs_delta_p_c_n = jnp.abs(n_p - n)
        L_p_c_n = jnp.clip((W_p_c + 1.0) / 2.0 - abs_delta_p_c_n, 0.0, L_max)
        A_chan_n = weight_scale * L_p_c_n
        A_chan_n *= (n >= 0) * (n < num_channels)
        sinogram_view_T = sinogram_view_T.at[n, :].add(A_chan_n.reshape((-1, 1)) * values)
    return sinogram_view_T


def horizontal_fan_back(sinogram_view_T, n_p, n_p_center, W_p_c, weight_scale,
                        psf_radius, coeff_power=1, use_stacked=0):
    """Back (adjoint) horizontal fan: gather each pixel's weighted channel rows.

    Args:
        sinogram_view_T (array, (num_channels, num_rows)): the view CHANNEL-MAJOR -- callers
            transpose up front so the per-pixel gather reads CONTIGUOUS rows (the adjoint of
            the forward helper's channel-major scatter).
        n_p / n_p_center / W_p_c / weight_scale / psf_radius: as in horizontal_fan_project.
        coeff_power (int, static): weights raised to this power (2 = the Hessian diagonal).
        use_stacked (int/bool): truthy = ONE stacked (psf_width * num_pixels, num_rows)
            gather covering every tap, then a reshape-sum over taps.  A measured GPU win
            only for parallel beam, whose back kernel has no vertical fan behind which the
            gather latency can hide; in the vertical-fan geometries it changes nothing in
            composition, so their policies leave it off.  Falsy = the per-tap gather + FMA
            (fused multiply-add) loop, which is also the best CPU form.  Concrete at trace
            time (ProjectorParams.back_stacked_gather).

    Returns:
        (num_pixels, num_rows) array of per-pixel weighted detector rows.
    """
    num_channels = sinogram_view_T.shape[0]
    num_pixels = n_p_center.shape[0]
    L_max = jnp.minimum(1.0, W_p_c)
    if use_stacked:
        n_offsets = jnp.arange(start=-psf_radius, stop=psf_radius + 1)
        n = n_p_center[None, :] + n_offsets[:, None]              # (psf_width, num_pixels)
        abs_delta_p_c_n = jnp.abs(n_p - n)
        L_p_c_n = jnp.clip((W_p_c + 1.0) / 2.0 - abs_delta_p_c_n, 0.0, L_max)
        A_chan_n = weight_scale * L_p_c_n
        A_chan_n = A_chan_n * ((n >= 0) & (n < num_channels))
        A_chan_n = A_chan_n ** coeff_power
        n = jnp.clip(n, 0, num_channels - 1)                      # weights already 0 outside
        # ONE (psf_width * num_pixels, num_rows) gather covering every tap:
        gathered = sinogram_view_T[n.reshape(-1), :]
        weighted = A_chan_n.reshape(-1)[:, None] * gathered
        return weighted.reshape(n.shape[0], num_pixels, -1).sum(axis=0)

    # The per-tap gather + fused-multiply-add loop (the historical formulation).  Out-of-range taps gather a
    # CLAMPED row (jax clamps out-of-bounds gather indices) with zero weight, so no explicit
    # clip is needed here.
    det_rows = jnp.zeros((num_pixels, sinogram_view_T.shape[1]))
    for n_offset in jnp.arange(start=-psf_radius, stop=psf_radius + 1):
        n = n_p_center + n_offset
        abs_delta_p_c_n = jnp.abs(n_p - n)
        L_p_c_n = jnp.clip((W_p_c + 1.0) / 2.0 - abs_delta_p_c_n, 0.0, L_max)
        A_chan_n = weight_scale * L_p_c_n
        A_chan_n *= (n >= 0) * (n < num_channels)
        A_chan_n = A_chan_n ** coeff_power
        det_rows = jnp.add(det_rows, A_chan_n.reshape((-1, 1)) * sinogram_view_T[n, :])
    return det_rows


# ──────────────────────────────────────────────────────────────────────────────
# The banded back vertical fan -- the shared contract (canonical note; the geometry files
# point here)
#
# Geometries whose back projection spreads one recon slice across a RANGE of detector rows
# (cone, multiaxis, translation) structure their back kernels as: HORIZONTAL fan once per
# view (it does not depend on the recon slice), then a VERTICAL fan evaluated over
# contiguous BANDS of global recon-slice indices [g0, g0 + L).  The banded vertical fan's
# contract, identical across those geometries:
#
#   * Physical z-coordinates come from the problem's recon_shape and the GLOBAL slice index
#     (k = g0 + k_local), never from the band length -- so any tiling of the slice axis
#     reproduces the full cylinder exactly (tiling-invariance; tested per geometry).
#   * A slice whose global index is at or beyond the real slice count receives zero, so the
#     zero-padding used to split the slice axis evenly across devices stays inert.
#   * Consumers: the single-device back projector walks the slice axis in fixed-size bands
#     with a ROLLED jax.lax.map (the band body compiles once and iterates at runtime, so
#     compiled-program size is independent of the slice count; the band size is the memory
#     knob).  back_project_one_view_to_band (horizontal fan once + ONE band) is the
#     per-band entry the multi-device reduce-scatter back projector uses.
#
# vertical_fan_band_gather below is the tap loop itself, shared VERBATIM by cone and
# translation (weights L / cos_alpha applied per tap).  MULTIAXIS deliberately keeps its own
# loop: its vertical fan is structurally different (pure-L interpolation weights with the
# mass-conserving amplitude applied POST-loop on the cylinder, mirroring its forward fan) --
# see MultiAxisParallelModel.back_vertical_fan_band_one_pixel.
# ──────────────────────────────────────────────────────────────────────────────
def vertical_fan_band_gather(detector_column_values, slice_indices, m_p, m_p_center, W_p_r,
                             weight_divisor, num_det_rows, num_recon_slices,
                             psf_radius, coeff_power=1):
    """Gather one pixel's detector column onto a band of global recon slices.

    Args:
        detector_column_values (array, (num_det_rows,)): this pixel's detector column.
        slice_indices (int array, (num_band_slices,)): GLOBAL slice indices g0 + arange(L).
        m_p (array, (num_band_slices,)): continuous projected detector-row coordinate
            per band slice.
        m_p_center (int array, (num_band_slices,)): rounded center row per band slice.
        W_p_r (scalar or (num_band_slices,)): projected voxel width in row units.
        weight_divisor (scalar or (num_band_slices,)): the geometry weight divisor
            (cos_alpha_p_z; divides the trapezoid term VERBATIM, preserving the historical
            arithmetic bit-for-bit).
        num_det_rows (int, static): detector rows.
        num_recon_slices (int, static): the REAL slice count (padded slices beyond it are
            zeroed, keeping padding inert).
        psf_radius (int, static): tap radius.
        coeff_power (int, static): weights raised to this power (2 = the Hessian diagonal).

    Returns:
        (num_band_slices,) voxel values for global slices [g0, g0 + L).
    """
    L_max = jnp.minimum(1, W_p_r)
    new_cylinder = jnp.zeros(slice_indices.shape[0])
    for m_offset in jnp.arange(start=-psf_radius, stop=psf_radius + 1):
        m = m_p_center + m_offset
        abs_delta_p_r_m = jnp.abs(m_p - m)
        L_p_r_m = jnp.clip((W_p_r + 1) / 2 - abs_delta_p_r_m, 0, L_max)
        A_row_m = L_p_r_m / weight_divisor
        A_row_m *= (m >= 0) * (m < num_det_rows)
        A_row_m = A_row_m ** coeff_power
        new_cylinder = jnp.add(new_cylinder, A_row_m * detector_column_values[m])
    # Padded global slices (index >= the real slice count) are inert.  No-op when
    # g0 + L <= num_recon_slices.
    new_cylinder = new_cylinder * (slice_indices < num_recon_slices)
    return new_cylinder


class Projectors:

    def __init__(self, tomography_model):

        self.tomography_model = tomography_model
        self.sparse_forward_project, self.sparse_back_project = None, None
        self.sparse_back_project_band = None   # set in create_projectors only for banded geometries
        self.create_projectors(tomography_model)

    def create_projectors(self, tomography_model):
        """
        Compute the forward and back projectors for this geometry and current view parameters

        Args:
            tomography_model (mbirjax.TomographyModel): An instance describing the current geometry and implementing the following 2 functions:

                * forward_project_pixel_batch_to_one_view (callable): jit-compilable function implementing :meth:`TomographyModel.forward_project_pixel_batch_to_one_view`
                * back_project_one_view_to_pixel_batch (callable): jit-compilable function implementing :meth:`TomographyModel.back_project_one_view_to_pixel_batch`

        Returns:
            Nothing, but the class variables `sparse_forward_project` and `sparse_back_project` are set to callable
            functions.  These are used to implement the following methods:

            * `sparse_forward_project`: :meth:`TomographyModel.sparse_forward_project`
            * `sparse_back_project`: :meth:`TomographyModel.sparse_back_project`

        Note:
            The returned functions will be jit compiled each time they are called with a new shape of input.  If
            called multiple times with the same shape of input, then the cached version will be used, which will
            give reduced execution time relative to the initial call.

            This method requires geometry-specific implementations of
            :meth:`TomographyModel.forward_project_pixel_batch_to_one_view` and
            :meth:`TomographyModel.back_project_one_view_to_pixel_batch`.

        """
        forward_project_pixel_batch_to_one_view = tomography_model.forward_project_pixel_batch_to_one_view
        back_project_one_view_to_pixel_batch = tomography_model.back_project_one_view_to_pixel_batch
        # Geometries whose back projection bleeds a slice across a RANGE of detector rows provide a
        # BANDED back kernel (cone; later translation/multiaxis), used by the sharded slice-band
        # path; parallel beam has none (it crops detector rows instead).
        back_project_one_view_to_band = getattr(tomography_model, 'back_project_one_view_to_band', None)

        # geometry_params already uses a shared namedtuple class (make_geometry_params); combine it
        # with the shapes into the module-level ProjectorParams type so the module-level projector
        # jit cache is shared across instances (see the note on ProjectorParams at the top).
        geometry_params = self.tomography_model.get_geometry_parameters()
        sinogram_shape, recon_shape = self.tomography_model.get_params(['sinogram_shape', 'recon_shape'])
        # sort_by_channel: the kernel-algorithm flag from the model's tile policy (selected at
        # device layout; the layout exists by the time create_projectors runs -- set_devices
        # precedes it in __init__ and in the set_params recompile).  Edge: configure_devices()
        # re-lays-out (new tiles) WITHOUT a projector rebuild, so this baked flag can go stale
        # across such a re-layout -- that costs only speed, never correctness (the reductions
        # are value-equal), and every sanctioned platform switch (use_gpu=...) goes through a
        # set_params recompile.
        tiles = self.tomography_model.tiles
        sort_by_channel = int(bool(tiles is not None and tiles.sort_by_channel))
        back_stacked_gather = int(bool(tiles is not None and tiles.back_stacked_gather))
        projector_params = ProjectorParams(sinogram_shape, recon_shape, geometry_params,
                                           sort_by_channel, back_stacked_gather)
        # Exposed for consumers outside the jitted drivers (the pallas back path reads
        # shapes/geometry/psf_radius from it -- fields that cannot go stale; the baked
        # kernel-algorithm flags carry the staleness caveat above).
        self.projector_params = projector_params

        view_params_name = self.tomography_model.get_params('view_params_name')
        # The view parameters are a RUNTIME input to the jitted projectors, not a baked
        # (closure-captured) constant: the current array is stored on this Projectors
        # object and passed as a traced argument on every call.  Because only the VALUES
        # are traced (the shape is static), TomographyModel.set_view_parameters can change
        # the angles/translations with NO recompile; a view-COUNT change is a geometry
        # change and rebuilds the projectors through set_params as before.
        self.view_params_array = jnp.asarray(self.tomography_model.get_params(view_params_name))
        # The batch-size knobs are NOT captured here: the public wrappers below read the
        # model's TILE POLICY (tm.tiles) AT CALL TIME, for the same reason
        # view_params_array is late-bound -- configure_devices() re-lays-out WITHOUT
        # recreating the projectors, and a construction-time capture would silently keep
        # running the first layout's values.  The knobs are STATIC jit arguments, so a
        # changed value retraces; it can never compute a wrong result.
        tm = self.tomography_model

        # The jitted drivers are MODULE-LEVEL functions (defined at the end of this file),
        # not per-instance closures, so their jit cache is SHARED across model instances: a
        # second model with the same geometry reuses the first model's compiled program
        # instead of re-tracing.  The geometry-specific pieces that used to be captured by
        # closure -- the per-view kernel, projector_params, and the two batch sizes -- are
        # passed as STATIC arguments (hashable; projector_params is already static in every
        # per-view kernel).  The view-parameter array stays a TRACED argument, so
        # set_view_parameters changes its values with no recompile.
        #
        # Keep references to the module-level jitted functions on this object so existing
        # introspection (e.g. _jit_sparse_forward_project._cache_size()) still works; the
        # cache they expose is now the one shared across instances.
        self._jit_sparse_forward_project = _jit_sparse_forward_project
        self._jit_sparse_back_project = _jit_sparse_back_project

        # ── Concrete integer scatter centers ─────────────────────────────
        # The wrappers below compute the horizontal fans' integer channel centers EAGERLY
        # (in the separate _jit_compute_scatter_centers program) and pass them into the
        # jitted drivers as concrete arrays -- see the scatter-centers note at the top of this
        # file.  channel_coord_fn is the geometry's float coordinate chain, so the centers
        # are consistent with the in-kernel weights by construction, and forward/back share
        # ONE center value (adjoint even at rounding ties).
        channel_coord_fn = tomography_model.compute_channel_coordinate

        # PERFORMANCE CONTRACT for these wrappers: NO eager array operations per call.
        # They run once per projector call -- hundreds of times per VCD iteration -- and one
        # eager gather/slice costs ~1 ms of host time (an eager per-call view-params gather
        # measured as +35% on a VCD recon).  owned_view_indices passes
        # THROUGH to the jitted programs, which gather in-trace, exactly as the drivers
        # always did.
        def scatter_centers(pixel_indices, pixels_major, owned_view_indices):
            n_pc = _jit_compute_scatter_centers(
                self.view_params_array, pixel_indices, channel_coord_fn=channel_coord_fn,
                projector_params=projector_params, pixels_major=pixels_major,
                owned_view_indices=owned_view_indices)
            # The CONCRETENESS contract: if these wrappers are ever swallowed by an outer
            # jit, the centers become tracers and the rounding-bug precondition silently
            # returns.  Guard loudly instead.
            assert not isinstance(n_pc, jax.core.Tracer), (
                'The projector wrappers must run OUTSIDE any jit: the integer scatter '
                'centers have to reach the projector programs as CONCRETE arrays '
                '(see plans/bugs_and_artifacts/jax rounding bug/phase_d_design.md).')
            return n_pc

        def num_owned(owned_view_indices):
            n = len(owned_view_indices)
            return n if n > 0 else int(self.view_params_array.shape[0])

        # Public entry points keep the original signatures; they read the CURRENT
        # view-parameter array off this object at call time (late binding), so
        # set_view_parameters takes effect on the next call with no recompile.
        # Above N_PC_SINGLE_CALL_MAX_BYTES they CHUNK the view axis by the op's view batch
        # (bounding the resident centers array) and reuse ONE compiled driver per full
        # chunk; the threshold only chooses where the view loop lives, never values.  The
        # chunked paths may do a few eager slices (bounded by the chunk count, amortized
        # over large calls); the SINGLE-call path -- every small/frequent call -- does none.
        def sparse_forward_project_public(voxel_values, pixel_indices, owned_view_indices=()):
            num_views_owned = num_owned(owned_view_indices)
            view_batch = tm.tiles.fwd_view_batch

            def one_call(owned_chunk):
                n_pc = scatter_centers(pixel_indices, pixels_major=True,
                                       owned_view_indices=owned_chunk)
                return _jit_sparse_forward_project(
                    self.view_params_array, voxel_values, pixel_indices, n_pc,
                    fwd_kernel=forward_project_pixel_batch_to_one_view,
                    projector_params=projector_params,
                    pixel_batch_size=tm.tiles.fwd_pixel_batch,
                    view_batch_size=view_batch,
                    owned_view_indices=owned_chunk)

            if 4 * num_views_owned * int(pixel_indices.shape[0]) <= N_PC_SINGLE_CALL_MAX_BYTES:
                return one_call(owned_view_indices)
            outputs = [one_call(owned_view_indices[start:start + view_batch]
                                if len(owned_view_indices) > 0
                                else np.arange(start, min(start + view_batch, num_views_owned)))
                       for start in range(0, num_views_owned, view_batch)]
            return jnp.concatenate(outputs, axis=0)

        def sparse_back_project_public(sinogram, pixel_indices, coeff_power=1, owned_view_indices=()):
            num_views_owned = num_owned(owned_view_indices)
            view_batch = tm.tiles.back_view_batch

            def one_call(sino_chunk, owned_chunk):
                n_pc = scatter_centers(pixel_indices, pixels_major=False,
                                       owned_view_indices=owned_chunk)
                return _jit_sparse_back_project(
                    self.view_params_array, sino_chunk, pixel_indices, n_pc,
                    back_kernel=back_project_one_view_to_pixel_batch,
                    projector_params=projector_params,
                    pixel_batch_size=tm.tiles.back_pixel_batch,
                    view_batch_size=view_batch,
                    coeff_power=coeff_power,
                    owned_view_indices=owned_chunk)

            single_call = (4 * num_views_owned * int(pixel_indices.shape[0])
                           <= N_PC_SINGLE_CALL_MAX_BYTES)
            # Chunking slices the sinogram's VIEW axis eagerly, which is only safe on a
            # single-device array (slicing a sharded axis replicates; the sharded paths
            # call this wrapper per device with local arrays, so they qualify).
            if not single_call and isinstance(sinogram, jax.Array) and len(sinogram.devices()) > 1:
                single_call = True
            if single_call:
                return one_call(sinogram, owned_view_indices)
            result = None
            for start in range(0, num_views_owned, view_batch):
                stop = min(start + view_batch, num_views_owned)
                owned_chunk = (owned_view_indices[start:stop] if len(owned_view_indices) > 0
                               else np.arange(start, stop))
                out = one_call(sinogram[start:stop], owned_chunk)
                result = out if result is None else _accumulate_donated(result, out)
            return result

        self.sparse_forward_project = sparse_forward_project_public
        self.sparse_back_project = sparse_back_project_public

        # Banded back projector (only for geometries with a banded kernel -- cone): projects a
        # view-owner's full views onto ONE global slice band [g0, g0 + num_band_slices), batched
        # and summed exactly like sparse_back_project so memory stays bounded by the batch sizes,
        # not by the view count.  Used by the sharded reduce-scatter (see
        # TomographyModel._back_project_view_shard_to_band).  g0 is traced (per band),
        # num_band_slices is static.
        if back_project_one_view_to_band is not None:
            def sparse_back_project_band_public(sinogram, pixel_indices, g0, num_band_slices,
                                                owned_view_indices=(), coeff_power=1):
                num_views_owned = num_owned(owned_view_indices)
                view_batch = tm.tiles.back_view_batch

                def one_call(sino_chunk, owned_chunk):
                    n_pc = scatter_centers(pixel_indices, pixels_major=False,
                                           owned_view_indices=owned_chunk)
                    return _jit_sparse_back_project_band(
                        self.view_params_array, sino_chunk, pixel_indices, n_pc,
                        g0, num_band_slices,
                        back_band_kernel=back_project_one_view_to_band,
                        projector_params=projector_params,
                        pixel_batch_size=tm.tiles.back_pixel_batch,
                        view_batch_size=view_batch,
                        coeff_power=coeff_power,
                        owned_view_indices=owned_chunk)

                single_call = (4 * num_views_owned * int(pixel_indices.shape[0])
                               <= N_PC_SINGLE_CALL_MAX_BYTES)
                if not single_call and isinstance(sinogram, jax.Array) and len(sinogram.devices()) > 1:
                    single_call = True   # never slice a sharded view axis; see sparse_back_project
                if single_call:
                    return one_call(sinogram, owned_view_indices)
                result = None
                for start in range(0, num_views_owned, view_batch):
                    stop = min(start + view_batch, num_views_owned)
                    owned_chunk = (owned_view_indices[start:stop] if len(owned_view_indices) > 0
                                   else np.arange(start, stop))
                    out = one_call(sinogram[start:stop], owned_chunk)
                    result = out if result is None else _accumulate_donated(result, out)
                return result

            self.sparse_back_project_band = sparse_back_project_band_public
            self._jit_sparse_back_project_band = _jit_sparse_back_project_band


def concatenate_function_in_batches(function, data_to_batch, batch_size):
    """
    Apply a given function to a set of data, batching over the first index, concatenating the results along axis=0.
    The function should operate on subsets of the form data_to_batch[start:start+batch_size] when function takes a
    single input or on analogous subsets of each element data_to_batch[j] when function takes multiple
    inputs.  The output of function should be an array or tuple of arrays.  These output are concatenated along
    the leading axis.

    The shape of each array is determined by the output(s) of function, which
    is concatenated along the leading axis.  If function returns a fixed shape output, then the result has
    size given by the total number batches (full or partial) times the length of the leading axis of the output.

    Args:
        function (callable): A function to be mapped over batches of the input data.  This will be called on
        the unpacked elements of data_to_batch (after batching) using a call of the form
        output_data = function(batch) when data_to_batch is a single array or
        output_data = function(*batch) when data_to_batch is a tuple.
        data_to_batch (jax array or tuple of arrays): An array of data to be batched and sent to function. If a tuple,
        then each element should have the same size leading axis.
        batch_size (int): The maximum number of entries to process at one time.

    Returns:
        An array or tuple of arrays.
    """
    data_to_batch = ensure_tuple(data_to_batch)

    # Apply the batch projector directly to an initial batch
    num_input_points = data_to_batch[0].shape[0]
    batch_size = num_input_points if batch_size is None else batch_size
    num_remaining = num_input_points % batch_size

    # If the input is a multiple of batch_size, then we'll do a full batch, otherwise just the excess.
    initial_batch_size = batch_size if num_remaining == 0 else num_remaining

    initial_batch = [data[:initial_batch_size] for data in data_to_batch]
    output_data = function(*initial_batch)

    # Then deal with the batches if there are any
    if batch_size < num_input_points:
        def wrapped_function(arg_list):
            return function(*arg_list)

        num_batches = (num_input_points - initial_batch_size) // batch_size
        output_shape = (num_batches, batch_size,)
        data_batched = [jnp.reshape(data[initial_batch_size:], output_shape + data.shape[1:])
                        for data in data_to_batch]

        # Apply the function in batches
        output_data_batched = jax.lax.map(wrapped_function, data_batched)

        # The output data may be a single array or a tuple or list of arrays
        # First unbatch the data by reshaping the first 2 dims to be the number of points in all the batches.
        # Using tree_map, this can be done on either a single array or a tuple of arrays and get either a
        # single array or a tuple back
        output_unbatched = jax.tree_util.tree_map(unbatch, output_data_batched)

        # Now stack the first partial batch with this unbatched result
        output_data = jax.tree_util.tree_map(concatenate_arrays, *(output_data, output_unbatched))

    return output_data


def unbatch(array):
    """
    Reshape a jax array from (n0, n1, ...) to (n0*n1, ...)

    Args:
        array (jax array): array to be reshaped

    Returns:
        jax array
    """
    return jax.numpy.reshape(array, (array.shape[0] * array.shape[1],) + array.shape[2:])


def concatenate_arrays(*arrays):
    """
    Helper function to concatenate a list or tuple of arrays along the leading axis.

    Args:
        *arrays: list of arrays, with compatible dimensions arrays[j].shape[1:]

    Returns:
        array of shape (n, ) + arrays[0].shape[1:], where n is the sum over j of arrays[j].shape[0]
    """
    return jax.numpy.concatenate(arrays, axis=0)


def sum_function_in_batches(function_to_sum, data_to_batch, batch_size, extra_args=()):
    """
    Apply a given function to a set of data, batching over the first index, summing the results.
    The function should operate on subsets of the form data_to_batch[start:start+batch_size] when function takes a
    single input or on analogous subsets of each element data_to_batch[j] when function takes multiple
    inputs.  The output of function should be a scalar or fixed size array.

    Args:
        function_to_sum (callable): A function to be mapped over batches of the input data.  This will be called on
        the unpacked elements of data_to_batch (after batching) and extra_args using a call of the form
        summed_data += function_to_sum(batched_data, *fixed_data) when data_to_batch is a single array or
        summed_data += function_to_sum(*batched_data, *fixed_data) when data_to_batch is a tuple.
        data_to_batch (jax array or tuple of arrays): The data to be processed in batches.  If a tuple, then each element
        should have the same size leading axis.
        batch_size (int): The maximum batch size.
        extra_args (tuple): Any additional arguments needed by function_to_sum

    Returns:
        jax array or scalar output of function_to_sum, summed over all the elements in data_to_batch.
    """
    data_to_batch = ensure_tuple(data_to_batch)
    extra_args = ensure_tuple(extra_args)

    def add_one_batch(summed_and_fixed_data, batched_data):
        """
        Apply the externally defined function function_to_sum to the data in the tuple batched_data
        and add the result to an existing result.  The existing result is the first element in the tuple
        summed_and_fixed_data.  Any remaining elements of summed_and_fixed_data are for additional arguments
        to function_to_sum.  batched_data and fixed_data are unpacked before calling function_to sum. The
         primary functionality is summed_data += function_to_sum(*batched_data, *fixed_data)

        Args:
            summed_and_fixed_data (tuple or list): The first element is an array of the shape returned by
            function_to_sum.  This shape should not depend on batched_data.  The remaining elements are
            extra arguments to be sent to function_to_sum.
            batched_data (tuple or list):  The data for use in function_to_sum.

        Returns:
            tuple of ([summed_data, *fixed_data], None)
        """
        summed_data = summed_and_fixed_data[0]
        fixed_data = summed_and_fixed_data[1:]
        output_to_add = function_to_sum(*batched_data, *fixed_data)
        summed_data = jax.tree_util.tree_map(jnp.add, *(summed_data, output_to_add))

        return [summed_data, *fixed_data], None

    # Apply the batch projector directly to an initial batch to get the initial output
    num_input_points = data_to_batch[0].shape[0]
    batch_size = num_input_points if batch_size is None else batch_size
    num_remaining = num_input_points % batch_size
    # If the input is a multiple of batch_size, then we'll do a full batch, otherwise just the excess.
    initial_batch_size = batch_size if num_remaining == 0 else num_remaining

    initial_batch = [data[:initial_batch_size] for data in data_to_batch]
    summed_output = function_to_sum(*initial_batch, *extra_args)

    # Then deal with the batches if there are any
    if batch_size < num_input_points:
        num_batches = (num_input_points - initial_batch_size) // batch_size
        output_shape = (num_batches, batch_size,)
        data_batched = [jnp.reshape(data[initial_batch_size:], output_shape + data.shape[1:])
                        for data in data_to_batch]

        # Set up a scan over the batches.
        initial_carry = [summed_output, *extra_args]
        final_carry, _ = jax.lax.scan(add_one_batch, initial_carry, data_batched)

        summed_output = final_carry[0]

    return summed_output


def ensure_tuple(var_args):
    """
    Convert a singleton to a one-element tuple if needed, and convert a list to a tuple
    Args:
        var_args: singleton or list or tuple

    Returns:
        tuple
    """
    # Check if var_args is already a tuple
    if isinstance(var_args, tuple):
        return var_args
    # Check if var_args is a list
    elif isinstance(var_args, list):
        return tuple(var_args)
    # Assume var_args is a single item if it's neither a list nor a tuple
    else:
        return (var_args, )


# ──────────────────────────────────────────────────────────────────────────────
# Module-level projector drivers (shared jit cache across model instances)
#
# These were per-instance closures inside Projectors.create_projectors.  Lifting them to
# module level -- with the geometry-specific per-view kernel, projector_params, and batch
# sizes as STATIC arguments and the view-parameter array as a TRACED argument -- lets
# jax.jit key its cache on those values, so two models with the same geometry SHARE one
# compiled program instead of each re-tracing (tracing dominates the per-model first-call
# cost).  Only the two top-level signatures (Projectors.sparse_forward_project /
# .sparse_back_project) are part of the public interface; these drivers are internal.
# ──────────────────────────────────────────────────────────────────────────────
def _sparse_forward_project(view_params_array, voxel_values, pixel_indices, n_p_centers,
                            fwd_kernel, projector_params, pixel_batch_size, view_batch_size,
                            owned_view_indices=()):
    """Forward project voxels to a sinogram, batching over pixels then views.

    Batches over pixels (sum_function_in_batches); within each pixel batch, maps the
    geometry's per-view forward kernel over batches of views (concatenate_function_in_batches)
    and sums the pixel-batch contributions.  fwd_kernel, projector_params, pixel_batch_size and
    view_batch_size are static; view_params_array / voxel_values / pixel_indices / n_p_centers
    are traced.

    n_p_centers is the CONCRETE (num_pixels, num_views) int32 channel-center array from
    _jit_compute_scatter_centers (pixels-major so the pixel scan batches it alongside
    voxel_values); each view's (pixel_batch,) row reaches the kernel as an input, so no round
    of a projection coordinate exists in this program's horizontal fans.
    """
    cur_view_params_array = view_params_array
    if len(owned_view_indices) > 0:
        # In-jit gather (traced; asarray also handles tuple-typed indices from tests).
        cur_view_params_array = view_params_array[jnp.asarray(owned_view_indices)]

    def forward_project_pixel_batch(local_values, local_pix_indices, local_n_pc):
        # local_n_pc: (pixel_batch, num_views) -> view-major for the view batching below
        # (a batch-sized int transpose, fused by XLA).
        local_n_pc_T = local_n_pc.T

        def forward_project_view_batch(view_params_batch, n_pc_batch):
            # vmap (not lax.map) over views: the per-view kernel reuses the same voxel batch,
            # so parallelizing over views trades a little memory for speed.
            return jax.vmap(fwd_kernel, in_axes=(None, None, 0, 0, None))(
                local_values, local_pix_indices, view_params_batch, n_pc_batch,
                projector_params)

        return concatenate_function_in_batches(forward_project_view_batch,
                                               (cur_view_params_array, local_n_pc_T),
                                               view_batch_size)

    return sum_function_in_batches(forward_project_pixel_batch,
                                   (voxel_values, pixel_indices, n_p_centers),
                                   pixel_batch_size)


_jit_sparse_forward_project = jax.jit(
    _sparse_forward_project,
    static_argnames=['fwd_kernel', 'projector_params', 'pixel_batch_size', 'view_batch_size'])


def _sparse_back_project(view_params_array, sinogram, pixel_indices, n_p_centers,
                         back_kernel, projector_params, pixel_batch_size, view_batch_size,
                         coeff_power=1, owned_view_indices=()):
    """Back project a sinogram to voxels, batching over views (summing) then pixels.

    Batches over views (sum_function_in_batches); within each view batch, maps the geometry's
    per-view back kernel over the views (vmap) and sums them, then batches over pixels
    (concatenate_function_in_batches).  back_kernel, projector_params, pixel_batch_size,
    view_batch_size and coeff_power are static; view_params_array / sinogram / pixel_indices /
    n_p_centers are traced.

    n_p_centers is the CONCRETE (num_views, num_pixels) int32 channel-center array from
    _jit_compute_scatter_centers (views-major so the view sum batches it alongside the
    sinogram); the SAME centers feed the forward projector, so the pair stays adjoint even at
    rounding ties.
    """
    cur_view_params_array = view_params_array
    if len(owned_view_indices) > 0:
        # In-jit gather (traced; asarray also handles tuple-typed indices from tests).
        cur_view_params_array = view_params_array[jnp.asarray(owned_view_indices)]

    def back_project_view_batch(local_view_batch, local_view_params_batch, local_n_pc,
                                local_pixel_indices):
        # local_n_pc: (view_batch, num_pixels) -> pixels-major for the pixel batching below.
        local_n_pc_T = local_n_pc.T

        def back_project_pixel_batch(pixel_indices_batch, n_pc_batch):
            # Map the per-view back kernel over the views in this batch (each view takes its
            # (pixel_batch,) column of centers -> in_axes=1), then sum over views.
            bp_vmap = jax.vmap(back_kernel, in_axes=(0, None, 0, 1, None, None))
            per_view_voxel_values_batch = bp_vmap(local_view_batch, pixel_indices_batch,
                                                  local_view_params_batch, n_pc_batch,
                                                  projector_params, coeff_power)
            # Driver-level reduce (shared across geometries, not the geometry kernel) -- give it its
            # own named_scope so profilers attribute it as a distinct region instead of leaving it
            # unscoped/unmapped.
            with jax.named_scope("projector/back/view_reduce"):
                return jnp.sum(per_view_voxel_values_batch, axis=0)

        return concatenate_function_in_batches(back_project_pixel_batch,
                                               (local_pixel_indices, local_n_pc_T),
                                               pixel_batch_size)

    return sum_function_in_batches(back_project_view_batch,
                                   (sinogram, cur_view_params_array, n_p_centers),
                                   view_batch_size, pixel_indices)


_jit_sparse_back_project = jax.jit(
    _sparse_back_project,
    static_argnames=['back_kernel', 'projector_params', 'pixel_batch_size', 'view_batch_size',
                     'coeff_power'])


def _sparse_back_project_band(view_params_array, sinogram, pixel_indices, n_p_centers,
                              g0, num_band_slices,
                              back_band_kernel, projector_params, pixel_batch_size, view_batch_size,
                              coeff_power=1, owned_view_indices=()):
    """Back project a sinogram onto the GLOBAL recon slice band [g0, g0 + num_band_slices).

    The banded analogue of _sparse_back_project: it batches over views (summing) then pixels in
    exactly the same way -- so the per-view-batch vmap stack and the per-pixel-batch transients are
    bounded by view_batch_size / pixel_batch_size, NOT by the number of views -- but calls the
    geometry's BANDED back kernel, which produces only the L = num_band_slices slices [g0, g0 + L)
    from the FULL view (no detector-row crop; for geometries whose back projection draws a slice
    from a RANGE of rows).  g0 is TRACED (a band's global start, so the bands of a slice axis do
    not retrace); num_band_slices is STATIC (it sets the output slice count); back_band_kernel /
    projector_params / batch sizes / coeff_power are static.  n_p_centers is the CONCRETE
    (num_views, num_pixels) int32 channel-center array (see _sparse_back_project).
    """
    cur_view_params_array = view_params_array
    if len(owned_view_indices) > 0:
        # In-jit gather (traced; asarray also handles tuple-typed indices from tests).
        cur_view_params_array = view_params_array[jnp.asarray(owned_view_indices)]

    def back_project_view_batch(local_view_batch, local_view_params_batch, local_n_pc,
                                local_pixel_indices):
        # local_n_pc: (view_batch, num_pixels) -> pixels-major for the pixel batching below.
        local_n_pc_T = local_n_pc.T

        def back_project_pixel_batch(pixel_indices_batch, n_pc_batch):
            # Map the per-view banded back kernel over the views in this batch, then sum over views.
            bp_vmap = jax.vmap(back_band_kernel, in_axes=(0, None, 0, 1, None, None, None, None))
            per_view_band = bp_vmap(local_view_batch, pixel_indices_batch, local_view_params_batch,
                                    n_pc_batch, projector_params, g0, num_band_slices, coeff_power)
            with jax.named_scope("projector/back/view_reduce"):   # driver reduce; see _sparse_back_project
                return jnp.sum(per_view_band, axis=0)

        return concatenate_function_in_batches(back_project_pixel_batch,
                                               (local_pixel_indices, local_n_pc_T),
                                               pixel_batch_size)

    return sum_function_in_batches(back_project_view_batch,
                                   (sinogram, cur_view_params_array, n_p_centers),
                                   view_batch_size, pixel_indices)


_jit_sparse_back_project_band = jax.jit(
    _sparse_back_project_band,
    static_argnames=['num_band_slices', 'back_band_kernel', 'projector_params',
                     'pixel_batch_size', 'view_batch_size', 'coeff_power'])
