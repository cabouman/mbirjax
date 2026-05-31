from functools import partial
from collections import namedtuple
from typing import Literal, Union, overload, Any

import numpy as np
import jax
import jax.numpy as jnp
import mbirjax as mj
import mbirjax._sharding as mjs
from mbirjax import TomographyModel, tomography_utils

ParallelBeamParamNames = mj.ParamNames | Literal['angles']


# ── FBP-filter kernel selection (Phase F1 experiment switch) ──────────────────
# fbp_filter applies a per-detector-row convolution to each device's sinogram
# block.  Three kernel structures are under evaluation:
#   "per_view"  : lax.map over views, vmap the row convolution within each view.
#                 FFT batch = view_batch_size * n_rows → memory scales with
#                 geometry (the H100 1624^3 OOM).
#   "flat"      : reshape (views, rows, channels) -> (views*rows, channels) and
#                 lax.map(batch_size=128) the row convolution — bounded batch but
#                 via the jax#27591-unsafe path, plus a body/tail padding hack.
#   "row_batch" : reshape to (M, B, c) and lax.map (no batch_size) with vmap over
#                 B — FFT work area ~ B*fft_len(c), bounded by B alone, #27591-free
#                 (see _apply_fbp_filter_row_batch).  Intended replacement for the
#                 other two; default-to-be after the GPU B-sweep.
# This constant selects which kernel fbp_filter uses; the comparison harness
# toggles it to measure time/memory/correctness.  TEMPORARY: once row_batch is
# confirmed on GPU, keep it and remove this switch and the other two.
_FBP_FILTER_KERNEL = "per_view"


def _split_body_tail(n, batch):
    """Split count n into (body, tail) where body is the largest multiple of batch.

    jax.lax.map pads the final batch when n is not divisible by the batch size,
    and that padding contaminates the cuFFT output, so we run an exact-multiple
    body (batched) plus a remainder tail (mapped without a fixed batch size).
    batch is capped to n so a small n never degenerates to an all-tail (slow,
    unbatched) pass.
    """
    batch = min(batch, n)
    body = (n // batch) * batch
    return body, n - body, batch


@partial(jax.jit, static_argnames='view_batch_size')
def _apply_fbp_filter_per_view(block, filter_arr, *, view_batch_size):
    """per_view kernel: lax.map over views, vmap the row convolution per view.

    Defined at module level and @jax.jit'd (view_batch_size static; block/filter
    shapes are static under jit) so the whole per-device kernel compiles to one
    fused executable.  This jit is load-bearing for multi-device scaling: without
    it the lax.map / fftconvolve / reshape run eagerly (op-by-op via Python),
    which on the CPU backend barely scales (~2x at 8 devices vs ~6.5x jitted —
    measured 2026-05-30, root cause of the research-vs-beta gap).  The JIT cache
    is keyed on this function's identity (stable, module-level) + static args +
    abstract input types, so compilation happens once per (shape, view_batch_size)
    and is shared across all threads and all fbp_filter calls.

    Args:
        block (jax array):      Shape (views, rows, channels), contiguous.
        filter_arr (jax array): Shape (2*channels - 1,), the FBP recon filter.
        view_batch_size (int):  Number of views per lax.map step.

    Returns:
        jax array of shape (views, rows, channels).
    """
    n_views = block.shape[0]
    body, tail, vbs = _split_body_tail(n_views, view_batch_size)

    def _convolve_row(row):
        return jax.scipy.signal.fftconvolve(row, filter_arr, mode='valid')

    def _apply_conv_to_view(view):
        # view: (rows, channels) — contiguous; vmap the convolution over rows.
        return jax.vmap(_convolve_row)(view)

    parts = []
    if body > 0:
        parts.append(jax.lax.map(_apply_conv_to_view, block[:body], batch_size=vbs))
    if tail > 0:
        parts.append(jax.lax.map(_apply_conv_to_view, block[body:]))
    return jnp.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]


# Cap for the flat kernel's lax.map batch size.  jax-ml/jax#27591: lax.map can
# produce WRONG results for large batch sizes (their repro: batch 128 correct,
# batch 512 garbage), so keep the flattened-row batch small.  128 is the value
# used elsewhere (fbp_filter's view_batch_size cap) and is on the safe side of
# the reported failure.
_FLAT_MAP_BATCH = 128


@partial(jax.jit, static_argnames='view_batch_size')
def _apply_fbp_filter_flat(block, filter_arr, *, view_batch_size):
    """flat kernel: reshape to (views*rows, channels), lax.map the per-row conv.

    The FBP filter is per detector row, so flattening the (views, rows) axes
    exposes views*rows independent rows — full parallel work regardless of how
    few views a device holds.  @jax.jit'd for the same reason as the per_view
    kernel (eager dispatch barely scales on CPU).

    Batch-size safety (jax-ml/jax#27591): lax.map can return wrong results for
    large batch sizes, so the flattened-row batch is capped at _FLAT_MAP_BATCH
    (128), NOT view_batch_size*n_rows (which could be tens of thousands).
    view_batch_size is accepted for signature parity with the per_view kernel but
    does not set the flat batch (the cap governs).

    Args:
        block (jax array):      Shape (views, rows, channels), contiguous.
        filter_arr (jax array): Shape (2*channels - 1,), the FBP recon filter.
        view_batch_size (int):  Accepted for parity; flat batch uses _FLAT_MAP_BATCH.

    Returns:
        jax array of shape (views, rows, channels).
    """
    n_views, n_rows, n_channels = block.shape
    rows = block.reshape(n_views * n_rows, n_channels)
    total_rows = n_views * n_rows
    body, tail, rbs = _split_body_tail(total_rows, _FLAT_MAP_BATCH)

    def _convolve_row(row):
        return jax.scipy.signal.fftconvolve(row, filter_arr, mode='valid')

    parts = []
    if body > 0:
        parts.append(jax.lax.map(_convolve_row, rows[:body], batch_size=rbs))
    if tail > 0:
        parts.append(jax.lax.map(_convolve_row, rows[body:]))
    filtered = jnp.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
    return filtered.reshape(n_views, n_rows, n_channels)


# Row batch (B) for the row_batch kernel: rows convolved per lax.map step.  The
# cuFFT work area is ~ B * fft_len(c), so B bounds the per-device filter memory
# independent of geometry (v, r) and device count.  Because work area also grows
# with c (fft_len(c) >~ 3c-2), B will likely need to become c-aware (B ~ budget /
# fft_len(c)); for now it is a fixed default, to be tuned by the GPU B-sweep.
_FBP_ROW_BATCH = 256


@partial(jax.jit, static_argnames='view_batch_size')
def _apply_fbp_filter_row_batch(block, filter_arr, *, view_batch_size):
    """row_batch kernel: reshape (v,r,c)->(v*r,c), batch B rows per lax.map step.

    The FBP filter is per detector row, so we flatten (views, rows) into one row
    axis, pad it up to a multiple of B, and fold it to (M, B, c) with M = v*r/B
    map steps.  Each step convolves B rows (vmap over the B), and lax.map scans
    the M steps WITHOUT a batch_size argument — so the cuFFT work area is
    ~ B * fft_len(c), bounded by B alone, independent of how many views/rows a
    device holds or how many devices there are.

    Two deliberate choices vs the `flat` kernel:
      - No `lax.map(batch_size=...)`: the scan processes one (B, c) chunk at a
        time and vmap supplies the B-way parallelism, so jax-ml/jax#27591 (wrong
        results for large lax.map batch_size) does NOT apply and B is a free knob.
      - Our own zero-pad + crop replaces the body/tail split: each row convolves
        independently, so the zero-padded rows produce benign output that is
        cropped away (no cuFFT-padding contamination).

    @jax.jit'd (view_batch_size static; block/filter shapes static) — the jit is
    load-bearing for multi-device scaling (eager dispatch barely scales).

    Args:
        block (jax array):      Shape (views, rows, channels), contiguous.
        filter_arr (jax array): Shape (2*channels - 1,), the FBP recon filter.
        view_batch_size (int):  Accepted for parity (and soon-deprecated); the
                                row batch B (_FBP_ROW_BATCH) governs the FFT batch.

    Returns:
        jax array of shape (views, rows, channels).
    """
    n_views, n_rows, n_channels = block.shape
    total_rows = n_views * n_rows
    batch = min(_FBP_ROW_BATCH, total_rows)        # don't pad past a tiny shard
    n_steps = (total_rows + batch - 1) // batch     # ceil → M
    padded = n_steps * batch

    rows = block.reshape(total_rows, n_channels)
    if padded > total_rows:
        # Append zero rows so the row axis is an exact multiple of batch; they
        # convolve to benign values and are cropped after the map.
        pad = jnp.zeros((padded - total_rows, n_channels), dtype=rows.dtype)
        rows = jnp.concatenate([rows, pad], axis=0)
    row_batches = rows.reshape(n_steps, batch, n_channels)

    def _convolve_row(row):
        return jax.scipy.signal.fftconvolve(row, filter_arr, mode='valid')

    def _convolve_row_batch(row_batch):    # row_batch: (batch, channels)
        return jax.vmap(_convolve_row)(row_batch)

    filtered = jax.lax.map(_convolve_row_batch, row_batches)   # (n_steps, batch, c_out)
    c_out = filtered.shape[-1]                         # == n_channels for 'valid'
    filtered = filtered.reshape(n_steps * batch, c_out)[:total_rows]
    return filtered.reshape(n_views, n_rows, c_out)


def _apply_fbp_filter(block, filter_arr, *, view_batch_size):
    """Dispatch to the FBP-filter kernel selected by _FBP_FILTER_KERNEL."""
    if _FBP_FILTER_KERNEL == "row_batch":
        return _apply_fbp_filter_row_batch(block, filter_arr, view_batch_size=view_batch_size)
    if _FBP_FILTER_KERNEL == "flat":
        return _apply_fbp_filter_flat(block, filter_arr, view_batch_size=view_batch_size)
    return _apply_fbp_filter_per_view(block, filter_arr, view_batch_size=view_batch_size)


class ParallelBeamModel(TomographyModel):
    """
    A class designed for handling forward and backward projections in a parallel beam geometry, extending the
    :ref:`TomographyModelDocs`. This class offers specialized methods and parameters tailored for parallel beam setups.

    This class inherits all methods and properties from the :ref:`TomographyModelDocs` and may override some
    to suit parallel beam geometrical requirements. See the documentation of the parent class for standard methods
    like setting parameters and performing projections and reconstructions.

    Parameters not included in the constructor can be set using the set_params method of :ref:`TomographyModelDocs`.
    Refer to :ref:`TomographyModelDocs` documentation for a detailed list of possible parameters.

    Args:
        sinogram_shape (tuple):
            Shape of the sinogram as a tuple in the form `(views, rows, channels)`, where 'views' is the number of
            different projection angles, 'rows' correspond to the number of detector rows, and 'channels' index columns of
            the detector that are assumed to be aligned with the rotation axis.
        angles (jnp.ndarray):
            A 1D array of projection angles, in radians, specifying the angle of each projection relative to the origin.

    Examples
    --------
    Initialize a parallel beam model with specific angles and sinogram shape:

    >>> import mbirjax
    >>> angles = jnp.array([0, jnp.pi/4, jnp.pi/2])
    >>> model = mj.ParallelBeamModel((180, 256, 10), angles)

    See Also
    --------
    TomographyModel : The base class from which this class inherits.
    """

    DIRECT_RECON_VIEW_BATCH_SIZE = TomographyModel.DIRECT_RECON_VIEW_BATCH_SIZE

    def __init__(self, sinogram_shape, angles):

        angles = jnp.asarray(angles)
        view_params_name = 'angles'
        super().__init__(sinogram_shape, angles=angles, view_params_name=view_params_name)

    @overload
    def get_params(self, parameter_names: Union[ParallelBeamParamNames, list[ParallelBeamParamNames]]) -> Any: ...

    def get_params(self, parameter_names) -> Any:
        return super().get_params(parameter_names)

    def get_magnification(self):
        """
        Compute the scale factor from a voxel at iso (at the origin on the center of rotation) to
        its projection on the detector.  For parallel beam, this is 1, but it may be parameter-dependent
        for other geometries.

        Returns:
            (float): magnification
        """
        magnification = 1.0
        return magnification

    def verify_valid_params(self):
        """
        Check that all parameters are compatible for a reconstruction.

        Note:
            Raises ValueError for invalid parameters.
        """
        super().verify_valid_params()
        sinogram_shape, angles = self.get_params(['sinogram_shape', 'angles'])

        if angles.shape[0] != sinogram_shape[0]:
            error_message = "Number view dependent parameter vectors must equal the number of views. \n"
            error_message += "Got {} for length of view-dependent parameters and "
            error_message += "{} for number of views.".format(angles.shape[0], sinogram_shape[0])
            raise ValueError(error_message)

        recon_shape = self.get_params('recon_shape')
        if recon_shape[2] != sinogram_shape[1]:
            error_message = "Number of recon slices must match number of sinogram rows. \n"
            error_message += "Got {} for recon_shape and {} for sinogram_shape".format(recon_shape, sinogram_shape)
            raise ValueError(error_message)

    def get_geometry_parameters(self):
        """
        Function to get a list of the primary geometry parameters for for parallel beam projection.

        Returns:
            namedtuple of required geometry parameters.
        """
        # First get the parameters managed by ParameterHandler
        geometry_param_names = ['delta_det_channel', 'det_channel_offset', 'delta_voxel']
        geometry_param_values = self.get_params(geometry_param_names)

        # Then get additional parameters:
        geometry_param_names += ['psf_radius']
        geometry_param_values.append(self.get_psf_radius())

        # Then create a namedtuple to access parameters by name in a way that can be jit-compiled.
        GeometryParams = namedtuple('GeometryParams', geometry_param_names)
        geometry_params = GeometryParams(*tuple(geometry_param_values))

        return geometry_params

    def get_psf_radius(self):
        """Computes the integer radius of the PSF kernel for parallel beam projection.
        """
        delta_det_channel, delta_voxel = self.get_params(['delta_det_channel', 'delta_voxel'])

        # Compute the maximum number of detector rows/channels on either side of the center detector hit by a voxel
        psf_radius = int(jnp.ceil(jnp.ceil(delta_voxel/delta_det_channel)/2))

        return psf_radius

    def auto_set_recon_geometry(self, no_compile=False, no_warning=False):
        """Compute the default recon size using the internal parameters delta_channel and delta_pixel plus
          the number of channels from the sinogram"""
        delta_det_row, delta_det_channel = self.get_params(['delta_det_row', 'delta_det_channel'])

        # Compute delta_voxel
        delta_voxel = self.get_params('delta_det_channel') / self.get_magnification()

        # Compute the recon_shape
        sinogram_shape = self.get_params('sinogram_shape')
        num_det_rows, num_det_channels = sinogram_shape[1:3]
        magnification = self.get_magnification()
        num_recon_rows = int(jnp.ceil(num_det_channels * delta_det_channel / (delta_voxel * magnification)))
        num_recon_cols = num_recon_rows
        num_recon_slices = int(jnp.round(num_det_rows * ((delta_det_row / delta_voxel) / magnification)))
        recon_shape = (num_recon_rows, num_recon_cols, num_recon_slices)

        self.set_params(no_compile=no_compile, no_warning=no_warning, recon_shape=recon_shape, delta_voxel=delta_voxel)

    @staticmethod
    @partial(jax.jit, static_argnames='projector_params')
    def forward_project_pixel_batch_to_one_view(voxel_values, pixel_indices, angle, projector_params):
        """
        Apply a parallel beam transformation to a set of voxel cylinders. These cylinders are assumed to have
        slices aligned with detector rows, so that a parallel beam maps a cylinder slice to a detector row.
        This function returns the resulting sinogram view.

        Args:
            voxel_values (jax array):  2D array of shape (num_pixels, num_recon_slices) of voxel values, where
                voxel_values[i, j] is the value of the voxel in slice j at the location determined by indices[i].
            pixel_indices (jax array of int):  1D vector of shape (len(pixel_indices), ) holding the indices into
                the flattened array of size num_rows x num_cols.
            angle (float):  Angle for this view
            projector_params (namedtuple):  tuple of (sinogram_shape, recon_shape, get_geometry_params())

        Returns:
            jax array of shape (num_det_rows, num_det_channels)
        """
        # Get all the geometry parameters - we use gp since geometry parameters is a named tuple and we'll access
        # elements using, for example, gp.delta_det_channel, so a longer name would be clumsy.
        gp = projector_params.geometry_params
        num_views, num_det_rows, num_det_channels = projector_params.sinogram_shape

        # Get the data needed for horizontal projection
        n_p, n_p_center, W_p_c, cos_alpha_p_xy = ParallelBeamModel.compute_proj_data(pixel_indices, angle, projector_params)
        L_max = jnp.minimum(1.0, W_p_c)

        # Allocate the sinogram array
        sinogram_view = jnp.zeros((num_det_rows, num_det_channels))

        # Do the projection
        for n_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius+1):
            n = n_p_center + n_offset
            abs_delta_p_c_n = jnp.abs(n_p - n)
            L_p_c_n = jnp.clip((W_p_c + 1.0) / 2.0 - abs_delta_p_c_n, 0.0, L_max)
            A_chan_n = gp.delta_voxel * L_p_c_n / cos_alpha_p_xy
            A_chan_n *= (n >= 0) * (n < num_det_channels)
            sinogram_view= sinogram_view.at[:, n].add(A_chan_n.reshape((1, -1)) * voxel_values.T)

        return sinogram_view

    @staticmethod
    @partial(jax.jit, static_argnames='projector_params')
    def back_project_one_view_to_pixel_batch(sinogram_view, pixel_indices, angle, projector_params, coeff_power=1):
        """
        Apply parallel back projection to a single sinogram view and return the resulting voxel cylinders.

        Args:
            sinogram_view (2D jax array): one view of the sinogram to be back projected.
                2D jax array of shape (num_det_rows)x(num_det_channels)
            pixel_indices (1D jax array of int):  indices into flattened array of size num_rows x num_cols.
            angle (float): The projection angle in radians for this view.
            projector_params (namedtuple): tuple of (sinogram_shape, recon_shape, get_geometry_params()).
            coeff_power (int): backproject using the coefficients of (A_ij ** coeff_power).
                Normally 1, but should be 2 when computing Hessian diagonal.
        Returns:
            jax array of shape (len(pixel_indices), num_det_rows)
        """
        # Get all the geometry parameters - we use gp since geometry parameters is a named tuple and we'll access
        # elements using, for example, gp.delta_det_channel, so a longer name would be clumsy.
        gp = projector_params.geometry_params
        num_views, num_det_rows, num_det_channels = projector_params.sinogram_shape

        num_pixels = pixel_indices.shape[0]

        # Get the data needed for horizontal projection
        n_p, n_p_center, W_p_c, cos_alpha_p_xy = ParallelBeamModel.compute_proj_data(pixel_indices, angle, projector_params)
        L_max = jnp.minimum(1.0, W_p_c)

        # Allocate the voxel cylinder array
        det_voxel_cylinder = jnp.zeros((num_pixels, num_det_rows))
        # jax.debug.breakpoint(num_frames=1)
        # Do the horizontal projection
        for n_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius + 1):
            n = n_p_center + n_offset
            abs_delta_p_c_n = jnp.abs(n_p - n)
            L_p_c_n = jnp.clip((W_p_c + 1.0) / 2.0 - abs_delta_p_c_n, 0.0, L_max)
            A_chan_n = gp.delta_voxel * L_p_c_n / cos_alpha_p_xy
            A_chan_n *= (n >= 0) * (n < num_det_channels)
            A_chan_n = A_chan_n ** coeff_power
            det_voxel_cylinder = jnp.add(det_voxel_cylinder, A_chan_n.reshape((-1, 1)) * sinogram_view[:, n].T)

        return det_voxel_cylinder

    @staticmethod
    def compute_proj_data(pixel_indices, angle, projector_params):
        """
        Compute the quantities n_p, n_p_center, W_p_c, cos_alpha_p_xy needed for vertical projection.

        Args:
            pixel_indices (1D jax array of int):  indices into flattened array of size num_rows x num_cols.
            angle (float): The projection angle in radians for this view.
            projector_params (namedtuple): tuple of (sinogram_shape, recon_shape, get_geometry_params()).

        Returns:
            n_p, n_p_center, W_p_c, cos_alpha_p_xy
        """
        # Get all the geometry parameters - we use gp since geometry parameters is a named tuple and we'll access
        # elements using, for example, gp.delta_det_channel, so a longer name would be clumsy.
        gp = projector_params.geometry_params

        num_views, num_det_rows, num_det_channels = projector_params.sinogram_shape
        recon_shape = projector_params.recon_shape

        # Convert the index into (i,j,k) coordinates corresponding to the indices into the 3D voxel array
        row_index, col_index = jnp.unravel_index(pixel_indices, recon_shape[:2])

        x_p = ParallelBeamModel.recon_ij_to_x(row_index, col_index, gp.delta_voxel, recon_shape, angle)

        det_center_channel = (num_det_channels - 1) / 2.0  # num_of_cols

        # Calculate indices on the detector grid
        n_p = (x_p + gp.det_channel_offset) / gp.delta_det_channel + det_center_channel
        n_p_center = jnp.round(n_p).astype(int)

        # Compute cos alpha for row and columns
        cos_alpha_p_xy = jnp.maximum(jnp.abs(jnp.cos(angle)),
                                     jnp.abs(jnp.sin(angle)))

        # Compute projected voxel width along columns and rows (in fraction of detector size)
        W_p_c = (gp.delta_voxel / gp.delta_det_channel) * cos_alpha_p_xy

        proj_data = (n_p, n_p_center, W_p_c, cos_alpha_p_xy)

        return proj_data

    @staticmethod
    def recon_ij_to_x(i, j, delta_voxel, recon_shape, angle):
        """
        Convert (i, j, k) indices into the recon volume to corresponding (x, y, z) coordinates.
        """
        num_recon_rows, num_recon_cols, num_recon_slices = recon_shape

        # Compute the un-rotated coordinates relative to iso
        # Note the change in order from (i, j) to (y, x)!!
        y_tilde = delta_voxel * (i - (num_recon_rows - 1) / 2.0)
        x_tilde = delta_voxel * (j - (num_recon_cols - 1) / 2.0)

        # Precompute cosine and sine of view angle, then do the rotation
        cosine = jnp.cos(angle)  # length = num_views
        sine = jnp.sin(angle)  # length = num_views

        x = cosine * x_tilde - sine * y_tilde
        y = sine * x_tilde + cosine * y_tilde

        return x

    def direct_recon(self, sinogram, filter_name="ramp", view_batch_size=DIRECT_RECON_VIEW_BATCH_SIZE):
        return self.fbp_recon(sinogram, filter_name=filter_name, view_batch_size=view_batch_size)

    def direct_filter(self, sinogram, filter_name="ramp", view_batch_size=DIRECT_RECON_VIEW_BATCH_SIZE):
        """
        Perform filtering on the given sinogram as needed for an FBP/FDK or other direct recon.

        Args:
            sinogram (jax array): The input sinogram with shape (num_views, num_rows, num_channels).
            filter_name (string, optional): Name of the filter to be used. Defaults to "ramp"
            view_batch_size (int, optional):  Size of view batches (used to limit memory use)

        Returns:
            filtered_sinogram (jax array): The sinogram after FBP filtering.
        """
        return self.fbp_filter(sinogram, filter_name=filter_name, view_batch_size=view_batch_size)

    def fbp_filter(self, sinogram, filter_name="ramp", view_batch_size=None):
        """
        Perform FBP filtering on the given sinogram.

        This is an internal sharded-contract method: it shards the sinogram on
        the view axis at entry (a no-op when no mesh is configured or when the
        input is already correctly sharded) and returns the filtered sinogram in
        that same view-native sharding.  It does NOT gather at exit, so pipelined
        callers (e.g. ``fbp_recon``/``direct_recon`` followed by back projection)
        keep the data on-device with zero host transfer.  Under the view-sharding
        scheme the ramp filter is per-view, so each device filters its own views
        with no cross-device communication.

        Args:
            sinogram (jax array): The input sinogram with shape (num_views, num_rows, num_channels).
            filter_name (string, optional): Name of the filter to be used. Defaults to "ramp"
            view_batch_size (int, optional):  Size of view batches (used to limit memory use)

        Returns:
            filtered_sinogram (jax array): The sinogram after FBP filtering, in
            view-native sharding (or a plain array when no mesh is configured).
        """
        # Shard on the view axis.  No-op when no mesh is configured (single
        # device) or when the input already carries this sharding.
        sinogram = self._shard_sinogram(sinogram)

        num_views, _, num_channels = sinogram.shape
        if view_batch_size is None:
            view_batch_size = self.view_batch_size_for_vmap
            max_view_batch_size = 128  # Limit the view batch size here and ConeBeam due to https://github.com/jax-ml/jax/issues/27591
            view_batch_size = min(view_batch_size, max_view_batch_size)

        # Generate the reconstruction filter with appropriate scaling.
        delta_voxel = self.get_params('delta_voxel')
        # Scaling factor adjusts the filter to account for voxel size, ensuring consistent reconstruction.
        # For a detailed theoretical derivation of this scaling factor, please refer to the zip file linked at
        # https://mbirjax.readthedocs.io/en/latest/theory.html
        scaling_factor = 1 / (delta_voxel ** 2)
        recon_filter = tomography_utils.generate_direct_recon_filter(num_channels, filter_name=filter_name)
        # Fold BOTH scalars — the voxel-size factor and the FBP weight pi/num_views
        # — into the filter, in place, so they cost nothing and each per-row
        # convolution output is already fully scaled.  Convolution is linear in
        # the filter, so scaling the filter scales every output row identically.
        # This replaces a post-kernel `filtered_sinogram * (pi/num_views)`, which
        # was an out-of-place, full-array multiply that promoted f32 -> f64 (np.pi
        # is float64), ~doubling peak memory and causing the 1-device GPU OOMs at
        # large sizes.  The in-place *= keeps the filter float32 (a tiny array).
        recon_filter *= scaling_factor * (np.pi / num_views)
        # Materialize the filter once as numpy; each device uploads its own copy.
        filter_np = np.asarray(recon_filter)

        # Each kernel computes its own (per-shard) body/tail batching internally
        # via _split_body_tail, so the split is always against the LOCAL
        # (per-device) view count — each device's shard holds only
        # num_views / n_devices views.  See _apply_fbp_filter and the kernels.

        if self.mesh is not None:
            # Multi-device: one thread per device, each filtering its own view
            # shard locally (no cross-device data movement).

            # Map each device to the local sinogram shard already resident on it.
            dev_to_shard = {s.device: s.data for s in sinogram.addressable_shards}

            def worker_process(i, device):
                # Per-device work: filter this device's contiguous block of views.
                # Runs inside run_per_device under jax.default_device(device), so
                # the small filter upload and the FFT all happen on `device` and
                # the result stays there (zero host transfer).
                shard = dev_to_shard[device]
                filter_jax = jnp.array(filter_np)   # tiny upload: 2*channels-1 floats
                return _apply_fbp_filter(
                    shard, filter_jax, view_batch_size=view_batch_size)

            # run_per_device fans worker_process out across the mesh devices (one
            # thread each) and returns the per-device results in device order,
            # each still resident on its own device.
            results = mjs.run_per_device(self.shard_devices, worker_process)

            # assemble_sharded stitches the per-device results back into one
            # logically-global NamedSharding array with no data movement (the
            # filtered shard for view-block i is already on device i).
            filtered_sinogram = mjs.assemble_sharded(
                results, sinogram.shape, sinogram.sharding)
        else:
            # Single-device path (no mesh configured): filter directly.
            filter_jax = jnp.array(filter_np)
            filtered_sinogram = _apply_fbp_filter(
                sinogram, filter_jax, view_batch_size=view_batch_size)

        # No post-scaling: the voxel factor and pi/num_views were folded into the
        # filter above, so each device's kernel output is already fully scaled.
        return filtered_sinogram

    def fbp_recon(self, sinogram, filter_name="ramp", view_batch_size=DIRECT_RECON_VIEW_BATCH_SIZE):
        """
        Perform filtered back-projection (FBP) reconstruction on the given sinogram.

        Our implementation uses standard filtering of the sinogram, then uses the adjoint of the forward projector to
        perform the backprojection.  This is different from many implementations, in which the backprojection is not
        exactly the adjoint of the forward projection.  For a detailed theoretical derivation of this implementation,
        see the zip file linked at this page: https://mbirjax.readthedocs.io/en/latest/theory.html

        Args:
            sinogram (jax array): The input sinogram with shape (num_views, num_rows, num_channels).
            filter_name (string, optional): Name of the filter to be used. Defaults to "ramp"
            view_batch_size (int, optional):  Size of view batches (used to limit memory use)

        Returns:
            recon (jax array): The reconstructed volume after FBP reconstruction.
        """

        filtered_sinogram = self.fbp_filter(sinogram, filter_name=filter_name, view_batch_size=view_batch_size)

        # Apply backprojection
        recon = self.back_project(filtered_sinogram)

        return recon

