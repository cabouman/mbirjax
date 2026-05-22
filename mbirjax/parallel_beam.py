from functools import partial
from collections import namedtuple
from typing import Literal, Union, overload, Any
import concurrent.futures
import numpy as np
import jax
import jax.numpy as jnp
import mbirjax as mj
from mbirjax import TomographyModel, tomography_utils

ParallelBeamParamNames = mj.ParamNames | Literal['angles']


@partial(jax.jit, static_argnames=('body_views', 'tail_views', 'view_batch_size'))
def _apply_fbp_filter_to_shard(shard, filter_arr, *, body_views, tail_views, view_batch_size):
    """Apply the FBP filter to a (views, local_rows, channels) shard on the current device.

    Defined at module level so JAX's JIT cache can be shared across all threads and all
    calls to fbp_filter.  The cache is keyed on:
      - this function's Python identity (stable — defined once)
      - the static args (body_views, tail_views, view_batch_size)
      - the abstract types of shard and filter_arr (shape + dtype)
    so compilation happens once per unique (size, view_batch_size) combination.

    Args:
        shard (jax array):      Shape (views, local_rows, channels), fully contiguous.
        filter_arr (jax array): Shape (2*channels - 1,), the FBP reconstruction filter.
        body_views (int):       Largest prefix of views divisible by view_batch_size.
        tail_views (int):       Remaining views (0 ≤ tail_views < view_batch_size).
        view_batch_size (int):  Batch size for lax.map over views.

    Returns:
        jax array of shape (views, local_rows, channels_filtered).
    """
    def _convolve_row(row):
        return jax.scipy.signal.fftconvolve(row, filter_arr, mode='valid')

    def _apply_conv_to_view(view):
        # view: (local_rows, channels) — contiguous; vmap over rows
        return jax.vmap(_convolve_row)(view)

    parts = []
    if body_views > 0:
        parts.append(jax.lax.map(_apply_conv_to_view, shard[:body_views],
                                 batch_size=view_batch_size))
    if tail_views > 0:
        # tail < view_batch_size — process one at a time to avoid cuFFT padding
        parts.append(jax.lax.map(_apply_conv_to_view, shard[body_views:]))
    return jnp.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]


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

        # Allocate the sinogram array. Use voxel_values.shape[1] rather than num_det_rows from
        # projector_params so that this function works correctly on a slice-sharded subset of the
        # full recon/sinogram (where the local slice count differs from the global num_det_rows).
        sinogram_view = jnp.zeros((voxel_values.shape[1], num_det_channels))

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

        # Allocate the voxel cylinder array. Use sinogram_view.shape[0] rather than num_det_rows
        # from projector_params so that this function works correctly on a slice-sharded subset
        # (where the local det-row count differs from the global num_det_rows).
        det_voxel_cylinder = jnp.zeros((num_pixels, sinogram_view.shape[0]))
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

    def fbp_filter(self, sinogram, filter_name="ramp", view_batch_size=100):
        """
        Perform FBP filtering on the given sinogram.

        Args:
            sinogram (jax array): The input sinogram with shape (num_views, num_rows, num_channels).
            filter_name (string, optional): Name of the filter to be used. Defaults to "ramp"
            view_batch_size (int, optional):  Size of view batches (used to limit memory use)

        Returns:
            filtered_sinogram (jax array): The sinogram after FBP filtering.
        """
        num_views, _, num_channels = sinogram.shape
        if view_batch_size is None:
            view_batch_size = self.view_batch_size_for_vmap
            max_view_batch_size = 128  # Limit the view batch size here and ConeBeam due to https://github.com/jax-ml/jax/issues/27591
            view_batch_size = min(view_batch_size, max_view_batch_size)

        # Generate the reconstruction filter with appropriate scaling
        delta_voxel = self.get_params('delta_voxel')
        # Scaling factor adjusts the filter to account for voxel size, ensuring consistent reconstruction.
        # For a detailed theoretical derivation of this scaling factor, please refer to the zip file linked at
        # https://mbirjax.readthedocs.io/en/latest/theory.html
        scaling_factor = 1 / (delta_voxel**2)

        # Single-device helpers (used by the else branch below).
        recon_filter = tomography_utils.generate_direct_recon_filter(num_channels, filter_name=filter_name)
        recon_filter *= scaling_factor

        def convolve_row(row):
            return jax.scipy.signal.fftconvolve(row, recon_filter, mode="valid")

        def apply_convolution_to_view(view):
            return jax.vmap(convolve_row)(view)

        if self.mesh is not None:
            # Multi-device path: Python threads, one per device, each calling the
            # module-level _apply_fbp_filter_to_shard via standard jax.jit.
            #
            # Why not shard_map or pmap:
            #   Both invoke XLA's SPMD partitioner on GPU backends, which changes how
            #   lax.map is lowered — adding extra intermediate materializations and
            #   producing a 3–6× slowdown even with a single device.  Threading bypasses
            #   SPMD: each thread uses the standard jit + XLA path and runs at
            #   single-device speed.  See .claude/parallel_options.md for the full analysis.
            #
            # Why not NamedSharding + jit:
            #   XLA's cuFFT thunk requires row-major contiguous memory
            #   (IsMonotonicWithDim0Major).  Sharding along axis 1 (rows) of a 3D array
            #   produces strided per-device views that fail this check.
            #
            # Data flow (all numpy until the per-device jnp.array upload):
            #   sinogram  (views, rows, channels)  → np.asarray → np.split along axis=1
            #   shard[i]  (views, local_rows, channels)  → jnp.array on device i
            #   filter_np (2*channels-1,) float32  → jnp.array on device i
            #   _apply_fbp_filter_to_shard → out (views, local_rows, channels_filtered)
            #   out → np.asarray (D→H, frees device memory)
            #   np.concatenate along axis=1 → jnp.array (uncommitted, default device)
            n_devices = self.mesh.devices.size
            pmap_devices = list(self.mesh.devices.flat)
            num_rows = sinogram.shape[1]
            local_rows = num_rows // n_devices

            # Split views into a body (largest prefix divisible by view_batch_size)
            # and a tail (remainder, 0 <= tail < view_batch_size).
            # lax.map with batch_size pads the last batch when num_views is not
            # divisible, and that padding contaminates cuFFT outputs (empirically
            # confirmed).  By splitting, neither piece is ever padded, so any
            # view_batch_size works — including when num_views is large or prime.
            #
            # These are computed once from num_views and reused for every shard.
            # That is correct because sharding is along axis 1 (rows): every shard
            # has shape (views, local_rows, channels) and therefore the same full
            # num_views along axis 0.  If sharding were ever changed to axis 0
            # (views), body_views/tail_views would need to be computed per-shard
            # from local_views = num_views // n_devices.
            body_views = (num_views // view_batch_size) * view_batch_size
            tail_views = num_views - body_views

            # ── Compute filter as numpy ──────────────────────────────────────
            # Reuse the float32 recon_filter computed above (in-place *= preserves
            # dtype).  A fresh np.asarray avoids the GPU scatter bug and gives each
            # thread an independent numpy source to upload to its assigned device.
            # DO NOT recompute via generate_direct_recon_filter(...) * scaling_factor:
            # that out-of-place multiply promotes float32 → float64, which is 2–32×
            # slower on GPU.
            filter_np = np.asarray(recon_filter)

            # ── Split sinogram along rows ────────────────────────────────────
            sino_np = np.asarray(sinogram)  # (views, rows, channels)
            row_splits = np.split(sino_np, n_devices, axis=1)
            # row_splits[i]: (views, local_rows, channels) — contiguous numpy shard

            # ── Dispatch one thread per device ───────────────────────────────
            # Each thread calls the module-level _apply_fbp_filter_to_shard,
            # which is a standard jax.jit function (not pmap/shard_map).
            # Using jax.default_device inside each thread assigns that thread's
            # operations to a specific device.  JAX is async: the GIL is held only
            # for the brief dispatch call, so both GPU kernels run concurrently.
            # The JIT cache is global (keyed on the module-level function object +
            # static args + abstract input types), so compilation happens once during
            # warmup and is reused by all threads on all subsequent calls.
            #
            # Why not pmap: on GPU backends, jax.pmap uses the same XLA SPMD
            # partitioner as shard_map, producing the same ~5× slowdown.  Threading
            # bypasses SPMD entirely — each thread compiles with the standard
            # (non-SPMD) XLA path and achieves single-device speed.
            results = [None] * n_devices

            def _filter_on_device(i):
                with jax.default_device(pmap_devices[i]):
                    shard_jax = jnp.array(row_splits[i])   # H→D for device i
                    filter_jax = jnp.array(filter_np)       # filter copy on device i
                    out = _apply_fbp_filter_to_shard(
                        shard_jax, filter_jax,
                        body_views=body_views,
                        tail_views=tail_views,
                        view_batch_size=view_batch_size)
                    jax.block_until_ready(out)
                results[i] = np.asarray(out)               # D→H, frees device memory

            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=n_devices) as executor:
                list(executor.map(_filter_on_device, range(n_devices)))

            # ── Gather ───────────────────────────────────────────────────────
            # Concatenate along local_rows → (views, rows, channels_filtered)
            filtered_sinogram = jnp.array(np.concatenate(results, axis=1))
        else:
            # Single-device path: place on sinogram_device and map over all views.
            sinogram = self._device_put(sinogram, self.sinogram_device)
            filtered_sinogram = jax.lax.map(apply_convolution_to_view, sinogram, batch_size=view_batch_size)
        filtered_sinogram *= jnp.pi / num_views  # scaling term
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

        # In multi-GPU mode, fbp_filter handles the device split internally via threading
        # (np.asarray → np.split → per-device jit → gather).  Pre-sharding here would
        # only add a pointless shard+gather roundtrip, so we skip it entirely.
        filtered_sinogram = self.fbp_filter(sinogram, filter_name=filter_name, view_batch_size=view_batch_size)

        # Apply backprojection.  filtered_sinogram is uncommitted after fbp_filter's gather.
        recon = self.back_project(filtered_sinogram)

        # _maybe_gather is a no-op here (recon is already uncommitted after pmap gather),
        # but kept for symmetry and to handle any future path where recon may be sharded.
        recon = self._maybe_gather(recon)

        return recon

