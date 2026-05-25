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

    # ------------------------------------------------------------------
    # Geometry-specific sharding hooks (parallel beam)
    # ------------------------------------------------------------------
    # In parallel beam, sinogram det-rows and recon slices share the same
    # physical axis (row i of the detector corresponds to slice i of the
    # recon), so both arrays are sharded along the same logical dimension.
    #   sinogram:   (views, det_rows, channels)  →  shard axis 1 (det_rows)
    #   flat_recon: (rows*cols, slices)           →  shard axis 1 (slices)
    #   3-D recon:  (rows, cols, slices)          →  shard axis 2 (slices)
    #
    # For a future cone-beam subclass, override these methods with the
    # geometry-appropriate axes and halo strategy; the VCD loop in the
    # base class calls only these hooks and requires no changes.

    def _shard_sinogram(self, sino):
        """Shard sinogram along det_rows (axis 1) — the parallel-beam slice axis."""
        return self._maybe_shard(sino, axis=1)

    def _gather_sinogram(self, sino):
        """Gather a det_row-sharded sinogram to an uncommitted single-device array."""
        return self._maybe_gather(sino)

    def _shard_recon(self, recon):
        """Shard recon along slices (axis 1 since this comes in as a 2D array (num_pixels, num_slices)."""
        return self._maybe_shard(recon, axis=1)

    def _gather_recon(self, recon):
        """Gather a slice-sharded recon to an uncommitted single-device array."""
        return self._maybe_gather(recon)

    def _extract_halos(self, flat_recon):
        """Return per-device boundary slices for the qggmrf inter-slice prior.

        flat_recon has shape (rows*cols, local_slices) on each device, sharded
        along axis 1.  The halo for device i is one slice of shape (rows*cols,):
          left_halo[i]  = last slice of device i-1  (or None for device 0)
          right_halo[i] = first slice of device i+1 (or None for last device)

        Cost: reads 2*(n_devices-1) slices from device memory to host numpy —
        approximately 2*(rows²*4) bytes total, negligible vs compute.
        """
        if self.mesh is None:
            return [None], [None]
        # Sort shards by their slice-axis start index so ordering is deterministic.
        shards = sorted(flat_recon.addressable_shards,
                        key=lambda s: s.index[1].start)
        left_halos  = [None] + [np.asarray(s.data[:, -1]) for s in shards[:-1]]
        right_halos = [np.asarray(s.data[:, 0])  for s in shards[1:]] + [None]
        return left_halos, right_halos

    def _prepare_voxels_for_projection(self, voxel_values):
        """Shard flat voxel_values along the slice axis when a mesh is configured.

        Called by TomographyModel.forward_project immediately after
        get_voxels_at_indices.  Sharding here ensures that the subsequent
        sparse_forward_project call receives a NamedSharding array and takes
        the multi-device threading path.

        When self.mesh is None, returns voxel_values unchanged (no-op).
        """
        if self.mesh is not None:
            return self._shard_recon(voxel_values)   # axis=2 for 2-D flat recon
        return voxel_values

    def _prepare_sinogram_for_backprojection(self, sinogram):
        """Shard sinogram along det_rows (axis 1) when a mesh is configured.

        Called by TomographyModel.back_project immediately before
        sparse_back_project.  Sharding here ensures that the subsequent
        sparse_back_project call receives a NamedSharding array and takes
        the multi-device threading path.

        When self.mesh is None, returns sinogram unchanged (no-op).
        """
        if self.mesh is not None:
            return self._shard_sinogram(sinogram)   # axis=1 for det_rows
        return sinogram

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

        if self.mesh is not None:
            n_devices = self.mesh.devices.size
            if sinogram_shape[1] % n_devices != 0:
                raise ValueError(
                    f"sinogram_shape[1] ({sinogram_shape[1]}) must be divisible by "
                    f"the number of sharding devices ({n_devices})."
                )

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

    def sparse_forward_project(self, voxel_values, pixel_indices, view_indices=None, output_device=None):
        """Multi-device forward projection via slice-parallel threading.

        When self.mesh is None, or when voxel_values is not a NamedSharding
        array (plain input), falls through to the base-class single-device path.

        When voxel_values IS a NamedSharding array:
          - Each per-device shard is already resident on its device (zero PCIe
            when called from the VCD hot path or via forward_project which calls
            _prepare_voxels_for_projection first).
          - One Python thread per device runs the full view/pixel batching
            pipeline on the local slice chunk using _device_context.
          - Returns a NamedSharding sinogram sharded along det_rows (axis 1).

        Contract: output sharding matches input sharding (sharded → sharded,
        plain → plain).  The parallel-beam geometry assumption (det_rows ==
        num_slices) is validated in verify_valid_params.

        Note: this method adds no entry/exit sharding; the caller (forward_project
        via _prepare_voxels_for_projection, or VCD directly) is responsible for
        providing an appropriately sharded voxel_values.
        """
        if self.mesh is None or not isinstance(
                getattr(voxel_values, 'sharding', None), jax.sharding.NamedSharding):
            return super().sparse_forward_project(
                voxel_values, pixel_indices, view_indices, output_device)

        view_indices, view_indices_batched, pixel_batch_boundaries, sinogram_shape = \
            self._compute_projection_batch_params(pixel_indices, view_indices)

        n_devices = self.mesh.devices.size
        pmap_devices = list(self.mesh.devices.flat)
        # Each device holds local_slices columns of the flat recon; assertion is
        # checked at configure_sharding time by verify_valid_params.
        local_slices = voxel_values.shape[1] // n_devices

        # Build device → local shard mapping (data already resident, zero PCIe).
        dev_to_chunk = {s.device: s.data for s in voxel_values.addressable_shards}

        # pixel_indices is tiny and identical on every device.
        pixel_np = np.asarray(pixel_indices)

        results = [None] * n_devices

        def _project_slice_chunk(i):
            dev = pmap_devices[i]
            with self._device_context(dev):
                voxel_chunk = dev_to_chunk[dev]        # already on device i — zero PCIe
                pix = jnp.array(pixel_np)              # tiny upload to device i
                sino_parts = []
                for vib in view_indices_batched:
                    sino_views = jnp.zeros((len(vib), local_slices, sinogram_shape[2]))
                    for k in range(len(pixel_batch_boundaries) - 1):
                        ps, pe = int(pixel_batch_boundaries[k]), int(pixel_batch_boundaries[k + 1])
                        sino_views = sino_views.block_until_ready()
                        sino_views = sino_views + self.projector_functions.sparse_forward_project(
                            voxel_chunk[ps:pe], pix[ps:pe], view_indices=vib)
                    sino_parts.append(sino_views)
                chunk_sino = jnp.concatenate(sino_parts)
                jax.block_until_ready(chunk_sino)
            results[i] = chunk_sino

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_devices) as executor:
            list(executor.map(_project_slice_chunk, range(n_devices)))

        # n_views_out may be < sinogram_shape[0] when called from VCD with a view subset.
        n_views_out = int(view_indices.shape[0])
        output_sino_shape = (n_views_out, sinogram_shape[1], sinogram_shape[2])
        sino_sharding = jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec(None, 'slices', None))
        return jax.make_array_from_single_device_arrays(
            shape=output_sino_shape, sharding=sino_sharding, arrays=results)

    def sparse_back_project(self, sinogram, pixel_indices, view_indices=None, coeff_power=1, output_device=None):
        """Multi-device back projection via slice-parallel threading.

        When self.mesh is None, or when sinogram is not a NamedSharding array
        (plain input), falls through to the base-class single-device path.

        When sinogram IS a NamedSharding array:
          - Each per-device shard (views, local_det_rows, channels) is already
            resident on its device (zero PCIe when called from back_project via
            _prepare_sinogram_for_backprojection, or directly from the VCD path).
          - One Python thread per device runs the full view/pixel batching
            pipeline on the local sinogram chunk using _device_context.
          - Returns a NamedSharding flat_recon sharded along slices (axis 1).

        Contract: output sharding matches input sharding (sharded → sharded,
        plain → plain).  The parallel-beam geometry assumption (det_rows ==
        num_slices) is validated in verify_valid_params.

        Note: this method adds no entry/exit sharding; the caller (back_project
        via _prepare_sinogram_for_backprojection, or VCD directly) is responsible
        for providing an appropriately sharded sinogram.
        """
        if self.mesh is None or not isinstance(
                getattr(sinogram, 'sharding', None), jax.sharding.NamedSharding):
            return super().sparse_back_project(
                sinogram, pixel_indices, view_indices, coeff_power, output_device)

        sinogram_shape = self.get_params('sinogram_shape')
        n_devices = self.mesh.devices.size
        pmap_devices = list(self.mesh.devices.flat)
        # Each device holds local_slices det-rows of the sinogram, corresponding to
        # local_slices recon slices (det_rows == slices for parallel beam).
        local_slices = sinogram_shape[1] // n_devices

        num_views = sinogram.shape[0]
        if view_indices is None:
            view_indices = jnp.arange(num_views)

        transfer_view_batch_size = self.view_batch_size_for_vmap
        num_view_batches = int(jnp.ceil(num_views / transfer_view_batch_size))
        view_indices_batched = jnp.array_split(view_indices, num_view_batches)

        num_pixels = len(pixel_indices)
        pixel_batch_boundaries = np.arange(start=0, stop=num_pixels, step=self.transfer_pixel_batch_size)
        pixel_batch_boundaries = np.append(pixel_batch_boundaries, num_pixels)

        # Build device → local shard mapping (data already resident, zero PCIe).
        dev_to_sino_shard = {s.device: s.data for s in sinogram.addressable_shards}

        # pixel_indices is tiny and identical on every device.
        pixel_np = np.asarray(pixel_indices)

        results = [None] * n_devices

        def _back_project_shard(i):
            dev = pmap_devices[i]
            with self._device_context(dev):
                sino_shard = dev_to_sino_shard[dev]    # (views, local_slices, channels)
                pix = jnp.array(pixel_np)              # tiny upload to device i
                recon_chunk = jnp.zeros((num_pixels, local_slices))
                for vib in view_indices_batched:
                    view_batch = sino_shard[vib]       # (batch_views, local_slices, channels)
                    voxel_batch_list = []
                    for k in range(len(pixel_batch_boundaries) - 1):
                        ps = int(pixel_batch_boundaries[k])
                        pe = int(pixel_batch_boundaries[k + 1])
                        voxel_batch = self.projector_functions.sparse_back_project(
                            view_batch, pix[ps:pe], view_indices=vib, coeff_power=coeff_power)
                        voxel_batch = voxel_batch.block_until_ready()
                        voxel_batch_list.append(voxel_batch)
                    recon_chunk = recon_chunk + jnp.concatenate(voxel_batch_list, axis=0)
                jax.block_until_ready(recon_chunk)
            results[i] = recon_chunk

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_devices) as executor:
            list(executor.map(_back_project_shard, range(n_devices)))

        # Assemble per-device recon chunks into a NamedSharding flat_recon.
        output_shape = (num_pixels, sinogram_shape[1])
        recon_sharding = jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec(None, 'slices'))
        return jax.make_array_from_single_device_arrays(
            shape=output_shape, sharding=recon_sharding, arrays=results)

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
        # Record whether the caller provided an already-sharded sinogram so we can
        # return a sharded result (zero PCIe for pipelined callers) or gather it
        # (backward-compatible plain array for user-facing calls).
        input_is_sharded = isinstance(getattr(sinogram, 'sharding', None),
                                      jax.sharding.NamedSharding)
        # Ensure sinogram is in sinogram-native sharding.  No-op if already correct;
        # scatters via numpy roundtrip (JAX bug workaround) if input is uncommitted.
        sinogram = self._shard_sinogram(sinogram)

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

        recon_filter = tomography_utils.generate_direct_recon_filter(num_channels, filter_name=filter_name)
        recon_filter *= scaling_factor

        # Split views into body (largest multiple of view_batch_size) + tail.
        # lax.map pads the last batch when num_views is not divisible, which
        # contaminates cuFFT outputs.  Splitting avoids any padding.
        body_views = (num_views // view_batch_size) * view_batch_size
        tail_views = num_views - body_views

        # Materialise filter as numpy once; each code path uploads its own copy
        # to the relevant device(s).  The in-place *= above preserves float32 —
        # recomputing with an out-of-place multiply promotes to float64,
        # which is 2–32× slower on GPU.
        filter_np = np.asarray(recon_filter)

        if self.mesh is not None:
            # ── Multi-device threading: zero PCIe ───────────────────────────
            #
            # Why threading over shard_map/pmap:
            #   Both invoke XLA's SPMD partitioner on GPU backends, adding extra
            #   intermediate materializations and a 3–6× slowdown.  Threading uses
            #   the standard jit + XLA path at full single-device speed.
            #
            # Why not NamedSharding + jit:
            #   XLA's cuFFT thunk requires row-major contiguous memory.  Sharding
            #   along axis 1 (rows) produces strided views that fail this check.
            #
            # Data flow (multi-device — sinogram stays on devices throughout):
            #   sinogram already sharded (det_rows axis) by _shard_sinogram above
            #   addressable_shards → shard.data already on device i  (zero PCIe)
            #   filter_np (2*channels-1,) float32  → jnp.array on device i (tiny)
            #   _apply_fbp_filter_to_shard → out  (still on device i)
            #   results[i] = out              stays on device — zero PCIe
            #   make_array_from_single_device_arrays → NamedSharding result
            n_devices = self.mesh.devices.size
            pmap_devices = list(self.mesh.devices.flat)

            # Map device → its local shard (already resident on that device).
            dev_to_shard = {s.device: s.data for s in sinogram.addressable_shards}

            results = [None] * n_devices

            def _filter_on_device(i):
                dev = pmap_devices[i]
                with jax.default_device(dev):
                    filter_jax = jnp.array(filter_np)   # small upload: 2*ch-1 floats
                    out = _apply_fbp_filter_to_shard(
                        dev_to_shard[dev], filter_jax,
                        body_views=body_views,
                        tail_views=tail_views,
                        view_batch_size=view_batch_size)
                    jax.block_until_ready(out)
                results[i] = out                         # stays on device — zero PCIe

            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=n_devices) as executor:
                list(executor.map(_filter_on_device, range(n_devices)))

            # Assemble per-device arrays into a NamedSharding array without any
            # device-to-host transfer.  Output shape == sinogram.shape because
            # fftconvolve(mode='valid') with a (2c-1)-length filter returns c channels.
            filtered_sinogram = jax.make_array_from_single_device_arrays(
                shape=sinogram.shape,
                sharding=sinogram.sharding,
                arrays=results)
        else:
            # ── Single-device path ───────────────────────────────────────────
            # Use the module-level @jax.jit _apply_fbp_filter_to_shard rather
            # than a local closure + lax.map.  A local closure is a new Python
            # object on every fbp_filter call, so JAX re-traces the computation
            # every time — O(sinogram_size) tracing overhead on top of compute.
            # _apply_fbp_filter_to_shard is stable (defined once at module level)
            # so its compiled form is cached by Python function identity and
            # subsequent calls skip re-tracing entirely.
            sinogram = self._device_put(sinogram, self.sinogram_device)
            with self._device_context(self.sinogram_device):
                filter_jax = jnp.array(filter_np)
                filtered_sinogram = _apply_fbp_filter_to_shard(
                    sinogram, filter_jax,
                    body_views=body_views,
                    tail_views=tail_views,
                    view_batch_size=view_batch_size)

        # Apply π/num_views scaling.  Works for both sharded and plain arrays.
        filtered_sinogram = filtered_sinogram * (jnp.pi / num_views)

        # Return sharded if the caller provided a sharded input (pipelined path);
        # otherwise gather to a plain uncommitted array (backward-compatible).
        return filtered_sinogram if input_is_sharded else self._gather_sinogram(filtered_sinogram)

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

        # fbp_filter handles sharding internally: sinogram is a plain input here, so
        # fbp_filter scatters it, filters each shard on-device, then gathers the result
        # back to a plain uncommitted array before returning.
        filtered_sinogram = self.fbp_filter(sinogram, filter_name=filter_name, view_batch_size=view_batch_size)

        # back_project receives a plain array and scatters/gathers internally in
        # multi-device mode via _prepare_sinogram_for_backprojection.
        recon = self.back_project(filtered_sinogram)
        recon = self._maybe_gather(recon)

        return recon

