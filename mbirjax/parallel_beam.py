import warnings
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


def _warn_view_batch_size_deprecated(view_batch_size):
    """Warn (only on explicit use) that view_batch_size is deprecated/ignored.

    The row filter kernel (tomography_utils.apply_row_filter) chooses its own
    batch (tomography_utils.ROW_FILTER_BATCH), so view_batch_size no longer
    affects parallel-beam FBP filtering.  Kept on the user-facing methods for
    back-compat; this fires only when a caller passes a non-None value.
    """
    if view_batch_size is not None:
        warnings.warn(
            "view_batch_size is deprecated and ignored for ParallelBeamModel FBP "
            "filtering; the row filter kernel "
            "(mbirjax.tomography_utils.apply_row_filter) sets its own batch "
            "(tomography_utils.ROW_FILTER_BATCH).",
            DeprecationWarning, stacklevel=3)


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

    def _supports_sharding(self):
        """Parallel beam has the placement/movement projector + prior path, so it runs on the
        always-on placement path (single-device auto-defaults to a trivial 1-device mesh)."""
        return True

    def verify_valid_params(self):
        """
        Check that all parameters are compatible for a reconstruction.

        Note:
            Raises ValueError for invalid parameters.
        """
        super().verify_valid_params()
        sinogram_shape, angles, voxel_row_aspect, voxel_slice_aspect = self.get_params(['sinogram_shape', 'angles', 'voxel_row_aspect', 'voxel_slice_aspect'])

        if voxel_row_aspect <= 0:
            error_message = "Voxel row aspect ratio must be positive. \n"
            error_message += "Got {} for voxel_row_aspect.".format(voxel_row_aspect)
            raise ValueError(error_message)

        if voxel_slice_aspect != 1.0:
            error_message = "Setting voxel slice aspect ratio is not supported for parallel beam model. \n"
            error_message += "Got {} for voxel_slice_aspect.".format(voxel_slice_aspect)
            raise ValueError(error_message)

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
        geometry_param_names = ['delta_det_channel', 'det_channel_offset', 'delta_voxel', 'voxel_row_aspect']
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
        delta_det_channel, delta_voxel, voxel_row_aspect = self.get_params(['delta_det_channel', 'delta_voxel', 'voxel_row_aspect'])

        delta_voxel_row = voxel_row_aspect * delta_voxel

        max_footprint = jnp.maximum(delta_voxel, delta_voxel_row)

        # Compute the maximum number of detector rows/channels on either side of the center detector hit by a voxel
        psf_radius = int(jnp.ceil(jnp.ceil(max_footprint / delta_det_channel) / 2))

        return psf_radius

    def auto_set_recon_geometry(self, no_compile=False, no_warning=False):
        """Compute the default recon size using the internal parameters delta_channel and delta_pixel plus
          the number of channels from the sinogram"""
        delta_det_row, delta_det_channel = self.get_params(['delta_det_row', 'delta_det_channel'])

        voxel_row_aspect = self.get_params('voxel_row_aspect')

        # Compute delta_voxel for each dimension
        delta_voxel = self.get_params('delta_det_channel') / self.get_magnification()
        delta_voxel_row = voxel_row_aspect * delta_voxel

        # Compute the recon_shape
        sinogram_shape = self.get_params('sinogram_shape')
        num_det_rows, num_det_channels = sinogram_shape[1:3]
        magnification = self.get_magnification()
        num_recon_rows = int(jnp.ceil(num_det_channels * delta_det_channel / (delta_voxel_row * magnification)))
        num_recon_cols = int(jnp.ceil(num_det_channels * delta_det_channel / (delta_voxel * magnification)))
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
        n_p, n_p_center, W_p_c, footprint_xy = ParallelBeamModel.compute_proj_data(pixel_indices, angle, projector_params)
        L_max = jnp.minimum(1.0, W_p_c)
        delta_voxel_row = gp.voxel_row_aspect * gp.delta_voxel

        # Size the detector-row axis from the actual input cylinder, not from
        # projector_params.sinogram_shape, so that a caller may pass a slice-band
        # of the cylinder (a contiguous subset of slices) and get back only the
        # corresponding detector rows.  Slice r maps only to detector row r in
        # parallel beam (the horizontal projection below mixes channels, never
        # rows), so restricting the input slices restricts the output rows with no
        # other change.  When the full cylinder is passed this equals num_det_rows,
        # so single-device behavior is unchanged.  This is the adjoint of the
        # row-sliced back projection kernel, and is what lets sharded forward
        # projection stream the slice axis in bands.
        num_input_slices = voxel_values.shape[1]

        # The horizontal projection scatters each pixel's contribution into its
        # detector channel n.  Build the view CHANNEL-MAJOR -- (channels, slices)
        # rather than (slices, channels) -- so the scatter writes a CONTIGUOUS row
        # (stride 1) instead of a column (stride num_det_channels).  A column stride
        # equal to a power-of-2 num_det_channels (e.g. 256/1024/2048 detectors)
        # aliases the CPU cache and runs several times slower at large slice counts;
        # the contiguous row access avoids that entirely and is faster regardless of
        # the channel count.  Transpose back to (slices, channels) on return (one
        # cheap pass, fused by XLA) so the output layout and all callers are
        # unchanged.
        sinogram_view_T = jnp.zeros((num_det_channels, num_input_slices))

        # Do the projection
        for n_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius+1):
            n = n_p_center + n_offset
            abs_delta_p_c_n = jnp.abs(n_p - n)
            L_p_c_n = jnp.clip((W_p_c + 1.0) / 2.0 - abs_delta_p_c_n, 0.0, L_max)
            A_chan_n = ((delta_voxel_row * gp.delta_voxel) / footprint_xy) * L_p_c_n
            A_chan_n *= (n >= 0) * (n < num_det_channels)
            sinogram_view_T = sinogram_view_T.at[n, :].add(A_chan_n.reshape((-1, 1)) * voxel_values)

        return sinogram_view_T.T

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
        n_p, n_p_center, W_p_c, footprint_xy = ParallelBeamModel.compute_proj_data(pixel_indices, angle, projector_params)
        L_max = jnp.minimum(1.0, W_p_c)
        delta_voxel_row = gp.voxel_row_aspect * gp.delta_voxel

        # Size the slice axis from the actual input view, not from
        # projector_params.sinogram_shape, so that a caller may pass a row-sliced
        # view (a contiguous subset of detector rows) and get back only the
        # corresponding recon slices.  Detector row r maps only to slice r in
        # parallel beam (the horizontal projection below mixes channels, never
        # rows), so restricting the input rows restricts the output slices with no
        # other change.  When the full view is passed this equals num_det_rows, so
        # single-device behavior is unchanged.  This is what lets sharded back
        # projection stream the slice axis in bands.
        num_input_rows = sinogram_view.shape[0]

        # The horizontal projection gathers each pixel's detector channel n.  Read
        # the view CHANNEL-MAJOR -- transpose to (channels, rows) up front so the
        # per-pixel gather reads a CONTIGUOUS row (stride 1) instead of a column
        # (stride num_det_channels).  A power-of-2 num_det_channels column stride
        # aliases the CPU cache and runs several times slower at large row counts;
        # the contiguous access avoids it (the adjoint of the forward kernel's
        # channel-major scatter).
        sinogram_view_T = sinogram_view.T            # (num_det_channels, num_input_rows)
        det_voxel_cylinder = jnp.zeros((num_pixels, num_input_rows))
        # Do the horizontal projection
        for n_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius + 1):
            n = n_p_center + n_offset
            abs_delta_p_c_n = jnp.abs(n_p - n)
            L_p_c_n = jnp.clip((W_p_c + 1.0) / 2.0 - abs_delta_p_c_n, 0.0, L_max)
            A_chan_n = ((delta_voxel_row * gp.delta_voxel) / footprint_xy) * L_p_c_n
            A_chan_n *= (n >= 0) * (n < num_det_channels)
            A_chan_n = A_chan_n ** coeff_power
            det_voxel_cylinder = jnp.add(det_voxel_cylinder, A_chan_n.reshape((-1, 1)) * sinogram_view_T[n, :])

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
            n_p, n_p_center, W_p_c, footprint_xy
        """
        # Get all the geometry parameters - we use gp since geometry parameters is a named tuple and we'll access
        # elements using, for example, gp.delta_det_channel, so a longer name would be clumsy.
        gp = projector_params.geometry_params

        num_views, num_det_rows, num_det_channels = projector_params.sinogram_shape
        recon_shape = projector_params.recon_shape

        delta_voxel_row = gp.voxel_row_aspect * gp.delta_voxel

        # Convert the index into (i,j,k) coordinates corresponding to the indices into the 3D voxel array
        row_index, col_index = jnp.unravel_index(pixel_indices, recon_shape[:2])

        x_p = ParallelBeamModel.recon_ij_to_x(row_index, col_index, gp.delta_voxel, delta_voxel_row, recon_shape, angle)

        det_center_channel = (num_det_channels - 1) / 2.0  # num_of_cols

        # Calculate indices on the detector grid
        n_p = (x_p + gp.det_channel_offset) / gp.delta_det_channel + det_center_channel
        n_p_center = jnp.round(n_p).astype(int)

        # Compute footprint for row and columns
        footprint_xy = jnp.maximum(jnp.abs(jnp.cos(angle)) * gp.delta_voxel, jnp.abs(jnp.sin(angle)) * delta_voxel_row)

        # Compute projected voxel width along columns and rows (in fraction of detector size)
        W_p_c = footprint_xy / gp.delta_det_channel

        proj_data = (n_p, n_p_center, W_p_c, footprint_xy)

        return proj_data

    @staticmethod
    def recon_ij_to_x(i, j, delta_voxel, delta_voxel_row, recon_shape, angle):
        """
        Convert (i, j, k) indices into the recon volume to corresponding (x, y, z) coordinates.
        """
        num_recon_rows, num_recon_cols, num_recon_slices = recon_shape

        # Compute the un-rotated coordinates relative to iso
        # Note the change in order from (i, j) to (y, x)!!
        y_tilde = delta_voxel_row * (i - (num_recon_rows - 1) / 2.0)
        x_tilde = delta_voxel * (j - (num_recon_cols - 1) / 2.0)

        # Precompute cosine and sine of view angle, then do the rotation
        cosine = jnp.cos(angle)  # length = num_views
        sine = jnp.sin(angle)  # length = num_views

        x = cosine * x_tilde - sine * y_tilde
        y = sine * x_tilde + cosine * y_tilde

        return x

    def direct_recon(self, sinogram, filter_name="ramp", view_batch_size=None, output_sharded=False):
        _warn_view_batch_size_deprecated(view_batch_size)
        return self.fbp_recon(sinogram, filter_name=filter_name, output_sharded=output_sharded)

    def direct_filter(self, sinogram, filter_name="ramp", view_batch_size=None, output_sharded=False):
        """
        Perform filtering on the given sinogram as needed for an FBP/FDK or other direct recon.

        This is a thin alias for :meth:`fbp_filter` and shares its contract: the
        input may be plain or view-sharded (a plain input is sharded at entry
        when sharding is on), and the OUTPUT form is chosen by ``output_sharded``
        — plain by default, the view-sharded device form when True.

        Args:
            sinogram (jax array): The input sinogram with shape (num_views, num_rows, num_channels).
            filter_name (string, optional): Name of the filter to be used. Defaults to "ramp"
            view_batch_size (int, optional): DEPRECATED and ignored (see fbp_filter).
            output_sharded (bool, optional): If False (default), return a plain
                array.  If True, return the view-sharded device form (on an
                unsharded model the output is the same either way).

        Returns:
            filtered_sinogram (jax array): The sinogram after FBP filtering --
            plain by default, view-sharded if ``output_sharded=True``.
        """
        _warn_view_batch_size_deprecated(view_batch_size)
        return self.fbp_filter(sinogram, filter_name=filter_name, output_sharded=output_sharded)

    def fbp_filter(self, sinogram, filter_name="ramp", view_batch_size=None, output_sharded=False):
        """
        Perform FBP filtering on the given sinogram.

        This is a **user-facing** method.  The input may be plain or view-sharded
        (a plain input is sharded on the view axis at entry when sharding is on);
        the OUTPUT form is chosen by ``output_sharded``: plain by default, the
        view-sharded device form when True.  Pipelined internal callers
        (``fbp_recon`` / ``direct_recon`` followed by back projection) pass
        ``output_sharded=True`` so the data stays on-device with zero host
        transfer.  Under the view-sharding scheme the ramp filter is per-view, so
        each device filters its own views with no cross-device communication.
        When no mesh is configured this simply filters the (single-device) array
        directly.

        Args:
            sinogram (jax array): The input sinogram with shape (num_views, num_rows, num_channels).
            filter_name (string, optional): Name of the filter to be used. Defaults to "ramp"
            view_batch_size (int, optional): DEPRECATED and ignored — the row
                filter kernel (tomography_utils.apply_row_filter) sets its own
                batch (tomography_utils.ROW_FILTER_BATCH).  Kept for back-compat.
            output_sharded (bool, optional): If False (default), return a plain
                array.  If True, return the view-sharded device form (on an
                unsharded model the output is the same either way).

        Returns:
            filtered_sinogram (jax array): The sinogram after FBP filtering --
            plain by default, view-sharded if ``output_sharded=True``.
        """
        _warn_view_batch_size_deprecated(view_batch_size)

        num_channels = sinogram.shape[2]
        # The FBP weight is pi / (number of REAL views): read it from the params (always the
        # problem's shapes), not from the array, whose view axis may be zero-padded for
        # sharding -- padded views contribute nothing, so they must not be counted here.
        num_views = self.get_params('sinogram_shape')[0]

        # Generate the reconstruction filter with appropriate scaling.
        delta_voxel, voxel_row_aspect = self.get_params(['delta_voxel', 'voxel_row_aspect'])
        delta_voxel_row = voxel_row_aspect * delta_voxel
        # Scaling factor adjusts the filter to account for voxel size, ensuring consistent reconstruction.
        # For a detailed theoretical derivation of this scaling factor, please refer to the zip file linked at
        # https://mbirjax.readthedocs.io/en/latest/theory.html
        scaling_factor = 1.0 / (delta_voxel * delta_voxel_row)
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

        # apply_row_filter batches rows internally (ROW_FILTER_BATCH), so each
        # device filters only its LOCAL view-shard's rows — no dependence on how
        # many views a device holds.  See tomography_utils.apply_row_filter.

        if self.is_sharded:
            # Multi-device: one thread per device, each filtering its own view
            # shard locally (no cross-device data movement).

            # Shard once at entry (no-op cost when already view-sharded) so the
            # per-device fan-out below sees every mesh device's shard.  Mirrors
            # fbp_recon's entry-shard; without it a plain input only has a shard
            # on device 0 and the per-device map misses the rest of the mesh.
            sinogram = self._shard_sinogram(sinogram)

            # Map each device to the local sinogram shard already resident on it.
            dev_to_shard = {s.device: s.data for s in sinogram.addressable_shards}

            def worker_process(i, device):
                # Per-device work: filter this device's contiguous block of views.
                # Runs inside run_per_device under jax.default_device(device), so
                # the small filter upload and the FFT all happen on `device` and
                # the result stays there (zero host transfer).
                shard = dev_to_shard[device]
                filter_jax = jnp.array(filter_np)   # tiny upload: 2*channels-1 floats
                return tomography_utils.apply_row_filter(shard, filter_jax)

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
            filtered_sinogram = tomography_utils.apply_row_filter(sinogram, filter_jax)

        # No post-scaling: the voxel factor and pi/num_views were folded into the
        # filter above, so each device's kernel output is already fully scaled.
        if output_sharded:
            return filtered_sinogram                     # keep the device form
        return self._gather_sinogram(filtered_sinogram)  # default: plain output

    def fbp_recon(self, sinogram, filter_name="ramp", view_batch_size=None, output_sharded=False):
        """
        Perform filtered back-projection (FBP) reconstruction on the given sinogram.

        Our implementation uses standard filtering of the sinogram, then uses the adjoint of the forward projector to
        perform the backprojection.  This is different from many implementations, in which the backprojection is not
        exactly the adjoint of the forward projection.  For a detailed theoretical derivation of this implementation,
        see the zip file linked at this page: https://mbirjax.readthedocs.io/en/latest/theory.html

        This is a **user-facing** method.  The input may be plain or sharded
        (a plain sinogram is sharded on the view axis once at entry when sharding
        is on); the OUTPUT form is chosen by ``output_sharded``.  Internally the
        pipeline stays on-device throughout — ``fbp_filter`` then
        ``back_project``, both called with ``output_sharded=True`` (zero
        intermediate host transfer).  By default the recon is gathered to a plain
        array at exit; with ``output_sharded=True`` it is returned slice-sharded
        (no host round-trip), so a sharded FBP result can feed a sharded consumer
        (e.g. the VCD init).  With no mesh configured the shard/gather are no-ops
        and the array flows through unchanged.

        Args:
            sinogram (jax array): The input sinogram with shape (num_views, num_rows, num_channels).
            filter_name (string, optional): Name of the filter to be used. Defaults to "ramp"
            view_batch_size (int, optional): DEPRECATED and ignored (see fbp_filter).
            output_sharded (bool, optional): If False (default), return a plain
                array.  If True, return the slice-sharded device form (on an
                unsharded model the output is the same either way).

        Returns:
            recon (jax array): The reconstructed volume — plain by default,
            slice-sharded if ``output_sharded=True``.
        """
        _warn_view_batch_size_deprecated(view_batch_size)

        # Shard once at entry so the filter receives view-sharded data (no-op
        # when no mesh is configured or already sharded).
        sinogram = self._shard_sinogram(sinogram)

        # Internal pipeline stage: keep the device form, no host transfer.
        filtered_sinogram = self.fbp_filter(sinogram, filter_name=filter_name,
                                            output_sharded=True)

        # Keep the recon in the device form through the pipeline; the exit
        # handling below is the single place the output form is decided.
        recon = self.back_project(filtered_sinogram, output_sharded=True)
        return recon if output_sharded else self._gather_recon(recon)

