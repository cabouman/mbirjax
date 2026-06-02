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

        # Allocate the voxel cylinder array.  Size the slice axis from the actual
        # input view, not from projector_params.sinogram_shape, so that a caller
        # may pass a row-sliced view (a contiguous subset of detector rows) and
        # get back only the corresponding recon slices.  Detector row r maps only
        # to slice r in parallel beam (the horizontal projection below mixes
        # channels, never rows), so restricting the input rows restricts the
        # output slices with no other change.  When the full view is passed this
        # equals num_det_rows, so single-device behavior is unchanged.  This is
        # what lets sharded back projection stream the slice axis in bands.
        num_input_rows = sinogram_view.shape[0]
        det_voxel_cylinder = jnp.zeros((num_pixels, num_input_rows))
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

    def direct_recon(self, sinogram, filter_name="ramp", view_batch_size=None):
        _warn_view_batch_size_deprecated(view_batch_size)
        return self.fbp_recon(sinogram, filter_name=filter_name)

    def direct_filter(self, sinogram, filter_name="ramp", view_batch_size=None):
        """
        Perform filtering on the given sinogram as needed for an FBP/FDK or other direct recon.

        This is a thin **internal** alias for :meth:`fbp_filter` and shares its
        sharded contract: it does NOT shard at entry or gather at exit.  Under a
        mesh the input must already be view-sharded and the output stays
        view-sharded; with no mesh the array is filtered directly.  The
        plain<->sharded boundary lives in the user-facing reconstruction methods
        (``fbp_recon`` / ``direct_recon``), not here.

        Args:
            sinogram (jax array): The input sinogram with shape (num_views, num_rows, num_channels).
                Under a mesh it must already be view-sharded.
            filter_name (string, optional): Name of the filter to be used. Defaults to "ramp"
            view_batch_size (int, optional): DEPRECATED and ignored (see fbp_filter).

        Returns:
            filtered_sinogram (jax array): The sinogram after FBP filtering, in
            the same (view) sharding as the input (or a plain array when no mesh
            is configured).
        """
        _warn_view_batch_size_deprecated(view_batch_size)
        return self.fbp_filter(sinogram, filter_name=filter_name)

    def fbp_filter(self, sinogram, filter_name="ramp", view_batch_size=None):
        """
        Perform FBP filtering on the given sinogram.

        This is an **internal** sharded-contract method.  It does NOT shard at
        entry or gather at exit: when a mesh is configured the input is expected
        to already be view-sharded (each device holds a contiguous block of
        views), and the filtered sinogram is returned in that same view sharding.
        Keeping the data on-device lets pipelined callers (``fbp_recon`` /
        ``direct_recon`` followed by back projection) run with zero host
        transfer.  The plain<->sharded boundary lives in the user-facing wrappers
        (``direct_filter`` / ``fbp_recon`` / ``direct_recon``), which shard at
        entry and unshard at exit.  Under the view-sharding scheme the ramp
        filter is per-view, so each device filters its own views with no
        cross-device communication.  When no mesh is configured this simply
        filters the (single-device) array directly.

        Args:
            sinogram (jax array): The input sinogram with shape (num_views, num_rows, num_channels).
                Under a mesh it must already be view-sharded (the user-facing
                wrappers establish this).
            filter_name (string, optional): Name of the filter to be used. Defaults to "ramp"
            view_batch_size (int, optional): DEPRECATED and ignored — the row
                filter kernel (tomography_utils.apply_row_filter) sets its own
                batch (tomography_utils.ROW_FILTER_BATCH).  Kept for back-compat.

        Returns:
            filtered_sinogram (jax array): The sinogram after FBP filtering, in
            the same (view) sharding as the input (or a plain array when no mesh
            is configured).
        """
        _warn_view_batch_size_deprecated(view_batch_size)

        num_views, _, num_channels = sinogram.shape

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

        # apply_row_filter batches rows internally (ROW_FILTER_BATCH), so each
        # device filters only its LOCAL view-shard's rows — no dependence on how
        # many views a device holds.  See tomography_utils.apply_row_filter.

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
        return filtered_sinogram

    def fbp_recon(self, sinogram, filter_name="ramp", view_batch_size=None):
        """
        Perform filtered back-projection (FBP) reconstruction on the given sinogram.

        Our implementation uses standard filtering of the sinogram, then uses the adjoint of the forward projector to
        perform the backprojection.  This is different from many implementations, in which the backprojection is not
        exactly the adjoint of the forward projection.  For a detailed theoretical derivation of this implementation,
        see the zip file linked at this page: https://mbirjax.readthedocs.io/en/latest/theory.html

        This is a **user-facing** method: its output sharding **matches its
        input**.  When a mesh is configured it shards the sinogram on the view
        axis once at entry, then runs the on-device pipeline — ``fbp_filter``
        (internal: sharded view -> sharded view) followed by ``back_project``
        (which, given sharded input, returns a slice-sharded recon).  The data
        stays resident on its devices throughout (zero intermediate host
        transfer).  If the original input was a plain array the recon is gathered
        at exit (plain in -> plain out); if it was already sharded the recon is
        returned slice-sharded (no host round-trip), so a sharded FBP result can
        feed a sharded consumer (e.g. the VCD init).  With no mesh configured the
        shard/gather are no-ops and the array flows through unchanged.

        Args:
            sinogram (jax array): The input sinogram with shape (num_views, num_rows, num_channels).
            filter_name (string, optional): Name of the filter to be used. Defaults to "ramp"
            view_batch_size (int, optional): DEPRECATED and ignored (see fbp_filter).

        Returns:
            recon (jax array): The reconstructed volume — plain if the input was
            plain, slice-sharded if the input was a sharded array.
        """
        _warn_view_batch_size_deprecated(view_batch_size)
        # Decide the exit handling from the ORIGINAL input (before the entry
        # shard, which would make any input look sharded).
        input_is_sharded = isinstance(getattr(sinogram, 'sharding', None),
                                      jax.sharding.NamedSharding)

        # Shard once at entry so the internal fbp_filter receives view-sharded
        # data (no-op when no mesh is configured or already sharded).
        sinogram = self._shard_sinogram(sinogram)

        # Internal pipeline stage: sharded view -> sharded view, no host transfer.
        filtered_sinogram = self.fbp_filter(sinogram, filter_name=filter_name)

        # back_project matches its input: the filtered sinogram is sharded, so it
        # returns a slice-sharded recon (no gather inside).  Match fbp_recon's own
        # input: gather to plain only if the original sinogram was plain.
        recon = self.back_project(filtered_sinogram)
        return recon if input_is_sharded else self._gather_recon(recon)

