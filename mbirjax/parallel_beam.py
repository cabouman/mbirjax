from functools import partial
from collections import namedtuple
from typing import Literal, Union, overload, Any

import numpy as np
import jax
import jax.numpy as jnp
import mbirjax as mj
import mbirjax._sharding as mjs
from mbirjax import TomographyModel, tomography_utils
from mbirjax.projectors import (horizontal_fan_project, horizontal_fan_back,
                                SORTED_CHANNEL_REDUCE_MIN_COLS)

ParallelBeamParamNames = mj.ParamNames | Literal['angles']


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

    def _back_project_view_shard_to_band(self, view_data, pixel_indices, g0, g1,
                                         owned_view_indices, coeff_power):
        """Parallel-beam specialization of the sharded slice-band back projection (overrides the
        base banded path): detector row r back-projects to slice r alone, so the slice band [g0, g1)
        IS detector rows [g0:g1).  Crop the detector-row axis and run the standard back projector
        (the kernel sizes its output slices from the input rows) -- cheaper than the base banded
        path, which would process the full detector rows.  Returns the per-view-owner partial band
        ``(num_pixels, g1 - g0)``."""
        return self.projector_functions.sparse_back_project(
            view_data[:, g0:g1, :], pixel_indices,
            owned_view_indices=owned_view_indices, coeff_power=coeff_power)

    # Measured GPU forward tiling (band x pixel-batch grid: fwd_band_pixel_sweep.py; results
    # digest in plans/experiments/projector_kernels/fwd_back_findings.md).  Model-level forward
    # speedups of 2-3.6x over the inherited defaults.  CPU measured the OPPOSITE direction
    # on both knobs, so these apply to GPU layouts only; the base (CPU) policy is untouched.
    _FWD_SLICE_BAND_GPU = 256    # the measured knee; whole-shard bands are WORSE at scale
    _FWD_PIXEL_BATCH_GPU = 8192  # larger sorts use the GPU better + fewer scan-carry steps;
                                 # 16384 is NOT uniformly better (non-monotonic in size)

    def _select_tile_policy(self, on_gpu, num_views, num_slices, n_devices):
        """Parallel-beam tiling: GPU forward gets the measured band/pixel-batch/sorted-reduce
        combination (constants above); everything else inherits the base policy."""
        tiles = super()._select_tile_policy(on_gpu, num_views, num_slices, n_devices)
        if not on_gpu:
            return tiles
        # The sorted channel reduction (defined at projectors.channel_scatter_reduce) pays a
        # per-call sort that amortizes over the band width;
        # with the 256-band policy the bands are wide everywhere except tiny problems, where the
        # BALANCED band width (bands split ±1, so ~slices_per_dev/n_bands) can drop below the
        # measured crossover and the scatter-add stays faster.
        slices_per_dev = -(-num_slices // max(1, n_devices))
        band = min(self._FWD_SLICE_BAND_GPU, max(1, slices_per_dev))
        balanced_band = slices_per_dev // max(1, -(-slices_per_dev // band))
        return tiles._replace(
            fwd_slice_band=self._FWD_SLICE_BAND_GPU,
            fwd_pixel_batch=self._FWD_PIXEL_BATCH_GPU,
            sort_by_channel=balanced_band >= SORTED_CHANNEL_REDUCE_MIN_COLS,
            # Back kernel: one stacked gather covering every psf tap.  A GPU win because the
            # back kernel is almost entirely gather-bound and parallel beam has no vertical
            # fan behind which the gather latency could hide; measured WORSE on CPU, which
            # keeps the per-tap loop (see projectors.horizontal_fan_back).
            back_stacked_gather=True,
        )

    def _forward_project_to_view_shards(self, devices, n_dev, num_slices, num_pixels,
                                     recon_shard_info, view_ranges, local_pixels):
        """Parallel-beam specialization of the sharded forward projection (overrides the base
        gather+monolithic path): band the slice axis and broadcast each band; each view-owner
        projects detector rows [g0:g1) from the band (row r <- slice r) and concatenates its
        row-bands -- never gathering the full cylinder.  Band sizing: the tile policy's
        fwd_slice_band when set (GPU: 256, measured -- see _select_tile_policy), else the
        memory-driven back formula (safe and conservative; forward's transient is even smaller,
        no n_dev-way gather)."""
        slices_per_dev = num_slices // n_dev
        band_len = self._slice_band_length(
            slices_per_dev, n_dev, num_pixels,
            fixed_band=self._fixed_slice_band('fwd'))
        band_bounds = self._balanced_slice_bounds(slices_per_dev, band_len)
        return self._forward_project_all_bands(
            band_bounds, recon_shard_info, view_ranges, local_pixels, devices)

    def _sino_row_padding(self):
        """Parallel beam ties detector row r to recon slice r (the kernels mix channels,
        never rows; recon_shape[2] == sinogram_shape[1] is enforced in
        verify_valid_params), so when the recon slice axis is padded for sharding the
        sinogram's row axis must present the SAME padded length: the entry placement
        zero-fills the row tail, keeping it exactly inert.  No padding -> None."""
        if self.recon_placement.is_padded:
            return 1, self.recon_placement.real_size, self.recon_placement.padded_size
        return None

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
        # The class is shared across instances (make_geometry_params) so the projectors' jit cache
        # is shared rather than re-traced per instance.
        geometry_params = self.make_geometry_params(geometry_param_names, geometry_param_values)

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
    def forward_project_pixel_batch_to_one_view(voxel_values, pixel_indices, angle, n_p_centers,
                                                projector_params):
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
            n_p_centers (jax array of int): (num_pixels,) integer channel centers for this view,
                computed OUTSIDE this program (see compute_channel_coordinate) -- the
                concrete-input contract; do not recompute in-kernel.
            projector_params (namedtuple):  tuple of (sinogram_shape, recon_shape, get_geometry_params())

        Returns:
            jax array of shape (num_det_rows, num_det_channels)
        """
        # Get all the geometry parameters - we use gp since geometry parameters is a named tuple and we'll access
        # elements using, for example, gp.delta_det_channel, so a longer name would be clumsy.
        gp = projector_params.geometry_params
        num_views, num_det_rows, num_det_channels = projector_params.sinogram_shape

        # Get the data needed for horizontal projection
        n_p, W_p_c, footprint_xy = ParallelBeamModel.compute_proj_data(pixel_indices, angle, projector_params)
        delta_voxel_row = gp.voxel_row_aspect * gp.delta_voxel

        # The output detector-row axis is sized from the actual input cylinder
        # (values.shape[1] inside the fan helper), NOT from projector_params: parallel beam
        # maps slice r only to detector row r, so a caller may pass a slice-BAND of the
        # cylinder and get back just the corresponding detector rows -- this is what lets
        # sharded forward projection stream the slice axis in bands (adjoint of the
        # row-sliced back kernel below).
        #
        # The horizontal projection is the shared trapezoid-rule fan (see
        # horizontal_fan_project in projectors.py, which also owns the channel-major
        # layout rationale and the sort_by_channel branch -- decided by the tile policy,
        # trace-time since projector_params is static).  Parallel beam's weight scale is the
        # in-plane voxel area over the footprint length, a per-view scalar.
        weight_scale = (delta_voxel_row * gp.delta_voxel) / footprint_xy
        sinogram_view_T = horizontal_fan_project(n_p, n_p_centers, W_p_c, weight_scale,
                                                 voxel_values, num_det_channels, gp.psf_radius,
                                                 use_sorted=projector_params.sort_by_channel)
        return sinogram_view_T.T

    @staticmethod
    @partial(jax.jit, static_argnames='projector_params')
    def back_project_one_view_to_pixel_batch(sinogram_view, pixel_indices, angle, n_p_centers,
                                             projector_params, coeff_power=1):
        """
        Apply parallel back projection to a single sinogram view and return the resulting voxel cylinders.

        Args:
            sinogram_view (2D jax array): one view of the sinogram to be back projected.
                2D jax array of shape (num_det_rows)x(num_det_channels)
            pixel_indices (1D jax array of int):  indices into flattened array of size num_rows x num_cols.
            angle (float): The projection angle in radians for this view.
            n_p_centers (jax array of int): (num_pixels,) integer channel centers for this view --
                the SAME concrete centers the forward kernel uses (see compute_channel_coordinate).
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

        # Get the data needed for horizontal projection
        n_p, W_p_c, footprint_xy = ParallelBeamModel.compute_proj_data(pixel_indices, angle, projector_params)
        delta_voxel_row = gp.voxel_row_aspect * gp.delta_voxel

        # The output slice axis is sized from the actual input view (sino_T.shape[1] inside
        # the fan helper), NOT from projector_params: detector row r maps only to slice r,
        # so a caller may pass a row-SLICED view and get back just the corresponding recon
        # slices -- this is what lets sharded back projection stream the slice axis in bands
        # (adjoint of the slice-banded forward above).
        #
        # The gather itself is the shared adjoint trapezoid-rule fan (see horizontal_fan_back
        # in projectors.py, which owns the channel-major layout rationale and the
        # back_stacked_gather branch -- ON for parallel beam on GPU, where the back kernel
        # has no vertical fan to hide the gather behind; decided by the tile policy,
        # trace-time since projector_params is static).
        sinogram_view_T = sinogram_view.T            # (num_det_channels, num_input_rows)
        weight_scale = (delta_voxel_row * gp.delta_voxel) / footprint_xy
        return horizontal_fan_back(sinogram_view_T, n_p, n_p_centers, W_p_c, weight_scale,
                                   gp.psf_radius, coeff_power=coeff_power,
                                   use_stacked=projector_params.back_stacked_gather)

    @staticmethod
    def compute_channel_coordinate(pixel_indices, single_view_params, projector_params):
        """Continuous projected channel coordinate n_p for one view (the float chain whose
        rounded value is the horizontal fans' integer center; see the base-class docstring
        for the concrete-input contract).  Reuses compute_proj_data verbatim so the
        centers are consistent with the in-kernel weights by construction."""
        return ParallelBeamModel.compute_proj_data(pixel_indices, single_view_params,
                                                   projector_params)[0]

    @staticmethod
    def compute_proj_data(pixel_indices, angle, projector_params):
        """
        Compute the quantities n_p, W_p_c, footprint_xy needed for horizontal projection.

        The integer center round(n_p) is deliberately NOT computed here: the projector
        wrappers round n_p OUTSIDE the projector programs (via compute_channel_coordinate)
        and pass the concrete centers into the kernels -- the horizontal-fan rounding-bug fix.

        Args:
            pixel_indices (1D jax array of int):  indices into flattened array of size num_rows x num_cols.
            angle (float): The projection angle in radians for this view.
            projector_params (namedtuple): tuple of (sinogram_shape, recon_shape, get_geometry_params()).

        Returns:
            n_p, W_p_c, footprint_xy
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

        # Calculate indices on the detector grid.  NOTE: no round here -- the integer center
        # is computed outside the projector programs (see compute_channel_coordinate).
        n_p = (x_p + gp.det_channel_offset) / gp.delta_det_channel + det_center_channel

        # Compute footprint for row and columns
        footprint_xy = jnp.maximum(jnp.abs(jnp.cos(angle)) * gp.delta_voxel, jnp.abs(jnp.sin(angle)) * delta_voxel_row)

        # Compute projected voxel width along columns and rows (in fraction of detector size)
        W_p_c = footprint_xy / gp.delta_det_channel

        proj_data = (n_p, W_p_c, footprint_xy)

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

    def direct_recon(self, sinogram, filter_name="ramp", output_sharded=False):
        return self.fbp_recon(sinogram, filter_name=filter_name, output_sharded=output_sharded)

    def direct_filter(self, sinogram, filter_name="ramp", output_sharded=False):
        """
        Perform filtering on the given sinogram as needed for an FBP/FDK or other direct recon.

        This is a thin alias for :meth:`fbp_filter` and shares its contract: the
        input may be plain or view-sharded (a plain input is sharded at entry
        when sharding is on), and the OUTPUT form is chosen by ``output_sharded``
        — numpy by default, the view-sharded device form when True.

        Args:
            sinogram (numpy or jax array): The input sinogram with shape (num_views, num_rows, num_channels).
            filter_name (string, optional): Name of the filter to be used. Defaults to "ramp"
            output_sharded (bool, optional): If False (default), return a numpy
                array.  If True, return the view-sharded device form (on an
                unsharded model the output is the same either way).

        Returns:
            filtered_sinogram (numpy or jax array): The sinogram after FBP filtering --
            numpy by default, view-sharded if ``output_sharded=True``.
        """
        return self.fbp_filter(sinogram, filter_name=filter_name, output_sharded=output_sharded)

    def fbp_filter(self, sinogram, filter_name="ramp", output_sharded=False):
        """
        Perform FBP filtering on the given sinogram.

        This is a **user-facing** method.  The input may be plain or view-sharded
        (a plain input is sharded on the view axis at entry when sharding is on);
        the OUTPUT form is chosen by ``output_sharded``: numpy by default, the
        view-sharded device form when True.  Pipelined internal callers
        (``fbp_recon`` / ``direct_recon`` followed by back projection) pass
        ``output_sharded=True`` so the data stays on-device with zero host
        transfer.  Under the view-sharding scheme the ramp filter is per-view, so
        each device filters its own views with no cross-device communication
        (a single device is the trivial 1-shard case).

        Args:
            sinogram (numpy or jax array): The input sinogram with shape (num_views, num_rows, num_channels).
            filter_name (string, optional): Name of the filter to be used. Defaults to "ramp"
            output_sharded (bool, optional): If False (default), return a numpy
                array.  If True, return the view-sharded device form (on an
                unsharded model the output is the same either way).

        Returns:
            filtered_sinogram (numpy or jax array): The sinogram after FBP filtering --
            numpy by default, view-sharded if ``output_sharded=True``.
        """
        # Voxel-size scaling factor: adjusts the filter to account for voxel size.  For
        # the theoretical derivation see the zip linked at
        # https://mbirjax.readthedocs.io/en/latest/theory.html
        # The FBP weight pi/num_views is folded into the filter by the shared method;
        # parallel beam has no FDK cosine pre-weight (row_weight=None).  The shared row
        # filter keeps the peak at the input+output floor and, when a mesh is configured,
        # filters each device's own view-shard locally (no cross-device movement).
        delta_voxel, voxel_row_aspect = self.get_params(['delta_voxel', 'voxel_row_aspect'])
        delta_voxel_row = voxel_row_aspect * delta_voxel
        scaling_factor = 1.0 / (delta_voxel * delta_voxel_row)
        return self._apply_direct_recon_filter(
            sinogram, filter_name, filter_scale=scaling_factor,
            output_sharded=output_sharded, row_weight=None)

    def fbp_recon(self, sinogram, filter_name="ramp", output_sharded=False):
        """
        Perform filtered back-projection (FBP) reconstruction on the given sinogram.

        Our implementation uses standard filtering of the sinogram, then uses the adjoint of the forward projector to
        perform the backprojection.  This is different from many implementations, in which the backprojection is not
        exactly the adjoint of the forward projection.  For a detailed theoretical derivation of this implementation,
        see the zip file linked at this page: https://mbirjax.readthedocs.io/en/latest/theory.html

        Note:
            FBP assumes the view angles are EQUALLY SPACED over the full angular range (the
            ``pi / num_views`` angular weight in the ramp filter).  On nonuniformly-spaced or
            limited-angle data it is only approximate and is best used as an initializer for the
            iterative ``recon()``, which corrects the angular weighting.

        This is a **user-facing** method.  The input may be plain or sharded
        (a plain sinogram is sharded on the view axis once at entry when sharding
        is on); the OUTPUT form is chosen by ``output_sharded``.  Internally the
        pipeline stays on-device throughout — ``fbp_filter`` then
        ``back_project``, both called with ``output_sharded=True`` (zero
        intermediate host transfer).  By default the recon is gathered to a numpy
        array at exit; with ``output_sharded=True`` it is returned slice-sharded
        (no host round-trip), so a sharded FBP result can feed a sharded consumer
        (e.g. the VCD init).  On a single device the shard/gather are trivial 1-shard
        operations.

        Args:
            sinogram (numpy or jax array): The input sinogram with shape (num_views, num_rows, num_channels).
            filter_name (string, optional): Name of the filter to be used. Defaults to "ramp"
            output_sharded (bool, optional): If False (default), return a numpy
                array.  If True, return the slice-sharded device form (on a single
                device the output is the same either way).

        Returns:
            recon (numpy or jax array): The reconstructed volume — numpy by default,
            slice-sharded if ``output_sharded=True``.
        """
        # Shard once at entry so the filter receives view-sharded data (a no-op
        # when already view-sharded; a single device is the trivial 1-shard case).
        sinogram = self._shard_sinogram(sinogram)

        # Internal pipeline stage: keep the device form, no host transfer.
        filtered_sinogram = self.fbp_filter(sinogram, filter_name=filter_name,
                                            output_sharded=True)

        # Keep the recon in the device form through the pipeline; the exit
        # handling below is the single place the output form is decided.
        recon = self.back_project(filtered_sinogram, output_sharded=True)
        return recon if output_sharded else self._gather_recon(recon)

