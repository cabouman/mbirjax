import jax
import jax.numpy as jnp
from functools import partial
from collections import namedtuple
import os
import warnings
from typing import Literal, Union, overload, Any

import numpy as np

import mbirjax as mj
from mbirjax import TomographyModel, tomography_utils, ParameterHandler
from mbirjax.projectors import (horizontal_fan_project, horizontal_fan_back,
                                vertical_fan_band_gather, SORTED_CHANNEL_REDUCE_MIN_COLS)

ConeBeamParamNames = mj.ParamNames | Literal['view_params_array', 'source_detector_dist', 'source_iso_dist', 'recon_slice_offset']

# Default slice-band size for the cone back projector's rolled vertical-fan loop.  The
# back projector computes the horizontal fan once per view, then walks the recon slice
# axis in bands of this many slices using a jax.lax.map.  The loop is ROLLED (the band
# body compiles once and iterates at runtime), so the compiled program size is
# independent of the slice count -- which can range from tens of slices on one device to
# thousands across many devices.  The band size bounds the per-band scratch to
# (pixel_batch x CONE_SLICE_BAND_SIZE); it is the back projector's memory knob, replacing
# the old entries_per_cylinder_batch geometry parameter.  128 matches that old default.
CONE_SLICE_BAND_SIZE = 128

# Detector-row batch size for the cone FORWARD vertical fan's rolled jax.lax.map over
# detector rows.  Like the back band size, this is a memory/compile knob that bounds the
# per-batch transient.  The forward projection does NOT band the slice axis (it stays
# monolithic -- see the banded-back section), so this chunks the OUTPUT detector rows
# instead.  It also replaces the old entries_per_cylinder_batch parameter; 128 matches the
# old default, so the forward output is unchanged.
CONE_FORWARD_DET_ROW_BATCH = 128


class ConeBeamModel(TomographyModel):
    """
    A class designed for handling forward and backward projections in a cone beam geometry, extending the
    :ref:`TomographyModelDocs`. This class offers specialized methods and parameters tailored for cone beam setups.

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
        angles (numpy or jax array):
            A 1D array of projection angles, in radians, specifying the angle of each projection relative to the origin.
        source_detector_dist (float): Distance between the X-ray source and the detector in units of ALU.
        source_iso_dist (float): Distance between the X-ray source and the center of rotation in units of ALU.
        helical_z_shifts (numpy or jax array, optional): Per-view axial shifts (ALU; same length as angles) for helical mode.
        use_curved_detector (bool, optional): Detector geometry type. Either False (default) or True implies each detector row has constant distance from source.

    Note:
        Additional parameter:

        **recon_slice_offset** (float, default=0) -
        This parameter controls the vertical offset of the reconstruction in ALU. If recon_slice_offset is positive, the region below iso is reconstructed.

    See Also
    --------
    TomographyModel : The base class from which this class inherits.
    """

    def __init__(self, sinogram_shape, angles, source_detector_dist, source_iso_dist, helical_z_shifts=None,
                 use_curved_detector=False):

        self.bp_psf_radius = 1

        if helical_z_shifts is None:
            # If helical_z_shifts is not provided or None,
            # then circular scan is helical with all-zero z shifts
            helical_z_shifts = jnp.zeros_like(angles)

        view_dependent_vecs = [vec.flatten() for vec in [angles, helical_z_shifts]]
        try:
            view_params_array = jnp.stack(view_dependent_vecs, axis=1)
        except ValueError as e:
            print(e)
            raise ValueError("Incompatible view dependent vector lengths:  all view-dependent vectors must have the "
                             "same length.")
        
        view_params_name = 'view_params_array'
        view_params_component_names = ['angles', 'helical_z_shifts']

        super().__init__(sinogram_shape, view_params_array=view_params_array,
                         source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist,
                         view_params_name=view_params_name, view_params_component_names=view_params_component_names,
                         recon_slice_offset=0.0, use_curved_detector=use_curved_detector)

    @overload
    def get_params(self, parameter_names: Union[ConeBeamParamNames, list[ConeBeamParamNames]]) -> Any: ...

    def get_params(self, parameter_names) -> Any:
        return super().get_params(parameter_names)
    
    def get_magnification(self):
        """
        Returns the magnification for the cone beam geometry.

        Returns:
            magnification = source_detector_dist / source_iso_dist
        """
        source_detector_dist, source_iso_dist = self.get_params(['source_detector_dist', 'source_iso_dist'])
        if jnp.isinf(source_detector_dist):
            magnification = 1
        else:
            magnification = source_detector_dist / source_iso_dist
        return magnification

    def verify_valid_params(self):
        """
        Check that all parameters are compatible for a reconstruction.

        Note:
            Raises ValueError for invalid parameters.
        """
        super().verify_valid_params()
        sinogram_shape, view_params_array, voxel_row_aspect, voxel_slice_aspect = self.get_params(['sinogram_shape', 'view_params_array', 'voxel_row_aspect', 'voxel_slice_aspect'])
        num_views, num_det_rows = sinogram_shape[:2]
        if view_params_array is None:
            raise ValueError("view_params_array was not set. This should be created in ConeBeamModel.__init__.")

        if view_params_array.shape != (num_views, 2):
            error_message = "Number view dependent parameter vectors must equal the number of views. \n"
            error_message += "Got {} for length of view-dependent parameters and "
            error_message += "{} for number of views.".format(view_params_array.shape[1], num_views)
            raise ValueError(error_message)
        
        if voxel_row_aspect <= 0:
            error_message = "Voxel row aspect ratio must be positive. \n"
            error_message += "Got {} for voxel_row_aspect.".format(voxel_row_aspect)
            raise ValueError(error_message)
        
        if voxel_slice_aspect <= 0:
            error_message = "Voxel slice aspect ratio must be positive. \n"
            error_message += "Got {} for voxel_slice_aspect.".format(voxel_slice_aspect)
            raise ValueError(error_message)

        # Check for cone angle > 45 degrees
        source_detector_dist, delta_det_row, det_row_offset = \
            self.get_params(['source_detector_dist', 'delta_det_row', 'det_row_offset'])
        half_detector_height = delta_det_row * num_det_rows / 2 + jnp.abs(det_row_offset)
        if half_detector_height > source_detector_dist:
            warnings.warn('Cone angle is more than 45 degrees.  This will likely produce recon artifacts.')

        # TODO:  Check for recon volume extending into the source
        # # Check for a potential division by zero or very small denominator
        # if (source_to_iso_dist - y) < 1e-3>:
        #     raise ValueError("Invalid geometry: Recon volume extends too close to source.")

    def get_geometry_parameters(self):
        """
        Function to get a list of the primary geometry parameters for cone beam projection.

        Returns:
            namedtuple of required geometry parameters.
        """
        # First get the parameters managed by ParameterHandler
        geometry_param_names = [
            'delta_det_row',
            'delta_det_channel',
            'det_row_offset',
            'det_channel_offset',
            'source_detector_dist',
            'delta_voxel',
            'voxel_row_aspect',
            'voxel_slice_aspect',
            'recon_slice_offset',
        ]
        geometry_param_values = self.get_params(geometry_param_names)

        # Then get additional parameters:
        geometry_param_names += ['magnification', 'psf_radius', 'bp_psf_radius',
                                 'use_curved_detector']
        geometry_param_values.append(self.get_magnification())
        geometry_param_values.append(self.get_psf_radius())
        geometry_param_values.append(self.bp_psf_radius)
        geometry_param_values.append(self.get_params('use_curved_detector'))

        # Then create a namedtuple to access parameters by name in a way that can be jit-compiled.
        geometry_params = self.make_geometry_params(geometry_param_names, geometry_param_values)

        return geometry_params

    def get_psf_radius(self):
        """
        Compute the integer radius of the PSF kernel for cone beam projection.
        """
        delta_det_row, delta_det_channel, source_detector_dist, recon_shape, delta_voxel, voxel_row_aspect, voxel_slice_aspect = self.get_params(
            ['delta_det_row', 'delta_det_channel', 'source_detector_dist', 'recon_shape', 'delta_voxel', 'voxel_row_aspect', 'voxel_slice_aspect'])
        magnification = self.get_magnification()
        
        delta_voxel_row = voxel_row_aspect * delta_voxel
        delta_voxel_slice = voxel_slice_aspect * delta_voxel

        # Compute minimum detector pitch
        delta_det = jnp.minimum(delta_det_row, delta_det_channel)

        # Compute maximum magnification
        if jnp.isinf(source_detector_dist):
            max_magnification = 1
            min_magnification = 1
        else:
            source_to_iso_dist = source_detector_dist / magnification
            half_width_x = 0.5 * recon_shape[1] * delta_voxel
            half_width_y = 0.5 * recon_shape[0] * delta_voxel_row
            half_xy_extent = jnp.maximum(half_width_x, half_width_y)
            # This isn't exactly the closest pixel since we're not accounting for rotation but for realistic cases it shouldn't matter.
            source_to_closest_pixel = source_to_iso_dist - half_xy_extent
            max_magnification = source_detector_dist / source_to_closest_pixel
            source_to_farthest_pixel = source_to_iso_dist + half_xy_extent
            min_magnification = source_detector_dist / source_to_farthest_pixel

        # Compute the maximum number of detector rows/channels on either side of the center detector hit by a voxel
        max_voxel_pitch = jnp.maximum(jnp.maximum(delta_voxel, delta_voxel_row), delta_voxel_slice)
        psf_radius = int(jnp.ceil(jnp.ceil((max_voxel_pitch * max_magnification / delta_det)) / 2))
        # Then repeat for the back projection from detector elements to voxels.
        # The voxels closest to the detector will be covered the most by a given detector element.
        # With magnification=1, the number of voxels per element would be delta_det / min_voxel_pitch
        min_voxel_pitch = jnp.minimum(jnp.minimum(delta_voxel, delta_voxel_row), delta_voxel_slice)
        max_voxels_per_detector = delta_det / (min_magnification * min_voxel_pitch)
        self.bp_psf_radius = int(jnp.ceil(jnp.ceil(max_voxels_per_detector) / 2))

        return psf_radius

    def auto_set_recon_geometry(self, no_compile=False, no_warning=False):
        """ Compute the automatic recon shape cone beam reconstruction.

        Lateral extent (xy) covers the detector field of view at iso.  Axial extent (z) covers the
        detector height at iso swept over any helical travel, then extends EACH end to the
        cone-beam visibility bound (the deepest z any measured ray reaches while crossing the
        reconstruction support), so material a ray passes through near the slab ends is
        representable rather than flashing into the end slices -- see the per-end extension
        comment below and plans/flash_remediation/phase_2d_remedies.html.
        """
        delta_det_row, delta_det_channel = self.get_params(['delta_det_row', 'delta_det_channel'])

        voxel_row_aspect, voxel_slice_aspect = self.get_params(['voxel_row_aspect', 'voxel_slice_aspect'])
        
        magnification = self.get_magnification()
        
        # Compute delta_voxel for each dimension
        delta_voxel = delta_det_channel / magnification
        delta_voxel_row = voxel_row_aspect * delta_voxel
        delta_voxel_slice = voxel_slice_aspect * delta_voxel

        # Compute the recon_shape
        sinogram_shape = self.get_params('sinogram_shape')
        num_det_rows, num_det_channels = sinogram_shape[1:3]
        num_recon_rows = int(jnp.round(num_det_channels * ((delta_det_channel / delta_voxel_row) / magnification)))
        num_recon_cols = int(jnp.round(num_det_channels * ((delta_det_channel / delta_voxel) / magnification)))
        
        # z coverage for helical
        z_shifts = self.get_params('view_params_array')[:,1]
        z_min = jnp.min(z_shifts)
        z_max = jnp.max(z_shifts)
        z_travel = z_max - z_min
        
        # Detector height mapped to iso
        H_iso = num_det_rows * (delta_det_row / magnification)
    
        # Base axial coverage: the detector height at iso, swept over the helix travel
        num_recon_slices = int(jnp.ceil((H_iso + z_travel) / delta_voxel_slice))
        num_recon_slices = max(1, num_recon_slices)

        # Center the recon about the helix travel (for circular, z_min=z_max=0 -> offset 0)
        recon_slice_offset = float(0.5 * (z_min + z_max))

        # ---- Per-end axial extension to the cone-beam visibility bound (see docstring) ----
        # An edge ray at height v diverges across the support to |z| = |v|*(SID + R)/SDD > |v|/mag;
        # extend each end by its own excess over H_iso/2 (ends differ under det_row_offset; helical
        # excesses attach at z_max/z_min).  Derivation: plans/flash_remediation/phase_2d_remedies.html.
        source_detector_dist, det_row_offset, det_channel_offset, use_ror_mask = self.get_params(
            ['source_detector_dist', 'det_row_offset', 'det_channel_offset', 'use_ror_mask'])
        support_radius = mj.get_support_radius((num_recon_rows, num_recon_cols), delta_voxel_row,
                                               delta_voxel, use_ror_mask=use_ror_mask)
        # Detector row-edge heights v in ALU, via the model's own detector convention (the outer
        # edges of the first and last rows sit at m = -0.5 and num_det_rows - 0.5; v does not
        # depend on the channel argument n, so n = 0 is arbitrary).
        _, v_row_low = self.detector_mn_to_uv(-0.5, 0.0, delta_det_channel, delta_det_row,
                                              det_channel_offset, det_row_offset,
                                              num_det_rows, num_det_channels)
        _, v_row_high = self.detector_mn_to_uv(num_det_rows - 0.5, 0.0, delta_det_channel,
                                               delta_det_row, det_channel_offset, det_row_offset,
                                               num_det_rows, num_det_channels)
        v_top = max(float(v_row_low), float(v_row_high))    # the +z detector edge
        v_bot = min(float(v_row_low), float(v_row_high))    # the -z detector edge
        # Iso-z reach per unit detector height at the far side of the support: written as
        # 1/mag + R/SDD, which equals (SID + R)/SDD for finite SDD and degrades gracefully to
        # 1/mag = 1 at SDD = inf (rays parallel in z, so the excess reduces to pure
        # det_row_offset compensation).
        z_per_v_far_side = 1.0 / float(magnification)
        if not jnp.isinf(source_detector_dist):
            z_per_v_far_side += support_radius / float(source_detector_dist)
        excess_top = max(0.0, v_top * z_per_v_far_side - float(H_iso) / 2)    # attaches at z_max
        excess_bot = max(0.0, -v_bot * z_per_v_far_side - float(H_iso) / 2)   # attaches at z_min
        num_slices_top = int(np.ceil(excess_top / delta_voxel_slice))
        num_slices_bot = int(np.ceil(excess_bot / delta_voxel_slice))
        num_recon_slices += num_slices_top + num_slices_bot
        # Recenter so the added slices land at the ends that need them
        recon_slice_offset += 0.5 * (num_slices_top - num_slices_bot) * float(delta_voxel_slice)

        recon_shape = (num_recon_rows, num_recon_cols, num_recon_slices)

        self.set_params(no_compile=no_compile, no_warning=no_warning, recon_shape=recon_shape, delta_voxel=delta_voxel, recon_slice_offset=recon_slice_offset)

    # Measured GPU forward pixel batch for LARGE problems (cone_fwd_tile_sweep.py):
    # ~1.5x at the 1024^3 class for 1.5-4 GB more memory (a deliberate trade), but
    # neutral-to-worse at the 512^3 class -- so it applies only above the slice threshold,
    # which sits between the two measured sizes.
    _FWD_PIXEL_BATCH_GPU_LARGE = 4096
    _FWD_PIXEL_BATCH_MIN_SLICES = 768   # 1024-class (1008 slices) qualifies; 512-class (448) does not

    def _select_tile_policy(self, on_gpu, num_views, num_slices, n_devices):
        """Cone tiling: on GPU the horizontal fan uses the SORTED channel reduction
        (defined at projectors.channel_scatter_reduce), and large problems get the measured
        larger forward pixel batch (constants above).

        The sorted reduce's columns are the FULL detector rows (cone forward is never
        row-banded), so its per-call sort amortizes at any realistic detector; the guard
        below only trips on tiny ones.

        Deliberately NOT set: ``back_stacked_gather``.  The stacked horizontal-fan gather
        wins in isolation but changes the FULL cone back kernel not at all -- the gather
        latency already hides behind the vertical-fan band work (cone_back_kernel_ab.py).
        Parallel back, with no vertical fan to hide behind, is where it wins.
        """
        tiles = super()._select_tile_policy(on_gpu, num_views, num_slices, n_devices)
        if not on_gpu:
            return tiles
        num_det_rows = self.get_params('sinogram_shape')[1]
        if num_slices >= self._FWD_PIXEL_BATCH_MIN_SLICES:
            tiles = tiles._replace(fwd_pixel_batch=self._FWD_PIXEL_BATCH_GPU_LARGE)
        # The fused-vfan back kernel (e5_cone_fused_back.py: 9.4x the XLA band sweep;
        # value contract = gradient 1e-5 / Hessian 1e-4, the affine-m rounding note in
        # _pallas_kernels.py).  n=1 -> the single-device short-circuit; n>1 -> the
        # per-owner band path; same availability gates as parallel.
        from mbirjax import _pallas_kernels
        return tiles._replace(
            sort_by_channel=num_det_rows >= SORTED_CHANNEL_REDUCE_MIN_COLS,
            back_pallas=_pallas_kernels.is_available() and n_devices == 1,
            back_pallas_band=_pallas_kernels.is_available() and n_devices > 1)

    @staticmethod
    @partial(jax.jit, static_argnames='projector_params')
    def forward_project_pixel_batch_to_one_view(voxel_values, pixel_indices, single_view_params, n_p_centers,
                                                projector_params):
        """
        Forward project a set of voxels determined by indices into the flattened array of size num_rows x num_cols.

        Args:
            voxel_values (jax array):  2D array of shape (num_indices, num_slices) of voxel values, where
                voxel_values[i, j] is the value of the voxel in slice j at the location determined by indices[i].
            pixel_indices (jax array of int):  1D vector of indices into flattened array of size num_rows x num_cols.
            single_view_params: These are the angle and helical_z_shift for the view being forward projected.
            n_p_centers (jax array of int): (num_pixels,) integer channel centers for this view,
                computed OUTSIDE this program (see compute_channel_coordinate).
                Consumed by the horizontal fan; the vertical fan's per-slice rounds remain
                in-jit (documented accepted risk; a per-slice precompute is not
                materializable).
            projector_params (namedtuple):  tuple of (sinogram_shape, recon_shape, get_geometry_params())

        Returns:
            jax array of shape (num_det_rows, num_det_channels)
        """
        recon_shape = projector_params.recon_shape
        num_recon_slices = recon_shape[2]
        if voxel_values.shape[0] != pixel_indices.shape[0] or len(voxel_values.shape) < 2 or \
                voxel_values.shape[1] != num_recon_slices:
            raise ValueError('voxel_values must have shape[0:2] = (num_indices, num_slices)')

        vertical_fan_projector = ConeBeamModel.forward_vertical_fan_pixel_batch_to_one_view
        horizontal_fan_projector = ConeBeamModel.forward_horizontal_fan_pixel_batch_to_one_view

        # named_scope tags the compiled program (HLO, XLA's intermediate representation) and
        # profiler traces with a stable, code-localized region name (profiling
        # attribution that survives recompiles + jax-version renames; see experiments/profiling).
        with jax.named_scope("cone/forward/vertical_fan"):
            new_voxel_values = vertical_fan_projector(voxel_values, pixel_indices, single_view_params, projector_params)
        with jax.named_scope("cone/forward/horizontal_fan"):
            sinogram_view = horizontal_fan_projector(new_voxel_values, pixel_indices, single_view_params,
                                                     n_p_centers, projector_params)

        return sinogram_view

    @staticmethod
    def forward_vertical_fan_pixel_batch_to_one_view(voxel_values, pixel_indices, single_view_params, projector_params):
        """
        Apply a fan beam forward projection in the vertical direction separately to each voxel determined by indices
        into the flattened array of size num_rows x num_cols.  This returns an array corresponding to the same pixel
        locations, but using values obtained from the projection of the original voxel cylinders onto a detector column,
        so the output array has size (len(pixel_indices), num_det_rows).

        Args:
            voxel_values (jax array):  2D array of shape (num_pixels, num_recon_slices) of voxel values, where
                voxel_values[i, j] is the value of the voxel in slice j at the location determined by indices[i].
            pixel_indices (jax array of int):  1D vector of shape (len(pixel_indices), ) holding the indices into
                the flattened array of size num_rows x num_cols.
            single_view_params: These are the angle and helical_z_shift for the view being forward projected.
            projector_params (namedtuple):  tuple of (sinogram_shape, recon_shape, get_geometry_params())

        Returns:
            jax array of shape (num_pixels, num_det_rows)
        """
        pixel_map = jax.vmap(ConeBeamModel.forward_vertical_fan_one_pixel_to_one_view,
                             in_axes=(0, 0, None, None))
        new_pixels = pixel_map(voxel_values, pixel_indices, single_view_params, projector_params)

        return new_pixels

    @staticmethod
    def forward_horizontal_fan_pixel_batch_to_one_view(voxel_values, pixel_indices, single_view_params,
                                                       n_p_centers, projector_params):
        """
        Apply a horizontal fan beam transformation to a set of voxel cylinders. These cylinders are assumed to have
        slices aligned with detector rows, so that a horizontal fan beam maps a cylinder slice to a detector row.
        This function returns the resulting sinogram view.

        Args:
            voxel_values (jax array):  2D array of shape (num_pixels, num_recon_slices) of voxel values, where
                voxel_values[i, j] is the value of the voxel in slice j at the location determined by indices[i].
            pixel_indices (jax array of int):  1D vector of shape (len(pixel_indices), ) holding the indices into
                the flattened array of size num_rows x num_cols.
            single_view_params: These are the angle and helical_z_shift for the view being forward projected.
            projector_params (namedtuple):  tuple of (sinogram_shape, recon_shape, get_geometry_params())

        Returns:
            jax array of shape (num_det_rows, num_det_channels)
        """
        # Get all the geometry parameters - we use gp since geometry parameters is a named tuple and we'll access
        # elements using, for example, gp.delta_det_channel, so a longer name would be clumsy.
        gp = projector_params.geometry_params
        num_views, num_det_rows, num_det_channels = projector_params.sinogram_shape

        # Get the data needed for horizontal projection
        n_p, W_p_c, footprint_xy = ConeBeamModel.compute_horizontal_data(pixel_indices, single_view_params, projector_params)
        delta_voxel_row = gp.voxel_row_aspect * gp.delta_voxel

        # The shared trapezoid-rule fan (see horizontal_fan_project in projectors.py, which
        # owns the channel-major layout rationale and the sort_by_channel branch -- for cone
        # the reduce columns are the FULL detector rows, so ConeBeamModel._select_tile_policy
        # enables the sorted reduce on GPU except at tiny detectors; trace-time branch since
        # projector_params is static).  Cone's weight scale -- in-plane voxel area over the
        # footprint length -- is a PER-PIXEL (num_pixels,) array (per-pixel magnification);
        # the helper broadcasts it.
        weight_scale = (delta_voxel_row * gp.delta_voxel) / footprint_xy
        sinogram_view_T = horizontal_fan_project(n_p, n_p_centers, W_p_c, weight_scale,
                                                 voxel_values, num_det_channels, gp.psf_radius,
                                                 use_sorted=projector_params.sort_by_channel)
        return sinogram_view_T.T

    @staticmethod
    def forward_vertical_fan_one_pixel_to_one_view(voxel_cylinder, pixel_index, single_view_params, projector_params):
        """
        Apply a fan beam forward projection in the vertical direction to the pixel determined by indices
        into the flattened array of size num_rows x num_cols.  This returns a vector obtained from the projection of
        the original voxel cylinder onto a detector column, so the output vector has length num_det_rows.

        Args:
            voxel_cylinder (jax array):  1D array of shape (num_recon_slices, ) of voxel values, where
                voxel_cylinder[j] is the value of the voxel in slice j at the location determined by pixel_index.
            pixel_index (int):  Index into the flattened array of size num_rows x num_cols.
            single_view_params: These are the angle and helical_z_shift for the view being forward projected.
            projector_params (namedtuple):  tuple of (sinogram_shape, recon_shape, get_geometry_params())

        Returns:
            jax array of shape (num_det_rows,)

        Note:
            This is a helper function used in vmap in :meth:`ConeBeamModel.forward_vertical_fan_pixel_batch_to_one_view`
        This method has the same signature and output as that method, except single int pixel_index is used
        in place of the 1D pixel_indices, and likewise only a single voxel cylinder is returned.
        """
        angle = single_view_params[0]
        helical_z_shift = single_view_params[1]
        
        # Get all the geometry parameters - we use gp since geometry parameters is a named tuple and we'll access
        # elements using, for example, gp.delta_det_channel, so a longer name would be clumsy.
        gp = projector_params.geometry_params
        num_views, num_det_rows, num_det_channels = projector_params.sinogram_shape
        recon_shape = projector_params.recon_shape
        num_slices = voxel_cylinder.shape[0]
        
        delta_voxel_slice = gp.voxel_slice_aspect * gp.delta_voxel
        
        # From pixel index, compute y and pixel_mag
        y, pixel_mag = ConeBeamModel.compute_y_mag_for_pixel(pixel_index, angle, recon_shape, projector_params)

        # The code above depends only on the pixel - a single point.  z is a potentially large vector
        # Here we compute cos_phi_p:  1 / cos_phi_p determines the projection length through a voxel
        # For computational efficiency, we use that to scale the voxel_cylinder values.
        # TODO:  possibly convert to a jitted function with donate_argnames to avoid copies for z, v, phi_p, cos_phi_p
        k = jnp.arange(len(voxel_cylinder))
        z = delta_voxel_slice * (k - (num_slices - 1) / 2.0) + (gp.recon_slice_offset - helical_z_shift)  # recon_ijk_to_xyz
        v = pixel_mag * z  # geometry_xyz_to_uv_mag
        # Compute vertical cone angle of voxels
        phi_p = jnp.arctan2(v, gp.source_detector_dist)  # compute_vertical_data_single_pixel
        cos_phi_p = jnp.cos(phi_p)  # We assume the vertical angle |phi_p| < 45 degrees so cos_alpha_p_z = cos_phi_p
        scaled_voxel_values = voxel_cylinder / cos_phi_p
        # End TODO

        # Get the length of projection of detector on vertical voxel profile (in fraction of voxel size)
        # This is also the slope of the map from voxel index to detector index
        W_p_r = (pixel_mag * delta_voxel_slice) / gp.delta_det_row
        slope_k_to_m = W_p_r
        L_max = jnp.minimum(1, W_p_r)  # Maximum fraction of a detector that can be covered by one voxel.

        # Set up detector row indices array (0, 10, 20, ..., 10*num_slice_batches)
        det_rows_per_batch = CONE_FORWARD_DET_ROW_BATCH
        det_rows_per_batch = min(det_rows_per_batch, num_det_rows)
        num_det_row_batches = (num_det_rows + det_rows_per_batch - 1) // det_rows_per_batch
        det_row_indices = det_rows_per_batch * jnp.arange(num_det_row_batches)

        det_center_row = (num_det_rows - 1) / 2.0
        row_batch = jnp.arange(det_rows_per_batch)

        # Set up a function to map over subsets of detector rows
        def create_det_column_rows(start_index):
            # We need to match the back projector, so we have to determine the fraction of each voxel that projects
            # to each detector.
            # First project the detector centers to the voxel cylinder
            m_center = start_index + row_batch  # Center of detector elements
            v_m = (m_center - det_center_row) * gp.delta_det_row - gp.det_row_offset  # Detector center in ALUs
            z_m = v_m / pixel_mag  # z coordinate of the projection of the center of the first detector element in this batch
            # Convert to voxel fractional index and find the center of each voxel
            k_m = (z_m - (gp.recon_slice_offset - helical_z_shift)) / delta_voxel_slice + (num_slices - 1) / 2.0
            k_m_center = jnp.round(k_m).astype(int)  # Center of the voxel hit by the center of the detector
            # Then map the center of the voxels back to the detector.
            m_p = slope_k_to_m * (k_m_center - k_m[0]) + m_center[0]  # Projection to detector of voxel centers

            # Allocate space
            new_column_batch = jnp.zeros(det_rows_per_batch)
            # Do the vertical projection
            for k_offset in jnp.arange(start=-gp.bp_psf_radius, stop=gp.bp_psf_radius+1):
                k_ind = k_m_center + k_offset  # Indices of the current set of voxels touched by the detector elements
                # The projection of these centers is the projection of k_m_center (which is m_p) plus
                # the offset times the slope of the map from voxel index to detector index
                abs_delta_p_r_m = jnp.abs(m_p + slope_k_to_m * k_offset - m_center)  # Distance from projection of center of voxel to center of detector
                A_row_k = jnp.clip((W_p_r + 1) / 2 - abs_delta_p_r_m, 0, L_max)  # Fraction of the detector hit by this voxel
                A_row_k *= (k_ind >= 0) * (k_ind < num_slices)
                new_column_batch = jnp.add(new_column_batch, A_row_k * scaled_voxel_values[k_ind])

            return new_column_batch, None

        det_column, _ = jax.lax.map(create_det_column_rows, det_row_indices)
        det_column = det_column.flatten()
        det_column = jax.lax.slice_in_dim(det_column, 0, num_det_rows)
        return det_column

    @staticmethod
    @partial(jax.jit, static_argnames=['projector_params', 'slice_band_size'])
    def back_project_one_view_to_pixel_batch(sinogram_view, pixel_indices, single_view_params, n_p_centers,
                                             projector_params, coeff_power=1, slice_band_size=None):
        """
        Back project one view to multiple pixels (voxel cylinders).

        The horizontal fan is computed ONCE per view (it does not depend on the recon
        slice), then the recon slice axis is walked in fixed-size bands with a ROLLED loop
        (jax.lax.map): the band body compiles once and iterates at runtime, so the
        compiled program does not grow with the slice count.  Each band back-projects onto
        its block of global slices via the banded vertical fan; the bands are stacked and
        cropped to the real slice count (the last band is padded up to a whole band and the
        banded kernel's global validity clip zeros that padded tail, so the crop is exact).

        Args:
            sinogram_view (2D jax array): one view of the sinogram to be back projected.
                2D jax array of shape (num_det_rows)x(num_det_channels)
            pixel_indices (1D jax array of int):  indices into flattened array of size num_rows x num_cols.
            single_view_params: These are the view dependent parameters for the view being back projected.
            projector_params (namedtuple): tuple of (sinogram_shape, recon_shape, get_geometry_params()).
            coeff_power (int): backproject using the coefficients of (A_ij ** coeff_power).
                Normally 1, but should be 2 when computing Hessian diagonal.
            slice_band_size (int or None): number of recon slices per band (the memory knob).
                None uses the module default CONE_SLICE_BAND_SIZE; tests pass a small value to
                exercise the multi-band assembly on a small geometry.  Static (sets shapes).

        Returns:
            The voxel values for all slices at the input index (i.e., a voxel cylinder) obtained by backprojecting
            the input sinogram view.  Shape (len(pixel_indices), num_recon_slices).
        """
        num_recon_slices = projector_params.recon_shape[2]
        num_pixels = pixel_indices.shape[0]

        # Horizontal fan once per view -> detector-based voxel cylinder (num_pixels, num_det_rows).
        # named_scope: stable, code-localized profiling regions (see experiments/profiling).
        with jax.named_scope("cone/back/pixel/horizontal_fan"):
            det_voxel_cylinder = ConeBeamModel.back_horizontal_fan_one_view_to_pixel_batch(
                sinogram_view, pixel_indices, single_view_params, n_p_centers, projector_params,
                coeff_power=coeff_power)

        # Tile the slice axis into uniform bands; jax.lax.map needs equal-shape iterations, so
        # the last band runs past num_recon_slices and is cropped off below.
        band_size = CONE_SLICE_BAND_SIZE if slice_band_size is None else slice_band_size
        band_size = min(band_size, num_recon_slices)
        num_bands = (num_recon_slices + band_size - 1) // band_size
        band_starts = band_size * jnp.arange(num_bands)        # g0 for each band (mapped over)

        def back_one_band(g0):
            # (num_pixels, band_size) back projection onto global slices [g0, g0 + band_size).
            return ConeBeamModel.back_vertical_fan_band_pixel_batch(
                det_voxel_cylinder, pixel_indices, single_view_params, projector_params,
                g0, band_size, coeff_power=coeff_power)

        with jax.named_scope("cone/back/pixel/vertical_fan"):
            bands = jax.lax.map(back_one_band, band_starts)    # (num_bands, num_pixels, band_size)
        # Reassemble into (num_pixels, num_bands * band_size): for pixel p and band b, slice
        # b*band_size + l is bands[b, p, l].  Then crop the padded tail back to the real count.  This
        # transpose-reassembly is UNIQUE to the pixel kernel (the band kernel returns one band) -> its
        # own region so its cost (vs the band kernel) is visible.
        with jax.named_scope("cone/back/pixel/assemble"):
            back_projection = jnp.transpose(bands, (1, 0, 2)).reshape(num_pixels, num_bands * band_size)
            back_projection = jax.lax.slice_in_dim(back_projection, 0, num_recon_slices, axis=1)

        return back_projection

    @staticmethod
    def back_horizontal_fan_one_view_to_pixel_batch(sinogram_view, pixel_indices, single_view_params,
                                                    n_p_centers, projector_params, coeff_power=1):
        """
        Apply the back projection of a horizontal fan beam transformation to a single sinogram view
        and return the resulting voxel cylinders.

        Args:
            sinogram_view (2D jax array): one view of the sinogram to be back projected.
                2D jax array of shape (num_det_rows)x(num_det_channels)
            pixel_indices (1D jax array of int):  indices into flattened array of size num_rows x num_cols.
            single_view_params: These are the angle and helical_z_shift for the view being back projected.
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
        n_p, W_p_c, footprint_xy = ConeBeamModel.compute_horizontal_data(pixel_indices, single_view_params, projector_params)
        delta_voxel_row = gp.voxel_row_aspect * gp.delta_voxel

        # The shared adjoint trapezoid-rule fan (see horizontal_fan_back in projectors.py,
        # which owns the channel-major layout rationale and the back_stacked_gather branch
        # -- deliberately NOT enabled by cone's policy: measured a 1.00x composition no-op,
        # the gather hides behind the vertical-fan band work).  The per-pixel weight scale
        # mirrors the forward fan.
        sinogram_view_T = sinogram_view.T            # (num_det_channels, num_det_rows)
        weight_scale = (delta_voxel_row * gp.delta_voxel) / footprint_xy
        return horizontal_fan_back(sinogram_view_T, n_p, n_p_centers, W_p_c, weight_scale,
                                   gp.psf_radius, coeff_power=coeff_power,
                                   use_stacked=projector_params.back_stacked_gather)

    # ──────────────────────────────────────────────────────────────────────────
    # Banded vertical fans + per-view banded kernels.  The shared contract (global-index
    # z anchoring / tiling-invariance, inert padding, single-device band walk + the
    # multi-device per-band entry) is documented ONCE at the canonical note above
    # projectors.vertical_fan_band_gather.
    #
    # Cone-specific: the FORWARD projection does NOT band -- a band of input slices reaches
    # a wide, mostly-empty window of detector rows, so a banded forward recomputes the
    # dominant horizontal stage per band for little memory gain (measured: 5-14x slower for
    # ~13-23% transient memory).  The multi-device forward instead gathers the slice
    # cylinder per pixel-batch and runs the monolithic forward.
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def back_vertical_fan_band_one_pixel(detector_column_values, pixel_index, single_view_params,
                                         projector_params, g0, num_band_slices, coeff_power=1):
        """Back vertical fan for one pixel, producing a BAND of output slices.

        Back-projects this pixel's detector column onto the GLOBAL recon slices
        [g0, g0+L) (output length L = num_band_slices) -- "band" names the output
        slice range.  compute_vertical_data_single_pixel already anchors z on the
        problem's recon_shape, so this just passes the global band indices; padded
        global slices (index >= the real slice count) are zeroed.  Concatenating the
        per-band outputs over a tiling of the slices reconstructs the monolithic
        vertical fan (the bands tile disjointly).
        """
        # Get the geometry parameters; short name gp since we access many fields.
        gp = projector_params.geometry_params
        num_views, num_det_rows, num_det_channels = projector_params.sinogram_shape
        num_recon_slices = projector_params.recon_shape[2]

        slice_indices = g0 + jnp.arange(num_band_slices)            # GLOBAL band indices
        m_p, m_p_center, W_p_r, cos_alpha_p_z = ConeBeamModel.compute_vertical_data_single_pixel(
            pixel_index, slice_indices, single_view_params, projector_params)
        # The shared banded vertical-fan gather (see vertical_fan_band_gather in
        # projectors.py; translation shares it verbatim -- weights L / cos_alpha, padded
        # slices zeroed).
        return vertical_fan_band_gather(detector_column_values, slice_indices, m_p,
                                        m_p_center, W_p_r, cos_alpha_p_z, num_det_rows,
                                        num_recon_slices, gp.psf_radius,
                                        coeff_power=coeff_power)

    @staticmethod
    def back_vertical_fan_band_pixel_batch(det_voxel_cylinder, pixel_indices, single_view_params,
                                           projector_params, g0, num_band_slices, coeff_power=1):
        """Vmap the banded back vertical fan over a pixel batch.

        ``det_voxel_cylinder`` is (num_pixels, num_det_rows); returns
        (num_pixels, num_band_slices) for global slices [g0, g0+L)."""
        pixel_map = jax.vmap(ConeBeamModel.back_vertical_fan_band_one_pixel,
                             in_axes=(0, 0, None, None, None, None, None))
        return pixel_map(det_voxel_cylinder, pixel_indices, single_view_params, projector_params,
                         g0, num_band_slices, coeff_power)

    @staticmethod
    @partial(jax.jit, static_argnames=['projector_params', 'num_band_slices'])
    def back_project_one_view_to_band(sinogram_view, pixel_indices, single_view_params, n_p_centers,
                                      projector_params, g0, num_band_slices, coeff_power=1):
        """Banded back projection of one view onto GLOBAL recon slices [g0, g0+L).

        Horizontal fan once (full det rows) -> banded vertical fan.  Returns
        (num_pixels, num_band_slices).  ``g0`` is traced; ``num_band_slices`` (= L)
        is static."""
        # named_scope tags the HLO/trace with a stable, code-localized region name (the vertical fan
        # is where the L1-bound transpose lives; see experiments/profiling/key_findings.md).
        with jax.named_scope("cone/back/band/horizontal_fan"):
            det_voxel_cylinder = ConeBeamModel.back_horizontal_fan_one_view_to_pixel_batch(
                sinogram_view, pixel_indices, single_view_params, n_p_centers, projector_params,
                coeff_power=coeff_power)
        with jax.named_scope("cone/back/band/vertical_fan"):
            return ConeBeamModel.back_vertical_fan_band_pixel_batch(
                det_voxel_cylinder, pixel_indices, single_view_params, projector_params,
                g0, num_band_slices, coeff_power=coeff_power)

    @staticmethod
    def compute_vertical_data_single_pixel(pixel_index, slice_indices, single_view_params, projector_params):
        """
        Compute the quantities m_p, m_p_center, W_p_r, cos_alpha_p_z needed for vertical projection.

        Args:
            pixel_index (int):  Index into flattened array of size num_rows x num_cols.
            slice_indices (array of int): Indices into the recon slices.
            single_view_params: These are the angle and helical_z_shift for the view being back projected.
            projector_params (namedtuple): tuple of (sinogram_shape, recon_shape, get_geometry_params()).

        Returns:
            m_p, m_p_center, W_p_r, cos_alpha_p_z
        """
        angle = single_view_params[0]
        helical_z_shift = single_view_params[1]
        
        # Get all the geometry parameters - we use gp since geometry parameters is a named tuple and we'll access
        # elements using, for example, gp.delta_det_channel, so a longer name would be clumsy.
        gp = projector_params.geometry_params

        num_views, num_det_rows, num_det_channels = projector_params.sinogram_shape
        recon_shape = projector_params.recon_shape
        num_recon_rows, num_recon_cols, num_recon_slices = recon_shape
        
        delta_voxel_slice = gp.voxel_slice_aspect * gp.delta_voxel

        # Convert the index into (i,j,k) coordinates corresponding to the indices into the 3D voxel array
        row_index, col_index = jnp.unravel_index(pixel_index, recon_shape[:2])
        # slice_indices = jnp.arange(num_recon_slices)

        x_p, y_p, z_p = ConeBeamModel.recon_ijk_to_xyz(row_index, col_index, slice_indices, gp.delta_voxel, gp.voxel_row_aspect,
                                                       gp.voxel_slice_aspect, recon_shape, gp.recon_slice_offset - helical_z_shift, angle)

        # Convert from xyz to coordinates on detector
        u_p, v_p, pixel_mag = ConeBeamModel.geometry_xyz_to_uv_mag(x_p, y_p, z_p, gp.source_detector_dist, gp.magnification,
                                                                   gp.use_curved_detector)
        # Convert from uv to index coordinates in detector and get the vector of center detector rows for this cylinder
        m_p, _ = ConeBeamModel.detector_uv_to_mn(u_p, v_p, gp.delta_det_channel, gp.delta_det_row, gp.det_channel_offset,
                                                 gp.det_row_offset, num_det_rows, num_det_channels)
        m_p_center = jnp.round(m_p).astype(int)

        # Compute vertical cone angle of pixel
        phi_p = jnp.arctan2(v_p, gp.source_detector_dist)

        # Compute cos alpha for row and columns
        cos_phi_p = jnp.cos(phi_p)  # We assume the vertical angle |phi_p| < 45 degrees so cos_alpha_p_z = cos_phi_p
        # cos_alpha_p_z = jnp.maximum(jnp.abs(cos_phi_p), jnp.abs(jnp.sin(phi_p)))

        # Get the length of projection of flattened voxel on detector (in fraction of detector size)
        W_p_r = pixel_mag * (delta_voxel_slice / gp.delta_det_row)   # * cos_alpha_p_z / cos_phi_p

        vertical_data = (m_p, m_p_center, W_p_r, cos_phi_p)  # cos_alpha_p_z)

        return vertical_data

    # GRADIENT only: the fused kernel's in-kernel affine-m carries a ~2e-5 Hessian
    # (squared-weight) error that VCD's grad/Hess division amplifies at low-Hessian
    # edge voxels -- the inc5 trajectory gate measured 8.5e-3 recon divergence with
    # cp=2 through the kernel.  The once-per-recon Hessian keeps the XLA path.
    _PALLAS_BACK_COEFF_POWERS = (1,)

    @staticmethod
    def compute_hfan_data(pixel_indices, single_view_params, projector_params):
        """(n_p, W_p_c, weight_scale) for the shared trapezoid weight builders
        (_pallas_kernels._jit_compute_back_weights): cone's horizontal-fan data with
        the weight scale of back_horizontal_fan_one_view_to_pixel_batch."""
        gp = projector_params.geometry_params
        n_p, W_p_c, footprint_xy = ConeBeamModel.compute_horizontal_data(
            pixel_indices, single_view_params, projector_params)
        weight_scale = (gp.voxel_row_aspect * gp.delta_voxel * gp.delta_voxel) / footprint_xy
        return n_p, W_p_c, weight_scale

    def _pallas_back_project_single_device(self, sinogram, pixel_indices,
                                           coeff_power=1, output_device=None):
        """Cone's pallas n=1 back driver: the fused-vfan kernel over the full slice
        range (see _pallas_kernels.cone_back_project_single_device)."""
        from mbirjax import _pallas_kernels
        return _pallas_kernels.cone_back_project_single_device(
            self, sinogram, pixel_indices, coeff_power=coeff_power,
            output_device=output_device)

    def _back_project_view_shard_to_band(self, view_data, pixel_indices, g0, g1,
                                         owned_view_indices, coeff_power):
        """Cone's per-owner banded back: the fused-vfan pallas kernel when enabled
        (full-row views -- a cone slice draws from a RANGE of rows, so no crop),
        else the base XLA banded path."""
        if (getattr(self.tiles, 'back_pallas_band', False)
                and coeff_power in self._PALLAS_BACK_COEFF_POWERS):
            from mbirjax import _pallas_kernels
            return _pallas_kernels.cone_back_project_band(
                self, view_data, pixel_indices, g0, g1 - g0,
                owned_view_indices=owned_view_indices, coeff_power=coeff_power)
        return super()._back_project_view_shard_to_band(
            view_data, pixel_indices, g0, g1, owned_view_indices, coeff_power)

    @staticmethod
    def compute_channel_coordinate(pixel_indices, single_view_params, projector_params):
        """Continuous projected channel coordinate n_p for one view (the float chain whose
        rounded value is the horizontal fans' integer center; see the base-class docstring
        for the concrete-input contract).  Reuses compute_horizontal_data verbatim
        so the centers are consistent with the in-kernel weights by construction."""
        return ConeBeamModel.compute_horizontal_data(pixel_indices, single_view_params,
                                                     projector_params)[0]

    @staticmethod
    def compute_horizontal_data(pixel_indices, single_view_params, projector_params):
        """
        Compute the quantities n_p, W_p_c, footprint_xy needed for horizontal projection.
        (The integer center round(n_p) is deliberately NOT computed here -- see
        compute_channel_coordinate.)

        Behavior differs by detector type:

        - **Flat** (use_curved_detector=False): Uses theta_p = arctan2(u_p, source_detector_dist) and
          includes a 1/cos(theta_p) foreshortening correction in W_p_c.
        - **Curved** (use_curved_detector=True): Uses theta_p = u_p / source_detector_dist
          (arc-length parameterisation) and omits the 1/cos(theta_p) term from W_p_c.

        Args:
            pixel_indices (1D jax array of int):  indices into flattened array of size num_rows x num_cols.
            single_view_params: These are the angle and helical_z_shift for the view being back projected.
            projector_params (namedtuple): tuple of (sinogram_shape, recon_shape, get_geometry_params()).

        Returns:
            n_p, W_p_c, footprint_xy
        """
        angle = single_view_params[0]
        helical_z_shift = single_view_params[1]
        
        # Get all the geometry parameters - we use gp since geometry parameters is a named tuple and we'll access
        # elements using, for example, gp.delta_det_channel, so a longer name would be clumsy.
        gp = projector_params.geometry_params

        num_views, num_det_rows, num_det_channels = projector_params.sinogram_shape
        recon_shape = projector_params.recon_shape
        num_recon_rows, num_recon_cols, num_recon_slices = recon_shape
        
        delta_voxel_row = gp.voxel_row_aspect * gp.delta_voxel

        # Convert the index into (i,j,k) coordinates corresponding to the indices into the 3D voxel array
        row_index, col_index = jnp.unravel_index(pixel_indices, recon_shape[:2])
        slice_index = jnp.arange(1)

        x_p, y_p, z_p = ConeBeamModel.recon_ijk_to_xyz(row_index, col_index, slice_index, gp.delta_voxel, gp.voxel_row_aspect,
                                                       gp.voxel_slice_aspect, recon_shape, gp.recon_slice_offset - helical_z_shift, angle)

        # Convert from xyz to coordinates on detector
        # pixel_mag should be kept in terms of magnification to allow for source_detector_dist = jnp.Inf
        u_p, v_p, pixel_mag = ConeBeamModel.geometry_xyz_to_uv_mag(x_p, y_p, z_p, gp.source_detector_dist, gp.magnification,
                                                                   gp.use_curved_detector)

        det_center_channel = (num_det_channels - 1) / 2.0  # num_of_cols

        # Calculate indices on the detector grid
        n_p = (u_p + gp.det_channel_offset) / gp.delta_det_channel + det_center_channel  # Sync with detector_uv_to_mn
        # NOTE: no round here -- the integer center is computed outside the projector
        # programs (see compute_channel_coordinate).

        # theta_p and W_p_c computation differ between flat and curved detectors
        if not gp.use_curved_detector: # 'flat'
            # Exact horizontal cone angle for flat detector
            theta_p = jnp.arctan2(u_p, gp.source_detector_dist)
            # Compute footprint for row and columns
            footprint_xy = jnp.maximum(jnp.abs(jnp.cos(angle - theta_p)) * gp.delta_voxel, jnp.abs(jnp.sin(angle - theta_p)) * delta_voxel_row)
            # Foreshortening correction: voxel footprint widens at oblique angles on a flat detector
            W_p_c = pixel_mag * (footprint_xy / gp.delta_det_channel) / jnp.cos(theta_p)
        else:  # 'curved'
            # u_p is arc-length, so effective angle is u_p / sdd
            theta_p = u_p / gp.source_detector_dist
            # Compute footprint for row and columns
            footprint_xy = jnp.maximum(jnp.abs(jnp.cos(angle - theta_p)) * gp.delta_voxel, jnp.abs(jnp.sin(angle - theta_p)) * delta_voxel_row)
            # No cos(theta_p) denominator: arc parameterisation absorbs the foreshortening
            W_p_c = pixel_mag * (footprint_xy / gp.delta_det_channel)

        horizontal_data = (n_p, W_p_c, footprint_xy)

        return horizontal_data

    @staticmethod
    def recon_ijk_to_xyz(i, j, k, delta_voxel, voxel_row_aspect, voxel_slice_aspect, recon_shape, recon_slice_offset, angle):
        """
        Convert (i, j, k) indices into the recon volume to corresponding (x, y, z) coordinates.
        """
        num_recon_rows, num_recon_cols, num_recon_slices = recon_shape
        
        delta_voxel_row = voxel_row_aspect * delta_voxel
        delta_voxel_slice = voxel_slice_aspect * delta_voxel

        # Compute the un-rotated coordinates relative to iso
        # Note the change in order from (i, j) to (y, x)!!
        y_tilde = delta_voxel_row * (i - (num_recon_rows - 1) / 2.0)
        x_tilde = delta_voxel * (j - (num_recon_cols - 1) / 2.0)

        # Precompute cosine and sine of view angle, then do the rotation
        cosine = jnp.cos(angle)  # length = num_views
        sine = jnp.sin(angle)  # length = num_views

        x = cosine * x_tilde - sine * y_tilde
        y = sine * x_tilde + cosine * y_tilde

        z = delta_voxel_slice * (k - (num_recon_slices - 1) / 2.0) + recon_slice_offset
        return x, y, z

    @staticmethod
    def geometry_xyz_to_uv_mag(x, y, z, source_detector_dist, magnification, use_curved_detector=False):
        """
        Convert (x, y, z) coordinates to (u, v) detector coordinates plus the pixel-dependent magnification.

        Behavior differs by detector type:
        - **Flat** (use_curved_detector=False): u = pixel_mag * x (linear perspective projection).
        - **Curved** (use_curved_detector=True): u = source_detector_dist * arctan2(x, source_iso_dist - y)
          (arc-length on a cylinder of radius source_detector_dist).
        """
        # Compute the magnification at this specific voxel
        # The following expression is valid even when source_detector_dist = jnp.Inf
        pixel_mag = 1 / (1 / magnification - y / source_detector_dist)

        # u computation depends on use_curved_detector
        if not use_curved_detector:  # 'flat'
            # Standard flat-panel
            u = pixel_mag * x
        else:  # 'curved'
            # u is arc-length on a cylinder of radius source_detector_dist
            source_iso_dist = source_detector_dist / magnification
            u = source_detector_dist * jnp.arctan2(x, (source_iso_dist - y))

        v = pixel_mag * z

        return u, v, pixel_mag

    @staticmethod
    @jax.jit
    def detector_uv_to_mn(u, v, delta_det_channel, delta_det_row, det_channel_offset, det_row_offset, num_det_rows,
                          num_det_channels):
        """
        Convert (u, v) detector coordinates to fractional indices (m, n) into the detector.

        Note:
            This version does not account for nonzero detector rotation.
        """
        # Account for small rotation of the detector
        # TODO:  In addition to including the rotation, we'd need to adjust the calculation of the channel as a
        #  function of slice.
        u_tilde = u  # jnp.cos(det_rotation) * u + jnp.sin(det_rotation) * v
        v_tilde = v  # -jnp.sin(det_rotation) * u + jnp.cos(det_rotation) * v

        # Get the center of the detector grid for columns and rows
        det_center_row = (num_det_rows - 1) / 2.0  # num_of_rows
        det_center_channel = (num_det_channels - 1) / 2.0  # num_of_cols

        # Calculate indices on the detector grid
        m = (v_tilde + det_row_offset) / delta_det_row + det_center_row
        n = (u_tilde + det_channel_offset) / delta_det_channel + det_center_channel  # Sync with compute_horizontal_data

        return m, n

    @staticmethod
    @jax.jit
    def detector_mn_to_uv(m, n, delta_det_channel, delta_det_row, det_channel_offset, det_row_offset, num_det_rows,
                      num_det_channels):
        """
        Convert fractional detector grid indices (m, n) into detector coordinates (u, v).

        Parameters:
            m: Fractional row index on the detector grid (vertical direction).
            n: Fractional channel index on the detector grid (horizontal direction).
            delta_det_channel: Spacing (pitch) of the detector channels (horizontal direction).
            delta_det_row: Spacing (pitch) of the detector rows (vertical direction).
            det_channel_offset: Offset in the detector channel (horizontal) direction.
            det_row_offset: Offset in the detector row (vertical) direction.
            num_det_rows: Total number of rows in the detector.
            num_det_channels: Total number of channels in the detector.

        Returns:
            u: Physical detector coordinate in the channel direction.
            v: Physical detector coordinate in the row direction.
        """
        # Calculate the center of the detector grid
        det_center_row = (num_det_rows - 1) / 2.0
        det_center_channel = (num_det_channels - 1) / 2.0

        # Compute detector coordinates (u, v)
        v = (m - det_center_row) * delta_det_row - det_row_offset
        u = (n - det_center_channel) * delta_det_channel - det_channel_offset

        return u, v

    @staticmethod
    @jax.jit
    def compute_y_mag_for_pixel(pixel_index, angle, recon_shape, projector_params):

        gp = projector_params.geometry_params
        row_index, col_index = jnp.unravel_index(pixel_index, recon_shape[:2])
        
        delta_voxel_row = gp.voxel_row_aspect * gp.delta_voxel

        # Compute the un-rotated coordinates relative to iso
        # Note the change in order from (i, j) to (y, x)!!
        y_tilde = delta_voxel_row * (row_index - (recon_shape[0] - 1) / 2.0)
        x_tilde = gp.delta_voxel * (col_index - (recon_shape[1] - 1) / 2.0)

        # Precompute cosine and sine of view angle, then do the rotation
        cosine = jnp.cos(angle)  # length = num_views
        sine = jnp.sin(angle)  # length = num_views

        y = sine * x_tilde + cosine * y_tilde

        # Convert from xyz to coordinates on detector
        pixel_mag = 1 / (1 / gp.magnification - y / gp.source_detector_dist)
        return y, pixel_mag

    def direct_recon(self, sinogram, filter_name="ramp", output_sharded=False):
        return self.fdk_recon(sinogram, filter_name=filter_name, output_sharded=output_sharded)

    def direct_filter(self, sinogram, filter_name="ramp", output_sharded=False):
        """
        Perform filtering on the given sinogram as needed for an FBP/FDK or other direct recon.

        Args:
            sinogram (numpy or jax array): The input sinogram with shape (num_views, num_rows, num_channels).
            filter_name (string, optional): Name of the filter to be used. Defaults to "ramp"
            output_sharded (bool, optional): If False (default), return a numpy array.  If True,
                return the view-sharded device form (on an unsharded model the output is the same
                either way).

        Returns:
            filtered_sinogram (numpy or jax array): The sinogram after FDK filtering -- numpy by default,
            view-sharded if output_sharded=True.
        """
        return self.fdk_filter(sinogram, filter_name=filter_name, output_sharded=output_sharded)

    def fdk_filter(self, sinogram, filter_name="ramp", output_sharded=False):
        """
        Perform FDK filtering on the given sinogram.

        Uses the shared row-batched filter kernel (tomography_utils.apply_row_filter via
        TomographyModel._apply_direct_recon_filter), the same path as
        ParallelBeamModel.fbp_filter, with an FDK cosine pre-weight applied per detector
        row.  The peak stays at the input+output floor (no full-sinogram out-of-place
        copy or concatenate), and when a mesh is configured each device filters its own
        view-shard locally.

        Args:
            sinogram (numpy or jax array): The input sinogram with shape (num_views, num_rows, num_channels).
            filter_name (string, optional): Name of the filter to be used. Defaults to "ramp"
            output_sharded (bool, optional): If False (default), return a numpy array.  If True,
                return the view-sharded device form (on an unsharded model the output is the same
                either way).

        Returns:
            filtered_sinogram (numpy or jax array): The sinogram after FDK filtering -- numpy by default,
            view-sharded if output_sharded=True.
        """
        # Detector geometry + voxel scaling -- the only FDK-specific pieces; the shared
        # row filter does the rest (bounded peak, jitted, sharded per view-shard).
        num_rows, num_channels = sinogram.shape[1], sinogram.shape[2]
        source_detector_dist = self.get_params('source_detector_dist')
        delta_voxel, delta_det_row, delta_det_channel, voxel_row_aspect, voxel_slice_aspect = self.get_params(
            ['delta_voxel', 'delta_det_row', 'delta_det_channel', 'voxel_row_aspect', 'voxel_slice_aspect'])
        det_row_offset, det_channel_offset = self.get_params(['det_row_offset', 'det_channel_offset'])
        delta_voxel_row = voxel_row_aspect * delta_voxel
        delta_voxel_slice = voxel_slice_aspect * delta_voxel
        voxel_volume = delta_voxel * delta_voxel_row * delta_voxel_slice
        # Magnification M_0 = source-detector distance / source-isocenter distance.
        M_0 = self.get_magnification()

        # FDK cosine pre-weight (rows, channels), view-INDEPENDENT.  Passed to the shared
        # kernel, which applies it per detector row inside the bounded scan (no full
        # out-of-place sinogram * weight copy).
        m_grid, n_grid = jnp.meshgrid(jnp.arange(num_rows), jnp.arange(num_channels), indexing='ij')
        u_grid, v_grid = self.detector_mn_to_uv(m_grid, n_grid, delta_det_channel, delta_det_row,
                                                det_channel_offset, det_row_offset, num_rows, num_channels)
        weight_map = source_detector_dist / jnp.sqrt(source_detector_dist ** 2 + u_grid ** 2 + v_grid ** 2)

        # FDK filter scale alpha (voxel-size factor); pi/num_views is folded in by the
        # shared method.  For the derivation see the theory zip at
        # https://mbirjax.readthedocs.io/en/latest/theory.html
        alpha = delta_det_row / (voxel_volume * M_0)
        return self._apply_direct_recon_filter(
            sinogram, filter_name, filter_scale=alpha,
            output_sharded=output_sharded, row_weight=weight_map)

    def helical_fdk_z_weight(self, recon, sinogram):
        """
        Scale each helical FDK slice by the inverse of the fraction of the scan that the slice is in view of the detector.

        Args:
            recon (jax array): The initial FDK reconstruction.
            sinogram (jax array): The input sinogram with shape (num_views, num_rows, num_channels).

        Returns:
            recon (jax array): The helical FDK reconstruction with correctly weighted slice intensity.
        """
        
        # Real sinogram shape from params: the passed ``sinogram`` may be view-sharded and
        # zero-padded, so num_views must be the real count for the coverage weight to be
        # correct.  (sinogram is otherwise unused here -- the weighting is per recon slice.)
        num_views, num_rows, num_channels = self.get_params('sinogram_shape')
        view_params_array = self.get_params('view_params_array')
        helical_z_shifts = view_params_array[:, 1]
        delta_voxel, voxel_slice_aspect, recon_shape, recon_slice_offset, delta_det_row = self.get_params(
            ['delta_voxel', 'voxel_slice_aspect', 'recon_shape', 'recon_slice_offset', 'delta_det_row']
        )
        M_0 = self.get_magnification()
        delta_voxel_slice = voxel_slice_aspect * delta_voxel

        # The slice->z map anchors on the REAL slice count (recon_shape[2], always the problem
        # size), but the weight is applied to the recon AS IT EXISTS ON THE DEVICES, whose slice
        # axis is zero-padded to a longer length when sharding does not divide evenly.  So build
        # the weight over the recon's device-form slice count while keeping the z anchor on the
        # real count, and force the weight to zero on the padded slices: they are identically
        # zero (the forced-zero invariant) and a padded slice is out of coverage (coverage 0 ->
        # num_views/0 = inf), so without this guard 0 * inf would make them NaN.  A no-op when
        # nothing is padded (device-form length == real length, k < num_real_slices everywhere).
        num_real_slices = recon_shape[2]
        num_device_slices = recon.shape[2]

        k = jnp.arange(num_device_slices)
        z_k = delta_voxel_slice * (k - (num_real_slices - 1) / 2.0) + recon_slice_offset
        det_half_height_iso = 0.5 * num_rows * delta_det_row / M_0
        visible = jnp.abs(z_k[:, None] - helical_z_shifts[None, :]) <= det_half_height_iso
        coverage = jnp.sum(visible, axis=1)
        # REAL slices can also have zero coverage: auto_set_recon_geometry extends the slab past
        # the central-ray height H_iso/2 (to the cone visibility bound), and this weight's
        # visibility criterion is the central-ray one -- so the extension slices at the travel
        # ends are grazed only by diverging edge rays and count as out of view here.  FDK has
        # essentially no data for them, so the honest initialization is 0 (the VCD iterations
        # fill them from the grazing rays and the prior); jnp.maximum keeps the discarded branch
        # of the where finite.
        z_weight = jnp.where(coverage > 0, num_views / jnp.maximum(coverage, 1), 0.0)
        # Padded device-form slices (k >= num_real_slices) stay zero as well: they are
        # identically zero by the forced-zero invariant and must remain so.
        z_weight = jnp.where(k < num_real_slices, z_weight, 0.0)
        recon = recon * z_weight[None, None, :]

        return recon

    def fdk_recon(self, sinogram, filter_name="ramp", output_sharded=False):
        """
        Perform FDK reconstruction on the given sinogram.

        Our implementation uses standard filtering of the sinogram, then uses the adjoint of the forward projector to
        perform the backprojection.  This is different from many implementations, in which the backprojection is not
        exactly the adjoint of the forward projection.  For a detailed theoretical derivation of this implementation,
        see the zip file linked at this page: https://mbirjax.readthedocs.io/en/latest/theory.html

        Note:
            FDK assumes the view angles are EQUALLY SPACED over the full angular range (the
            ``pi / num_views`` angular weight in the ramp filter), and does not apply short-scan
            (Parker-style) redundancy weighting; on nonuniformly-spaced, limited-angle, or short
            scans it is only approximate.  For helical scans it is approximate regardless.  FDK is
            best used as an initializer for the iterative ``recon()``.

        Args:
            sinogram (numpy or jax array): The input sinogram with shape (num_views, num_rows, num_channels).
            filter_name (string, optional): Name of the filter to be used. Defaults to "ramp"
            output_sharded (bool, optional): If False (default), return a numpy array.  If
                True, return the slice-sharded device form (on an unsharded model the output
                is the same either way).

        Returns:
            recon (numpy or jax array): The reconstructed volume after FDK reconstruction -- numpy by
            default, slice-sharded if ``output_sharded=True``.
        """
        # Shard once at entry so the filter receives view-sharded data (a no-op when already
        # view-sharded; a single device is the trivial 1-shard case).  The pipeline then stays
        # on-device throughout
        # -- fdk_filter then back_project, both output_sharded=True (zero host transfer) --
        # exactly like ParallelBeamModel.fbp_recon.
        sinogram = self._shard_sinogram(sinogram)
        filtered_sinogram = self.fdk_filter(sinogram, filter_name=filter_name, output_sharded=True)
        recon = self.back_project(filtered_sinogram, output_sharded=True)

        # Slice-dependent weighting for helical recon: a per-recon-slice multiply (the recon's
        # slice axis), so it works directly on the slice-sharded device form -- each device
        # scales its own slices.  Circular scans (z_range == 0) skip it and stay on-device.
        helical_z_shifts = self.get_params('view_params_array')[:, 1]
        z_range = jnp.max(helical_z_shifts) - jnp.min(helical_z_shifts)
        if z_range > 0:
            from mbirjax._utils import _called_by
            if not _called_by('vcd_recon') and self.get_params('verbose') > 0:
                warnings.warn('Using FDK for helical direct reconstruction. This will produce an approximate reconstruction with artifacts, but is suitable for MBIR initialization.')
            recon = self.helical_fdk_z_weight(recon, sinogram)

        # Single place the output form is decided.
        return recon if output_sharded else self._gather_recon(recon)

    def split_sino_recon(self, sino, weights=None, half_overlap=5, init_recon=None, max_iterations=15, stop_threshold_change_pct=0.2,
                         first_iteration=0, compute_prior_loss=False, logfile_path='~/.mbirjax/logs/recon.log', print_logs=True,
                         align_split_grid=False):
        """
        This function reduces memory usage for cone beam MBIR reconstruction by approximately a factor of 2
        by splitting the detector rows into two overlapping halves, reconstructing each half separately,
        and stitching the reconstructions together.

        The function can be called with the same arguments as TomographyModel.recon(), and it should return a
        reconstruction which is approximately equal to the reconstruction returned by TomographyModel.recon().

        Each half keeps ``half_overlap`` detector rows past the iso row, and its reconstruction
        extends ``half_overlap_recon`` slices past the split, where half_overlap_recon =
        ceil(half_overlap_sino * (1 + R/SID) * rho) + 2 with rho = delta_det_row/(magnification *
        delta_voxel_slice) and R the recon-support radius.  The (1 + R/SID) factor makes every
        slice the kept rows can SEE representable (the cone-divergence bound of
        auto_set_recon_geometry, evaluated at the iso ray); without it each half is axially
        truncated at its extension end, which shows up as alternating stripes at the stitch seam
        on real scans (see plans/flash_remediation/).  The +2 margin covers the voxel footprint
        and the worst-case half-slice misalignment between the sinogram cut row and the recon
        split slice.

        Args:
            sino (jnp.ndarray | np.ndarray): Full sinogram of shape (num_views, num_rows, num_cols).
            weights (jnp.ndarray | np.ndarray, optional): Optional sinogram weights with the same shape as `sino`.
            half_overlap (int): Number of overlapping detector rows past the iso row per half (when
                recon slices are coarser than the iso-mapped rows, the row overlap is scaled up so
                it still spans ``half_overlap`` slices).  The recon overlap is derived from it by
                the geometry formula above.
            init_recon (optional): Same as in the recon method.
            max_iterations (int, optional): Same as in the recon method.
            stop_threshold_change_pct (float, optional): Same as in the recon method.
            first_iteration (int, optional): Same as in the TomographyModel.recon() method.
            compute_prior_loss (bool, optional): Same as in the TomographyModel.recon() method.
            logfile_path (str, optional): Same as in the TomographyModel.recon() method.  The two
                halves' logs are merged into this single file, each under a section header.
            print_logs (bool, optional): Same as in the TomographyModel.recon() method.
            align_split_grid (bool, optional): If True, align the recon split slice with the
                sinogram cut row: first by choosing the cut row (effective only when rho != 1,
                where the row and slice grids are incommensurate), then by shifting the whole
                recon grid by the sub-slice residual (at most half a slice).  Alignment removes
                the seam-stripe driver outright, but the shifted output samples the object at
                z-positions up to delta_voxel_slice/2 away from what recon() would use -- an
                equally valid reconstruction that is NOT registration-identical to recon(), which
                is why it is opt-in.  The applied shift is reported in the returned dictionary
                under 'split_params'.  Defaults to False.

        Returns:
            Tuple[np.ndarray, dict]: the reconstructed volume (numpy array), and a
                metadata dictionary containing recon and model parameters for each
                half, plus 'split_params' (the overlaps and any alignment shift used).

        Raises:
            ValueError: If inputs are missing or shapes are inconsistent, if half_overlap < 2,
                or if the geometry has nonzero helical z-shifts.
            AssertionError: If array dimensions are invalid.

        Example:
            >>> import jax.numpy as jnp
            >>> import mbirjax as mj
            >>> sino = jnp.ones((180, 64, 64))  # (views, rows, cols)
            >>> model = mj.ConeBeamModel(sinogram_shape=sino.shape,
            ...                          angles=jnp.linspace(0, jnp.pi, 180),
            ...                          source_detector_dist=1000.0,
            ...                          source_iso_dist=500.0)
            >>> recon, recon_info = model.split_sino_recon(sino, half_overlap=4)
            >>> # 64 detector-height slices + 2 automatic visibility-extension slices
            >>> # at each end (see auto_set_recon_geometry):
            >>> recon.shape  # Quilted reconstruction volume
            (64, 64, 68)
        """
        # -------- Basic validation --------
        if half_overlap < 2:
            raise ValueError('half_overlap must be >= 2.')
        helical_z_shifts = self.get_params('view_params_array')[:, 1]
        if any(helical_z_shifts != 0):
            raise ValueError('helical_z_shifts must be zero.')
        if sino is None:
            raise ValueError("sino must be provided.")
        if not (hasattr(sino, "ndim") and sino.ndim == 3):
            raise AssertionError("sino must be a 3D array shaped (num_views, num_rows, num_cols).")
        if weights is not None and getattr(weights, "shape", None) != sino.shape:
            raise AssertionError("weights, if provided, must have the same shape as sino.")

        # Operate on the host: split here and let each half's recon re-shard its own half, so the full
        # sinogram is never on the devices at once (the memory saving).  np.asarray is a no-op for a
        # numpy input and gathers a device/sharded input once; the per-half slices below are then
        # cheap host views.
        sino = np.asarray(sino)
        if weights is not None:
            weights = np.asarray(weights)
        if init_recon is not None and isinstance(init_recon, jax.Array):
            # Same host-side treatment as sino/weights, for two reasons.  (1) Slicing a slice-sharded
            # volume along its sharded axis REPLICATES the half onto every device (a full half-recon
            # copy per device) just before the half-recon builds its working set; host slicing keeps
            # only one half's arrays device-resident at a time -- the point of this function.
            # (2) _gather_recon crops any zero-padded slices of a device-form volume; without the crop,
            # the bottom-half slice init_recon[:, :, -num_slices:] would pick up padding zeros instead
            # of the real bottom slices.
            init_recon = self._gather_recon(init_recon)

        # Get parameters for later use
        num_views, full_num_rows, num_cols = sino.shape

        # -------- parameters needed to create top and bottom models --------
        delta_det_row = self.get_params('delta_det_row')
        full_det_row_offset = self.get_params('det_row_offset')
        delta_voxel = self.get_params('delta_voxel')
        voxel_slice_aspect, voxel_row_aspect = self.get_params(['voxel_slice_aspect', 'voxel_row_aspect'])
        delta_voxel_slice = voxel_slice_aspect * delta_voxel
        full_recon_shape = self.get_params('recon_shape')
        full_recon_slice_offset = self.get_params('recon_slice_offset')
        source_iso_dist, use_ror_mask = self.get_params(['source_iso_dist', 'use_ror_mask'])
        magnification = self.get_magnification()

        # Get recon shape parameters
        full_recon_rows, full_recon_cols, full_recon_slices = full_recon_shape

        # -------- Overlaps: detector rows kept past the cut, recon slices kept past the split --------
        # Sino overlap: when recon slices are coarser than the iso-mapped rows, scale the row
        # overlap up so it still spans about half_overlap slices (the knob keeps its meaning in
        # both unit systems).
        delta_detector_row_at_iso = max(delta_det_row / magnification, 1e-12)
        ratio_pixel_to_sino_pitch = delta_voxel_slice / delta_detector_row_at_iso
        if ratio_pixel_to_sino_pitch > 1:
            half_overlap_sino = int(jnp.round(half_overlap * ratio_pixel_to_sino_pitch))
        else:
            half_overlap_sino = half_overlap
        # Recon overlap from geometry (see docstring): every slice the kept rows can SEE must be
        # representable, or each half is axially truncated at its extension end and the seam
        # stripes.  rho = slices seen per kept row at the iso ray; (1 + R/SID) is the
        # cone-divergence bound evaluated at the far side of the support (the split-seam analogue
        # of auto_set_recon_geometry's per-end extension); +2 covers the voxel footprint and the
        # worst-case half-slice cut/split misalignment.
        rho = 1.0 / ratio_pixel_to_sino_pitch
        support_radius = mj.get_support_radius(full_recon_shape, voxel_row_aspect * delta_voxel,
                                               delta_voxel, use_ror_mask=use_ror_mask)
        half_overlap_recon = int(np.ceil(
            half_overlap_sino * (1.0 + support_radius / float(source_iso_dist)) * rho)) + 2

        """
        Compute detector shape parameters for top and bottom sinograms
        """
        # -------- Choose the detector row nearest to iso (the cut row) --------
        det_iso_row_float = ((full_num_rows - 1) / 2.0) + (full_det_row_offset / delta_det_row)
        det_iso_row_index = int(jnp.round(det_iso_row_float))

        # Validate iso-row index is inside (0, num_rows)
        if not (0 < det_iso_row_index < full_num_rows):
            raise ValueError(
                f"Computed det_iso_row_index={det_iso_row_index} is out of valid range (0, {full_num_rows-1}). "
            )

        # -------- Optional cut/split alignment (align_split_grid) --------
        # The seam-stripe driver is the SUB-SLICE misalignment eps between the cut row's
        # iso-mapped plane and the split slice, eps = wrap(split_offset - cut_offset_rows * rho)
        # with both round-off terms in [-1/2, 1/2].  Two mechanisms remove it: choosing a
        # different cut row changes eps by rho per row (effective only when rho != 1; at rho == 1
        # the row and slice grids are commensurate and eps is invariant under every index
        # choice), and a sub-slice shift of the whole recon grid removes the residual exactly.
        # The shift moves the OUTPUT grid by up to half a slice -- an equally valid sampling that
        # is not registration-identical to recon(), which is why this is opt-in.
        grid_shift_alu = 0.0
        if align_split_grid:
            def _wrap_half(x):
                return (x + 0.5) % 1.0 - 0.5
            iso_float_unshifted = (full_recon_slices - 1) / 2.0 - full_recon_slice_offset / delta_voxel_slice
            split_off_unshifted = round(iso_float_unshifted) - iso_float_unshifted
            candidates = [c for c in range(det_iso_row_index - 2, det_iso_row_index + 3)
                          if 0 < c < full_num_rows]
            eps_by_cut = {c: _wrap_half(split_off_unshifted - (c - det_iso_row_float) * rho)
                          for c in candidates}
            det_iso_row_index = min(eps_by_cut, key=lambda c: abs(eps_by_cut[c]))
            grid_shift_alu = -float(eps_by_cut[det_iso_row_index]) * delta_voxel_slice
            full_recon_slice_offset = full_recon_slice_offset + grid_shift_alu

        # -------- Compute the recon slice nearest to iso (the split slice) --------
        full_recon_iso_slice_index_float = (full_recon_slices - 1) / 2.0 - full_recon_slice_offset / delta_voxel_slice
        split_index = int(jnp.round(full_recon_iso_slice_index_float))
        top_num_slices = split_index + 1

        # Compute the offset of the split from iso.
        # This will be used to slightly shift the slices so that they align with a standard reconstruction.
        split_offset = split_index - full_recon_iso_slice_index_float

        # Residual sub-slice cut/split misalignment, for the returned split_params (0 up to float
        # noise when align_split_grid=True; the +2 overlap margin absorbs it when False).
        split_cut_mismatch = float((split_offset - (det_iso_row_index - det_iso_row_float) * rho
                                    + 0.5) % 1.0 - 0.5)

        # Fallback: if the split leaves either half too thin -- an empty half sinogram, or fewer
        # kept slices than the recon overlap (the stitch spans 2 * half_overlap_recon slices) --
        # then warn and do a normal MBIR recon.
        if (split_index < 1 or split_index > full_recon_slices - 2
                or top_num_slices < half_overlap_recon
                or full_recon_slices - top_num_slices < half_overlap_recon):
            warnings.warn(
                "split_index is too close to the volume boundary for the recon overlap; "
                "falling back to standard MBIR reconstruction.",
                UserWarning,
            )
            return self.recon(
                sino,
                weights=weights,
                init_recon=init_recon,
                max_iterations=max_iterations,
                stop_threshold_change_pct=stop_threshold_change_pct,
                first_iteration=first_iteration,
                compute_prior_loss=compute_prior_loss,
                logfile_path=logfile_path,
                print_logs=print_logs,
            )

        # -------- Per-half scalar parameters (cheap; the heavy arrays + models are built one half at a
        # time in _recon_one_half below, so only ONE half's inputs are resident at once) --------
        top_lo, top_hi = 0, min(det_iso_row_index + half_overlap_sino, full_num_rows)
        bot_lo, bot_hi = max(det_iso_row_index - half_overlap_sino, 0), full_num_rows

        top_recon_shape = (full_recon_shape[0], full_recon_shape[1], top_num_slices + half_overlap_recon)
        bot_recon_shape = (full_recon_shape[0], full_recon_shape[1], (full_recon_shape[2] - top_num_slices) + half_overlap_recon)
        top_recon_slice_offset = (+half_overlap_recon - (top_recon_shape[2]-1)/2 + 0 + split_offset) * delta_voxel_slice
        bot_recon_slice_offset = (-half_overlap_recon + (bot_recon_shape[2]-1)/2 + 1 + split_offset) * delta_voxel_slice

        full_det_center = (full_num_rows - 1) / 2.0

        # NOTE: earlier versions applied a quarter-sine taper to each half's overlap-row weights.
        # With the geometry-derived recon overlap above, the overlap data is fully explainable by
        # each half's extended slices, so the taper is unnecessary -- and the flash-remediation
        # campaign showed it is a regime-dependent suppressor that fails at coarse downsampling
        # while the matched extension fixes the seam in every regime tested
        # (plans/flash_remediation/).  The taper is therefore retired.

        # Regularization params come from the FULL sinogram; the halves copy them and set
        # auto_regularize_flag=False so they do not re-derive from their partial sinograms.
        self.auto_set_regularization_params(sino)

        def _recon_one_half(lo, hi, recon_shape, recon_slice_offset, is_top, half_logfile_path):
            """Reconstruct one detector-row half on the host; return (host_recon, recon_dict).

            Builds the half's model, sinogram slice, and weights, runs recon, and gathers the result to
            the host.  All the heavy state (the half model, the device recon) is local, so it is
            released when this returns -- only ONE half's inputs are resident at a time, which is
            the point of doing half a recon at a time.  The returned reconstruction is a host array.
            """
            num_rows = hi - lo
            det_center = (num_rows - 1) / 2.0
            det_row_offset = full_det_row_offset + (full_det_center - (det_center + lo)) * delta_det_row

            # Half model: copy the parent's parameters, set this half's detector/recon geometry, then pin
            # to the parent's devices (the fixed-GPU purpose; copy_ct_model does not carry the layout, so
            # an explicit configure_devices on the parent would otherwise be dropped).  configure_devices
            # rebuilds the placement for THIS half's recon shape; a non-dividing slice axis pads inertly.
            model = mj.copy_ct_model(self, new_num_det_rows=num_rows)
            model.set_params(det_row_offset=det_row_offset)
            model.set_params(auto_regularize_flag=False)
            model.set_params(recon_shape=recon_shape)
            model.set_params(recon_slice_offset=recon_slice_offset)
            if self.shard_devices is not None:
                model.configure_devices(self.shard_devices)

            # Sinogram and weight slices are host VIEWS (nothing mutates them; weights=None passes
            # through so the half recon uses its constant-weight path with no ones array built).
            sino_half = sino[:, lo:hi, :]
            weights_half = None if weights is None else weights[:, lo:hi, :]

            # init_recon slice: the top half takes the first recon_shape[2] slices, the bottom the last.
            half_init = None
            if init_recon is not None:
                half_init = init_recon[:, :, :recon_shape[2]] if is_top else init_recon[:, :, -recon_shape[2]:]

            recon_half, recon_dict = model.recon(sino_half, weights=weights_half, init_recon=half_init,
                                                 max_iterations=max_iterations,
                                                 stop_threshold_change_pct=stop_threshold_change_pct,
                                                 first_iteration=first_iteration,
                                                 compute_prior_loss=compute_prior_loss,
                                                 logfile_path=half_logfile_path, print_logs=print_logs)
            # recon() already returns a host NumPy array (its output_sharded=False gather), so the
            # half is on the host here -- no device_get needed.
            return recon_half, recon_dict

        # -------- Reconstruct the halves ONE AT A TIME (the top half is built, recon'd, gathered to the
        # host, and freed before the bottom half is built), so only one half's sino/weights/model and one
        # half's device recon are resident at any moment. --------
        # Each half logs to its own temp file; the two are merged into logfile_path afterward
        # (in finally, so any half logs written before a failure are preserved).
        if logfile_path:
            log_path = os.path.expanduser(logfile_path)
            half_log_paths = (log_path + '.top', log_path + '.bot')
        else:
            log_path, half_log_paths = None, (None, None)
        try:
            recon_top_half, recon_top_dict = _recon_one_half(top_lo, top_hi, top_recon_shape,
                                                             top_recon_slice_offset, is_top=True,
                                                             half_logfile_path=half_log_paths[0])
            recon_bot_half, recon_bot_dict = _recon_one_half(bot_lo, bot_hi, bot_recon_shape,
                                                             bot_recon_slice_offset, is_top=False,
                                                             half_logfile_path=half_log_paths[1])
        finally:
            if log_path:
                mj.merge_log_files(log_path, zip(('split_sino_recon: top half', 'split_sino_recon: bottom half'),
                                                 half_log_paths))

        # -------- Stitch together top and bottom reconstructions --------
        # Both halves were device_get'd to the host above, so stitch_arrays (host-preserving) assembles
        # the full volume ON THE HOST -- the full recon is never rebuilt on a single device, which would
        # defeat the half-at-a-time memory saving (and OOM for a recon too large to fit whole on the GPUs).
        # half_overlap_recon is used on both sides of the seam, so total overlap is 2 * half_overlap_recon.
        # ramp_overlap determines which slices are blended, which is usually less than 2 * half_overlap_recon
        # to avoid possible boundary effects.  ramp_overlap should be even so that it applies equally to slices on
        # either side of the seam.
        ramp_overlap = 4
        ramp_overlap = min(ramp_overlap, half_overlap_recon)
        ramp_overlap -= ramp_overlap % 2  # ensure even
        recon_full = mj.stitch_arrays([recon_top_half, recon_bot_half], axis=2,
                                      overlap=2 * half_overlap_recon, ramp_overlap=ramp_overlap)

        # -------- Construct full reconstruction dictionary --------
        recon_full_dict = {'recon_params_top': recon_top_dict['recon_params'],
                           'recon_params_bottom': recon_bot_dict['recon_params'],
                           'recon_log_top': recon_top_dict['recon_log'],
                           'recon_log_bottom': recon_bot_dict['recon_log'],
                           'notes_top': recon_top_dict['notes'],
                           'notes_bottom': recon_bot_dict['notes'],
                           'model_params_top': recon_top_dict['model_params'],
                           'model_params_bottom': recon_bot_dict['model_params'],
                           # The overlaps actually used, the residual sub-slice cut/split
                           # misalignment, and any align_split_grid shift of the OUTPUT grid
                           # (in ALU; the returned volume samples a grid whose
                           # recon_slice_offset differs from the model's by this amount).
                           'split_params': {'half_overlap_sino': int(half_overlap_sino),
                                            'half_overlap_recon': int(half_overlap_recon),
                                            'align_split_grid': bool(align_split_grid),
                                            'grid_shift_alu': float(grid_shift_alu),
                                            'split_cut_mismatch_slices': split_cut_mismatch}, }

        return recon_full, recon_full_dict
