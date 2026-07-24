import warnings
import jax
import jax.numpy as jnp
from functools import partial
from collections import namedtuple

import numpy as np

import mbirjax as mj
from mbirjax.projectors import (horizontal_fan_project, horizontal_fan_back,
                                vertical_fan_band_gather)


# Default slice-band size for the translation back projector's rolled vertical-fan loop.
# The back projector computes the horizontal fan once per view, then walks the recon slice
# axis in bands of this many slices using a jax.lax.map.  The loop is ROLLED (the band body
# compiles once and iterates at runtime), so the compiled program size is independent of the
# slice count -- which can range from tens of slices on one device to thousands across many.
# The band size bounds the per-band scratch to (pixel_batch x band_size); it is the back
# projector's memory knob, replacing the old entries_per_cylinder_batch geometry parameter.
# 128 matches that old default.
TRANSLATION_SLICE_BAND_SIZE = 128

# Detector-row batch size for the FORWARD vertical fan's rolled jax.lax.map over detector
# rows.  Like the back band size, this is a memory/compile knob that bounds the per-batch
# transient.  The forward projection does NOT band the slice axis (it stays monolithic), so
# this chunks the OUTPUT detector rows instead.  It also replaces the old
# entries_per_cylinder_batch parameter; 128 matches the old default, so the forward output is
# unchanged.
TRANSLATION_FORWARD_DET_ROW_BATCH = 128


class TranslationModel(mj.TomographyModel):
    """
    This class implements the translation tomography geometry in which each view is a cone beam projection of a translated object.
    This geometry is useful for 3D imaging of thin objects.

    This class inherits all methods and properties from the :ref:`TomographyModelDocs` and may override some
    to suit the translation tomography geometrical requirements. See the documentation of the parent class for standard methods
    like setting parameters and performing projections and reconstructions.

    Args:
        sinogram_shape (tuple): Shape of the sinogram as (num_views, num_rows, num_channels),
            where 'num_views' is the number of translation steps, 'num_rows' is the number of detector rows,
            and 'num_channels' is the number of detector columns.
        translation_vectors (jax array or numpy array): A (num_views, 3) array of translations (x, y, z) in ALUs.
            Each vector specifies how the object is translated for each view.
            Positive x shifts the object left, z shifts up, and y shifts away from the source.
        source_detector_dist (float): Distance from the X-ray source to the detector.
        source_iso_dist (float): Distance from the X-ray source to the isocenter.

    See Also:
        mbirjax.TomographyModel: Base class with standard methods like `set_params` and `reconstruct`.

    Example:
        .. code-block:: python

            import jax.numpy as jnp
            from mbirjax.translation_model import TranslationModel

            sinogram_shape = (180, 256, 256)
            translation_vectors = jnp.zeros((180, 3))
            model = TranslationModel(
                sinogram_shape=sinogram_shape,
                translation_vectors=translation_vectors,
                source_detector_dist=500.0,
                source_iso_dist=500.0
            )
            model.auto_set_recon_geometry()
    """

    def __init__(self, sinogram_shape, translation_vectors, source_detector_dist, source_iso_dist):

        translation_vectors = jnp.asarray(translation_vectors)
        view_params_name = 'translation_vectors'

        super().__init__(sinogram_shape, translation_vectors=translation_vectors, source_detector_dist=source_detector_dist,
                         source_iso_dist=source_iso_dist, view_params_name=view_params_name, qggmrf_nbr_wts=(0.1, 1, 1,), use_ror_mask=False)
        
        self.set_params(max_alpha=1.3)  # We override this value due to observed instabilities with larger values

    def get_magnification(self):
        """
        Returns the magnification for the cone beam geometry.

        Returns:
            magnification = source_detector_dist / source_iso_dist
        """
        source_detector_dist, source_iso_dist = self.get_params(['source_detector_dist', 'source_iso_dist'])
        if jnp.isinf(source_detector_dist):
            raise ValueError('Distance from source to detector is infinite, which means all translated projections have the same information.')
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
        voxel_row_aspect, voxel_slice_aspect = self.get_params(['voxel_row_aspect', 'voxel_slice_aspect'])

        if voxel_row_aspect <= 0:
            error_message = "Voxel row aspect ratio must be positive. \n"
            error_message += "Got {} for voxel_row_aspect.".format(voxel_row_aspect)
            raise ValueError(error_message)

        if voxel_slice_aspect <= 0:
            error_message = "Voxel slice aspect ratio must be positive. \n"
            error_message += "Got {} for voxel_slice_aspect.".format(voxel_slice_aspect)
            raise ValueError(error_message)

    def get_geometry_parameters(self):
        """
        Function to get a list of the primary geometry parameters for translation model projection.

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
        ]
        geometry_param_values = self.get_params(geometry_param_names)

        # Then get additional parameters:
        geometry_param_names += ['magnification', 'psf_radius']
        geometry_param_values.append(self.get_magnification())
        geometry_param_values.append(self.get_psf_radius())

        # Then create a namedtuple to access parameters by name in a way that can be jit-compiled.
        # The class is shared across instances (make_geometry_params) so the projectors' jit cache
        # is shared rather than re-traced per instance.
        geometry_params = self.make_geometry_params(geometry_param_names, geometry_param_values)

        return geometry_params

    def get_psf_radius(self):
        """
        Compute the integer radius of the PSF kernel for cone beam projection.
        """
        delta_det_row, delta_det_channel, source_iso_dist, source_detector_dist, recon_shape, delta_voxel, translation_vectors, voxel_row_aspect, voxel_slice_aspect = self.get_params(
            ['delta_det_row', 'delta_det_channel', 'source_iso_dist', 'source_detector_dist', 'recon_shape', 'delta_voxel', 'translation_vectors', 'voxel_row_aspect', 'voxel_slice_aspect'])
        magnification = self.get_magnification()

        delta_voxel_row = voxel_row_aspect * delta_voxel
        delta_voxel_slice = voxel_slice_aspect * delta_voxel

        # Compute minimum detector pitch
        delta_det = jnp.minimum(delta_det_row, delta_det_channel)

        # Find the maximum and minimum translation vectors
        max_translation = jnp.amax(translation_vectors, axis=0)
        min_translation = jnp.amin(translation_vectors, axis=0)

        # Compute maximum magnification
        if jnp.isinf(source_detector_dist):
            raise ValueError('Distance from source to detector is infinite, which means all translated projections have the same information.')
        else:
            # Determine the distance from the source to the closest voxel.
            source_to_closest_pixel = source_iso_dist - (0.5 * recon_shape[0] * delta_voxel_row) - max_translation[1]
            # Determine the maximum magnification.
            max_magnification = source_detector_dist / source_to_closest_pixel

            if source_to_closest_pixel < 0:
                raise ValueError('Reconstruction volume extends into source - no valid projection in this case.')

        # Compute the maximum number of detector rows/channels on either side of the center detector hit by a voxel
        max_voxel_pitch = jnp.maximum(delta_voxel, delta_voxel_slice)
        psf_radius = int(jnp.ceil(jnp.ceil((max_voxel_pitch * max_magnification / delta_det)) / 2))
        if psf_radius > 4:
            warnings.warn('A single voxel may project onto 100 or more detector elements, which may lead to artifacts. Consider using smaller voxels.')
        return psf_radius

    def auto_set_recon_geometry(self, no_compile=False, no_warning=False):
        """ Compute the automatic recon shape translation reconstruction.
        """
        # Get model parameters
        sinogram_shape = self.get_params('sinogram_shape')
        source_detector_dist, source_iso_dist = self.get_params(['source_detector_dist', 'source_iso_dist'])
        delta_det_row, delta_det_channel = self.get_params(['delta_det_row', 'delta_det_channel'])
        translation_vectors = self.get_params('translation_vectors')
        voxel_row_aspect, voxel_slice_aspect = self.get_params(['voxel_row_aspect', 'voxel_slice_aspect'])

        # Calculate the reconstruction geometry parameters
        recon_shape, delta_voxel, voxel_row_aspect = mj.utilities.calc_tct_recon_params(source_detector_dist,
                                                                                        source_iso_dist, delta_det_row,
                                                                                        delta_det_channel,
                                                                                        sinogram_shape,
                                                                                        translation_vectors,
                                                                                        voxel_row_aspect,
                                                                                        voxel_slice_aspect)

        self.set_params(no_compile=no_compile, no_warning=no_warning, recon_shape=recon_shape, delta_voxel=delta_voxel,
                        voxel_row_aspect=voxel_row_aspect)

    def _check_lateral_truncation(self, sino_indicator):
        """No-op override: in translation tomography the object (typically a plate wider than
        the detector) routinely spans the whole field of view, so edge-touching sinogram
        support is the normal operating condition rather than the lateral-truncation defect
        the base check warns about (see TomographyModel._check_lateral_truncation)."""
        return

    def _select_tile_policy(self, on_gpu, num_views, num_slices, n_devices):
        """Translation tiling: the base policy, unchanged -- with two MEASURED deliberate
        non-settings.

        Deliberately NOT set: ``sort_by_channel``.  The forward horizontal fan carries the
        SORTED-channel-reduction branch (defined at projectors.channel_scatter_reduce; kept
        for experiments), but at
        translation's REAL detector shapes (~1900x3000 TCT panels) the sorted reduce is
        4.5-6.5x SLOWER than the scatter: with few views and thousands of channels there
        are only ~2 scatter collisions per channel -- nothing for the sort to win back
        (pixel_count_crossover_ab.py; the collision-ratio guard constant in projectors.py
        records the crossover).  Small square test shapes DO show a win, but at
        sub-millisecond stakes; the production shapes dominate the decision.  The wide psf
        translation runs at large cone angle (psf_radius 2-3) degrades the sorted form
        further.

        Deliberately NOT set: ``back_stacked_gather``.  The stacked horizontal-fan gather
        wins in isolation but changes the FULL translation back kernel not at all -- the
        gather latency hides behind the vertical-fan band work (mt_back_kernel_ab.py,
        measured at psf_radius 1 and 3).
        """
        return super()._select_tile_policy(on_gpu, num_views, num_slices, n_devices)

    @staticmethod
    @partial(jax.jit, static_argnames='projector_params')
    def forward_project_pixel_batch_to_one_view(voxel_values, pixel_indices, translation_vector, n_p_centers,
                                                projector_params):
        """
        Forward project a set of voxels determined by indices into the flattened array of size num_rows x num_cols.

        Args:
            voxel_values (jax array):  2D array of shape (num_indices, num_slices) of voxel values, where
                voxel_values[i, j] is the value of the voxel in slice j at the location determined by indices[i].
            pixel_indices (jax array of int):  1D vector of indices into flattened array of size num_rows x num_cols.
            translation_vector (jax array of floats):  1D translation vector in ALU units for this view.
            n_p_centers (jax array of int): (num_pixels,) integer channel centers for this view,
                computed OUTSIDE this program (see compute_channel_coordinate).
            projector_params (namedtuple):  tuple of (sinogram_shape, recon_shape, get_geometry_params())

        Returns:
            jax array of shape (num_det_rows, num_det_channels)
        """
        recon_shape = projector_params.recon_shape
        num_recon_slices = recon_shape[2]
        if voxel_values.shape[0] != pixel_indices.shape[0] or len(voxel_values.shape) < 2 or \
                voxel_values.shape[1] != num_recon_slices:
            raise ValueError('voxel_values must have shape[0:2] = (num_indices, num_slices)')

        vertical_fan_projector = TranslationModel.forward_vertical_fan_pixel_batch_to_one_view
        horizontal_fan_projector = TranslationModel.forward_horizontal_fan_pixel_batch_to_one_view

        new_voxel_values = vertical_fan_projector(voxel_values, pixel_indices, translation_vector, projector_params)
        sinogram_view = horizontal_fan_projector(new_voxel_values, pixel_indices, translation_vector,
                                                 n_p_centers, projector_params)

        return sinogram_view

    @staticmethod
    def forward_vertical_fan_pixel_batch_to_one_view(voxel_values, pixel_indices, translation_vector, projector_params):
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
            translation_vector (jax array of floats):  1D translation vector in ALU units for this view.
            projector_params (namedtuple):  tuple of (sinogram_shape, recon_shape, get_geometry_params())

        Returns:
            jax array of shape (num_pixels, num_det_rows)
        """
        pixel_map = jax.vmap(TranslationModel.forward_vertical_fan_one_pixel_to_one_view,
                             in_axes=(0, 0, None, None))
        new_pixels = pixel_map(voxel_values, pixel_indices, translation_vector, projector_params)

        return new_pixels

    @staticmethod
    def forward_horizontal_fan_pixel_batch_to_one_view(voxel_values, pixel_indices, translation_vector,
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
            translation_vector (jax array of floats):  1D translation vector in ALU units for this view.
            projector_params (namedtuple):  tuple of (sinogram_shape, recon_shape, get_geometry_params())

        Returns:
            jax array of shape (num_det_rows, num_det_channels)
        """

        # Get all the geometry parameters - we use gp since geometry parameters is a named tuple and we'll access
        # elements using, for example, gp.delta_det_channel, so a longer name would be clumsy.
        gp = projector_params.geometry_params
        num_views, num_det_rows, num_det_channels = projector_params.sinogram_shape

        # Get the data needed for horizontal projection
        n_p, W_p_c, cos_theta_p = TranslationModel.compute_horizontal_data(pixel_indices, translation_vector, projector_params)
        delta_voxel_row = gp.voxel_row_aspect * gp.delta_voxel

        # The shared trapezoid-rule fan (see horizontal_fan_project in projectors.py, which
        # owns the channel-major layout rationale and the sort_by_channel branch --
        # deliberately NOT enabled by translation's policy: the channel-collision cliff at
        # real TCT shapes, see _select_tile_policy).  The weight scale is PER-PIXEL
        # (pixel-dependent cone angle); precomputing delta_voxel_row / cos_theta_p once here
        # (instead of the historical per-tap dvr * L / cos) is a deliberate last-bit-level
        # reassociation, accepted for the shared-helper form.
        weight_scale = delta_voxel_row / cos_theta_p
        sinogram_view_T = horizontal_fan_project(n_p, n_p_centers, W_p_c, weight_scale,
                                                 voxel_values, num_det_channels, gp.psf_radius,
                                                 use_sorted=projector_params.sort_by_channel)
        return sinogram_view_T.T

    @staticmethod
    def forward_vertical_fan_one_pixel_to_one_view(voxel_cylinder, pixel_index, translation_vector, projector_params):
        """
        Apply a fan beam forward projection in the vertical direction to the pixel determined by indices
        into the flattened array of size num_rows x num_cols.  This returns a vector obtained from the projection of
        the original voxel cylinder onto a detector column, so the output vector has length num_det_rows.

        Args:
            voxel_cylinder (jax array):  1D array of shape (num_recon_slices, ) of voxel values, where
                voxel_cylinder[j] is the value of the voxel in slice j at the location determined by pixel_index.
            pixel_index (int):  Index into the flattened array of size num_rows x num_cols.
            translation_vector (jax array of floats):  1D translation vector in ALU units for this view.
            projector_params (namedtuple):  tuple of (sinogram_shape, recon_shape, get_geometry_params())

        Returns:
            jax array of shape (num_det_rows,)

        Note:
            This is a helper function used in vmap in :meth:`TranslationModel.forward_vertical_fan_pixel_batch_to_one_view`
        This method has the same signature and output as that method, except single int pixel_index is used
        in place of the 1D pixel_indices, and likewise only a single voxel cylinder is returned.
        """
        # Get all the geometry parameters - we use gp since geometry parameters is a named tuple and we'll access
        # elements using, for example, gp.delta_det_channel, so a longer name would be clumsy.
        gp = projector_params.geometry_params
        num_views, num_det_rows, num_det_channels = projector_params.sinogram_shape
        recon_shape = projector_params.recon_shape
        num_recon_rows, num_recon_cols, num_recon_slices = recon_shape
        # Anchor the slice<->z map on the REAL slice count from params (recon_shape), NOT the
        # input cylinder length, so the forward stays consistent with the back projector (which
        # anchors z on recon_shape) and the validity clip below is a GLOBAL test (k < S_real).
        # The forward is monolithic -- it is never banded (decision C) -- so the global slice
        # index equals the local index here; the batch method asserts the per-pixel cylinder is
        # num_recon_slices long, so this is value-identical today and becomes the inert-padding
        # anchor once the slice axis is zero-padded for sharding.
        num_slices = num_recon_slices

        delta_voxel_slice = gp.voxel_slice_aspect * gp.delta_voxel

        # From pixel index, compute y and pixel_mag
        y, pixel_mag = TranslationModel.compute_y_mag_for_pixel(pixel_index, translation_vector, recon_shape,
                                                                projector_params)

        # The code above depends only on the pixel - a single point.  z is a potentially large vector
        # Here we compute cos_phi_p:  1 / cos_phi_p determines the projection length through a voxel
        # For computational efficiency, we use that to scale the voxel_cylinder values.
        ### possibly convert to a jitted function with donate_argnames to avoid copies for z, v, phi_p, cos_phi_p
        k = jnp.arange(num_slices)
        z = delta_voxel_slice * (k - (num_recon_slices - 1) / 2.0) - translation_vector[2]  # recon_ijk_to_xyz
        v = pixel_mag * z  # geometry_xyz_to_uv_mag
        # Compute vertical cone angle of voxels
        phi_p = jnp.arctan2(v, gp.source_detector_dist)  # compute_vertical_data_single_pixel
        cos_phi_p = jnp.cos(phi_p)  # We assume the vertical angle |phi_p| < 45 degrees so cos_alpha_p_z = cos_phi_p
        scaled_voxel_values = voxel_cylinder / cos_phi_p
        # End

        # Get the length of projection of detector on vertical voxel profile (in fraction of voxel size)
        # This is also the slope of the map from voxel index to detector index
        W_p_r = (pixel_mag * delta_voxel_slice) / gp.delta_det_row
        slope_k_to_m = W_p_r
        L_max = jnp.minimum(1, W_p_r)  # Maximum fraction of a detector that can be covered by one voxel.

        # Set up detector row indices array (0, 10, 20, ..., 10*num_slice_batches)
        det_rows_per_batch = TRANSLATION_FORWARD_DET_ROW_BATCH
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
            k_m = (z_m + translation_vector[2]) / delta_voxel_slice + (num_slices - 1) / 2.0
            k_m_center = jnp.round(k_m).astype(int)  # Center of the voxel hit by the center of the detector
            # Then map the center of the voxels back to the detector.
            m_p = slope_k_to_m * (k_m_center - k_m[0]) + m_center[0]  # Projection to detector of voxel centers

            # Allocate space
            new_column_batch = jnp.zeros(det_rows_per_batch)
            # Do the vertical projection
            for k_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius + 1):
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

        The horizontal fan is computed ONCE per view (it does not depend on the recon slice),
        then the recon slice axis is walked in fixed-size bands with a ROLLED loop
        (jax.lax.map): the band body compiles once and iterates at runtime, so the compiled
        program does not grow with the slice count.  Each band back-projects onto its block of
        global slices via the banded vertical fan; the bands are stacked and cropped to the
        real slice count (the last band is padded up to a whole band and the banded kernel's
        global validity clip zeros that padded tail, so the crop is exact).

        Args:
            sinogram_view (2D jax array): one view of the sinogram to be back projected.
                2D jax array of shape (num_det_rows)x(num_det_channels)
            pixel_indices (1D jax array of int):  indices into flattened array of size num_rows x num_cols.
            single_view_params: These are the view dependent parameters for the view being back projected.
            projector_params (namedtuple): tuple of (sinogram_shape, recon_shape, get_geometry_params()).
            coeff_power (int): backproject using the coefficients of (A_ij ** coeff_power).
                Normally 1, but should be 2 when computing Hessian diagonal.
            slice_band_size (int or None): number of recon slices per band (the memory knob).
                None uses the module default TRANSLATION_SLICE_BAND_SIZE; tests pass a small
                value to exercise the multi-band assembly on a small geometry.  Static (sets shapes).

        Returns:
            The voxel values for all slices at the input index (i.e., a voxel cylinder) obtained by backprojecting
            the input sinogram view.  Shape (len(pixel_indices), num_recon_slices).
        """
        num_recon_slices = projector_params.recon_shape[2]
        num_pixels = pixel_indices.shape[0]

        # Horizontal fan once per view -> detector-based voxel cylinder (num_pixels, num_det_rows).
        det_voxel_cylinder = TranslationModel.back_horizontal_fan_one_view_to_pixel_batch(
            sinogram_view, pixel_indices, single_view_params, n_p_centers, projector_params,
            coeff_power=coeff_power)

        # Tile the slice axis into uniform bands; jax.lax.map needs equal-shape iterations, so
        # the last band runs past num_recon_slices and is cropped off below.
        band_size = TRANSLATION_SLICE_BAND_SIZE if slice_band_size is None else slice_band_size
        band_size = min(band_size, num_recon_slices)
        num_bands = (num_recon_slices + band_size - 1) // band_size
        band_starts = band_size * jnp.arange(num_bands)        # g0 for each band (mapped over)

        def back_one_band(g0):
            # (num_pixels, band_size) back projection onto global slices [g0, g0 + band_size).
            return TranslationModel.back_vertical_fan_band_pixel_batch(
                det_voxel_cylinder, pixel_indices, single_view_params, projector_params,
                g0, band_size, coeff_power=coeff_power)

        bands = jax.lax.map(back_one_band, band_starts)        # (num_bands, num_pixels, band_size)
        # Reassemble into (num_pixels, num_bands * band_size): for pixel p and band b, slice
        # b*band_size + l is bands[b, p, l].  Then crop the padded tail back to the real count.
        back_projection = jnp.transpose(bands, (1, 0, 2)).reshape(num_pixels, num_bands * band_size)
        back_projection = jax.lax.slice_in_dim(back_projection, 0, num_recon_slices, axis=1)

        return back_projection

    @staticmethod
    def back_horizontal_fan_one_view_to_pixel_batch(sinogram_view, pixel_indices, translation_vector,
                                                    n_p_centers, projector_params, coeff_power=1):
        """
        Apply the back projection of a horizontal fan beam transformation to a single sinogram view
        and return the resulting voxel cylinders.

        Args:
            sinogram_view (2D jax array): one view of the sinogram to be back projected.
                2D jax array of shape (num_det_rows)x(num_det_channels)
            pixel_indices (1D jax array of int):  indices into flattened array of size num_rows x num_cols.
            translation_vector (jax array of floats):  1D translation vector in ALU units for this view.
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
        n_p, W_p_c, cos_theta_p = TranslationModel.compute_horizontal_data(pixel_indices, translation_vector, projector_params)
        delta_voxel_row = gp.voxel_row_aspect * gp.delta_voxel

        # The shared adjoint trapezoid-rule fan (see horizontal_fan_back in projectors.py,
        # which owns the channel-major layout rationale and the back_stacked_gather branch
        # -- deliberately NOT enabled by translation's policy: measured a 1.00-1.02x
        # composition no-op behind the vertical fan).  Weight scale mirrors the forward fan
        # (per-pixel; same accepted last-bit-level reassociation).
        sinogram_view_T = sinogram_view.T            # (num_det_channels, num_det_rows)
        weight_scale = delta_voxel_row / cos_theta_p
        return horizontal_fan_back(sinogram_view_T, n_p, n_p_centers, W_p_c, weight_scale,
                                   gp.psf_radius, coeff_power=coeff_power,
                                   use_stacked=projector_params.back_stacked_gather)

    # ──────────────────────────────────────────────────────────────────────────
    # Banded vertical fan + per-view banded back kernels.  The shared contract (global-index
    # z anchoring / tiling-invariance, inert padding, single-device band walk in bands of
    # TRANSLATION_SLICE_BAND_SIZE + the multi-device per-band entry) is documented ONCE at
    # the canonical note above projectors.vertical_fan_band_gather.
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def back_vertical_fan_band_one_pixel(detector_column_values, pixel_index, translation_vector,
                                         projector_params, g0, num_band_slices, coeff_power=1):
        """Back vertical fan for one pixel, producing a BAND of output slices.

        Back-projects this pixel's detector column onto the GLOBAL recon slices
        [g0, g0+L) (output length L = num_band_slices) -- "band" names the output slice
        range.  compute_vertical_data_single_pixel already anchors z on the problem's
        recon_shape, so this just passes the global band indices; padded global slices
        (index >= the real slice count) are zeroed.  Concatenating the per-band outputs over
        a tiling of the slices reconstructs the full vertical fan (the bands tile disjointly).

        Args:
            detector_column_values (1D jax array): shape (num_det_rows,), this pixel's detector column.
            pixel_index (int):  Index into flattened array of size num_rows x num_cols.
            translation_vector (jax array of floats):  1D translation vector in ALU units for this view.
            projector_params (namedtuple): tuple of (sinogram_shape, recon_shape, get_geometry_params()).
            g0 (int): global index of the first slice in the band (traced).
            num_band_slices (int): band length L (static).
            coeff_power (int): backproject using the coefficients of (A_ij ** coeff_power).
                Normally 1, but should be 2 when computing Hessian diagonal.

        Returns:
            1D jax array of shape (num_band_slices,) of voxel values.
        """
        # Get all the geometry parameters - we use gp since geometry parameters is a named tuple and we'll access
        # elements using, for example, gp.delta_det_channel, so a longer name would be clumsy.
        gp = projector_params.geometry_params
        num_views, num_det_rows, num_det_channels = projector_params.sinogram_shape
        num_recon_slices = projector_params.recon_shape[2]

        slice_indices = g0 + jnp.arange(num_band_slices)            # GLOBAL band indices
        m_p, m_p_center, W_p_r, cos_alpha_p_z = TranslationModel.compute_vertical_data_single_pixel(
            pixel_index, slice_indices, translation_vector, projector_params)
        # The shared banded vertical-fan gather (see vertical_fan_band_gather in
        # projectors.py; cone shares it verbatim -- weights L / cos_alpha, padded slices
        # zeroed).
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
        pixel_map = jax.vmap(TranslationModel.back_vertical_fan_band_one_pixel,
                             in_axes=(0, 0, None, None, None, None, None))
        return pixel_map(det_voxel_cylinder, pixel_indices, single_view_params, projector_params,
                         g0, num_band_slices, coeff_power)

    @staticmethod
    @partial(jax.jit, static_argnames=['projector_params', 'num_band_slices'])
    def back_project_one_view_to_band(sinogram_view, pixel_indices, single_view_params, n_p_centers,
                                      projector_params, g0, num_band_slices, coeff_power=1):
        """Banded back projection of one view onto GLOBAL recon slices [g0, g0+L).

        Horizontal fan once (full det rows) -> banded vertical fan.  Returns
        (num_pixels, num_band_slices).  ``g0`` is traced; ``num_band_slices`` (= L) is
        static.  This is the per-band entry the multi-device reduce-scatter back projector
        uses on each view-owner's shard."""
        det_voxel_cylinder = TranslationModel.back_horizontal_fan_one_view_to_pixel_batch(
            sinogram_view, pixel_indices, single_view_params, n_p_centers, projector_params,
            coeff_power=coeff_power)
        return TranslationModel.back_vertical_fan_band_pixel_batch(
            det_voxel_cylinder, pixel_indices, single_view_params, projector_params,
            g0, num_band_slices, coeff_power=coeff_power)

    @staticmethod
    def compute_vertical_data_single_pixel(pixel_index, slice_indices, translation_vector, projector_params):
        """
        Compute the quantities m_p, m_p_center, W_p_r, cos_alpha_p_z needed for vertical projection.

        Args:
            pixel_index (int):  Index into flattened array of size num_rows x num_cols.
            slice_indices (array of int): Indices into the recon slices.
            translation_vector (jax array of floats):  1D translation vector in ALU units for this view.
            projector_params (namedtuple): tuple of (sinogram_shape, recon_shape, get_geometry_params()).

        Returns:
            m_p, m_p_center, W_p_r, cos_alpha_p_z
        """
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

        x_p, y_p, z_p = TranslationModel.recon_ijk_to_xyz(row_index, col_index, slice_indices, recon_shape, gp.delta_voxel, gp.voxel_row_aspect, gp.voxel_slice_aspect, translation_vector)

        # Convert from xyz to coordinates on detector
        u_p, v_p, pixel_mag = TranslationModel.geometry_xyz_to_uv_mag(x_p, y_p, z_p, gp.source_detector_dist, gp.magnification)

        # Convert from uv to index coordinates in detector and get the vector of center detector rows for this cylinder
        m_p, _ = TranslationModel.detector_uv_to_mn(u_p, v_p, gp.delta_det_channel, gp.delta_det_row, gp.det_channel_offset,
                                                    gp.det_row_offset, num_det_rows, num_det_channels)

        m_p_center = jnp.round(m_p).astype(int)

        # Compute vertical cone angle of pixel
        phi_p = jnp.arctan2(v_p, gp.source_detector_dist)

        # Compute cos alpha for row and columns
        cos_phi_p = jnp.cos(phi_p)  # We assume the vertical angle |phi_p| < 45 degrees so cos_alpha_p_z = cos_phi_p
        # cos_alpha_p_z = jnp.maximum(jnp.abs(cos_phi_p), jnp.abs(jnp.sin(phi_p)))

        # Get the length of projection of flattened voxel on detector (in fraction of detector size)
        W_p_r = pixel_mag * (delta_voxel_slice / gp.delta_det_row)  # * cos_alpha_p_z / cos_phi_p

        vertical_data = (m_p, m_p_center, W_p_r, cos_phi_p)  # cos_alpha_p_z)

        return vertical_data

    @staticmethod
    def compute_channel_coordinate(pixel_indices, single_view_params, projector_params):
        """Continuous projected channel coordinate n_p for one view (the float chain whose
        rounded value is the horizontal fans' integer center; see the base-class docstring
        for the concrete-input contract).  Reuses compute_horizontal_data verbatim
        so the centers are consistent with the in-kernel weights by construction."""
        return TranslationModel.compute_horizontal_data(pixel_indices, single_view_params,
                                                        projector_params)[0]

    @staticmethod
    def compute_horizontal_data(pixel_indices, translation_vector, projector_params):
        """
        Compute the quantities n_p, W_p_c, cos_theta_p needed for horizontal projection.
        (The integer center round(n_p) is deliberately NOT computed here -- see
        compute_channel_coordinate.)

        Args:
            pixel_indices (1D jax array of int):  indices into flattened array of size num_rows x num_cols.
            translation_vector (jax array of floats):  1D translation vector in ALU units for this view.
            projector_params (namedtuple): tuple of (sinogram_shape, recon_shape, get_geometry_params()).

        Returns:
            n_p, n_p_center, W_p_c, cos_theta_p
        """
        # Get all the geometry parameters - we use gp since geometry parameters is a named tuple and we'll access
        # elements using, for example, gp.delta_det_channel, so a longer name would be clumsy.
        gp = projector_params.geometry_params

        num_views, num_det_rows, num_det_channels = projector_params.sinogram_shape
        recon_shape = projector_params.recon_shape
        num_recon_rows, num_recon_cols, num_recon_slices = recon_shape

        # Convert the index into (i,j,k) coordinates corresponding to the indices into the 3D voxel array
        row_index, col_index = jnp.unravel_index(pixel_indices, recon_shape[:2])
        slice_index = jnp.arange(1)

        x_p, y_p, _ = TranslationModel.recon_ijk_to_xyz(row_index, col_index, slice_index, recon_shape, gp.delta_voxel, gp.voxel_row_aspect, gp.voxel_slice_aspect, translation_vector)

        # Convert from xyz to coordinates on detector
        # pixel_mag should be kept in terms of magnification to allow for source_detector_dist = jnp.Inf
        pixel_mag = 1 / (1 / gp.magnification - y_p / gp.source_detector_dist)
        # Compute the physical position that this voxel projects onto the detector
        u_p = pixel_mag * x_p
        det_center_channel = (num_det_channels - 1) / 2.0  # num_of_cols

        # Calculate indices on the detector grid.  NOTE: no round here -- the integer center
        # is computed outside the projector programs (see compute_channel_coordinate).
        n_p = (u_p + gp.det_channel_offset) / gp.delta_det_channel + det_center_channel  # Sync with detector_uv_to_mn

        # Compute horizontal and vertical cone angle of pixel
        theta_p = jnp.arctan2(u_p, gp.source_detector_dist)

        # Compute cosine alpha
        cos_theta_p = jnp.cos(theta_p)

        # Compute projected voxel width along columns and rows (in fraction of detector size)
        W_p_c = pixel_mag * (gp.delta_voxel / gp.delta_det_channel)

        horizontal_data = (n_p, W_p_c, cos_theta_p)

        return horizontal_data

    @staticmethod
    def recon_ijk_to_xyz(i, j, k, recon_shape, delta_voxel, voxel_row_aspect, voxel_slice_aspect, translation_vector):
        """
        Convert (i, j, k) indices into the recon volume to corresponding (x, y, z) coordinates.
        """
        num_recon_rows, num_recon_cols, num_recon_slices = recon_shape

        delta_voxel_row = voxel_row_aspect * delta_voxel
        delta_voxel_slice = voxel_slice_aspect * delta_voxel

        y = delta_voxel_row * (i - (num_recon_rows - 1) / 2.0) - translation_vector[1]
        x = delta_voxel * (j - (num_recon_cols - 1) / 2.0) - translation_vector[0]
        z = delta_voxel_slice * (k - (num_recon_slices - 1) / 2.0) - translation_vector[2]
        return x, y, z

    @staticmethod
    def geometry_xyz_to_uv_mag(x, y, z, source_detector_dist, magnification):
        """
        Convert (x, y, z) coordinates to to (u, v) detector coordinates plus the pixel-dependent magnification.
        """
        # Compute the magnification at this specific voxel
        # The following expression is valid even when source_detector_dist = jnp.Inf
        pixel_mag = 1 / (1 / magnification - y / source_detector_dist)

        # Compute the physical position that this voxel projects onto the detector
        u = pixel_mag * x
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
        # Get the center of the detector grid for columns and rows
        det_center_row = (num_det_rows - 1) / 2.0  # num_of_rows
        det_center_channel = (num_det_channels - 1) / 2.0  # num_of_cols

        # Calculate indices on the detector grid
        m = (v + det_row_offset) / delta_det_row + det_center_row
        n = (u + det_channel_offset) / delta_det_channel + det_center_channel  # Sync with compute_horizontal_data

        return m, n

    @staticmethod
    @jax.jit
    def compute_y_mag_for_pixel(pixel_index, translation_vector, recon_shape, projector_params):

        gp = projector_params.geometry_params
        row_index, col_index = jnp.unravel_index(pixel_index, recon_shape[:2])
        num_recon_rows, num_recon_cols, num_recon_slices = recon_shape

        delta_voxel_row = gp.voxel_row_aspect * gp.delta_voxel

        y = delta_voxel_row * (row_index - (num_recon_rows - 1) / 2.0) - translation_vector[1]

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
            output_sharded (bool, optional): If False (default), return a numpy array; if True,
                return the view-sharded device form (on a single device the same either way).

        Returns:
            filtered_sinogram (numpy or jax array): The sinogram after FBP filtering.
        """
        return self.fdk_filter(sinogram, filter_name=filter_name, output_sharded=output_sharded)

    def fdk_filter(self, sinogram, filter_name="ramp", output_sharded=False):
        """
        Perform FDK filtering on the given sinogram.

        Uses the shared row-batched filter kernel (tomography_utils.apply_row_filter via
        TomographyModel._apply_direct_recon_filter), the same path as
        ParallelBeamModel.fbp_filter and ConeBeamModel.fdk_filter, with an FDK cosine
        pre-weight applied per detector row.  The peak stays at the input+output floor (no
        full-sinogram out-of-place ``sinogram * weight`` copy, no concatenate, no f64
        promotion of the sinogram), and the fragile per-view ``lax.map`` (jax-ml/jax#27591)
        is gone.  Through the shared sharded row-filter, ``output_sharded=True`` returns the
        view-sharded device form (on a single device the result is the same either way).

        Args:
            sinogram (numpy or jax array): The input sinogram with shape (num_views, num_rows, num_channels).
            filter_name (string, optional): Name of the filter to be used. Defaults to "ramp"
            output_sharded (bool, optional): If False (default), return a numpy array; if True,
                return the view-sharded device form (on a single device the same either way).

        Returns:
            filtered_sinogram (numpy or jax array): The sinogram after FDK filtering.
        """
        # Detector geometry + voxel scaling -- the only FDK-specific pieces; the shared row
        # filter does the rest (bounded peak, jitted, sharded per view-shard once ported).
        # TranslationModel reuses ConeBeamModel.detector_mn_to_uv (it has no detector map of
        # its own), matching the fan architecture it shares with cone.
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
        u_grid, v_grid = mj.ConeBeamModel.detector_mn_to_uv(m_grid, n_grid, delta_det_channel, delta_det_row,
                                                            det_channel_offset, det_row_offset, num_rows, num_channels)
        weight_map = source_detector_dist / jnp.sqrt(source_detector_dist ** 2 + u_grid ** 2 + v_grid ** 2)

        # FDK filter scale alpha (voxel-size factor); pi/num_views is folded in by the
        # shared method.  For the derivation see the theory zip at
        # https://mbirjax.readthedocs.io/en/latest/theory.html
        alpha = delta_det_row / (voxel_volume * M_0)
        return self._apply_direct_recon_filter(
            sinogram, filter_name, filter_scale=alpha,
            output_sharded=output_sharded, row_weight=weight_map)

    def fdk_recon(self, sinogram, filter_name="ramp", output_sharded=False):
        """
        Perform FDK reconstruction on the given sinogram.

        Our implementation uses standard filtering of the sinogram, then uses the adjoint of the forward projector to
        perform the backprojection.  This is different from many implementations, in which the backprojection is not
        exactly the adjoint of the forward projection.  For a detailed theoretical derivation of this implementation,
        see the zip file linked at this page: https://mbirjax.readthedocs.io/en/latest/theory.html

        Note:
            The ``pi / num_views`` angular weight in the filter assumes EQUALLY SPACED views over
            the conventional full angular range and applies no short-scan (Parker-style) redundancy
            weighting.  Translation tomography is an inherently limited-angle geometry, so this
            direct FDK reconstruction is only approximate; it is intended as an initializer for the
            iterative ``recon()``, which absorbs a global angular mis-weighting in a few iterations.

        Args:
            sinogram (numpy or jax array): The input sinogram with shape (num_views, num_rows, num_channels).
            filter_name (string, optional): Name of the filter to be used. Defaults to "ramp"
            output_sharded (bool, optional): If False (default), return a numpy array; if True,
                return the view-sharded device form (on a single device the same either way).

        Returns:
            recon (numpy or jax array): The reconstructed volume after FDK reconstruction.
        """

        # Keep the pipeline on-device (matches ParallelBeamModel.fbp_recon): fdk_filter returns the
        # view-sharded form (no host round-trip), back_project consumes it directly and decides the
        # output form.  fdk_filter shards a plain input itself, so no explicit entry shard is needed.
        filtered_sinogram = self.fdk_filter(sinogram, filter_name=filter_name, output_sharded=True)
        recon = self.back_project(filtered_sinogram, output_sharded=output_sharded)

        return recon

    def _get_estimate_of_recon_std(self, sinogram, sino_indicator):
        """
        Estimate the standard deviation of the reconstruction from the sinogram.  This is used to scale sigma_prox and
        sigma_x in MBIR reconstruction.
        This version accounts for anisotropic row pitch in translation geometry

        Args:
            sinogram (ndarray): 3D jax array containing sinogram with shape (num_views, num_det_rows, num_det_channels).
            sino_indicator (ndarray): a binary mask that indicates the region of sinogram support; same shape as sinogram.
        """
        # Get parameters
        delta_voxel = self.get_params('delta_voxel')
        recon_shape = self.get_params('recon_shape')
        voxel_row_aspect = self.get_params('voxel_row_aspect')

        delta_voxel_row = voxel_row_aspect * delta_voxel

        # Compute the typical magnitude of a sinogram value
        typical_sinogram_value = np.average(np.abs(sinogram), weights=sino_indicator)

        # Compute a typical projection path length
        # For TCT, we will assume that the projections are along the row direction,
        # and we will assume that the object fills approximately half the distance along the rows.
        fraction_of_fill = 0.5
        typical_path_length = fraction_of_fill * recon_shape[0] * delta_voxel_row

        # Compute a typical recon value by dividing average sinogram value by a typical projection path length
        recon_std = typical_sinogram_value / typical_path_length

        return recon_std