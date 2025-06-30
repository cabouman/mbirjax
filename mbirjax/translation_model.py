import warnings
import jax
import jax.numpy as jnp
from functools import partial
from collections import namedtuple

import numpy as np

import mbirjax


class TranslationModel(mbirjax.TomographyModel):
    """
    Implements a tomography model using translation-based cone-beam projections.

    This class inherits from `mbirjax.TomographyModel` and specializes it for scenarios where the object undergoes
    translations instead of rotations during image acquisition. The forward and backward projections are computed using
    per-view translation vectors.

    Note: This geometry reduces regularization along the row (y) direction by setting `qggmrf_nbr_wts=(0.5, 1, 1)`,
    which helps compensates for anisotropy intrinsic to the translation geometry.

    Args:
        sinogram_shape (tuple): Shape of the sinogram as (num_views, num_rows, num_channels),
            where 'num_views' is the number of translation steps, 'num_rows' is the number of detector rows,
            and 'num_channels' is the number of detector columns.
        translation_vectors (jax array or numpy array): A (num_views, 3) array of translations (x, y, z) in ALUs.
            Each vector specifies how the object is translated for each view.
            Positive x shifts the object left, z shifts up, and y shifts away from the source.
        source_detector_dist (float): Distance from the X-ray source to the detector.
        source_iso_dist (float): Distance from the X-ray source to the isocenter.


    Usage Notes:
        - Reconstruction parameters should typically be set using `set_params`.
        - The row voxel thickness can be adjusted with `delta_recon_row`, while the voxel column and slice spacing
          can be controlled via the base parameter `delta_voxel`.

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
            model.set_params(delta_recon_row=2.0)
            model.auto_set_recon_size(sinogram_shape)
    """
    def __init__(self, sinogram_shape, translation_vectors, source_detector_dist, source_iso_dist):

        self.use_ror_mask = False
        self.entries_per_cylinder_batch = 128
        translation_vectors = jnp.asarray(translation_vectors)
        view_params_name = 'translation_vectors'

        super().__init__(sinogram_shape, translation_vectors=translation_vectors, source_detector_dist=source_detector_dist,
                         source_iso_dist=source_iso_dist, view_params_name=view_params_name, qggmrf_nbr_wts=(0.5, 1, 1,))

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

    def get_geometry_parameters(self):
        """
        Function to get a list of the primary geometry parameters for translation model projection.

        Returns:
            namedtuple of required geometry parameters.
        """
        # First get the parameters managed by ParameterHandler
        geometry_param_names = \
            ['delta_det_row', 'delta_det_channel', 'det_row_offset', 'det_channel_offset',
             'source_detector_dist', 'delta_voxel', 'delta_recon_row']
        geometry_param_values = self.get_params(geometry_param_names)

        # Then get additional parameters:
        geometry_param_names += ['magnification', 'psf_radius',
                                 'entries_per_cylinder_batch']
        geometry_param_values.append(self.get_magnification())
        geometry_param_values.append(self.get_psf_radius())
        geometry_param_values.append(self.entries_per_cylinder_batch)

        # Then create a namedtuple to access parameters by name in a way that can be jit-compiled.
        GeometryParams = namedtuple('GeometryParams', geometry_param_names)
        geometry_params = GeometryParams(*tuple(geometry_param_values))

        return geometry_params

    def get_psf_radius(self):
        """
        Compute the integer radius of the PSF kernel for cone beam projection.
        """
        delta_det_row, delta_det_channel, source_iso_dist, source_detector_dist, recon_shape, delta_voxel, delta_recon_row, translation_vectors = self.get_params(
            ['delta_det_row', 'delta_det_channel', 'source_iso_dist', 'source_detector_dist', 'recon_shape', 'delta_voxel', 'delta_recon_row', 'translation_vectors'])
        magnification = self.get_magnification()

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
            source_to_closest_pixel = source_iso_dist - (0.5 * recon_shape[0] * delta_recon_row) - max_translation[1]
            # Determine the maximum magnification.
            max_magnification = source_detector_dist / source_to_closest_pixel

            if source_to_closest_pixel < 0:
                raise ValueError('Reconstruction volume extends into source - no valid projection in this case.')

        # Compute the maximum number of detector rows/channels on either side of the center detector hit by a voxel
        psf_radius = int(jnp.ceil(jnp.ceil((delta_voxel * max_magnification / delta_det)) / 2))
        if psf_radius > 4:
            warnings.warn('A single voxel may project onto 100 or more detector elements, which may lead to artifacts. Consider using smaller voxels.')
        return psf_radius

    def auto_set_recon_size(self, sinogram_shape, no_compile=True, no_warning=False):
        """ Compute the automatic recon shape translation reconstruction.
        """
        # Get model parameters
        source_detector_dist, source_iso_dist = self.get_params(['source_detector_dist', 'source_iso_dist'])
        delta_det_row, delta_det_channel = self.get_params(['delta_det_row', 'delta_det_channel'])
        magnification = self.get_magnification()
        delta_voxel = self.get_params('delta_voxel')
        num_views, num_det_rows, num_det_channels = sinogram_shape
        translation_vectors = self.get_params('translation_vectors')

        # Calculate the width and height of the detector in ALU
        detect_box = jnp.array([delta_det_channel*num_det_channels, delta_det_row*num_det_rows])

        # Compute 1/2 the cone angle in radians
        half_cone_angle = jnp.arctan2(jnp.max(detect_box), 2*source_detector_dist)

        # Compute the row pitch based on a heuristic
        delta_recon_row = np.sqrt(2) * (delta_voxel / jnp.tan(half_cone_angle))
        delta_recon_row = float(delta_recon_row)

        # Compute cube = (width, depth, height) of the scanned region in ALU
        max_translation = jnp.amax(translation_vectors, axis=0)  # Translate object right/up when positive
        min_translation = jnp.amin(translation_vectors, axis=0)  # Translate object left/down when negative
        cube = max_translation - min_translation

        # Compute recon_box = (width, height) of the reconstruction box in voxels
        recon_box = jnp.ceil((jnp.array([cube[0], cube[2]]) + detect_box/magnification/2)/delta_voxel)

        # Use a heuristic to determine a reasonable number of slices
        num_recon_rows = jnp.ceil((num_views*num_det_channels*num_det_rows)/(recon_box[0]*recon_box[1]))
        # Make sure the object extends no further than half way to the source
        max_recon_rows = jnp.floor((source_iso_dist - cube[1]) / delta_recon_row)
        if max_recon_rows < 1:
            print(f"[Error] Computed max_recon_rows = {max_recon_rows} < 1. This suggests the object extends beyond the source.")
        num_recon_rows = jnp.minimum(num_recon_rows, max_recon_rows)

        # Set the parameters to their computed values
        num_recon_cols, num_recon_slices = recon_box
        num_recon_cols = int(num_recon_cols)
        num_recon_rows = int(num_recon_rows)
        num_recon_slices = int(num_recon_slices)
        recon_shape = (num_recon_rows, num_recon_cols, num_recon_slices)
        self.set_params(no_compile=no_compile, no_warning=no_warning, recon_shape=recon_shape, delta_recon_row=delta_recon_row)


    @staticmethod
    @partial(jax.jit, static_argnames='projector_params')
    def forward_project_pixel_batch_to_one_view(voxel_values, pixel_indices, translation_vector, projector_params):
        """
        Forward project a set of voxels determined by indices into the flattened array of size num_rows x num_cols.

        Args:
            voxel_values (jax array):  2D array of shape (num_indices, num_slices) of voxel values, where
                voxel_values[i, j] is the value of the voxel in slice j at the location determined by indices[i].
            pixel_indices (jax array of int):  1D vector of indices into flattened array of size num_rows x num_cols.
            translation_vector (jax array of floats):  1D translation vector in ALU units for this view.
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
        sinogram_view = horizontal_fan_projector(new_voxel_values, pixel_indices, translation_vector, projector_params)

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
    def forward_horizontal_fan_pixel_batch_to_one_view(voxel_values, pixel_indices, translation_vector, projector_params):
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
        n_p, n_p_center, W_p_c, cos_theta_p = TranslationModel.compute_horizontal_data(pixel_indices, translation_vector, projector_params)
        L_max = jnp.minimum(1, W_p_c)

        # Allocate the sinogram array
        sinogram_view = jnp.zeros((num_det_rows, num_det_channels))

        # Do the horizontal projection
        for n_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius + 1):
            n = n_p_center + n_offset
            abs_delta_p_c_n = jnp.abs(n_p - n)
            L_p_c_n = jnp.clip((W_p_c + 1) / 2 - abs_delta_p_c_n, 0, L_max)
            A_chan_n = L_p_c_n / cos_theta_p
            A_chan_n *= (n >= 0) * (n < num_det_channels)
            sinogram_view = sinogram_view.at[:, n].add(A_chan_n.reshape((1, -1)) * voxel_values.T)

        return sinogram_view

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
        num_slices = voxel_cylinder.shape[0]

        # From pixel index, compute y and pixel_mag
        y, pixel_mag = TranslationModel.compute_y_mag_for_pixel(pixel_index, translation_vector, recon_shape,
                                                                projector_params)

        # The code above depends only on the pixel - a single point.  z is a potentially large vector
        # Here we compute cos_phi_p:  1 / cos_phi_p determines the projection length through a voxel
        # For computational efficiency, we use that to scale the voxel_cylinder values.
        ### possibly convert to a jitted function with donate_argnames to avoid copies for z, v, phi_p, cos_phi_p
        k = jnp.arange(len(voxel_cylinder))
        z = gp.delta_voxel * (k - (num_recon_slices - 1) / 2.0) - translation_vector[2] # recon_ijk_to_xyz
        v = pixel_mag * z  # geometry_xyz_to_uv_mag
        # Compute vertical cone angle of voxels
        phi_p = jnp.arctan2(v, gp.source_detector_dist)  # compute_vertical_data_single_pixel
        cos_phi_p = jnp.cos(phi_p)  # We assume the vertical angle |phi_p| < 45 degrees so cos_alpha_p_z = cos_phi_p
        scaled_voxel_values = voxel_cylinder / cos_phi_p
        # End

        # Get the length of projection of detector on vertical voxel profile (in fraction of voxel size)
        # This is also the slope of the map from voxel index to detector index
        W_p_r = (pixel_mag * gp.delta_voxel) / gp.delta_det_row
        slope_k_to_m = W_p_r
        L_max = jnp.minimum(1, W_p_r)  # Maximum fraction of a detector that can be covered by one voxel.

        # Set up detector row indices array (0, 10, 20, ..., 10*num_slice_batches)
        det_rows_per_batch = gp.entries_per_cylinder_batch
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
            k_m = (z_m + translation_vector[2]) / gp.delta_voxel + (num_slices - 1) / 2.0
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
    @partial(jax.jit, static_argnames='projector_params')
    def back_project_one_view_to_pixel_batch(sinogram_view, pixel_indices, single_view_params, projector_params,
                                             coeff_power=1):
        """
        Use vmap to do a backprojection from one view to multiple pixels (voxel cylinders).

        Args:
            sinogram_view (2D jax array): one view of the sinogram to be back projected.
                2D jax array of shape (num_det_rows)x(num_det_channels)
            pixel_indices (1D jax array of int):  indices into flattened array of size num_rows x num_cols.
            single_view_params: These are the view dependent parameters for the view being back projected.
            projector_params (namedtuple): tuple of (sinogram_shape, recon_shape, get_geometry_params()).
            coeff_power (int): backproject using the coefficients of (A_ij ** coeff_power).
                Normally 1, but should be 2 when computing Hessian diagonal.

        Returns:
            The voxel values for all slices at the input index (i.e., a voxel cylinder) obtained by backprojecting
            the input sinogram view.
        """

        vertical_fan_projector = TranslationModel.back_vertical_fan_one_view_to_pixel_batch
        horizontal_fan_projector = TranslationModel.back_horizontal_fan_one_view_to_pixel_batch

        det_voxel_cylinder = horizontal_fan_projector(sinogram_view, pixel_indices, single_view_params,
                                                      projector_params, coeff_power=coeff_power)
        back_projection = vertical_fan_projector(det_voxel_cylinder, pixel_indices, single_view_params,
                                                 projector_params, coeff_power=coeff_power)

        return back_projection

    @staticmethod
    def back_horizontal_fan_one_view_to_pixel_batch(sinogram_view, pixel_indices, translation_vector,
                                                    projector_params, coeff_power=1):
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
        n_p, n_p_center, W_p_c, cos_theta_p = TranslationModel.compute_horizontal_data(pixel_indices, translation_vector, projector_params)
        L_max = jnp.minimum(1, W_p_c)

        # Allocate the voxel cylinder array
        det_voxel_cylinder = jnp.zeros((num_pixels, num_det_rows))

        # Do the horizontal projection
        for n_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius + 1):
            n = n_p_center + n_offset
            abs_delta_p_c_n = jnp.abs(n_p - n)
            L_p_c_n = jnp.clip((W_p_c + 1) / 2 - abs_delta_p_c_n, 0, L_max)
            A_chan_n = L_p_c_n / cos_theta_p
            A_chan_n *= (n >= 0) * (n < num_det_channels)
            A_chan_n = A_chan_n ** coeff_power
            det_voxel_cylinder = jnp.add(det_voxel_cylinder, A_chan_n.reshape((-1, 1)) * sinogram_view[:, n].T)

        return det_voxel_cylinder

    @staticmethod
    def back_vertical_fan_one_view_to_pixel_batch(det_voxel_cylinder, pixel_indices, single_view_params,
                                                  projector_params, coeff_power=1):
        """
        Apply a fan beam backward projection in the vertical direction to the pixel determined by indices
        into the flattened array of size num_rows x num_cols.  This returns a vector obtained from the projection of
        the detector-based voxel cylinders onto voxel cylinders in recon space, so the output vector has length
        num_recon_slices.

        Args:
            det_voxel_cylinder (2D jax array): 2D array of shape (num_pixels, num_det_rows) of voxel values, where
                det_voxel_cylinder[i, j] is the value of the voxel in row j at the location determined by indices[i].
            pixel_indices (1D jax array of int):  indices into flattened array of size num_rows x num_cols.
            single_view_params: These are the view dependent parameters for the view being back projected.
            projector_params (namedtuple): tuple of (sinogram_shape, recon_shape, get_geometry_params()).
            coeff_power (int): backproject using the coefficients of (A_ij ** coeff_power).
                Normally 1, but should be 2 when computing Hessian diagonal.

        Returns:
            2D jax array of shape (num_pixels, num_recon_slices) of voxel values.
        """
        pixel_map = jax.vmap(TranslationModel.back_vertical_fan_one_view_to_one_pixel,
                             in_axes=(0, 0, None, None, None))
        new_pixels = pixel_map(det_voxel_cylinder, pixel_indices, single_view_params, projector_params, coeff_power)

        return new_pixels

    @staticmethod
    def back_vertical_fan_one_view_to_one_pixel(detector_column_values, pixel_index, translation_vector, projector_params,
                                                coeff_power=1):
        """
        Apply the back projection of a vertical fan beam transformation to a single voxel cylinder and return the column
        vector of the resulting values.

        Args:
            detector_column_values (1D jax array): 1D array of shape (num_det_rows,) of voxel values, where
                detector_column_values[i, j] is the value of the voxel in row j at the location determined by indices[i].
            pixel_index (int):  Index into flattened array of size num_rows x num_cols.
            translation_vector (jax array of floats):  1D translation vector in ALU units for this view.
            projector_params (namedtuple): tuple of (sinogram_shape, recon_shape, get_geometry_params()).
            coeff_power (int): backproject using the coefficients of (A_ij ** coeff_power).
                Normally 1, but should be 2 when computing Hessian diagonal.

        Returns:
            1D jax array of shape (num_recon_slices,) of voxel values.
        """
        # Get all the geometry parameters - we use gp since geometry parameters is a named tuple and we'll access
        # elements using, for example, gp.delta_det_channel, so a longer name would be clumsy.
        gp = projector_params.geometry_params

        num_views, num_det_rows, num_det_channels = projector_params.sinogram_shape
        recon_shape = projector_params.recon_shape
        num_recon_rows, num_recon_cols, num_recon_slices = recon_shape

        # Set up slice indices array (0, slices_per_batch, 2*slices_per_batch, ..., num_slice_batches*slices_per_batch)
        num_slices = num_recon_slices
        slices_per_batch = gp.entries_per_cylinder_batch
        slices_per_batch = min(slices_per_batch, num_slices)
        num_slice_batches = (num_slices + slices_per_batch - 1) // slices_per_batch
        slice_indices = slices_per_batch * jnp.arange(num_slice_batches)

        # Set up a function to map over the slices of the cylinder
        # Here we can use a map over subsections of the voxel cylinder because we are indexing by slice,
        # so there is no overlap from one section to the next.
        def create_voxel_cylinder_slices(start_index):
            # Allocate space
            new_cylinder = jnp.zeros(slices_per_batch)
            # Get the data needed for vertical projection
            cur_slice_indices = start_index + jnp.arange(slices_per_batch)
            m_p, m_p_center, W_p_r, cos_alpha_p_z = TranslationModel.compute_vertical_data_single_pixel(pixel_index, cur_slice_indices, translation_vector, projector_params)
            L_max = jnp.minimum(1, W_p_r)  # Maximum fraction of a detector that can be covered by one voxel.

            # Do the vertical projection
            for m_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius + 1):
                m = m_p_center + m_offset
                abs_delta_p_r_m = jnp.abs(m_p - m)  # Distance from projection of center of voxel to center of detector
                L_p_r_m = jnp.clip((W_p_r + 1) / 2 - abs_delta_p_r_m, 0, L_max)
                A_row_m = L_p_r_m / cos_alpha_p_z
                A_row_m *= (m >= 0) * (m < num_det_rows)
                A_row_m = A_row_m ** coeff_power
                new_cylinder = jnp.add(new_cylinder, A_row_m * detector_column_values[m])

            return new_cylinder, None

        recon_voxel_cylinder, _ = jax.lax.map(create_voxel_cylinder_slices, slice_indices)
        recon_voxel_cylinder = recon_voxel_cylinder.flatten()
        recon_voxel_cylinder = jax.lax.slice_in_dim(recon_voxel_cylinder, 0, num_recon_slices)
        return recon_voxel_cylinder

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

        # Convert the index into (i,j,k) coordinates corresponding to the indices into the 3D voxel array
        row_index, col_index = jnp.unravel_index(pixel_index, recon_shape[:2])
        # slice_indices = jnp.arange(num_recon_slices)

        x_p, y_p, z_p = TranslationModel.recon_ijk_to_xyz(row_index, col_index, slice_indices, recon_shape, gp.delta_voxel, gp.delta_recon_row, translation_vector)

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
        W_p_r = pixel_mag * (gp.delta_voxel / gp.delta_det_row)  # * cos_alpha_p_z / cos_phi_p

        vertical_data = (m_p, m_p_center, W_p_r, cos_phi_p)  # cos_alpha_p_z)

        return vertical_data

    @staticmethod
    def compute_horizontal_data(pixel_indices, translation_vector, projector_params):
        """
        Compute the quantities n_p, n_p_center, W_p_c, cos_alpha_p_xy needed for vertical projection.

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

        x_p, y_p, _ = TranslationModel.recon_ijk_to_xyz(row_index, col_index, slice_index, recon_shape, gp.delta_voxel, gp.delta_recon_row, translation_vector)

        # Convert from xyz to coordinates on detector
        # pixel_mag should be kept in terms of magnification to allow for source_detector_dist = jnp.Inf
        pixel_mag = 1 / (1 / gp.magnification - y_p / gp.source_detector_dist)
        # Compute the physical position that this voxel projects onto the detector
        u_p = pixel_mag * x_p
        det_center_channel = (num_det_channels - 1) / 2.0  # num_of_cols

        # Calculate indices on the detector grid
        n_p = (u_p + gp.det_channel_offset) / gp.delta_det_channel + det_center_channel  # Sync with detector_uv_to_mn
        n_p_center = jnp.round(n_p).astype(int)

        # Compute horizontal and vertical cone angle of pixel
        theta_p = jnp.arctan2(u_p, gp.source_detector_dist)

        # Compute cosine alpha
        cos_theta_p = jnp.cos(theta_p)

        # Compute projected voxel width along columns and rows (in fraction of detector size)
        W_p_c = pixel_mag * (gp.delta_voxel / gp.delta_det_channel)

        horizontal_data = (n_p, n_p_center, W_p_c, cos_theta_p)

        return horizontal_data

    @staticmethod
    def recon_ijk_to_xyz(i, j, k, recon_shape, delta_voxel, delta_recon_row, translation_vector):
        """
        Convert (i, j, k) indices into the recon volume to corresponding (x, y, z) coordinates.
        """
        num_recon_rows, num_recon_cols, num_recon_slices = recon_shape

        y = delta_recon_row * (i - (num_recon_rows - 1) / 2.0) - translation_vector[1]
        x = delta_voxel * (j - (num_recon_cols - 1) / 2.0) - translation_vector[0]
        z = delta_voxel * (k - (num_recon_slices - 1) / 2.0) - translation_vector[2]
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

        y = gp.delta_recon_row * (row_index - (num_recon_rows - 1) / 2.0) - translation_vector[1]

        # Convert from xyz to coordinates on detector
        pixel_mag = 1 / (1 / gp.magnification - y / gp.source_detector_dist)
        return y, pixel_mag


