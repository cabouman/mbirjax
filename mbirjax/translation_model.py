import warnings
import jax
import jax.numpy as jnp
from functools import partial
from collections import namedtuple
import mbirjax
from mbirjax import TomographyModel


class TranslationModel(TomographyModel):
    """
    A class designed for handling forward and backward projections using translations with a cone beam source. This extends
    :ref:`TomographyModelDocs`. This class offers specialized methods and parameters tailored for translation mode.

    This class inherits all methods and properties from the :ref:`TomographyModelDocs` and may override some
    to suit translation mode geometrical requirements. See the documentation of the parent class for standard methods
    like setting parameters and performing projections and reconstructions.

    Parameters not included in the constructor can be set using the set_params method of :ref:`TomographyModelDocs`.
    Refer to :ref:`TomographyModelDocs` documentation for a detailed list of possible parameters.

    Args:
        sinogram_shape (tuple):
            Shape of the sinogram as a tuple in the form `(views, rows, channels)`, where 'views' is the number of
            different translations, 'rows' correspond to the number of detector rows, and 'channels' index columns of
            the detector that are assumed to be aligned with the rotation axis.
        translation_vectors (jnp.ndarray):
            A 2D array of translation vectors in ALUs, specifying the translation of the object relative to the origin.  A vector of (x, y) translates the object right x units and down y units, as seen looking from source to detector.

    See Also
    --------
    TomographyModel : The base class from which this class inherits.
    """
    def __init__(self, sinogram_shape, translation_vectors, source_detector_dist, source_iso_dist, recon_width):
        # Convert the view-dependent vectors to an array
        # This is more complicated than needed with only a single view-dependent vector but is included to
        # illustrate the process as shown in TemplateModel
        view_dependent_vecs = [vec.flatten() for vec in translation_vectors]
        self.bp_psf_radius = 1
        self.entries_per_cylinder_batch = 128
        self.recon_width = recon_width
        try:
            view_params_array = jnp.stack(view_dependent_vecs, axis=0)
        except ValueError as e:
            raise ValueError("Incompatible view dependent vector lengths:  all view-dependent vectors must have the "
                             "same length.")

        super().__init__(sinogram_shape, view_params_array=view_params_array,
                         source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)

    @classmethod
    def from_file(cls, filename):
        """
        Construct a TranslationModel from parameters saved using save_params()

        Args:
            filename (str): Name of the file containing parameters to load.

        Returns:
            TranslationModel with the specified parameters.
        """
        # Load the parameters and convert to use the TranslationModel keywords.
        required_param_names = ['sinogram_shape', 'source_detector_dist', 'source_iso_dist']
        required_params, params = mbirjax.ParameterHandler.load_param_dict(filename, required_param_names, values_only=True)

        # Collect the required parameters into a separate dictionary and remove them from the loaded dict.
        translation_vectors = params['view_params_array']
        del params['view_params_array']
        required_params['translation_vectors'] = translation_vectors

        # Get an instance with the required parameters, then set any optional parameters
        new_model = cls(**required_params)
        new_model.set_params(**params)
        return new_model

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

    def get_psf_radius(self):
        """
        Compute the integer radius of the PSF kernel for cone beam projection.
        """
        delta_det_row, delta_det_channel, source_detector_dist, recon_shape, delta_voxel = self.get_params(
            ['delta_det_row', 'delta_det_channel', 'source_detector_dist', 'recon_shape', 'delta_voxel'])
        magnification = self.get_magnification()

        # Compute minimum detector pitch
        delta_det = jnp.minimum(delta_det_row, delta_det_channel)

        # Compute maximum magnification
        if jnp.isinf(source_detector_dist):
            raise ValueError('Distance from source to detector is infinite, which means all translated projections have the same information.')
        else:
            source_to_iso_dist = source_detector_dist / magnification
            # Determine the closest and farthest points from the source to determine max and min magnification.
            # iso is at the center of the recon volume, so we move half the length to get max/min distances.
            # This doesn't give exactly the closest pixel (which is really in the corner) since we're not accounting
            # for rotation, but for realistic cases it shouldn't matter.
            source_to_closest_pixel = source_to_iso_dist - 0.5 * jnp.maximum(recon_shape[0], recon_shape[1])*delta_voxel
            max_magnification = source_detector_dist / source_to_closest_pixel
            source_to_farthest_pixel = source_to_iso_dist + jnp.maximum(recon_shape[0], recon_shape[1])*delta_voxel
            min_magnification = source_detector_dist / source_to_farthest_pixel

        if max_magnification < 0:
            raise ValueError('Reconstruction volume extends into source - no valid projection in this case.')

        # Compute the maximum number of detector rows/channels on either side of the center detector hit by a voxel
        psf_radius = int(jnp.ceil(jnp.ceil((delta_voxel * max_magnification / delta_det)) / 2))
        # Then repeat for the back projection from detector elements to voxels.
        # The voxels closest to the detector will be covered the most by a given detector element.
        # With magnification=1, the number of voxels per element would be delta_det / delta_voxel
        max_voxels_per_detector = delta_det / (min_magnification * delta_voxel)
        self.bp_psf_radius = int(jnp.ceil(jnp.ceil(max_voxels_per_detector) / 2))
        if psf_radius > 1:
            warnings.warn('A single voxel may project onto several detector elements, which may lead to artifacts. Consider using smaller voxels.')
        return psf_radius

    def auto_set_recon_size(self, sinogram_shape, no_compile=True, no_warning=False):
        """ Compute the automatic recon shape cone beam reconstruction.
        """
        delta_det_row, delta_det_channel = self.get_params(['delta_det_row', 'delta_det_channel'])
        num_det_rows, num_det_channels = sinogram_shape[1:3]
        det_height = num_det_rows * delta_det_row
        det_width = num_det_channels * delta_det_channel

        # A positive det_row_offset causes a pixel seen in row i of the original sinogram to be seen in row i + index_offset
        # of the new sinogram (i.e., the detector moves up, the observed pixel moves down).
        det_row_offset, det_channel_offset = self.get_params(['det_row_offset', 'det_channel_offset'])
        # Get the bounds of the detector
        v_top = - (num_det_rows / 2) * delta_det_row - det_row_offset
        v_bottom = (num_det_rows / 2) * delta_det_row - det_row_offset
        u_left = - (num_det_channels / 2) * delta_det_channel - det_channel_offset
        u_right = (num_det_channels / 2) * delta_det_channel - det_channel_offset

        # Get the inner and outer coordinates of the recon along the iso line
        ymin_recon = - self.recon_width / 2
        source_detector_dist = self.get_params('source_detector_distance')
        max_magnification = source_detector_dist / ymin_recon
        # Determine the voxel dimensions in terms of the detector elements and magnification
        delta_voxel_x = delta_det_channel / max_magnification
        delta_voxel_z = delta_det_row / max_magnification
        delta_voxel_min = min(delta_voxel_x, delta_voxel_z)
        # Determine delta_voxel_y - same aspect ratio as the triangle source-detector-(detector max)
        ratio_vert = max(v_top, v_bottom) / source_detector_dist
        delta_voxel_y = ratio_vert * delta_voxel_z
        ratio_horiz = max(u_left, u_right) / source_detector_dist
        delta_voxel_y = min(delta_voxel_y, ratio_horiz * delta_voxel_x)

        # Determine the extent of the translations
        # TODO

        delta_voxel = self.get_params('delta_voxel')
        num_recon_rows = int(jnp.round(num_det_channels * ((delta_det_channel / delta_voxel) / magnification)))
        num_recon_cols = num_recon_rows
        num_recon_slices = int(jnp.round(num_det_rows * ((delta_det_row / delta_voxel) / magnification)))

        recon_shape = (num_recon_rows, num_recon_cols, num_recon_slices)
        self.set_params(no_compile=no_compile, no_warning=no_warning, recon_shape=recon_shape)

    @staticmethod
    @partial(jax.jit, static_argnames='projector_params')
    def forward_project_pixel_batch_to_one_view(voxel_values, pixel_indices, translation_vector, projector_params):
        """
        Forward project a set of voxels determined by indices into the flattened array of size num_rows x num_cols.

        Args:
            voxel_values (jax array):  2D array of shape (num_indices, num_slices) of voxel values, where
                voxel_values[i, j] is the value of the voxel in slice j at the location determined by indices[i].
            pixel_indices (jax array of int):  1D vector of indices into flattened array of size num_rows x num_cols.
            translation_vector (jax array of floats):  2D translation vector in ALU units for this view.
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
            translation_vector (jax array of floats):  2D translation vector in ALU units for this view.
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
            translation_vector (jax array of floats):  2D translation vector in ALU units for this view.
            projector_params (namedtuple):  tuple of (sinogram_shape, recon_shape, get_geometry_params())

        Returns:
            jax array of shape (num_det_rows, num_det_channels)
        """

        # Get all the geometry parameters - we use gp since geometry parameters is a named tuple and we'll access
        # elements using, for example, gp.delta_det_channel, so a longer name would be clumsy.
        gp = projector_params.geometry_params
        num_views, num_det_rows, num_det_channels = projector_params.sinogram_shape

        # Get the data needed for horizontal projection
        n_p, n_p_center, W_p_c, cos_alpha_p_xy = TranslationModel.compute_horizontal_data(pixel_indices, translation_vector, projector_params)
        L_max = jnp.minimum(1, W_p_c)

        # Allocate the sinogram array
        sinogram_view = jnp.zeros((num_det_rows, num_det_channels))

        # Do the horizontal projection
        for n_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius + 1):
            n = n_p_center + n_offset
            abs_delta_p_c_n = jnp.abs(n_p - n)
            L_p_c_n = jnp.clip((W_p_c + 1) / 2 - abs_delta_p_c_n, 0, L_max)
            A_chan_n = gp.delta_voxel * L_p_c_n / cos_alpha_p_xy
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
            translation_vector (jax array of floats):  2D translation vector in ALU units for this view.
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
        num_slices = voxel_cylinder.shape[0]

        # From pixel index, compute y and pixel_mag
        y, pixel_mag = TranslationModel.compute_y_mag_for_pixel(pixel_index, translation_vector, recon_shape, projector_params)

        # The code above depends only on the pixel - a single point.  z is a potentially large vector
        # Here we compute cos_phi_p:  1 / cos_phi_p determines the projection length through a voxel
        # For computational efficiency, we use that to scale the voxel_cylinder values.
        # TODO:  possibly convert to a jitted function with donate_argnames to avoid copies for z, v, phi_p, cos_phi_p
        k = jnp.arange(len(voxel_cylinder))
        z = gp.delta_voxel * (k - (num_slices - 1) / 2.0) + gp.recon_slice_offset + translation_vector[1]  # recon_ijk_to_xyz
        v = pixel_mag * z  # geometry_xyz_to_uv_mag
        # Compute vertical cone angle of voxels
        phi_p = jnp.arctan2(v, gp.source_detector_dist)  # compute_vertical_data_single_pixel
        cos_phi_p = jnp.cos(phi_p)  # We assume the vertical angle |phi_p| < 45 degrees so cos_alpha_p_z = cos_phi_p
        scaled_voxel_values = voxel_cylinder / cos_phi_p
        # End TODO

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
            k_m = (z_m - gp.recon_slice_offset) / gp.delta_voxel + (num_slices - 1) / 2.0
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
    def compute_horizontal_data(pixel_indices, translation_vector, projector_params):
        """
        Compute the quantities n_p, n_p_center, W_p_c, cos_alpha_p_xy needed for vertical projection.

        Args:
            pixel_indices (1D jax array of int):  indices into flattened array of size num_rows x num_cols.
            translation_vector (jax array of floats):  2D translation vector in ALU units for this view.
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
        slice_index = jnp.arange(1)

        x_p, y_p, _ = TranslationModel.recon_ijk_to_xyz(row_index, col_index, slice_index, gp.delta_voxel,
                                                     recon_shape, gp.recon_slice_offset, translation_vector)

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

        # Compute cos alpha for row and columns
        angle = 0.0  # For translation mode, we assume the object is aligned with the detector.
        cos_alpha_p_xy = jnp.maximum(jnp.abs(jnp.cos(angle - theta_p)),
                                    jnp.abs(jnp.sin(angle - theta_p)))

        # Compute projected voxel width along columns and rows (in fraction of detector size)
        W_p_c = pixel_mag * (gp.delta_voxel / gp.delta_det_channel) * (cos_alpha_p_xy / jnp.cos(theta_p))

        horizontal_data = (n_p, n_p_center, W_p_c, cos_alpha_p_xy)

        return horizontal_data


    @staticmethod
    def recon_ijk_to_xyz(i, j, k, delta_voxel, recon_shape, recon_slice_offset, translation_vector):
        """
        Convert (i, j, k) indices into the recon volume to corresponding (x, y, z) coordinates.
        """
        num_recon_rows, num_recon_cols, num_recon_slices = recon_shape

        # Compute the un-rotated coordinates relative to iso
        # Note the change in order from (i, j) to (y, x)!!
        y_tilde = delta_voxel * (i - (num_recon_rows - 1) / 2.0)
        x_tilde = delta_voxel * (j - (num_recon_cols - 1) / 2.0)

        # Precompute cosine and sine of view angle, then do the rotation
        angle = 0.0  # For translation mode, we assume the object is aligned with the detector.
        cosine = jnp.cos(angle)  # length = num_views
        sine = jnp.sin(angle)  # length = num_views

        x = cosine * x_tilde - sine * y_tilde + translation_vector[0]
        y = sine * x_tilde + cosine * y_tilde

        z = delta_voxel * (k - (num_recon_slices - 1) / 2.0) + recon_slice_offset + translation_vector[1]
        return x, y, z
