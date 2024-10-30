import jax
import jax.numpy as jnp
from functools import partial
from collections import namedtuple
import warnings
import mbirjax
from mbirjax.tomography_utils import generate_filter  # see fdk_recon method


class ConeBeamModel(mbirjax.TomographyModel):
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
        angles (ndarray or jax array):
            A 1D array of projection angles, in radians, specifying the angle of each projection relative to the origin.
        source_detector_dist (float): Distance between the X-ray source and the detector in units of ALU.
        source_iso_dist (float): Distance between the X-ray source and the center of rotation in units of ALU.

    Note:
        One additional parameter for ConeBeamModel that can be set using set_params() is

        **recon_slice_offset** (float, default=0) -
        Vertical offset of the image in ALU. If recon_slice_offset is positive, we reconstruct the region below iso.
    """

    def __init__(self, sinogram_shape, angles, source_detector_dist, source_iso_dist):
        # Convert the view-dependent vectors to an array
        # This is more complicated than needed with only a single view-dependent vector but is included to
        # illustrate the process as shown in TemplateModel
        view_dependent_vecs = [vec.flatten() for vec in [angles]]
        self.bp_psf_radius = 1
        self.entries_per_cylinder_batch = 128
        self.slice_range_length = 0
        try:
            view_params_array = jnp.stack(view_dependent_vecs, axis=1)
        except ValueError as e:
            raise ValueError("Incompatible view dependent vector lengths:  all view-dependent vectors must have the "
                             "same length.")

        super().__init__(sinogram_shape, view_params_array=view_params_array,
                         source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist,
                         recon_slice_offset=0.0)

    @classmethod
    def from_file(cls, filename):
        """
        Construct a ConeBeamModel from parameters saved using save_params()

        Args:
            filename (str): Name of the file containing parameters to load.

        Returns:
            ConeBeamModel with the specified parameters.
        """
        # Load the parameters and convert to use the ConeBeamModel keywords.
        required_param_names = ['sinogram_shape', 'source_detector_dist', 'source_iso_dist']
        required_params, params = mbirjax.ParameterHandler.load_param_dict(filename, required_param_names, values_only=True)

        # Collect the required parameters into a separate dictionary and remove them from the loaded dict.
        angles = params['view_params_array']
        del params['view_params_array']
        required_params['angles'] = angles

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

    def verify_valid_params(self):
        """
        Check that all parameters are compatible for a reconstruction.

        Note:
            Raises ValueError for invalid parameters.
        """
        super().verify_valid_params()
        sinogram_shape, view_params_array = self.get_params(['sinogram_shape', 'view_params_array'])

        if view_params_array.shape[0] != sinogram_shape[0]:
            error_message = "Number view dependent parameter vectors must equal the number of views. \n"
            error_message += "Got {} for length of view-dependent parameters and "
            error_message += "{} for number of views.".format(view_params_array.shape[0], sinogram_shape[0])
            raise ValueError(error_message)

        # Check for cone angle > 45 degrees
        source_detector_dist, delta_det_row, det_row_offset = \
            self.get_params(['source_detector_dist', 'delta_det_row', 'det_row_offset'])
        half_detector_height = delta_det_row * sinogram_shape[1] / 2 + jnp.abs(det_row_offset)
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
        geometry_param_names = \
            ['delta_det_row', 'delta_det_channel', 'det_row_offset', 'det_channel_offset',
             'source_detector_dist', 'delta_voxel', 'recon_slice_offset']
        geometry_param_values = self.get_params(geometry_param_names)

        # Then get additional parameters:
        geometry_param_names += ['magnification', 'psf_radius', 'bp_psf_radius',
                                 'entries_per_cylinder_batch', 'slice_range_length']
        geometry_param_values.append(self.get_magnification())
        geometry_param_values.append(self.get_psf_radius())
        geometry_param_values.append(self.bp_psf_radius)
        geometry_param_values.append(self.entries_per_cylinder_batch)
        geometry_param_values.append(self.slice_range_length)

        # Then create a namedtuple to access parameters by name in a way that can be jit-compiled.  
        GeometryParams = namedtuple('GeometryParams', geometry_param_names)
        geometry_params = GeometryParams(*tuple(geometry_param_values))

        return geometry_params

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
            max_magnification = 1
            min_magnification = 1
        else:
            source_to_iso_dist = source_detector_dist / magnification
            # This isn't exactly the closest pixel since we're not accounting for rotation but for realistic cases it shouldn't matter.
            source_to_closest_pixel = source_to_iso_dist - jnp.maximum(recon_shape[0], recon_shape[1])*delta_voxel
            max_magnification = source_detector_dist / source_to_closest_pixel
            source_to_farthest_pixel = source_to_iso_dist + jnp.maximum(recon_shape[0], recon_shape[1])*delta_voxel
            min_magnification = source_detector_dist / source_to_farthest_pixel

        # Compute the maximum number of detector rows/channels on either side of the center detector hit by a voxel
        psf_radius = int(jnp.ceil(jnp.ceil((delta_voxel * max_magnification / delta_det)) / 2))
        # Then repeat for the back projection from detector elements to voxels.
        # The voxels closest to the detector will be covered the most by a given detector element.
        # With magnification=1, the number of voxels per element would be delta_det / delta_voxel
        max_voxels_per_detector = delta_det / (min_magnification * delta_voxel)
        self.bp_psf_radius = int(jnp.ceil(jnp.ceil(max_voxels_per_detector) / 2))

        self.slice_range_length = int(1 + 2 * self.bp_psf_radius + \
                                  jnp.ceil(self.entries_per_cylinder_batch * max_voxels_per_detector))

        return psf_radius

    def auto_set_recon_size(self, sinogram_shape, no_compile=True, no_warning=False):
        """ Compute the automatic recon shape cone beam reconstruction.
        """
        delta_det_row, delta_det_channel = self.get_params(['delta_det_row', 'delta_det_channel'])
        delta_voxel = self.get_params('delta_voxel')
        num_det_rows, num_det_channels = sinogram_shape[1:3]
        magnification = self.get_magnification()

        num_recon_rows = int(jnp.round(num_det_channels * ((delta_det_channel / delta_voxel) / magnification)))
        num_recon_cols = num_recon_rows
        num_recon_slices = int(jnp.round(num_det_rows * ((delta_det_row / delta_voxel) / magnification)))

        recon_shape = (num_recon_rows, num_recon_cols, num_recon_slices)
        self.set_params(no_compile=no_compile, no_warning=no_warning, recon_shape=recon_shape)

    @staticmethod
    @partial(jax.jit, static_argnames='projector_params')
    def forward_project_pixel_batch_to_one_view(voxel_values, pixel_indices, angle, projector_params):
        """
        Forward project a set of voxels determined by indices into the flattened array of size num_rows x num_cols.

        Args:
            voxel_values (jax array):  2D array of shape (num_indices, num_slices) of voxel values, where
                voxel_values[i, j] is the value of the voxel in slice j at the location determined by indices[i].
            pixel_indices (jax array of int):  1D vector of indices into flattened array of size num_rows x num_cols.
            angle (float):  Angle for this view
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

        new_voxel_values = vertical_fan_projector(voxel_values, pixel_indices, angle, projector_params)
        sinogram_view = horizontal_fan_projector(new_voxel_values, pixel_indices, angle, projector_params)

        return sinogram_view

    @staticmethod
    def forward_vertical_fan_pixel_batch_to_one_view(voxel_values, pixel_indices, angle, projector_params):
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
            angle (float):  Angle for this view
            projector_params (namedtuple):  tuple of (sinogram_shape, recon_shape, get_geometry_params())

        Returns:
            jax array of shape (num_pixels, num_det_rows)
        """
        pixel_map = jax.vmap(ConeBeamModel.forward_vertical_fan_one_pixel_to_one_view,
                             in_axes=(0, 0, None, None))
        new_pixels = pixel_map(voxel_values, pixel_indices, angle, projector_params)

        return new_pixels

    @staticmethod
    def forward_horizontal_fan_pixel_batch_to_one_view(voxel_values, pixel_indices, angle, projector_params):
        """
        Apply a horizontal fan beam transformation to a set of voxel cylinders. These cylinders are assumed to have
        slices aligned with detector rows, so that a horizontal fan beam maps a cylinder slice to a detector row.
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
        n_p, n_p_center, W_p_c, cos_alpha_p_xy = ConeBeamModel.compute_horizontal_data(pixel_indices, angle, projector_params)
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
    def forward_vertical_fan_one_pixel_to_one_view(voxel_cylinder, pixel_index, angle, projector_params):
        """
        Apply a fan beam forward projection in the vertical direction to the pixel determined by indices
        into the flattened array of size num_rows x num_cols.  This returns a vector obtained from the projection of
        the original voxel cylinder onto a detector column, so the output vector has length num_det_rows.

        Args:
            voxel_cylinder (jax array):  1D array of shape (num_recon_slices, ) of voxel values, where
                voxel_cylinder[j] is the value of the voxel in slice j at the location determined by pixel_index.
            pixel_index (int):  Index into the flattened array of size num_rows x num_cols.
            angle (float):  Angle for this view.
            projector_params (namedtuple):  tuple of (sinogram_shape, recon_shape, get_geometry_params())

        Returns:
            jax array of shape (num_det_rows,)

        Note:
            This is a helper function used in vmap in :meth:`ConeBeamModel.forward_vertical_fan_pixel_batch_to_one_view`
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
        y, pixel_mag = ConeBeamModel.compute_y_mag_for_pixel(pixel_index, angle, recon_shape, projector_params)

        # The code above depends only on the pixel - a single point.  z is a potentially large vector
        # Here we compute cos_phi_p:  1 / cos_phi_p determines the projection length through a voxel
        # For computational efficiency, we use that to scale the voxel_cylinder values.
        # TODO:  possibly convert to a jitted function with donate_argnames to avoid copies for z, v, phi_p, cos_phi_p
        k = jnp.arange(len(voxel_cylinder))
        z = gp.delta_voxel * (k - (num_slices - 1) / 2.0) + gp.recon_slice_offset  # recon_ijk_to_xyz
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
    @partial(jax.jit, static_argnames='projector_params')
    def back_project_one_view_to_pixel_batch(sinogram_view, pixel_indices, single_view_params, projector_params, coeff_power=1):
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

        vertical_fan_projector = ConeBeamModel.back_vertical_fan_one_view_to_pixel_batch
        horizontal_fan_projector = ConeBeamModel.back_horizontal_fan_one_view_to_pixel_batch

        det_voxel_cylinder = horizontal_fan_projector(sinogram_view, pixel_indices, single_view_params,
                                                      projector_params, coeff_power=coeff_power)
        back_projection = vertical_fan_projector(det_voxel_cylinder, pixel_indices, single_view_params,
                                                 projector_params, coeff_power=coeff_power)

        return back_projection

    @staticmethod
    def back_horizontal_fan_one_view_to_pixel_batch(sinogram_view, pixel_indices, angle,
                                                    projector_params, coeff_power=1):
        """
        Apply the back projection of a horizontal fan beam transformation to a single sinogram view
        and return the resulting voxel cylinders.

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
        n_p, n_p_center, W_p_c, cos_alpha_p_xy = ConeBeamModel.compute_horizontal_data(pixel_indices, angle, projector_params)
        L_max = jnp.minimum(1, W_p_c)

        # Allocate the voxel cylinder array
        det_voxel_cylinder = jnp.zeros((num_pixels, num_det_rows))

        # Do the horizontal projection
        for n_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius + 1):
            n = n_p_center + n_offset
            abs_delta_p_c_n = jnp.abs(n_p - n)
            L_p_c_n = jnp.clip((W_p_c + 1) / 2 - abs_delta_p_c_n, 0, L_max)
            A_chan_n = gp.delta_voxel * L_p_c_n / cos_alpha_p_xy
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
        pixel_map = jax.vmap(ConeBeamModel.back_vertical_fan_one_view_to_one_pixel,
                             in_axes=(0, 0, None, None, None))
        new_pixels = pixel_map(det_voxel_cylinder, pixel_indices, single_view_params, projector_params, coeff_power)

        return new_pixels

    @staticmethod
    def back_vertical_fan_one_view_to_one_pixel(detector_column_values, pixel_index, angle, projector_params,
                                                coeff_power=1):
        """
        Apply the back projection of a vertical fan beam transformation to a single voxel cylinder and return the column
        vector of the resulting values.

        Args:
            detector_column_values (1D jax array): 1D array of shape (num_det_rows,) of voxel values, where
                detector_column_values[i, j] is the value of the voxel in row j at the location determined by indices[i].
            pixel_index (int):  Index into flattened array of size num_rows x num_cols.
            angle (float): The projection angle in radians for this view.
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
        num_slices = detector_column_values.shape[0]
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
            m_p, m_p_center, W_p_r, cos_alpha_p_z = ConeBeamModel.compute_vertical_data_single_pixel(pixel_index, cur_slice_indices, angle,
                                                                                                     projector_params)
            L_max = jnp.minimum(1, W_p_r)  # Maximum fraction of a detector that can be covered by one voxel.

            # Do the vertical projection
            for m_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius+1):
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
    def compute_vertical_data_single_pixel(pixel_index, slice_indices, angle, projector_params):
        """
        Compute the quantities m_p, m_p_center, W_p_r, cos_alpha_p_z needed for vertical projection.

        Args:
            pixel_index (int):  Index into flattened array of size num_rows x num_cols.
            slice_indices (array of int): Indices into the recon slices.
            angle (float): The projection angle in radians for this view.
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

        x_p, y_p, z_p = ConeBeamModel.recon_ijk_to_xyz(row_index, col_index, slice_indices, gp.delta_voxel,
                                                       recon_shape, gp.recon_slice_offset, angle)

        # Convert from xyz to coordinates on detector
        u_p, v_p, pixel_mag = ConeBeamModel.geometry_xyz_to_uv_mag(x_p, y_p, z_p, gp.source_detector_dist, gp.magnification)
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
        W_p_r = pixel_mag * (gp.delta_voxel / gp.delta_det_row)   # * cos_alpha_p_z / cos_phi_p

        vertical_data = (m_p, m_p_center, W_p_r, cos_phi_p)  # cos_alpha_p_z)

        return vertical_data

    @staticmethod
    def compute_horizontal_data(pixel_indices, angle, projector_params):
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
        num_recon_rows, num_recon_cols, num_recon_slices = recon_shape

        # Convert the index into (i,j,k) coordinates corresponding to the indices into the 3D voxel array
        row_index, col_index = jnp.unravel_index(pixel_indices, recon_shape[:2])
        slice_index = jnp.arange(1)

        x_p, y_p, _ = ConeBeamModel.recon_ijk_to_xyz(row_index, col_index, slice_index, gp.delta_voxel,
                                                     recon_shape, gp.recon_slice_offset, angle)

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
        cos_alpha_p_xy = jnp.maximum(jnp.abs(jnp.cos(angle - theta_p)),
                                    jnp.abs(jnp.sin(angle - theta_p)))

        # Compute projected voxel width along columns and rows (in fraction of detector size)
        W_p_c = pixel_mag * (gp.delta_voxel / gp.delta_det_channel) * (cos_alpha_p_xy / jnp.cos(theta_p))

        horizontal_data = (n_p, n_p_center, W_p_c, cos_alpha_p_xy)

        return horizontal_data

    @staticmethod
    def recon_ijk_to_xyz(i, j, k, delta_voxel, recon_shape, recon_slice_offset, angle):
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

        z = delta_voxel * (k - (num_recon_slices - 1) / 2.0) + recon_slice_offset
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
    def compute_y_mag_for_pixel(pixel_index, angle, recon_shape, projector_params):

        gp = projector_params.geometry_params
        row_index, col_index = jnp.unravel_index(pixel_index, recon_shape[:2])

        # Compute the un-rotated coordinates relative to iso
        # Note the change in order from (i, j) to (y, x)!!
        y_tilde = gp.delta_voxel * (row_index - (recon_shape[0] - 1) / 2.0)
        x_tilde = gp.delta_voxel * (col_index - (recon_shape[1] - 1) / 2.0)

        # Precompute cosine and sine of view angle, then do the rotation
        cosine = jnp.cos(angle)  # length = num_views
        sine = jnp.sin(angle)  # length = num_views

        y = sine * x_tilde + cosine * y_tilde

        # Convert from xyz to coordinates on detector
        pixel_mag = 1 / (1 / gp.magnification - y / gp.source_detector_dist)
        return y, pixel_mag
    
    def fdk_recon(self, sinogram, filter_name="ramp"):
        # TODO write docstring, mention this only coveres planar detector in doc string.

        num_views, num_rows, num_channels = sinogram.shape
        filter = generate_filter(num_channels, filter_name=filter_name)
        source_detector_dist, source_iso_dist = self.get_params(['source_detector_dist', 'source_iso_dist'])

        # Define the s and v coordinates (in pixel units)
        s = (jnp.arange(num_channels) - (num_channels // 2))
        v = (jnp.arange(num_rows - 1, -1, -1) - (num_rows // 2))  # Reversed to start from the top

        # Create the meshgrid
        v_grid, s_grid = jnp.meshgrid(v, s, indexing='ij')

        # Combine the s and v grids into a single matrix of coordinate pairs
        coordinates = jnp.stack((s_grid, v_grid), axis=-1)

        pre_weight = source_iso_dist / jnp.sqrt(source_iso_dist**2 + jnp.sum(coordinates**2, axis=-1))

        # Apply the pre-weighting factor to the sinogram
        weighted_sinogram = sinogram * pre_weight

        # Define convolution for a single row (across its channels)
        def convolve_row(row):
            return jnp.convolve(row, filter, mode="valid")

        # Apply above convolve func across each row of a view
        def apply_convolution_to_view(view):
            return jax.vmap(convolve_row)(view)

        # Apply convolution across the channels of the weighted sinogram per each fixed view & row
        filtered_sinogram = jax.vmap(apply_convolution_to_view)(weighted_sinogram)

        recon = self.back_project(filtered_sinogram)
        recon *= (jnp.pi / num_views) * (source_detector_dist / source_iso_dist)
        # recon *= (1 / (2 * num_views * source_iso_dist ** 2))
        recon *= 2 * jnp.pi / num_views

        return recon

    def fdk_recon_old(self, sinogram):
        num_views, num_rows, num_channels = sinogram.shape
        source_detector_dist, source_iso_dist = self.get_params(['source_detector_dist', 'source_iso_dist'])

        # Create the ramp filter in time domain (sinc functions)
        ramp_filter = generate_filter(num_channels, filter="Ram-Lak")

        # Flatten the sinogram (from 3D to 2D)
        sinogram_flattened = sinogram.reshape(-1, sinogram.shape[-1])  # Shape: (num_views * num_rows, num_channels)

        # Define convolution func that will be applied to each channel
        def convolve_row(channel):
            return jnp.convolve(channel, ramp_filter, mode="valid")

        # Apply convolution to all channels, iterating over (views x rows)
        filtered_sinogram_flattened = jax.vmap(convolve_row)(sinogram_flattened)

        # Reshape the filtered sinogram back to (views, rows, channels)
        filtered_sinogram = filtered_sinogram_flattened.reshape(num_views, num_rows, -1)

        # Geometric weighting with the correct 1/(2 * num_views * src_orig^2) factor
        t_val = jnp.linspace(-num_channels / 2, num_channels / 2, num_channels)
        weighting = (
            source_detector_dist / jnp.sqrt(
            source_detector_dist ** 2 + t_val ** 2 + (jnp.arange(num_rows)[:, None] - num_rows / 2) ** 2
            )
            )
        weighting *= (1 / (2 * num_views * source_iso_dist ** 2))

        # Apply the weighting to the filtered sinogram, back project
        filtered_sinogram *= weighting[None, :, :]
        recon = self.back_project(filtered_sinogram)
        # recon *= jnp.pi / num_views  # scaling term

        return recon