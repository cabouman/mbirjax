import jax
import jax.numpy as jnp
from functools import partial
from collections import namedtuple
from mbirjax import TomographyModel, ParameterHandler


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
        angles (jnp.ndarray):
            A 1D array of projection angles, in radians, specifying the angle of each projection relative to the origin.
        source_detector_dist (float): Distance between the X-ray source and the detector in units of ALU.
        source_iso_dist (float): Distance between the X-ray source and the center of rotation in units of ALU.
        recon_slice_offset (float, optional, default=0): Vertical offset of the image in ALU.
            If recon_slice_offset is positive, we reconstruct the region below iso.
        det_rotation (float, optional, default=0):  Angle in radians between the projection of the object rotation axis
            and the detector vertical axis, where positive describes a clockwise rotation of the detector as seen from the source.
    """

    def __init__(self, sinogram_shape, angles, source_detector_dist, source_iso_dist,
                 recon_slice_offset=0.0, det_rotation=0.0):
        # Convert the view-dependent vectors to an array
        # This is more complicated than needed with only a single view-dependent vector but is included to
        # illustrate the process as shown in TemplateModel
        view_dependent_vecs = [vec.flatten() for vec in [angles]]
        try:
            view_params_array = jnp.stack(view_dependent_vecs, axis=1)
        except ValueError as e:
            raise ValueError("Incompatible view dependent vector lengths:  all view-dependent vectors must have the "
                             "same length.")

        super().__init__(sinogram_shape, view_params_array=view_params_array,
                         source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist,
                         recon_slice_offset=recon_slice_offset, det_rotation=det_rotation)

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
        params = ParameterHandler.load_param_dict(filename, values_only=True)
        angles = params['view_params_array']
        del params['view_params_array']
        return cls(angles=angles, **params)

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
            ['delta_det_row', 'delta_det_channel', 'det_row_offset', 'det_channel_offset', 'det_rotation',
             'source_detector_dist', 'delta_voxel', 'recon_slice_offset']
        geometry_param_values = self.get_params(geometry_param_names)

        # Then get additional parameters:
        geometry_param_names += ['magnification', 'psf_radius']
        geometry_param_values.append(self.get_magnification())
        geometry_param_values.append(self.get_psf_radius())

        # Then create a namedtuple to access parameters by name in a way that can be jit-compiled.  
        GeometryParams = namedtuple('GeometryParams', geometry_param_names)
        geometry_params = GeometryParams(*tuple(geometry_param_values))

        return geometry_params

    def get_psf_radius(self):
        """Computes the integer radius of the PSF kernel for cone beam projection.
        """
        delta_det_row, delta_det_channel, source_detector_dist, recon_shape, delta_voxel = self.get_params(
            ['delta_det_row', 'delta_det_channel', 'source_detector_dist', 'recon_shape', 'delta_voxel'])
        magnification = self.get_magnification()

        # Compute minimum detector pitch
        delta_det = jnp.minimum(delta_det_row, delta_det_channel)

        # Compute maximum magnification
        if jnp.isinf(source_detector_dist):
            max_magnification = 1
        else:
            source_to_iso_dist = source_detector_dist/magnification
            source_to_closest_pixel = source_to_iso_dist - jnp.maximum(recon_shape[0], recon_shape[1])*delta_voxel
            max_magnification = source_detector_dist/source_to_closest_pixel

        # Compute the maximum number of detector rows/channels on either side of the center detector hit by a voxel
        psf_radius = int(jnp.ceil(jnp.ceil((delta_voxel*max_magnification/delta_det))/2))

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

        # Define the horizontal projector, which will be vmapped over slices.
        def project_slice(sinogram_view_row, voxel_values_slice):
            for n_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius+1):
                n = n_p_center + n_offset
                abs_delta_p_c_n = jnp.abs(n_p - n)
                L_p_c_n = jnp.clip((W_p_c + 1) / 2 - abs_delta_p_c_n, 0, L_max)
                A_chan_n = gp.delta_voxel * L_p_c_n / cos_alpha_p_xy
                A_chan_n *= (n >= 0) * (n < num_det_channels)
                sinogram_view_row = sinogram_view_row.at[n].add(A_chan_n * voxel_values_slice)

            return sinogram_view_row

        sinogram_view = jax.vmap(project_slice, in_axes=(0, 1))(sinogram_view, voxel_values)

        return sinogram_view

    @staticmethod
    def forward_vertical_fan_one_pixel_to_one_view(voxel_values, pixel_index, angle, projector_params):
        """
        Apply a fan beam forward projection in the vertical direction to the pixel determined by indices
        into the flattened array of size num_rows x num_cols.  This returns a vector obtained from the projection of
        the original voxel cylinder onto a detector column, so the output vector has length num_det_rows.

        Args:
            voxel_values (jax array):  2D array of shape (num_pixels, num_recon_slices) of voxel values, where
                voxel_values[i, j] is the value of the voxel in slice j at the location determined by indices[i].
            pixel_index (int):  Index into the flattened array of size num_rows x num_cols.
            angle (float):  Angle for this view.
            projector_params (namedtuple):  tuple of (sinogram_shape, recon_shape, get_geometry_params())

        Returns:
            jax array of shape (num_pixels, num_det_rows)

        Note:
            This is a helper function used in vmap in :meth:`ConeBeamModel.forward_vertical_fan_pixel_batch_to_one_view`
        This method has the same signature and output as that method, except single int pixel_index is used
        in place of the 1D pixel_indices, and likewise only a single voxel cylinder is returned.
        """
        # Get all the geometry parameters - we use gp since geometry parameters is a named tuple and we'll access
        # elements using, for example, gp.delta_det_channel, so a longer name would be clumsy.
        gp = projector_params.geometry_params
        num_views, num_det_rows, num_det_channels = projector_params.sinogram_shape

        # Get the data needed for vertical projection
        m_p, m_p_center, W_p_r, cos_alpha_p_z = ConeBeamModel.compute_vertical_data_single_pixel(pixel_index, angle,
                                                                                                 projector_params)
        L_max = jnp.minimum(1, W_p_r)

        # Allocate the output cylinder
        new_voxel_cylinder = jnp.zeros(num_det_rows)

        # Do the vertical projection
        for m_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius+1):
            m = m_p_center + m_offset
            abs_delta_p_r_m = jnp.abs(m_p - m)
            L_p_r_m = jnp.clip((W_p_r + 1) / 2 - abs_delta_p_r_m, 0, L_max)
            A_row_m = L_p_r_m / cos_alpha_p_z
            A_row_m *= (m >= 0) * (m < num_det_rows)
            new_voxel_cylinder = new_voxel_cylinder.at[m].add(A_row_m * voxel_values)

        return new_voxel_cylinder

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

        # Define the horizontal projector, which will be vmapped over slices.
        def project_slice(det_voxel_row, sinogram_view_row):
            for n_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius+1):
                n = n_p_center + n_offset
                abs_delta_p_c_n = jnp.abs(n_p - n)
                L_p_c_n = jnp.clip((W_p_c + 1) / 2 - abs_delta_p_c_n, 0, L_max)
                A_chan_n = gp.delta_voxel * L_p_c_n / cos_alpha_p_xy
                A_chan_n *= (n >= 0) * (n < num_det_channels)
                A_chan_n = A_chan_n ** coeff_power
                det_voxel_row = jnp.add(det_voxel_row, A_chan_n * sinogram_view_row[n])

            return det_voxel_row

        det_voxel_cylinder = jax.vmap(project_slice, in_axes=(1, 0), out_axes=1)(det_voxel_cylinder, sinogram_view)

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

        # Get the data needed for vertical projection
        m_p, m_p_center, W_p_r, cos_alpha_p_z = ConeBeamModel.compute_vertical_data_single_pixel(pixel_index, angle,
                                                                                                 projector_params)
        L_max = jnp.minimum(1, W_p_r)

        # Allocate space
        recon_voxel_cylinder = jnp.zeros(num_recon_slices)

        # Do the vertical projection for this pixel.
        for m_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius+1):
            m = m_p_center + m_offset
            abs_delta_p_r_m = jnp.abs(m_p - m)
            L_p_r_m = jnp.clip((W_p_r + 1) / 2 - abs_delta_p_r_m, 0, L_max)
            A_row_m = L_p_r_m / cos_alpha_p_z
            A_row_m *= (m >= 0) * (m < num_det_rows)
            A_row_m = A_row_m ** coeff_power
            recon_voxel_cylinder = jnp.add(recon_voxel_cylinder, A_row_m * detector_column_values[m])

        return recon_voxel_cylinder

    @staticmethod
    def compute_vertical_data_single_pixel(pixel_index, angle, projector_params):
        """
        Compute the quantities m_p, m_p_center, W_p_r, cos_alpha_p_z needed for vertical projection.

        Args:
            pixel_index (int):  Index into flattened array of size num_rows x num_cols.
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
        slice_index = jnp.arange(num_recon_slices)

        x_p, y_p, z_p = ConeBeamModel.recon_ijk_to_xyz(row_index, col_index, slice_index, gp.delta_voxel,
                                                       recon_shape, gp.recon_slice_offset, angle)

        # Convert from xyz to coordinates on detector
        u_p, v_p, pixel_mag = ConeBeamModel.geometry_xyz_to_uv_mag(x_p, y_p, z_p, gp.source_detector_dist, gp.magnification)
        # Convert from uv to index coordinates in detector and get the vector of center detector rows for this cylinder
        m_p, _ = ConeBeamModel.detector_uv_to_mn(u_p, v_p, gp.delta_det_channel, gp.delta_det_row, gp.det_channel_offset,
                                                 gp.det_row_offset, num_det_rows, num_det_channels, gp.det_rotation)
        m_p_center = jnp.round(m_p).astype(int)

        # Compute vertical cone angle of pixel
        phi_p = jnp.arctan2(v_p, gp.source_detector_dist)

        # Compute cos alpha for row and columns
        cos_phi_p = jnp.cos(phi_p)
        cos_alpha_p_z = jnp.maximum(jnp.abs(cos_phi_p), jnp.abs(jnp.sin(phi_p)))

        # Get the length of projection of flattened voxel on detector (in fraction of detector size)
        W_p_r = pixel_mag * (gp.delta_voxel / gp.delta_det_row) * cos_alpha_p_z / cos_phi_p

        vertical_data = (m_p, m_p_center, W_p_r, cos_alpha_p_z)

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
        slice_index = jnp.arange(num_recon_slices)

        x_p, y_p, _ = ConeBeamModel.recon_ijk_to_xyz(row_index, col_index, slice_index, gp.delta_voxel,
                                                       recon_shape, gp.recon_slice_offset, angle)

        # Convert from xyz to coordinates on detector
        # pixel_mag should be kept in terms of magnification to allow for source_detector_dist = jnp.Inf
        pixel_mag = 1 / (1 / gp.magnification - y_p / gp.source_detector_dist)
        # Compute the physical position that this voxel projects onto the detector
        u_p = pixel_mag * x_p
        det_center_channel = (num_det_channels - 1) / 2.0  # num_of_cols

        # Calculate indices on the detector grid
        n_p = (u_p / gp.delta_det_channel) + det_center_channel + gp.det_channel_offset
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
    def recon_ijk_to_xyz(i, j, k, delta_voxel, recon_shape,
                         recon_slice_offset, angle):
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
    @partial(jax.jit, static_argnames='det_rotation')
    def detector_uv_to_mn(u, v, delta_det_channel, delta_det_row, det_channel_offset, det_row_offset, num_det_rows,
                          num_det_channels, det_rotation=0):
        """
        Convert (u, v) detector coordinates to fractional indices (m, n) into the detector.

        Note:
            This version does not account for nonzero detector rotation.
        """
        if det_rotation != 0:
            raise ValueError('Nonzero det_rotation is not implemented.')

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
        n = (u_tilde + det_channel_offset) / delta_det_channel + det_center_channel

        return m, n
