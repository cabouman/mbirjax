import warnings

import jax
import jax.numpy as jnp
from functools import partial
from mbirjax import TomographyModel


class ConeBeamModel(TomographyModel):
    """
    A class designed for handling forward and backward projections in a cone beam geometry, extending the
    :ref:`TomographyModelDocs`. This class offers specialized methods and parameters tailored for cone beam setups.

    This class inherits all methods and properties from the :ref:`TomographyModelDocs` and may override some
    to suit parallel beam geometrical requirements. See the documentation of the parent class for standard methods
    like setting parameters and performing projections and reconstructions.

    Args:
        sinogram_shape (tuple):
            Shape of the sinogram as a tuple in the form `(views, rows, channels)`, where 'views' is the number of
            different projection angles, 'rows' correspond to the number of detector rows, and 'channels' index columns of
            the detector that are assumed to be aligned with the rotation axis.
        angles (jnp.ndarray):
            A 1D array of projection angles, in radians, specifying the angle of each projection relative to the origin.
        source_detector_dist (float): Distance between the X-ray source and the detector in units of ALU.
        source_iso_dist (float): Distance between the X-ray source and the center of rotation in units of ALU.
        det_row_offset (float, optional, default=0): Distance = (detector iso row) - (center of detector rows) in ALU.
        det_channel_offset (float, optional, default=0): Distance = (detector iso channel) - (center of detector channels) in ALU.
        recon_slice_offset (float, optional, default=0): Vertical offset of the image in ALU.
            If recon_slice_offset is positive, we reconstruct the region below iso.
        det_rotation (float, optional, default=0):  Angle in radians between the projection of the object rotation axis
            and the detector vertical axis, where positive describes a clockwise rotation of the detector as seen from the source.
        **kwargs (dict):
            Additional keyword arguments that are passed to the :ref:`TomographyModelDocs` constructor. These can
            include settings and configurations specific to the tomography model such as noise models or image dimensions.
            Refer to :ref:`TomographyModelDocs` documentation for a detailed list of possible parameters.
    """

    def __init__(self, sinogram_shape, angles, source_detector_dist, source_iso_dist,
                 recon_slice_offset=0.0, det_rotation=0.0, **kwargs):
        # Convert the view-dependent vectors to an array
        # This is more complicated than needed with only a single view-dependent vector but is included to
        # illustrate the process as shown in TemplateModel
        view_dependent_vecs = [vec.flatten() for vec in [angles]]
        try:
            view_params_array = jnp.stack(view_dependent_vecs, axis=1)
        except ValueError as e:
            raise ValueError("Incompatible view dependent vector lengths:  all view-dependent vectors must have the "
                             "same length.")
        magnification = source_detector_dist / source_iso_dist
        super().__init__(sinogram_shape, view_params_array=view_params_array,
                         source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist,
                         recon_slice_offset=recon_slice_offset, det_rotation=det_rotation,
                         **kwargs)

    def get_magnification(self):
        """
        Returns the magnification for the cone beam geometry.

        Returns:
            magnification = source_detector_dist / source_iso_dist
        """
        source_detector_dist, source_iso_dist = self.get_params(['source_detector_dist', 'source_iso_dist'])
        magnification = source_detector_dist / source_iso_dist
        return magnification

    def verify_valid_params(self):
        """
        Check that all parameters are compatible for a reconstruction.
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
        Function to get a list of the primary geometry parameters for projection.

        Returns:
            List of required geometry parameters.
        """
        geometry_params = self.get_params(
            ['delta_det_row', 'delta_det_channel', 'det_row_offset', 'det_channel_offset', 'det_rotation',
             'source_detector_dist', 'delta_voxel','recon_slice_offset'])
        geometry_params.append(self.get_magnification())

        return geometry_params

    @staticmethod
    def back_project_one_view_to_pixel(sinogram_view, pixel_index, angle, projector_params, coeff_power=1):
        """
        Calculate the backprojection at a specified recon pixel cylinder given a sinogram view and model parameters.
        Also supports computation of the diagonal hessian when coeff_power = 2.

        Args:
            sinogram_view (jax array): one view of the sinogram to be back projected.
                2D jnp array of shape (num_det_rows)x(num_det_channels)
            pixel_index (int): Index of pixel cylinder to back project.  This index is converted to a 2D index
             using i, j = unravel_index(pixel_index, (num_recon_rows, num_recon_cols)).
            angle (float): The angle in radians for this view.
            projector_params (tuple):  tuple of (sinogram_shape, recon_shape, get_geometry_params()).
            coeff_power (int): Normally 1, but should be 2 when computing diagonal hessian 2.

        Returns:
            back_projection (jnp array): 1D array of length (number of slices) obtained by backprojecting the
                given view onto the voxel cylinder specified by the given pixel.
        """
        # Get the part of the system matrix and channel indices for this voxel cylinder
        sinogram_view_shape = (1,) + sinogram_view.shape  # Adjoin a leading 1 to indicate a single view sinogram
        view_projector_params = (sinogram_view_shape,) + projector_params[1:]

        # Compute sparse system matrices for rows and columns
        # Bij_value, Bij_channel, Cij_value, Cij_row are all shaped [(num pixels)*(num slices)]x(2p+1)
        pixel_index = jnp.array(pixel_index).reshape((1, 1))
        Bij_value, Bij_channel, Cij_value, Cij_row = ConeBeamModel.compute_sparse_Bij_Cij_single_view(pixel_index,
                                                                                                      angle,
                                                                                                      view_projector_params)

        # Generate full index arrays for rows and columns
        # Expand Cij_row and Cij_channel for broadcasting
        Cij_value_expanded = Cij_value[0, :, :, None]  # Shape (Nv, 2p+1, 1)
        Bij_value_expanded = Bij_value[0, :, None, :]  # Shape (Nv, 1, 2p+1)

        # Expand Cij_row and Cij_channel for broadcasting
        rows_expanded = Cij_row[0, :, :, None]  # Shape (Nv, 2p+1, 1)
        channels_expanded = Bij_channel[0, :, None, :]  # Shape (Nv, 1, 2p+1)

        # Create sinogram_array with shape (Nv x psf_width x psf_width)
        sinogram_array = sinogram_view[rows_expanded, channels_expanded]
        # Compute back projection
        # coeff_power = 1 normally; coeff_power = 2 when computing diagonal of hessian
        back_projection = jnp.sum(sinogram_array * ((Bij_value_expanded * Cij_value_expanded) ** coeff_power),
                                  axis=(1, 2))
        return back_projection

    @staticmethod
    def forward_project_pixels_to_one_view(voxel_values, pixel_indices, angle, projector_params):
        """
        Forward project a set of voxels determined by indices into the flattened array of size num_rows x num_cols.

        Args:
            voxel_values (jax array):  2D array of shape (num_indices, num_slices) of voxel values, where
                voxel_values[i, j] is the value of the voxel in slice j at the location determined by indices[i].
            pixel_indices (jax array of int):  1D vector of indices into flattened array of size num_rows x num_cols.
            angle (float):  Angle for this view
            projector_params (tuple):  tuple of (sinogram_shape, recon_shape, get_geometry_params())
            sinogram_shape (tuple): Sinogram shape (num_views, num_det_rows, num_det_channels)

        Returns:
            jax array of shape (num_det_rows, num_det_channels)
        """
        recon_shape = projector_params[1]
        num_recon_slices = recon_shape[2]
        if voxel_values.shape[0] != pixel_indices.shape[0] or len(voxel_values.shape) < 2 or \
                voxel_values.shape[1] != num_recon_slices:
            raise ValueError('voxel_values must have shape[0:2] = (num_indices, num_slices)')

        new_voxel_values = ConeBeamModel.forward_project_vertical_fan_beam_one_view(voxel_values, pixel_indices, angle,
                                                                                     projector_params)
        sinogram_view = ConeBeamModel.forward_project_horizontal_fan_beam_one_view(new_voxel_values, pixel_indices,
                                                                                  angle, projector_params)
        return sinogram_view


    @staticmethod
    @partial(jax.jit, static_argnames='projector_params')
    def compute_sparse_Bij_Cij_single_view(pixel_indices, angle, projector_params, p=1):
        """
        Calculate the separable sparse system matrices for a subset of voxels and a single view.
        It returns a sparse matrix specified by the system matrix values and associated detector column index.
        Since this is for parallel beam geometry, the values are assumed to be the same for each row/slice.

        Args:
            pixel_indices (jax array of int):  1D vector of indices into flattened array of size num_rows x num_cols.
            angle (float):  Angle for this single view
            projector_params (tuple):  tuple of (sinogram_shape, recon_shape, get_geometry_params())
            p (int, optional, default=1):  # This is the assumed number of channels per side

        Returns:
            Bij_value, Bij_column, Cij_value, Cij_row (jnp array): Each with shape (num voxels)x(num slices)x(2p+1)
        """
        warnings.warn('Compiling for indices length = {}'.format(pixel_indices.shape))
        warnings.warn('Using hard-coded detectors per side.  These should be set dynamically based on the geometry.')

        # Get all the geometry parameters
        geometry_params = projector_params[2]
        (delta_det_channel, delta_det_row, det_channel_offset, det_row_offset, det_rotation, source_detector_dist,
         delta_voxel, recon_slice_offset, magnification) = geometry_params

        num_views, num_det_rows, num_det_channels = projector_params[0]
        recon_shape = projector_params[1]
        num_recon_rows, num_recon_cols, num_recon_slices = recon_shape

        # Convert the index into (i,j,k) coordinates corresponding to the indices into the 3D voxel array
        num_indices = pixel_indices.size
        row_index, col_index = jnp.unravel_index(pixel_indices, recon_shape[:2])

        # Replicate along the slice axis
        i = jnp.tile(row_index[:, :, None], reps=(1, num_recon_slices, 1))
        j = jnp.tile(col_index[:, :, None], reps=(1, num_recon_slices, 1))
        k = jnp.tile(jnp.arange(num_recon_slices)[None, :, None], reps=(num_indices, 1, 1))

        # All the following objects should have shape (num pixels)x(num slices)x1
        # x, y, z
        # u, v
        # mp, np
        # cone_angle_channel
        # cone_angle_row
        # cos_alpha_col
        # cos_alpha_row
        # W_col has shape
        # W_row has shape
        # mp has shape
        # np has shape 

        # Convert from ijk to coordinates about iso
        x, y, z = ConeBeamModel.recon_ijk_to_xyz(i, j, k, delta_voxel, recon_shape,
                                   recon_slice_offset, angle)

        # Convert from xyz to coordinates on detector
        u, v, pixel_mag = ConeBeamModel.geometry_xyz_to_uv_mag(x, y, z, source_detector_dist, magnification)

        # Convert from uv to index coordinates in detector
        mp, np = ConeBeamModel.detector_uv_to_mn(u, v, delta_det_channel, delta_det_row, det_channel_offset,
                                                 det_row_offset, num_det_rows, num_det_channels, det_rotation)

        # Compute cone angle of pixel along columns and rows
        cone_angle_channel = jnp.arctan2(u, source_detector_dist)
        cone_angle_row = jnp.arctan2(v, source_detector_dist)

        # Compute cos alpha for row and columns
        cos_alpha_col = jnp.maximum(jnp.abs(jnp.cos(angle - cone_angle_channel)),
                                    jnp.abs(jnp.sin(angle - cone_angle_channel)))
        cos_alpha_row = jnp.maximum(jnp.abs(jnp.cos(cone_angle_row)), jnp.abs(jnp.sin(cone_angle_row)))

        # Compute projected voxel width along columns and rows
        W_col = pixel_mag * (delta_voxel / delta_det_channel) * (cos_alpha_col / jnp.cos(cone_angle_channel))
        W_row = pixel_mag * (delta_voxel / delta_det_row) * (cos_alpha_row / jnp.cos(cone_angle_row))

        # ################
        # Compute the Bij matrix entries
        # Compute a jnp channel index array with shape [(num pixels)*(num slices)]x1
        Bij_channel = jnp.round(np).astype(int)

        # Compute a jnp channel index array with shape [(num pixels)*(num slices)]x(2p+1)
        Bij_channel = jnp.concatenate([Bij_channel + j for j in range(-p, p + 1)], axis=-1)

        # Compute the distance of each channel from the center of the voxel
        # Should be shape [(num pixels)*(num slices)]x(2p+1)
        delta_channel = jnp.abs(Bij_channel - np)

        # Calculate L = length of intersection between detector element and projection of flattened voxel
        # Should be shape [(num pixels)*(num slices)]x(2p+1)
        tmp1 = (W_col + 1) / 2.0  # length = num_indices
        tmp2 = (W_col - 1) / 2.0  # length = num_indices
        L_channel = jnp.maximum(tmp1 - jnp.maximum(jnp.abs(tmp2), delta_channel), 0)

        # Compute Bij sparse matrix with shape [(num pixels)*(num slices)]x(2p+1)
        Bij_value = (delta_voxel / cos_alpha_col) * L_channel
        Bij_value = Bij_value * (Bij_channel >= 0) * (Bij_channel < num_det_channels)

        # ################
        # Compute the Cij matrix entries
        # Compute a jnp row index array with shape [(num pixels)*(num slices)]x1
        Cij_row = jnp.round(mp).astype(int)

        # Compute a jnp row index array with shape [(num pixels)*(num slices)]x(2p+1)
        Cij_row = jnp.concatenate([Cij_row + j for j in range(-p, p + 1)], axis=-1)

        # Compute the distance of each row from the center of the voxel
        # Should be shape [(num pixels)*(num slices)]x(2p+1)
        delta_row = jnp.abs(Cij_row - mp)

        # Calculate L = length of intersection between detector element and projection of flattened voxel
        # Should be shape [(num pixels)*(num slices)]x(2p+1)
        tmp1 = (W_row + 1) / 2.0  # length = num_indices
        tmp2 = (W_row - 1) / 2.0  # length = num_indices
        L_row = jnp.maximum(tmp1 - jnp.maximum(jnp.abs(tmp2), delta_row), 0)

        # Compute Cij sparse matrix with shape [(num pixels)*(num slices)]x(2p+1)
        Cij_value = (1 / cos_alpha_row) * L_row
        # Zero out any out-of-bounds values
        Cij_value = Cij_value * (Cij_row >= 0) * (Cij_row < num_det_rows)

        return Bij_value, Bij_channel, Cij_value, Cij_row

    @staticmethod
    def forward_project_vertical_fan_beam_one_view(voxel_values, pixel_indices, angle, projector_params):

        pixel_map = jax.vmap(ConeBeamModel.forward_project_vertical_fan_beam_one_pixel_one_view,
                             in_axes=(0, 0, None, None))
        new_pixels = pixel_map(voxel_values, pixel_indices, angle, projector_params)

        return new_pixels

    @staticmethod
    @partial(jax.jit, static_argnames='projector_params')
    def forward_project_horizontal_fan_beam_one_view(voxel_values, pixel_indices, angle, projector_params, p=1):
        """
        Apply a horizontal fan beam transformation to a single voxel cylinder that has slices aligned with detector
        rows and return the resulting sinogram view.

        Args:
            voxel_values:
            pixel_indices:
            angle:
            projector_params:
            p:

        Returns:

        """

        # Get all the geometry parameters
        geometry_params = projector_params[2]
        (delta_det_channel, delta_det_row, det_channel_offset, det_row_offset, det_rotation, source_detector_dist,
         delta_voxel, recon_slice_offset, magnification) = geometry_params

        num_views, num_det_rows, num_det_channels = projector_params[0]
        recon_shape = projector_params[1]
        num_recon_rows, num_recon_cols, num_recon_slices = recon_shape

        # Convert the index into (i,j,k) coordinates corresponding to the indices into the 3D voxel array
        row_index, col_index = jnp.unravel_index(pixel_indices, recon_shape[:2])
        slice_index = jnp.arange(num_recon_slices)

        x_p, y_p, z_p = ConeBeamModel.recon_ijk_to_xyz(row_index, col_index, slice_index, delta_voxel,
                                                       recon_shape, recon_slice_offset, angle)

        # Convert from xyz to coordinates on detector
        pixel_mag = 1 / (1 / magnification - y_p / source_detector_dist)  # This should be kept in terms of magnification
        # Compute the physical position that this voxel projects onto the detector
        u_p = pixel_mag * x_p
        det_center_channel = (num_det_channels - 1) / 2.0  # num_of_cols

        # Calculate indices on the detector grid
        n_p = (u_p / delta_det_channel) + det_center_channel + det_channel_offset
        n_p_center = jnp.round(n_p).astype(int)

        # Compute horizontal cone angle of pixel
        theta_p = jnp.arctan2(u_p, source_detector_dist)

        # Compute cos alpha for row and columns
        cos_alpha_p_xy = jnp.maximum(jnp.abs(jnp.cos(angle - theta_p)),
                                    jnp.abs(jnp.sin(angle - theta_p)))

        # Compute projected voxel width along columns and rows
        W_p_c = pixel_mag * (delta_voxel / delta_det_channel) * (cos_alpha_p_xy / jnp.cos(theta_p))
        L_max = jnp.minimum(1, W_p_c)

        # Allocate the sinogram array
        sinogram_view = jnp.zeros((num_det_rows, num_det_channels))
        # TODO:  convert to vmap over m / z
        # Computed values needed to finish the forward projection:
        # n_p_center, n_p, W_p_c, cos_alpha_p_xy
        for n_offset in jnp.arange(start=-p, stop=p+1):
            n = n_p_center + n_offset
            abs_delta_p_c_n = jnp.abs(n_p - n)
            L_p_c_n = jnp.clip((W_p_c + 1) / 2 - abs_delta_p_c_n, 0, L_max)
            A_chan_n = delta_voxel * L_p_c_n / cos_alpha_p_xy
            A_chan_n *= (n >= 0) * (n < num_det_channels)
            sinogram_view = sinogram_view.at[:, n].add(A_chan_n[None, :] * voxel_values.T)

        return sinogram_view

    @staticmethod
    @partial(jax.jit, static_argnames='projector_params')
    def forward_project_vertical_fan_beam_one_pixel_one_view(voxel_values, pixel_index, angle, projector_params,
                                                             p=1):
        """
        Apply a vertical fan beam transformation to a single voxel cylinder and return the column vector
        of the resulting values.

        Args:
            voxel_values:
            pixel_index:
            angle:
            projector_params:
            p:

        Returns:

        """

        # Get all the geometry parameters
        geometry_params = projector_params[2]
        (delta_det_channel, delta_det_row, det_channel_offset, det_row_offset, det_rotation, source_detector_dist,
         delta_voxel, recon_slice_offset, magnification) = geometry_params

        num_views, num_det_rows, num_det_channels = projector_params[0]
        recon_shape = projector_params[1]
        num_recon_rows, num_recon_cols, num_recon_slices = recon_shape

        # Convert the index into (i,j,k) coordinates corresponding to the indices into the 3D voxel array
        row_index, col_index = jnp.unravel_index(pixel_index, recon_shape[:2])
        slice_index = jnp.arange(num_recon_slices)

        x_p, y_p, z_p = ConeBeamModel.recon_ijk_to_xyz(row_index, col_index, slice_index, delta_voxel,
                                                       recon_shape, recon_slice_offset, angle)

        # Convert from xyz to coordinates on detector
        u_p, v_p, pixel_mag = ConeBeamModel.geometry_xyz_to_uv_mag(x_p, y_p, z_p, source_detector_dist, magnification)
        # Convert from uv to index coordinates in detector and get the vector of center detector rows for this cylinder
        m_p, _ = ConeBeamModel.detector_uv_to_mn(u_p, v_p, delta_det_channel, delta_det_row, det_channel_offset,
                                                 det_row_offset, num_det_rows, num_det_channels, det_rotation)
        m_p_center = jnp.round(m_p).astype(int)

        # Compute vertical cone angle of pixel
        phi_p = jnp.arctan2(v_p, source_detector_dist)

        # Compute cos alpha for row and columns
        cos_phi_p = jnp.cos(phi_p)
        cos_alpha_p_z = jnp.maximum(jnp.abs(cos_phi_p), jnp.abs(jnp.sin(phi_p)))

        # Get the length of projection of flattened voxel on detector (in fraction of detector size)
        W_p_r = pixel_mag * (delta_voxel / delta_det_row) * cos_alpha_p_z / cos_phi_p

        new_voxel_cylinder = jnp.zeros(num_det_rows)
        L_max = jnp.minimum(1, W_p_r)

        # Computed values needed to finish the forward projection:
        # m_p_center, m_p, W_p_r, cos_alpha_p_z
        for m_offset in jnp.arange(start=-p, stop=p+1):
            m = m_p_center + m_offset
            abs_delta_p_r_m = jnp.abs(m_p - m)
            L_p_r_m = jnp.clip((W_p_r + 1) / 2 - abs_delta_p_r_m, 0, L_max)
            A_row_m = L_p_r_m / cos_alpha_p_z
            A_row_m *= (m >= 0) * (m < num_det_rows)
            new_voxel_cylinder = new_voxel_cylinder.at[m].add(A_row_m * voxel_values)

        return new_voxel_cylinder

    @staticmethod
    @partial(jax.jit, static_argnames='projector_params')
    def back_project_vertical_fan_beam_one_pixel_one_view(voxel_values, pixel_index, angle, projector_params, p=1):
        """
        Apply the back projection of a vertical fan beam transformation to a single voxel cylinder and return the column
        vector of the resulting values.

        Args:
            voxel_values:
            pixel_index:
            angle:
            projector_params:
            p:

        Returns:

        """

        # Get all the geometry parameters
        geometry_params = projector_params[2]
        (delta_det_channel, delta_det_row, det_channel_offset, det_row_offset, det_rotation, source_detector_dist,
         delta_voxel, recon_slice_offset, magnification) = geometry_params

        num_views, num_det_rows, num_det_channels = projector_params[0]
        recon_shape = projector_params[1]
        num_recon_rows, num_recon_cols, num_recon_slices = recon_shape

        # Convert the index into (i,j,k) coordinates corresponding to the indices into the 3D voxel array
        row_index, col_index = jnp.unravel_index(pixel_index, recon_shape[:2])
        slice_index = jnp.arange(num_recon_slices)

        x_p, y_p, z_p = ConeBeamModel.recon_ijk_to_xyz(row_index, col_index, slice_index, delta_voxel,
                                                       recon_shape, recon_slice_offset, angle)

        # Convert from xyz to coordinates on detector
        u_p, v_p, pixel_mag = ConeBeamModel.geometry_xyz_to_uv_mag(x_p, y_p, z_p, source_detector_dist, magnification)
        # Convert from uv to index coordinates in detector and get the vector of center detector rows for this cylinder
        m_p, _ = ConeBeamModel.detector_uv_to_mn(u_p, v_p, delta_det_channel, delta_det_row, det_channel_offset,
                                                 det_row_offset, num_det_rows, num_det_channels, det_rotation)
        m_p_center = jnp.round(m_p).astype(int)

        # Compute vertical cone angle of pixel
        phi_p = jnp.arctan2(v_p, source_detector_dist)

        # Compute cos alpha for row and columns
        cos_phi_p = jnp.cos(phi_p)
        cos_alpha_p_z = jnp.maximum(jnp.abs(cos_phi_p), jnp.abs(jnp.sin(phi_p)))

        # Get the length of projection of flattened voxel on detector (in fraction of detector size)
        W_p_r = pixel_mag * (delta_voxel / delta_det_row) * cos_alpha_p_z / cos_phi_p

        recon_voxel_cylinder = jnp.zeros(num_recon_slices)
        L_max = jnp.minimum(1, W_p_r)

        # Computed values needed to finish the forward projection:
        # m_p_center, m_p, W_p_r, cos_alpha_p_z
        for m_offset in jnp.arange(start=-p, stop=p+1):
            m = m_p_center + m_offset
            abs_delta_p_r_m = jnp.abs(m_p - m)
            L_p_r_m = jnp.clip((W_p_r + 1) / 2 - abs_delta_p_r_m, 0, L_max)
            A_row_m = L_p_r_m / cos_alpha_p_z
            A_row_m *= (m >= 0) * (m < num_det_rows)
            recon_voxel_cylinder = jnp.add(recon_voxel_cylinder, A_row_m * voxel_values[m])

        return recon_voxel_cylinder

    @staticmethod
    @jax.jit
    def recon_ijk_to_xyz(i, j, k, delta_voxel, recon_shape,
                         recon_slice_offset, angle):

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
    @jax.jit
    def geometry_xyz_to_uv_mag(x, y, z, source_detector_dist, magnification):
        # Compute the magnification at this specific voxel
        # This can be done with the following 2 lines, but by taking the reciprocal of pixel_mag, we can express it in
        # terms of magnification and source_detector_dist, which remains valid even source_detector_dist = np.Inf.
        # source_to_iso_dist = source_detector_dist / magnification
        # pixel_mag = source_detector_dist / (source_to_iso_dist - y)
        pixel_mag = 1 / (1 / magnification - y / source_detector_dist)

        # Compute the physical position that this voxel projects onto the detector
        u = pixel_mag * x
        v = pixel_mag * z

        return u, v, pixel_mag

    @staticmethod
    @partial(jax.jit, static_argnames='det_rotation')
    def detector_uv_to_mn(u, v, delta_det_channel, delta_det_row, det_channel_offset, det_row_offset, num_det_rows,
                          num_det_channels, det_rotation=0):
        # Account for small rotation of the detector
        # TODO:  In addition to including the rotation, we'd need to adjust the calculation of the channel as a
        #  function of slice.
        u_tilde = u  # jnp.cos(det_rotation) * u + jnp.sin(det_rotation) * v
        v_tilde = v  # -jnp.sin(det_rotation) * u + jnp.cos(det_rotation) * v

        # Get the center of the detector grid for columns and rows
        det_center_row = (num_det_rows - 1) / 2.0  # num_of_rows
        det_center_channel = (num_det_channels - 1) / 2.0  # num_of_cols

        # Calculate indices on the detector grid
        m = (v_tilde / delta_det_row) + det_center_row + det_row_offset
        n = (u_tilde / delta_det_channel) + det_center_channel + det_channel_offset

        return m, n
