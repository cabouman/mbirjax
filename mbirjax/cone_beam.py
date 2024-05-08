import warnings

import jax
import jax.numpy as jnp
from functools import partial
from mbirjax import TomographyModel


class ConeBeamModel(TomographyModel):
    """
    A class designed for handling forward and backward projections in a parallel beam geometry, extending the
    :ref:`TomographyModelDocs`. This class offers specialized methods and parameters tailored for parallel beam setups.

    This class inherits all methods and properties from the :ref:`TomographyModelDocs` and may override some
    to suit parallel beam geometrical requirements. See the documentation of the parent class for standard methods
    like setting parameters and performing projections and reconstructions.

    Args:
        angles (jnp.ndarray):
            A 1D array of projection angles, in radians, specifying the angle of each projection relative to the origin.
        sinogram_shape (tuple):
            Shape of the sinogram as a tuple in the form `(views, rows, channels)`, where 'views' is the number of
            different projection angles, 'rows' correspond to the number of detector rows, and 'channels' index columns of
            the detector that are assumed to be aligned with the rotation axis.
        source_detector_dist (float) – Distance between the X-ray source and the detector in units of ALU.
        det_channel_offset (float) – Distance = (projected center of rotation) - (center of detector channels) in ALU.
        det_row_offset (float) – Distance = (projected perpendicular to center of rotation) - (center of detector rows) in ALU.
        image_slice_offset (float) – Vertical offset of the image in ALU.
        angles (float, ndarray) – 1D array of view angles in radians.
        **kwargs (dict):
            Additional keyword arguments that are passed to the :ref:`TomographyModelDocs` constructor. These can
            include settings and configurations specific to the tomography model such as noise models or image dimensions.
            Refer to :ref:`TomographyModelDocs` documentation for a detailed list of possible parameters.
    """

    def __init__(self, sinogram_shape, source_detector_dist, det_row_offset=0.0, recon_slice_offset=0.0, det_rotation=0.0, angles, **kwargs):
        # Convert the view-dependent vectors to an array
        # This is more complicated than needed with only a single view-dependent vector but is included to
        # illustrate the process as shown in TemplateModel
        view_dependent_vecs = [vec.flatten() for vec in [angles]]
        try:
            view_params_array = jnp.stack(view_dependent_vecs, axis=1)
        except ValueError as e:
            raise ValueError("Incompatible view dependent vector lengths:  all view-dependent vectors must have the "
                             "same length.")
        super().__init__(sinogram_shape, source_detector_dist=source_detector_dist, det_row_offset=det_row_offset, recon_slice_offset=recon_slice_offset, view_params_array=view_params_array, det_rotation=det_rotation, **kwargs)

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

        recon_shape = self.get_params('recon_shape')
        if recon_shape[2] != sinogram_shape[1]:
            error_message = "Number of recon slices must match number of sinogram rows. \n"
            error_message += "Got {} for recon_shape and {} for sinogram_shape".format(recon_shape, sinogram_shape)
            raise ValueError(error_message)

    def get_geometry_parameters(self):
        """
        Function to get a list of the primary geometry parameters for projection.

        Returns:
            List of required geometry parameters.
        """
        geometry_params = self.get_params(['delta_det_channel', 'delta_det_row', 'det_channel_offset', 'det_row_offset', 'det_rotation', 'source_detector_dist', 'magnification', 'delta_pixel_recon', 'recon_slice_offset'])

        return geometry_params

    @staticmethod
    def back_project_one_view_to_voxel(sinogram_view, voxel_index, angle, projector_params, coeff_power=1):
        """
        Calculate the backprojection value at a specified recon voxel given a sinogram view and various parameters.
        This code uses the distance driven projector.

        Args:
            sinogram_view (jax array): one view of the sinogram to be back projected
            voxel_index: the integer index into flattened recon - need to apply unravel_index(voxel_index, recon_shape) to get i, j, k
            angle (float): The angle in radians for this view.
            projector_params (tuple):  tuple of (sinogram_shape, recon_shape, get_geometry_params()).
            coeff_power: [int] backproject using the coefficients of (A_ij ** coeff_power).
                Normally 1, but should be 2 when computing theta 2.

        Returns:
            The value of the voxel for all slices at the input index (i.e., a voxel cylinder) obtained by backprojecting
            the input sinogram view.
        """

        # Get the part of the system matrix and channel indices for this voxel
        sinogram_view_shape = (1,) + sinogram_view.shape  # Adjoin a leading 1 to indicate a single view sinogram
        view_projector_params = (sinogram_view_shape,) + projector_params[1:]
        Aji, channel_index = ConeBeamModel.compute_Aji_channel_index(voxel_index, angle, view_projector_params)

        # Extract out the relevant entries from the sinogram
        sinogram_array = sinogram_view[:, channel_index.T.flatten()]

        # Compute dot product
        return jnp.sum(sinogram_array * (Aji**coeff_power), axis=1)

    @staticmethod
    def forward_project_voxels_one_view(voxel_values, voxel_indices, angle, projector_params):
        """
        Forward project a set of voxels determined by indices into the flattened array of size num_rows x num_cols.

        Args:
            voxel_values (jax array):  2D array of shape (num_indices, num_slices) of voxel values, where
                voxel_values[i, j] is the value of the voxel in slice j at the location determined by indices[i].
            voxel_indices (jax array of int):  1D vector of indices into flattened array of size num_rows x num_cols.
            angle (float):  Angle for this view
            projector_params (tuple):  tuple of (sinogram_shape, recon_shape, get_geometry_params())
            sinogram_shape (tuple): Sinogram shape (num_views, num_det_rows, num_det_channels)

        Returns:
            jax array of shape (num_det_rows, num_det_channels)
        """
        if voxel_values.ndim != 2:
            raise ValueError('voxel_values must have shape (num_indices, num_slices)')

        # Get the geometry parameters and the system matrix and channel indices
        num_views, num_det_rows, num_det_channels = projector_params[0]
        Aji, channel_index = ConeBeamModel.compute_Aji_channel_index(voxel_indices, angle, projector_params)

        # Add axes to be able to broadcast while multiplying.
        # sinogram_values has shape num_indices x (2p+1) x num_slices
        sinogram_values = (Aji[:, :, None] * voxel_values[:, None, :])

        # Now sum over indices into the locations specified by channel_index.
        # Directly using index_add for indexed updates
        sinogram_view = jnp.zeros((num_det_rows, num_det_channels))  # num_det_rows x num_det_channels

        # Apply the vectorized update function with a vmap over slices
        # sinogram_view is num_det_rows x num_det_channels, sinogram_values is num_indices x (2p+1) x num_det_rows
        sinogram_values = sinogram_values.transpose((2, 0, 1)).reshape((num_det_rows, -1))
        sinogram_view = sinogram_view.at[:, channel_index.flatten()].add(sinogram_values)
        del Aji, channel_index
        return sinogram_view


    @staticmethod
    @partial(jax.jit, static_argnums=3)
    def compute_Aij_voxels_single_view( voxel_indices, angle, projector_params, p=1 ):
        """
        Calculate the sparse system matrix for a subset of voxels and a single view.
        It returns a sparse matrix specified by the system matrix values and associated detector column index.
        Since this is for parallel beam geometry, the values are assumed to be the same for each row/slice.

        Args:
            voxel_indices (jax array of int):  1D vector of indices into flattened array of size num_rows x num_cols.
            angle (float):  Angle for this single view
            projector_params (tuple):  tuple of (sinogram_shape, recon_shape, get_geometry_params())
            p (int, optional, default=1):  # This is the assumed number of channels per side

        Returns:
            Aji_value (num indices, 2p+1), Aji_channel (num indices, 2p+1)
        """
        def recon_ijk_to_xyz( i, j, k, delta_pixel_recon, num_recon_rows, num_recon_cols, recon_slice_offset, angle):
            # Compute the un-rotated coordinates relative to iso
            x_tilde, y_tilde, z = delta_pixel_recon * (jnp.array([i, j, k]) - jnp.array([num_recon_rows, num_recon_cols]) / 2.0)
            z += recon_slice_offset  # Adding slice offset to z only

            # Compute the rotated coordinates relative to iso
            x = jnp.cos(angle) * x_tilde - jnp.sin(angle) * y_tilde  # corrected minus sign here
            y = jnp.sin(angle) * x_tilde + jnp.cos(angle) * y_tilde

            return x, y, z

        def geometry_xyz_to_uv_mag(x, y, z, source_detector_dist, magnification):
            # Compute the source to iso distance
            source_to_iso_dist = source_detector_dist / magnification

            # Check for a potential division by zero or very small denominator
            if (source_to_iso_dist - y) == 0 :
                raise ValueError("Invalid geometry: Denominator in pixel magnification calculation becomes zero.")

            # Compute the magnification at this specific voxel
            pixel_mag = source_detector_dist / (source_to_iso_dist - y)

            # Compute the physical position that this voxel projects onto the detector
            u = pixel_mag * x
            v = pixel_mag * z

            return u, v, pixel_mag

        def detector_uv_to_mn( self, u, v, det_rotation, delta_det_channel, delta_det_row, det_channel_offset, det_row_offset, num_det_rows, num_det_channels):
            # Account for small rotation of the detector
            u_tilde = jnp.cos(det_rotation) * u + jnp.sin(det_rotation) * v
            v_tilde = -jnp.sin(det_rotation) * u + jnp.cos(det_rotation) * v

            # Get the center of the detector grid for columns and rows
            det_center_channels = (num_det_channels - 1) / 2.0  # num_of_cols
            det_center_rows = (num_det_rows - 1) / 2.0  # num_of_rows

            # Calculate indices on the detector grid
            n = (u_tilde / delta_det_channel) + det_center_channels + det_channel_offset
            m = (v_tilde / delta_det_row) + det_center_rows + det_row_offset

            return m, n

        warnings.warn('Compiling for indices length = {}'.format(voxel_indices.shape))
        warnings.warn('Using hard-coded detectors per side.  These should be set dynamically based on the geometry.')

        # Get all the geometry parameters
        geometry_params = projector_params[2]
        delta_det_channel, delta_det_row, det_channel_offset, det_row_offset, det_rotation, source_detector_dist, magnification, delta_pixel_recon, recon_slice_offset = geometry_params

        num_views, num_det_rows, num_det_channels = projector_params[0]
        num_recon_rows, num_recon_cols = projector_params[1][:2]

        # Convert the index into (i,j,k) coordinates corresponding to the indices into the 3D voxel array
        recon_shape_2d = (num_recon_rows, num_recon_cols)
        row_index, col_index = jnp.unravel_index(voxel_indices, recon_shape_2d)

        # TODO: We need something here so that we handle slices. Right now there is no k
        # We might need to break this into 2 functions, one for B and one for C. Not sure.

        # Convert from ijk to coordinates about iso
        x, y, z = recon_ijk_to_xyz(i, j, k, delta_pixel_recon, num_recon_rows, num_recon_cols, recon_slice_offset, angle)

        # Convert from xyz to coordinates on detector
        u, v, pixel_mag = geometry_xyz_to_uv_mag(x, y, z, source_detector_dist, magnification)

        # Convert from uv to index coordinates in detector
        m, n = detector_uv_to_mn(u, v, det_rotation, delta_det_channel, delta_det_row, det_channel_offset, det_row_offset, num_det_rows, num_det_channels)

        # Compute cone angle of pixel along columns and rows
        cone_angle_channel = jnp.arctan2(u, source_detector_dist)
        cone_angle_row = jnp.arctan2(v, source_detector_dist)

        # Compute cos alpha for row and columns
        cos_alpha_col = jnp.maximum(jnp.abs(jnp.cos(angle - cone_angle_channel)), jnp.abs(jnp.sin(angle - cone_angle_channel)))
        cos_alpha_row = jnp.maximum(jnp.abs(jnp.cos(cone_angle_row)), jnp.abs(jnp.sin(cone_angle_row)))

        # Compute compute projected voxel width along columns and rows
        W_col = pixel_mag * (delta_pixel_recon / delta_det_channel) * (cos_alpha_col / jnp.cos(cone_angle_channel))
        W_row = pixel_mag * (delta_pixel_recon / delta_det_row) * (cos_alpha_row / jnp.cos(cone_angle_row))



        # Compute the location on the detector in ALU of the projected center of the voxel
        x_pos_on_detector = x_pos_rot + channel_center  # length = num_indices

        # Compute a jnp array with 2p+1 entries that are the channel indices of the relevant channels
        Aij_channel = jnp.round(x_pos_on_detector / delta_det_channel).astype(int)  # length = num_indices
        Aij_channel = Aij_channel.reshape((-1, 1))

        # Compute channel indices for 2p+1 adjacent channels at each view angle
        # Should be num_indices x 2p+1
        # Aij_channel = jnp.concatenate([Aij_channel - 1, Aij_channel, Aij_channel + 1], axis=-1)
        Aij_channel = jnp.concatenate([Aij_channel + j for j in range(-p, p+1)], axis=-1)

        # Compute the distance of each channel from the projected center of the voxel
        delta = jnp.abs(
            Aij_channel * delta_det_channel - x_pos_on_detector.reshape((-1, 1)))  # Should be num_indices x 2p+1

        # Calculate L = length of intersection between detector element and projection of flattened voxel
        tmp1 = (W + delta_det_channel) / 2.0  # length = num_indices
        tmp2 = (W - delta_det_channel) / 2.0  # length = num_indices
        Lv = jnp.maximum(tmp1 - jnp.maximum(jnp.abs(tmp2), delta), 0)  # Should be num_indices x 2p+1

        # Compute the values of Aij
        Aji_value = (delta_pixel_recon / cos_alpha) * (Lv / delta_det_channel)  # Should be num_indices x 2p+1
        Aji_value = Aji_value * (Aij_channel >= 0) * (Aij_channel < num_det_channels)
        return Aji_value, Aij_channel

