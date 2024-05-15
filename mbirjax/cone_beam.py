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

    def __init__(self, sinogram_shape, angles, source_detector_dist, magnification,
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
        super().__init__(sinogram_shape, source_detector_dist=source_detector_dist,
                         recon_slice_offset=recon_slice_offset, det_rotation=det_rotation,
                         view_params_array=view_params_array, **kwargs)
        self.set_params(magnification=magnification)

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
             'source_detector_dist', 'magnification', 'delta_voxel_xy', 'delta_voxel_z', 'recon_slice_offset'])

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
        # jax.debug.breakpoint()
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
        if voxel_values.ndim != 2:
            raise ValueError('voxel_values must have shape (num_indices, num_slices)')
        pixel_indices = pixel_indices.reshape((-1, 1))

        # Get the geometry parameters and the system matrix and channel indices
        num_views, num_det_rows, num_det_channels = projector_params[0]
        Bij_value, Bij_channel, Cij_value, Cij_row = ConeBeamModel.compute_sparse_Bij_Cij_single_view(pixel_indices,
                                                                                                      angle,
                                                                                                      projector_params)

        # Determine the size of the sinogram based on max indices + 1 for 0-based indexing
        Nr = num_det_rows
        Nc = num_det_channels

        # Allocate the sinogram array
        sinogram_view = jnp.zeros((Nr, Nc))

        # Compute the outer products and scale by voxel_values
        # First, compute the outer product of Bij_value and Cij_value
        outer_products = jnp.einsum('kmi,kmj->kmij', Cij_value, Bij_value)
        sinogram_entries = outer_products * voxel_values[:, :, None, None]

        # Expand Cij_row and Cij_channel for broadcasting
        rows_expanded = Cij_row[:, :, :, None]  # Shape (Nv, 2p+1, 1)
        channels_expanded = Bij_channel[:, :, None, :]  # Shape (Nv, 1, 2p+1)

        # Flatten the arrays to index into sinogram view
        flat_sinogram_entries = sinogram_entries.reshape(-1)
        flat_rows = jnp.tile(rows_expanded, reps=(1, 1, 1, sinogram_entries.shape[3]))
        flat_rows = flat_rows.reshape(-1)
        flat_channels = jnp.tile(channels_expanded, reps=(1, 1, sinogram_entries.shape[2], 1))
        flat_channels = flat_channels.reshape(-1)

        # Aggregate the results into sinogram_view
        sinogram_view = sinogram_view.at[flat_rows, flat_channels].add(flat_sinogram_entries)
        # jax.debug.breakpoint()
        return sinogram_view

    @staticmethod
    @partial(jax.jit, static_argnums=2)
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
         magnification, delta_voxel_xy, delta_voxel_z, recon_slice_offset) = geometry_params

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
        x, y, z = ConeBeamModel.recon_ijk_to_xyz(i, j, k, delta_voxel_xy, delta_voxel_z, recon_shape,
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
        W_col = pixel_mag * (delta_voxel_xy / delta_det_channel) * (cos_alpha_col / jnp.cos(cone_angle_channel))
        W_row = pixel_mag * (delta_voxel_xy / delta_det_row) * (cos_alpha_row / jnp.cos(cone_angle_row))

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
        Bij_value = (delta_voxel_xy / cos_alpha_col) * L_channel
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
    @partial(jax.jit, static_argnums=2)
    def compute_sparse_Bij_Cij_single_view_new(pixel_indices, angle, projector_params, p=1):
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
         magnification, delta_voxel_xy, delta_voxel_z, recon_slice_offset) = geometry_params

        num_views, num_det_rows, num_det_channels = projector_params[0]
        recon_shape = projector_params[1]
        num_recon_rows, num_recon_cols, num_recon_slices = recon_shape

        # Convert the index into (i,j,k) coordinates corresponding to the indices into the 3D voxel array
        num_indices = pixel_indices.size
        row_index, col_index = jnp.unravel_index(pixel_indices, recon_shape[:2])

        slice_index = jnp.arange(num_recon_slices)

        x, y, z = ConeBeamModel.recon_ijk_to_xyz(row_index, col_index, slice_index, delta_voxel_xy, delta_voxel_z,
                                                 recon_shape, recon_slice_offset, angle)

        # Convert from xyz to coordinates on detector
        u, v, pixel_mag = ConeBeamModel.geometry_xyz_to_uv_mag(x, y, z, source_detector_dist, magnification)

    @staticmethod
    @jax.jit
    def recon_ijk_to_xyz(i, j, k, delta_voxel_xy, delta_voxel_z, recon_shape,
                         recon_slice_offset, angle):

        num_recon_rows, num_recon_cols, num_recon_slices = recon_shape

        # Compute the un-rotated coordinates relative to iso
        # Note the change in order from (i, j) to (y, x)!!
        y_tilde = delta_voxel_xy * (i - (num_recon_rows - 1) / 2.0)
        x_tilde = delta_voxel_xy * (j - (num_recon_cols - 1) / 2.0)

        # Precompute cosine and sine of view angle, then do the rotation
        cosine = jnp.cos(angle)  # length = num_views
        sine = jnp.sin(angle)  # length = num_views

        x = cosine * x_tilde - sine * y_tilde
        y = sine * x_tilde + cosine * y_tilde

        z = delta_voxel_z * (k - (num_recon_slices - 1) / 2.0) + recon_slice_offset
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
                          num_det_channels, det_rotation):
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
