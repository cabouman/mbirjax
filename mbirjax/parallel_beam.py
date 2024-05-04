import warnings

import jax
import jax.numpy as jnp

from mbirjax import TomographyModel, Projectors


class ParallelBeamModel(TomographyModel):
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
        **kwargs (dict):
            Additional keyword arguments that are passed to the :ref:`TomographyModelDocs` constructor. These can
            include settings and configurations specific to the tomography model such as noise models or image dimensions.
            Refer to :ref:`TomographyModelDocs` documentation for a detailed list of possible parameters.

    Examples
    --------
    Initialize a parallel beam model with specific angles and sinogram shape:
    >>> import mbirjax
    >>> angles = jnp.array([0, jnp.pi/4, jnp.pi/2])
    >>> model = mbirjax.ParallelBeamModel(angles, (180, 256, 10))

    See Also
    --------
    TomographyModel : The base class from which this class inherits.
    """

    def __init__(self, angles, sinogram_shape, **kwargs):

        super().__init__(sinogram_shape, angles=angles, **kwargs)
        projector_functions = Projectors(self, self.forward_project_voxels_one_view, self.back_project_one_view_to_voxel)
        self._sparse_forward_project = projector_functions._sparse_forward_project
        self._sparse_back_project = projector_functions._sparse_back_project
        self._compute_hessian_diagonal = projector_functions._compute_hessian_diagonal

    def verify_valid_params(self):
        """
        Check that all parameters are compatible for a reconstruction.
        """
        super().verify_valid_params()
        sinogram_shape, angles = self.get_params(['sinogram_shape', 'angles'])

        if len(angles) != sinogram_shape[0]:
            error_message = "Number of angles must equal the number of views. \n"
            error_message += "Got {} for number of angles and {} for number of views.".format(len(angles),
                                                                                              sinogram_shape[0])
            raise ValueError(error_message)

    def get_geometry_parameters(self):
        """
        Convenience function to get a list of the primary geometry parameters for projection.

        Returns:
            List of delta_det_channel, det_channel_offset, delta_pixel_recon,
            num_recon_rows, num_recon_cols, num_recon_slices
        """
        geometry_params = self.get_params(['delta_det_channel', 'det_channel_offset', 'delta_pixel_recon',
                                           'num_recon_rows', 'num_recon_cols', 'num_recon_slices'])

        return geometry_params

    @staticmethod
    def back_project_one_view_to_voxel(sinogram_view, voxel_index, angle, geometry_params, coeff_power=1):
        """
        Calculate the backprojection value at a specified recon voxel given a sinogram view and various parameters.
        This code uses the distance driven projector.

        Args:
            sinogram_view (jax array): one view of the sinogram to be back projected
            voxel_index: the integer index into flattened recon - need to apply unravel_index(voxel_index, recon_shape) to get i, j, k
            angle:
            geometry_params:
            coeff_power: [int] backproject using the coefficients of (A_ij ** coeff_power).
                Normally 1, but should be 2 when computing theta 2.

        Returns:
            The value of the voxel at the input index obtained by backprojecting the input sinogram.
        """

        # Get the part of the system matrix and channel indices for this voxel
        Aji, channel_index = ParallelBeamModel.compute_Aji_channel_index(voxel_index, angle, geometry_params,
                                                                         (1,) + sinogram_view.shape)

        # Extract out the relevant entries from the sinogram
        sinogram_array = sinogram_view[:, channel_index.T.flatten()]

        # Compute dot product
        return jnp.sum(sinogram_array * (Aji**coeff_power), axis=1)

    @staticmethod
    def forward_project_voxels_one_view(voxel_values, voxel_indices, angle, geometry_params, sinogram_shape):
        """
        Forward project a set of voxels determined by indices into the flattened array of size num_rows x num_cols.

        Args:
            voxel_values (jax array):  2D array of shape (num_indices, num_slices) of voxel values, where
                voxel_values[i, j] is the value of the voxel in slice j at the location determined by indices[i].
            voxel_indices (jax array of int):  1D vector of indices into flattened array of size num_rows x num_cols.
            angle (float):  angle for this view
            geometry_params (list): Geometry parameters from get_geometry_params()
            sinogram_shape (tuple): Sinogram shape (num_views, num_det_rows, num_det_channels)

        Returns:
            jax array of shape (num_det_rows, num_det_channels)
        """
        if voxel_values.ndim != 2:
            raise ValueError('voxel_values must have shape (num_indices, num_slices)')

        # Get the geometry parameters and the system matrix and channel indices
        num_views, num_det_rows, num_det_channels = sinogram_shape
        Aji, channel_index = ParallelBeamModel.compute_Aji_channel_index(voxel_indices, angle, geometry_params,
                                                                         sinogram_shape)

        # Add axes to be able to broadcast while multiplying.
        # sinogram_values has shape num_indices x (2P+1) x num_slices
        sinogram_values = (Aji[:, :, None] * voxel_values[:, None, :])

        # Now sum over indices into the locations specified by channel_index.
        # Directly using index_add for indexed updates
        sinogram_view = jnp.zeros((num_det_rows, num_det_channels))  # num_det_rows x num_det_channels

        # Apply the vectorized update function with a vmap over slices
        # sinogram_view is num_det_rows x num_det_channels, sinogram_values is num_indices x (2P+1) x num_det_rows
        sinogram_values = sinogram_values.transpose((2, 0, 1)).reshape((num_det_rows, -1))
        sinogram_view = sinogram_view.at[:, channel_index.flatten()].add(sinogram_values)
        del Aji, channel_index
        return sinogram_view

    @staticmethod
    @jax.jit
    def compute_Aji_channel_index(voxel_indices, angles, geometry_params, sinogram_shape):

        # TODO:  P should be included in function signature with a partial on the jit
        warnings.warn('Compiling for indices length = {}'.format(voxel_indices.shape))
        warnings.warn('Using hard-coded detectors per side.  These should be set dynamically based on the geometry.')
        P = 1  # This is the assumed number of channels per side

        # Get all the geometry parameters
        delta_det_channel, det_channel_offset, delta_pixel_recon, num_recon_rows, num_recon_cols = geometry_params[:-1]

        num_views, num_det_rows, num_det_channels = sinogram_shape

        # Convert the index into (i,j,k) coordinates corresponding to the indices into the 3D voxel array
        recon_shape = (num_recon_rows, num_recon_cols)
        row_index, col_index = jnp.unravel_index(voxel_indices, recon_shape)

        # Compute the x,y position of the voxel relative to the center of rotation
        # Assumes: rows index top to bottom; slice is viewed from the top; rotation of object is clockwise
        y_pos = -delta_pixel_recon * (row_index - ((num_recon_rows - 1.0) / 2.0))  # length = num_indices
        x_pos = delta_pixel_recon * (col_index - ((num_recon_cols - 1.0) / 2.0))

        # Compute projection of the scalar center-of-rotation onto the detector in ALUs
        channel_center = (delta_det_channel * (num_det_channels - 1.0) / 2.0) + det_channel_offset

        # Precompute cosine and sine of view angle
        # angles = angles.reshape((-1, 1))  # Reshape to be a column vector of size num_views x 1
        cosine = jnp.cos(angles)    # length = num_views
        sine = jnp.sin(angles)      # length = num_views

        # Rotate coordinates of pixel
        x_pos_rot = cosine * x_pos + sine * y_pos  # length = num_indices
        # y_pos_rot = -sine*x_pos + cosine*y_pos

        # Calculate cos alpha = cos ( smallest angle between source-voxel line and voxel edge )
        cos_alpha = jnp.maximum(jnp.abs(cosine), jnp.abs(sine))  # length = num_indices

        # Calculate W = length of projection of flattened voxel on detector
        W = delta_pixel_recon * cos_alpha  # length = num_indices

        # Compute the location on the detector in ALU of the projected center of the voxel
        x_pos_on_detector = x_pos_rot + channel_center  # length = num_indices

        # Compute a jnp array with 2P+1 entries that are the channel indices of the relevant channels
        # Hardwired for P=1, i.e., 3 pixel window
        channel_index = jnp.round(x_pos_on_detector / delta_det_channel).astype(int)  # length = num_indices
        channel_index = channel_index.reshape((-1, 1))
        # Compute channel indices for 3 adjacent channels at each view angle
        # Should be num_indices x 3
        channel_index = jnp.concatenate([channel_index - 1, channel_index, channel_index + 1], axis=-1)

        # Compute the distance of each channel from the projected center of the voxel
        delta = jnp.abs(
            channel_index * delta_det_channel - x_pos_on_detector.reshape((-1, 1)))  # Should be num_indices x 3

        # Calculate L = length of intersection between detector element and projection of flattened voxel
        tmp1 = (W + delta_det_channel) / 2.0  # length = num_indices
        tmp2 = (W - delta_det_channel) / 2.0  # length = num_indices
        Lv = jnp.maximum(tmp1 - jnp.maximum(jnp.abs(tmp2), delta), 0)  # Should be num_indices x 3

        # Compute the values of Aij
        Aji = (delta_pixel_recon / cos_alpha) * (Lv / delta_det_channel)  # Should be num_indices x 3
        Aji = Aji * (channel_index >= 0) * (channel_index < num_det_channels)
        return Aji, channel_index
