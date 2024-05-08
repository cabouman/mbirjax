import warnings

import jax
import jax.numpy as jnp
from functools import partial
from mbirjax import TomographyModel


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
    >>> model = mbirjax.ParallelBeamModel((180, 256, 10), angles)

    See Also
    --------
    TomographyModel : The base class from which this class inherits.
    """

    def __init__(self, sinogram_shape, angles, **kwargs):
        # Convert the view-dependent vectors to an array
        # This is more complicated than needed with only a single view-dependent vector but is included to
        # illustrate the process as shown in TemplateModel
        view_dependent_vecs = [vec.flatten() for vec in [angles]]
        try:
            view_params_array = jnp.stack(view_dependent_vecs, axis=1)
        except ValueError as e:
            raise ValueError("Incompatible view dependent vector lengths:  all view-dependent vectors must have the "
                             "same length.")
        super().__init__(sinogram_shape, view_params_array=view_params_array, **kwargs)

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
            List of delta_det_channel, det_channel_offset, delta_pixel_recon,
            num_recon_rows, num_recon_cols, num_recon_slices
        """
        geometry_params = self.get_params(['delta_det_channel', 'det_channel_offset', 'delta_pixel_recon'])

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
                Normally 1, but should be 2 for compute_hessian_diagonal.

        Returns:
            The value of the voxel for all slices at the input index (i.e., a voxel cylinder) obtained by backprojecting
            the input sinogram view.
        """

        # Get the part of the system matrix and channel indices for this voxel
        sinogram_view_shape = (1,) + sinogram_view.shape  # Adjoin a leading 1 to indicate a single view sinogram
        view_projector_params = (sinogram_view_shape,) + projector_params[1:]
        Aij_value, Aij_index = ParallelBeamModel.compute_Aij_voxels_single_view(voxel_index, angle, view_projector_params)

        # Extract out the relevant entries from the sinogram
        sinogram_array = sinogram_view[:, Aij_index.T.flatten()]

        # Compute dot product
        return jnp.sum(sinogram_array * (Aij_value**coeff_power), axis=1)

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

        Returns:
            jax array of shape (num_det_rows, num_det_channels)
        """
        if voxel_values.ndim != 2:
            raise ValueError('voxel_values must have shape (num_indices, num_slices)')

        # Get the geometry parameters and the system matrix and channel indices
        num_views, num_det_rows, num_det_channels = projector_params[0]
        Aij_value, Aij_channel = ParallelBeamModel.compute_Aij_voxels_single_view(voxel_indices, angle, projector_params)

        # Add axes to be able to broadcast while multiplying.
        # sinogram_values has shape num_indices x (2p+1) x num_slices
        sinogram_values = (Aij_value[:, :, None] * voxel_values[:, None, :])

        # Now sum over indices into the locations specified by Aij_channel.
        # Directly using index_add for indexed updates
        sinogram_view = jnp.zeros((num_det_rows, num_det_channels))  # num_det_rows x num_det_channels

        # Apply the vectorized update function with a vmap over slices
        # sinogram_view is num_det_rows x num_det_channels, sinogram_values is num_indices x (2p+1) x num_det_rows
        sinogram_values = sinogram_values.transpose((2, 0, 1)).reshape((num_det_rows, -1))
        sinogram_view = sinogram_view.at[:, Aij_channel.flatten()].add(sinogram_values)
        del Aij_value, Aij_channel
        return sinogram_view


    @staticmethod
    @partial(jax.jit, static_argnums=3)
    def compute_Aij_voxels_single_view(voxel_indices, angle, projector_params, p=1):
        """
        Calculate the sparse system matrix for a subset of voxels and a single view.
        The function returns a sparse matrix specified by the matrix values and associated detector column index.
        Since this is for parallel beam geometry, the values are assumed to be the same for each row/slice pair.

        Args:
            voxel_indices (jax array of int):  1D vector of indices into flattened array of size num_rows x num_cols.
            angle (float):  Angle for this single view
            projector_params (tuple):  tuple of (sinogram_shape, recon_shape, get_geometry_params())
            p (int, optional, default=1):  # This is the assumed number of channels per side

        Returns:
            Aij_value (num indices, 2p+1), Aji_channel (num indices, 2p+1)
        """
        warnings.warn('Compiling for indices length = {}'.format(voxel_indices.shape))
        warnings.warn('Using hard-coded detectors per side.  These should be set dynamically based on the geometry.')

        # Get all the geometry parameters
        geometry_params = projector_params[2]
        delta_det_channel, det_channel_offset, delta_pixel_recon = geometry_params

        num_views, num_det_rows, num_det_channels = projector_params[0]
        num_recon_rows, num_recon_cols = projector_params[1][:2]

        # Convert the index into (i,j,k) coordinates corresponding to the indices into the 3D voxel array
        recon_shape_2d = (num_recon_rows, num_recon_cols)
        row_index, col_index = jnp.unravel_index(voxel_indices, recon_shape_2d)

        # Compute the x,y position of the voxel relative to the center of rotation
        # Assumes: rows index top to bottom; slice is viewed from the top; rotation of object is clockwise
        y_pos = -delta_pixel_recon * (row_index - ((num_recon_rows - 1.0) / 2.0))  # length = num_indices
        x_pos = delta_pixel_recon * (col_index - ((num_recon_cols - 1.0) / 2.0))

        # Compute projection of the scalar center-of-rotation onto the detector in ALUs
        channel_center = (delta_det_channel * (num_det_channels - 1.0) / 2.0) + det_channel_offset

        # Precompute cosine and sine of view angle
        cosine = jnp.cos(angle)    # length = num_views
        sine = jnp.sin(angle)      # length = num_views

        # Rotate coordinates of pixel
        x_pos_rot = cosine * x_pos + sine * y_pos  # length = num_indices
        # y_pos_rot = -sine*x_pos + cosine*y_pos

        # Calculate cos alpha = cos ( smallest angle between source-voxel line and voxel edge )
        cos_alpha = jnp.maximum(jnp.abs(cosine), jnp.abs(sine))  # length = num_indices

        # Calculate W = length of projection of flattened voxel on detector
        W = delta_pixel_recon * cos_alpha  # length = num_indices

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
        Aij_value = (delta_pixel_recon / cos_alpha) * (Lv / delta_det_channel)  # Should be num_indices x 2p+1
        Aij_value = Aij_value * (Aij_channel >= 0) * (Aij_channel < num_det_channels)
        return Aij_value, Aij_channel
