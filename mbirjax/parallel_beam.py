import warnings

import jax
from jax import numpy as jnp, lax

from mbirjax import TomographyModel


class ParallelBeamModel(TomographyModel):
    """
    A class designed for handling forward and backward projections in a parallel beam geometry, extending the
    :class:`TomographyModel`. This class offers specialized methods and parameters tailored for parallel beam setups.

    This class inherits all methods and properties from the :class:`~mbirjax.TomographyModel` and may override some
    to suit parallel beam geometrical requirements. See the documentation of the parent class for standard methods
    like setting parameters and performing projections and reconstructions.

    Parameters
    ----------
    angles : jnp.ndarray
        A 1D array of projection angles, in radians, specifying the angle of each projection relative to the origin.
    sinogram_shape : tuple
        Shape of the sinogram as a tuple in the form `(views, rows, channels)`, where 'views' is the number of
        different projection angles, 'rows' correspond to the number of detector rows, and 'channels' index columns of
        the detector that are assumed to be aligned with the rotation axis..
    **kwargs : dict
        Additional keyword arguments that are passed to the :class:`~mbirjax.TomographyModel` constructor. These can
        include settings and configurations specific to the tomography model such as noise models or image dimensions.
        Refer to the :class:`~mbirjax.TomographyModel` documentation for a detailed list of possible parameters.

    Examples
    --------
    Initialize a parallel beam model with specific angles and sinogram shape:

    >>> angles = jnp.array([0, jnp.pi/4, jnp.pi/2])
    >>> model = ParallelBeamModel(angles, (180, 256, 10))

    See Also
    --------
    TomographyModel : The base class from which this class inherits.
    """


    def __init__(self, angles, sinogram_shape, **kwargs):

        super().__init__(sinogram_shape, angles=angles, **kwargs)

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
        Get a list of the primary geometry parameters.
        Returns:
            List of delta_det_channel, det_channel_offset, delta_pixel_recon, num_recon_rows, num_recon_cols
        """
        geometry_params = self.get_params(['delta_det_channel', 'det_channel_offset', 'delta_pixel_recon'])
        num_recon_rows, num_recon_cols = self.get_params(['num_recon_rows', 'num_recon_cols'])

        # TODO:  Need to include detector rows/recon slices
        return geometry_params + [num_recon_rows, num_recon_cols]

    def compile_projectors(self):
        """
        Compute the forward and back projectors for this set of angles
        Args:

        Returns:
            A callable of the form back_projector(sinogram, indices), where sinogram is an array with shape
            (views, rows, channels) and indices is a 1D array of integer indices into the flattened reconstruction
            volume.
            The output of a call to back_projector is the backprojection of the sinogram onto the corresponding voxels.

        Note:
            The returned function will be jit compiled each time it is called with a new shape of input.  If it is
            called multiple times with the same shape of input, then the cached version will be used, which will
            give reduced execution time relative to the initial call.
        """
        geometry_params = self.get_geometry_parameters()
        cos_sin_angles = self._get_cos_sin_angles(self.get_params('angles'))
        sinogram_shape = self.get_params('sinogram_shape')

        def sparse_back_project_fcn(sinogram, indices, voxel_batch_size=None, coeff_power=1):
            return ParallelBeamModel.back_project_to_voxels_scan(sinogram, indices, cos_sin_angles, geometry_params,
                                                                 voxel_batch_size=voxel_batch_size,
                                                                 coeff_power=coeff_power)

        def sparse_forward_project_fcn(voxel_values, voxel_indices, view_batch_size=None):
            """
            Compute the sinogram obtained by forward projecting the specified voxels. The voxels
            are determined using 2D indices into a flattened array of shape (num_rows, num_cols),
            and for each such 2D index, the voxels in all slices at that location are projected.

            Args:
                voxel_values: 2D array of shape (len(voxel_indices), num_slices) of voxel values
                voxel_indices: 1D array of indices into a flattened array of shape (num_rows, num_cols)
                view_batch_size: Number of views to process in one batch (to limit memory use)

            Returns:
                3D array of shape (num_views, num_det_rows, num_det_cols)
            """
            num_views = cos_sin_angles.shape[1]
            forward_vmap = jax.vmap(self.forward_project_voxels_one_view, in_axes=(None, None, 1, None, None))

            if view_batch_size is None or view_batch_size >= num_views:
                sinogram = forward_vmap(voxel_values, voxel_indices, cos_sin_angles, geometry_params, sinogram_shape)
            else:
                num_batches = num_views // view_batch_size
                cos_sin_angles_batched = jnp.reshape(cos_sin_angles.T[0:num_batches * view_batch_size],
                                                     (num_batches, view_batch_size, 2))
                cos_sin_angles_batched = cos_sin_angles_batched.transpose((0, 2, 1))

                def forward_map(cs_angle_batch):
                    return forward_vmap(voxel_values, voxel_indices, cs_angle_batch, geometry_params,
                                        sinogram_shape)

                sinogram = jax.lax.map(forward_map, cos_sin_angles_batched)
                sinogram = jnp.reshape(sinogram, (num_batches * view_batch_size,) + sinogram.shape[2:])
                num_remaining = num_views - num_batches * view_batch_size
                if num_remaining > 0:
                    end_batch = cos_sin_angles[:, -num_remaining:]
                    end_views = forward_map(end_batch)
                    sinogram = jnp.concatenate((sinogram, end_views), axis=0)

            return sinogram

        self.sparse_forward_project = jax.jit(sparse_forward_project_fcn, static_argnums=(2,))
        self.sparse_back_project = jax.jit(sparse_back_project_fcn, static_argnums=(2,))

    @staticmethod
    def back_project_to_voxels_vmap(sinogram, voxel_indices, cos_sin_angles, geometry_params, coeff_power=1,
                                    voxel_batch_size=None):
        """
        Use jax.vmap to backproject one view at a time and then sum over voxels.
        Args:
            sinogram:
            voxel_indices:
            cos_sin_angles:
            geometry_params:
            coeff_power:
            voxel_batch_size:

        Returns:

        """
        num_voxels = voxel_indices.shape[0]
        backward_vmap = jax.vmap(ParallelBeamModel.back_project_one_view_to_voxels, in_axes=(0, None, 1, None, None))
        if voxel_batch_size is None or voxel_batch_size >= num_voxels:
            bp_all_views = backward_vmap(sinogram, voxel_indices, cos_sin_angles, geometry_params, coeff_power)
            bp = jnp.sum(bp_all_views, axis=0)
        else:
            num_batches = num_voxels // voxel_batch_size
            voxel_inds_batched = jnp.reshape(voxel_indices[0:num_batches * voxel_batch_size],
                                             (num_batches, voxel_batch_size))

            def backward_map(voxel_inds_batch):
                return backward_vmap(sinogram, voxel_inds_batch, cos_sin_angles, geometry_params, coeff_power)

            bp_all_views = jax.lax.map(backward_map, voxel_inds_batched)
            bp = jnp.sum(bp_all_views, axis=1)
            bp = jnp.reshape(bp, (num_batches * voxel_batch_size, ) + bp.shape[2:])
            num_remaining = num_voxels - num_batches * voxel_batch_size
            if num_remaining > 0:
                end_batch = voxel_indices[-num_remaining:]
                end_bp_all_views = backward_map(end_batch)
                end_bp = jnp.sum(end_bp_all_views, axis=0)
                bp = jnp.concatenate((bp, end_bp), axis=0)

        return bp

    @staticmethod
    def back_project_to_voxels_scan(sinogram, voxel_indices, cos_sin_angles, geometry_params, coeff_power=1,
                                    voxel_batch_size=None):
        """
        Use jax.lax.scan to backproject one view at a time and accumulate the results in the specified voxels.
        Args:
            sinogram:
            voxel_indices:
            cos_sin_angles:
            geometry_params:
            coeff_power:
            voxel_batch_size:

        Returns:

        """
        # jax.lax.scan applies a function to each entry indexed by the leading dimension of its input, the incorporates
        # the output of that function into an accumulator.  Here we apply accumulate and project to one sinogram
        # view and the cos and sin of the corresponding angle.
        initial_bp = jnp.zeros((voxel_indices.shape[0], sinogram.shape[1]))
        extra_args = voxel_indices, geometry_params, coeff_power
        initial_carry = [extra_args, initial_bp]
        sino_angles = (sinogram, cos_sin_angles.T)
        # Use lax.scan to process each (slice, angle) pair of 'sino_angles'
        final_carry, _ = lax.scan(ParallelBeamModel.accumulate_and_project, initial_carry, sino_angles)

        return final_carry[1]

    @staticmethod
    def accumulate_and_project(carry, sino_angle_pair):
        """

        Args:
            carry:
            sino_angle_pair:

        Returns:

        """
        extra_args, accumulated = carry
        sinogram_view, cos_sin_angle = sino_angle_pair
        voxel_indices, geometry_params, coeff_power = extra_args
        bp_view = ParallelBeamModel.back_project_one_view_to_voxels(sinogram_view, voxel_indices, cos_sin_angle,
                                                                    geometry_params, coeff_power)
        accumulated += bp_view
        del bp_view
        return [extra_args, accumulated], None

    @staticmethod
    @jax.jit
    def back_project_one_view_to_voxels(sinogram_view, voxel_indices, cos_sin_angle, geometry_params, coeff_power=1):
        """

        Args:
            sinogram_view:
            voxel_indices:
            cos_sin_angle:
            geometry_params:
            coeff_power:

        Returns:

        """
        bp_vmap = jax.vmap(ParallelBeamModel.back_project_one_view_to_voxel, in_axes=(None, 0, None, None, None))
        bp = bp_vmap(sinogram_view, voxel_indices, cos_sin_angle, geometry_params, coeff_power)
        return bp

    @staticmethod
    def back_project_one_view_to_voxel(sinogram_view, voxel_index, cos_sin_angle, geometry_params, coeff_power=1):
        """
        Calculate the backprojection value at a specified recon voxel given a sinogram view and various parameters.
        This code uses the distance driven projector.
        Args:
            sinogram_view: [jax array] one view of the sinogram to be back projected
            voxel_index: the integer index into flattened recon - need to apply unravel_index(voxel_index, recon_shape) to get i, j, k
            cos_sin_angle:
            geometry_params:
            coeff_power: [int] backproject using the coefficients of (A_ij ** coeff_power).
                Normally 1, but should be 2 when computing theta 2.
        Returns:
            The value of the voxel at the input index obtained by backprojecting the input sinogram.
        """

        # Get the geometry parameters and the system matrix and channel indices
        num_det_rows, num_det_channels = sinogram_view.shape
        num_slices = num_det_rows

        Aji, channel_index = ParallelBeamModel.compute_Aji_channel_index(voxel_index, cos_sin_angle, geometry_params,
                                                                         (1,) + sinogram_view.shape)

        # Extract out the relevant entries from the sinogram
        sinogram_array = sinogram_view[:, channel_index.T.flatten()]

        # Compute dot product
        return jnp.sum(sinogram_array * (Aji**coeff_power), axis=1)

    def compute_hessian_diagonal(self, weights=None, voxel_batch_size=None):
        """
        Computes the diagonal of the Hessian matrix, which is computed by doing a backprojection of the weight
        matrix except using the square of the coefficients in the backprojection to a given voxel.
        One of weights or sinogram_shape must be not None. If weights is not None, it must be an array with the same
        shape as the sinogram to be backprojected.  If weights is None, then a weights matrix will computed as an
        array of ones of size sinogram_shape.
        Args:
            weights (ndarray or None): The weights with shape (views, rows, channels)
            voxel_batch_size:

        Returns:
            An array that is the same size as the reconstruction.
        """
        sinogram_shape = self.get_params('sinogram_shape')
        if weights is None:
            weights = jnp.ones(sinogram_shape)
        elif weights.shape != sinogram_shape:
            error_message = 'Weights must be constant or an array of the same shape as sinogram'
            error_message += '\nGot weights.shape = {}, but sinogram.shape = {}'.format(weights.shape, sinogram_shape)
            raise ValueError(error_message)
        geometry_params = self.get_geometry_parameters()
        angles = self.get_params('angles')
        cos_sin_angles = self._get_cos_sin_angles(angles)

        num_recon_rows, num_recon_cols = geometry_params[-2:]
        num_recon_slices = weights.shape[1]
        max_index = num_recon_rows * num_recon_cols
        indices = jnp.arange(max_index)

        hessian_diagonal = self.sparse_back_project(weights, indices, voxel_batch_size=voxel_batch_size, coeff_power=2)

        return hessian_diagonal.reshape((num_recon_rows, num_recon_cols, num_recon_slices))

    @staticmethod
    def forward_project_voxels_one_view(voxel_values, voxel_indices, cos_sin_angle, geometry_params, sinogram_shape):
        """
        Forward project a set of voxels determined by indices into the flattened array of size num_rows x num_cols.
        Args:
            voxel_values (jax array):  2D array of shape (num_indices, num_slices) of voxel values, where
                voxel_values[i, j] is the value of the voxel in slice j at the location determined by indices[i].
            voxel_indices (jax array of int):  1D vector of indices into flattened array of size num_rows x num_cols.
            cos_sin_angle (jax array):  2D array of cosines and sines from _get_cos_sin_angles()
            geometry_params (list): Geometry parameters from get_geometry_params()
            sinogram_shape (tuple): Sinogram shape

        Returns:
            jax array of shape sinogram_shape
        """
        if voxel_values.ndim != 2:
            raise ValueError('voxel_values must have shape (num_indices, num_slices)')
        num_slices = voxel_values.shape[1]

        # Get the geometry parameters and the system matrix and channel indices
        num_views, num_det_rows, num_det_channels = sinogram_shape
        Aji, channel_index = ParallelBeamModel.compute_Aji_channel_index(voxel_indices, cos_sin_angle, geometry_params,
                                                                         sinogram_shape)

        # Add axes to be able to broadcast while multiplying.
        # sinogram_values has shape num_indices x (2P+1) x num_slices
        sinogram_values = (Aji[:, :, None] * voxel_values[:, None, :])

        # Now sum over indices into the locations specified by channel_index.
        # Directly using index_add for indexed updates
        sinogram_view = jnp.zeros((num_slices, num_det_channels))  # num_det_rows x num_det_channels

        # Apply the vectorized update function with a vmap over slices
        # sinogram_view is num_slices x num_det_channels, sinogram_values is num_indices x (2P+1) x num_slices
        sinogram_values = sinogram_values.transpose((2, 0, 1)).reshape((num_slices, -1))
        sinogram_view = sinogram_view.at[:, channel_index.flatten()].add(sinogram_values)
        del Aji, channel_index
        return sinogram_view

    @staticmethod
    @jax.jit
    def backproject_to_voxel(sinogram, voxel_index, cos_sin_angles, geometry_params, coeff_power=1):
        """
        Calculate the backprojection value at a specified recon voxel given the sinogram and various parameters.
        This code uses the distance driven projector.
        Args:
            sinogram: [jax array] the sinogram to be back projected
            voxel_index: the integer index into flattened recon - need to apply unravel_index(voxel_index, recon_shape) to get i, j, k
            cos_sin_angles:
            geometry_params:
            coeff_power: [int] backproject using the coefficients of (A_ij ** coeff_power).
                Normally 1, but should be 2 when computing theta 2.
        Returns:
            The value of the voxel at the input index obtained by backprojecting the input sinogram.
        """

        # Get the geometry parameters and the system matrix and channel indices
        num_views, num_det_rows, num_det_channels = sinogram.shape
        num_slices = num_det_rows

        Aji, channel_index = ParallelBeamModel.compute_Aji_channel_index(voxel_index, cos_sin_angles, geometry_params,
                                                                         sinogram.shape)

        # Extract out the relevant entries from the sinogram
        view_index = jnp.arange(num_views)
        view_index = jnp.concatenate((view_index, view_index, view_index))  # Should be 3*num_views
        sinogram_array = sinogram[view_index, :, channel_index.T.flatten()]

        # Compute dot product
        return jnp.sum(sinogram_array * (Aji.T.reshape((-1, 1))**coeff_power), axis=0)

    @staticmethod
    @jax.jit
    def compute_Aji_channel_index(voxel_indices, cos_sin_angles, geometry_params, sinogram_shape):

        # TODO:  P should be included in function signature with a partial on the jit
        warnings.warn('Compiling for indices length = {}'.format(voxel_indices.shape))
        warnings.warn('Using hard-coded detectors per side.  These should be set dynamically based on the geometry.')
        P = 1  # This is the assumed number of channels per side

        # Get all the geometry parameters
        delta_det_channel, det_channel_offset, delta_pixel_recon, num_recon_rows, num_recon_cols = geometry_params

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
        cosine = cos_sin_angles[0:1].T  # jnp.cos(angles)    # length = num_views
        sine = cos_sin_angles[1:2].T  # jnp.sin(angles)      # length = num_views

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
