import warnings

import jax
from jax import numpy as jnp, lax

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
        Convenience function to get a list of the primary geometry parameters for projection.

        Returns:
            List of delta_det_channel, det_channel_offset, delta_pixel_recon,
            num_recon_rows, num_recon_cols, num_recon_slices
        """
        geometry_params = self.get_params(['delta_det_channel', 'det_channel_offset', 'delta_pixel_recon',
                                           'num_recon_rows', 'num_recon_cols', 'num_recon_slices'])

        return geometry_params

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
        angles = self.get_params('angles')
        sinogram_shape = self.get_params('sinogram_shape')
        voxel_batch_size, view_batch_size = self.get_params(['voxel_batch_size', 'view_batch_size'])

        def sparse_forward_project_fcn(voxel_values, voxel_indices):
            """
            Compute the sinogram obtained by forward projecting the specified voxels. The voxels
            are determined using 2D indices into a flattened array of shape (num_rows, num_cols),
            and for each such 2D index, the voxels in all slices at that location are projected.

            Args:
                voxel_values (ndarray or jax array): 2D array of shape (len(voxel_indices), num_slices) of voxel values
                voxel_indices (ndarray or jax array): 1D array of indices into a flattened array of shape (num_rows, num_cols)

            Returns:
                3D array of shape (num_views, num_det_rows, num_det_cols)
            """
            num_voxels = len(voxel_indices)
            if voxel_batch_size is None or voxel_batch_size >= num_voxels:
                sinogram = sparse_forward_project_voxel_batch(voxel_values, voxel_indices)
            else:
                num_batches = num_voxels // voxel_batch_size
                length_of_batches = num_batches * voxel_batch_size
                voxel_values_batched = jnp.reshape(voxel_values[:length_of_batches],
                                                    (num_batches, voxel_batch_size, -1))
                voxel_indices_batched = jnp.reshape(voxel_indices[:length_of_batches], (num_batches, voxel_batch_size))

                # Set up a scan over the voxel batches.  We'll project one batch to a sinogram,
                # then add that to the accumulated sinogram
                initial_sinogram = jnp.zeros(sinogram_shape)
                initial_carry = [view_batch_size, initial_sinogram]
                values_indices = (voxel_values_batched, voxel_indices_batched)
                final_carry, _ = lax.scan(forward_project_accumulate, initial_carry, values_indices)

                # Get the sinogram from these batches, and add in any leftover voxels
                sinogram = final_carry[1]
                num_remaining = num_voxels - num_batches * voxel_batch_size
                if num_remaining > 0:
                    end_batch_values = voxel_values[-num_remaining:]
                    end_batch_indices = voxel_indices[-num_remaining:]
                    sinogram += sparse_forward_project_voxel_batch(end_batch_values, end_batch_indices)

            return sinogram

        def forward_project_accumulate(carry, values_indices_batch):
            """

            Args:
                carry:
                values_indices_batch:

            Returns:

            """
            view_batch_size, cur_sino = carry
            voxel_values, voxel_indices = values_indices_batch
            cur_sino += sparse_forward_project_voxel_batch(voxel_values, voxel_indices)

            return [view_batch_size, cur_sino], None

        def sparse_forward_project_voxel_batch(voxel_values, voxel_indices):
            """
            Compute the sinogram obtained by forward projecting the specified voxels. The voxels
            are determined using 2D indices into a flattened array of shape (num_rows, num_cols),
            and for each such 2D index, the voxels in all slices at that location are projected.

            Args:
                voxel_values: 2D array of shape (len(voxel_indices), num_slices) of voxel values
                voxel_indices: 1D array of indices into a flattened array of shape (num_rows, num_cols)

            Returns:
                3D array of shape (num_views, num_det_rows, num_det_cols)
            """
            num_views = len(angles)
            forward_vmap = jax.vmap(self.forward_project_voxels_one_view, in_axes=(None, None, 0, None, None))

            if view_batch_size is None or view_batch_size >= num_views:
                sinogram = forward_vmap(voxel_values, voxel_indices, angles, geometry_params, sinogram_shape)
            else:
                num_batches = num_views // view_batch_size
                angles_batched = jnp.reshape(angles[0:num_batches * view_batch_size],(num_batches, view_batch_size))

                def forward_map(angle_batch):
                    return forward_vmap(voxel_values, voxel_indices, angle_batch, geometry_params,
                                        sinogram_shape)

                sinogram = jax.lax.map(forward_map, angles_batched)
                sinogram = jnp.reshape(sinogram, (num_batches * view_batch_size,) + sinogram.shape[2:])
                num_remaining = num_views - num_batches * view_batch_size
                if num_remaining > 0:
                    end_batch = angles[-num_remaining:]
                    end_views = forward_map(end_batch)
                    sinogram = jnp.concatenate((sinogram, end_views), axis=0)

            return sinogram

        def sparse_back_project_fcn(sinogram, indices, coeff_power=1):
            num_voxels = len(indices)
            if voxel_batch_size is None or voxel_batch_size >= num_voxels:
                voxel_values = ParallelBeamModel.back_project_to_voxels_scan(sinogram, indices, angles, geometry_params,
                                                                             coeff_power=coeff_power)
            else:
                num_batches = num_voxels // voxel_batch_size
                length_of_batches = num_batches * voxel_batch_size
                indices_batched = jnp.reshape(indices[:length_of_batches], (num_batches, voxel_batch_size))

                def backward_map(indices_batch):
                    return ParallelBeamModel.back_project_to_voxels_scan(sinogram, indices_batch, angles, geometry_params,
                                                                         coeff_power=coeff_power)

                voxel_values = jax.lax.map(backward_map, indices_batched)
                voxel_values = voxel_values.reshape((length_of_batches, -1))
                num_remaining = num_voxels - num_batches * voxel_batch_size
                if num_remaining > 0:
                    end_batch = indices[-num_remaining:]
                    end_values = backward_map(end_batch)
                    voxel_values = jnp.concatenate((voxel_values, end_values), axis=0)
            return voxel_values

        self._sparse_forward_project = jax.jit(sparse_forward_project_fcn)
        self._sparse_back_project = jax.jit(sparse_back_project_fcn, static_argnums=(2,))

    @staticmethod
    def back_project_to_voxels_scan(sinogram, voxel_indices, angles, geometry_params, coeff_power=1):
        """
        Use jax.lax.scan to backproject one view at a time and accumulate the results in the specified voxels.

        Args:
            sinogram:
            voxel_indices:
            angles:
            geometry_params:
            voxel_batch_size:
            coeff_power:

        Returns:

        """
        # TODO:  Implement voxel_batch_size
        # jax.lax.scan applies a function to each entry indexed by the leading dimension of its input, then incorporates
        # the output of that function into an accumulator.  Here we apply backproject_accumulate to one sinogram
        # view and the corresponding angle.
        initial_bp = jnp.zeros((voxel_indices.shape[0], sinogram.shape[1]))
        extra_args = voxel_indices, geometry_params, coeff_power
        initial_carry = [extra_args, initial_bp]
        sino_angles = (sinogram, angles)
        # Use lax.scan to process each (slice, angle) pair of 'sino_angles'
        final_carry, _ = lax.scan(ParallelBeamModel.backproject_accumulate, initial_carry, sino_angles)

        return final_carry[1]

    @staticmethod
    def backproject_accumulate(carry, view_angle_pair):
        """

        Args:
            carry:
            view_angle_pair:

        Returns:

        """
        extra_args, accumulated = carry
        sinogram_view, angle = view_angle_pair
        voxel_indices, geometry_params, coeff_power = extra_args
        bp_view = ParallelBeamModel.back_project_one_view_to_voxels(sinogram_view, voxel_indices, angle,
                                                                    geometry_params, coeff_power)
        accumulated += bp_view
        del bp_view
        return [extra_args, accumulated], None

    @staticmethod
    @jax.jit
    def back_project_one_view_to_voxels(sinogram_view, voxel_indices, angle, geometry_params, coeff_power=1):
        """

        Args:
            sinogram_view:
            voxel_indices:
            angle:
            geometry_params:
            coeff_power:

        Returns:

        """
        bp_vmap = jax.vmap(ParallelBeamModel.back_project_one_view_to_voxel, in_axes=(None, 0, None, None, None))
        bp = bp_vmap(sinogram_view, voxel_indices, angle, geometry_params, coeff_power)
        return bp

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
    @jax.jit
    def backproject_to_voxel(sinogram, voxel_index, angles, geometry_params, coeff_power=1):
        """
        Calculate the backprojection value at a specified recon voxel given the sinogram and various parameters.
        This code uses the distance driven projector.

        Args:
            sinogram: [jax array] the sinogram to be back projected
            voxel_index: the integer index into flattened recon - need to apply unravel_index(voxel_index, recon_shape) to get i, j, k
            angles:
            geometry_params:
            coeff_power: [int] backproject using the coefficients of (A_ij ** coeff_power).
                Normally 1, but should be 2 when computing theta 2.

        Returns:
            The value of the voxel at the input index obtained by backprojecting the input sinogram.
        """

        # Get the geometry parameters and the system matrix and channel indices
        num_views = sinogram.shape[0]

        Aji, channel_index = ParallelBeamModel.compute_Aji_channel_index(voxel_index, angles, geometry_params,
                                                                         sinogram.shape)

        # Extract out the relevant entries from the sinogram
        view_index = jnp.arange(num_views)
        view_index = jnp.concatenate((view_index, view_index, view_index))  # Should be 3*num_views
        sinogram_array = sinogram[view_index, :, channel_index.T.flatten()]

        # Compute dot product
        return jnp.sum(sinogram_array * (Aji.T.reshape((-1, 1))**coeff_power), axis=0)

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

    def compute_hessian_diagonal(self, weights=None):
        """
        Computes the diagonal of the Hessian matrix, which is computed by doing a backprojection of the weight
        matrix except using the square of the coefficients in the backprojection to a given voxel.
        One of weights or sinogram_shape must be not None. If weights is not None, it must be an array with the same
        shape as the sinogram to be backprojected.  If weights is None, then a weights matrix will be computed as an
        array of ones of size sinogram_shape.

        Args:
            weights (ndarray or None): The weights with shape (views, rows, channels)

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

        num_recon_rows, num_recon_cols, num_recon_slices = geometry_params[-3:]
        max_index = num_recon_rows * num_recon_cols
        indices = jnp.arange(max_index)

        hessian_diagonal = self._sparse_back_project(weights, indices, coeff_power=2)

        return hessian_diagonal.reshape((num_recon_rows, num_recon_cols, num_recon_slices))
