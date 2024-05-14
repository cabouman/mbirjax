import warnings

import jax
import jax.numpy as jnp


class Projectors:

    def __init__(self, tomography_model, forward_core, backward_core):

        self.tomography_model = tomography_model
        self.sparse_forward_project, self.sparse_back_project, self.compute_hessian_diagonal = None, None, None
        self.create_projectors(forward_core, backward_core)

    def create_projectors(self, forward_project_pixels_to_one_view, back_project_one_view_to_pixel):
        """
        Compute the forward and back projectors for this geometry and current view parameters
        Args:

        Returns:
            Nothing, but the class variables `sparse_forward_project`, `sparse_back_project`, and
            `compute_hessian_diagonal` are set to callable functions.  These are used to implement the following
            methods:

            * `sparse_forward_project`: :meth:`TomographyModel.sparse_forward_project`
            * `sparse_back_project`: :meth:`TomographyModel.sparse_back_project`
            * `compute_hessian_diagonal`: :meth:`TomographyModel.compute_hessian_diagonal`

        Note:
            The returned functions will be jit compiled each time they are called with a new shape of input.  If
            called multiple times with the same shape of input, then the cached version will be used, which will
            give reduced execution time relative to the initial call.

            This method requires geometry-specific implementations of
            :meth:`TomographyModel.forward_project_pixels_to_one_view` and
            :meth:`TomographyModel.back_project_one_view_to_pixel`.
        """
        geometry_params = self.tomography_model.get_geometry_parameters()
        sinogram_shape, recon_shape = self.tomography_model.get_params(['sinogram_shape', 'recon_shape'])
        projector_params = (tuple(sinogram_shape), tuple(recon_shape), tuple(geometry_params))
        view_params_array = self.tomography_model.get_params('view_params_array')
        pixel_batch_size, view_batch_size = self.tomography_model.get_params(['pixel_batch_size', 'view_batch_size'])

        def sparse_forward_project_fcn2(voxel_values, pixel_indices):
            num_pixels = len(pixel_indices)
            forward_vmap = jax.vmap(forward_project_pixels_to_one_view, in_axes=(0, 0))
            if pixel_batch_size is None or pixel_batch_size >= num_pixels:
                sinogram_view = forward_vmap(voxel_values, pixel_indices)
            else:
                num_batches = num_pixels // pixel_batch_size
                length_of_batches = num_batches * pixel_batch_size
                indices_batched = jnp.reshape(indices[:length_of_batches], (num_batches, pixel_batch_size))

                def backward_map(indices_batch):
                    return back_project_to_voxels_scan(sinogram, indices_batch, coeff_power=coeff_power)

                voxel_values = jax.lax.map(backward_map, indices_batched)
                voxel_values = voxel_values.reshape((length_of_batches, -1))
                num_remaining = num_pixels - num_batches * pixel_batch_size
                if num_remaining > 0:
                    end_batch = indices[-num_remaining:]
                    end_values = backward_map(end_batch)
                    voxel_values = jnp.concatenate((voxel_values, end_values), axis=0)
            return voxel_values

        def sparse_forward_project_fcn(voxel_values, pixel_indices):
            """
            Compute the sinogram obtained by forward projecting the specified voxels. The voxels
            are determined using 2D indices into a flattened array of shape (num_rows, num_cols),
            and for each such 2D index, the voxels in all slices at that location are projected.

            Args:
                voxel_values (ndarray or jax array): 2D array of shape (len(pixel_indices), num_slices) of voxel values
                pixel_indices (ndarray or jax array): 1D array of indices into a flattened array of shape (num_rows, num_cols)

            Returns:
                3D array of shape (num_views, num_det_rows, num_det_cols)
            """
            num_pixels = len(pixel_indices)
            if pixel_batch_size is None or pixel_batch_size >= num_pixels:
                sinogram = sparse_forward_project_voxel_batch(voxel_values, pixel_indices)
            else:
                num_batches = num_pixels // pixel_batch_size
                length_of_batches = num_batches * pixel_batch_size
                voxel_values_batched = jnp.reshape(voxel_values[:length_of_batches],
                                                    (num_batches, pixel_batch_size, -1))
                pixel_indices_batched = jnp.reshape(pixel_indices[:length_of_batches], (num_batches, pixel_batch_size))

                # Set up a scan over the voxel batches.  We'll project one batch to a sinogram,
                # then add that to the accumulated sinogram
                initial_sinogram = jnp.zeros(sinogram_shape)
                initial_carry = [view_batch_size, initial_sinogram]
                values_indices = (voxel_values_batched, pixel_indices_batched)
                final_carry, _ = jax.lax.scan(forward_project_accumulate, initial_carry, values_indices)

                # Get the sinogram from these batches, and add in any leftover voxels
                sinogram = final_carry[1]
                num_remaining = num_pixels - num_batches * pixel_batch_size
                if num_remaining > 0:
                    end_batch_values = voxel_values[-num_remaining:]
                    end_batch_indices = pixel_indices[-num_remaining:]
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
            voxel_values, pixel_indices = values_indices_batch
            cur_sino += sparse_forward_project_voxel_batch(voxel_values, pixel_indices)

            return [view_batch_size, cur_sino], None

        def sparse_forward_project_voxel_batch(voxel_values, pixel_indices):
            """
            Compute the sinogram obtained by forward projecting the specified voxels. The voxels
            are determined using 2D indices into a flattened array of shape (num_rows, num_cols),
            and for each such 2D index, the voxels in all slices at that location are projected.

            Args:
                voxel_values: 2D array of shape (len(pixel_indices), num_slices) of voxel values
                pixel_indices: 1D array of indices into a flattened array of shape (num_rows, num_cols)

            Returns:
                3D array of shape (num_views, num_det_rows, num_det_cols)
            """
            num_views = view_params_array.shape[0]
            forward_vmap = jax.vmap(forward_project_pixels_to_one_view, in_axes=(None, None, 0, None))

            if view_batch_size is None or view_batch_size >= num_views:
                sinogram = forward_vmap(voxel_values, pixel_indices, view_params_array, projector_params)
            else:
                num_batches = num_views // view_batch_size
                view_params_batched = jnp.reshape(view_params_array[0:num_batches * view_batch_size],
                                                  (num_batches, view_batch_size, -1))

                @jax.jit
                def forward_map(view_params_batch):
                    return forward_vmap(voxel_values, pixel_indices, view_params_batch, projector_params)

                sinogram = jax.lax.map(forward_map, view_params_batched)
                sinogram = jnp.reshape(sinogram, (num_batches * view_batch_size,) + sinogram.shape[2:])
                num_remaining = num_views - num_batches * view_batch_size
                if num_remaining > 0:
                    end_batch = view_params_array[-num_remaining:]
                    end_views = forward_map(end_batch)
                    sinogram = jnp.concatenate((sinogram, end_views), axis=0)

            return sinogram

        def sparse_back_project_fcn(sinogram, pixel_indices, coeff_power=1):
            num_pixels = len(pixel_indices)
            if pixel_batch_size is None or pixel_batch_size >= num_pixels:
                voxel_values = back_project_to_voxels_scan(sinogram, pixel_indices, coeff_power=coeff_power)
            else:
                num_batches = num_pixels // pixel_batch_size
                length_of_batches = num_batches * pixel_batch_size
                indices_batched = jnp.reshape(pixel_indices[:length_of_batches], (num_batches, pixel_batch_size))

                def backward_map(indices_batch):
                    return back_project_to_voxels_scan(sinogram, indices_batch, coeff_power=coeff_power)

                voxel_values = jax.lax.map(backward_map, indices_batched)
                voxel_values = voxel_values.reshape((length_of_batches, -1))
                num_remaining = num_pixels - num_batches * pixel_batch_size
                if num_remaining > 0:
                    end_batch = pixel_indices[-num_remaining:]
                    end_values = backward_map(end_batch)
                    voxel_values = jnp.concatenate((voxel_values, end_values), axis=0)
            return voxel_values

        def back_project_to_voxels_scan(sinogram, pixel_indices, coeff_power=1):
            """
            Use jax.lax.scan to backproject one view at a time and accumulate the results in the specified voxels.
            The individual backprojections from each view must be added to get the full backprojection.  This is
            done using the helper function backproject_accumulate.

            Args:
                sinogram:
                pixel_indices:
                coeff_power:

            Returns:

            """
            # jax.lax.scan applies a function to each entry indexed by the leading dimension of its input, then
            # incorporates the output of that function into an accumulator.  Here we apply backproject_accumulate to
            # one sinogram view and the corresponding view_params.
            initial_bp = jnp.zeros((pixel_indices.shape[0], sinogram.shape[1]))
            extra_args = pixel_indices, coeff_power
            initial_carry = [extra_args, initial_bp]
            sino_view_params = (sinogram, view_params_array)
            # Use lax.scan to process each (slice, view_params) pair of 'sino_view_params'
            final_carry, _ = jax.lax.scan(backproject_accumulate, initial_carry, sino_view_params)

            return final_carry[1]

        def backproject_accumulate(carry, view_params_pair):
            """

            Args:
                carry:
                view_params_pair:

            Returns:

            """
            extra_args, accumulated = carry
            sinogram_view, view_params = view_params_pair
            pixel_indices, coeff_power = extra_args
            bp_view = back_project_one_view_to_pixels(sinogram_view, pixel_indices, view_params, coeff_power)
            accumulated += bp_view
            del bp_view
            return [extra_args, accumulated], None

        @jax.jit
        def back_project_one_view_to_pixels(sinogram_view, pixel_indices, view_params, coeff_power=1):
            """
            Use vmap to do a backprojection from one view to multiple voxels.

            Args:
                sinogram_view:
                pixel_indices:
                view_params:
                coeff_power:

            Returns:

            """
            bp_vmap = jax.vmap(back_project_one_view_to_pixel, in_axes=(None, 0, None, None, None))
            bp = bp_vmap(sinogram_view, pixel_indices, view_params, projector_params, coeff_power)
            return bp

        def compute_hessian_diagonal(weights=None):
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
            if weights is None:
                weights = jnp.ones(sinogram_shape)
            elif weights.shape != sinogram_shape:
                error_message = 'Weights must be constant or an array of the same shape as sinogram'
                error_message += '\nGot weights.shape = {}, but sinogram.shape = {}'.format(weights.shape, sinogram_shape)
                raise ValueError(error_message)

            num_recon_rows, num_recon_cols, num_recon_slices = recon_shape[:3]
            max_index = num_recon_rows * num_recon_cols
            indices = jnp.arange(max_index)

            hessian_diagonal = self.sparse_back_project(weights, indices, coeff_power=2)

            return hessian_diagonal.reshape((num_recon_rows, num_recon_cols, num_recon_slices))

        # Set the compiled projectors and Hessian function
        projector_functions = (jax.jit(sparse_forward_project_fcn),
                               jax.jit(sparse_back_project_fcn, static_argnums=(2,)),
                               compute_hessian_diagonal)
        self.sparse_forward_project, self.sparse_back_project, self.compute_hessian_diagonal = projector_functions
