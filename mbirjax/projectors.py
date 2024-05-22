import warnings
from functools import partial
import jax
import jax.numpy as jnp


class Projectors:

    def __init__(self, tomography_model, forward_core, backward_core):

        self.tomography_model = tomography_model
        self.sparse_forward_project, self.sparse_back_project, self.compute_hessian_diagonal = None, None, None
        self.create_projectors(forward_core, backward_core)

    def create_projectors(self, forward_project_pixel_batch_to_one_view, back_project_one_view_to_pixel_batch):
        """
        Compute the forward and back projectors for this geometry and current view parameters
        
        Args:
            forward_project_pixel_batch_to_one_view (callable): jit-compilable function implementing
                :meth:`TomographyModel.forward_project_pixel_batch_to_one_view`
            back_project_one_view_to_pixel_batch (callable): jit-compilable function implementing
                :meth:`TomographyModel.back_project_one_view_to_pixel_batch`

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
            :meth:`TomographyModel.forward_project_pixel_batch_to_one_view` and
            :meth:`TomographyModel.back_project_one_view_to_pixel_batch`.
        """
        geometry_params = self.tomography_model.get_geometry_parameters()
        sinogram_shape, recon_shape = self.tomography_model.get_params(['sinogram_shape', 'recon_shape'])
        projector_params = (tuple(sinogram_shape), tuple(recon_shape), tuple(geometry_params))
        view_params_array = self.tomography_model.get_params('view_params_array')
        pixel_batch_size, view_batch_size = self.tomography_model.get_params(['pixel_batch_size', 'view_batch_size'])

        def sparse_forward_project_fcn(voxel_values, pixel_indices):
            """
            Compute the sinogram obtained by forward projecting the specified voxels. The voxel sare determined
            using 2D indices into a flattened array of shape (num_recon_rows, num_recon_cols),
            and for each such 2D index, the voxels in all slices at that location are projected.

            This function batches over pixels, applies sparse_forward_project_pixel_batch to each batch,
            then adds the results to get the sinogram.

            Args:
                voxel_values (ndarray or jax array): 2D array of shape (len(pixel_indices), num_slices) of voxel values
                pixel_indices (ndarray or jax array): 1D array of indices into a flattened array of shape (num_rows, num_cols)

            Returns:
                3D array of shape (num_views, num_det_rows, num_det_cols)
            """
            num_pixels = pixel_indices.shape[0]
            # Apply the batch projector directly to a batch if the batch is small enough.
            if pixel_batch_size is None or pixel_batch_size >= num_pixels:
                sinogram = sparse_forward_project_pixel_batch(voxel_values, pixel_indices)
            # Otherwise subdivide into batches, apply the batch projector, and then add.
            else:
                num_batches = num_pixels // pixel_batch_size
                length_of_batches = num_batches * pixel_batch_size
                voxel_values_batched = jnp.reshape(voxel_values[:length_of_batches],
                                                    (num_batches, pixel_batch_size,) + voxel_values.shape[1:])
                pixel_indices_batched = jnp.reshape(pixel_indices[:length_of_batches], (num_batches, pixel_batch_size))

                # Set up a scan over the voxel batches.  We'll project one batch to a sinogram,
                # then add that to the accumulated sinogram
                initial_sinogram = jnp.zeros(sinogram_shape)
                initial_carry = [initial_sinogram]
                values_indices = (voxel_values_batched, pixel_indices_batched)
                final_carry, _ = jax.lax.scan(forward_project_accumulate, initial_carry, values_indices)

                # Get the sinogram from these batches, and add in any leftover voxels
                sinogram = final_carry[0]
                num_remaining = num_pixels - num_batches * pixel_batch_size
                if num_remaining > 0:
                    end_batch_values = voxel_values[-num_remaining:]
                    end_batch_indices = pixel_indices[-num_remaining:]
                    sinogram += sparse_forward_project_pixel_batch(end_batch_values, end_batch_indices)

            return sinogram

        def forward_project_accumulate(carry, values_indices_batch):
            """

            Args:
                carry:
                values_indices_batch:

            Returns:

            """
            cur_sino = carry[0]
            voxel_values, pixel_indices = values_indices_batch
            cur_sino += sparse_forward_project_pixel_batch(voxel_values, pixel_indices)

            return [cur_sino], None

        def sparse_forward_project_pixel_batch(voxel_values, pixel_indices):
            """
            Compute the sinogram obtained by forward projecting the specified batch of voxels. The voxels
            are determined using 2D indices into a flattened array of shape (num_rows, num_cols),
            and for each such 2D index, the voxels in all slices at that location are projected.

            This function creates batches of views and collects the results to form the full sinogram.

            Args:
                voxel_values: 2D array of shape (len(pixel_indices), num_slices) of voxel values
                pixel_indices: 1D array of indices into a flattened array of shape (num_rows, num_cols)

            Returns:
                3D array of shape (num_views, num_det_rows, num_det_cols)
            """
            num_views = view_params_array.shape[0]

            def forward_project_single_view(single_view_params):
                # Use closure to define a mappable function that operates on a single view with the given voxel values.
                return forward_project_pixel_batch_to_one_view(voxel_values, pixel_indices,
                                                               single_view_params, projector_params)

            def forward_project_view_batch(view_params_batch):
                # Map the single view function over a batch of views.
                # To parallelize over views, we can use jax.vmap here instead of jax.lax.map, but this may use extra
                # memory since the voxel values are required for each view.

                sino_view_batch = jax.vmap(forward_project_single_view)(view_params_batch)

                return sino_view_batch

            # Apply the function on a single batch of views if the batch is small enough.
            if view_batch_size is None or view_batch_size >= num_views:
                sinogram = forward_project_view_batch(view_params_array)
            # Otherwise break the views up into batches and apply the function to each batch use another level of map.
            else:
                num_batches = num_views // view_batch_size
                view_params_batched = jnp.reshape(view_params_array[0:num_batches * view_batch_size],
                                                  (num_batches, view_batch_size, -1))

                sinogram = jax.lax.map(forward_project_view_batch, view_params_batched)
                sinogram = jnp.reshape(sinogram, (num_batches * view_batch_size,) + sinogram.shape[2:])
                num_remaining = num_views - num_batches * view_batch_size
                if num_remaining > 0:
                    end_batch = view_params_array[-num_remaining:]
                    end_views = forward_project_view_batch(end_batch)
                    sinogram = jnp.concatenate((sinogram, end_views), axis=0)

            return sinogram

        def sparse_back_project_fcn_new(sinogram, pixel_indices, coeff_power=1):
            """
            Compute the voxel values obtained by back projecting the sinogram to the specified voxels. The voxels
            are determined using 2D indices into a flattened array of shape (num_recon_rows, num_recon_cols),
            and for each such 2D index, the voxels in all slices at that location are projected.

            This function batches over views, applies sparse_back_project_view_batch to each batch,
            then adds the results to get the voxel values.

            Args:
                sinogram (ndarray or jax array): 3D array of shape (num_views, num_det_rows, num_det_cols)
                pixel_indices (ndarray or jax array): 1D array of indices into a flattened array of shape
                (num_recon_rows, num_recon_cols)
                coeff_power (int): backproject using the coefficients of (A_ij ** coeff_power).
                Normally 1, but should be 2 when computing Hessian diagonal.

            Returns:
                2D array of shape (num_pixels, num_recon_slices)
            """
            num_views = sinogram.shape[0]
            num_pixels = pixel_indices.shape[0]
            num_recon_slices = recon_shape[2]
            # Apply the batch projector directly to a batch if the batch is small enough.
            if view_batch_size is None or view_batch_size >= num_views:
                voxel_values = sparse_back_project_view_batch(sinogram, view_params_array, pixel_indices, coeff_power)
            # Otherwise subdivide into batches, apply the batch projector, and then add.
            else:
                num_batches = num_views // view_batch_size
                length_of_batches = num_batches * view_batch_size
                sinogram_batched = jnp.reshape(sinogram[:length_of_batches],
                                               (num_batches, view_batch_size,) + sinogram.shape[1:])
                view_params_batched = jnp.reshape(view_params_array[:length_of_batches],
                                                  (num_batches, view_batch_size,) + view_params_array.shape[1:])

                # Set up a scan over the view batches.  We'll project one batch to voxels,
                # then add that to the accumulated voxel_values
                initial_voxel_values = jnp.zeros((num_pixels, num_recon_slices))
                initial_carry = [initial_voxel_values, pixel_indices, coeff_power]
                sino_and_params = (sinogram_batched, view_params_batched)
                final_carry, _ = jax.lax.scan(back_project_accumulate, initial_carry, sino_and_params)

                # Get the voxel values from these batches and add in any leftover views
                voxel_values = final_carry[0]
                num_remaining = num_views - num_batches * view_batch_size
                if num_remaining > 0:
                    end_batch_views = sinogram[-num_remaining:]
                    end_batch_params = view_params_array[-num_remaining:]
                    voxel_values += sparse_back_project_view_batch(end_batch_views, end_batch_params,
                                                                   pixel_indices, coeff_power)

            return voxel_values

        def back_project_accumulate(carry, view_and_params_batch):
            cur_voxel_values, local_pixel_indices, local_coeff_power = carry
            view_batch, view_params_batch = view_and_params_batch
            cur_voxel_values += sparse_back_project_view_batch(view_batch, view_params_batch, local_pixel_indices, local_coeff_power)

            return [cur_voxel_values, local_pixel_indices, local_coeff_power], None

        def sparse_back_project_view_batch(view_batch, view_params_batch, pixel_indices, coeff_power):
            """
            This function creates batches of pixels and collects the results to form the full sinogram.
            Also, since the geometry-specific projectors map between a batch of voxel cylinders and a single view,
            we need to map over views and add the results to get the correct back projection for each voxel.

            Args:
                view_batch:
                view_params_batch:
                pixel_indices:
                coeff_power:

            Returns:

            """
            num_pixels = pixel_indices.shape[0]

            def back_project_pixel_batch(pixel_indices_batch):
                # Apply back_project_one_view_to_pixel_batch to each pixel batch and each view
                # Add over the views and concatenate over the pixels
                bp_vmap = jax.vmap(back_project_one_view_to_pixel_batch, in_axes=(0, None, 0, None, None))
                per_view_voxel_values_batch = bp_vmap(view_batch, pixel_indices_batch, view_params_batch,
                                                      projector_params, coeff_power)

                voxel_values_batch = jnp.sum(per_view_voxel_values_batch, axis=0)
                return voxel_values_batch

            # Apply the function on a single batch of pixels if the batch is small enough
            if pixel_batch_size is None or pixel_batch_size >= num_pixels:

                new_voxel_values = back_project_pixel_batch(pixel_indices)

            # Otherwise batch the pixels and map over the batches.
            else:
                num_batches = num_pixels // pixel_batch_size
                length_of_batches = num_batches * pixel_batch_size
                pixel_indices_batched = jnp.reshape(pixel_indices[:length_of_batches], (num_batches, pixel_batch_size))

                batched_voxel_values = jax.lax.map(back_project_pixel_batch, pixel_indices_batched)
                new_voxel_values = batched_voxel_values.reshape((length_of_batches,) + batched_voxel_values.shape[2:])

                # Add in any leftover pixels
                num_remaining = num_pixels - num_batches * pixel_batch_size
                if num_remaining > 0:
                    end_batch_indices = pixel_indices[-num_remaining:]
                    end_batch_voxel_values = back_project_pixel_batch(end_batch_indices)
                    new_voxel_values = jnp.concatenate((new_voxel_values, end_batch_voxel_values), axis=0)

            return new_voxel_values

        def sparse_back_project_fcn(sinogram, pixel_indices, coeff_power=1):
            """
            Compute the voxel values obtained by back projecting the sinogram to the specified voxels. The voxels
            are determined using 2D indices into a flattened array of shape (num_recon_rows, num_recon_cols),
            and for each such 2D index, the voxels in all slices at that location are projected.

            Args:
                sinogram (ndarray or jax array): 3D array of shape (num_views, num_det_rows, num_det_cols)
                pixel_indices (ndarray or jax array): 1D array of indices into a flattened array of shape
                (num_recon_rows, num_recon_cols)

            Returns:
                2D array of shape (num_pixels, num_recon_slices)
            """
            new_voxel_values = sparse_back_project_fcn_new(sinogram, pixel_indices, coeff_power)
            return new_voxel_values

            num_pixels = len(pixel_indices)
            if pixel_batch_size is None or pixel_batch_size >= num_pixels:
                voxel_values = back_project_to_pixels_scan(sinogram, pixel_indices, coeff_power=coeff_power)
            else:
                num_batches = num_pixels // pixel_batch_size
                length_of_batches = num_batches * pixel_batch_size
                indices_batched = jnp.reshape(pixel_indices[:length_of_batches], (num_batches, pixel_batch_size))

                def backward_map(indices_batch):
                    return back_project_to_pixels_scan(sinogram, indices_batch, coeff_power=coeff_power)

                voxel_values = jax.lax.map(backward_map, indices_batched)
                voxel_values = voxel_values.reshape((length_of_batches, -1))
                num_remaining = num_pixels - num_batches * pixel_batch_size
                if num_remaining > 0:
                    end_batch = pixel_indices[-num_remaining:]
                    end_values = backward_map(end_batch)
                    voxel_values = jnp.concatenate((voxel_values, end_values), axis=0)
            return voxel_values

        def back_project_to_pixels_scan(sinogram, pixel_indices, coeff_power=1):
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
            num_recon_slices = recon_shape[2]
            initial_bp = jnp.zeros((pixel_indices.shape[0], num_recon_slices))
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
            bp_view = back_project_one_view_to_pixel_batch(sinogram_view, pixel_indices, view_params, projector_params, coeff_power)
            accumulated += bp_view
            del bp_view
            return [extra_args, accumulated], None

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
                               jax.jit(sparse_back_project_fcn, static_argnames='coeff_power'),
                               compute_hessian_diagonal)
        self.sparse_forward_project, self.sparse_back_project, self.compute_hessian_diagonal = projector_functions
