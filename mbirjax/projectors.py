from collections import namedtuple
import jax
import jax.numpy as jnp


class Projectors:

    def __init__(self, tomography_model):

        self.tomography_model = tomography_model
        self.sparse_forward_project, self.sparse_back_project, self.compute_hessian_diagonal = None, None, None
        self.create_projectors(tomography_model)

    def create_projectors(self, tomography_model):
        """
        Compute the forward and back projectors for this geometry and current view parameters

        Args:
            tomography_model (mbirjax.TomographyModel): An instance describing the current geometry and implementing the following 2 functions:

                * forward_project_pixel_batch_to_one_view (callable): jit-compilable function implementing :meth:`TomographyModel.forward_project_pixel_batch_to_one_view`
                * back_project_one_view_to_pixel_batch (callable): jit-compilable function implementing :meth:`TomographyModel.back_project_one_view_to_pixel_batch`

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
        forward_project_pixel_batch_to_one_view = tomography_model.forward_project_pixel_batch_to_one_view
        back_project_one_view_to_pixel_batch = tomography_model.back_project_one_view_to_pixel_batch

        geometry_params = self.tomography_model.get_geometry_parameters()
        sinogram_shape, recon_shape = self.tomography_model.get_params(['sinogram_shape', 'recon_shape'])

        # Combine the needed parameters into a named tuple for named access compatible with jit
        projector_param_names = ['sinogram_shape', 'recon_shape', 'geometry_params']
        projector_param_values = (sinogram_shape, recon_shape, geometry_params)
        ProjectorParams = namedtuple('ProjectorParams', projector_param_names)
        projector_params = ProjectorParams(*tuple(projector_param_values))

        view_params_array = self.tomography_model.get_params('view_params_array')
        pixel_batch_size, view_batch_size = self.tomography_model.get_params(['pixel_batch_size', 'view_batch_size'])

        def sparse_forward_project_fcn(voxel_values, pixel_indices, view_indices=()):
            """
            Compute the sinogram obtained by forward projecting the specified voxels. The voxel sare determined
            using 2D indices into a flattened array of shape (num_recon_rows, num_recon_cols),
            and for each such 2D index, the voxels in all slices at that location are projected.

            This function batches over pixels, applies sparse_forward_project_pixel_batch to each batch,
            then adds the results to get the sinogram.

            Args:
                voxel_values (ndarray or jax array): 2D array of shape (len(pixel_indices), num_slices) of voxel values
                pixel_indices (ndarray or jax array): 1D array of indices into a flattened array of shape (num_rows, num_cols)
                view_indices (ndarray or jax array, optional): 1D array of indices into the view parameters array.
                    If None, then all views are used.

            Returns:
                3D array of shape (num_views, num_det_rows, num_det_cols), where num_views is len(view_indices) if view_indices is not None
            """

            batch_size = pixel_batch_size
            function_to_sum = sparse_forward_project_pixel_batch
            data_to_batch = (voxel_values, pixel_indices)  # Apply ensure_tuple
            extra_args = (view_indices, )
            summed_output = sum_function_in_batches(function_to_sum, data_to_batch, batch_size, extra_args)
            return summed_output

        def sparse_forward_project_pixel_batch(voxel_values, pixel_indices, view_indices=()):
            """
            Compute the sinogram obtained by forward projecting the specified batch of voxels. The voxels
            are determined using 2D indices into a flattened array of shape (num_rows, num_cols),
            and for each such 2D index, the voxels in all slices at that location are projected.

            This function creates batches of views and collects the results to form the full sinogram.

            Args:
                voxel_values: 2D array of shape (len(pixel_indices), num_slices) of voxel values
                pixel_indices: 1D array of indices into a flattened array of shape (num_rows, num_cols)
                view_indices (ndarray or jax array, optional): 1D array of indices into the view parameters array.
                    If None, then all views are used.

            Returns:
                3D array of shape (num_views, num_det_rows, num_det_cols)
            """
            cur_view_params_array = view_params_array
            if len(view_indices) > 0:
                cur_view_params_array = view_params_array[view_indices]

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

            sinogram = concatenate_function_in_batches(forward_project_view_batch, cur_view_params_array, view_batch_size)

            return sinogram

        def sparse_back_project_fcn(sinogram, pixel_indices, coeff_power=1, view_indices=()):
            """
            Compute the voxel values obtained by back projecting the sinogram to the specified voxels. The voxels
            are determined using 2D indices into a flattened array of shape (num_recon_rows, num_recon_cols),
            and for each such 2D index, the voxels in all slices at that location are projected.

            This function batches over views, applies sparse_back_project_view_batch to each batch,
            then adds the results to get the voxel values.

            Args:
                sinogram (ndarray or jax array): 3D array of shape (cur_num_views, num_det_rows, num_det_cols), where
                    cur_num_views is recon_shape[0] if view_indices is () and len(view_indices)
                    otherwise, in which case the views in sinogram should match those indicated by view_indices.
                pixel_indices (ndarray or jax array): 1D array of indices into a flattened array of shape
                (num_recon_rows, num_recon_cols)
                coeff_power (int): backproject using the coefficients of (A_ij ** coeff_power).
                    Normally 1, but should be 2 when computing Hessian diagonal.
                view_indices (ndarray or jax array, optional): 1D array of indices into the view parameters array.
                    If None, then all views are used.

            Returns:
                2D array of shape (num_pixels, num_recon_slices)
            """

            cur_view_params_array = view_params_array
            if len(view_indices) > 0:
                cur_view_params_array = view_params_array[view_indices]

            batch_size = view_batch_size
            function_to_sum = sparse_back_project_view_batch
            data_to_batch = (sinogram, cur_view_params_array)  # Apply ensure_tuple
            extra_args = (pixel_indices, coeff_power, )
            summed_output = sum_function_in_batches(function_to_sum, data_to_batch, batch_size, extra_args)
            return summed_output

        def sparse_back_project_view_batch(view_batch, view_params_batch, pixel_indices, coeff_power):
            """
            This function creates batches of pixels and collects the results to form the full sinogram.
            Also, since the geometry-specific projectors map between a batch of voxel cylinders and a single view,
            we need to map over views and add the results to get the correct back projection for each voxel.

            Args:
                view_batch (ndarray or jax array): 3D array of shape (cur_num_views, num_det_rows, num_det_cols)
                view_params_batch (jax array): 1D or 2D array of parameters, view_params_batch[i] describes view_batch[i]
                pixel_indices (ndarray or jax array): 1D array of indices into a flattened array of shape
                (num_recon_rows, num_recon_cols)
                coeff_power (int): backproject using the coefficients of (A_ij ** coeff_power).
                    Normally 1, but should be 2 when computing Hessian diagonal.

            Returns:
                jax array of size (pixel_indices.shape[0], num_recon_slices)
            """
            def back_project_pixel_batch(pixel_indices_batch):
                """
                Apply back_project_one_view_to_pixel_batch to each pixel batch and each view
                Add over the views and concatenate over the pixels.

                Args:
                    pixel_indices_batch:

                Returns:
                    jax array of size (pixel_indices.shape[0], num_recon_slices)
                """
                #
                bp_vmap = jax.vmap(back_project_one_view_to_pixel_batch, in_axes=(0, None, 0, None, None))
                per_view_voxel_values_batch = bp_vmap(view_batch, pixel_indices_batch, view_params_batch,
                                                      projector_params, coeff_power)

                voxel_values_batch = jnp.sum(per_view_voxel_values_batch, axis=0)
                return voxel_values_batch

            new_voxel_values = concatenate_function_in_batches(back_project_pixel_batch, pixel_indices, pixel_batch_size)
            return new_voxel_values

        def compute_hessian_diagonal(weights=None, view_indices=()):
            """
            Computes the diagonal of the Hessian matrix, which is computed by doing a backprojection of the weight
            matrix except using the square of the coefficients in the backprojection to a given voxel.
            One of weights or sinogram_shape must be not None. If weights is not None, it must be an array with the same
            shape as the sinogram to be backprojected.  If weights is None, then a weights matrix will be computed as an
            array of ones of size sinogram_shape.

            Args:
               weights (ndarray or jax array or None, optional): 3D array of shape
                    (cur_num_views, num_det_rows, num_det_cols), where cur_num_views is recon_shape[0]
                    if view_indices is () and len(view_indices) otherwise, in which case the views in weights should
                    match those indicated by view_indices.  Defaults to all 1s.
               view_indices (ndarray or jax array, optional): 1D array of indices into the view parameters array.
                    If None, then all views are used.

            Returns:
                An array that is the same size as the reconstruction.
            """
            num_views = len(view_indices) if len(view_indices) != 0 else sinogram_shape[0]
            if weights is None:
                weights = jnp.ones((num_views,) + sinogram_shape[1:])
            elif weights.shape != (num_views,) + sinogram_shape[1:]:
                error_message = 'Weights must be constant or an array compatible with sinogram'
                error_message += '\nGot weights.shape = {}, but sinogram.shape = {}'.format(weights.shape, sinogram_shape)
                raise ValueError(error_message)

            num_recon_rows, num_recon_cols, num_recon_slices = recon_shape[:3]
            max_index = num_recon_rows * num_recon_cols
            indices = jnp.arange(max_index)

            hessian_diagonal = self.sparse_back_project(weights, indices, coeff_power=2, view_indices=view_indices)

            return hessian_diagonal.reshape((num_recon_rows, num_recon_cols, num_recon_slices))

        # Set the compiled projectors and Hessian function
        projector_functions = (jax.jit(sparse_forward_project_fcn),
                               jax.jit(sparse_back_project_fcn, static_argnames='coeff_power'),
                               compute_hessian_diagonal)
        self.sparse_forward_project, self.sparse_back_project, self.compute_hessian_diagonal = projector_functions


def concatenate_function_in_batches(function, data_to_batch, batch_size):
    """
    Apply a given function to a set of data, batching over the first index, concatenating the results along axis=0.
    The function should operate on subsets of the form data_to_batch[start:start+batch_size] when function takes a
    single input or on analogous subsets of each element data_to_batch[j] when function takes multiple
    inputs.  The output of function should be an array or tuple of arrays.  These output are concatenated along
    the leading axis.

    The shape of each array is determined by the output(s) of function, which
    is concatenated along the leading axis.  If function returns a fixed shape output, then the result has
    size given by the total number batches (full or partial) times the length of the leading axis of the output.

    Args:
        function (callable): A function to be mapped over batches of the input data.  This will be called on
        the unpacked elements of data_to_batch (after batching) using a call of the form
        output_data = function(batch) when data_to_batch is a single array or
        output_data = function(*batch) when data_to_batch is a tuple.
        data_to_batch (jax array or tuple of arrays): An array of data to be batched and sent to function. If a tuple,
        then each element should have the same size leading axis.
        batch_size (int): The maximum number of entries to process at one time.

    Returns:
        An array or tuple of arrays.
    """
    data_to_batch = ensure_tuple(data_to_batch)

    # Apply the batch projector directly to an initial batch
    num_input_points = data_to_batch[0].shape[0]
    batch_size = num_input_points if batch_size is None else batch_size
    num_remaining = num_input_points % batch_size

    # If the input is a multiple of batch_size, then we'll do a full batch, otherwise just the excess.
    initial_batch_size = batch_size if num_remaining == 0 else num_remaining

    initial_batch = [data[:initial_batch_size] for data in data_to_batch]
    output_data = function(*initial_batch)

    # Then deal with the batches if there are any
    if batch_size < num_input_points:
        def wrapped_function(arg_list):
            return function(*arg_list)

        num_batches = (num_input_points - initial_batch_size) // batch_size
        output_shape = (num_batches, batch_size,)
        data_batched = [jnp.reshape(data[initial_batch_size:], output_shape + data.shape[1:])
                        for data in data_to_batch]

        # Apply the function in batches
        output_data_batched = jax.lax.map(wrapped_function, data_batched)

        # The output data may be a single array or a tuple or list of arrays
        # First unbatch the data by reshaping the first 2 dims to be the number of points in all the batches.
        # Using tree_map, this can be done on either a single array or a tuple of arrays and get either a
        # single array or a tuple back
        output_unbatched = jax.tree_util.tree_map(unbatch, output_data_batched)

        # Now stack the first partial batch with this unbatched result
        output_data = jax.tree_util.tree_map(concatenate_arrays, *(output_data, output_unbatched))

    return output_data


def unbatch(array):
    """
    Reshape a jax array from (n0, n1, ...) to (n0*n1, ...)

    Args:
        array (jax array): array to be reshaped

    Returns:
        jax array
    """
    return jax.numpy.reshape(array, (array.shape[0] * array.shape[1],) + array.shape[2:])


def concatenate_arrays(*arrays):
    """
    Helper function to concatenate a list or tuple of arrays along the leading axis.

    Args:
        *arrays: list of arrays, with compatible dimensions arrays[j].shape[1:]

    Returns:
        array of shape (n, ) + arrays[0].shape[1:], where n is the sum over j of arrays[j].shape[0]
    """
    return jax.numpy.concatenate(arrays, axis=0)


def sum_function_in_batches(function_to_sum, data_to_batch, batch_size, extra_args=()):
    """
    Apply a given function to a set of data, batching over the first index, summing the results.
    The function should operate on subsets of the form data_to_batch[start:start+batch_size] when function takes a
    single input or on analogous subsets of each element data_to_batch[j] when function takes multiple
    inputs.  The output of function should be a scalar or fixed size array.

    Args:
        function_to_sum (callable): A function to be mapped over batches of the input data.  This will be called on
        the unpacked elements of data_to_batch (after batching) and extra_args using a call of the form
        summed_data += function_to_sum(batched_data, *fixed_data) when data_to_batch is a single array or
        summed_data += function_to_sum(*batched_data, *fixed_data) when data_to_batch is a tuple.
        data_to_batch (jax array or tuple of arrays): The data to be processed in batches.  If a tuple, then each element
        should have the same size leading axis.
        batch_size (int): The maximum batch size.
        extra_args (tuple): Any additional arguments needed by function_to_sum

    Returns:
        jax array or scalar output of function_to_sum, summed over all the elements in data_to_batch.
    """
    data_to_batch = ensure_tuple(data_to_batch)
    extra_args = ensure_tuple(extra_args)

    def add_one_batch(summed_and_fixed_data, batched_data):
        """
        Apply the externally defined function function_to_sum to the data in the tuple batched_data
        and add the result to an existing result.  The existing result is the first element in the tuple
        summed_and_fixed_data.  Any remaining elements of summed_and_fixed_data are for additional arguments
        to function_to_sum.  batched_data and fixed_data are unpacked before calling function_to sum. The
         primary functionality is summed_data += function_to_sum(*batched_data, *fixed_data)

        Args:
            summed_and_fixed_data (tuple or list): The first element is an array of the shape returned by
            function_to_sum.  This shape should not depend on batched_data.  The remaining elements are
            extra arguments to be sent to function_to_sum.
            batched_data (tuple or list):  The data for use in function_to_sum.

        Returns:
            tuple of ([summed_data, *fixed_data], None)
        """
        summed_data = summed_and_fixed_data[0]
        fixed_data = summed_and_fixed_data[1:]
        output_to_add = function_to_sum(*batched_data, *fixed_data)
        summed_data = jax.tree_util.tree_map(jnp.add, *(summed_data, output_to_add))

        return [summed_data, *fixed_data], None

    # Apply the batch projector directly to an initial batch to get the initial output
    num_input_points = data_to_batch[0].shape[0]
    batch_size = num_input_points if batch_size is None else batch_size
    num_remaining = num_input_points % batch_size
    # If the input is a multiple of batch_size, then we'll do a full batch, otherwise just the excess.
    initial_batch_size = batch_size if num_remaining == 0 else num_remaining

    initial_batch = [data[:initial_batch_size] for data in data_to_batch]
    summed_output = function_to_sum(*initial_batch, *extra_args)

    # Then deal with the batches if there are any
    if batch_size < num_input_points:
        num_batches = (num_input_points - initial_batch_size) // batch_size
        output_shape = (num_batches, batch_size,)
        data_batched = [jnp.reshape(data[initial_batch_size:], output_shape + data.shape[1:])
                        for data in data_to_batch]

        # Set up a scan over the batches.
        initial_carry = [summed_output, *extra_args]
        final_carry, _ = jax.lax.scan(add_one_batch, initial_carry, data_batched)

        summed_output = final_carry[0]

    return summed_output


def ensure_tuple(var_args):
    """
    Convert a singleton to a one-element tuple if needed, and convert a list to a tuple
    Args:
        var_args: singleton or list or tuple

    Returns:
        tuple
    """
    # Check if var_args is already a tuple
    if isinstance(var_args, tuple):
        return var_args
    # Check if var_args is a list
    elif isinstance(var_args, list):
        return tuple(var_args)
    # Assume var_args is a single item if it's neither a list nor a tuple
    else:
        return (var_args, )