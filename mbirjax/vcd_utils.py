import warnings
import numpy as np
import jax.numpy as jnp
import jax
import mbirjax.bn256 as bn
import mbirjax.preprocess as mjp


def stitch_arrays(array_list, overlap, axis=2):
    """
    Concatenate JAX arrays along one axis while linearly blending a fixed overlap
    between adjacent arrays.

    This behaves like `jnp.concatenate` except that for each adjacent pair, the
    first `overlap_length` elements of the second array and the last
    `overlap_length` elements of the current result are combined by a piece-wise linear cross‑fade.

    All non‑`axis` dimensions must match across inputs.

    Args:
        array_list (list[jax.Array]): Sequence of 2+ JAX arrays to stitch.
        overlap (int): Number of elements to blend between each adjacent pair.
            Must be `>= 1` and not exceed the length of any input along `axis`.
        axis (int, optional): Axis along which to stitch. Defaults to 2.

    Returns:
        jax.Array: Stitched array. Its shape equals the input shape with the
        length along `axis` equal to:

            sum(len_k) - (len(array_list) - 1) * overlap_length

        where `len_k` are the lengths of each input along `axis`.

    Raises:
        ValueError: If fewer than two arrays are provided, if non‑`axis`
            dimensions differ, or if any array is shorter than
            `overlap_length` along `axis`.

    Example:
        >>> import jax.numpy as jnp
        >>> a0 = jnp.arange(2*2*5).reshape(2, 2, 5)
        >>> a1 = jnp.arange(2*2*6).reshape(2, 2, 6)
        >>> out = stitch_arrays([a0, a1], overlap=3, axis=2)
        >>> out.shape
        (2, 2, 8)

        # 8 comes from 5 + 6 - 3 (one overlap between two arrays).
    """
    # Check for valid input
    if not isinstance(array_list, list) or len(array_list) < 2:
        raise ValueError('array_list must be a list of 2 or more jax arrays.')
    for dim in range(array_list[0].ndim):
        lengths = [array.shape[dim] for array in array_list]
        if dim != axis:
            if np.amax(lengths) != np.amin(lengths):
                raise ValueError('The shapes of the arrays in array_list must be the same except in the dimension specified by axis.')
        if dim == axis:
            if np.amin(lengths) < overlap:
                raise ValueError('Each array must have length at least overlap in the dimension specified by axis.')

    # Create a piecewise linear weight array:
    # 0 for first 25%, linear ramp 0→1 over middle 50%, 1 for final 25%.
    t = jnp.linspace(0, 1, overlap)
    weights = jnp.clip((t - 0.25) / 0.5, 0.0, 1.0)
    weights_shape = np.ones(array_list[0].ndim, dtype=int)
    weights_shape[0] = len(weights)
    weights = weights.reshape(weights_shape)

    # Start with the first array in the list
    stitched = jnp.swapaxes(array_list[0], 0, axis)

    # Iterate through each subsequent array in the list
    for next_array in array_list[1:]:
        # Extract the overlap from the current end of the stitched array and the beginning of the next array
        overlap_current = stitched[-overlap:]
        next_array = jnp.swapaxes(next_array, 0, axis)
        overlap_next = next_array[:overlap]

        # Weighted average for the overlapping part
        weighted_overlap = (1 - weights) * overlap_current + weights * overlap_next

        # Replace the overlap in the stitched array
        stitched = jnp.concatenate([stitched[:-overlap], weighted_overlap], axis=0)

        # Append the non-overlapping remainder of the next array
        stitched = jnp.concatenate([stitched, next_array[overlap:]], axis=0)

    return jnp.swapaxes(stitched, 0, axis)


def get_2d_ror_mask(recon_shape, *, crop_radius_pixels=0, crop_radius_fraction=0.0):
    """
    Get a binary mask for the region of reconstruction.  By default, the mask is the largest possible circle
    inscribed on the longest edge of the 2D recon_shape[0:2].  The radius of this circle can be reduced by
    setting crop_radius_pixels or crop_radius_fraction, either of which is subtracted from the radius.  Only one
    of these can be nonzero. Negative values are clipped to 0.

    Args:
        recon_shape (tuple): Shape of recon in (rows, columns, slices)
        crop_radius_pixels (int): Number of pixels to subtract from the radius before creating the mask.
        crop_radius_fraction (float): Fraction to subtract from the radius before creating the mask.

    Returns:
        A binary mask for the region of reconstruction.
    """
    # Set up a mask to zero out points outside the ROR
    if crop_radius_pixels != 0 and crop_radius_fraction != 0.0:
        raise ValueError('Only one of crop_radius_pixels and crop_radius_fraction can be nonzero.')

    num_recon_rows, num_recon_cols = recon_shape[:2]
    row_center = (num_recon_rows - 1) / 2
    col_center = (num_recon_cols - 1) / 2

    radius = max(row_center, col_center)

    crop_radius = int(radius * crop_radius_fraction)
    crop_radius = max(crop_radius, crop_radius_pixels)
    crop_radius = max(crop_radius, 0)
    radius -= crop_radius

    col_coords = np.arange(num_recon_cols) - col_center
    row_coords = np.arange(num_recon_rows) - row_center

    coords = np.meshgrid(col_coords, row_coords)  # Note the interchange of rows and columns for meshgrid
    mask = coords[0]**2 + coords[1]**2 <= radius**2
    mask = mask[:, :]
    return mask


def gen_set_of_pixel_partitions(recon_shape, granularity, output_device=None, use_ror_mask=True):
    """
    Generates a collection of voxel partitions for an array of specified partition sizes.
    This function creates a tuple of randomly generated 2D voxel partitions.

    Args:
        recon_shape (tuple): Shape of recon in (rows, columns, slices)
        granularity (list or tuple):  List of num_subsets to use for each partition
        output_device (jax device): Device on which to place the output of the partition
        use_ror_mask (bool): Flag to indicate whether to mask out a circular RoR

    Returns:
        tuple: A tuple of 2D arrays each representing a partition of voxels into the specified number of subsets.
    """
    partitions = []
    for num_subsets in granularity:
        partition = gen_pixel_partition(recon_shape, num_subsets, use_ror_mask=use_ror_mask)
        partitions += [jax.device_put(partition, output_device),]

    return partitions


def gen_pixel_partition_grid(recon_shape, num_subsets, use_ror_mask=True):

    small_tile_side = np.ceil(np.sqrt(num_subsets)).astype(int)
    num_subsets = small_tile_side ** 2
    num_small_tiles = [np.ceil(recon_shape[k] / small_tile_side).astype(int) for k in [0, 1]]

    single_subset_inds = np.random.permutation(num_subsets).reshape((small_tile_side, small_tile_side))
    subset_inds = np.tile(single_subset_inds, num_small_tiles)
    subset_inds = subset_inds[:recon_shape[0], :recon_shape[1]]

    ror_mask = get_2d_ror_mask(recon_shape[:2]) if use_ror_mask else 1
    subset_inds = (subset_inds + 1) * ror_mask - 1  # Get a - at each location outside the mask, subset_ind at other points
    subset_inds = subset_inds.flatten()
    num_inds = len(np.where(subset_inds > -1)[0])

    if num_subsets > num_inds:
        # num_subsets = len(indices)
        warning = '\nThe number of partition subsets is greater than the number of pixels in the region of '
        warning += 'reconstruction.  \nReducing the number of subsets to equal the number of indices.'
        warnings.warn(warning)
        subset_inds = subset_inds[subset_inds > -1]
        return jnp.array(subset_inds).reshape((-1, 1))

    flat_inds = []
    max_points = 0
    min_points = subset_inds.size
    nonempty_subsets = np.unique(subset_inds[subset_inds>=0])
    for k in nonempty_subsets:
        cur_inds = np.where(subset_inds == k)[0]
        flat_inds.append(cur_inds)  # Get all the indices for each subset
        max_points = max(max_points, cur_inds.size)
        min_points = min(min_points, cur_inds.size)

    extra_point_inds = np.random.randint(min_points, size=(max_points - min_points + 1,))
    for k in range(len(nonempty_subsets)):
        cur_inds = flat_inds[k]
        num_extra_points = max_points - cur_inds.size
        if num_extra_points > 0:
            extra_subset_inds = (k + 1 + np.arange(num_extra_points, dtype=int)) % len(nonempty_subsets)
            new_point_inds = [flat_inds[extra_subset_inds[j]][extra_point_inds[j]] for j in range(num_extra_points)]
            flat_inds[k] = np.concatenate((cur_inds, new_point_inds))
    flat_inds = np.array(flat_inds)

    # Reorganize into subsets, then sort each subset
    indices = jnp.array(flat_inds)

    return jnp.array(indices)


def gen_pixel_partition(recon_shape, num_subsets, use_ror_mask=True):
    """
    Generates a partition of pixel indices into specified number of subsets for use in tomographic reconstruction algorithms.
    The function ensures that each subset contains an equal number of pixels, suitable for VCD reconstruction.

    Args:
        recon_shape (tuple): Shape of recon in (rows, columns, slices)
        num_subsets (int): The number of subsets to divide the pixel indices into.
        use_ror_mask (bool): Flag to indicate whether to mask out a circular RoR

    Raises:
        ValueError: If the number of subsets specified is greater than the total number of pixels in the grid.

    Returns:
        jnp.array: A JAX array where each row corresponds to a subset of pixel indices, sorted within each subset.
    """
    # Determine the 2D indices within the RoR
    num_recon_rows, num_recon_cols = recon_shape[:2]
    max_index_val = num_recon_rows * num_recon_cols
    indices = np.arange(max_index_val, dtype=np.int32)

    # Mask off indices that are outside the region of reconstruction
    if use_ror_mask:
        mask = get_2d_ror_mask(recon_shape)
        mask = mask.flatten()
        indices = indices[mask == 1]
    if num_subsets > len(indices):
        num_subsets = len(indices)
        warning = '\nThe number of partition subsets is greater than the number of pixels in the region of '
        warning += 'reconstruction.  \nReducing the number of subsets to equal the number of indices.'
        warnings.warn(warning)

    # Determine the number of indices to repeat to make the total number divisible by num_subsets
    num_indices_per_subset = int(np.ceil((len(indices) / num_subsets)))
    array_size = num_subsets * num_indices_per_subset
    num_extra_indices = array_size - len(indices)  # Note that this is >=0 since otherwise the ValueError is raised
    indices = np.random.permutation(indices)

    # Enlarge the array to the desired length by adding random indices that are not in the final subset
    num_non_final_indices = (num_subsets - 1) * num_indices_per_subset
    extra_indices = np.random.choice(indices[:num_non_final_indices], size=num_extra_indices, replace=False)
    indices = np.concatenate((indices, extra_indices))

    # Reorganize into subsets, then sort each subset
    indices = indices.reshape(num_subsets, indices.size // num_subsets)
    indices = jnp.sort(indices, axis=1)

    return jnp.array(indices)


def gen_pixel_partition_blue_noise(recon_shape, num_subsets, use_ror_mask=True):
    """
    Generates a partition of pixel indices into specified number of subsets for use in tomographic reconstruction algorithms.
    The function ensures that each subset contains an equal number of pixels, suitable for VCD reconstruction.

    Args:
        recon_shape (tuple): Shape of recon in (rows, columns, slices)
        num_subsets (int): The number of subsets to divide the pixel indices into.
        use_ror_mask (bool): Flag to indicate whether to mask out a circular RoR

    Raises:
        ValueError: If the number of subsets specified is greater than the total number of pixels in the grid.

    Returns:
        jnp.array: A JAX array where each row corresponds to a subset of pixel indices, sorted within each subset.
    """
    pattern = bn.bn256
    num_tiles = [np.ceil(recon_shape[k] / pattern.shape[k]).astype(int) for k in [0, 1]]
    ror_mask = get_2d_ror_mask(recon_shape) if use_ror_mask else 1

    single_subset_inds = np.floor(pattern / (2**16 / num_subsets)).astype(int)

    # Repeat each bn subset to do the tiling
    subset_inds = np.tile(single_subset_inds, num_tiles)
    subset_inds = subset_inds[:recon_shape[0], :recon_shape[1]]
    subset_inds = (subset_inds + 1) * ror_mask - 1  # Get a -1 at each location outside the mask, subset_ind at other points
    subset_inds = subset_inds.flatten()
    num_valid_inds = np.sum(subset_inds >= 0)
    if num_subsets > num_valid_inds:
        return gen_pixel_partition(recon_shape, num_subsets)

    flat_inds = []
    max_points = 0
    min_points = subset_inds.size
    for k in range(num_subsets):
        cur_inds = np.where(subset_inds == k)[0]
        flat_inds.append(cur_inds)  # Get all the indices for each subset
        max_points = max(max_points, cur_inds.size)
        min_points = min(min_points, cur_inds.size)

    if min_points == 0:
        return gen_pixel_partition(recon_shape, num_subsets)

    extra_point_inds = np.random.randint(low=0, high=min_points, size=(max_points - min_points + 1,))

    for k in range(num_subsets):
        cur_inds = flat_inds[k]
        num_extra_points = max_points - cur_inds.size
        if num_extra_points > 0:
            extra_subset_inds = (k + 1 + np.arange(num_extra_points, dtype=int)) % num_subsets
            new_point_inds = [flat_inds[extra_subset_inds[j]][extra_point_inds[j]] for j in range(num_extra_points)]
            flat_inds[k] = np.concatenate((cur_inds, new_point_inds))
    flat_inds = jnp.array(flat_inds)
    flat_inds = jnp.sort(flat_inds, axis=1)

    return flat_inds


def gen_partition_sequence(partition_sequence, max_iterations):
    """
    Generates a sequence of voxel partitions of the specified length by extending the sequence
    with the last element if necessary.
    """
    # Get sequence from params and convert it to a np array
    partition_sequence = jnp.array(partition_sequence)

    # Check if the sequence needs to be extended
    current_length = partition_sequence.size
    if max_iterations > current_length:
        # Calculate the number of additional elements needed
        extra_elements_needed = max_iterations - current_length
        # Get the last element of the array
        last_element = partition_sequence[-1]
        # Create an array of the last element repeated the necessary number of times
        extension_array = np.full(extra_elements_needed, last_element)
        # Concatenate the original array with the extension array
        extended_partition_sequence = np.concatenate((partition_sequence, extension_array))
    else:
        # If no extension is needed, slice the original array to the desired length
        extended_partition_sequence = partition_sequence[:max_iterations]

    return extended_partition_sequence


def gen_full_indices(recon_shape, use_ror_mask=True):
    """
    Generates a full array of voxels in the region of reconstruction.
    This is useful for computing forward projections.
    """
    partition = gen_pixel_partition(recon_shape, num_subsets=1, use_ror_mask=use_ror_mask)
    full_indices = partition[0]

    return full_indices


def gen_weights_mar(ct_model, sinogram, init_recon=None, metal_threshold=None, beta=1.0, gamma=3.0):
    """
    Generates the weights used for reducing metal artifacts in MBIR reconstruction.

    This function computes sinogram weights that help to reduce metal artifacts.
    More specifically, it computes weights with the form:

        weights = exp( -(sinogram/beta) * ( 1 + gamma * delta(metal) ) )

    delta(metal) denotes a binary mask indicating the sino entries that contain projections of metal.
    Providing ``init_recon`` yields better metal artifact reduction.
    If not provided, the metal segmentation is generated directly from the sinogram.

    Args:
        sinogram (jax array): 3D jax array containing sinogram with shape (num_views, num_det_rows, num_det_channels).
        init_recon (jax array, optional): An initial reconstruction used to identify metal voxels. If not provided, Otsu's method is used to directly segment sinogram into metal regions.
        metal_threshold (float, optional): Values in ``init_recon`` above ``metal_threshold`` are classified as metal. If not provided, Otsu's method is used to segment ``init_recon``.
        beta (float, optional): Scalar value in range :math:`>0`.
            A larger ``beta`` improves the noise uniformity, but too large a value may increase the overall noise level.
        gamma (float, optional): Scalar value in range :math:`>=0`.
            A larger ``gamma`` reduces the weight of sinogram entries with metal, but too large a value may reduce image quality inside the metal regions.

    Returns:
        (jax array): Weights used in mbircone reconstruction, with the same array shape as ``sinogram``
    """
    # If init_recon is not provided, then identify the distorted sino entries with Otsu's thresholding method.
    if init_recon is None:
        print("init_recon is not provided. Automatically determine distorted sinogram entries with Otsu's method.")
        # assuming three categories: metal, non_metal, and background.
        [bk_thresh_sino, metal_thresh_sino] = mjp.multi_threshold_otsu(sinogram, classes=3)
        print("Distorted sinogram threshold = ", metal_thresh_sino)
        delta_metal = jnp.array(sinogram > metal_thresh_sino, dtype=jnp.dtype(jnp.float32), device=ct_model.main_device)

    # If init_recon is provided, identify the distorted sino entries by forward projecting init_recon.
    else:
        if metal_threshold is None:
            print("Metal_threshold calculated with Otsu's method.")
            # assuming three categories: metal, non_metal, and background.
            [bk_threshold, metal_threshold] = mjp.multi_threshold_otsu(init_recon, classes=3)

        print("metal_threshold = ", metal_threshold)
        # Identify metal voxels
        metal_mask = jnp.array(init_recon > metal_threshold, dtype=jnp.dtype(jnp.float32), device=ct_model.main_device)
        # Forward project metal mask to generate a sinogram mask
        metal_mask_projected = ct_model.forward_project(metal_mask)

        # metal mask in the sinogram domain, where 1 means a distorted sino entry, and 0 else.
        delta_metal = jnp.array(metal_mask_projected > 0.0, dtype=jnp.dtype(jnp.float32), device=ct_model.main_device)

    # weights for undistorted sino entries
    weights = jnp.exp(-sinogram*(1+gamma*delta_metal)/beta)

    return weights


def gen_weights(sinogram, weight_type):
    """
    Compute optional weights used in MBIR reconstruction based on the noise model.

    The weights should be proportional to the inverse variance of the noise for each sinogram entry.
    They can be used to improve reconstruction quality.

    Args:
        sinogram (jax.Array): A 3D JAX array of shape (num_views, num_det_rows, num_det_channels)
            representing the sinogram.
        weight_type (str): The type of noise model to use for weighting. Must be one of:
            - 'unweighted': Use uniform weights (all ones).
            - 'transmission': Use exponential decay, `exp(-sinogram)`.
            - 'transmission_root': Use square-root decay, `exp(-sinogram / 2)`.
            - 'emission': Use reciprocal decay, `1 / (abs(sinogram) + 0.1)`.

    Returns:
        jax.Array: A 3D array of weights with the same shape as the input sinogram.

    Raises:
        Exception: If `weight_type` is not one of the supported options.

    Note:
        For transmission noise models, sinogram values should not be excessively large (e.g., > 5),
        as this corresponds to near-zero transmission, which is not physically meaningful in typical X-ray imaging.

    Example:
        >>> sinogram = jnp.ones((180, 64, 128))
        >>> weights = gen_weights(sinogram, weight_type='transmission_root')
        >>> weights.shape
        (180, 64, 128)
    """
    weight_list = []
    num_views = sinogram.shape[0]
    batch_size = 128
    main_device = jax.devices('cpu')[0]
    try:
        gpus = jax.devices('gpu')
        worker_device = gpus[0]
    except RuntimeError:
        worker_device = main_device

    for i in range(0, num_views, batch_size):
        sino_batch = jax.device_put(sinogram[i:min(i + batch_size, num_views)], worker_device)

        if weight_type == 'unweighted':
            weights = jnp.ones(sino_batch.shape)
        elif weight_type == 'transmission':
            weights = jnp.exp(-sino_batch)
        elif weight_type == 'transmission_root':
            weights = jnp.exp(-sino_batch / 2)
        elif weight_type == 'emission':
            weights = 1.0 / (jnp.absolute(sino_batch) + 0.1)
        else:
            raise Exception("gen_weights: undefined weight_type {}".format(weight_type))
        weight_list.append(jax.device_put(weights, main_device))

    weights = jnp.concatenate(weight_list, axis=0)
    return weights
