import warnings
import numpy as np
import jax.numpy as jnp
import jax
import mbirjax.bn256 as bn
import mbirjax.preprocess as mjp

def generate_2d_ror_mask(recon_shape, *, delta_voxel=1.0, voxel_row_aspect=1.0, radius_alu=None, crop_radius_pixels=0, crop_radius_fraction=0.0):
    """
    Generate a binary mask for the region of reconstruction.
    By default, the mask is the largest possible centered circle in ALU space that fits inside recon_shape[:2], unless radius_alu is provided.
    The radius can be reduced by setting crop_radius_pixels or crop_radius_fraction, either of which is subtracted from the radius.  Only one of
    these can be nonzero. Negative values are clipped to 0.
    
    Args:
        recon_shape (tuple): Shape of recon in (rows, columns, slices), or just (rows, columns).
        delta_voxel (float): Column voxel pitch in ALU.
        voxel_row_aspect (float): Physical row voxel size relative to column voxel size.
        radius_alu (float | None): Radius of mask in ALU. If None, use largest inscribed radius.
        crop_radius_pixels (int): Number of column-pixel-equivalent pixels to subtract from radius.
        crop_radius_fraction (float): Fraction to subtract from radius.

    Returns:
        np.ndarray: Boolean 2D binary mask.
    """
    # Set up a mask to zero out points outside the ROR
    if crop_radius_pixels != 0 and crop_radius_fraction != 0.0:
        raise ValueError("Only one of crop_radius_pixels and crop_radius_fraction can be nonzero.")

    if delta_voxel <= 0:
        raise ValueError("delta_voxel must be positive.")

    if voxel_row_aspect <= 0:
        raise ValueError("voxel_row_aspect must be positive.")

    num_recon_rows, num_recon_cols = recon_shape[:2]
    row_center = (num_recon_rows - 1) / 2
    col_center = (num_recon_cols - 1) / 2

    col_coords_alu = (np.arange(num_recon_cols) - col_center) * delta_voxel
    row_coords_alu = (np.arange(num_recon_rows) - row_center) * voxel_row_aspect * delta_voxel

    col_grid_alu, row_grid_alu = np.meshgrid(col_coords_alu, row_coords_alu)

    if radius_alu is None:
        auto_num_recon_rows = int(np.round(num_recon_cols / voxel_row_aspect))

        col_radius_pixels = (num_recon_cols - 1) / 2
        row_radius_pixels = (auto_num_recon_rows - 1) / 2

        col_radius_alu = col_radius_pixels * delta_voxel
        row_radius_alu = row_radius_pixels * voxel_row_aspect * delta_voxel
    else:
        if radius_alu <= 0:
            raise ValueError("radius_alu must be positive.")
        col_radius_alu = float(radius_alu)
        row_radius_alu = float(radius_alu)

    if crop_radius_fraction != 0.0:
        col_radius_alu *= max(1.0 - crop_radius_fraction, 0.0)
        row_radius_alu *= max(1.0 - crop_radius_fraction, 0.0)
    else:
        crop_radius_alu = max(crop_radius_pixels, 0) * delta_voxel
        col_radius_alu -= crop_radius_alu
        row_radius_alu -= crop_radius_alu

    mask = (col_grid_alu / col_radius_alu) ** 2 + (row_grid_alu / row_radius_alu) ** 2 <= 1.0
    mask = mask[:, :]
    print(np.sum(mask))
    return mask

def get_2d_ror_mask(recon_shape, *, delta_voxel=1.0, voxel_row_aspect=1.0, ror_mask_option="auto", crop_radius_pixels=0, crop_radius_fraction=0.0):
    """
    Resolve the selected binary mask for the region of reconstruction.

    Args:
        recon_shape (tuple): Shape of recon in (rows, columns, slices), or just (rows, columns).
        delta_voxel (float): Column voxel pitch in ALU.
        voxel_row_aspect (float): Physical row voxel size relative to column voxel size.
        ror_mask_option (default is 'auto'):
            None:
                No mask.
            "auto":
                The mask is the largest possible centered circle in ALU space that fits inside recon_shape[:2].
            float:
                The mask is a circle in ALU space with radius = ror_mask_option
            2D array:
                Use a custom binary mask. Must have shape recon_shape[:2].
        crop_radius_pixels (int): Number of column-pixel-equivalent pixels to subtract from radius.
        crop_radius_fraction (float): Fraction to subtract from radius.
    
    Returns:
        np.ndarray: Boolean 2D binary mask.
    """
    if ror_mask_option is None:
        return 1

    elif isinstance(ror_mask_option, str):
        if ror_mask_option != "auto":
            raise ValueError(
                "ror_mask_option must be None, 'auto', a nonnegative float radius in ALU, "
                "or a 2D binary array."
            )

        return generate_2d_ror_mask(
            recon_shape,
            delta_voxel=delta_voxel,
            voxel_row_aspect=voxel_row_aspect,
            radius_alu=None,
            crop_radius_pixels=crop_radius_pixels,
            crop_radius_fraction=crop_radius_fraction
        )

    elif np.isscalar(ror_mask_option):
        radius_alu = float(ror_mask_option)
        if radius_alu <= 0:
            raise ValueError("ror_mask_option radius must be positive.")

        return generate_2d_ror_mask(
            recon_shape,
            delta_voxel=delta_voxel,
            voxel_row_aspect=voxel_row_aspect,
            radius_alu=radius_alu,
            crop_radius_pixels=crop_radius_pixels,
            crop_radius_fraction=crop_radius_fraction
        )
    
    else: # user-provided mask
        mask = np.asarray(ror_mask_option)
    
        if mask.shape != tuple(recon_shape[:2]):
            raise ValueError(
                "Custom ror_mask_option must have shape recon_shape[:2]. "
                f"Got mask.shape={mask.shape}, expected {tuple(recon_shape[:2])}."
            )
    
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError("Custom ror_mask_option must contain only 0s and 1s.")
    
        return mask


def gen_set_of_pixel_partitions(recon_shape, granularity, delta_voxel=1.0, voxel_row_aspect=1.0, output_device=None, ror_mask_option="auto"):
    """
    Generates a collection of voxel partitions for an array of specified partition sizes.
    This function creates a tuple of randomly generated 2D voxel partitions.

    Args:
        recon_shape (tuple): Shape of recon in (rows, columns, slices)
        granularity (list or tuple):  List of num_subsets to use for each partition
        delta_voxel (float): Column voxel pitch in ALU.
        voxel_row_aspect (float): Aspect ratio of recon voxel rows relative to columns
        output_device (jax device): Device on which to place the output of the partition
        ror_mask_option: None, 'auto', float radius in ALU, or 2D binary mask.

    Returns:
        tuple: A tuple of 2D arrays each representing a partition of voxels into the specified number of subsets.
    """
    partitions = []
    for num_subsets in granularity:
        partition = gen_pixel_partition(recon_shape, num_subsets, delta_voxel=delta_voxel, voxel_row_aspect=voxel_row_aspect, ror_mask_option=ror_mask_option)
        partitions += [jax.device_put(partition, output_device),]

    return partitions


def gen_pixel_partition_grid(recon_shape, num_subsets, delta_voxel=1.0, voxel_row_aspect=1.0, ror_mask_option="auto"):

    small_tile_side = np.ceil(np.sqrt(num_subsets)).astype(int)
    num_subsets = small_tile_side ** 2
    num_small_tiles = [np.ceil(recon_shape[k] / small_tile_side).astype(int) for k in [0, 1]]

    single_subset_inds = np.random.permutation(num_subsets).reshape((small_tile_side, small_tile_side))
    subset_inds = np.tile(single_subset_inds, num_small_tiles)
    subset_inds = subset_inds[:recon_shape[0], :recon_shape[1]]

    ror_mask = get_2d_ror_mask(recon_shape, delta_voxel=delta_voxel, voxel_row_aspect=voxel_row_aspect, ror_mask_option=ror_mask_option)
    subset_inds = (subset_inds + 1) * ror_mask - 1 # Get a - at each location outside the mask, subset_ind at other points 
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


def gen_pixel_partition(recon_shape, num_subsets, delta_voxel=1.0, voxel_row_aspect=1.0, ror_mask_option="auto"):
    """
    Generates a partition of pixel indices into specified number of subsets for use in tomographic reconstruction algorithms.
    The function ensures that each subset contains an equal number of pixels, suitable for VCD reconstruction.

    Args:
        recon_shape (tuple): Shape of recon in (rows, columns, slices)
        num_subsets (int): The number of subsets to divide the pixel indices into.
        delta_voxel (float): Column voxel pitch in ALU.
        voxel_row_aspect (float): Aspect ratio of recon voxel rows relative to columns
        ror_mask_option: None, 'auto', float radius in ALU, or 2D binary mask.

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
    if ror_mask_option is not None:
        mask = get_2d_ror_mask(recon_shape, delta_voxel=delta_voxel, voxel_row_aspect=voxel_row_aspect, ror_mask_option=ror_mask_option)
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


def gen_pixel_partition_blue_noise(recon_shape, num_subsets, delta_voxel=1.0, voxel_row_aspect=1.0, ror_mask_option="auto"):
    """
    Generates a partition of pixel indices into specified number of subsets for use in tomographic reconstruction algorithms.
    The function ensures that each subset contains an equal number of pixels, suitable for VCD reconstruction.

    Args:
        recon_shape (tuple): Shape of recon in (rows, columns, slices)
        num_subsets (int): The number of subsets to divide the pixel indices into.
        delta_voxel (float): Column voxel pitch in ALU.
        voxel_row_aspect (float): Aspect ratio of recon voxel rows relative to columns
        ror_mask_option: None, 'auto', float radius in ALU, or 2D binary mask.

    Raises:
        ValueError: If the number of subsets specified is greater than the total number of pixels in the grid.

    Returns:
        jnp.array: A JAX array where each row corresponds to a subset of pixel indices, sorted within each subset.
    """
    pattern = bn.bn256
    num_tiles = [np.ceil(recon_shape[k] / pattern.shape[k]).astype(int) for k in [0, 1]]
    ror_mask = get_2d_ror_mask(recon_shape, delta_voxel=delta_voxel, voxel_row_aspect=voxel_row_aspect, ror_mask_option=ror_mask_option)

    single_subset_inds = np.floor(pattern / (2**16 / num_subsets)).astype(int)

    # Repeat each bn subset to do the tiling
    subset_inds = np.tile(single_subset_inds, num_tiles)
    subset_inds = subset_inds[:recon_shape[0], :recon_shape[1]]
    subset_inds = (subset_inds + 1) * ror_mask - 1  # Get a -1 at each location outside the mask, subset_ind at other points
    subset_inds = subset_inds.flatten()
    num_valid_inds = np.sum(subset_inds >= 0)
    if num_subsets > num_valid_inds:
        return gen_pixel_partition(recon_shape, num_subsets, delta_voxel=delta_voxel, voxel_row_aspect=voxel_row_aspect, ror_mask_option=ror_mask_option)

    flat_inds = []
    max_points = 0
    min_points = subset_inds.size
    for k in range(num_subsets):
        cur_inds = np.where(subset_inds == k)[0]
        flat_inds.append(cur_inds)  # Get all the indices for each subset
        max_points = max(max_points, cur_inds.size)
        min_points = min(min_points, cur_inds.size)

    if min_points == 0:
        return gen_pixel_partition(recon_shape, num_subsets, delta_voxel=delta_voxel, voxel_row_aspect=voxel_row_aspect, ror_mask_option=ror_mask_option)

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


def gen_full_indices(recon_shape, delta_voxel=1.0, voxel_row_aspect=1.0, ror_mask_option="auto"):
    """
    Generates a full array of voxels in the region of reconstruction.
    This is useful for computing forward projections.
    """
    partition = gen_pixel_partition(recon_shape, num_subsets=1, delta_voxel=delta_voxel, voxel_row_aspect=voxel_row_aspect, ror_mask_option=ror_mask_option)
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
