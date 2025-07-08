import warnings
from enum import Enum
from typing import Union
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import jax.numpy as jnp
import jax
import mbirjax as mj
import mbirjax.bn256 as bn
import mbirjax.preprocess as mjp


class ObjectType(str, Enum):
    SHEPP_LOGAN = 'shepp-logan'
    CUBE = 'cube'


class ModelType(str, Enum):
    PARALLEL = 'parallel'
    CONE = 'cone'
    TRANSLATION = 'translation'


def generate_demo_data(
    object_type: Union[ObjectType, str] = ObjectType.SHEPP_LOGAN,
    model_type: Union[ModelType, str] = ModelType.CONE,
    num_views: int = 64,
    num_det_rows: int = 96,
    num_det_channels: int = 128,
    num_x_translations: int = 7,
    num_z_translations: int = 7,
    x_spacing: float = 22,
    z_spacing: float = 22
) -> (np.ndarray, np.ndarray):
    """
    Create a simple object and a sinogram for demonstration purposes.

    This function will create a 3D volume (aka object or phantom) of the specified type, then use the model type and
    parameters to create a simulated sinogram.  The object type 'shepp-logan' gives a simplified version of the
    classic Shepp-Logan test phantom, and type 'cube' gives a simple cube object.

    The output sinogram is a 3D numpy array with shape (num_views, num_det_rows, num_det_channels).  Each 2D array
    sinogram[view_index] is a simulated image from the detector, with num_det_rows indicating the vertical size
    and num_det_channels representing the horizontal size.

    Args:
        object_type (str, optional): One of 'shepp-logan' or 'cube'.  Defaults to 'shepp-logan'.
        model_type (str, optional): One of 'parallel', 'cone', or 'translation'.  Defaults to 'cone'.
        num_views (int, optional):  Number of views in the output sinogram.  Defaults to 64. Ignored when model_type is 'translation'
        num_det_rows (int, optional): Number of rows (vertical) in the output sinogram.  Defaults to 40.
        num_det_channels (int, optional): Number of channels (horizontal) in the output sinogram.  Defaults to 128.
        num_x_translations (int, optional): Number of horizontal translations for translation mode.  Defaults to 7.
        num_z_translations (int, optional): Number of vertical translations for translation mode.  Defaults to 7.
        x_spacing (float, optional): Horizontal spacing between translations in ALU.  Defaults to 22.
        z_spacing (float, optional): Vertical spacing between translations in ALU.  Defaults to 22.

    Returns:
        tuple: (object, sinogram, params)
            - object (np.ndarray): a volume with shape (num_det_channels, num_det_channels, num_det_rows)
            - sinogram (np.ndarray): a sinogram with shape (num_views, num_det_rows, num_det_channels)
            - params (dict): a dict containing 'angles' and, if model_type is 'cone', then also 'source_detector_dist' and 'source_iso_dist'
    """
    # Coerce types to Enum
    object_type = ObjectType(object_type)
    model_type = ModelType(model_type)

    start_angle = -np.pi
    end_angle = np.pi

    # Initialize model

    if model_type == ModelType.PARALLEL:
        sinogram_shape = (num_views, num_det_rows, num_det_channels)
        angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)
        ct_model_for_generation = mj.ParallelBeamModel(sinogram_shape, angles)
        params = {'angles': angles}
    elif model_type == ModelType.CONE:
        # For cone beam geometry, we need to describe the distances source to detector and source to rotation axis.
        # np.Inf is an allowable value, in which case this is essentially parallel beam
        source_detector_dist = 4 * num_det_channels
        source_iso_dist = source_detector_dist
        sinogram_shape = (num_views, num_det_rows, num_det_channels)
        angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)
        ct_model_for_generation = mj.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist,
                                                   source_iso_dist=source_iso_dist)
        params = {'angles': angles, 'source_detector_dist': source_detector_dist, 'source_iso_dist': source_iso_dist}
    elif model_type == ModelType.TRANSLATION:
        source_iso_dist = np.min(num_det_rows, num_det_channels) / 2
        source_detector_dist = source_iso_dist
        translation_vectors = gen_translation_vectors(num_x_translations, num_z_translations, x_spacing, z_spacing)
        num_views = translation_vectors.shape[0]
        sinogram_shape = (num_views, num_det_rows, num_det_channels)
        ct_model_for_generation = mj.TranslationModel(sinogram_shape, translation_vectors, source_detector_dist=source_detector_dist,
                                                      source_iso_dist=source_iso_dist)
        params = {'translation_vectors': translation_vectors}
    else:
        raise ValueError(f'Invalid model type. Expected one of {[m.value for m in ModelType]}, got {model_type}')

    # Generate phantom
    print('Creating phantom')
    recon_shape = ct_model_for_generation.get_params('recon_shape')
    device = ct_model_for_generation.main_device
    if object_type == ObjectType.SHEPP_LOGAN:
        phantom = mj.generate_3d_shepp_logan_low_dynamic_range(recon_shape, device=device)
    elif object_type == ObjectType.CUBE:
        phantom = gen_cube_phantom(recon_shape, device=device)
    else:
        raise ValueError(f'Invalid object type. Expected one of {[o.value for o in ObjectType]}, got {object_type}')

    # Generate synthetic sinogram data
    print('Creating sinogram')
    sinogram = ct_model_for_generation.forward_project(phantom)
    sinogram = np.asarray(sinogram)

    return phantom, sinogram, params


def gen_translation_vectors(num_x_translations, num_z_translations, x_spacing, z_spacing):
    """
    Generate translation vectors for lateral (x) and axial (z) displacements.

    Args:
        num_x_translations (int): Number of x-direction translations
        num_z_translations (int): Number of z-direction translations
        x_spacing (float): Spacing between x translations in ALU
        z_spacing (float): Spacing between z translations in ALU

    Returns:
        np.ndarray: Array of shape (num_views, 3) with translation vectors [dx, dy, dz]
    """
    num_views = num_x_translations * num_z_translations
    translation_vectors = np.zeros((num_views, 3))

    x_center = (num_x_translations - 1) / 2
    z_center = (num_z_translations - 1) / 2

    idx = 0
    for row in range(num_z_translations):
        for col in range(num_x_translations):
            dx = (col - x_center) * x_spacing
            dz = (row - z_center) * z_spacing
            dy = 0
            translation_vectors[idx] = [dx, dy, dz]
            idx += 1

    return translation_vectors


def stitch_arrays(array_list, overlap_length, axis=2):
    """
    Combine a list of jax arrays in a way similar to concatenate, except that adjacent arrays are assumed to overlap
    by a specified number of elements, so those elements are created as a weighted average of the overlapping elements.
    The weight varies linearly throughout the overlap.  For example, if arrays a0, a1 have size 5x5x10, 5x5x12
    respectively, with an overlap length of 4 in axis 2, then the output array has size 5x5x18, the overlapping
    entries are [:, :, 6+j] for j = 0, 1, 2, 3, and the entries in [:, :, 6+j] are given by
    (1 - (j + 1) / (4 + 1)) * a0[:, :, 6+j] + (j + 1) / (4 + 1) * a1[:, :, 6+j]

    Args:
        array_list (list of jax arrays):  List of jax arrays
        overlap_length (int):  Number of overlapping entries between adjacent arrays
        axis (int, optional): Axis along which to combine arrays

    Returns:
        jax array:  The stitched arrays
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
            if np.amin(lengths) < overlap_length:
                raise ValueError('Each array must have length at least overlap_length in the dimension specified by axis.')

    # Create a linear weight array
    weights = jnp.linspace(0, 1, overlap_length + 1, endpoint=False)[1:]
    weights_shape = np.ones(array_list[0].ndim).astype(int)
    weights_shape[0] = len(weights)
    weights = weights.reshape(weights_shape)

    # Start with the first array in the list
    stitched = jnp.swapaxes(array_list[0], 0, axis)

    # Iterate through each subsequent array in the list
    for next_array in array_list[1:]:
        # Extract the overlap from the current end of the stitched array and the beginning of the next array
        overlap_current = stitched[-overlap_length:]
        next_array = jnp.swapaxes(next_array, 0, axis)
        overlap_next = next_array[:overlap_length]

        # Weighted average for the overlapping part
        weighted_overlap = (1 - weights) * overlap_current + weights * overlap_next

        # Replace the overlap in the stitched array
        stitched = jnp.concatenate([stitched[:-overlap_length], weighted_overlap], axis=0)

        # Append the non-overlapping remainder of the next array
        stitched = jnp.concatenate([stitched, next_array[overlap_length:]], axis=0)

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


def gen_cube_phantom(recon_shape, device=None):
    """Code to generate a simple phantom """
    # Compute phantom height and width
    num_recon_rows, num_recon_cols, num_recon_slices = recon_shape[:3]
    phantom_rows = num_recon_rows // 4  # Phantom height
    phantom_cols = num_recon_cols // 4  # Phantom width

    # Allocate phantom memory
    phantom = np.zeros((num_recon_rows, num_recon_cols, num_recon_slices))

    # Compute start and end locations
    start_rows = (num_recon_rows - phantom_rows) // 2
    stop_rows = (num_recon_rows + phantom_rows) // 2
    start_cols = (num_recon_cols - phantom_cols) // 2
    stop_cols = (num_recon_cols + phantom_cols) // 2
    for slice_index in np.arange(num_recon_slices):
        shift_cols = int(slice_index * phantom_cols / num_recon_slices)
        phantom[start_rows:stop_rows, (shift_cols + start_cols):(shift_cols + stop_cols), slice_index] = 1.0 / max(
            phantom_rows, phantom_cols)

    return jnp.array(phantom, device=device)


@jax.jit
def add_ellipsoid(current_volume, grids, z_locations, x0, y0, z0, a, b, c, angle=0, intensity=1.0):
    """
    Add an ellipsoid to an existing jax array.  This is done using lax.scan over the z slices to avoid
    using really large arrays when the volume is large.

    Args:
        current_volume (jax array): 3D volume
        grids (tuple):  A tuple of x_grid, y_grid, i_grid, j_grid obtained as in generate_3d_shepp_logan_low_dynamic_range
        z_locations (jax array): A 1D array of z coordinates of the volume
        x0 (float): x center for the ellipsoid
        y0 (float): y center for the ellipsoid
        z0 (float): z center for the ellipsoid
        a (float): x radius
        b (float): y radius
        c (float): z radius
        angle (float): angle of rotation of the ellipsoid in the xy plane around (x0, y0)
        intensity (float): The constant value of the ellipsoid to be added.

    Returns:
        3D jax array: current_volume + ellipsoid
    """

    # Unpack the grids and determine the xy locations for this angle
    x_grid, y_grid, i_grid, j_grid = grids
    cos_angle = jnp.cos(jnp.deg2rad(angle))
    sin_angle = jnp.sin(jnp.deg2rad(angle))
    Xr = cos_angle * (x_grid - x0) + sin_angle * (y_grid - y0)
    Yr = -sin_angle * (x_grid - x0) + cos_angle * (y_grid - y0)

    # Determine which xy locations will be updated for this ellipsoid
    xy_norm = Xr**2 / a**2 + Yr**2 / b**2

    def add_slice_vmap(volume_slice, z):
        return volume_slice + intensity * ((xy_norm + (z - z0)**2 / c**2) <= 1).astype(float)

    volume_map = jax.vmap(add_slice_vmap, in_axes=(2, 0), out_axes=2)
    current_volume = volume_map(current_volume, z_locations)

    return current_volume


def _gen_ellipsoid(x_grid, y_grid, z_grid, x0, y0, z0, a, b, c, gray_level, alpha=0, beta=0, gamma=0):
    """
    Return an image with a 3D ellipsoid in a 3D plane with a center of [x0,y0,z0] and ...

    Args:
        x_grid(jax array): 3D grid of X coordinate values.
        y_grid(jax array): 3D grid of Y coordinate values.
        z_grid(jax array): 3D grid of Z coordinate values.
        x0(float): horizontal center of ellipsoid.
        y0(float): vertical center of ellipsoid.
        z0(float): normal center of ellipsoid.
        a(float): X-axis radius.
        b(float): Y-axis radius.
        c(float): Z-axis radius.
        gray_level(float): Gray level for the ellipse.
        alpha(float): [Default=0.0] counter-clockwise angle of rotation by X-axis in radians.
        beta(float): [Default=0.0] counter-clockwise angle of rotation by Y-axis in radians.
        gamma(float): [Default=0.0] counter-clockwise angle of rotation by Z-axis in radians.

    Return:
        ndarray: 3D array with the same shape as x_grid, y_grid, and z_grid

    """
    # Generate Rotation Matrix.
    rx = np.array([[1, 0, 0], [0, np.cos(-alpha), -np.sin(-alpha)], [0, np.sin(-alpha), np.cos(-alpha)]])
    ry = np.array([[np.cos(-beta), 0, np.sin(-beta)], [0, 1, 0], [-np.sin(-beta), 0, np.cos(-beta)]])
    rz = np.array([[np.cos(-gamma), -np.sin(-gamma), 0], [np.sin(-gamma), np.cos(-gamma), 0], [0, 0, 1]])
    r = np.dot(rx, np.dot(ry, rz))

    cor = np.array([x_grid.flatten() - x0, y_grid.flatten() - y0, z_grid.flatten() - z0])

    image = ((np.dot(r[0], cor)) ** 2 / a ** 2 + (np.dot(r[1], cor)) ** 2 / b ** 2 + (
        np.dot(r[2], cor)) ** 2 / c ** 2 <= 1.0) * gray_level

    return image.reshape(x_grid.shape)


def generate_3d_shepp_logan_reference(phantom_shape):
    """
    Generate a 3D Shepp Logan phantom based on below reference.

    Kak AC, Slaney M. Principles of computerized tomographic imaging. Page.102. IEEE Press, New York, 1988. https://engineering.purdue.edu/~malcolm/pct/CTI_Ch03.pdf

    Args:
        phantom_shape (tuple or list of ints): num_rows, num_cols, num_slices

    Return:
        out_image: 3D array, num_slices*num_rows*num_cols

    Note:
        This function produces 6 intermediate arrays that each have shape phantom_shape, so if phantom_shape is
        large, then this will use a lot of peak memory.
    """

    # The function describing the phantom is defined as the sum of 10 ellipsoids inside a 2×2×2 cube:
    sl3d_paras = [
        {'x0': 0.0, 'y0': 0.0, 'z0': 0.0, 'a': 0.69, 'b': 0.92, 'c': 0.9, 'gamma': 0, 'gray_level': 2.0},
        {'x0': 0.0, 'y0': 0.0, 'z0': 0.0, 'a': 0.6624, 'b': 0.874, 'c': 0.88, 'gamma': 0, 'gray_level': -0.98},
        {'x0': -0.22, 'y0': 0.0, 'z0': -0.25, 'a': 0.41, 'b': 0.16, 'c': 0.21, 'gamma': 108, 'gray_level': -0.02},
        {'x0': 0.22, 'y0': 0.0, 'z0': -0.25, 'a': 0.31, 'b': 0.11, 'c': 0.22, 'gamma': 72, 'gray_level': -0.02},
        {'x0': 0.0, 'y0': 0.35, 'z0': -0.25, 'a': 0.21, 'b': 0.25, 'c': 0.5, 'gamma': 0, 'gray_level': 0.02},
        {'x0': 0.0, 'y0': 0.1, 'z0': -0.25, 'a': 0.046, 'b': 0.046, 'c': 0.046, 'gamma': 0, 'gray_level': 0.02},
        {'x0': -0.08, 'y0': -0.65, 'z0': -0.25, 'a': 0.046, 'b': 0.023, 'c': 0.02, 'gamma': 0, 'gray_level': 0.01},
        {'x0': 0.06, 'y0': -0.65, 'z0': -0.25, 'a': 0.046, 'b': 0.023, 'c': 0.02, 'gamma': 90, 'gray_level': 0.01},
        {'x0': 0.06, 'y0': -0.105, 'z0': 0.625, 'a': 0.056, 'b': 0.04, 'c': 0.1, 'gamma': 90, 'gray_level': 0.02},
        {'x0': 0.0, 'y0': 0.1, 'z0': 0.625, 'a': 0.056, 'b': 0.056, 'c': 0.1, 'gamma': 0, 'gray_level': -0.02}
    ]

    num_rows, num_cols, num_slices = phantom_shape
    axis_x = np.linspace(-1.0, 1.0, num_cols)
    axis_y = np.linspace(1.0, -1.0, num_rows)
    axis_z = np.linspace(-1.0, 1.0, num_slices)

    x_grid, y_grid, z_grid = np.meshgrid(axis_x, axis_y, axis_z)
    image = x_grid * 0.0

    for el_paras in sl3d_paras:
        image += _gen_ellipsoid(x_grid=x_grid, y_grid=y_grid, z_grid=z_grid, x0=el_paras['x0'], y0=el_paras['y0'],
                               z0=el_paras['z0'],
                               a=el_paras['a'], b=el_paras['b'], c=el_paras['c'],
                               gamma=el_paras['gamma'] / 180.0 * np.pi,
                               gray_level=el_paras['gray_level'])

    return image.transpose((1, 0, 2))


def generate_3d_shepp_logan_low_dynamic_range(phantom_shape, device=None):
    """
    Generates a 3D Shepp-Logan phantom with specified dimensions.

    Args:
        phantom_shape (tuple): Phantom shape in (rows, columns, slices).
        device (jax device): Device on which to place the output phantom.

    Returns:
        ndarray: A 3D numpy array of shape phantom_shape representing the voxel intensities of the phantom.

    Note:
        This function uses a memory-efficient approach to generating large phantoms.
    """
    # Get space for the result and set up the grids for add_ellipsoid
    with jax.default_device(device):
        phantom = jnp.zeros(phantom_shape, device=device)
    N, M, P = phantom_shape
    x_locations = jnp.linspace(-1, 1, N)
    y_locations = jnp.linspace(-1, 1, M)
    z_locations = jnp.linspace(-1, 1, P)
    x_grid, y_grid = jnp.meshgrid(x_locations, y_locations, indexing='ij')
    i_grid, j_grid = jnp.meshgrid(jnp.arange(N), jnp.arange(M), indexing='ij')
    grids = (x_grid, y_grid, i_grid, j_grid)

    # Main ellipsoid
    phantom = add_ellipsoid(phantom, grids, z_locations, 0, 0, 0, 0.69, 0.92, 0.9, intensity=1)
    # Smaller ellipsoids and other structures
    phantom = add_ellipsoid(phantom, grids, z_locations, 0, 0.0184, 0, 0.6624, 0.874, 0.88, intensity=-0.8)
    phantom = add_ellipsoid(phantom, grids, z_locations, 0.22, 0, 0, 0.41, 0.16, 0.21, angle=108, intensity=-0.2)
    phantom = add_ellipsoid(phantom, grids, z_locations, -0.22, 0, 0, 0.31, 0.11, 0.22, angle=72, intensity=-0.2)
    phantom = add_ellipsoid(phantom, grids, z_locations, 0, 0.35, 0, 0.21, 0.25, 0.5, intensity=0.1)
    phantom = add_ellipsoid(phantom, grids, z_locations, 0, 0.1, 0, 0.046, 0.046, 0.046, intensity=0.1)
    phantom = add_ellipsoid(phantom, grids, z_locations, 0, -0.1, 0, 0.046, 0.046, 0.046, intensity=0.1)
    phantom = add_ellipsoid(phantom, grids, z_locations, -0.08, -0.605, 0, 0.046, 0.023, 0.02, angle=0, intensity=0.1)
    phantom = add_ellipsoid(phantom, grids, z_locations, 0, -0.605, 0, 0.023, 0.023, 0.02, angle=0, intensity=0.1)

    return phantom


def gen_translation_phantom(recon_shape, option, words, fill_rate=0.05, font_size=20):
    """
    Generate a synthetic ground truth phantom based on the selected option.

    Args:
        option (str): Phantom type to generate. Options are 'dots' or 'text'.
        recon_shape (tuple[int, int, int]): Shape of the reconstruction volume.
        words (list[str]): List of ASCII words to render.
        fill_rate (float, optional): Fill rate of the reconstruction volume. Default is 0.05.
        font_size (int, optional): Font size of the ASCII words. Default is 20.

    Returns:
        np.ndarray: Generated phantom volume.
    """
    if option == 'dots':
        return gen_dot_phantom(recon_shape, fill_rate)
    elif option == 'text':
        return gen_text_phantom(recon_shape, words, font_size)
    else:
        raise ValueError(f"Unsupported phantom option: {option}")


def gen_dot_phantom(recon_shape, fill_rate):
    """
    Generate a synthetic ground truth reconstruction volume.

    Args:
        recon_shape (tuple[int, int, int]): Shape of the reconstruction volume.
        fill_rate (float): Fill rate of the reconstruction volume.

    Returns:
        np.ndarray: Ground truth reconstruction volume with sparse binary features.
    """
    np.random.seed(42)
    gt_recon = np.zeros(recon_shape, dtype=np.float32)

    y_pad = recon_shape[0] // 6
    central_start = y_pad
    central_end = recon_shape[0] - y_pad

    row_size = recon_shape[1] * recon_shape[2]
    num_ones_per_row = int(row_size * fill_rate)

    for row_idx in range(central_start, central_end):
        flat_row = gt_recon[row_idx].flatten()
        positions_ones = np.random.choice(row_size, num_ones_per_row, replace=False)
        flat_row[positions_ones] = 1.0
        gt_recon[row_idx] = flat_row.reshape(recon_shape[1:])

    return gt_recon


def gen_text_phantom(recon_shape, words, font_size, font_path="DejaVuSans.ttf"):
    """
    Generate a 3D text phantom with binary word patterns embedded in specific slices.

    Args:
        recon_shape (tuple[int, int, int]): Shape of the phantom volume (num_rows, num_cols, num_slices).
        words (list[str]): List of ASCII words to render.
        font_size (int): Font size of ASCII words.
        font_path (str, optional): Path to the TrueType font file. Default is "DejaVuSans.ttf".

    Returns:
        np.ndarray: A 3D numpy array of shape `recon_shape` containing the text phantom.
    """
    positions = []
    for i in range(1, len(words) + 1):
        positions.append((recon_shape[0] // (len(words) + 1) * i, recon_shape[1] // 2, recon_shape[2] // 2))

    array_size = np.minimum(recon_shape[1], recon_shape[2])

    phantom = np.zeros(recon_shape, dtype=np.float32)
    try:
        font = ImageFont.truetype(font_path, size=font_size)
    except OSError:
        from pathlib import Path
        fallback_paths = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS fallback
            "/Library/Fonts/Arial.ttf",  # Additional macOS path
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux fallback
        ]
        for fallback in fallback_paths:
            if Path(fallback).exists():
                font = ImageFont.truetype(fallback, size=font_size)
                break
        else:
            raise FileNotFoundError(
                f"Could not find a usable font. Tried the following paths:\n"
                + "\n".join(fallback_paths)
                + "\nPlease install one of these fonts or specify a valid font_path."
            )

    for word, (r, c, s) in zip(words, positions):
        img = Image.new('L', (array_size, array_size), 0)
        draw = ImageDraw.Draw(img)

        text_box = draw.textbbox((0, 0), word, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]

        x = (array_size - text_width) // 2 - text_box[0]
        y = (array_size - text_height) // 2 - text_box[1]
        draw.text((x, y), word, fill=1, font=font)

        word_array = np.array(img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT))
        word_array = (word_array > 0).astype(np.float32)

        # Crop or pad word_array to fit in the recon volume
        r_start = max(0, r)
        r_end = min(recon_shape[0], r + 1)
        c_start = max(0, c - array_size // 2)
        c_end = min(recon_shape[1], c_start + array_size)
        s_start = max(0, s - array_size // 2)
        s_end = min(recon_shape[2], s_start + array_size)

        word_crop = word_array[:(c_end - c_start), :(s_end - s_start)]
        phantom[r_start:r_end, c_start:c_end, s_start:s_end] = word_crop

    return phantom


def gen_weights_mar(ct_model, sinogram, init_recon=None, metal_threshold=None, beta=1.0, gamma=3.0):
    """
    Implementation of TomographyModel.gen_weights_mar.  If metal_threshold is not provided, it will be estimated using
     Otsu's method.  If init_recon is provided, it will be used along with metal threshold to estimate the region
     containing metal, and this metal region will be forward projected, with the projection used to estimate the weights.
    The weights are placed on ct_model.main_device.
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
