import numpy as np
import jax.numpy as jnp


def get_2d_ror_mask(recon_shape):
    """
    Get a binary mask for the region of reconstruction.

    Args:
        recon_shape (tuple): Shape of recon in (rows, columns, slices)

    Returns:
        A binary mask for the region of reconstruction.
    """
    # Set up a mask to zero out points outside the ROR
    num_recon_rows, num_recon_cols = recon_shape[:2]
    row_center = (num_recon_rows - 1) / 2
    col_center = (num_recon_cols - 1) / 2

    radius = max(row_center, col_center)

    col_coords = np.arange(num_recon_cols) - col_center
    row_coords = np.arange(num_recon_rows) - row_center

    coords = np.meshgrid(col_coords, row_coords)  # Note the interchange of rows and columns for meshgrid
    mask = coords[0]**2 + coords[1]**2 <= radius**2
    mask = mask[:, :]
    return mask


def gen_set_of_pixel_partitions(recon_shape, granularity):
    """
    Generates a collection of voxel partitions for an array of specified partition sizes.
    This function creates a tuple of randomly generated 2D voxel partitions.

    Args:
        recon_shape (tuple): Shape of recon in (rows, columns, slices)
        granularity (list or tuple):  List of num_subsets to use for each partition

    Returns:
        tuple: A tuple of 2D arrays each representing a partition of voxels into the specified number of subsets.
    """
    partitions = ()
    for num_subsets in granularity:
        partition = gen_pixel_partition(recon_shape, num_subsets)
        partitions += (partition,)

    return partitions


def gen_pixel_partition(recon_shape, num_subsets):
    """
    Generates a partition of pixel indices into specified number of subsets for use in tomographic reconstruction algorithms.
    The function ensures that each subset contains an equal number of pixels, suitable VCD reconstruction.

    Args:
        recon_shape (tuple): Shape of recon in (rows, columns, slices)
        num_subsets (int): The number of subsets to divide the pixel indices into.

    Raises:
        ValueError: If the number of subsets specified is greater than the total number of pixels in the grid.

    Returns:
        jnp.array: A JAX array where each row corresponds to a subset of pixel indices, sorted within each subset.
    """
    # Determine the 2D indices within the RoR
    num_recon_rows, num_recon_cols = recon_shape[:2]
    max_index_val = num_recon_rows * num_recon_cols
    indices = np.arange(max_index_val, dtype=np.int32)
    if num_subsets > max_index_val:
        raise ValueError('num_subsets must not be larger than max_index_val')

    # Mask off indices that are outside the region of reconstruction
    mask = get_2d_ror_mask(recon_shape)
    mask = mask.flatten()
    indices = indices[mask == 1]

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
    indices = np.sort(indices, axis=1)

    return jnp.array(indices)


def gen_indices_d2(recon_shape, block_width):
    """
    Generates an index array for a 2D reconstruction using a block size of block_width x block_width

    Args:
        recon_shape (tuple): Shape of recon in (rows, columns, slices)
        block_width (int): side length of block

    Returns:

    """
    num_recon_rows, num_recon_cols = recon_shape[:2]
    max_index_val = num_recon_rows * num_recon_cols

    # Make sure rows and columns are divisible by block_width, but make sure to round up
    num_recon_rows = block_width * ((num_recon_rows // block_width) + (num_recon_rows % block_width))
    num_recon_cols = block_width * ((num_recon_cols // block_width) + (num_recon_cols % block_width))

    # Generate an array, but make sure its values don't go outside the valid range
    indices = np.arange(num_recon_rows * num_recon_cols) % max_index_val

    indices = indices.reshape(num_recon_rows, num_recon_cols // block_width, block_width)
    indices = np.transpose(indices, axes=(2, 1, 0))
    indices = indices.reshape(block_width, num_recon_cols // block_width, num_recon_rows // block_width, block_width)
    indices = np.transpose(indices, axes=(0, 3, 2, 1))
    indices = indices.reshape(block_width * block_width,
                              (num_recon_rows // block_width) * (num_recon_cols // block_width))

    return jnp.array(indices)


def gen_cube_phantom(recon_shape):
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

    return jnp.array(phantom)


def ellipsoid(x0, y0, z0, a, b, c, N, M, P, angle=0, intensity=1.0):
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, M)
    z = np.linspace(-1, 1, P)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    cos_angle = np.cos(np.deg2rad(angle))
    sin_angle = np.sin(np.deg2rad(angle))
    Xr = cos_angle * (X - x0) + sin_angle * (Y - y0)
    Yr = -sin_angle * (X - x0) + cos_angle * (Y - y0)
    Zr = Z - z0

    ellipsoid = (Xr**2 / a**2 + Yr**2 / b**2 + Zr**2 / c**2) <= 1
    return ellipsoid.astype(float) * intensity


def _gen_ellipsoid(x_grid, y_grid, z_grid, x0, y0, z0, a, b, c, gray_level, alpha=0, beta=0, gamma=0):
    """
    Return an image with a 3D ellipsoid in a 3D plane with a center of [x0,y0,z0] and ...

    Args:
        x_grid(float): 3D grid of X coordinate values.
        y_grid(float): 3D grid of Y coordinate values.
        z_grid(float): 3D grid of Z coordinate values.
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
        num_rows: int, number of rows.
        num_cols: int, number of cols.
        num_slices: int, number of slices.

    Return:
        out_image: 3D array, num_slices*num_rows*num_cols
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

    return image


def generate_3d_shepp_logan(phantom_shape):
    """
    Generates a 3D Shepp-Logan phantom with specified dimensions.

    Args:
        phantom_shape (tuple): Phantom shape in (rows, columns, slices).

    Returns:
        ndarray: A 3D numpy array of shape phantom_shape representing the voxel intensities of the phantom.
    """
    phantom = np.zeros(phantom_shape)
    N, M, P = phantom_shape

    # Main ellipsoid
    phantom += ellipsoid(0, 0, 0, 0.69, 0.92, 0.9, N, M, P, intensity=1)
    # Smaller ellipsoids and other structures
    phantom += ellipsoid(0, 0.0184, 0, 0.6624, 0.874, 0.88, N, M, P, intensity=-0.8)
    phantom += ellipsoid(0.22, 0, 0, 0.41, 0.16, 0.21, N, M, P, angle=108, intensity=-0.2)
    phantom += ellipsoid(-0.22, 0, 0, 0.31, 0.11, 0.22, N, M, P, angle=72, intensity=-0.2)
    phantom += ellipsoid(0, 0.35, 0, 0.21, 0.25, 0.5, N, M, P, intensity=0.1)
    phantom += ellipsoid(0, 0.1, 0, 0.046, 0.046, 0.046, N, M, P, intensity=0.1)
    phantom += ellipsoid(0, -0.1, 0, 0.046, 0.046, 0.046, N, M, P, intensity=0.1)
    phantom += ellipsoid(-0.08, -0.605, 0, 0.046, 0.023, 0.02, N, M, P, angle=0, intensity=0.1)
    phantom += ellipsoid(0, -0.605, 0, 0.023, 0.023, 0.02, N, M, P, angle=0, intensity=0.1)

    return jnp.array(phantom)
