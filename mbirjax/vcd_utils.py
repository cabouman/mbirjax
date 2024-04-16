import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt


def get_2d_ror_mask(num_recon_rows, num_recon_cols):
    """
    Get a binary mask for the region of reconstruction.
    Returns:
        A binary mask for the region of reconstruction.
    """
    # Set up a mask to zero out points outside the ROR
    row_center = (num_recon_rows - 1) / 2
    col_center = (num_recon_cols - 1) / 2

    radius = max(row_center, col_center)

    col_coords = np.arange(num_recon_cols) - col_center
    row_coords = np.arange(num_recon_rows) - row_center

    coords = np.meshgrid(col_coords, row_coords)  # Note the interchange of rows and columns for meshgrid
    mask = coords[0] ** 2 + coords[1] ** 2 <= radius ** 2
    mask = mask[:, :]
    return mask

def get_voxels_at_indices(recon, indices):
    """
    This function seems to produce a bug, but I don't understand why. Charlie
    """
    num_rows = recon.shape[0]
    print(f'recon shape: {recon.shape}; indices shape: {indices.shape}')
    voxel_values = recon.reshape((-1, num_rows))[indices]
    print(f'voxel_values shape: {voxel_values.shape}')
    return voxel_values


def gen_voxel_partition( num_recon_rows, num_recon_cols, num_subsets):
    """
    Generates a partition of voxel indices into specified number of subsets for use in tomographic reconstruction algorithms.
    The function ensures that each subset contains an equal number of voxels, suitable VCD reconstruction.
    Parameters:
        num_recon_rows (int): The number of rows in the reconstruction grid.
        num_recon_cols (int): The number of columns in the reconstruction grid.
        num_subsets (int): The number of subsets to divide the voxel indices into.
    Raises:
        ValueError: If the number of subsets specified is greater than the total number of voxels in the grid.
    Returns:
        jnp.array: A JAX array where each row corresponds to a subset of voxel indices, sorted within each subset.
    """
    # Determine the 2D indices within the RoR
    max_index_val = num_recon_rows * num_recon_cols
    indices = np.arange(max_index_val, dtype=np.int32)
    if num_subsets > max_index_val:
        raise ValueError('num_subsets must not be larger than max_index_val')

    # Mask off indices that are outside the region of reconstruction
    mask = get_2d_ror_mask(num_recon_rows, num_recon_cols)
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


def gen_indices_d2(num_recon_rows, num_recon_cols, block_width):
    """
    Generates an index array for a 2D reconstruction using a block size of block_width x block_width
    """
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

def gen_phantom(num_recon_rows, num_recon_cols, num_recon_slices=1):
    """Code to generate a simple phantom """
    # Compute phantom height and width
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
        phantom[start_rows:stop_rows, (shift_cols+start_cols):(shift_cols+stop_cols), slice_index] = 1.0 / max(phantom_rows,phantom_cols)

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

def generate_3d_shepp_logan( N, M, P ):
    """
    Generates a 3D Shepp-Logan phantom with specified dimensions.
    Args:
        N (int): Number of voxels along the x-axis (width).
        M (int): Number of voxels along the y-axis (height).
        P (int): Number of voxels along the z-axis (depth).
    Returns:
        ndarray: A 3D numpy array of shape (N, M, P) representing the voxel intensities of the phantom.
    """
    phantom = np.zeros((N, M, P))

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

def plot_granularity_and_loss( granularity_sequences, losses, labels, granularity_ylim=None, loss_ylim=None, fig_title=None ):
    """
    Plots multiple granularity and loss data sets on a single figure with separate subplots, using fixed scales for all plots.

    Args:
        granularity_sequences (list of lists): A list containing different granularity sequences.
        losses (list of lists): A list containing different loss data corresponding to the granularity sequences.
        labels (list of str): Labels for each subplot to distinguish between different data sets.
        granularity_ylim (tuple, optional): Limits for the granularity axis (y-limits), applied to all plots.
        loss_ylim (tuple, optional): Limits for the loss axis (y-limits), applied to all plots.
    """
    num_plots = len(granularity_sequences)
    fig, axes = plt.subplots(nrows=1, ncols=num_plots, figsize=(5 * num_plots, 4), sharey='row')
    fig.suptitle(fig_title)

    if num_plots == 1:
        axes = [axes]  # Make it iterable for a single subplot scenario

    for ax, granularity_sequence, loss, label in zip(axes, granularity_sequences, losses, labels):
        index = list(range(len(granularity_sequence)))

        # Plot granularity sequence on the first y-axis
        ax1 = ax
        ax1.stem(index, granularity_sequence, label='Granularity Sequence', basefmt=" ", linefmt='b', markerfmt='bo')
        ax1.set_ylabel('Granularity', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        if granularity_ylim:
            ax1.set_ylim(granularity_ylim)  # Apply fixed y-limit for granularity

        # Create a second y-axis for the loss
        ax2 = ax1.twinx()
        ax2.plot(index, loss, label='Loss', color='r')
        ax2.set_ylabel('Loss', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_yscale('log')
        if loss_ylim:
            ax2.set_ylim(loss_ylim)  # Apply fixed y-limit for loss, ensure log scale is considered

        # Set labels and legends
        ax1.set_xlabel('Iteration Number')
        ax.set_title(label)

        # Add legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.tight_layout()
    plt.show()

def debug_plot_indices(num_recon_rows, num_recon_cols, indices, recon_at_indices=None, num_recon_slices=1, title='Debug Plot'):
    """
    Visualizes indices on a reconstruction grid and optionally displays reconstruction data at these indices.

    Parameters:
        num_recon_rows (int): Number of rows in the reconstruction grid.
        num_recon_cols (int): Number of columns in the reconstruction grid.
        indices (array): Flat indices in the reconstruction grid to be highlighted or modified.
        recon_at_indices (array, optional): Values to set at specified indices in the reconstruction grid. If provided,
                                            displays the reconstruction at these indices across slices.
        num_recon_slices (int): Number of slices in the reconstruction grid, default is 1.
        title (str): Title for the plot.

    Usage:
        When recon_at_indices is not provided, the function visualizes the indices on a 2D grid.
        When recon_at_indices is provided, it also shows the reconstructed values at these indices in 3D.

    Example:
        debug_plot_indices_or_reconstruction(100, 100, [5050, 10001], recon_at_indices=[1, -1], num_recon_slices=5, title='Recon Visualization')
    """
    # Create an empty grid
    recon = np.zeros((num_recon_rows * num_recon_cols, num_recon_slices))

    # Create a mask for indices
    mask = np.zeros(num_recon_rows * num_recon_cols)
    mask[indices] = 1  # Highlight indices
    mask = mask.reshape((num_recon_rows, num_recon_cols))

    # If reconstruction data is provided, add it to the recon grid
    if recon_at_indices is not None:
        for i, idx in enumerate(indices):
            recon[idx, :] += recon_at_indices[i]

    # Visualization
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    # Display the mask of indices
    im_mask = ax[0].imshow(mask, cmap='viridis')
    ax[0].set_title('Mask of Indices')
    plt.colorbar(im_mask, ax=ax[0])

    if recon_at_indices is not None:
        # Display the reconstructed values if provided
        im_recon = ax[1].imshow(recon[:, 0].reshape((num_recon_rows, num_recon_cols)), cmap='viridis')
        ax[1].set_title(f'{title} at Indices')
        plt.colorbar(im_recon, ax=ax[1])
    else:
        ax[1].axis('off')  # Turn off the second subplot if no reconstruction data is provided

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def debug_plot_partitions( partitions, num_recon_rows, num_recon_cols ):
    """
    Visualizes a set of partitions as color images in a single row, where each partition is represented by a different color.

    Parameters:
        partitions (tuple of arrays): A tuple where each element is a 2D numpy array representing a partition.
        num_recon_rows (int): Number of rows in the original image grid.
        num_recon_cols (int): Number of columns in the original image grid.
    """
    plt.rcParams.update({'font.size': 24})  # Adjust font size here
    num_partitions = len(partitions)
    fig, axes = plt.subplots(nrows=1, ncols=num_partitions, figsize=(5 * num_partitions, 5))

    for i, partition in enumerate(partitions):
        # Create an empty image array to fill with subset colors
        image = np.zeros((num_recon_rows * num_recon_cols), dtype=int)

        # Assign a unique color (integer label) to each subset
        for subset_index, indices in enumerate(partition):
            image[indices.flatten()] = subset_index + 1  # Color code starts from 1 upwards

        # Reshape the image array back to 2D format
        image = image.reshape((num_recon_rows, num_recon_cols))

        # Plotting
        if num_partitions == 1:
            ax = axes
        else:
            ax = axes[i]

        cax = ax.imshow(image, cmap='nipy_spectral', interpolation='nearest')
        ax.set_title(f'{len(partition)} Partitions')
        ax.axis('off')  # Turn off axis labels

    plt.tight_layout()
    plt.show()

