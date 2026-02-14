import numpy as np
import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import tqdm
import mbirjax as mj


def debug_plot_partitions(partitions, recon_shape):
    """
    Visualizes a set of partitions as color images in a single row, where each partition is represented by a different color.

    Parameters:
        partitions (tuple of arrays): A tuple where each element is a 2D numpy array representing a partition.
        recon_shape (tuple): Shape of phantom in (rows, columns, slices).
    """
    num_recon_rows, recon_shape = recon_shape[:2]
    font_size = plt.rcParams["font.size"]
    plt.rcParams.update({'font.size': 24})  # Adjust font size here
    num_partitions = len(partitions)
    fig = plt.figure(figsize=(23, 5))
    # ImageGrid spacing units are inches; tune these for row/column spacing density.
    grid_axes_pad = (0.20, 0.30)   # (horizontal, vertical) spacing between subplots
    grid_cbar_pad = 0.10           # spacing between last column and colorbar
    grid_cbar_size = "2.5%"        # colorbar width
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(1, num_partitions),
        axes_pad=grid_axes_pad,
        share_all=False,
        aspect=False,
        cbar_mode=None,
        cbar_location="right",
        cbar_pad=grid_cbar_pad,
        cbar_size=grid_cbar_size,
    )

    for i, partition in enumerate(partitions):
        # Create an empty image array to fill with subset colors
        image = np.zeros((num_recon_rows * recon_shape), dtype=int)

        # Assign a unique color (integer label) to each subset
        for subset_index, indices in enumerate(partition):
            image[indices.flatten()] = subset_index + 1  # Color code starts from 1 upwards

        # Reshape the image array back to 2D format
        image = image.reshape((num_recon_rows, recon_shape))

        # Plotting
        ax = grid[i]
        im = ax.imshow(
            image,
            cmap='nipy_spectral',
            aspect='equal',
            extent=(0.0, 1.0, 0.0, 1.0),
            origin='upper',
            interpolation='nearest',
        )
        if len(partition) == 1:
            ax.set_title(f'{len(partition)} subset')
        else:
            ax.set_title(f'{len(partition)} subsets')
        ax.axis('off')  # Turn off axis labels

    plt.tight_layout()
    plt.show()

    plt.rcParams.update({'font.size': font_size})  # Reset the font size


if __name__ == "__main__":

    # Generate sequence of partition images for Figure 1
    recon_shape = (32, 32, 1)
    num_angles = recon_shape[0] // 2
    angles = np.linspace(0, np.pi, num=num_angles, endpoint=False)
    num_views = len(angles)
    sinogram_shape = (num_views, 1, recon_shape[0])
    ct_model = mj.ParallelBeamModel(sinogram_shape, angles)

    granularity = [1, 4, 16, 64, 128]
    use_ror_mask = True
    use_grid_subsets = True
    min_num_indices = 32  # len(partitions[-1][0])

    if use_grid_subsets:
        gen_partition = mj.gen_pixel_partition_grid
    else:
        gen_partition = mj.gen_pixel_partition

    partitions = []
    for num_subsets in granularity:
        partition = gen_partition(recon_shape, num_subsets, use_ror_mask=use_ror_mask)
        partitions.append(partition)

    # Plot the set of partitions
    debug_plot_partitions(partitions=partitions, recon_shape=recon_shape)

    vmin = 0.1
    log_vmin = np.log10(vmin)
    log_vmax = None
    # Plot A^T A for each of the subsets
    ncols = len(partitions)
    fig = plt.figure(figsize=(23, 10))
    # ImageGrid spacing units are inches; tune these for row/column spacing density.
    grid_axes_pad = (0.20, 0.30)   # (horizontal, vertical) spacing between subplots
    grid_cbar_pad = 0.10           # spacing between last column and colorbar
    grid_cbar_size = "2.5%"        # colorbar width
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(2, ncols),
        axes_pad=grid_axes_pad,
        share_all=False,
        aspect=False,
        cbar_mode="single",
        cbar_location="right",
        cbar_pad=grid_cbar_pad,
        cbar_size=grid_cbar_size,
    )
    plt.rcParams.update({'font.size': 24})
    # Build restricted dense matrix M = (A^T A)[I, I], where I = partitions[0][0].
    # Each column is A^T A applied to a basis vector e_j in the restricted index set.
    restricted_indices = np.asarray(partitions[0][0]).flatten()
    num_restricted = restricted_indices.size
    num_pixels = int(np.prod(recon_shape))
    at_a_restricted = np.zeros((num_restricted, num_restricted), dtype=np.float32)

    basis_vector = np.zeros(num_pixels, dtype=np.float32)
    for col, pixel_index in tqdm(
        enumerate(restricted_indices),
        total=num_restricted,
        desc="Building (A^T A)[I, I]",
    ):
        basis_vector[pixel_index] = 1.0
        basis_image = basis_vector.reshape(recon_shape)
        sinogram = ct_model.forward_project(basis_image)
        back_projection = ct_model.back_project(sinogram)
        back_projection_flat = np.asarray(back_projection).reshape(-1)
        at_a_restricted[:, col] = back_projection_flat[restricted_indices]
        basis_vector[pixel_index] = 0.0

    log_abs_at_a_restricted = np.log10(np.clip(np.abs(at_a_restricted), vmin, None))
    log_vmax = log_abs_at_a_restricted.max()

    # Plot A^T A for each of the subsets
    # For each partition, we'll map the first-subset indices into row/col positions of restricted_indices.
    # First get the dict from restricted indices into rows/cols of at_a_restricted
    restricted_position_map = {int(idx): pos for pos, idx in enumerate(restricted_indices)}
    for i, partition in enumerate(partitions):
        cur_subset_indices = np.asarray(partition[0]).flatten()
        keep_positions = np.array(
            [restricted_position_map[int(idx)] for idx in cur_subset_indices if int(idx) in restricted_position_map],
            dtype=int,
        )
        cur_at_a = log_abs_at_a_restricted[np.ix_(keep_positions, keep_positions)]
        print(
            f"Partition {i}: subset size={cur_subset_indices.size}, "
            f"kept size={keep_positions.size}, block shape={cur_at_a.shape}"
        )

        # Plot the full A^T A
        top_ax = grid[i]
        im = top_ax.imshow(
            cur_at_a,
            cmap='viridis',
            aspect='equal',
            vmin=log_vmin,
            vmax=log_vmax,
            extent=(0.0, 1.0, 0.0, 1.0),
            origin='upper',
            interpolation='nearest',
        )
        num_subsets = len(partition)
        # if num_subsets == 1:
        #     title = '{} subset\n'.format(num_subsets)
        # else:
        #     title = '{} subsets\n'.format(num_subsets)
        title = r'$A^T A$: '
        cur_num_rows = cur_at_a.shape[0]
        if cur_num_rows < 1000:
            title += '{} x {}'.format(cur_num_rows, cur_num_rows)
        else:
            if cur_num_rows < 10000:
                display_rows = np.round(cur_num_rows / 1000.0, decimals=1)
            else:
                display_rows = np.round(cur_num_rows / 1000.0).astype(int)
                title += '{}K x {}K'.format(display_rows, display_rows)
        top_ax.set_title(title)  # (r"Heatmap of $log |(A^T A)[I, I]|$")
        top_ax.axis("off")

        # Then the zoomed A^T A
        zoom_log_at_a = cur_at_a[:min_num_indices, :min_num_indices]
        bottom_ax = grid[ncols + i]
        bottom_ax.imshow(
            zoom_log_at_a,
            cmap='viridis',
            aspect='equal',
            vmin=log_vmin,
            vmax=log_vmax,
            extent=(0.0, 1.0, 0.0, 1.0),
            origin='upper',
            interpolation='nearest',
        )
        cur_num_rows = zoom_log_at_a.shape[0]
        title = '{} x {} corner'.format(cur_num_rows, cur_num_rows)
        bottom_ax.set_title(title)  # (r"Heatmap of $log |(A^T A)[I, I]|$")
        bottom_ax.axis("off")

    # Single colorbar spanning both rows.
    norm = mpl.colors.Normalize(vmin=log_vmin, vmax=log_vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap='viridis')
    sm.set_array([])
    cbar = grid.cbar_axes[0].colorbar(sm)
    cbar.ax.set_ylabel(r'$\log_{10}|A^T A|$')
    plt.show()
