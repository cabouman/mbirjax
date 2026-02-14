import numpy as np
import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
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
    plt.rcParams.update({'font.size': 24})  # Adjust font size here
    num_partitions = len(partitions)
    fig, axes = plt.subplots(nrows=1, ncols=num_partitions, figsize=(5 * num_partitions, 5))

    for i, partition in enumerate(partitions):
        # Create an empty image array to fill with subset colors
        image = np.zeros((num_recon_rows * recon_shape), dtype=int)

        # Assign a unique color (integer label) to each subset
        for subset_index, indices in enumerate(partition):
            image[indices.flatten()] = subset_index + 1  # Color code starts from 1 upwards

        # Reshape the image array back to 2D format
        image = image.reshape((num_recon_rows, recon_shape))

        # Plotting
        if num_partitions == 1:
            ax = axes
        else:
            ax = axes[i]

        cax = ax.imshow(image, cmap='nipy_spectral', interpolation='nearest')
        if len(partition) == 1:
            ax.set_title(f'{len(partition)} Subset')
        else:
            ax.set_title(f'{len(partition)} Subsets')
        ax.axis('off')  # Turn off axis labels

    plt.tight_layout()
    plt.show()


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
    use_grid_subsets = False
    plot_full_images = False
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
    # mj.debug_plot_partitions(partitions=partitions, recon_shape=recon_shape)

    # for i, partition in enumerate(partitions):
    #     # Create an empty image array to fill with subset colors
    #     subset = np.zeros((recon_shape[0] * recon_shape[1]), dtype=int)
    #
    #     # Assign nonzeros to this subset
    #     indices = partition[0]
    #     subset[indices.flatten()] = 1.0
    #
    #     # Reshape the image array back to 2D format
    #     subset = subset.reshape(recon_shape)
    #
    #     sinogram = ct_model.forward_project(subset)
    #     back_projection = ct_model.back_project(sinogram)
    #
    #     sinogram /= sinogram.max()
    #     back_projection /= back_projection.max()
    #
    #     _, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    #     ax[0].imshow(subset[:, :, 0])
    #     ax[0].set_title(r"Subset, $S$")
    #     ax[0].axis('off')
    #
    #     ax[1].imshow(back_projection[:, :, 0])
    #     ax[1].set_title(r"$A^T A S$")
    #     ax[1].axis('off')
    #
    #     plt.tight_layout()
    #     plt.show()

    vmin = 0.1
    log_vmin = np.log10(vmin)
    log_vmax = None
    # Plot A^T A for each of the subsets
    ncols = len(partitions)
    fig, ax = plt.subplots(nrows=2, ncols=ncols, figsize=(23, 10))

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
        im = ax[0, i].imshow(cur_at_a, cmap='viridis', aspect='equal', vmin=log_vmin, vmax=log_vmax)
        title = '# subsets = {}\n'.format(len(partitions[i]))
        title += r'$A^T A$: '
        cur_num_rows = cur_at_a.shape[0]
        title += '{} x {}'.format(cur_num_rows, cur_num_rows)
        ax[0, i].set_title(title)  # (r"Heatmap of $log |(A^T A)[I, I]|$")
        ax[0, i].axis("off")

        # Then the zoomed A^T A
        zoom_log_at_a = cur_at_a[:min_num_indices, :min_num_indices]
        ax[1, i].imshow(zoom_log_at_a, cmap='viridis', aspect='equal', vmin=log_vmin, vmax=log_vmax)
        title = r'$A^T A$ zoom: '
        cur_num_rows = zoom_log_at_a.shape[0]
        title += '{} x {}'.format(cur_num_rows, cur_num_rows)
        ax[1, i].set_title(title)  # (r"Heatmap of $log |(A^T A)[I, I]|$")
        ax[1, i].axis("off")



    # # For each partition, we'll map the first-subset indices into row/col positions of restricted_indices.
    # # First get the dict from restricted indices into rows/cols of at_a_restricted
    # for i, partition in enumerate(partitions):
    #     cur_subset_indices = np.asarray(partition[0]).flatten()
    #     if not plot_full_images:
    #         cur_subset_indices = cur_subset_indices[:min_num_indices]
    #     # Build restricted dense matrix M = (A^T A)[I, I], where I = partitions[0][0].
    #     # Each column is A^T A applied to a basis vector e_j in the restricted index set.
    #     restricted_indices = cur_subset_indices
    #     num_restricted = restricted_indices.size
    #     num_pixels = int(np.prod(recon_shape))
    #     at_a_restricted = np.zeros((num_restricted, num_restricted), dtype=np.float32)
    #
    #     basis_vector = np.zeros(num_pixels, dtype=np.float32)
    #     for col, pixel_index in tqdm(
    #         enumerate(restricted_indices),
    #         total=num_restricted,
    #         desc="Building (A^T A)[I, I]",
    #     ):
    #         basis_vector[pixel_index] = 1.0
    #         basis_image = basis_vector.reshape(recon_shape)
    #         sinogram = ct_model.forward_project(basis_image)
    #         back_projection = ct_model.back_project(sinogram)
    #         back_projection_flat = np.asarray(back_projection).reshape(-1)
    #         at_a_restricted[:, col] = back_projection_flat[restricted_indices]
    #         basis_vector[pixel_index] = 0.0
    #     log_abs_at_a_restricted = np.log10(np.clip(np.abs(at_a_restricted), vmin, None))
    #     if log_vmax is None:
    #         log_vmax = log_abs_at_a_restricted.max()
    #     print(f"Restricted index count: {num_restricted}")
    #     print(f"(A^T A)[I, I] shape: {at_a_restricted.shape}")
    #     print(f"log abs(M) min: {log_abs_at_a_restricted.min():.6g}")
    #     print(f"log abs(M) max: {log_abs_at_a_restricted.max():.6g}")
    #
    #     im = ax[i].imshow(log_abs_at_a_restricted, cmap='viridis', aspect='equal', vmin=log_vmin, vmax=log_vmax)
    #     title = '# subsets = {}\n'.format(len(partitions[i]))
    #     if plot_full_images:
    #         title += r'$A^T A$: '
    #     else:
    #         title += r'$A^T A$ zoom: '
    #     cur_num_rows = log_abs_at_a_restricted.shape[0]
    #     title += '{} x {}'.format(cur_num_rows, cur_num_rows)
    #     ax[i].set_title(title)  # (r"Heatmap of $log |(A^T A)[I, I]|$")
    #     ax[i].axis("off")
    # plt.colorbar(label=r"$log |M_{ij}|$")
    # # Single colorbar for the whole figure:
    import matplotlib as mpl
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    norm = mpl.colors.Normalize(vmin=log_vmin, vmax=log_vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap='viridis')
    sm.set_array([])

    # Force colorbar to use all subplot axes as one group
    cbar = fig.colorbar(
        sm,  # or im from first imshow
        ax=ax.ravel().tolist(),
        location="right",
        pad=0.02
    )
    # plt.tight_layout()
    plt.show()
