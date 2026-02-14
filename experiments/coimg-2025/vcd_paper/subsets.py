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
    angles = np.linspace(0, np.pi, num=16)
    num_views = len(angles)
    sinogram_shape = (num_views, 1, recon_shape[0])
    ct_model = mj.ParallelBeamModel(sinogram_shape, angles)

    granularity = [1, 4, 16, 64]
    use_ror_mask = True
    use_grid = True
    if use_grid:
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
    #     fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
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

    min_num_indices = len(partitions[-1][0])
    vmin = 1e-2
    log_abs_at_a_restricted = np.log10(np.abs(at_a_restricted) + vmin)
    vmax = log_abs_at_a_restricted.max()
    print(f"Restricted index count: {num_restricted}")
    print(f"(A^T A)[I, I] shape: {at_a_restricted.shape}")
    print(f"log abs(M) min: {log_abs_at_a_restricted.min():.6g}")
    print(f"log abs(M) max: {log_abs_at_a_restricted.max():.6g}")

    # Plot A^T A for each of the subsets
    ncols = len(partitions)
    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(10, 5))
    # For each partition, we'll map the first-subset indices into row/col positions of restricted_indices.
    # First get the dict from restricted indices into rows/cols of at_a_restricted
    restricted_position_map = {int(idx): pos for pos, idx in enumerate(restricted_indices)}
    restricted_blocks = []
    for i, partition in enumerate(partitions):
        cur_subset_indices = np.asarray(partition[0]).flatten()
        keep_positions = np.array(
            [restricted_position_map[int(idx)] for idx in cur_subset_indices if int(idx) in restricted_position_map],
            dtype=int,
        )
        cur_restricted = at_a_restricted[np.ix_(keep_positions, keep_positions)]
        restricted_blocks.append(cur_restricted)
        print(
            f"Partition {i}: subset size={cur_subset_indices.size}, "
            f"kept size={keep_positions.size}, block shape={cur_restricted.shape}"
        )
        # if cur_restricted.shape[0] > min_num_indices:
        #     cur_restricted = cur_restricted[:min_num_indices, :min_num_indices]
        ax[i].imshow(cur_restricted, cmap='viridis', aspect='equal')
        ax[i].set_title(r"Heatmap of $log |(A^T A)[I, I]|$")
        ax[i].axis("off")
    # plt.colorbar(label=r"$log |M_{ij}|$")
    plt.tight_layout()
    plt.show()
