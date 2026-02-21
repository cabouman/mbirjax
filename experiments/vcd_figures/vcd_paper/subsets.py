"""
Visualize subset structure and restricted Gram blocks for a parallel-beam projector.

The script builds partitions of reconstruction pixels for multiple subset counts and
compares the corresponding restricted blocks of the normal operator A^T A.
It also computes a Fourier-conjugated version of the same restricted operator.
"""

from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import mbirjax as mj
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import tqdm


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the subset and restricted-operator visualization."""

    recon_shape: tuple[int, int, int] = (32, 32, 1)
    granularity: tuple[int, ...] = (1, 4, 16, 64, 128)
    use_grid_subsets: bool = True
    min_num_indices: int = 32
    font_size: int = 20
    vmin: float = 0.1
    grid_axes_pad: tuple[float, float] = (0.02, 0.5)
    grid_cbar_pad: float = 0.10
    grid_cbar_size: str = "2.5%"
    plot_fourier_conjugated: bool = True


def plot_partitions(partitions, recon_shape, grid):
    """
    Plot the partition labels for each subset scheme on the first ImageGrid row.

    Parameters
    ----------
    partitions : list
        Each entry is a partition; a partition is an iterable of 1D flat pixel indices.
    recon_shape : tuple[int, int, int]
        Reconstruction volume shape as (rows, cols, slices).
    grid : ImageGrid
        Target ImageGrid with at least ``len(partitions)`` axes in the first row.
    """
    num_recon_rows, num_recon_cols = recon_shape[:2]
    original_font_size = plt.rcParams["font.size"]
    plt.rcParams.update({"font.size": 24})

    for i, partition in enumerate(partitions):
        labels_image = np.zeros((num_recon_rows * num_recon_cols), dtype=int)
        for subset_index, indices in enumerate(partition):
            labels_image[np.asarray(indices).flatten()] = subset_index + 1
        labels_image = labels_image.reshape((num_recon_rows, num_recon_cols))

        ax = grid[i]
        ax.imshow(
            labels_image,
            cmap="nipy_spectral",
            aspect="equal",
            extent=(0.0, 1.0, 0.0, 1.0),
            origin="upper",
            interpolation="nearest",
        )
        subset_word = "subset" if len(partition) == 1 else "subsets"
        ax.set_title(f"{len(partition)} {subset_word}")
        ax.axis("off")

    plt.rcParams.update({"font.size": original_font_size})


def make_parallel_beam_model(recon_shape):
    """Create the parallel-beam model used by this experiment."""
    num_angles = recon_shape[0] // 2
    angles = np.linspace(0, np.pi, num=num_angles, endpoint=False)
    sinogram_shape = (len(angles), 1, recon_shape[0])
    return mj.ParallelBeamModel(sinogram_shape, angles)


def generate_partitions(recon_shape, granularity, use_grid_subsets):
    """Generate one partition per requested subset count."""
    gen_partition = mj.gen_pixel_partition_grid if use_grid_subsets else mj.gen_pixel_partition
    return [
        gen_partition(recon_shape, num_subsets, use_ror_mask=True)
        for num_subsets in granularity
    ]


def order_indices_center_out(indices, recon_shape):
    """
    Order flat indices by L1 distance from image center.

    This ordering is used to make low-frequency Fourier content appear near the
    top-left corner of matrix visualizations.
    """
    indices = np.asarray(indices).flatten()
    inds_2d = np.unravel_index(indices, recon_shape[:2])
    mid = (recon_shape[0] / 2, recon_shape[1] / 2)
    d_l1 = [np.abs(ind[0] - mid[0]) + np.abs(ind[1] - mid[1]) for ind in zip(*inds_2d)]
    sort_order = np.argsort(d_l1)
    return np.asarray(indices[sort_order], dtype=int)


def build_restricted_matrices(ct_model, recon_shape, restricted_indices):
    """
    Build restricted normal-operator matrices over index set I.

    Returns
    -------
    ata_restricted : ndarray
        Real-space restricted matrix (A^T A)[I, I].
    fourier_ata_restricted : ndarray
        Fourier-conjugated restricted matrix over the same index set.
    """
    num_restricted = restricted_indices.size
    num_pixels = int(np.prod(recon_shape))
    ata_restricted = np.zeros((num_restricted, num_restricted), dtype=np.float32)
    fourier_ata_restricted = np.zeros((num_restricted, num_restricted, 2), dtype=np.float32)

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
        ata_restricted[:, col] = back_projection_flat[restricted_indices]

        ifft_basis_image = np.fft.ifft2(np.fft.ifftshift(basis_image))
        real_ifft_basis = np.real(ifft_basis_image)
        imag_ifft_basis = np.imag(ifft_basis_image)

        real_fp_ifft = ct_model.forward_project(real_ifft_basis)
        imag_fp_ifft = ct_model.forward_project(imag_ifft_basis)

        real_bp_ifft = ct_model.back_project(real_fp_ifft)
        imag_bp_ifft = ct_model.back_project(imag_fp_ifft)
        bp_ifft = real_bp_ifft + 1j * imag_bp_ifft

        fourier_bp = np.fft.fftshift(np.fft.fft2(bp_ifft))
        fourier_bp_flat = np.asarray(fourier_bp).reshape(-1)
        fourier_bp_flat = fourier_bp_flat[restricted_indices]
        fourier_ata_restricted[:, col] = np.array([np.real(fourier_bp_flat), np.imag(fourier_bp_flat)]).T

        basis_vector[pixel_index] = 0.0

    return ata_restricted, fourier_ata_restricted


def format_matrix_size(n):
    """Format matrix dimension for subplot titles."""
    if n < 1000:
        return f"{n} x {n}"
    if n < 10000:
        display_rows = np.round(n / 1000.0, decimals=1)
    else:
        display_rows = np.round(n / 1000.0).astype(int)
    return f"{display_rows}K x {display_rows}K"


def plot_restricted_blocks(grid, partitions, log_abs_matrix, restricted_indices, ncols, cfg):
    """Plot full and zoomed restricted blocks for each partition."""
    log_vmin = np.log10(cfg.vmin)
    log_vmax = float(log_abs_matrix.max())

    restricted_position_map = {int(idx): pos for pos, idx in enumerate(restricted_indices)}
    for i, partition in enumerate(partitions):
        cur_subset_indices = np.asarray(partition[0]).flatten()
        keep_positions = np.array(
            [restricted_position_map[int(idx)] for idx in cur_subset_indices if int(idx) in restricted_position_map],
            dtype=int,
        )
        cur_block = log_abs_matrix[np.ix_(keep_positions, keep_positions)]
        # num_inds = len(restricted_indices)
        # full_matrix = - np.ones((num_inds, num_inds), dtype=cur_block.dtype)
        # full_matrix[np.ix_(keep_positions, keep_positions)] = cur_block
        # cur_block = full_matrix
        print(
            f"Partition {i}: subset size={cur_subset_indices.size}, "
            f"kept size={keep_positions.size}, block shape={cur_block.shape}"
        )

        top_ax = grid[ncols + i]
        top_ax.imshow(
            cur_block,
            cmap="viridis",
            aspect="equal",
            vmin=log_vmin,
            vmax=log_vmax,
            extent=(0.0, 1.0, 0.0, 1.0),
            origin="upper",
            interpolation="nearest",
        )
        top_ax.set_title(r"$A^T A$: " + format_matrix_size(cur_block.shape[0]))
        top_ax.axis("off")

        zoom_log_block = cur_block[: cfg.min_num_indices, : cfg.min_num_indices]
        bottom_ax = grid[2 * ncols + i]
        bottom_ax.imshow(
            zoom_log_block,
            cmap="viridis",
            aspect="equal",
            vmin=log_vmin,
            vmax=log_vmax,
            extent=(0.0, 1.0, 0.0, 1.0),
            origin="upper",
            interpolation="nearest",
        )
        bottom_ax.set_title(f"{zoom_log_block.shape[0]} x {zoom_log_block.shape[0]} corner")
        bottom_ax.axis("off")

    norm = mpl.colors.Normalize(vmin=log_vmin, vmax=log_vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    cbar = grid.cbar_axes[0].colorbar(sm)
    cbar.ax.set_ylabel(r"$\log_{10}|A^T A|$")


def main():
    """Run the subset-structure and restricted-operator experiment."""
    cfg = ExperimentConfig()

    plt.rcParams.update({"font.size": cfg.font_size})
    fig = plt.figure(figsize=(4 * len(cfg.granularity), 15))
    ct_model = make_parallel_beam_model(cfg.recon_shape)
    partitions = generate_partitions(cfg.recon_shape, cfg.granularity, cfg.use_grid_subsets)
    ncols = len(partitions)

    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(3, ncols),
        axes_pad=cfg.grid_axes_pad,
        share_all=False,
        aspect=False,
        cbar_mode="single",
        cbar_location="right",
        cbar_pad=cfg.grid_cbar_pad,
        cbar_size=cfg.grid_cbar_size,
    )

    plot_partitions(partitions=partitions, recon_shape=cfg.recon_shape, grid=grid)

    # Reference index set I is the first subset from the 1-subset partition.
    restricted_indices = order_indices_center_out(partitions[0][0], cfg.recon_shape)
    ata_restricted, fourier_ata_restricted = build_restricted_matrices(
        ct_model, cfg.recon_shape, restricted_indices
    )
    matrix_to_plot = None
    if cfg.plot_fourier_conjugated:
        matrix_to_plot = fourier_ata_restricted[:, :, 0] ** 2 + fourier_ata_restricted[:, :, 1] ** 2
        matrix_to_plot = np.sqrt(matrix_to_plot)
    else:
        matrix_to_plot = ata_restricted
    log_abs_matrix = np.log10(np.clip(np.abs(matrix_to_plot), cfg.vmin, None))
    plot_restricted_blocks(
        grid=grid,
        partitions=partitions,
        log_abs_matrix=log_abs_matrix,
        restricted_indices=restricted_indices,
        ncols=ncols,
        cfg=cfg,
    )
    plt.show()


if __name__ == "__main__":
    main()
