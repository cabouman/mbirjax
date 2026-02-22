"""
Visualize subset structure and restricted Gram blocks for a parallel-beam projector.

The script builds partitions of reconstruction pixels for multiple subset counts and
compares the corresponding restricted blocks of the normal operator A^T A.
It also computes a Fourier-conjugated version of the same restricted operator.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import mbirjax as mj
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import tqdm


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the subset and restricted-operator visualization."""

    recon_shape: tuple[int, int, int] = (64, 64, 1)
    granularity: tuple[int, ...] = (1, 4, 16, 64, 128)
    use_grid_subsets: bool = True
    use_ror_mask: bool = True
    min_num_indices: int = 32
    font_size: int = 20
    vmin: float = 0.1
    grid_axes_pad: tuple[float, float] = (0.02, 0.5)
    grid_cbar_pad: float = 0.10
    grid_cbar_size: str = "2.5%"
    plot_fourier_conjugated: bool = True
    use_preconditioner: bool = True
    batch_size: int = 2048


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
    sinogram_shape = (len(angles), recon_shape[2], recon_shape[0])
    return mj.ParallelBeamModel(sinogram_shape, angles)


def generate_partitions(recon_shape, granularity, use_grid_subsets, use_ror_mask):
    """Generate one partition per requested subset count."""
    gen_partition = mj.gen_pixel_partition_grid if use_grid_subsets else mj.gen_pixel_partition
    return [
        gen_partition(recon_shape, num_subsets, use_ror_mask=use_ror_mask)
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


def order_frequency_indices_radial(recon_shape):
    """Order 2D shifted-FFT frequencies by radial distance from DC."""
    num_rows, num_cols = recon_shape[:2]
    row_coords, col_coords = np.indices((num_rows, num_cols))
    row_center = num_rows / 2.0
    col_center = num_cols / 2.0
    radial_dist = np.sqrt((row_coords - row_center) ** 2 + (col_coords - col_center) ** 2).reshape(-1)
    flat_indices = np.arange(num_rows * num_cols, dtype=int)
    sort_order = np.lexsort((flat_indices, radial_dist))
    return flat_indices[sort_order]


def compute_inverse_hessian_diagonal(ct_model):
    """Compute a numerically safe inverse Hessian diagonal over the full recon grid."""
    hessian_diagonal = np.asarray(ct_model.compute_hessian_diagonal()).reshape(-1).astype(np.float32)
    safe_hessian = np.maximum(hessian_diagonal, np.finfo(np.float32).eps)
    return 1.0 / safe_hessian


def build_restricted_matrices(
    recon_shape,
    restricted_indices,
    inverse_hessian_diagonal=None,
    use_preconditioner=True,
    batch_size=2048,
):
    """
    Build restricted matrices in one stacked projection/backprojection call.

    Each restricted basis vector is placed in a reconstruction slice. Slices are
    processed in batches to control memory.
    """
    num_rows, num_cols = recon_shape[:2]
    restricted_indices = np.asarray(restricted_indices, dtype=int).reshape(-1)
    num_restricted = restricted_indices.size

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    ata_restricted = np.zeros((num_restricted, num_restricted), dtype=np.float32)
    row_inds_all, col_inds_all = np.unravel_index(restricted_indices, (num_rows, num_cols))

    for start in tqdm(range(0, num_restricted, batch_size), desc="Building (A^T A)[I, I] batched"):
        end = min(start + batch_size, num_restricted)
        cur_batch_size = end - start
        cur_indices = restricted_indices[start:end]

        stacked_recon_shape = (num_rows, num_cols, cur_batch_size)
        stacked_model = make_parallel_beam_model(stacked_recon_shape)

        basis_stack = np.zeros(stacked_recon_shape, dtype=np.float32)
        basis_stack[row_inds_all[start:end], col_inds_all[start:end], np.arange(cur_batch_size)] = 1.0

        sinogram = stacked_model.forward_project(basis_stack)
        back_projection = stacked_model.back_project(sinogram)
        back_projection_flat = np.asarray(back_projection).reshape(num_rows * num_cols, cur_batch_size)
        ata_restricted[:, start:end] = back_projection_flat[restricted_indices, :]

    if use_preconditioner:
        if inverse_hessian_diagonal is None:
            raise ValueError("inverse_hessian_diagonal is required when use_preconditioner=True.")
        ata_restricted = ata_restricted * inverse_hessian_diagonal[restricted_indices][:, None]

    return np.asarray(ata_restricted, dtype=np.float32)


def build_fourier_response_abs_matrix(
    recon_shape,
    subset_indices,
    frequency_indices,
    inverse_hessian_diagonal=None,
    use_preconditioner=True,
    batch_size=2048,
):
    """
    Build |F P (D^{-1}) A^T A P F^{-1}| by stacking frequency inputs as slices.

    Frequency columns are processed in batches to control memory.
    """
    num_rows, num_cols = recon_shape[:2]
    num_freq = num_rows * num_cols
    subset_indices = np.asarray(subset_indices, dtype=int).reshape(-1)
    frequency_indices = np.asarray(frequency_indices, dtype=int).reshape(-1)
    if subset_indices.size == 0:
        raise ValueError("subset_indices must be non-empty.")
    if frequency_indices.size != num_freq:
        raise ValueError("frequency_indices must include all frequencies.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    subset_mask = jnp.zeros((num_rows * num_cols,), dtype=jnp.float32).at[subset_indices].set(1.0)
    subset_mask_2d = subset_mask.reshape((num_rows, num_cols))
    freq_order_jnp = jnp.asarray(frequency_indices)
    inverse_hessian_2d = None
    if use_preconditioner:
        if inverse_hessian_diagonal is None:
            raise ValueError("inverse_hessian_diagonal is required when use_preconditioner=True.")
        inverse_hessian_2d = jnp.asarray(inverse_hessian_diagonal.reshape((num_rows, num_cols)))

    response_abs = jnp.zeros((num_freq, num_freq), dtype=jnp.float32)
    for start in tqdm(range(0, num_freq, batch_size), desc="Building Fourier response batched"):
        end = min(start + batch_size, num_freq)
        cur_batch_size = end - start
        cur_freq_indices = frequency_indices[start:end]

        stacked_recon_shape = (num_rows, num_cols, cur_batch_size)
        stacked_model = make_parallel_beam_model(stacked_recon_shape)

        freq_basis_stack = jnp.zeros(stacked_recon_shape, dtype=jnp.complex64)
        freq_rows, freq_cols = np.unravel_index(cur_freq_indices, (num_rows, num_cols))
        freq_basis_stack = freq_basis_stack.at[freq_rows, freq_cols, jnp.arange(cur_batch_size)].set(1.0 + 0.0j)

        spatial_basis_stack = jnp.fft.ifft2(
            jnp.fft.ifftshift(freq_basis_stack, axes=(0, 1)),
            axes=(0, 1),
            norm="ortho",
        )

        real_spatial_stack = jnp.real(spatial_basis_stack) * subset_mask_2d[..., None]
        imag_spatial_stack = jnp.imag(spatial_basis_stack) * subset_mask_2d[..., None]

        real_sinogram = stacked_model.forward_project(real_spatial_stack)
        imag_sinogram = stacked_model.forward_project(imag_spatial_stack)
        real_back_stack = stacked_model.back_project(real_sinogram)
        imag_back_stack = stacked_model.back_project(imag_sinogram)
        complex_back_stack = real_back_stack + 1j * imag_back_stack

        if use_preconditioner:
            complex_back_stack = complex_back_stack * inverse_hessian_2d[..., None]

        complex_back_stack = complex_back_stack * subset_mask_2d[..., None]

        fourier_response_stack = jnp.fft.fftshift(
            jnp.fft.fft2(complex_back_stack, axes=(0, 1), norm="ortho"),
            axes=(0, 1),
        )

        fourier_response_flat = fourier_response_stack.reshape((num_freq, cur_batch_size))
        ordered_rows = jnp.take(fourier_response_flat, freq_order_jnp, axis=0)
        response_abs = response_abs.at[:, start:end].set(jnp.abs(ordered_rows).astype(jnp.float32))

    return response_abs


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
    cbar_label = r"$\log_{10}|(D^{-1})A^T A|$" if cfg.use_preconditioner else r"$\log_{10}|A^T A|$"
    cbar.ax.set_ylabel(cbar_label)


def plot_full_response_blocks(grid, partitions, log_abs_matrices, ncols, cfg, cbar_label, start_row=1):
    """Plot full and zoomed response matrices (already log-scaled)."""
    log_vmin = np.log10(cfg.vmin)
    log_vmax = max(float(log_abs_matrix.max()) for log_abs_matrix in log_abs_matrices)

    for i, (partition, log_abs_matrix) in enumerate(zip(partitions, log_abs_matrices)):
        m = np.round(np.sqrt(partition.shape[0])).astype(int)
        top_ax = grid[start_row * ncols + i]
        top_ax.imshow(
            log_abs_matrix,
            cmap="viridis",
            aspect="equal",
            vmin=log_vmin,
            vmax=log_vmax,
            extent=(0.0, 1.0, 0.0, 1.0),
            origin="upper",
            interpolation="nearest",
        )
        top_ax.set_title(f"{m} x {m} stride")
        top_ax.axis("off")

        zoom_log_block = log_abs_matrix[: cfg.min_num_indices, : cfg.min_num_indices]
        bottom_ax = grid[(start_row + 1) * ncols + i]
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
    cbar.ax.set_ylabel(cbar_label)


def main():
    """Run the subset-structure and restricted-operator experiment."""
    cfg = ExperimentConfig()
    is_fourier_mode = cfg.plot_fourier_conjugated

    plt.rcParams.update({"font.size": cfg.font_size})
    fig_height = 10 if is_fourier_mode else 15
    fig = plt.figure(figsize=(4 * len(cfg.granularity), fig_height))
    ct_model = make_parallel_beam_model(cfg.recon_shape)
    inverse_hessian_diagonal = None
    if cfg.use_preconditioner:
        inverse_hessian_diagonal = compute_inverse_hessian_diagonal(ct_model)
    partitions = generate_partitions(cfg.recon_shape, cfg.granularity, cfg.use_grid_subsets, cfg.use_ror_mask)
    ncols = len(partitions)

    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(2 if is_fourier_mode else 3, ncols),
        axes_pad=cfg.grid_axes_pad,
        share_all=False,
        aspect=False,
        cbar_mode="single",
        cbar_location="right",
        cbar_pad=cfg.grid_cbar_pad,
        cbar_size=cfg.grid_cbar_size,
    )

    if not is_fourier_mode:
        plot_partitions(partitions=partitions, recon_shape=cfg.recon_shape, grid=grid)

    # Reference index set I is the first subset from the 1-subset partition.
    if cfg.plot_fourier_conjugated:
        frequency_indices = order_frequency_indices_radial(cfg.recon_shape)
        log_abs_matrices = []
        for partition in partitions:
            subset_indices = np.asarray(partition[0]).flatten()
            response_abs = build_fourier_response_abs_matrix(
                recon_shape=cfg.recon_shape,
                subset_indices=subset_indices,
                frequency_indices=frequency_indices,
                inverse_hessian_diagonal=inverse_hessian_diagonal,
                use_preconditioner=cfg.use_preconditioner,
                batch_size=cfg.batch_size,
            )
            # Normalize each response to a common peak so subset-size trends are shape-based, not scale-based.
            response_abs = response_abs / jnp.maximum(jnp.max(response_abs), jnp.finfo(jnp.float32).eps)
            log_abs_matrices.append(np.asarray(jnp.log10(jnp.clip(response_abs, cfg.vmin, None))))
        plot_full_response_blocks(
            grid=grid,
            partitions=partitions,
            log_abs_matrices=log_abs_matrices,
            ncols=ncols,
            cfg=cfg,
            cbar_label=(
                r"$\log_{10}|F P (D^{-1}) A^T A P F^{-1}|$"
                if cfg.use_preconditioner
                else r"$\log_{10}|F P A^T A P F^{-1}|$"
            ),
            start_row=0,
        )
    else:
        restricted_indices = order_indices_center_out(partitions[0][0], cfg.recon_shape)
        ata_restricted = build_restricted_matrices(
            cfg.recon_shape,
            restricted_indices,
            inverse_hessian_diagonal=inverse_hessian_diagonal,
            use_preconditioner=cfg.use_preconditioner,
            batch_size=cfg.batch_size,
        )
        log_abs_matrix = np.log10(np.clip(np.abs(ata_restricted), cfg.vmin, None))
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
