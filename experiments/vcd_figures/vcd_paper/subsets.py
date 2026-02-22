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

    recon_shape: tuple[int, int, int] = (32, 32, 1)
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
    ct_model, recon_shape, restricted_indices, inverse_hessian_diagonal=None, use_preconditioner=True
):
    """
    Build restricted normal-operator matrices over index set I.

    Returns
    -------
    ata_restricted : ndarray
        Real-space restricted matrix ((D^{-1}) A^T A)[I, I] if enabled; otherwise (A^T A)[I, I].
    """
    num_restricted = restricted_indices.size
    num_pixels = int(np.prod(recon_shape))
    ata_restricted = np.zeros((num_restricted, num_restricted), dtype=np.float32)

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
        ata_col = back_projection_flat[restricted_indices]
        if use_preconditioner:
            if inverse_hessian_diagonal is None:
                raise ValueError("inverse_hessian_diagonal is required when use_preconditioner=True.")
            ata_col = ata_col * inverse_hessian_diagonal[restricted_indices]
        ata_restricted[:, col] = ata_col

        basis_vector[pixel_index] = 0.0

    return ata_restricted


def build_fourier_response_abs_matrix(
    ct_model,
    recon_shape,
    subset_indices,
    frequency_indices,
    inverse_hessian_diagonal=None,
    use_preconditioner=True,
):
    """
    Build |F P (D^{-1}) A^T A P F^{-1}| over full-frequency rows/cols using one spatial subset.

    Parameters
    ----------
    subset_indices : ndarray
        Flat spatial indices defining subset projector P.
    frequency_indices : ndarray
        Flat shifted-FFT indices defining row/column ordering.
    """
    num_rows, num_cols = recon_shape[:2]
    num_freq = num_rows * num_cols
    subset_indices = np.asarray(subset_indices, dtype=int).reshape(-1)
    frequency_indices = np.asarray(frequency_indices, dtype=int).reshape(-1)
    if subset_indices.size == 0:
        raise ValueError("subset_indices must be non-empty.")

    subset_indices_jnp = jnp.asarray(subset_indices)
    frequency_indices_jnp = jnp.asarray(frequency_indices)
    inverse_hessian_subset = None
    if use_preconditioner:
        inverse_hessian_subset = jnp.asarray(inverse_hessian_diagonal[subset_indices])[:, None]

    fourier_response_abs = jnp.zeros((num_freq, num_freq), dtype=jnp.float32)
    freq_basis_flat = jnp.zeros(num_freq, dtype=jnp.complex64)

    for col, freq_index in tqdm(
        enumerate(frequency_indices),
        total=num_freq,
        desc="Building |F P A^T A P F^{-1}|",
    ):
        freq_basis_flat = freq_basis_flat.at[freq_index].set(1.0 + 0.0j)
        freq_basis_2d = freq_basis_flat.reshape((num_rows, num_cols))
        spatial_basis_2d = jnp.fft.ifft2(
            jnp.fft.ifftshift(freq_basis_2d, axes=(0, 1)),
            axes=(0, 1),
            norm="ortho",
        )

        real_spatial_subset = jnp.take(jnp.real(spatial_basis_2d).reshape(-1), subset_indices_jnp)[:, None]
        imag_spatial_subset = jnp.take(jnp.imag(spatial_basis_2d).reshape(-1), subset_indices_jnp)[:, None]

        real_sinogram = ct_model.sparse_forward_project(
            real_spatial_subset, subset_indices_jnp, output_device=ct_model.sinogram_device
        )
        imag_sinogram = ct_model.sparse_forward_project(
            imag_spatial_subset, subset_indices_jnp, output_device=ct_model.sinogram_device
        )

        real_back_subset = ct_model.sparse_back_project(
            real_sinogram, subset_indices_jnp, output_device=ct_model.main_device
        )
        imag_back_subset = ct_model.sparse_back_project(
            imag_sinogram, subset_indices_jnp, output_device=ct_model.main_device
        )
        complex_back_subset = real_back_subset + 1j * imag_back_subset
        if use_preconditioner:
            if inverse_hessian_diagonal is None:
                raise ValueError("inverse_hessian_diagonal is required when use_preconditioner=True.")
            complex_back_subset = complex_back_subset * inverse_hessian_subset

        back_projection_flat = jnp.zeros(num_freq, dtype=jnp.complex64)
        back_projection_flat = back_projection_flat.at[subset_indices_jnp].set(complex_back_subset[:, 0])
        back_projection_2d = back_projection_flat.reshape((num_rows, num_cols))

        fourier_response_2d = jnp.fft.fftshift(
            jnp.fft.fft2(back_projection_2d, axes=(0, 1), norm="ortho"),
            axes=(0, 1),
        )
        fourier_response_col = jnp.take(fourier_response_2d.reshape(-1), frequency_indices_jnp)
        fourier_response_abs = fourier_response_abs.at[:, col].set(jnp.abs(fourier_response_col).astype(jnp.float32))

        freq_basis_flat = freq_basis_flat.at[freq_index].set(0.0 + 0.0j)

    return fourier_response_abs


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
                ct_model=ct_model,
                recon_shape=cfg.recon_shape,
                subset_indices=subset_indices,
                frequency_indices=frequency_indices,
                inverse_hessian_diagonal=inverse_hessian_diagonal,
                use_preconditioner=cfg.use_preconditioner,
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
            ct_model,
            cfg.recon_shape,
            restricted_indices,
            inverse_hessian_diagonal=inverse_hessian_diagonal,
            use_preconditioner=cfg.use_preconditioner,
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
