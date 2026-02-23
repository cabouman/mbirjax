"""
One-sweep spectral response experiment for VCD paper figures.

This script:
1) Builds a real image whose shifted 2D Fourier magnitude is flat (random phase).
2) Generates sinogram data from that image.
3) Runs exactly one VCD sweep for each partition size in `granularity`.
4) Plots |FFT(recon_after_one_sweep)| maps and radial gain curves.

The reconstruction updates come from `ParallelBeamModel.recon`, so forward/prior
terms and step-size logic match the production reconstruction path.
"""

from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import warnings

try:
    from .subsets import make_parallel_beam_model
except ImportError:
    from subsets import make_parallel_beam_model

num_iterations = 4
granularity_list = (1, 4, 16, 64, 128)
gd_seq, gd_label = [num_iterations * [0], f'1 subset, {num_iterations} iterations']
icd_seq, icd_label = [num_iterations * [4], f'128 subsets, {num_iterations} iterations']
vcd_seq, vcd_label = (0, 1, 2, 3), 'VCD'


@dataclass(frozen=True)
class SweepConfig:
    recon_shape: tuple[int, int, int] = (1000, 1000, 1)
    granularity: tuple[int, ...] = granularity_list
    experiment_sequences: tuple[tuple[int, ...], ...] | None = (gd_seq, icd_seq, vcd_seq)
    experiment_labels: tuple[str, ...] | None = (gd_label, icd_label, vcd_label)
    random_phase_seed: int = 0
    partition_seed: int = 0
    map_vmin: float = 1e-2
    map_vmax: float = 1.0
    radial_vmin: float = 1e-3
    radial_bin_width: int = 4
    radial_skip_bins: int = 2
    gt_support_threshold_fraction: float = 1e-2
    normalize_maps_by_peak: bool = True
    use_loaded_image: bool = True
    use_ror_mask: bool = False
    radial_full_circle_only: bool = True
    font_size: int = 16

    # ==================== CHECKPOINT TOGGLES ====================
    # Set `enable_checkpoints=False` to disable all intermediate plots.
    enable_checkpoints: bool = False
    checkpoint_show_target_image: bool = False
    checkpoint_show_target_fft_mag: bool = False
    checkpoint_show_sinogram: bool = False
    checkpoint_show_first_recon: bool = False
    checkpoint_show_first_recon_fft: bool = False
    # ============================================================


def generate_flat_magnitude_target(recon_shape, seed):
    """
    Generate a real target image with approximately flat shifted-FFT magnitude.

    A random real seed image is transformed to Fourier domain and phase-only
    normalized so |FFT| is ~1 everywhere, preserving Hermitian symmetry.
    """
    num_rows, num_cols, num_slices = recon_shape
    if num_slices != 1:
        raise ValueError("This script currently expects recon_shape[2] == 1.")

    rng = np.random.default_rng(seed)
    seed_image = rng.standard_normal((num_rows, num_cols)).astype(np.float32)
    seed_fft = np.fft.fftshift(np.fft.fft2(seed_image, norm="ortho"))
    flat_fft = seed_fft / np.maximum(np.abs(seed_fft), np.finfo(np.float32).eps)
    target_2d = np.real(np.fft.ifft2(np.fft.ifftshift(flat_fft), norm="ortho")).astype(np.float32)
    target = target_2d[..., None]
    return target, flat_fft


def compute_shifted_fft_magnitude(image_2d):
    """Return shifted 2D FFT magnitude."""
    fft_2d = np.fft.fftshift(np.fft.fft2(image_2d, norm="ortho"))
    return np.abs(fft_2d)


def compute_radial_profile(image_2d, full_circle_only=True, bin_width=1, skip_bins=0, valid_mask=None):
    """Compute radial mean profile of a 2D image centered at DC (shifted FFT layout)."""
    num_rows, num_cols = image_2d.shape
    row_coords, col_coords = np.indices((num_rows, num_cols))
    row_center = num_rows // 2
    col_center = num_cols // 2
    radii = np.sqrt((row_coords - row_center) ** 2 + (col_coords - col_center) ** 2)
    bin_indices = np.floor(radii).astype(int)

    if full_circle_only:
        max_full_radius = min(
            row_center,
            num_rows - 1 - row_center,
            col_center,
            num_cols - 1 - col_center,
        )
        max_bin = int(max_full_radius)
        valid = bin_indices <= max_bin
    else:
        max_bin = int(bin_indices.max())
        valid = np.ones_like(bin_indices, dtype=bool)

    if valid_mask is not None:
        valid = valid & valid_mask.astype(bool)

    if bin_width <= 0:
        raise ValueError("bin_width must be >= 1.")
    if skip_bins < 0:
        raise ValueError("skip_bins must be >= 0.")

    binned_vals = (bin_indices[valid] // bin_width).ravel()
    img_vals = image_2d[valid].ravel()

    max_coarse_bin = int(max_bin // bin_width)
    sums = np.bincount(binned_vals, weights=img_vals, minlength=max_coarse_bin + 1)
    counts = np.bincount(binned_vals, minlength=max_coarse_bin + 1)
    profile = sums / np.maximum(counts, 1)
    radius_values = (np.arange(max_coarse_bin + 1, dtype=np.float32) + 0.5) * float(bin_width)
    radius_values /= max(float(max_bin), 1.0)
    if skip_bins > 0:
        if skip_bins >= profile.size:
            raise ValueError("skip_bins is too large for the computed number of radial bins.")
        profile = profile[skip_bins:]
        radius_values = radius_values[skip_bins:]
    return radius_values, profile.astype(np.float32)


def run_recon_for_sequence(model, sinogram, partition_sequence):
    """Run reconstruction for a specified partition-index sequence."""
    model.set_params(partition_sequence=list(partition_sequence))
    max_iterations = len(partition_sequence)
    recon, recon_dict = model.recon(
        sinogram,
        init_recon=0,
        max_iterations=max_iterations,
        first_iteration=0,
        stop_threshold_change_pct=0.0,
        compute_prior_loss=False,
        print_logs=False,
    )
    return np.asarray(recon), recon_dict


def maybe_show_checkpoint(cfg, condition, plot_fn):
    """Execute a plotting callback only when checkpoint toggles are enabled."""
    if cfg.enable_checkpoints and condition:
        plot_fn()


def plot_summary(gt_image, recon_images, gt_freq_map, freq_maps, radial_profiles, radii, labels, cfg):
    """Plot spatial images, Fourier maps, and radial curves."""
    n_experiments = len(freq_maps)
    ncols = n_experiments + 1  # Extra left column for ground truth.
    fig = plt.figure(figsize=(4 * ncols, 11))
    grid = fig.add_gridspec(3, ncols, height_ratios=[1.0, 1.0, 0.9])
    norm = mpl.colors.LogNorm(vmin=cfg.map_vmin, vmax=cfg.map_vmax)

    # Top row: spatial images.
    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(gt_image, cmap="gray", origin="upper", interpolation="nearest")
    ax.set_title("GT image")
    ax.axis("off")
    for i, (recon_image, label) in enumerate(zip(recon_images, labels), start=1):
        ax = fig.add_subplot(grid[0, i])
        ax.imshow(recon_image, cmap="gray", origin="upper", interpolation="nearest")
        ax.set_title(label)
        ax.axis("off")

    # Middle row: Fourier magnitude maps.
    ax = fig.add_subplot(grid[1, 0])
    ax.imshow(
        np.clip(gt_freq_map, cfg.map_vmin, None),
        cmap="viridis",
        norm=norm,
        origin="upper",
        interpolation="nearest",
    )
    ax.set_title("GT |F image|")
    ax.axis("off")
    for i, (freq_map, label) in enumerate(zip(freq_maps, labels), start=1):
        ax = fig.add_subplot(grid[1, i])
        ax.imshow(
            np.clip(freq_map, cfg.map_vmin, None),
            cmap="viridis",
            norm=norm,
            origin="upper",
            interpolation="nearest",
        )
        ax.set_title(label)
        ax.axis("off")

    cax = fig.add_axes([0.92, 0.34, 0.015, 0.46])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.set_ylabel(r"$|\mathcal{F} x|$")

    # Bottom row: radial curves.
    ax_curve = fig.add_subplot(grid[2, :])
    for profile, label in zip(radial_profiles, labels):
        ax_curve.semilogy(radii, np.clip(profile, cfg.radial_vmin, None), linewidth=2, label=label)
    ax_curve.set_xlabel("Normalized radial frequency")
    ax_curve.set_ylabel(r"Radial mean of $|\mathcal{F}(x-x_{gt})|$ on GT support")
    title_suffix = " (full-circle support)" if cfg.radial_full_circle_only else ""
    ax_curve.set_title(
        f"Post-Sequence Radial Error Spectrum on GT Support{title_suffix} "
        f"[bin_width={cfg.radial_bin_width}, skip_bins={cfg.radial_skip_bins}, "
        f"support>={cfg.gt_support_threshold_fraction:.1e}*max]"
    )
    ax_curve.grid(True, which="both", alpha=0.3)
    ax_curve.legend(loc="best")

    # fig.tight_layout(rect=[0.0, 0.0, 0.9, 1.0])
    plt.show()


def resolve_experiments(cfg):
    """
    Resolve experiment sequences/labels from config.

    If cfg.experiment_sequences is None, default to one-sweep runs for each partition.
    """
    if cfg.experiment_sequences is None:
        sequences = tuple((idx,) for idx in range(len(cfg.granularity)))
        labels = tuple(f"{cfg.granularity[idx]} subsets (1 sweep)" for idx in range(len(cfg.granularity)))
        return sequences, labels

    sequences = tuple(tuple(int(val) for val in seq) for seq in cfg.experiment_sequences)
    if len(sequences) == 0:
        raise ValueError("experiment_sequences must contain at least one sequence.")

    max_index = len(cfg.granularity) - 1
    for seq in sequences:
        if len(seq) == 0:
            raise ValueError("Each experiment sequence must contain at least one partition index.")
        for idx in seq:
            if idx < 0 or idx > max_index:
                raise ValueError(
                    f"Invalid partition index {idx} in experiment_sequences; expected 0..{max_index}."
                )

    if cfg.experiment_labels is None:
        labels = tuple(
            f"seq={list(seq)} ({len(seq)} sweeps, last={cfg.granularity[seq[-1]]} subsets)"
            for seq in sequences
        )
    else:
        if len(cfg.experiment_labels) != len(sequences):
            raise ValueError("experiment_labels length must match experiment_sequences length.")
        labels = tuple(cfg.experiment_labels)

    return sequences, labels


def main():
    cfg = SweepConfig()
    plt.rcParams.update({"font.size": cfg.font_size})

    if cfg.use_loaded_image:
        target_2d = load_image().astype(np.float32)
        recon_shape = (target_2d.shape[0], target_2d.shape[1], 1)
        if recon_shape != cfg.recon_shape:
            warnings.warn(
                f"use_loaded_image=True: overriding recon_shape from {cfg.recon_shape} to {recon_shape}.",
                stacklevel=2,
            )
        target = target_2d[..., None]
        target_fft = np.fft.fftshift(np.fft.fft2(target_2d, norm="ortho"))
    else:
        recon_shape = cfg.recon_shape
        target, target_fft = generate_flat_magnitude_target(recon_shape, cfg.random_phase_seed)
        target_2d = target[..., 0]

    ct_model = make_parallel_beam_model(recon_shape)
    ct_model.use_ror_mask = cfg.use_ror_mask
    ct_model.set_params(granularity=cfg.granularity)
    print(f"use_ror_mask={ct_model.use_ror_mask}, radial_full_circle_only={cfg.radial_full_circle_only}")

    sinogram = np.asarray(ct_model.forward_project(target))
    experiment_sequences, experiment_labels = resolve_experiments(cfg)
    print(f"num_experiments={len(experiment_sequences)}")

    # ==================== CHECKPOINT BLOCK: TARGET/SINOGRAM ====================
    maybe_show_checkpoint(
        cfg,
        cfg.checkpoint_show_target_image,
        lambda: (
            plt.figure(figsize=(5, 4)),
            plt.imshow(target_2d, cmap="gray"),
            plt.title("Checkpoint: Target Image"),
            plt.colorbar(),
            plt.tight_layout(),
            plt.show(),
        ),
    )
    maybe_show_checkpoint(
        cfg,
        cfg.checkpoint_show_target_fft_mag,
        lambda: (
            plt.figure(figsize=(5, 4)),
            plt.imshow(np.abs(target_fft), cmap="viridis", norm=mpl.colors.LogNorm(vmin=1e-2, vmax=2.0)),
            plt.title("Checkpoint: |FFT(Target)|"),
            plt.colorbar(),
            plt.tight_layout(),
            plt.show(),
        ),
    )
    maybe_show_checkpoint(
        cfg,
        cfg.checkpoint_show_sinogram,
        lambda: (
            plt.figure(figsize=(6, 4)),
            plt.imshow(sinogram[:, 0, :], cmap="gray", aspect="auto"),
            plt.title("Checkpoint: Sinogram"),
            plt.xlabel("Detector channel"),
            plt.ylabel("View"),
            plt.colorbar(),
            plt.tight_layout(),
            plt.show(),
        ),
    )
    # ========================================================================

    recon_images = []
    freq_maps = []
    radial_profiles = []
    labels = []
    shared_radii = None
    gt_freq_raw = compute_shifted_fft_magnitude(target_2d)
    support_threshold = cfg.gt_support_threshold_fraction * max(float(gt_freq_raw.max()), np.finfo(np.float32).eps)
    gt_support_mask = gt_freq_raw >= support_threshold
    support_fraction = float(np.mean(gt_support_mask))
    print(
        f"GT support threshold={support_threshold:.3e} "
        f"({cfg.gt_support_threshold_fraction:.1e} * max), support_fraction={support_fraction:.3f}"
    )

    # Use identical random partitions across experiments for fair comparison.
    for experiment_index, (partition_sequence, experiment_label) in enumerate(
        zip(experiment_sequences, experiment_labels)
    ):
        np.random.seed(cfg.partition_seed)
        recon, recon_dict = run_recon_for_sequence(ct_model, sinogram, partition_sequence)
        recon_2d = recon[..., 0]
        freq_mag_raw = compute_shifted_fft_magnitude(recon_2d)
        error_freq_mag = compute_shifted_fft_magnitude(recon_2d - target_2d)

        recon_images.append(recon_2d.astype(np.float32))
        freq_mag = freq_mag_raw
        if cfg.normalize_maps_by_peak:
            freq_mag = freq_mag / max(float(freq_mag.max()), np.finfo(np.float32).eps)

        freq_maps.append(freq_mag.astype(np.float32))
        radii, radial_profile = compute_radial_profile(
            error_freq_mag,
            full_circle_only=cfg.radial_full_circle_only,
            bin_width=cfg.radial_bin_width,
            skip_bins=cfg.radial_skip_bins,
            valid_mask=gt_support_mask,
        )
        shared_radii = radii
        radial_profiles.append(radial_profile)
        labels.append(experiment_label)

        recon_params = recon_dict["recon_params"]
        alpha_value = recon_params["alpha_values"][-1] if len(recon_params["alpha_values"]) > 0 else np.nan
        fm_rmse = recon_params["fm_rmse"][-1] if len(recon_params["fm_rmse"]) > 0 else np.nan
        mean_intensity_error = float(np.mean(recon_2d - target_2d))
        print(
            f"Experiment {experiment_index}: sequence={list(partition_sequence)}, "
            f"alpha={alpha_value:.4f}, fm_rmse={fm_rmse:.4f}, "
            f"mean_intensity_error={mean_intensity_error:.4e}, "
            f"fft_peak={float(freq_mag.max()):.4f}, fft_error_peak={float(error_freq_mag.max()):.4f}"
        )

        # ==================== CHECKPOINT BLOCK: FIRST RECON ====================
        if experiment_index == 0:
            maybe_show_checkpoint(
                cfg,
                cfg.checkpoint_show_first_recon,
                lambda: (
                    plt.figure(figsize=(5, 4)),
                    plt.imshow(recon_2d, cmap="gray"),
                    plt.title("Checkpoint: Recon After One Sweep (First Partition)"),
                    plt.colorbar(),
                    plt.tight_layout(),
                    plt.show(),
                ),
            )
            maybe_show_checkpoint(
                cfg,
                cfg.checkpoint_show_first_recon_fft,
                lambda: (
                    plt.figure(figsize=(5, 4)),
                    plt.imshow(np.clip(freq_mag, cfg.map_vmin, None), cmap="viridis",
                               norm=mpl.colors.LogNorm(vmin=cfg.map_vmin, vmax=cfg.map_vmax)),
                    plt.title("Checkpoint: |FFT(One-Sweep Recon)| (First Partition)"),
                    plt.colorbar(),
                    plt.tight_layout(),
                    plt.show(),
                ),
            )
        # ====================================================================

    if shared_radii is None:
        raise RuntimeError("No reconstructions were generated; check granularity configuration.")

    gt_freq_map = compute_shifted_fft_magnitude(target_2d)
    if cfg.normalize_maps_by_peak:
        gt_freq_map = gt_freq_map / max(float(gt_freq_map.max()), np.finfo(np.float32).eps)

    plot_summary(
        gt_image=target_2d,
        recon_images=recon_images,
        gt_freq_map=gt_freq_map,
        freq_maps=freq_maps,
        radial_profiles=radial_profiles,
        radii=shared_radii,
        labels=labels,
        cfg=cfg,
    )


def load_image():
    """
    Load an image for use as a baseline phantom
    Returns: 2D numpy array
    """
    import mbirjax as mj
    # phantom = mj.load_data_hdf5('./data/recon.h5')[0]
    # phantom = phantom.transpose((1, 2, 0))
    # phantom = mj.preprocess.apply_cylindrical_mask(phantom, radial_margin=70)
    # image = np.array(phantom[100:, 100:, 1])
    # image = np.clip(image, 0, 0.05)
    # image = image / 0.05

    num_views = 600
    num_det_rows = 200
    num_det_channels = 1000

    # Generate simulated data
    # In a real application you would not have the phantom, but we include it here for later display purposes
    phantom, sinogram, params = mj.generate_demo_data(object_type='shepp-logan', model_type='parallel',
                                                      num_views=num_views, num_det_rows=num_det_rows,
                                                      num_det_channels=num_det_channels)
    image = phantom[:, :, num_det_rows // 2]
    image = np.array(image)
    image = image / np.max(image)

    plt.imshow(image, cmap="gray")
    plt.show()
    return image


if __name__ == "__main__":
    main()
