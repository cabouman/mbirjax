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
    normalize_maps_by_peak: bool = True
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


def compute_radial_profile(image_2d, full_circle_only=True):
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
        binned_vals = bin_indices[valid].ravel()
        img_vals = image_2d[valid].ravel()
    else:
        max_bin = int(bin_indices.max())
        binned_vals = bin_indices.ravel()
        img_vals = image_2d.ravel()

    sums = np.bincount(binned_vals, weights=img_vals, minlength=max_bin + 1)
    counts = np.bincount(binned_vals, minlength=max_bin + 1)
    profile = sums / np.maximum(counts, 1)
    radius_values = np.arange(max_bin + 1, dtype=np.float32)
    radius_values /= max(float(max_bin), 1.0)
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


def plot_summary(freq_maps, radial_profiles, radii, labels, cfg):
    """Plot one-sweep Fourier maps (top) and radial curves (bottom)."""
    ncols = len(freq_maps)
    fig = plt.figure(figsize=(4 * ncols, 8))
    grid = fig.add_gridspec(2, ncols, height_ratios=[1.0, 0.9])
    norm = mpl.colors.LogNorm(vmin=cfg.map_vmin, vmax=cfg.map_vmax)

    for i, (freq_map, label) in enumerate(zip(freq_maps, labels)):
        ax = fig.add_subplot(grid[0, i])
        ax.imshow(
            np.clip(freq_map, cfg.map_vmin, None),
            cmap="viridis",
            norm=norm,
            origin="upper",
            interpolation="nearest",
        )
        ax.set_title(label)
        ax.axis("off")

    cax = fig.add_axes([0.92, 0.54, 0.015, 0.34])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.set_ylabel(r"$|\mathcal{F} x|$")

    ax_curve = fig.add_subplot(grid[1, :])
    for profile, label in zip(radial_profiles, labels):
        ax_curve.semilogy(radii, np.clip(profile, cfg.radial_vmin, None), linewidth=2, label=label)
    ax_curve.set_xlabel("Normalized radial frequency")
    ax_curve.set_ylabel(r"Radial mean of $|\mathcal{F} x|$")
    title_suffix = " (full-circle support)" if cfg.radial_full_circle_only else ""
    ax_curve.set_title(f"Post-Sequence Radial Spectral Gain{title_suffix}")
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

    target, target_fft = generate_flat_magnitude_target(cfg.recon_shape, cfg.random_phase_seed)
    target_2d = target[..., 0]

    ct_model = make_parallel_beam_model(cfg.recon_shape)
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

    freq_maps = []
    radial_profiles = []
    labels = []
    shared_radii = None

    # Use identical random partitions across experiments for fair comparison.
    for experiment_index, (partition_sequence, experiment_label) in enumerate(
        zip(experiment_sequences, experiment_labels)
    ):
        np.random.seed(cfg.partition_seed)
        recon, recon_dict = run_recon_for_sequence(ct_model, sinogram, partition_sequence)
        recon_2d = recon[..., 0]
        freq_mag = compute_shifted_fft_magnitude(recon_2d)

        if cfg.normalize_maps_by_peak:
            freq_mag = freq_mag / max(float(freq_mag.max()), np.finfo(np.float32).eps)

        freq_maps.append(freq_mag.astype(np.float32))
        radii, radial_profile = compute_radial_profile(
            freq_maps[-1], full_circle_only=cfg.radial_full_circle_only
        )
        shared_radii = radii
        radial_profiles.append(radial_profile)
        labels.append(experiment_label)

        recon_params = recon_dict["recon_params"]
        alpha_value = recon_params["alpha_values"][-1] if len(recon_params["alpha_values"]) > 0 else np.nan
        fm_rmse = recon_params["fm_rmse"][-1] if len(recon_params["fm_rmse"]) > 0 else np.nan
        print(
            f"Experiment {experiment_index}: sequence={list(partition_sequence)}, "
            f"alpha={alpha_value:.4f}, fm_rmse={fm_rmse:.4f}, "
            f"fft_peak={float(freq_mag.max()):.4f}"
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

    plot_summary(freq_maps, radial_profiles, shared_radii, labels, cfg)


if __name__ == "__main__":
    main()
