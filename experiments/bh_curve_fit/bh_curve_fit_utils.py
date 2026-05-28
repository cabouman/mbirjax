import matplotlib.pyplot as plt
import numpy as np
import mbirjax as mj
import mbirjax.preprocess as mjp

def plot_bh_correction_curve(x_path_length, fitted_bh_params, output_path=None, show=True):
    """
    Plot the fitted beam-hardening curve and a linear reference curve.

    The linear reference has slope equal to the fitted curve gradient at zero.
    """
    # Linear reference with slope equal to the gradient of the fitted curve at 0.
    epsilon = 1e-6
    fitted_at_zero = mjp.apply_beam_hardening_curve(0.0, fitted_bh_params)
    fitted_at_epsilon = mjp.apply_beam_hardening_curve(epsilon, fitted_bh_params)
    slope_at_zero = (fitted_at_epsilon - fitted_at_zero) / epsilon
    projection_ideal = slope_at_zero * x_path_length

    # Fitted BH correction curve.
    projection_fitted_bh = mjp.apply_beam_hardening_curve(x_path_length, fitted_bh_params)

    # ------------------------------------------
    # Plot
    # ------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    ax.plot(
        x_path_length,
        projection_ideal,
        color='black',
        linestyle='--',
        linewidth=2.4,
        alpha=0.85,
        label='Linear Projection Curve'
    )

    ax.plot(
        x_path_length,
        projection_fitted_bh,
        color='green',
        linestyle='-',
        linewidth=3.0,
        label=r'Fitted BH Curve $h^*(p)$'
    )

    ax.set_xlabel('Path Length [mm]')
    ax.set_ylabel('Projection Value')
    ax.grid(True, linestyle=':', alpha=0.6)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    ax.legend(loc='best', frameon=True)

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_inverse_correction_curve(projection_values, baseline_corrected_projection, cheb_corrected_projection, output_path=None, show=True):
    """
    Plot baseline and Chebyshev inverse projection-linearization curves.
    """
    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    ax.plot(
        projection_values,
        baseline_corrected_projection,
        color='tab:blue',
        linewidth=3.0,
        label='Polynomial BH Correction Function(baseline)')
    ax.plot(
        projection_values,
        cheb_corrected_projection,
        color='darkorange',
        linewidth=2.8,
        label='Chebyshev Inverse BH Correction Function(proposed)')

    ax.set_xlabel('Measured Projection Value')
    ax.set_ylabel('Corrected Projection Value')
    ax.set_title('Beam-Hardening Correction Functions')
    ax.grid(True, linestyle=':', alpha=0.6)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    ax.legend(loc='best', frameon=True)
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def create_uniform_index(angle_candidates, end_index, num_views):
    """
    Select ``num_views`` indices uniformly from the top ``end_index`` angles.
    """
    angles = np.asarray(angle_candidates)
    num_angles = angles.shape[0]

    if not (1 <= end_index <= num_angles):
        raise ValueError(
            f'end_index must be between 1 and {num_angles}, got {end_index}.')
    if num_views < 2:
        raise ValueError(f'num_views must be at least 2, got {num_views}.')

    sorted_desc_indices = np.argsort(angles)[::-1]
    truncated_indices = sorted_desc_indices[:end_index]

    step = (end_index - 1) / (num_views - 1)
    chosen_positions = [int(np.floor(step * i)) for i in range(num_views - 1)]
    chosen_positions.append(end_index - 1)

    uniform_indices = [int(truncated_indices[pos]) for pos in chosen_positions]

    return uniform_indices
