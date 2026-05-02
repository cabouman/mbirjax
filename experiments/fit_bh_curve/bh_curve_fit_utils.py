import matplotlib.pyplot as plt
import numpy as np
import mbirjax as mj


def plot_bh_correction_curve(x_path_length, fitted_bh_params):
    """
    Plot the fitted beam-hardening curve and a linear reference curve.

    The linear reference has slope equal to the fitted curve gradient at zero.
    """
    # Linear reference with slope equal to the gradient of the fitted curve at 0.
    epsilon = 1e-6
    fitted_at_zero = mj.apply_fitted_beam_hardening_curve(0.0, fitted_bh_params)
    fitted_at_epsilon = mj.apply_fitted_beam_hardening_curve(epsilon, fitted_bh_params)
    slope_at_zero = (fitted_at_epsilon - fitted_at_zero) / epsilon
    projection_ideal = slope_at_zero * x_path_length

    # Fitted BH correction curve.
    projection_fitted_bh = mj.apply_fitted_beam_hardening_curve(
        x_path_length,
        fitted_bh_params,
    )

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
    plt.show()
    plt.close()