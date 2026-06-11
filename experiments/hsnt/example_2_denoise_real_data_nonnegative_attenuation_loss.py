"""
Hyperspectral Dehydration & Rehydration
---------------------------------------

This script demonstrates the use of dehydration and rehydration for hyperspectral data denoising.
Multiple real hyperspectral neutron datasets are available for the purpose.
"""

import os
import numpy as np
import time
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')

import jax
import jax.numpy as jnp
import jax.lax as lax

from mbirjax.hsnt import import_hsnt_data_hdf5, export_hsnt_data_hdf5
from plot_utils import plot_images, plot_spectra


@jax.jit
def newton_update(W, H, T):
    """JAX-optimized Newton update with automatic differentiation and line search.

    Args:
        W: Feature matrix (spatial pixels × num_materials), JAX array
        H: Spectral basis matrix (num_materials × spectral channels), JAX array
        T: Data term matrix (spatial pixels × spectral channels), JAX array

    Returns:
        Updated (W, H) pair as JAX arrays
    """
    def loss_fn(W_h_pair):
        W_, H_ = W_h_pair
        X = W_ @ H_
        return (jnp.exp(-X) + T * X).sum()

    # Use JAX's automatic differentiation
    loss_grad_fn = jax.grad(loss_fn)

    X = W @ H
    Z = jnp.exp(-X)
    init_loss = (Z + T * X).sum()

    # Compute gradients via automatic differentiation
    grad_W, grad_H = loss_grad_fn((W, H))

    # Compute Hessian diagonal approximation manually (kept explicit for numerical stability)
    d2L_dW2 = Z @ (H.T**2)
    d2L_dH2 = (W.T**2) @ Z

    dW = grad_W / (d2L_dW2 + 1e-10)
    dH = grad_H / (d2L_dH2 + 1e-10)

    # Line search over learning rates (vectorized)
    learning_rates = jnp.logspace(0.5, -3, 15)

    def line_search_step(carry, lr):
        W_best, H_best, loss_best, found = carry
        W_temp = jnp.maximum(W - lr * dW, 1e-10)
        H_temp = jnp.maximum(H - lr * dH, 1e-10)
        X_temp = W_temp @ H_temp
        temp_loss = (jnp.exp(-X_temp) + T * X_temp).sum()

        # Update if loss improved
        improved = temp_loss < loss_best
        W_best = jnp.where(improved, W_temp, W_best)
        H_best = jnp.where(improved, H_temp, H_best)
        loss_best = jnp.where(improved, temp_loss, loss_best)
        found = found | improved

        return (W_best, H_best, loss_best, found), None

    (W_new, H_new, _, _), _ = lax.scan(line_search_step, (W, H, init_loss, False), learning_rates)

    return W_new, H_new

@jax.jit
def multiplicative_update(W, H, T):
    """JAX-optimized multiplicative update for non-negative factorization.

    Args:
        W: Feature matrix (spatial pixels × num_materials), JAX array
        H: Spectral basis matrix (num_materials × spectral channels), JAX array
        T: Data term matrix (spatial pixels × spectral channels), JAX array

    Returns:
        Updated (W, H) pair as JAX arrays
    """
    Z = jnp.exp(-W @ H)

    W_mult = ((Z @ H.T) / (T @ H.T) + 1) / 2
    H_mult = ((W.T @ Z) / (W.T @ T) + 1) / 2

    return W * W_mult, H * H_mult

@jax.jit
def optimize(T, num_materials, max_steps, rel_tol):
    """Optimize W, H using Newton and multiplicative updates."""
    num_pixels = T.shape[0]
    num_wavelengths = T.shape[1]

    # Fixed seed for reproducibility
    key = jax.random.PRNGKey(129)

    # ===== Newton Optimization =====
    def newton_cond(state):
        _, _, _, i, converged = state
        return (i < max_steps) & (~converged)

    def newton_body(state):
        W, H, prev_loss, i, converged = state
        W_new, H_new = newton_update(W, H, T)
        loss_new = (jnp.exp(-W_new @ H_new) + T * (W_new @ H_new)).sum()
        is_converged = jnp.abs(loss_new - prev_loss) / (prev_loss + 1e-10) < rel_tol
        W_out = jnp.where(converged, W, W_new)
        H_out = jnp.where(converged, H, H_new)
        loss_out = jnp.where(converged, prev_loss, loss_new)
        return (W_out, H_out, loss_out, i + 1, converged | is_converged)

    key1, key2 = jax.random.split(key)
    W_newt_init = jax.random.uniform(key1, shape=(num_pixels, num_materials), dtype=jnp.float32)
    H_newt_init = jax.random.uniform(key2, shape=(num_materials, num_wavelengths), dtype=jnp.float32)

    prev_loss_newt = (jnp.exp(-W_newt_init @ H_newt_init) + T * (W_newt_init @ H_newt_init)).sum()
    state_newt = (W_newt_init, H_newt_init, prev_loss_newt, 0, False)
    state_newt = lax.while_loop(newton_cond, newton_body, state_newt)
    W_newt, H_newt, _, _, _ = state_newt

    # ===== Multiplicative Optimization =====
    def mu_cond(state):
        _, _, _, i, converged = state
        return (i < max_steps) & (~converged)

    def mu_body(state):
        W, H, prev_loss, i, converged = state
        W_new, H_new = multiplicative_update(W, H, T)
        loss_new = (jnp.exp(-W_new @ H_new) + T * (W_new @ H_new)).sum()
        is_converged = jnp.abs(loss_new - prev_loss) / (prev_loss + 1e-10) < rel_tol
        W_out = jnp.where(converged, W, W_new)
        H_out = jnp.where(converged, H, H_new)
        loss_out = jnp.where(converged, prev_loss, loss_new)
        return (W_out, H_out, loss_out, i + 1, converged | is_converged)

    key3, key4 = jax.random.split(key)
    W_mu_init = jax.random.uniform(key3, shape=(num_pixels, num_materials), dtype=jnp.float32)
    H_mu_init = jax.random.uniform(key4, shape=(num_materials, num_wavelengths), dtype=jnp.float32)

    prev_loss_mu = (jnp.exp(-W_mu_init @ H_mu_init) + T * (W_mu_init @ H_mu_init)).sum()
    state_mu = (W_mu_init, H_mu_init, prev_loss_mu, 0, False)
    state_mu = lax.while_loop(mu_cond, mu_body, state_mu)
    W_mu, H_mu, _, _, _ = state_mu

    return W_newt, H_newt, W_mu, H_mu


def main():
    start_time = time.time()

    # Choose dataset from '0.8C_Ni_cylinder', '1.6C_Ni_cylinder', '2.4C_Ni_cylinder', '4.8C_Ni_cylinder', '9.6C_Ni_cylinder'
    dataset_name = '0_8c_Ni_cylinder_dataset'
    input_path = './input_data/processed_data_0_8c_Ni_cylinder.h5'  # path to import input noisy data
    output_path = './output_data/'  # path to export output denoised data
    os.makedirs(output_path, exist_ok=True)  # Make output directory if it does not exist

    # Denoiser parameters
    num_materials = 3  # Number of materials
    verbose = 2  # Verbosity level

    # Display parameters
    display_wave_idx = 100  # Wavelength index of displayed images
    display_pix_idx = [200, 200]  # Pixel index [row, column] of displayed spectra
    vmax = 2  # Maximum pixel value for displayed images
    vmin = 0  # Minimum pixel value for displayed images
    y_lim_attenuation = (0, 3)  # (y_min, y_max) to set y-axis range for attenuation spectra
    y_lim_transmission = (0, 0.75)  # (y_min, y_max) to set y-axis range for transmission spectra

    # Import real hyperspectral data
    hsnt_data, metadata = import_hsnt_data_hdf5(input_path, dataset_name)
    wavelengths = metadata['wavelengths']

    if verbose >= 1:
        print("Hyperspectral data shape: ", hsnt_data.shape)

    T = np.exp(-hsnt_data).reshape(-1, hsnt_data.shape[-1])

    # Convert to JAX array
    T_jax = jnp.asarray(T, dtype=jnp.float32)

    # Perform hyperspectral denoising
    W_newt, H_newt, W_mu, H_mu = optimize(
        T_jax, num_materials=num_materials, max_steps=1000, rel_tol=1e-8
    )

    # Convert results to NumPy arrays and reshape to original data shape
    hsnt_denoised_newt = np.array(W_newt @ H_newt).reshape(hsnt_data.shape)
    hsnt_denoised_mu = np.array(W_mu @ H_mu).reshape(hsnt_data.shape)

    # Write out denoised/rehydrated data
    export_hsnt_data_hdf5(os.path.join(output_path, dataset_name+'_dataset_denoised_newt.h5'), hsnt_denoised_newt, metadata)
    export_hsnt_data_hdf5(os.path.join(output_path, dataset_name+'_dataset_denoised_mu.h5'), hsnt_denoised_mu, metadata)

    # Plot hyperspectral projections and spectra
    if verbose > 1:
        plot_images(images=[hsnt_data[0, :, :, display_wave_idx],
                            hsnt_denoised_newt[0, :, :, display_wave_idx],
                            hsnt_denoised_mu[0, :, :, display_wave_idx]],
                    titles=['Fig (a): Noisy hyperspectral projection \n\nWavelength index: ' + str(display_wave_idx),
                            'Fig (b): Newton Denoised hyperspectral projection \n\nWavelength index: ' + str(display_wave_idx),
                            'Fig (c): Multiplicative Denoised hyperspectral projection \n\nWavelength index: ' + str(display_wave_idx)],
                    vmax=vmax, vmin=vmin,
                    filename="cylinder_nnal.png")

        plot_spectra(spectra=[hsnt_data[0, display_pix_idx[0], display_pix_idx[1], :],
                              hsnt_denoised_newt[0, display_pix_idx[0], display_pix_idx[1], :],
                              hsnt_denoised_mu[0, display_pix_idx[0], display_pix_idx[1], :]],
                     labels=['Noisy', 'Newton', 'Multiplicative'],
                     title='Single pixel spectra (attenuation) for noisy and denoised data',
                     x_label='wavelength (Angstrom)',
                     y_label='attenuation',
                     y_lim=y_lim_attenuation,
                     wavelengths=wavelengths,
                     filename="cylinder_attenuation_nnal.png")

        plot_spectra(spectra=[np.exp(-hsnt_data[0, display_pix_idx[0], display_pix_idx[1], :]),
                              np.exp(-hsnt_denoised_newt[0, display_pix_idx[0], display_pix_idx[1], :]),
                              np.exp(-hsnt_denoised_mu[0, display_pix_idx[0], display_pix_idx[1], :])],
                     labels=['Noisy', 'Newton', 'Multiplicative'],
                     title='Single pixel spectra (transmission) for noisy and denoised data',
                     x_label='wavelength (Angstrom)',
                     y_label='transmission',
                     y_lim=y_lim_transmission,
                     wavelengths=wavelengths,
                     filename="cylinder_transmission_nnal.png")

    print('Total time elapsed: ', time.time() - start_time, ' seconds')

    plt.show()


if __name__ == "__main__":
    main()
