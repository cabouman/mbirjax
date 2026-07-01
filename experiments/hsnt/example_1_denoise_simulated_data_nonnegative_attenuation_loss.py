"""
Hyperspectral Dehydration & Rehydration
---------------------------------------

This script demonstrates the use of dehydration and rehydration for hyperspectral data denoising.
A simulated hyperspectral neutron dataset containing three materials (Ni, Cu, and Al) is used for the purpose.
"""

import os
import numpy as np
import time
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')

import jax
import jax.numpy as jnp
from jax import lax

from scipy import optimize

from mbirjax.hsnt import dehydrate, generate_hyper_data
from plot_utils import plot_images


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
    learning_rates = jnp.logspace(0.33, -1, 5)

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

def optimize_newt_body(T, num_materials, max_steps, rel_tol):
    """Optimize W, H using Newton updates."""
    num_pixels = T.shape[0]
    num_wavelengths = T.shape[1]

    # Fixed seed for reproducibility
    key = jax.random.PRNGKey(jnp.array(129, dtype=int))

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
    W_newt, H_newt, _, i, _ = state_newt

    return W_newt, H_newt, i
optimize_newt = jax.jit(optimize_newt_body, static_argnames=['num_materials', 'max_steps', 'rel_tol'])

def optimize_mu_body(T, num_materials, max_steps, rel_tol):
    """Optimize W, H using multiplicative updates."""
    num_pixels = T.shape[0]
    num_wavelengths = T.shape[1]

    # Fixed seed for reproducibility
    key = jax.random.PRNGKey(129)

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
    W_mu, H_mu, _, i, _ = state_mu

    return W_mu, H_mu, i
optimize_mu = jax.jit(optimize_mu_body, static_argnames=['num_materials', 'max_steps', 'rel_tol'])

def main():
    # Simulation parameters
    num_angles = 1  # Number of projection angles
    detector_rows = 64  # Number of rows in the detector
    detector_columns = 64  # Number of columns in the detector
    dosage_rate = 3  # Neutron dosage rate
    material_density = {"Ni": 0.25, "Cu": 0.25, "Al": 0.75}  # Define material density (vol. fraction)
    dataset_type = 'attenuation'  # Choose between 'attenuation' or 'transmission'

    # Denoiser parameters
    num_materials = 3  # Number of materials
    verbose = 0  # Verbosity level

    # Fix seed for random number generation
    np.random.seed(129)

    # Load theoretical linear attenuation coefficients for Ni, Cu, and Al
    material_basis_path = './binaries/'
    filename = os.path.join(material_basis_path, 'material_basis.npy')
    material_basis = np.load(filename)

    # Generate simulated noisy hyperspectral data and ground truth
    [noisy_hyper_projection, _, gt_hyper_projection] = generate_hyper_data(material_basis,
                                                                           num_angles=num_angles,
                                                                           detector_rows=detector_rows,
                                                                           detector_columns=detector_columns,
                                                                           dosage_rate=dosage_rate,
                                                                           material_density=material_density,
                                                                           verbose=verbose)
    noisy_hyper_projection = np.nan_to_num(noisy_hyper_projection, nan=0.0, posinf=0.0, neginf=0.0)  # Replace any NaNs or infs with zeros
    T = np.exp(-noisy_hyper_projection).reshape(-1, gt_hyper_projection.shape[-1])

    # Spoof simulated projection data which is not returned by generate_hyper_data
    height = detector_rows // 3
    width = detector_columns // 2
    thickness = 20 * np.sqrt((width//2)**2 - np.linspace(-width // 2, width // 2, width)**2)/ width
    material_projection = np.zeros((num_angles, detector_rows, detector_columns, num_materials)).astype(np.float32)
    material_projection[:, :height, width // 2:width + width // 2, 0] = material_density["Ni"] * thickness
    material_projection[:, 2 * height:, width // 2:width + width // 2, 1] = material_density["Cu"] * thickness
    material_projection[:, height:2 * height, width // 2:width + width // 2, 2] = material_density["Al"] * thickness
    material_projection = material_projection.reshape(-1, num_materials)

    # Perform hyperspectral denoising (dehydrate + rehydrate)
    W, H, _ = dehydrate(noisy_hyper_projection,
                        dataset_type=dataset_type,
                        num_materials=num_materials,
                        safety_factor=1,
                        verbose=verbose)
    W = W.reshape(np.prod(gt_hyper_projection.shape[:-1]), num_materials)
    H = H.reshape(num_materials, gt_hyper_projection.shape[-1])

    ### Refine using nonnegative attenuation loss

    # Convert to JAX array
    T_jax = jnp.asarray(T, dtype=jnp.float32)

    # Perform hyperspectral denoising
    W_newt, H_newt, i_newt = optimize_newt(
        T_jax, num_materials=num_materials, max_steps=1000, rel_tol=1e-8
    )
    start_time = time.time()
    W_newt, H_newt, i_newt = optimize_newt(
        T_jax, num_materials=num_materials, max_steps=1000, rel_tol=1e-8
    )
    print('Newton reconstruction completed in: ', time.time() - start_time, ' seconds after ', i_newt, ' iterations')
    W_mu, H_mu, i_mu = optimize_mu(
        T_jax, num_materials=num_materials, max_steps=1000, rel_tol=1e-8
    )
    start_time = time.time()
    W_mu, H_mu, i_mu = optimize_mu(
        T_jax, num_materials=num_materials, max_steps=1000, rel_tol=1e-8
    )
    print('Multiplicative reconstruction completed in: ', time.time() - start_time, ' seconds after ', i_mu, ' iterations')

    W_newt = np.asarray(W_newt, dtype=np.float64)
    H_newt = np.asarray(H_newt, dtype=np.float64)
    W_mu = np.asarray(W_mu, dtype=np.float64)
    H_mu = np.asarray(H_mu, dtype=np.float64)

    # Compute least squares estimate of material coefficients for current projections
    theta_frob = np.linalg.lstsq(H.T, material_basis.T)[0].T
    theta_newt1 = np.linalg.lstsq(H_newt.T, material_basis.T)[0].T
    theta_mu1 = np.linalg.lstsq(H_mu.T, material_basis.T)[0].T

    def compute_rmse(W, H):
        def f(theta):
            theta = theta.reshape(num_materials, num_materials)
            W_trans = W @ np.linalg.pinv(theta)
            H_trans = theta @ H
            rmse = np.linalg.norm(material_basis - H_trans)**2 / material_basis.size + \
                   np.linalg.norm(material_projection - W_trans)**2 / material_projection.size
            return rmse
        return f

    simp = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]
    opts = {'method': 'Nelder-Mead', 'options': {'maxiter': 1000, 'initial_simplex': simp}, 'x0': np.zeros(num_materials**2)}

    theta_newt2 = optimize.minimize(compute_rmse(W_newt, H_newt), **opts).x.reshape(num_materials, num_materials)
    theta_mu2 = optimize.minimize(compute_rmse(W_mu, H_mu), **opts).x.reshape(num_materials, num_materials)

    # Plot reconstructed spectra
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.figure(figsize=(12, 16))
    plt.suptitle('Material attenuation spectra reconstructions')
    for i, (spectra, title) in enumerate([
            (material_basis, 'Ground Truth'),
            (theta_frob @ H, r'$L^2$ Loss'),
            (theta_newt1 @ H_newt, 'Quasi-Newton (Method 1)'),
            (theta_mu1 @ H_mu, 'Mann-Multiplicative (Method 1)'),
            #(theta_newt2 @ H_newt, 'Quasi-Newton (Method 2)'),
            #(theta_mu2 @ H_mu, 'Mann-Multiplicative (Method 2)'),
        ]):
        ax = plt.subplot(4, 1, i + 1)
        ax.plot(spectra[0], label='Ni')
        ax.plot(spectra[1], label='Cu')
        ax.plot(spectra[2], label='Al')

        ax.set_title(title)
        ax.set_xlabel('wavelength index')
        if i == 0:
            ax.set_ylabel('attenuation')
        else:
            ax.set_yticklabels([])
        ax.set_ylim(0, 1.1)
        ax.legend(loc='upper left')
    plt.savefig('example_1_nonnegative_attenuation_loss_spectra_reconstruction.png')

    # Plot reconstructed material coefficient maps
    plt.figure(figsize=(12, 16))
    plt.suptitle('Material projection reconstructions')
    row_max = np.max(material_projection, axis=0).reshape(1, 1, num_materials)
    image_dims = (detector_rows, detector_columns, num_materials)
    for i, (image, title) in enumerate([
            (material_projection.reshape(image_dims) / row_max, 'Ground Truth'),
            ((W @ np.linalg.pinv(theta_frob)).reshape(image_dims) / row_max, '$L^2$ Loss'),
            ((W_newt @ np.linalg.pinv(theta_newt1)).reshape(image_dims) / row_max, 'Quasi-Newton (Method 1)'),
            ((W_mu @ np.linalg.pinv(theta_mu1)).reshape(image_dims) / row_max, 'Mann-Multiplicative (Method 1)'),
            ((W_newt @ np.linalg.pinv(theta_newt2)).reshape(image_dims) / row_max, 'Quasi-Newton (Method 2)'),
            ((W_mu @ np.linalg.pinv(theta_mu2)).reshape(image_dims) / row_max, 'Mann-Multiplicative (Method 2)'),
        ]):
        ax = plt.subplot(3, 2, i + 1)
        ax.set_title(title)
        ax.imshow(image)
    plt.savefig('example_1_nonnegative_attenuation_loss_material_maps.png')

    # plt.show()

if __name__ == "__main__":
    main()
