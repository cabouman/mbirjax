"""
Hyperspectral Dehydration & Rehydration
---------------------------------------

This script demonstrates the use of dehydration and rehydration for hyperspectral data denoising.
A simulated hyperspectral neutron dataset containing three materials (Ni, Cu, and Al) is used for the purpose.
"""

import os
import shutil
import numpy as np
import time
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')
from PIL import Image
import jax
import jax.numpy as jnp
import jax.lax as lax
from functools import partial

from mbirjax.hsnt import dehydrate, rehydrate, generate_hyper_data
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


def create_optimization_fn(N, rel_tol):
    """Create a vmappable optimization function that handles one seed.

    Args:
        N: Maximum iterations
        rel_tol: Relative tolerance for convergence

    Returns:
        vmappable function that takes (seed_idx, T, material_basis, material_projection)
    """
    @jax.jit
    def optimize_for_seed(seed_idx, T, material_basis, material_projection):
        """Pure function to optimize W, H for a single seed and compute MSEs."""
        num_pixels = T.shape[0]
        num_wavelengths = T.shape[1]
        num_materials = material_basis.shape[0]

        # ===== Newton Optimization =====
        def newton_cond(state):
            W, H, prev_loss, i, converged = state
            return (i < N) & (~converged)

        def newton_body(state):
            W, H, prev_loss, i, converged = state
            W_new, H_new = newton_update(W, H, T)
            loss_new = (jnp.exp(-W_new @ H_new) + T * (W_new @ H_new)).sum()
            is_converged = jnp.abs(loss_new - prev_loss) / (prev_loss + 1e-10) < rel_tol
            W_out = jnp.where(converged, W, W_new)
            H_out = jnp.where(converged, H, H_new)
            loss_out = jnp.where(converged, prev_loss, loss_new)
            return (W_out, H_out, loss_out, i + 1, converged | is_converged)

        # Random initialization based on seed_idx
        key = jax.random.PRNGKey(jnp.array(seed_idx, dtype=int))
        key1, key2 = jax.random.split(key)
        W_newt_init = jax.random.uniform(key1, shape=(num_pixels, num_materials), dtype=jnp.float32)
        H_newt_init = jax.random.uniform(key2, shape=(num_materials, num_wavelengths), dtype=jnp.float32)

        prev_loss_newt = (jnp.exp(-W_newt_init @ H_newt_init) + T * (W_newt_init @ H_newt_init)).sum()
        state_newt = (W_newt_init, H_newt_init, prev_loss_newt, 0, False)
        state_newt = lax.while_loop(newton_cond, newton_body, state_newt)
        W_newt, H_newt, _, _, _ = state_newt

        # ===== Multiplicative Optimization =====
        def mu_cond(state):
            W, H, prev_loss, i, converged = state
            return (i < N) & (~converged)

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

        # ===== Compute MSEs =====
        # Use jnp.linalg operations for JAX compatibility
        theta_newt = jnp.linalg.lstsq(H_newt.T, material_basis.T, rcond=None)[0].T
        theta_mu = jnp.linalg.lstsq(H_mu.T, material_basis.T, rcond=None)[0].T

        Ds_newt = jnp.linalg.norm(theta_newt @ H_newt - material_basis) ** 2 / material_basis.size
        Ds_mu = jnp.linalg.norm(theta_mu @ H_mu - material_basis) ** 2 / material_basis.size

        Dm_newt = jnp.linalg.norm(W_newt @ jnp.linalg.pinv(theta_newt) - material_projection) ** 2 / material_projection.size
        Dm_mu = jnp.linalg.norm(W_mu @ jnp.linalg.pinv(theta_mu) - material_projection) ** 2 / material_projection.size

        return Ds_newt, Ds_mu, Dm_newt, Dm_mu

    # Return vmapped version (vectorizes over first axis: seed_idx)
    return jax.vmap(optimize_for_seed, in_axes=(0, None, None, None))


def main():
    # Simulation parameters
    num_angles = 1  # Number of projection angles
    detector_rows = 64  # Number of rows in the detector
    detector_columns = 64  # Number of columns in the detector
    dosage_rate = 50  # Neutron dosage rate
    material_density = {"Ni": 0.25, "Cu": 0.25, "Al": 0.75}  # Define material density (vol. fraction)
    dataset_type = 'attenuation'  # Choose between 'attenuation' or 'transmission'

    # Denoiser parameters
    num_materials = 3  # Number of materials
    verbose = 0  # Verbosity level
    N = 1000  # Maximum number of fine-tuning iterations

    # GPU detection and device configuration
    try:
        gpu_devices = jax.devices('gpu')
        print(f"GPU available: {len(gpu_devices)} GPU device(s) detected")
    except RuntimeError:
        print("No GPU available, using CPU")

    # Load theoretical linear attenuation coefficients for Ni, Cu, and Al
    material_basis_path = './binaries/'
    filename = os.path.join(material_basis_path, 'material_basis.npy')
    material_basis_0 = np.load(filename)

    # Create vmapped optimization function once (outside main loop)
    rel_tol = 1e-8
    vmapped_optimize = create_optimization_fn(N, rel_tol)

    with open('denoising_results_1.csv', 'w') as f:
        f.write('num_pixels,num_wavelengths,random_seed,MSE_spectra_newt,MSE_spectra_mu,MSE_materials_newt,MSE_materials_mu\n')

        for size in np.logspace(2, 8, 7, base=2, dtype=int):
            detector_rows = size
            detector_columns = size
            for downsample in np.logspace(np.log2(1200), 0, 10, base=2, dtype=int):
                material_basis = material_basis_0[:, ::downsample]

                print(f'\nSimulating data with size {size}x{size}, downsample factor {downsample}...')
                start_time = time.time()

                # Generate ONE set of data (shared across all seeds for this configuration)
                [noisy_hyper_projection,
                _,
                gt_hyper_projection] = \
                    generate_hyper_data(material_basis,
                                        num_angles=num_angles,
                                        detector_rows=detector_rows,
                                        detector_columns=detector_columns,
                                        dosage_rate=dosage_rate,
                                        material_density=material_density,
                                        verbose=verbose)
                noisy_hyper_projection = np.nan_to_num(noisy_hyper_projection, nan=0.0, posinf=0.0, neginf=0.0)
                T = np.exp(-noisy_hyper_projection).reshape(-1, gt_hyper_projection.shape[-1])
                print(f'Simulation completed in {time.time() - start_time:.2f} seconds')
                start_time = time.time()

                # Spoof simulated projection data for 3 materials (Ni, Cu, and Al)
                height = detector_rows // 3
                width = detector_columns // 2
                thickness = 20 * np.sqrt((width//2)**2 - np.linspace(-width // 2, width // 2, width)**2) / width
                material_projection = np.zeros((num_angles, detector_rows, detector_columns, num_materials)).astype(np.float32)
                material_projection[:, :height, width // 2:width + width // 2, 0] = material_density["Ni"] * thickness
                material_projection[:, 2 * height:, width // 2:width + width // 2, 1] = material_density["Cu"] * thickness
                material_projection[:, height:2 * height, width // 2:width + width // 2, 2] = material_density["Al"] * thickness
                material_projection = material_projection.reshape(-1, num_materials)

                # Convert to JAX arrays once
                T_jax = jnp.asarray(T, dtype=jnp.float32)
                material_basis_jax = jnp.asarray(material_basis, dtype=jnp.float32)
                material_projection_jax = jnp.asarray(material_projection, dtype=jnp.float32)

                # Vectorized optimization across all 10 seeds at once
                seeds = jnp.arange(10, dtype=jnp.int32)
                Ds_newt_batch, Ds_mu_batch, Dm_newt_batch, Dm_mu_batch = vmapped_optimize(
                    seeds, T_jax, material_basis_jax, material_projection_jax
                )

                # Convert results to NumPy and write to CSV
                Ds_newt_np = np.array(Ds_newt_batch)
                Ds_mu_np = np.array(Ds_mu_batch)
                Dm_newt_np = np.array(Dm_newt_batch)
                Dm_mu_np = np.array(Dm_mu_batch)

                print(f'Optimization completed in {time.time() - start_time:.2f} seconds')

                for seed in range(10):
                    f.write(f'{size**2},{material_basis.shape[1]},{seed},'
                           f'{Ds_newt_np[seed]},{Ds_mu_np[seed]},'
                           f'{Dm_newt_np[seed]},{Dm_mu_np[seed]}\n')
                f.flush()

if __name__ == "__main__":
    main()
