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

import jax.numpy as jnp

from mbirjax.hsnt import dehydrate, generate_hyper_data, nnal_factorization, compare_spectra


def main():
    # Simulation parameters
    num_angles = 1  # Number of projection angles
    detector_rows = 64  # Number of rows in the detector
    detector_columns = 64  # Number of columns in the detector
    dosage_rate = 3  # Neutron dosage rate
    material_density = {"Ni": 0.25, "Cu": 0.25, "Al": 0.75}  # Define material density (vol. fraction)
    dataset_type = 'attenuation'  # Choose between 'attenuation' or 'transmission'

    # Denoiser parameters
    num_materials_fit = 3  # Number of materials in reconstructed subspace
    verbose = 0  # Verbosity level

    # Fix seed for random number generation
    np.random.seed(129)

    # Load theoretical linear attenuation coefficients for Ni, Cu, and Al
    material_basis_path = './binaries/'
    filename = os.path.join(material_basis_path, 'material_basis.npy')
    material_basis = np.load(filename)
    num_materials_true = material_basis.shape[0]

    # Generate simulated noisy hyperspectral data and ground truth
    [noisy_hyper_projection, _, gt_hyper_projection] = generate_hyper_data(
        material_basis,
                                                                           num_angles=num_angles,
                                                                           detector_rows=detector_rows,
                                                                           detector_columns=detector_columns,
                                                                           dosage_rate=dosage_rate,
                                                                           material_density=material_density,
        verbose=verbose
    )
    noisy_hyper_projection = np.nan_to_num(noisy_hyper_projection, nan=0.0, posinf=0.0, neginf=0.0)  # Replace any NaNs or infs with zeros
    T = np.exp(-noisy_hyper_projection).reshape(-1, gt_hyper_projection.shape[-1])

    # Spoof simulated projection data which is not returned by generate_hyper_data
    height = detector_rows // 3
    width = detector_columns // 2
    thickness = 20 * np.sqrt((width//2)**2 - np.linspace(-width // 2, width // 2, width)**2)/ width
    material_projection = np.zeros((num_angles, detector_rows, detector_columns, num_materials_true)).astype(np.float32)
    material_projection[:, :height, width // 2:width + width // 2, 0] = material_density["Ni"] * thickness
    material_projection[:, 2 * height:, width // 2:width + width // 2, 1] = material_density["Cu"] * thickness
    material_projection[:, height:2 * height, width // 2:width + width // 2, 2] = material_density["Al"] * thickness
    material_projection = material_projection.reshape(-1, num_materials_true)

    # Perform hyperspectral denoising (dehydrate + rehydrate)
    print("Performing L2 factorization...")
    start_time = time.time()
    W, H, _ = dehydrate(noisy_hyper_projection,
                        dataset_type=dataset_type,
                        num_materials=num_materials_fit,
                        safety_factor=1,
                        verbose=verbose)
    W = W.reshape(np.prod(gt_hyper_projection.shape[:-1]), -1)
    H = H.reshape(-1, gt_hyper_projection.shape[-1])
    print('L2 factorization completed in: ', time.time() - start_time, ' seconds')

    ### Refine using nonnegative attenuation loss

    # Convert to JAX array
    T_jax = jnp.asarray(T, dtype=jnp.float32)

    kwargs = {
        'num_materials': num_materials_fit,
        'max_steps': 5000,
        'batch_size': None,
    }

    # Perform hyperspectral denoising
    start_time = time.time()
    W_newt1, H_newt1, i_newt = nnal_factorization(
        T_jax, method='quasi_newton', **kwargs
    )
    print('Newton reconstruction completed in: ', time.time() - start_time, ' seconds after ', i_newt, ' iterations')
    start_time = time.time()
    W_newt, H_newt, i_newt = nnal_factorization(
        T_jax, method='quasi_newton', **kwargs
    )
    print('Newton reconstruction completed in: ', time.time() - start_time, ' seconds after ', i_newt, ' iterations')
    start_time = time.time()
    W_mu1, H_mu1, i_mu = nnal_factorization(
        T_jax, method='mann_multiplicative', **kwargs
    )
    print('Multiplicative reconstruction completed in: ', time.time() - start_time, ' seconds after ', i_mu, ' iterations')
    start_time = time.time()
    W_mu, H_mu, i_mu = nnal_factorization(
        T_jax, method='mann_multiplicative', **kwargs
    )
    print('Multiplicative reconstruction completed in: ', time.time() - start_time, ' seconds after ', i_mu, ' iterations')

    W_newt = np.asarray(W_newt, dtype=np.float64)
    H_newt = np.asarray(H_newt, dtype=np.float64)
    W_mu = np.asarray(W_mu, dtype=np.float64)
    H_mu = np.asarray(H_mu, dtype=np.float64)

    # Compute least squares estimate of material coefficients for current projections
    theta_frob = np.linalg.lstsq(H.T, material_basis.T)[0].T
    theta_newt = np.linalg.lstsq(H_newt.T, material_basis.T)[0].T
    theta_mu = np.linalg.lstsq(H_mu.T, material_basis.T)[0].T

    # Plot reconstructed spectra
    compare_spectra(
        spectra_groups=[
            theta_frob @ H,
            theta_newt @ H_newt,
            theta_mu @ H_mu,
        ],
        ground_truth=material_basis,
        labels=['Ni', 'Cu', 'Al'],
        subtitles=[
            r'L$^2$ Loss',
            'Quasi-Newton',
            'Mann-Multiplicative'
        ],
        title=f'Material attenuation spectra reconstructions, Dosage: {dosage_rate}',
        x_label='Wavelength index',
        y_label='Attenuation',
        y_lim=(0, 1.1),
        filename='example_1_nonnegative_attenuation_loss_spectra_reconstruction.png'
    )

    # Plot reconstructed material coefficient maps
    plt.figure(figsize=(12, 12))
    plt.suptitle('Material projection reconstructions')
    row_max = np.max(material_projection, axis=0).reshape(1, 1, num_materials_true)
    image_dims = (detector_rows, detector_columns, num_materials_true)
    for i, (image, title) in enumerate([
            (material_projection.reshape(image_dims) / row_max, 'Ground Truth'),
            ((W @ np.linalg.pinv(theta_frob)).reshape(image_dims) / row_max, '$L^2$ Loss'),
            ((W_newt @ np.linalg.pinv(theta_newt)).reshape(image_dims) / row_max, 'Quasi-Newton'),
            ((W_mu @ np.linalg.pinv(theta_mu)).reshape(image_dims) / row_max, 'Mann-Multiplicative'),
        ]):
        ax = plt.subplot(2, 2, i + 1)
        ax.set_title(title)
        ax.imshow(image)
    plt.savefig('example_1_nonnegative_attenuation_loss_material_maps.png')

    # plt.show()

if __name__ == "__main__":
    main()
