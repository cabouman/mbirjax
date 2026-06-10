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

from mbirjax.hsnt import dehydrate, rehydrate, generate_hyper_data
from plot_utils import plot_images, plot_spectra


def newton_update(W, H, T):
    X = W @ H
    Z = np.exp(-X)
    init_loss = (Z + T * X).sum()

    dL_dW = (T - Z) @ H.T
    dL_dH = W.T @ (T - Z)

    d2L_dW2 = Z @ H.T**2
    d2L_dH2 = W.T**2 @ Z

    dW = dL_dW / (d2L_dW2 + 1e-10)
    dH = dL_dH / (d2L_dH2 + 1e-10)

    for learning_rate in np.logspace(0.5, -3, 15):
        W_temp = np.maximum(W - learning_rate * dW, 1e-10)
        H_temp = np.maximum(H - learning_rate * dH, 1e-10)
        X_temp = W_temp @ H_temp
        temp_loss = (np.exp(-X_temp) + T * X_temp).sum()
        if temp_loss < init_loss:
            W = W_temp
            H = H_temp
            break

    return W, H

def multiplicative_update(W, H, T):
    Z = np.exp(-W @ H)

    W_mult = ((Z @ H.T) / (T @ H.T) + 1) / 2
    H_mult = ((W.T @ Z) / (W.T @ T) + 1) / 2

    return W * W_mult, H * H_mult


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
    verbose = 2  # Verbosity level
    N = 1000  # Maximum number of fine-tuning iterations

    # Load theoretical linear attenuation coefficients for Ni, Cu, and Al
    material_basis_path = './binaries/'
    filename = os.path.join(material_basis_path, 'material_basis.npy')
    material_basis_0 = np.load(filename)

    with open('denoising_results.csv', 'w') as f:
        f.write('num_pixels,num_wavelengths,random_seed,MSE_spectra_newt,MSE_spectra_mu,MSE_materials_newt,MSE_materials_mu\n')

        for size in np.logspace(2, 8, 7, base=2, dtype=int):
            detector_rows = size
            detector_columns = size
            for downsample in np.logspace(np.log2(1200), 0, 10, base=2, dtype=int):
                material_basis = material_basis_0[:, ::downsample]
                for seed in range(10):
                    np.random.seed(seed)

                    print(f'\nSimulating data with size {size}x{size}, downsample factor {downsample}, and seed {seed}...')

                    # Generate simulated noisy hyperspectral data and ground truth
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
                    noisy_hyper_projection = np.nan_to_num(noisy_hyper_projection, nan=0.0, posinf=0.0, neginf=0.0)  # Replace any NaNs or infs with zeros
                    T = np.exp(-noisy_hyper_projection).reshape(-1, gt_hyper_projection.shape[-1])

                    # Spoof simulated projection data for 3 materials (Ni, Cu, and Al)
                    height = detector_rows // 3
                    width = detector_columns // 2
                    thickness = 20 * np.sqrt((width//2)**2 - np.linspace(-width // 2, width // 2, width)**2)/ width
                    material_projection = np.zeros((num_angles, detector_rows, detector_columns, num_materials)).astype(np.float32)
                    material_projection[:, :height, width // 2:width + width // 2, 0] = material_density["Ni"] * thickness
                    material_projection[:, 2 * height:, width // 2:width + width // 2, 1] = material_density["Cu"] * thickness
                    material_projection[:, height:2 * height, width // 2:width + width // 2, 2] = material_density["Al"] * thickness
                    material_projection = material_projection.reshape(-1, num_materials)

                    # Refine using nonnegative attenuation loss
                    W_newt = np.random.rand(np.prod(gt_hyper_projection.shape[:-1]), num_materials)
                    H_newt = np.random.rand(num_materials, gt_hyper_projection.shape[-1])
                    W_mu = W_newt.copy()
                    H_mu = H_newt.copy()
                    prev_loss_newt = (np.exp(-W_newt @ H_newt) + T * (W_newt @ H_newt)).sum()
                    prev_loss_mu = prev_loss_newt

                    rel_tol = 1e-8

                    for i in range(N):
                        W_newt, H_newt = newton_update(W_newt, H_newt, T)
                        loss_newt = (np.exp(-W_newt @ H_newt) + T * (W_newt @ H_newt)).sum()
                        if abs(loss_newt - prev_loss_newt) / prev_loss_newt < rel_tol:
                            break
                        prev_loss_newt = loss_newt

                    for i in range(N):
                        W_mu, H_mu = multiplicative_update(W_mu, H_mu, T)
                        loss_mu = (np.exp(-W_mu @ H_mu) + T * (W_mu @ H_mu)).sum()
                        if abs(loss_mu - prev_loss_mu) / prev_loss_mu < rel_tol:
                            break
                        prev_loss_mu = loss_mu

                    # Compute least squares estimate of material coefficients for current projections
                    theta_newt = np.linalg.lstsq(H_newt.T, material_basis.T)[0].T
                    theta_mu = np.linalg.lstsq(H_mu.T, material_basis.T)[0].T

                    # Compute MSE of material spectra
                    Ds_newt = np.linalg.norm(theta_newt @ H_newt - material_basis) ** 2 / material_basis.size
                    Ds_mu = np.linalg.norm(theta_mu @ H_mu - material_basis) ** 2 / material_basis.size

                    # Compute MSE of material projections
                    Dm_newt = np.linalg.norm(W_newt @ np.linalg.pinv(theta_newt) - material_projection) ** 2 / material_projection.size
                    Dm_mu = np.linalg.norm(W_mu @ np.linalg.pinv(theta_mu) - material_projection) ** 2 / material_projection.size

                    f.write(f'{size**2},{material_basis.shape[1]},{seed},{Ds_newt},{Ds_mu},{Dm_newt},{Dm_mu}\n')
                    f.flush()

if __name__ == "__main__":
    main()
