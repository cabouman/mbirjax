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
    N = 300  # Number of fine-tuning iterations

    # Display parameters
    display_wave_idx = 200  # Wavelength index of displayed images
    display_pix_idx = [10, 32]  # Pixel index [row, column] of displayed spectra
    vmax = 2.5  # Maximum pixel value for displayed images
    vmin = 0  # Minimum pixel value for displayed images
    y_lim_attenuation = (1, 2.7)  # (y_min, y_max) to set y-axis range for attenuation spectra
    y_lim_transmission = (0, 0.35)  # (y_min, y_max) to set y-axis range for transmission spectra
    output_path = './output_plots/'  # path to export output plots

    # Delete old output plots if they exist
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

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
    T = np.exp(-noisy_hyper_projection).reshape(-1, gt_hyper_projection.shape[-1])

    # Perform hyperspectral denoising (dehydrate + rehydrate)
    W, H, _ = dehydrate(noisy_hyper_projection,
                        dataset_type=dataset_type,
                        num_materials=num_materials,
                        safety_factor=1,
                        verbose=verbose)
    frob_hyper_projection = rehydrate([W, H, dataset_type])
    W = W.reshape(np.prod(gt_hyper_projection.shape[:-1]), num_materials)
    H = H.reshape(num_materials, gt_hyper_projection.shape[-1])

    frob_loss = (np.exp(-frob_hyper_projection) + T.reshape(gt_hyper_projection.shape) * frob_hyper_projection).sum()
    gt_loss = (np.exp(-gt_hyper_projection) + T.reshape(gt_hyper_projection.shape) * gt_hyper_projection).sum() / frob_loss
    newton_loss = 1  # Initialize to 1 since we will be comparing to the Frobenius loss

    # Refine using nonnegative attenuation loss
    W_newt = W.copy()
    H_newt = H.copy()
    W_mu = W_newt.copy()
    H_mu = H_newt.copy()
    for i in range(N):  # Run for a fixed number of iterations
        print(f'Iteration {i + 1}/{N}')

        # Newton update
        Z_newt = np.exp(-W_newt @ H_newt)

        dL_dA = (T - Z_newt) @ H_newt.T
        dL_dB = W_newt.T @ (T - Z_newt)

        d2L_dA2 = Z_newt @ H_newt.T**2
        d2L_dB2 = W_newt.T**2 @ Z_newt

        dW = dL_dA / (d2L_dA2 + 1e-10)
        dH = dL_dB / (d2L_dB2 + 1e-10)

        # Compute learning rate using line search
        for learning_rate in np.logspace(-3, 0, 20):
            W_temp = np.maximum(W_newt - learning_rate * dW, 1e-10)
            H_temp = np.maximum(H_newt - learning_rate * dH, 1e-10)
            proj = W_temp @ H_temp
            temp_loss = (np.exp(-proj) + T * proj).sum() / frob_loss
            if temp_loss > newton_loss:
                break
            W_newt = W_temp
            H_newt = H_temp
            newton_loss = temp_loss

        # Multiplicative update
        Z_mu = np.exp(-W_mu @ H_mu)

        W_mult = ((Z_mu @ H_mu.T) / (T @ H_mu.T) + 1) / 2
        H_mult = ((W_mu.T @ Z_mu) / (W_mu.T @ T) + 1) / 2

        W_mu = W_mu * W_mult
        H_mu = H_mu * H_mult

        newton_hyper_projection = (W_newt @ H_newt).reshape(gt_hyper_projection.shape)
        mult_hyper_projection = (W_mu @ H_mu).reshape(gt_hyper_projection.shape)

        mult_loss = (np.exp(-mult_hyper_projection) + T.reshape(gt_hyper_projection.shape) * mult_hyper_projection).sum() / frob_loss

        # Plot hyperspectral projections and spectra
        if verbose > 1:
            plot_images(images=[gt_hyper_projection[0, :, :, display_wave_idx],
                                noisy_hyper_projection[0, :, :, display_wave_idx],
                                frob_hyper_projection[0, :, :, display_wave_idx],
                                newton_hyper_projection[0, :, :, display_wave_idx],
                                mult_hyper_projection[0, :, :, display_wave_idx]],
                        titles=[f'Fig (a): Ground truth hyperspectral projection\nWavelength index: {display_wave_idx}\n',
                                f'Fig (b): Noisy hyperspectral projection\nWavelength index: {display_wave_idx}\n',
                                f'Fig (c): Frobenius hyperspectral projection\nWavelength index: {display_wave_idx}\n',
                                f'Fig (d): Newton hyperspectral projection\nWavelength index: {display_wave_idx}\nIteration: {i+1}/{N}',
                                f'Fig (e): Multiplicative hyperspectral projection\nWavelength index: {display_wave_idx}\nIteration: {i+1}/{N}'],
                        vmax=vmax, vmin=vmin,
                        filename=f'{output_path}/example_1_nonnegative_attenuation_loss_projection_{i}.png')

            plot_spectra(spectra=[noisy_hyper_projection[0, display_pix_idx[0], display_pix_idx[1], :],
                                  frob_hyper_projection[0, display_pix_idx[0], display_pix_idx[1], :],
                                  newton_hyper_projection[0, display_pix_idx[0], display_pix_idx[1], :],
                                  mult_hyper_projection[0, display_pix_idx[0], display_pix_idx[1], :],
                                  gt_hyper_projection[0, display_pix_idx[0], display_pix_idx[1], :]],
                        labels=[f'Noisy (Dosage: {dosage_rate})',
                                f'Frobenius (Rel Loss: 1.00000)',
                                f'Denoised Newton (Rel Loss: {newton_loss:.5f})',
                                f'Denoised MU (Rel Loss: {mult_loss:.5f})',
                                f'Ground Truth (Rel Loss: {gt_loss:.5f})'],
                        title=f'Single pixel spectra (attenuation) for noisy and denoised data\nIteration: {i+1}/{N}',
                        x_label='wavelength index',
                        y_label='attenuation',
                        y_lim=y_lim_attenuation,
                        filename=f'{output_path}/example_1_nonnegative_attenuation_loss_spectra_{i}.png')

            plot_spectra(spectra=[np.exp(-noisy_hyper_projection[0, display_pix_idx[0], display_pix_idx[1], :]),
                                  np.exp(-frob_hyper_projection[0, display_pix_idx[0], display_pix_idx[1], :]),
                                  np.exp(-newton_hyper_projection[0, display_pix_idx[0], display_pix_idx[1], :]),
                                  np.exp(-mult_hyper_projection[0, display_pix_idx[0], display_pix_idx[1], :]),
                                  np.exp(-gt_hyper_projection[0, display_pix_idx[0], display_pix_idx[1], :])],
                        labels=[f'Noisy (Dosage: {dosage_rate})',
                                f'Frobenius (Rel Loss: 1.00000)',
                                f'Denoised Newton (Rel Loss: {newton_loss:.5f})',
                                f'Denoised MU (Rel Loss: {mult_loss:.5f})',
                                f'Ground Truth (Rel Loss: {gt_loss:.5f})'],
                        title=f'Single pixel spectra (transmission) for noisy and denoised data\nIteration: {i+1}/{N}',
                        x_label='wavelength index',
                        y_label='transmission',
                        y_lim=y_lim_transmission,
                        filename=f'{output_path}/example_1_nonnegative_attenuation_loss_spectra_transmission_{i}.png')

            plt.close('all')

    # Save GIFs of projections and spectra across iterations
    if verbose > 1:
        images = []
        for i in range(N):
            images.append(Image.open(f'{output_path}/example_1_nonnegative_attenuation_loss_projection_{i}.png'))
        images[0].save('example_1_nonnegative_attenuation_loss_projection.gif', save_all=True, append_images=images[1:], delay=0.01, loop=0)
        images = []
        for i in range(N):
            images.append(Image.open(f'{output_path}/example_1_nonnegative_attenuation_loss_spectra_{i}.png'))
        images[0].save('example_1_nonnegative_attenuation_loss_spectra.gif', save_all=True, append_images=images[1:], delay=0.01, loop=0)
        images = []
        for i in range(N):
            images.append(Image.open(f'{output_path}/example_1_nonnegative_attenuation_loss_spectra_transmission_{i}.png'))
        images[0].save('example_1_nonnegative_attenuation_loss_spectra_transmission.gif', save_all=True, append_images=images[1:], delay=0.01, loop=0)

if __name__ == "__main__":
    main()
