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
    learning_rate = 0.01  # Learning rate for Newton updates
    N = 100  # Number of fine-tuning iterations

    # Display parameters
    display_wave_idx = 200  # Wavelength index of displayed images
    display_pix_idx = [10, 32]  # Pixel index [row, column] of displayed spectra
    vmax = 2.5  # Maximum pixel value for displayed images
    vmin = 0  # Minimum pixel value for displayed images
    y_lim_attenuation = (1, 2.7)  # (y_min, y_max) to set y-axis range for attenuation spectra
    y_lim_transmission = (0, 0.35)  # (y_min, y_max) to set y-axis range for transmission spectra
    output_path = './output_plots/'  # path to export output plots
    os.makedirs(output_path, exist_ok=True)  # Make output directory if it does not exist

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

    # Refine using nonnegative attenuation loss
    W_newt = W.copy().reshape(np.prod(gt_hyper_projection.shape[:-1]), num_materials)
    H_newt = H.copy().reshape(num_materials, gt_hyper_projection.shape[-1])
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

        W_newt = np.maximum(W_newt - learning_rate * dL_dA / (d2L_dA2 + 1e-10), 1e-10)
        H_newt = np.maximum(H_newt - learning_rate * dL_dB / (d2L_dB2 + 1e-10), 1e-10)

        # Multiplicative update
        Z_mu = np.exp(-W_mu @ H_mu)

        W_mult = ((Z_mu @ H_mu.T) / (T @ H_mu.T) + 1) / 2
        H_mult = ((W_mu.T @ Z_mu) / (W_mu.T @ T) + 1) / 2

        W_mu = W_mu * W_mult
        H_mu = H_mu * H_mult

        newton_hyper_projection = (W_newt @ H_newt).reshape(gt_hyper_projection.shape)
        mult_hyper_projection = (W_mu @ H_mu).reshape(gt_hyper_projection.shape)

        # Plot hyperspectral projections and spectra
        if verbose > 1:
            plot_images(images=[gt_hyper_projection[0, :, :, display_wave_idx],
                                noisy_hyper_projection[0, :, :, display_wave_idx],
                                frob_hyper_projection[0, :, :, display_wave_idx],
                                newton_hyper_projection[0, :, :, display_wave_idx],
                                mult_hyper_projection[0, :, :, display_wave_idx]],
                        titles=['Fig (a): Ground truth hyperspectral projection \n\nWavelength index: ' + str(display_wave_idx),
                                'Fig (b): Noisy hyperspectral projection \n\nWavelength index: ' + str(display_wave_idx),
                                'Fig (c): Frobenius hyperspectral projection \n\nWavelength index: ' + str(display_wave_idx),
                                'Fig (d): Newton hyperspectral projection \n\nWavelength index: ' + str(display_wave_idx),
                                'Fig (e): Multiplicative hyperspectral projection \n\nWavelength index: ' + str(display_wave_idx)],
                        vmax=vmax, vmin=vmin,
                        filename=f'{output_path}/example_1_nonnegative_attenuation_loss_projection_{i}.png')

            plot_spectra(spectra=[noisy_hyper_projection[0, display_pix_idx[0], display_pix_idx[1], :],
                                  frob_hyper_projection[0, display_pix_idx[0], display_pix_idx[1], :],
                                  newton_hyper_projection[0, display_pix_idx[0], display_pix_idx[1], :],
                                  mult_hyper_projection[0, display_pix_idx[0], display_pix_idx[1], :],
                                  gt_hyper_projection[0, display_pix_idx[0], display_pix_idx[1], :]],
                        labels=['Noisy', 'Frobenius', 'Denoised Newton', 'Denoised MU', 'Ground Truth'],
                        title='Single pixel spectra (attenuation) for noisy and denoised data',
                        x_label='wavelength index',
                        y_label='attenuation',
                        y_lim=y_lim_attenuation,
                        filename=f'{output_path}/example_1_nonnegative_attenuation_loss_spectra_{i}.png')

            plot_spectra(spectra=[np.exp(-noisy_hyper_projection[0, display_pix_idx[0], display_pix_idx[1], :]),
                                  np.exp(-frob_hyper_projection[0, display_pix_idx[0], display_pix_idx[1], :]),
                                  np.exp(-newton_hyper_projection[0, display_pix_idx[0], display_pix_idx[1], :]),
                                  np.exp(-mult_hyper_projection[0, display_pix_idx[0], display_pix_idx[1], :]),
                                  np.exp(-gt_hyper_projection[0, display_pix_idx[0], display_pix_idx[1], :])],
                        labels=['Noisy', 'Frobenius', 'Denoised Newton', 'Denoised MU', 'Ground Truth'],
                        title='Single pixel spectra (transmission) for noisy and denoised data',
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
