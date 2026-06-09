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
    N = 500  # Number of fine-tuning iterations

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

    gt_loss = (np.exp(-gt_hyper_projection) + T.reshape(gt_hyper_projection.shape) * gt_hyper_projection).sum()
    frob_loss = (np.exp(-frob_hyper_projection) + T.reshape(gt_hyper_projection.shape) * frob_hyper_projection).sum() / gt_loss
    newton_loss = np.inf

    # Refine using nonnegative attenuation loss
    W_newt = np.random.rand(*W.shape)
    H_newt = np.random.rand(*H.shape)
    W_mu = W_newt.copy()
    H_mu = H_newt.copy()
    D_newt = []
    D_mu = []
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
        for learning_rate in np.logspace(-2, 0, 5):
            W_temp = np.maximum(W_newt - learning_rate * dW, 1e-10)
            H_temp = np.maximum(H_newt - learning_rate * dH, 1e-10)
            proj = W_temp @ H_temp
            temp_loss = (np.exp(-proj) + T * proj).sum() / gt_loss
            if temp_loss > newton_loss:
                break
            W_newt = W_temp
            H_newt = H_temp
            newton_hyper_projection = proj.reshape(gt_hyper_projection.shape)
            newton_loss = temp_loss

        # Multiplicative update
        Z_mu = np.exp(-W_mu @ H_mu)

        W_mult = ((Z_mu @ H_mu.T) / (T @ H_mu.T) + 1) / 2
        H_mult = ((W_mu.T @ Z_mu) / (W_mu.T @ T) + 1) / 2

        W_mu = W_mu * W_mult
        H_mu = H_mu * H_mult
        mult_hyper_projection = (W_mu @ H_mu).reshape(gt_hyper_projection.shape)
        mult_loss = (np.exp(-mult_hyper_projection) + T.reshape(gt_hyper_projection.shape) * mult_hyper_projection).sum() / gt_loss

        # Compute least squares estimate of material coefficients for current projections
        theta_newt = np.linalg.lstsq(H_newt.T, material_basis.T)[0].T
        theta_mu = np.linalg.lstsq(H_mu.T, material_basis.T)[0].T

        # Compute MSE of material spectra
        D_newt.append(np.linalg.norm(theta_newt @ H_newt - material_basis) ** 2 / material_basis.size)
        D_mu.append(np.linalg.norm(theta_mu @ H_mu - material_basis) ** 2 / material_basis.size)

        # Plot hyperspectral projections and spectra
        if verbose > 1 and (i % 20 == 0 or i == N - 1):  # Plot at regular intervals and the last iteration
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
                                f'Frobenius (Rel Loss: {frob_loss:.5f})',
                                f'Denoised Newton (Rel Loss: {newton_loss:.5f})',
                                f'Denoised MU (Rel Loss: {mult_loss:.5f})',
                                f'Ground Truth'],
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
                                f'Frobenius (Rel Loss: {frob_loss:.5f})',
                                f'Denoised Newton (Rel Loss: {newton_loss:.5f})',
                                f'Denoised MU (Rel Loss: {mult_loss:.5f})',
                                f'Ground Truth'],
                        title=f'Single pixel spectra (transmission) for noisy and denoised data\nIteration: {i+1}/{N}',
                        x_label='wavelength index',
                        y_label='transmission',
                        y_lim=y_lim_transmission,
                        filename=f'{output_path}/example_1_nonnegative_attenuation_loss_spectra_transmission_{i}.png')

            plt.close('all')

    # Plot MSE of spectra reconstructions across iterations
    plt.figure(figsize=(10, 6))
    plt.plot(D_newt, label='Newton')
    plt.plot(D_mu, label='Multiplicative')
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.title('Mean Squared Error of Spectra Reconstructions')
    plt.legend()
    plt.ylim(0, 0.005)
    plt.tight_layout()
    plt.savefig(f'example_1_nonnegative_attenuation_loss_distance_to_material_basis_{gt_hyper_projection.shape[-1]}.png')

    # Plot reconstructed spectra
    plot_spectra(spectra=[(theta_newt @ H_newt)[0],
                          (theta_newt @ H_newt)[1],
                          (theta_newt @ H_newt)[2],
                          (theta_mu @ H_mu)[0],
                          (theta_mu @ H_mu)[1],
                          (theta_mu @ H_mu)[2],
                          material_basis[0],
                          material_basis[1],
                          material_basis[2]],
                labels=["Ni Recon (Newton)", "Cu Recon (Newton)", "Al Recon (Newton)",
                        "Ni Recon (MU)", "Cu Recon (MU)", "Al Recon (MU)",
                        "Ni Basis", "Cu Basis", "Al Basis"],
                title='Material attenuation spectra reconstructions',
                x_label='wavelength index',
                y_label='attenuation',
                filename=f'example_1_nonnegative_attenuation_loss_spectra_reconstruction.png')

    # Plot reconstructed material coefficient maps
    plot_images(images=[(W_newt @ np.linalg.pinv(theta_newt)).reshape(detector_rows, detector_columns, num_materials),
                        (W_mu @ np.linalg.pinv(theta_mu)).reshape(detector_rows, detector_columns, num_materials)],
                titles=[f'Fig (a): Newton material coefficient maps\nIteration: {N}',
                        f'Fig (b): Multiplicative material coefficient maps\nIteration: {N}]'],
                filename='example_1_nonnegative_attenuation_loss_material_maps.png')

    # Save GIFs of projections and spectra across iterations
    if verbose > 1:
        images = []
        for i in range(N):
            if os.path.exists(f'{output_path}/example_1_nonnegative_attenuation_loss_projection_{i}.png'):
                images.append(Image.open(f'{output_path}/example_1_nonnegative_attenuation_loss_projection_{i}.png'))
        images[0].save('example_1_nonnegative_attenuation_loss_projection.gif', save_all=True, append_images=images[1:], delay=0.01, loop=0)
        images = []
        for i in range(N):
            if os.path.exists(f'{output_path}/example_1_nonnegative_attenuation_loss_spectra_{i}.png'):
                images.append(Image.open(f'{output_path}/example_1_nonnegative_attenuation_loss_spectra_{i}.png'))
        images[0].save('example_1_nonnegative_attenuation_loss_spectra.gif', save_all=True, append_images=images[1:], delay=0.01, loop=0)
        images = []
        for i in range(N):
            if os.path.exists(f'{output_path}/example_1_nonnegative_attenuation_loss_spectra_transmission_{i}.png'):
                images.append(Image.open(f'{output_path}/example_1_nonnegative_attenuation_loss_spectra_transmission_{i}.png'))
        images[0].save('example_1_nonnegative_attenuation_loss_spectra_transmission.gif', save_all=True, append_images=images[1:], delay=0.01, loop=0)

    plt.show()

if __name__ == "__main__":
    main()
