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

from mbirjax.hsnt import create_hsnt_metadata, export_hsnt_data_hdf5
from mbirjax.preprocess import read_tif_stack_dir
from plot_utils import plot_images, plot_spectra


def main():
    start_time = time.time()

    # Choose dataset from '0.8C_Ni_cylinder', '1.6C_Ni_cylinder', '2.4C_Ni_cylinder', '4.8C_Ni_cylinder', '9.6C_Ni_cylinder'
    dataset_name = '2_4c'
    input_path = './input_data/'  # path to import input noisy data
    output_path = './output_data/'  # path to export output denoised data
    os.makedirs(output_path, exist_ok=True)  # Make output directory if it does not exist

    # Denoiser parameters
    num_materials = 3  # Number of materials
    verbose = 2  # Verbosity level
    test_denoise = False  # If True, test hyper_denoise; if False, test dehydrate + rehydrate sequence

    # Display parameters
    display_wave_idx = 500  # Wavelength index of displayed images
    display_pix_idx = [200, 200]  # Pixel index [row, column] of displayed spectra
    vmax = 2  # Maximum pixel value for displayed images
    vmin = 0  # Minimum pixel value for displayed images
    y_lim_attenuation = (0, 3)  # (y_min, y_max) to set y-axis range for attenuation spectra
    y_lim_transmission = (0, 0.75)  # (y_min, y_max) to set y-axis range for transmission spectra

    # Fix seed for random number generation
    np.random.seed(129)

    # Import real hyperspectral data
    object_scan = read_tif_stack_dir(os.path.join(input_path, dataset_name, "Ni_cylinder_projections"))
    blank_scan = read_tif_stack_dir(os.path.join(input_path, dataset_name, "open_beam", "observation_01"))
    hsnt_data = object_scan / blank_scan

    if verbose >= 1:
        print("Hyperspectral data shape: ", hsnt_data.shape)

    # Perform hyperspectral denoising (dehydrate + rehydrate)
    T = hsnt_data.reshape(-1, hsnt_data.shape[-1])
    W = np.random.rand(T.shape[0], num_materials)  # Random initialization of W
    H = np.random.rand(num_materials, T.shape[1])  # Random initialization of H
    learning_rate = 0.2
    for i in range(100):  # Run for a fixed number of iterations
        Z = np.exp(-W @ H)

        dL_dA = (T - Z) @ H.T
        dL_dB = W.T @ (T - Z)

        d2L_dA2 = Z @ H.T**2
        d2L_dB2 = W.T**2 @ Z

        W = np.maximum(W - learning_rate * dL_dA / (d2L_dA2 + 1e-10), 1e-10)
        H = np.maximum(H - learning_rate * dL_dB / (d2L_dB2 + 1e-10), 1e-10)

    hsnt_denoised = (W @ H).reshape(hsnt_data.shape)

    # Plot hyperspectral projections and spectra
    if verbose > 1:
        plot_images(images=[hsnt_data[:, :, display_wave_idx],
                            hsnt_denoised[:, :, display_wave_idx]],
                    titles=['Fig (a): Noisy hyperspectral projection \n\nWavelength index: ' + str(display_wave_idx),
                            'Fig (b): Denoised hyperspectral projection \n\nWavelength index: ' + str(display_wave_idx)],
                    vmax=vmax, vmin=vmin)

        plot_spectra(spectra=[hsnt_data[display_pix_idx[0], display_pix_idx[1], :],
                              hsnt_denoised[display_pix_idx[0], display_pix_idx[1], :]],
                     labels=['Noisy', 'Denoised'],
                     title='Single pixel spectra (attenuation) for noisy and denoised data',
                     x_label='wavelength (Angstrom)',
                     y_label='attenuation',
                     y_lim=y_lim_attenuation,
                     wavelengths=None)

        plot_spectra(spectra=[np.exp(-hsnt_data[display_pix_idx[0], display_pix_idx[1], :]),
                              np.exp(-hsnt_denoised[display_pix_idx[0], display_pix_idx[1], :])],
                     labels=['Noisy', 'Denoised'],
                     title='Single pixel spectra (transmission) for noisy and denoised data',
                     x_label='wavelength (Angstrom)',
                     y_label='transmission',
                     y_lim=y_lim_transmission,
                     wavelengths=None)

    print('Total time elapsed: ', time.time() - start_time, ' seconds')

    plt.show()

    # Write out denoised/rehydrated data
    filename_denoised = os.path.join(output_path, dataset_name+'_dataset_denoised.h5')
    export_hsnt_data_hdf5(filename_denoised, hsnt_denoised, dict())


if __name__ == "__main__":
    main()
