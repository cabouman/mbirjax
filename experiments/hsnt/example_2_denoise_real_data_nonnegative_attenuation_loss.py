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

from mbirjax.hsnt import hyper_denoise, dehydrate, rehydrate, import_hsnt_data_hdf5, export_hsnt_data_hdf5
from plot_utils import plot_images, plot_spectra


def main():
    start_time = time.time()

    # Choose dataset from '0.8C_Ni_cylinder', '1.6C_Ni_cylinder', '2.4C_Ni_cylinder', '4.8C_Ni_cylinder', '9.6C_Ni_cylinder'
    dataset_name = '2_4c_Ni_cylinder_dataset'
    input_path = './input_data/2_4c/processed_data_2_4c_Ni_cylinder.h5'  # path to import input noisy data
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

    # Fix seed for random number generation
    np.random.seed(129)

    # Import real hyperspectral data
    hsnt_data, metadata = import_hsnt_data_hdf5(input_path, dataset_name)
    dataset_type = metadata['dataset_type']
    wavelengths = metadata['wavelengths']

    if verbose >= 1:
        print("Hyperspectral data shape: ", hsnt_data.shape)

    T = np.exp(-hsnt_data).reshape(-1, hsnt_data.shape[-1])

    # Perform hyperspectral denoising (dehydrate + rehydrate)
    W, H, _ = dehydrate(hsnt_data,
                        dataset_type=dataset_type,
                        num_materials=num_materials,
                        safety_factor=1,
                        verbose=verbose)
    W = W.reshape(-1, num_materials)
    H = H.reshape(num_materials, -1)

    # Refine using nonnegative attenuation loss
    for i in range(100):  # Run for a fixed number of iterations
        print(f'Iteration {i + 1}/100')

        # Multiplicative update
        Z = np.exp(-W @ H)

        W_mult = ((Z @ H.T) / (T @ H.T) + 1) / 2
        H_mult = ((W.T @ Z) / (W.T @ T) + 1) / 2

        W *= W_mult
        H *= H_mult

    hsnt_denoised = (W @ H).reshape(hsnt_data.shape)

    # Write out denoised/rehydrated data
    filename_denoised = os.path.join(output_path, dataset_name+'_dataset_denoised_nnal.h5')
    export_hsnt_data_hdf5(filename_denoised, hsnt_denoised, metadata)

    # Plot hyperspectral projections and spectra
    if verbose > 1:
        plot_images(images=[hsnt_data[0, :, :, display_wave_idx],
                            hsnt_denoised[0, :, :, display_wave_idx]],
                    titles=['Fig (a): Noisy hyperspectral projection \n\nWavelength index: ' + str(display_wave_idx),
                            'Fig (b): Denoised hyperspectral projection \n\nWavelength index: ' + str(display_wave_idx)],
                    vmax=vmax, vmin=vmin,
                    filename="cylinder_nnal.png")

        plot_spectra(spectra=[hsnt_data[0, display_pix_idx[0], display_pix_idx[1], :],
                              hsnt_denoised[0, display_pix_idx[0], display_pix_idx[1], :]],
                     labels=['Noisy', 'Denoised'],
                     title='Single pixel spectra (attenuation) for noisy and denoised data',
                     x_label='wavelength (Angstrom)',
                     y_label='attenuation',
                     y_lim=y_lim_attenuation,
                     wavelengths=wavelengths,
                     filename="cylinder_attenuation_nnal.png")

        plot_spectra(spectra=[np.exp(-hsnt_data[0, display_pix_idx[0], display_pix_idx[1], :]),
                              np.exp(-hsnt_denoised[0, display_pix_idx[0], display_pix_idx[1], :])],
                     labels=['Noisy', 'Denoised'],
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
