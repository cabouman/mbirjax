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
    dataset_name = '2.4C_Ni_cylinder'
    input_path = './input_data/'  # path to import input noisy data
    output_path = './output_data/'  # path to export output denoised data
    os.makedirs(output_path, exist_ok=True)  # Make output directory if it does not exist

    # Denoiser parameters
    subspace_dimension = None  # Subspace dimension
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
    filename = os.path.join(input_path, dataset_name+'_dataset.h5')
    hsnt_data, metadata_dict = import_hsnt_data_hdf5(filename, dataset_name)
    dataset_type = metadata_dict['dataset_type']
    wavelengths = metadata_dict['wavelengths']

    if verbose >= 1:
        print("Hyperspectral data shape: ", hsnt_data.shape)

    if test_denoise:
        if verbose >= 1:
            print("Running hyperspectral denoising (i.e., dehydrate + rehydrate)")
        hsnt_denoised = hyper_denoise(hsnt_data, dataset_type=dataset_type, subspace_dimension=subspace_dimension, verbose=verbose)
    else:
        if verbose >= 1:
            print("Running hyperspectral dehydrate followed by rehydrate (i.e., denoising)")
        hsnt_dehydrated = dehydrate(hsnt_data, dataset_type=dataset_type, subspace_dimension=subspace_dimension, verbose=verbose)
        hsnt_denoised = rehydrate(hsnt_dehydrated)

        # Write out dehydrated data
        filename_dehydrated = os.path.join(output_path, dataset_name+'_dataset_dehydrated.h5')
        export_hsnt_data_hdf5(filename_dehydrated, hsnt_dehydrated, metadata_dict)

    # Write out denoised/rehydrated data
    filename_denoised = os.path.join(output_path, dataset_name+'_dataset_denoised.h5')
    export_hsnt_data_hdf5(filename_denoised, hsnt_denoised, metadata_dict)

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
                     wavelengths=wavelengths)

        plot_spectra(spectra=[np.exp(-hsnt_data[display_pix_idx[0], display_pix_idx[1], :]),
                              np.exp(-hsnt_denoised[display_pix_idx[0], display_pix_idx[1], :])],
                     labels=['Noisy', 'Denoised'],
                     title='Single pixel spectra (transmission) for noisy and denoised data',
                     x_label='wavelength (Angstrom)',
                     y_label='transmission',
                     y_lim=y_lim_transmission,
                     wavelengths=wavelengths)

    print('Total time elapsed: ', time.time() - start_time, ' seconds')

    plt.show()


if __name__ == "__main__":
    main()
