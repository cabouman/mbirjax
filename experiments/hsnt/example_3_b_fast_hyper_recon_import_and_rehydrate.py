"""
Hyperspectral Dehydration & Rehydration
---------------------------------------

Examples 3(a) + 3(b) demonstrate the use of dehydration and rehydration for fast hyperspectral reconstruction.
A simulated hyperspectral neutron dataset containing three materials (Ni, Cu, and Al) is used for the purpose.
This script - example 3(b) - imports the dehydrated reconstructions from 3(a) and performs rehydration.
"""

import os
import numpy as np
import time
import matplotlib.pyplot as plt

from mbirjax.hsnt import import_hsnt_data_hdf5, rehydrate
from plot_utils import plot_images, plot_spectra


def main():
    start_time = time.time()

    # Set input path and dataset name
    input_path = './processed_data/'  # Path to import data
    dataset_name = "hsnt_dehydrated_recons"  # Name of the input dataset

    # Display parameters
    display_wave_idx = 200  # Wavelength index of displayed image
    display_slice_idx = 8  # Slice index of displayed image
    display_vox_idx = [32, 32, 8]  # Voxel index [column, column, row] of displayed spectra
    vmax = 0.1  # Maximum voxel value for displayed image
    vmin = 0  # Minimum voxel value for displayed image
    y_lim_attenuation = (0.04, 0.09)  # (y_min, y_max) to set y-axis range for attenuation spectra
    y_lim_transmission = (0.91, 0.96)  # (y_min, y_max) to set y-axis range for transmission spectra

    # Import dehydrated reconstructions from HDF5 file
    filename = os.path.join(input_path, dataset_name + ".h5")
    hsnt_dehydrated_recons, metadata_dehydrated = import_hsnt_data_hdf5(filename, dataset_name)

    # Perform rehydration
    hsnt_recons = rehydrate(hsnt_dehydrated_recons)

    # Plot hyperspectral projections and spectra
    plot_images(images=[hsnt_recons[:, :, display_slice_idx, display_wave_idx]],
                titles=['Hyperspectral reconstruction' +
                        '\n\nSlice index: ' + str(display_slice_idx) +
                        '\nWavelength index: ' + str(display_wave_idx)],
                vmax=vmax, vmin=vmin)

    plot_spectra(spectra=[hsnt_recons[display_vox_idx[0], display_vox_idx[1], display_vox_idx[2], :]],
                 title='Single voxel spectra (attenuation) for reconstructed data',
                 x_label='wavelength index',
                 y_label='attenuation',
                 y_lim=y_lim_attenuation)

    print('Total time elapsed: ', time.time() - start_time, ' seconds')

    plt.show()


if __name__ == "__main__":
    main()
