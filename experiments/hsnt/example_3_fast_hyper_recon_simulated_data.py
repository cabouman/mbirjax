"""
Hyperspectral Dehydration & Rehydration
---------------------------------------

This script demonstrates the use of dehydration and rehydration for fast hyperspectral reconstruction.
A simulated hyperspectral neutron dataset containing three materials (Ni, Cu, and Al) is used for the purpose.
"""

import os
import numpy as np
import time
import matplotlib.pyplot as plt
import mbirjax as mj

from mbirjax.hsnt import dehydrate, rehydrate, generate_hyper_data
from plot_utils import plot_images, plot_spectra


def main():
    start_time = time.time()

    # Simulation parameters
    num_angles = 16  # Number of view angles
    detector_rows = 64  # Number of rows in the detector
    detector_columns = 64  # Number of columns in the detector
    dosage_rate = 300  # Neutron dosage rate
    material_density = {"Ni": 0.25, "Cu": 0.25, "Al": 0.75}  # Define material density (vol. fraction)
    dataset_type = 'attenuation'  # Choose between 'attenuation' or 'transmission'

    # Fast hyperspectral reconstruction parameters
    num_materials = 3  # Number of materials
    recon_snr_db = 20  # Assumed SNR for the reconstruction
    verbose = 2  # Verbosity level

    # Display parameters
    display_wave_idx = 200  # Wavelength index of displayed image
    display_slice_idx = 8  # Slice index of displayed image
    display_vox_idx = [32, 32, 8]  # Voxel index [column, column, row] of displayed spectra
    vmax = 0.1  # Maximum voxel value for displayed image
    vmin = 0  # Minimum voxel value for displayed image
    y_lim_attenuation = (0.04, 0.09)  # (y_min, y_max) to set y-axis range for attenuation spectra
    y_lim_transmission = (0.91, 0.96)  # (y_min, y_max) to set y-axis range for transmission spectra

    # Fix seed for random number generation
    np.random.seed(129)

    # Load theoretical linear attenuation coefficients for Ni, Cu, and Al
    material_basis_path = './binaries/'
    filename = os.path.join(material_basis_path, 'material_basis.npy')
    material_basis = np.load(filename)

    # Generate simulated noisy hyperspectral projection data
    [hsnt_data, angles, _] = generate_hyper_data(material_basis,
                                                 num_angles=num_angles,
                                                 detector_rows=detector_rows,
                                                 detector_columns=detector_columns,
                                                 dosage_rate=dosage_rate,
                                                 material_density=material_density,
                                                 verbose=verbose)

    # MBIR model setup
    mj_model = mj.ParallelBeamModel((num_angles, detector_rows, detector_columns), angles)
    mj_model.set_params(snr_db=recon_snr_db, verbose=0)

    # Perform fast hyperspectral reconstruction (dehydrate + MBIR + rehydrate)
    hsnt_dehydrated = dehydrate(hsnt_data, dataset_type=dataset_type, num_materials=num_materials, verbose=verbose)

    [subspace_data, subspace_basis, dataset_type] = hsnt_dehydrated
    subspace_dimension = subspace_data.shape[-1]
    subspace_recons = np.zeros((detector_columns, detector_columns, detector_rows, subspace_dimension))
    for idx in range(subspace_dimension):
        print("Reconstructing data for subspace index: " + str(idx))
        subspace_recons[:, :, :, idx], _ = mj_model.recon(subspace_data[:, :, :, idx])
    hsnt_dehydrated_recons = [subspace_recons, subspace_basis, dataset_type]

    hsnt_recons = rehydrate(hsnt_dehydrated_recons)

    # Plot hyperspectral projections and spectra
    if verbose > 1:
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

        plot_spectra(spectra=[np.exp(-hsnt_recons[display_vox_idx[0], display_vox_idx[1], display_vox_idx[2], :])],
                     title='Single voxel spectra (transmission) for reconstructed data',
                     x_label='wavelength index',
                     y_label='transmission',
                     y_lim=y_lim_transmission)

    print('Total time elapsed: ', time.time() - start_time, ' seconds')

    plt.show()


if __name__ == "__main__":
    main()
