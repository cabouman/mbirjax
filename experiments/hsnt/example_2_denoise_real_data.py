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
plt.style.use('tableau-colorblind10')

import jax
import jax.numpy as jnp
import jax.lax as lax

from mbirjax.hsnt import import_hsnt_data_hdf5, export_hsnt_data_hdf5, optimize_newt, optimize_mu
from plot_utils import plot_images, plot_spectra


def main():
    # Choose dataset from '0.8C_Ni_cylinder', '1.6C_Ni_cylinder', '2.4C_Ni_cylinder', '4.8C_Ni_cylinder', '9.6C_Ni_cylinder'
    dataset_name = '0_8c_Ni_cylinder_dataset'
    input_path = './input_data/processed_data_0_8c_Ni_cylinder.h5'  # path to import input noisy data
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

    # Import real hyperspectral data
    hsnt_data, metadata = import_hsnt_data_hdf5(input_path, dataset_name)

    if verbose >= 1:
        print("Hyperspectral data shape: ", hsnt_data.shape)

    T = np.exp(-hsnt_data).reshape(-1, hsnt_data.shape[-1])

    # Convert to JAX array
    T_jax = jnp.asarray(T, dtype=jnp.float32)

    # Perform hyperspectral denoising
    start_time = time.time()
    W_newt, H_newt, i_newt = optimize_newt(
        T_jax, num_materials=num_materials, max_steps=1000, rel_tol=1e-8
    )
    print('Newton reconstruction completed in: ', time.time() - start_time, ' seconds after ', i_newt, ' iterations')
    start_time = time.time()
    W_mu, H_mu, i_mu = optimize_mu(
        T_jax, num_materials=num_materials, max_steps=1000, rel_tol=1e-8
    )
    print('Multiplicative reconstruction completed in: ', time.time() - start_time, ' seconds after ', i_mu, ' iterations')

    # Convert results to NumPy arrays and reshape to original data shape
    hsnt_denoised_newt = np.array(W_newt @ H_newt).reshape(hsnt_data.shape)
    hsnt_denoised_mu = np.array(W_mu @ H_mu).reshape(hsnt_data.shape)

    # Write out denoised/rehydrated data
    export_hsnt_data_hdf5(os.path.join(output_path, dataset_name+'_dataset_denoised_newt.h5'), hsnt_denoised_newt, metadata)
    export_hsnt_data_hdf5(os.path.join(output_path, dataset_name+'_dataset_denoised_mu.h5'), hsnt_denoised_mu, metadata)

    # Plot hyperspectral projections and spectra
    if verbose > 1:
        num_images=3
        images=[hsnt_data[0, :, :, display_wave_idx],
                hsnt_denoised_newt[0, :, :, display_wave_idx],
                hsnt_denoised_mu[0, :, :, display_wave_idx]]
        titles=['Original', 'Quasi-Newton', 'Mann-Multiplicative']
        filename="cylinder_nnal.png"
        plt.rcParams['figure.constrained_layout.use'] = True
        plt.rc('font', size=40)
        fig = plt.figure(figsize=(10, 10 * num_images), dpi=160 / num_images)
        fig.suptitle(f'Material Projections\nWavelength index: {display_wave_idx}')
        for idx in range(num_images):
            ax = fig.add_subplot(num_images, 1, idx + 1)
            ax.set_title(titles[idx])
            ax.imshow(images[idx], vmin=vmin, vmax=vmax, cmap='gray')
            if idx != num_images - 1:
                ax.set_xticklabels([])
        plt.savefig(filename, dpi=100)

        plt.figure(figsize=(30, 15))
        plt.plot(hsnt_data[0, display_pix_idx[0], display_pix_idx[1], :], label='Original')
        plt.plot(hsnt_denoised_newt[0, display_pix_idx[0], display_pix_idx[1], :], label='Quasi-Newton Denoised')
        plt.plot(hsnt_denoised_mu[0, display_pix_idx[0], display_pix_idx[1], :], label='Mann-Multiplicative Denoised')
        plt.title('Single pixel spectra')
        plt.xlabel('wavelength index')
        plt.ylabel('attenuation')
        plt.ylim(y_lim_attenuation)
        plt.legend()
        plt.savefig("cylinder_attenuation_nnal.png")

        plt.figure(figsize=(30, 15))
        plt.plot(np.exp(-hsnt_data[0, display_pix_idx[0], display_pix_idx[1], :]), label='Original')
        plt.plot(np.exp(-hsnt_denoised_newt[0, display_pix_idx[0], display_pix_idx[1], :]), label='Quasi-Newton Denoised')
        plt.plot(np.exp(-hsnt_denoised_mu[0, display_pix_idx[0], display_pix_idx[1], :]), label='Mann-Multiplicative Denoised')
        plt.title('Single pixel spectra (transmission)')
        plt.xlabel('wavelength index')
        plt.ylabel('transmission')
        plt.ylim(y_lim_transmission)
        plt.legend()
        plt.savefig("cylinder_transmission_nnal.png")

    plt.show()


if __name__ == "__main__":
    main()
