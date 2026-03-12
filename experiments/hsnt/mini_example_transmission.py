"""
Hyperspectral Dehydration & Rehydration
---------------------------------------

This mini script tests the performance of the algorithm for low quality transmission data.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from mbirjax.hsnt import hyper_denoise


def main():
    start_time = time.time()

    # Parameters
    num_materials = 1
    max_iter=300

    # Load noisy transmission data and corresponding wavelength values
    noisy_data = np.load("./input_data/test_transmission_data_0.8C.npy")
    wavelengths = np.load("./input_data/test_wavelengths_0.8C.npy")

    # Denoise using wrong (attenuation) and right (transmission) mode
    denoised_data_a = hyper_denoise(noisy_data, dataset_type='attenuation', num_materials=num_materials, max_iter=max_iter)
    denoised_data_b = hyper_denoise(noisy_data, dataset_type='transmission', num_materials=num_materials, max_iter=max_iter)


    # Plot attenuation mode results
    plt.figure()
    plt.plot(wavelengths, noisy_data[200,200], '.', color='gray', markersize = 1, label = 'raw pixel')
    plt.plot(wavelengths, denoised_data_a[200,200], '.', color='navy', markersize = 1, label = 'denoised pixel')
    plt.plot(wavelengths, np.mean(noisy_data[160:350,160:350], axis=(0,1)), '.', color='red', markersize = 1, 
             label = 'raw avg over pixels')
    plt.plot(wavelengths, np.mean(denoised_data_a[160:350,160:350], axis=(0,1)), '.', color='green', markersize = 1, 
             label = 'denoised avg over pixels')
    plt.ylim(-0.5,1.05)
    plt.grid(linestyle='--')
    plt.legend(markerscale=10, loc='best', ncol=2)
    plt.title(f'Pixel Spectra at (200,200) using "attenuation"', fontweight='bold', fontsize=12, y=1.02)

    # Plot transmission mode results
    plt.figure()
    plt.plot(wavelengths, noisy_data[200,200], '.', color='gray', markersize = 1, label = 'raw pixel')
    plt.plot(wavelengths, denoised_data_b[200,200], '.', color='navy', markersize = 1, label = 'denoised pixel')
    plt.plot(wavelengths, np.mean(noisy_data[160:350,160:350], axis=(0,1)), '.', color='red', markersize = 1, 
             label = 'raw avg over pixels')
    plt.plot(wavelengths, np.mean(denoised_data_b[160:350,160:350], axis=(0,1)), '.', color='green', markersize = 1, 
             label = 'denoised avg over pixels')
    plt.ylim(-0.5,1.05)
    plt.grid(linestyle='--')
    plt.legend(markerscale=10, loc='best', ncol=2)
    plt.title(f'Pixel Spectra at (200,200) using "transmission"', fontweight='bold', fontsize=12, y=1.02)

    print('Total time elapsed: ', time.time() - start_time, ' seconds')

    plt.show()


if __name__ == "__main__":
    main()
