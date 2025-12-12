"""
Hyperspectral Dehydration & Rehydration
---------------------------------------

This script demonstrates the use of dehydration and rehydration for hyperspectral data denoising.
Multiple real hyperspectral neutron datasets are available for the purpose.
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple
import csv

import os
import numpy as np
import time
import matplotlib.pyplot as plt

from mbirjax.hsnt import hyper_denoise, dehydrate, rehydrate, import_hsnt_data_hdf5, export_hsnt_data_hdf5
from plot_utils import plot_images, plot_spectra
import mbirjax as mj


def read_measurement_csv(
    data_dir: str | Path,
    filename: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read measurement data from a CSV file and return (calibration_values, data_cube).

    Expected CSV layout:
      - Column 0: first coordinate  (odd integers in {-25,-23,...,-1,1,...,25})
      - Column 1: second coordinate (same)
      - Columns 2..: measurement values.
      - Row 0 has NO coordinates; columns 2.. are calibration values.

    Returns
    -------
    calibration_values : np.ndarray of shape (N,)
    data_cube : np.ndarray of shape (26, 26, N)
        data_cube[i, j, :] contains the measurement vector for coordinate
        (coord0, coord1) mapped via (coord + 25) // 2.
    """

    path = Path(data_dir) / filename

    # --- Read CSV into a nested list of strings ---
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            # skip entirely empty rows
            if not any(field.strip() for field in row):
                continue
            rows.append(row)

    if len(rows) == 0:
        raise ValueError("CSV file contains no data.")

    # --- Convert the first row to calibration values ---
    # Columns 2..end
    try:
        calibration_values = np.array(
            [float(x) for x in rows[0][2:]],
            dtype=float
        )
    except ValueError:
        raise ValueError("Calibration row contains non-numeric values.")

    num_measurements = calibration_values.shape[0]

    # --- Process data rows ---
    coord_list_0 = []
    coord_list_1 = []
    meas_list = []

    for row in rows[1:]:
        if len(row) < 2:
            continue  # skip malformed rows

        try:
            c0 = int(row[0])
            c1 = int(row[1])
        except ValueError:
            # skip rows that don't contain coordinates
            continue

        # measurement values
        vals = np.array([float(x) for x in row[2:]], dtype=float)
        if vals.size != num_measurements:
            raise ValueError(
                f"Row has {vals.size} measurement columns, but expected {num_measurements}."
            )

        coord_list_0.append(c0)
        coord_list_1.append(c1)
        meas_list.append(vals)

    coord0 = np.array(coord_list_0, dtype=int)
    coord1 = np.array(coord_list_1, dtype=int)
    measurements = np.vstack(meas_list) if meas_list else np.zeros((0, num_measurements))

    # --- Map odd coordinates in [-25,25]\{0} to indices 0..25 ---
    def coord_to_index(c: np.ndarray) -> np.ndarray:
        if np.any(c == 0):
            raise ValueError("Coordinate 0 encountered; expected odd integers ±1..±25.")
        if np.any((c < -25) | (c > 25) | (c % 2 == 0)):
            raise ValueError("Coordinates must be odd integers in [-25,-1] ∪ [1,25].")
        return (c + 25) // 2

    idx0 = coord_to_index(coord0)
    idx1 = coord_to_index(coord1)

    # --- Fill 3D array ---
    data_cube = np.full((max(idx0)+1, max(idx1)+1, num_measurements), np.nan, dtype=float)

    for i, j, vec in zip(idx0, idx1, measurements):
        data_cube[i, j, :] = vec

    return calibration_values, data_cube

def main():
    start_time = time.time()

    # Choose dataset
    # 'Nilotinib-P, 50X, 785, 1200_01.csv', 'Nilotinib-P, 50X, 785, 1200_01(1).csv'
    # 'Nilotinib-AS, 50X, 785, 1200_01.csv', 'Nilotinib-AS, 50X, 785, 1200_01(1).csv'
    dataset_name = 'Nilotinib-AS, 50X, 785, 1200_01.csv'  # 'Nilotinib-AS, 50X, 785, 1200_01.csv'
    input_path = './raman_data/'  # path to import input noisy data
    output_path = './raman_data/'  # path to export output denoised data
    os.makedirs(output_path, exist_ok=True)  # Make output directory if it does not exist

    calibration_values, data_cube = read_measurement_csv(input_path, dataset_name)
    hsnt_data = data_cube / calibration_values[None, None, :]
    # mj.slice_viewer(hsnt_data)
    num_bins = hsnt_data.shape[2]
    hsnt_flat = hsnt_data.reshape((-1, num_bins))
    hsnt_mean = np.mean(hsnt_flat, axis=0)
    hsnt_centered = hsnt_flat - hsnt_mean
    u, s, vh = np.linalg.svd(hsnt_centered, full_matrices=False)

    for inds in [[0, 1, 2], [3, 4, 5]]:
        rgb = s[inds, None] * vh[inds] @ hsnt_centered.T
        rgb_flat = rgb.T
        rgb = rgb.T.reshape(hsnt_data.shape[:2] + (3,))
        rgb2 = np.percentile(rgb, 2)
        rgb98 = np.percentile(rgb, 98)
        rgb = (rgb - rgb2) / (rgb98 - rgb2)
        plt.figure()
        plt.imshow(rgb)
        plt.title(f"PCA: ({dataset_name})")
        plt.figure()
        for k in inds:
            plt.plot(vh[k] * (-1)**(k+1))
        plt.title('PCA components {}'.format(inds))
    plt.show()

    # Denoiser parameters
    subspace_dimension = 3  # Subspace dimension
    loss_type = 'frobenius' # 'kullback-leibler'  # 'frobenius'

    verbose = 2  # Verbosity level
    test_denoise = False  # If True, test hyper_denoise; if False, test dehydrate + rehydrate sequence

    # Display parameters
    display_wave_idx = 500  # Wavelength index of displayed images
    display_pix_idx = [20, 15]  # Pixel index [row, column] of displayed spectra
    vmax = 2  # Maximum pixel value for displayed images
    vmin = 0  # Minimum pixel value for displayed images
    y_lim_attenuation = (0, 3)  # (y_min, y_max) to set y-axis range for attenuation spectra
    y_lim_transmission = (0, 0.75)  # (y_min, y_max) to set y-axis range for transmission spectra

    # Fix seed for random number generation
    np.random.seed(129)

    # Import real hyperspectral data
    dataset_type = 'attenuation'
    metadata = {'dataset_name': dataset_name, 'dataset_type': dataset_type}
    wavelengths = None

    if verbose >= 1:
        print("Hyperspectral data shape: ", hsnt_data.shape)

    if test_denoise:
        if verbose >= 1:
            print("Running hyperspectral denoising (i.e., dehydrate + rehydrate)")
        hsnt_denoised = hyper_denoise(hsnt_data, beta_loss=loss_type, dataset_type=dataset_type, subspace_dimension=subspace_dimension, verbose=verbose)
    else:
        if verbose >= 1:
            print("Running hyperspectral dehydrate followed by rehydrate (i.e., denoising)")
        hsnt_dehydrated = dehydrate(hsnt_data, beta_loss=loss_type, dataset_type=dataset_type, subspace_dimension=subspace_dimension, verbose=verbose)
        hsnt_denoised = rehydrate(hsnt_dehydrated)

        plt.figure()
        plt.imshow(hsnt_dehydrated[0])
        plt.title(f"DeHy: ({dataset_name})")
        plt.figure()
        plt.plot(hsnt_dehydrated[1].T)
        # Write out dehydrated data
        filename_dehydrated = os.path.join(output_path, dataset_name+'_dataset_dehydrated.h5')
        export_hsnt_data_hdf5(filename_dehydrated, hsnt_dehydrated, metadata)


    mj.slice_viewer(hsnt_data, np.abs(hsnt_denoised - hsnt_data), vmin=0, vmax=0.5)

    # Write out denoised/rehydrated data
    filename_denoised = os.path.join(output_path, dataset_name+'_dataset_denoised.h5')
    export_hsnt_data_hdf5(filename_denoised, hsnt_denoised, metadata)

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
