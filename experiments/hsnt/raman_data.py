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

    # Choose dataset
    # 'Nilotinib-P, 50X, 785, 1200_01.csv', 'Nilotinib-P, 50X, 785, 1200_01(1).csv'
    # 'Nilotinib-AS, 50X, 785, 1200_01.csv', 'Nilotinib-AS, 50X, 785, 1200_01(1).csv'
    dataset_name = 'Nilotinib-P, 50X, 785, 1200_01.csv'  # 'Nilotinib-AS, 50X, 785, 1200_01.csv'
    input_path = './raman_data/'  # path to import input noisy data
    output_path = './raman_data/'  # path to export output denoised data
    os.makedirs(output_path, exist_ok=True)  # Make output directory if it does not exist

    # Load the data and do PCA
    calibration_values, data_cube = read_measurement_csv(input_path, dataset_name)
    hsnt_data = data_cube / calibration_values[None, None, :]
    # mj.slice_viewer(hsnt_data)

    # Start HSNT
    # Define the components to use and determine the range
    ind_sets = [[0], [1, 2], [3, 4, 5]]
    unique_inds = np.unique(np.concatenate(ind_sets))
    subspace_dimension = np.amax(unique_inds) + 1  # Subspace dimension
    loss_type = 'frobenius' # 'kullback-leibler'  # 'frobenius'

    verbose = 2  # Verbosity level

    # Fix seed for random number generation
    np.random.seed(129)

    # Import real hyperspectral data
    dataset_type = 'transmission'

    if verbose >= 1:
        print("Hyperspectral data shape: ", hsnt_data.shape)
        print("Running hyperspectral dehydrate followed by rehydrate (i.e., denoising)")

    hsnt_dehydrated = dehydrate(hsnt_data, beta_loss=loss_type, dataset_type=dataset_type,
                                subspace_dimension=subspace_dimension, verbose=verbose)

    # Set up for plotting
    nrows = len(ind_sets)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 5*nrows))
    fig.suptitle(f"NNMF: ({dataset_name})\n")

    hsnt_spectra = hsnt_dehydrated[1]
    hsnt_image_per_component = hsnt_dehydrated[0]
    cum_inds = []
    for j, inds in enumerate(ind_sets):
        cum_inds = cum_inds + inds
        hsnt_denoised = hsnt_image_per_component[:, :, cum_inds] @ hsnt_spectra[cum_inds]
        nrmse = np.linalg.norm(hsnt_denoised - hsnt_data)**2 / np.linalg.norm(hsnt_data)**2
        print('NRMSE for NNMF for inds = {}: {:.4g}%'.format(cum_inds, 100 * nrmse))

        # Fill in an extra channel if needed
        image = hsnt_image_per_component[:, :, inds]
        if image.shape[2] == 2:
            image = np.stack([image[:, :, 0], image[:, :, 1], 0*(image[:, :, 0] + image[:, :, 1])/2], axis=2)
        # Normalize by 2 and 98 percentile
        image2 = np.percentile(image, 2)
        image98 = np.percentile(image, 98)
        image = (image - image2) / (image98 - image2)
        image = np.clip(image, 0, 1)

        if len(ind_sets) > 1:
            ax_img, ax_plot = axes[j]
        else:
            ax_img, ax_plot = axes
        # Left: image
        ax_img.imshow(image)
        ax_img.set_title(
            f"Image using component(s) {inds} only"
        )
        ax_img.axis("off")

        # Right: NNMF components
        for k in inds:
            ax_plot.plot(hsnt_spectra[k])

        ax_plot.set_title(f"NNMF components {inds}")
        ax_plot.set_xlabel("Index")
        ax_plot.set_ylabel("Amplitude")

        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
