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
        row = rows[0][2:]
        for i, x in enumerate(row):
            if x.find(",") != -1:
                x = x.replace(',', '.')
                row[i] = float(x)
            elif float(x) < 10000:
                row[i] = float(x)
            else:
                row[i] = float(x) / 1000
        raman_shifts = np.array(row, dtype=float)
    except ValueError:
        raise ValueError("Calibration row contains non-numeric values.")

    num_measurements = raman_shifts.shape[0]

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
        row = row[2:]
        for i, x in enumerate(row):
            if x.find(",") != -1:
                x = x.replace(',', '.')
                row[i] = float(x)
            elif float(x) < 10000:
                row[i] = float(x)
            else:
                row[i] = float(x) / 1000.0
        vals = np.array(row, dtype=float)

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

    # --- Map integer coordinates to indices 0...
    def coord_to_index(c: np.ndarray) -> np.ndarray:
        min_index = np.min(c)
        unique_inds = np.unique(c)
        step = unique_inds[1] - unique_inds[0]
        return (c - min_index) // step

    idx0 = coord_to_index(coord0)
    idx1 = coord_to_index(coord1)

    # --- Fill 3D array ---
    data_cube = np.full((max(idx0)+1, max(idx1)+1, num_measurements), np.nan, dtype=float)

    for i, j, vec in zip(idx0, idx1, measurements):
        data_cube[i, j, :] = vec

    return raman_shifts, data_cube


def run_pca(data_cube, dataset_name, ind_sets, raman_shifts):
    num_bins = data_cube.shape[2]
    hsnt_flat = data_cube.reshape((-1, num_bins))
    hsnt_mean = np.mean(hsnt_flat, axis=0)
    hsnt_centered = hsnt_flat - hsnt_mean
    u, s, vh = np.linalg.svd(hsnt_centered, full_matrices=False)
    # Compute symmetric y-limits across selected components
    ymax = 0.0    # Define the components to use and determine the range
    unique_inds = np.unique(np.concatenate(ind_sets))
    for k in unique_inds:
        comp = vh[k] * (-1) ** (k + 1)
        ymax = max(ymax, np.max(np.abs(comp)))

    # Add a small margin so curves don't touch the border
    ymax *= 1.05

    # Show PCA components and images obtained using the specified subset of components
    plt.figure()
    plt.plot(raman_shifts, hsnt_mean)
    plt.title('Mean spectrum')

    # Set up for plotting
    nrows = len(ind_sets)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 5*nrows))
    fig.suptitle(f"PCA: ({dataset_name})\n")

    for j, inds in enumerate(ind_sets):
        recon_svd = u[:, inds] @ (s[inds, None] * vh[inds])
        nrmse = np.linalg.norm(recon_svd - hsnt_centered) / np.linalg.norm(hsnt_centered)
        pct_explained = 1 - nrmse**2
        recon_svd += hsnt_mean
        recon_svd = recon_svd.reshape(data_cube.shape)
        print('Pct variance for PCA recon with inds = {}: {}'.format(inds, pct_explained))

        # Create a false-color image using the specified components of the PCA as the RGB channels
        image = s[inds, None] * vh[inds] @ hsnt_centered.T
        image_flat = image.T
        image = image_flat.reshape(data_cube.shape[:2] + (image_flat.shape[1],))
        # Fill in an extra channel if needed
        if image.shape[2] == 2:
            image = np.stack([image[:, :, 0], image[:, :, 1], 0*(image[:, :, 0] + image[:, :, 1])/2], axis=2)
        # Normalize by 2 and 98 percentile
        image2 = np.percentile(image, 2)
        image98 = np.percentile(image, 98)
        image = (image - image2) / (image98 - image2)
        image = np.clip(image, 0, 1)

        # Plot the false color image next to the plot of the PCA components
        if len(ind_sets) > 1:
            ax_img, ax_plot = axes[j]
        else:
            ax_img, ax_plot = axes

        # Left: image
        ax_img.imshow(image)
        ax_img.set_title(
            f"Component(s) {inds} explain {100 * pct_explained:.4g}% of variance"
        )
        ax_img.axis("off")

        # Right: PCA components
        for k in inds:
            ax_plot.plot(raman_shifts, vh[k] * (-1) ** (k + 1))

        ax_plot.set_ylim(-ymax, ymax)
        ax_plot.set_title(f"PCA components {inds}")
        ax_plot.set_xlabel("Raman shift")
        ax_plot.set_ylabel("Amplitude")

        plt.tight_layout()


def main():

    # Choose dataset
    # 'Nilotinib-P, 50X, 785, 1200_01.csv', 'Nilotinib-P, 50X, 785, 1200_01(1).csv'
    # 'Nilotinib-AS, 50X, 785, 1200_01.csv', 'Nilotinib-AS, 50X, 785, 1200_01(1).csv'
    dataset_name = 'Nilotinib-AS, 20X, 785, 1200_01.csv'  # 'Nilotinib-AS, 50X, 785, 1200_01.csv'
    input_path = './raman_data/'  # path to import input noisy data
    output_path = './raman_data/'  # path to export output denoised data
    os.makedirs(output_path, exist_ok=True)  # Make output directory if it does not exist

    ind_sets = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    # Load the data and do PCA
    raman_shifts, data_cube = read_measurement_csv(input_path, dataset_name)
    # data_cube = data_cube[::2, ::2]
    # mj.slice_viewer(data_cube / data_cube[:, :, 0:1], cmap='hsv')

    run_pca(data_cube, dataset_name, ind_sets, raman_shifts)

    # Start HSNT
    # Define the components to use and determine the range
    unique_inds = np.unique(np.concatenate(ind_sets))
    subspace_dimension = np.amax(unique_inds) + 1  # Subspace dimension
    loss_type = 'frobenius' # 'kullback-leibler'  # 'frobenius'

    verbose = 2  # Verbosity level

    # Fix seed for random number generation
    np.random.seed(129)

    # Import real hyperspectral data
    dataset_type = 'attenuation'

    if verbose >= 1:
        print("Hyperspectral data shape: ", data_cube.shape)
        print("Running hyperspectral dehydrate followed by rehydrate (i.e., denoising)")

    hsnt_dehydrated = dehydrate(data_cube, beta_loss=loss_type, dataset_type=dataset_type,
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
        nrmse = np.linalg.norm(hsnt_denoised - data_cube)**2 / np.linalg.norm(data_cube)**2
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
            ax_plot.plot(raman_shifts, hsnt_spectra[k])

        ax_plot.set_title(f"NNMF components {inds}")
        ax_plot.set_ylim([hsnt_spectra.min(), hsnt_spectra.max()])
        ax_plot.set_xlabel("Raman shift")
        ax_plot.set_ylabel("Amplitude")

        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
