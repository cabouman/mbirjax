# This script is for reconstructing Translation CT data from Zeiss scanner

import mbirjax as mj
import mbirjax.preprocess as mjp
import numpy as np
import os
import pprint
import sys
import subprocess
import importlib.util

def main():
    # Download and extract data
    download_dir = "./purdue_p_data"
    dataset_url = "/depot/bouman/data/Translation/purdue_p_xrm.tgz"
    dataset_dir = mj.download_and_extract(dataset_url, download_dir)

    # Load and preprocess data
    sino, translation_params, optional_params = mjp.zeiss_tct.compute_sino_and_params(dataset_dir, crop_pixels_bottom=53)

    # Estimate and subtract the per-view background
    # background = [np.mean(sino[j][sino[j]<0.04]) for j in range(sino.shape[0])]
    # sino = sino - np.array(background)[:, None, None]

    # Initialize model for reconstruction.
    tct_model = mj.TranslationModel(**translation_params)
    tct_model.set_params(**optional_params)
    tct_model.set_params(sharpness=2.0)
    # tct_model.auto_set_recon_geometry(sino.shape)
    recon_shape = tct_model.get_params('recon_shape')
    delta_det_channel = tct_model.get_params('delta_det_channel')

    # Set parameters for recon
    tct_model.set_params(positivity_flag=True)
    tct_model.set_params(recon_shape=(40,)+recon_shape[1:])
    #tct_model.set_params(delta_recon_row=10 * delta_det_channel)
    tct_model.set_params(partition_sequence=5*[0,] + 100*[1, 3,])
    tct_model.set_params(qggmrf_nbr_wts=[0.1, 1, 1])

    # Print model parameters and display translation array
    translation_vectors = translation_params['translation_vectors']
    tct_model.print_params()
    delta_voxel, delta_recon_row = tct_model.get_params(['delta_voxel', 'delta_recon_row'])
    translation_vectors_display = translation_vectors.copy()
    translation_vectors_display[:, 0] /= delta_voxel
    translation_vectors_display[:, 2] /= delta_voxel
    translation_vectors_display[:, 1] /= delta_recon_row
    # mj.display_translation_vectors(translation_vectors_display, recon_shape)

    # Compute sino indicator
    sino_indicator = tct_model._get_sino_indicator(sino)

    # View sinogram and sino indicator
    mj.slice_viewer(sino, sino_indicator, slice_axis=0, vmin=0, vmax=0.3,
                    title='Original sinogram (left), Sino indicator (right)', slice_label='View')


if __name__ == '__main__':
    main()
