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
    tct_model.auto_set_recon_shape(sino.shape)
    recon_shape = tct_model.get_params('recon_shape')

    # Set parameters for recon
    tct_model.set_params(recon_shape=(20,)+recon_shape[1:])
    tct_model.set_params(delta_recon_row=10)
    tct_model.set_params(partition_sequence=50*[0,] + 100*[2,] + [3,])
    tct_model.set_params(qggmrf_nbr_wts=[0.01, 1, 1])

    # Print model parameters and display translation array
    translation_vectors = translation_params['translation_vectors']
    tct_model.print_params()
    delta_voxel, delta_recon_row = tct_model.get_params(['delta_voxel', 'delta_recon_row'])
    translation_vectors_display = translation_vectors.copy()
    translation_vectors_display[:, 0] /= delta_voxel
    translation_vectors_display[:, 2] /= delta_voxel
    translation_vectors_display[:, 1] /= delta_recon_row
    mj.display_translation_vectors(translation_vectors_display, recon_shape)

    # View sinogram
    mj.slice_viewer(sino, slice_axis=0, title='Original sinogram', slice_label='View')

    # Generate weights array - for an initial reconstruction, use weights = None, then modify if needed.
    weights = None

    # Perform MBIR reconstruction
    direct_recon, direct_dict = tct_model.recon(sino, weights=weights, max_iterations=0)
    mbir_recon, mbir_dict = tct_model.recon(sino, weights=weights, max_iterations=200)

    # Save reconstruction results
    output_path = './output/'  # path to store output recon
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, f'purdue_p_recon.h5')
    mj.export_recon_hdf5(output_path, mbir_recon, recon_dict=mbir_dict, top_margin=0, bottom_margin=0)

    sino_from_recon = tct_model.forward_project(mbir_recon)
    mj.slice_viewer(sino, sino_from_recon, title='Original sinogram and forward projected recon', slice_axis=0)

    # Display Results
    mj.slice_viewer(direct_recon.transpose(0, 2, 1), mbir_recon.transpose(0, 2, 1), data_dicts=[direct_dict, mbir_dict],
                    title='Direct recon (left) and MBIR recon (right)', slice_axis=0, vmin=-0.002, vmax=0.02)



if __name__ == '__main__':
    main()
