# This script is for reconstructing Translation CT BGA data from Zeiss scanner

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
    download_dir = "./purdue_BGA_data"
    dataset_url = "/depot/bouman/data/Translation/purdue_BGA_xrm.tgz"
    dataset_dir = mj.download_and_extract(dataset_url, download_dir)

    # Load and preprocess data (returns a ready-to-reconstruct model and a data-specific weight mask)
    sino, tct_model, weights = mjp.zeiss_tct.get_sino_and_model(dataset_dir)

    # tct_model.set_params(sharpness=2.0)
    recon_shape = tct_model.get_params('recon_shape')
    voxel_row_aspect = tct_model.get_params('voxel_row_aspect')
    delta_det_channel = tct_model.get_params('delta_det_channel')

    # Set parameters for recon
    tct_model.set_params(positivity_flag=True)
    tct_model.set_params(voxel_row_aspect=voxel_row_aspect / 2)
    tct_model.set_params(recon_shape=(recon_shape[0] * 2, 2900, 2500))
    tct_model.set_params(partition_sequence=[0, 2, 4, 6])
    tct_model.set_params(qggmrf_nbr_wts=[0.1, 1, 1])

    # Print model parameters and display translation array
    translation_vectors = np.array(tct_model.get_params('translation_vectors'))
    tct_model.print_params()
    delta_voxel, voxel_row_aspect, voxel_slice_aspect = tct_model.get_params(['delta_voxel', 'voxel_row_aspect', 'voxel_slice_aspect'])
    translation_vectors_display = translation_vectors.copy()
    translation_vectors_display[:, 0] /= delta_voxel
    translation_vectors_display[:, 2] /= (voxel_slice_aspect * delta_voxel)
    translation_vectors_display[:, 1] /= (voxel_row_aspect * delta_voxel)
    mj.display_translation_vectors(translation_vectors_display, recon_shape)

    # View sinogram
    mj.slice_viewer(sino, slice_axis=0, title='Original sinogram', slice_label='View')

    # Generate weights array - for an initial reconstruction, use weights = None, then modify if needed.
    weights = weights

    # Perform MBIR reconstruction
    direct_recon, direct_dict = tct_model.recon(sino, weights=weights, max_iterations=0)
    mbir_recon, mbir_dict = tct_model.recon(sino, weights=weights, max_iterations=200, stop_threshold_change_pct=0.2)

    # Save reconstruction results
    output_path = './output/'  # path to store output recon
    os.makedirs(output_path, exist_ok=True)
    fdk_path = os.path.join(output_path, f"TCT_BGA_fdk_recon.h5")
    mj.export_recon_hdf5(fdk_path, direct_recon, recon_dict=direct_dict, top_margin=0, bottom_margin=0)
    mbir_path = os.path.join(output_path, f"TCT_BGA_mbir_recon.h5")
    mj.export_recon_hdf5(mbir_path, mbir_recon, recon_dict=mbir_dict, top_margin=0, bottom_margin=0)

    # sino_from_recon = tct_model.forward_project(mbir_recon)
    # mj.slice_viewer(sino, sino_from_recon, title='Original sinogram and forward projected recon', slice_axis=0)

    # Display Results
    mj.slice_viewer(direct_recon.transpose(0, 2, 1), mbir_recon.transpose(0, 2, 1), data_dicts=[direct_dict, mbir_dict],
                    vmin = 0, vmax=0.5,
                    title='Direct recon (left) and MBIR recon (right)', slice_axis=0)



if __name__ == '__main__':
    main()