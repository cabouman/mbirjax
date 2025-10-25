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

    # Initialize model for reconstruction.
    tct_model = mj.TranslationModel(**translation_params)
    tct_model.set_params(**optional_params)
    tct_model.set_params(sharpness=0.0)
    recon_shape = tct_model.get_params('recon_shape')

    # Print model parameters and display translation array
    translation_vectors = translation_params['translation_vectors']
    tct_model.print_params()
    mj.display_translation_vectors(translation_vectors, recon_shape)

    # View sinogram
    mj.slice_viewer(sino, slice_axis=0, title='Original sinogram', slice_label='View')

    # Generate weights array - for an initial reconstruction, use weights = None, then modify if needed.
    weights = None

    # Perform MBIR reconstruction
    recon, recon_params = tct_model.recon(sino, init_recon=0, weights=weights, max_iterations=80)

    # Save reconstruction results
    output_path = './output/'  # path to store output recon
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, f'purdue_p_recon.h5')
    mj.export_recon_hdf5(output_path, recon, recon_dict=recon_params, top_margin=0, bottom_margin=0)

    sino_from_recon = tct_model.forward_project(recon)
    mj.slice_viewer(sino, sino_from_recon, title='Original sinogram and forward projected recon', slice_axis=0)

    # Display Results
    mj.slice_viewer(recon.transpose(0, 2, 1), title='MBIR reconstruction', slice_axis=0)



if __name__ == '__main__':
    main()
