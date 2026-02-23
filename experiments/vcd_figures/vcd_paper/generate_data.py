import numpy as np
import os
import time
import pprint
import jax
import jax.numpy as jnp
import jax.lax as lax


import mbirjax as mj
import mbirjax.preprocess as mjp
import pprint

pp = pprint.PrettyPrinter(indent=4)

if __name__ == "__main__":
    print('This script is demonstrates the preprocessing and reconstruction of NSI an dataset\
    \n\t using both FDK and MBIR reconstruction.\n')

    # recon parameters
    sharpness = 1.5
    snr_db = 30.0
    downsample = 2
    verbose = 2              # Print, but do not display plots
    crop_det_rows = 765

    # User defined paths
    output_path = './data/'   # path to store output recon images
    os.makedirs(output_path, exist_ok=True)  # mkdir if directory does not exist
    download_dir = './demo_data/'            # Directory to store downloaded data

    dataset_url = 'https://www.datadepot.rcac.purdue.edu/bouman/elec_board/Vertical_Elec_Board_1200.tgz'
    print(f"Selected dataset URL: {dataset_url}")

    # Download and extract data. Then set path to NSI scan directory.
    dataset_dir = mj.download_and_extract(dataset_url, download_dir)

    # preprocessing parameters
    downsample_factor = [downsample, downsample]  # downsample factor of scan view images along detector rows and detector columns.
    subsample_view_factor = downsample  # view subsample factor.

    print("\n************** NSI dataset preprocessing **************")
    sino, cone_beam_params, optional_params = \
        mjp.nsi.compute_sino_and_params(dataset_dir, crop_pixels_top=crop_det_rows, crop_pixels_bottom=crop_det_rows,
                                                       downsample_factor=downsample_factor,
                                                       subsample_view_factor=subsample_view_factor)

    print("\n************** Set up MBIRJAX model **************")
    # Construct cone beam object using NSI parameters
    ct_model = mj.ConeBeamModel(**cone_beam_params)

    # Set optional NSI geometry parameters
    ct_model.set_params(**optional_params)

    # Set user determined parameter values
    ct_model.set_params(sharpness=sharpness, snr_db=snr_db, verbose=verbose)

    # Print out model parameters
    ct_model.print_params()

    print("\n************** Calculate sinogram weights **************")
    weights = mj.gen_weights(sino, weight_type='transmission_root')

    print("\n************** Perform FDK reconstruction **************")
    # ##########################
    # Perform FDK reconstruction
    fdk_recon = ct_model.direct_recon(sino)
    #mj.slice_viewer(fdk_recon)

    print("\n************** Perform MBIR reconstruction **************")
    # #### Perform MBIR reconstruction
    time0 = time.time()
    mbir_recon, mbir_recon_dict = ct_model.recon(sino, weights=weights, max_iterations=100)
    mbir_recon.block_until_ready()
    elapsed = time.time() - time0
    print('Elapsed time for recon is {:.3f} seconds'.format(elapsed))

    # #### Print out parameters used in recon
    pprint.pprint(mbir_recon_dict['recon_params'])

    # #### Save MBIR reconstruction to HDF5 file output
    mj.export_recon_hdf5(os.path.join(output_path, "recon.h5"), mbir_recon, recon_dict=mbir_recon_dict, remove_flash=True,
                         top_margin=0, bottom_margin=0)

    if verbose >1:
        # Display FDK versus MBIR
        vmin = 0
        vmax = downsample_factor[0] * 0.025
        mj.slice_viewer(fdk_recon, mbir_recon, data_dicts=[None, mbir_recon_dict], vmin=0, vmax=vmax, slice_label= ["FDK Recon", "MBIR Recon"], title='Axial Slice')
