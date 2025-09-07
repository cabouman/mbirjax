import warnings
import time
import numpy as np
import jax.numpy as jnp
import mbirjax as mj
import mbirjax.preprocess as mjp
import os


if __name__ == "__main__":
    print('This script demonstrates split-sino reconstruction.\n')

    # Set user determined parameters
    display_comparison = True       # Displays comparison of reconstructions, but slows script
    recon_slice_offset = 0.0        # Offset of reconstruction slice down (positive) or up (negative)
    downsample = 16                 # Downsample rows and columns of scan views for speed
    pixel_scale = 1.0               # Scaling factor for pixel pitch

    output_path = './output'
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        warnings.warn(
            f'Created output directory {output_path}. For faster I/O on clusters, consider symlinking to scratch, e.g.,\n'
            f'  ln -s /scratch/gautschi/<username>/results {output_path}')

    # path to store and extract the NSI data and metadata.
    download_dir = './demo_data/'

    # NSI file path
    dataset_url = 'https://www.datadepot.rcac.purdue.edu/bouman/data/demo_nsi_vert_metal_all_views.tgz'

    # Download and extract data. Then set path to NSI scan directory.
    dataset_dir = mj.download_and_extract(dataset_url, download_dir)

    # preprocessing parameters
    downsample_factor = [downsample, downsample]  # downsample factor of scan view images along detector rows and detector columns.
    subsample_view_factor = downsample  # view subsample factor.

    # recon parameters
    sharpness = 1.0
    snr_db = 30.0

    print("\n************** NSI dataset preprocessing **************")
    sino, cone_beam_params, optional_params = mjp.nsi.compute_sino_and_params(dataset_dir,
                                                                              downsample_factor=downsample_factor,
                                                                              subsample_view_factor=subsample_view_factor)

    print("\n***************** Set up MBIRJAX model ****************")
    # Construct cone beam object using NSI parameters
    ct_model = mj.ConeBeamModel(**cone_beam_params)

    # Set optional NSI geometry parameters
    ct_model.set_params(**optional_params)

    # Set user determined parameter values
    ct_model.set_params(sharpness=sharpness, snr_db=snr_db, verbose=1)
    ct_model.set_params(recon_slice_offset=recon_slice_offset)
    delta_voxel = ct_model.get_params('delta_voxel')
    ct_model.set_params(delta_voxel=delta_voxel * pixel_scale)

    # Print out model parameters
    ct_model.print_params()

    if display_comparison:
        print("\n***************** Compute standard recon ****************")
        recon_std, recon_dict_std = ct_model.recon(sino)  # weights can be passed as third arg if available

    print("\n***************** Compute split sino recon ****************")
    t0 = time.time()
    recon, recon_dict = ct_model.recon_split_sino(sino)  # weights can be passed as third arg if available
    t1 = time.time()

    if display_comparison:
        print(f"Stitched recon shape: {recon.shape}   (elapsed: {t1 - t0:.1f}s)")
        mj.slice_viewer(recon, recon_std, data_dicts=[recon_dict, recon_dict_std], slice_axis=1, title="Split Sino Recon (left) vs Standard Recon (right)")
        diff = recon - recon_std
        mj.slice_viewer(diff, slice_axis=1, title="Recon Difference")
    else:
        mj.slice_viewer(recon, data_dicts=recon_dict, slice_axis=1, title="Split Sino Recon")
