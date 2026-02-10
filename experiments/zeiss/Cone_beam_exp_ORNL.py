### This script is for reconstructing cone beam CT data from ORNL Zeiss scanner

import os
import sys
import numpy as np
import jax.numpy as jnp
import pprint
import mbirjax as mj
import mbirjax.preprocess as mjp

pp = pprint.PrettyPrinter(indent=4)

def main():
    # Recon parameters
    sharpness = 1.5
    snr_db = 35.0
    downsample_factor = 2
    subsample_view_factor = 2

    # Path to the dataset
    dataset_dir = '/depot/bouman/data/ORNL/versa/ParAM-Round-1_Z62.txrm'

    # Output path
    output_path = './output/'  # path to store output recon images

    # Load the sinogram and metadata
    print("\n********** Load sinogram and metadata from the data **************")
    sinogram, cone_beam_params, optional_params, metadata = mjp.zeiss_cb.compute_sino_and_params(dataset_dir, downsample_factor=(downsample_factor, downsample_factor),
                                                                                                 subsample_view_factor=subsample_view_factor,
                                                                                                 is_preprocessed=False)

    # Construct cone beam model
    print("\n********** Construct cone beam model **************")
    ct_model = mj.ConeBeamModel(**cone_beam_params)
    ct_model.set_params(**optional_params)

    # Rerun auto-parameter functions because we changed the assumed detector pitch
    ct_model.auto_set_recon_geometry(sinogram.shape) # Reset default recon shape

    # Sharpness and snr_db
    ct_model.set_params(sharpness=sharpness, snr_db=snr_db)

    # Display the sinogram
    mj.slice_viewer(sinogram, slice_axis=0, title='Original sinogram')

    # Print out model parameters
    ct_model.print_params()

    # Perform FDK reconstruction
    print("\n********** Perform FDK reconstruction **************")
    direct_recon = ct_model.direct_recon(sinogram)

    # Perform sinogram per-view alignment
    print("\n********** Perform sinogram alignment **************")
    sinogram = mjp.sino_view_alignment(ct_model, sinogram, direct_recon)

    # Weights
    weights = mj.gen_weights(sinogram, weight_type='transmission_root')

    # Perform FDK reconstruction
    print("\n********** Perform FDK reconstruction after alignment **************")
    direct_recon = ct_model.direct_recon(sinogram)

    # Perform MBIR reconstruction
    print("\n********** Perform MBIR reconstruction after alignment **************")
    mbir_recon, recon_dict = ct_model.recon(sinogram, weights=weights)

    # Save recon to hdf5
    print("\n*********** save mbir and fdk recon in h5 format *************")
    os.makedirs(output_path, exist_ok=True)  # mkdir if directory does not exist
    fdk_path = os.path.join(output_path, f"cone_fdk_recon.h5")
    mj.export_recon_hdf5(fdk_path, direct_recon, recon_dict=None)
    mbir_path = os.path.join(output_path, f"cone_mbir_recon.h5")
    mj.export_recon_hdf5(mbir_path, mbir_recon, recon_dict=None, remove_flash=True)
    print("FDK recon saved to {}".format(os.path.abspath(fdk_path)))
    print("MBIR recon saved to {}".format(os.path.abspath(mbir_path)))

    # Display the results
    mj.slice_viewer(direct_recon, mbir_recon, slice_axis=2,
                    slice_label=['FDK', 'MBIR'],
                    title='Comparison between FDK and MBIR reconstructions')


if __name__ == '__main__':
    main()