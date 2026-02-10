### This script is for testing image alignment method on real sinogram

import os
import sys
import numpy as np
import jax.numpy as jnp
import time
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
    sinogram, cone_beam_params, optional_params, metadata = mjp.zeiss_cb.compute_sino_and_params(dataset_dir,downsample_factor=(downsample_factor, downsample_factor),
                                                                                                 subsample_view_factor=subsample_view_factor,
                                                                                                 is_preprocessed=False)

    # Construct cone beam model
    print("\n********** Construct cone beam model **************")
    ct_model = mj.ConeBeamModel(**cone_beam_params)
    ct_model.set_params(**optional_params)

    # Sharpness and weights
    ct_model.set_params(sharpness=sharpness, snr_db=snr_db)
    weights = mj.gen_weights(sinogram, weight_type='transmission_root')

    # Display the sinogram
    mj.slice_viewer(sinogram, slice_axis=0, title='Original sinogram')

    # Print out model parameters
    ct_model.print_params()

    # Perform FDK reconstruction
    print("\n********** Perform FDK reconstruction **************")
    direct_recon = ct_model.direct_recon(sinogram)

    # Perform sinogram per-view alignment
    print("\n********** Perform sinogram alignment **************")
    t0 = time.time()
    aligned_sino = mjp.sino_view_alignment(ct_model, sinogram, direct_recon)
    t1 = time.time()
    print(f"Elapsed time for sinogram alignment: {t1 - t0:.3f}s")

    # Visualize the direct_recon before and after sinogram alignment
    direct_recon_aligned = ct_model.direct_recon(aligned_sino)
    mj.slice_viewer(direct_recon, direct_recon_aligned, slice_axis=2,
                    slice_label=['FDK before alignment', 'FDK after alignment'],
                    title='Comparison of FDK before and after alignment')

    # Visualize the MBIR recon before and after sinogram alignment
    print("\n********** Perform MBIR reconstruction before and after alignment **************")
    mbir_recon, recon_dict = ct_model.recon(sinogram, weights=weights)

    aligned_sino_weights = mj.gen_weights(aligned_sino, weight_type='transmission_root')
    mbir_recon_aligned, recon_dict = ct_model.recon(aligned_sino, weights=aligned_sino_weights)

    mj.slice_viewer(mbir_recon, mbir_recon_aligned, slice_axis=2,
                    slice_label=['MBIR before alignment', 'MBIR after alignment'],
                    title='Comparison of MBIR before and after alignment')

if __name__ == '__main__':
    main()