# This script is for reconstructing parallel beam CT data from Zeiss scanner

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
    sharpness = 0.25
    view_scale = 0.8

    # Path to the dataset
    dataset_dir = '/depot/bouman/data/Zeiss/foam512R1N3000.txrm'

    # Output path
    output_path = './output/'  # path to store output recon images

    # Load the sinogram and metadata
    print("\n********** Load sinogram and metadata from the data **************")
    sinogram, cone_beam_params, optional_params, metadata = mjp.zeiss_cone_beam.compute_sino_and_params(dataset_dir)

    # Construct cone beam model
    print("\n********** Construct cone beam model **************")
    ct_model = mj.ConeBeamModel(**cone_beam_params)
    ct_model.set_params(**optional_params)
    ct_model.set_params(sharpness=sharpness)
    weights = mj.gen_weights(sinogram, weight_type='transmission_root')

    # Display the sinogram
    mj.slice_viewer(sinogram, slice_axis=1, title='Original sinogram')

    # Print out model parameters
    ct_model.print_params()

    # Perform FDK reconstruction
    print("\n********** Perform FDK reconstruction **************")
    direct_recon = ct_model.direct_recon(sinogram)

    # Perform MBIR reconstruction
    print("\n********** Perform MBIR reconstruction **************")
    mbir_recon, recon_dict = ct_model.recon(sinogram, weights=weights)

    # Save recon to hdf5
    print("\n*********** save mar and fdk recon in h5 format *************")
    os.makedirs(output_path, exist_ok=True)  # mkdir if directory does not exist
    fdk_path = os.path.join(output_path, f"cone_fdk_recon.h5")
    mj.export_recon_hdf5(fdk_path, direct_recon, recon_dict=None)
    mbir_path = os.path.join(output_path, f"cone_mbir_recon.h5")
    mj.export_recon_hdf5(mbir_path, mbir_recon, recon_dict=None, remove_flash=True)
    print("FDK recon saved to {}".format(os.path.abspath(fdk_path)))
    print("MBIR recon saved to {}".format(os.path.abspath(mbir_path)))

    # Determine viewing range
    vmax = view_scale * max(jnp.max(direct_recon), jnp.max(mbir_recon))
    vmin = 0

    # Display the results
    mj.slice_viewer(direct_recon, mbir_recon, vmin=vmin, vmax=vmax, slice_axis=2,
                    slice_label=['FDK', 'MBIR'],
                    title='Comparison between FDK and MBIR reconstructions')

if __name__ == '__main__':
    main()