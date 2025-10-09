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
    sharpness = 1.0

    # Path to the dataset
    dataset_dir = '/depot/bouman/data/Zeiss/foam512R1N3000.txrm'

    # Output path
    output_path = './output/'  # path to store output recon images

    # Load the sinogram and metadata
    print("\n********** Load sinogram and metadata from the data **************")
    sinogram, metadata = mjp.zeiss.read_txrm(dataset_dir)

    # Load geometry parameters from the metadata
    angles = metadata["thetas"]  # in radians
    det_pixel_pitch = metadata["det_pixel_pitch"]  # in um
    source_iso_dist = metadata['source_iso_dist'][0] # in mm
    iso_detector_dist = metadata['iso_det_dist'][0] # in mm
    source_iso_dist = np.abs(source_iso_dist)  # in mm
    source_detector_dist = np.abs(source_iso_dist) + np.abs(iso_detector_dist)  # in mm

    # Convert geometry parameters from zeiss into mbirjax format
    # It seems that Zeiss detector pixel has equal width and height
    source_detector_dist *= 1000  # mm to um
    source_iso_dist *= 1000  # mm to um
    delta_det_channel = det_pixel_pitch
    delta_det_row = det_pixel_pitch
    delta_voxel = delta_det_channel * (source_iso_dist/source_detector_dist)

    # Construct cone beam model
    print("\n********** Construct cone beam model **************")
    ct_model = mj.ConeBeamModel(sinogram_shape=sinogram.shape, angles=angles, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)
    ct_model.set_params(delta_det_channel=delta_det_channel, delta_det_row=delta_det_row, delta_voxel=delta_voxel, sharpness=sharpness, verbose=1)
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

    # Display the results
    mj.slice_viewer(direct_recon, mbir_recon, vmin=0, slice_axis=2,
                    slice_label=['FDK', 'MBIR'],
                    title='Comparison between FDK and MBIR reconstructions')


if __name__ == '__main__':
    main()