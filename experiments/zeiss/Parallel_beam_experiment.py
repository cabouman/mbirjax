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
    num_slices = 10

    # Path to the dataset
    dataset_dir = '/depot/bouman/data/Zeiss/foam512R1N3000.txrm'

    # Output path
    output_path = './output/'   # path to store output recon images

    # Load the sinogram and metadata
    print("\n********** Load sinogram and metadata from the data **************")
    sinogram, metadata = mjp.zeiss.read_txrm(dataset_dir)

    # Load reconstruction parameters from the metadata
    angles = metadata["thetas"]
    # Get pixel size in units of cm
    pixel_size = metadata["det_pixel_pitch"] / 10000

    # Select out a typically small number of slices from the important region of the sample.
    num_views, num_det_rows, num_det_channels = sinogram.shape
    num_slices = np.minimum(num_det_rows, num_slices)
    crop_pixels_top = (num_det_rows - num_slices) // 2
    crop_pixels_bottom = (num_det_rows - num_slices) // 2

    # Crop out desired rows from each view of the sinogram
    print("\n********** Crop out desired rows from each view **************")
    Nr_lo = crop_pixels_top
    Nr_hi = num_det_rows - crop_pixels_bottom

    sinogram = sinogram[:, Nr_lo:Nr_hi, :]

    # Construct parallel beam model
    print("\n********** Construct parallel beam model **************")
    ct_model = mj.ParallelBeamModel(sinogram_shape=sinogram.shape, angles=angles)
    ct_model.set_params(sharpness=sharpness, verbose=1)
    weights = mj.gen_weights(sinogram, weight_type='transmission_root')

    # Display the sinogram
    mj.slice_viewer(sinogram, slice_axis=1, title='Original sinogram')

    # Print out model parameters
    ct_model.print_params()

    # Perform FBP reconstruction
    print("\n********** Perform MBIR reconstruction **************")
    direct_recon = ct_model.direct_recon(sinogram)
    direct_recon /= pixel_size  # convert to units of 1/cm

    # Perform MBIR reconstruction
    print("\n********** Perform MBIR reconstruction **************")
    mbir_recon, recon_dict = ct_model.recon(sinogram, weights=weights)
    mbir_recon /= pixel_size  # convert to units of 1/cm

    # Mask out Region of Interest (ROI)
    mbir_recon = mjp.apply_cylindrical_mask(mbir_recon)

    # Save recon to hdf5
    print("\n*********** save mar and fdk recon in h5 format *************")
    os.makedirs(output_path, exist_ok=True)  # mkdir if directory does not exist
    fbp_path = os.path.join(output_path, f"parallel_fbp_recon.h5")
    mj.export_recon_hdf5(fbp_path, direct_recon, recon_dict=None)
    mbir_path = os.path.join(output_path, f"parallel_mbir_recon.h5")
    mj.export_recon_hdf5(mbir_path, mbir_recon, recon_dict=None, remove_flash=True)
    print("FBP recon saved to {}".format(os.path.abspath(fbp_path)))
    print("FDK recon saved to {}".format(os.path.abspath(mbir_path)))

    # Display the results
    vmax = 10
    mj.slice_viewer(direct_recon, mbir_recon, vmin=0, vmax=vmax, slice_axis=2,
                    slice_label=['FBP', 'MBIR'],
                    title='Comparison between FBP and MBIR reconstructions')


if __name__ == '__main__':
    main()