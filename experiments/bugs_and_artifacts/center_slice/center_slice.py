### This script is for reconstructing cone beam CT data from ORNL Zeiss scanner

import os
import sys, os
import numpy as np
import jax.numpy as jnp
import pprint
import mbirjax as mj
import mbirjax.preprocess as mjp
import pickle

pp = pprint.PrettyPrinter(indent=4)

def main():
    # Recon parameters
    sharpness = 1.5
    snr_db = 35.0
    downsample_factor = 3       # Spatial downsampling
    subsample_view_factor = 5   # View  downsampling

    # Determine which dataset to use
    dataset_index = 4       # Index into one of the datasets below
    use_local_data = True   # If True, then use local_data_directory/filename.  Otherwise, use the data_depot path.
    local_data_directory = '../../zeiss/data'  # Directory for local testing - you should copy the files below into this directory

    # Path to the dataset
    depot_data_sets = [
        '/depot/bouman/data/ORNL/versa/ParAM-Round-1_Z62.txrm',             # 0: Cylinder with rods and notches
        '/depot/bouman/data/ORNL/versa/SiC-SiC_CompositeFFOV_tomo-A.txrm',  # 1:
        '/depot/bouman/data/Zeiss/purdue/Scan_tomo-A.txrm',                 # 2:
        '/depot/bouman/data/Zeiss/purdue_BGA/17U1-250TC-Normal_Tomo_HART_360_HART.txrm',    # 3: Solder drops, high-angle
        '/depot/bouman/data/Zeiss/purdue_BGA/17U1-250TC-Normal_Tomo_No_HART.txrm',          # 4: Solder drops, equiangle
        '/depot/bouman/data/Zeiss/foam512R1N3000_raw_scan.txrm'                             # 5: Synthetic foam data
    ]

    dataset_path = depot_data_sets[dataset_index]
    if use_local_data:
        filename = os.path.basename(dataset_path)
        dataset_path = os.path.join(local_data_directory, filename)

    # Output path
    output_path = './output/'  # path to store output recon images

    # Load the sinogram and metadata
    print("\n********** Load sinogram and metadata from the data **************")
    load_file = False
    flip_sino_up_down = False
    params_file = 'zeiss_params.pkl'
    data_file = 'zeiss_data.npz'
    if load_file:
        with open(params_file, "rb") as f:
            cone_beam_params, optional_params = pickle.load(f)
        data = np.load(data_file)
        direct_recon, weights, sinogram = data['direct_recon'], data['weights'], data['sinogram']
        crop_size = 0
        if crop_size > 0:
            direct_recon = direct_recon[:, :, crop_size:-crop_size]
            weights = weights[:, crop_size:-crop_size, :]
            sinogram = sinogram[:, crop_size:-crop_size, :]
            cone_beam_params['sinogram_shape'] = sinogram.shape
        if flip_sino_up_down:
            direct_recon = direct_recon[:, :, ::-1]
            weights = weights[:, ::-1, :]
            sinogram = sinogram[:, ::-1, :]
    else:
        crop_size = 400
        sinogram, cone_beam_params, optional_params = mjp.zeiss_cb.compute_sino_and_params(dataset_path, downsample_factor=(downsample_factor, downsample_factor),
                                                                                           subsample_view_factor=subsample_view_factor, crop_pixels_bottom=crop_size, crop_pixels_top=crop_size)

    # Construct cone beam model
    print("\n********** Construct cone beam model **************")
    ct_model = mj.ConeBeamModel(**cone_beam_params)
    ct_model.set_params(**optional_params)

    # Rerun auto-parameter functions because we changed the assumed detector pitch
    ct_model.auto_set_recon_geometry(sinogram.shape) # Reset default recon shape

    # Sharpness and snr_db
    ct_model.set_params(sharpness=sharpness, snr_db=snr_db)

    # Display the sinogram
    # mj.slice_viewer(sinogram, slice_axis=0, title='Original sinogram')

    # Print out model parameters
    ct_model.print_params()

    if not load_file:
        # Perform FDK reconstruction
        print("\n********** Perform FDK reconstruction **************")
        direct_recon = ct_model.direct_recon(sinogram)
        # mj.slice_viewer(direct_recon, slice_axis=2, title='Direct reconstruction')

        # Perform sinogram per-view alignment
        # print("\n********** Perform sinogram alignment **************")
        # sinogram = mjp.align_sino_views(ct_model, sinogram, direct_recon)

        # Weights
        weights = mj.gen_weights(sinogram, weight_type='transmission_root')

        np.savez('zeiss_data.npz', direct_recon=direct_recon, weights=weights, sinogram=sinogram)
        with open("zeiss_params.pkl", "wb") as f:
            pickle.dump((cone_beam_params, optional_params), f)
        # Perform FDK reconstruction
        # print("\n********** Perform FDK reconstruction after alignment **************")
        # direct_recon = ct_model.direct_recon(sinogram)

    if direct_recon is None:
        print("\n********** Perform FDK reconstruction **************")
        direct_recon = ct_model.direct_recon(sinogram)
    # mj.slice_viewer(jnp.swapaxes(direct_recon, 0, 2), slice_axis=1)
    # Perform MBIR reconstruction
    print("\n********** Perform MBIR reconstruction **************")
    mbir_recon, recon_dict = ct_model.recon(sinogram, init_recon=direct_recon, weights=weights, max_iterations=4)

    # # Save recon to hdf5
    # print("\n*********** save mbir and fdk recon in h5 format *************")
    # os.makedirs(output_path, exist_ok=True)  # mkdir if directory does not exist
    # fdk_path = os.path.join(output_path, f"cone_fdk_recon.h5")
    # mj.export_recon_hdf5(fdk_path, direct_recon, recon_dict=None)
    # mbir_path = os.path.join(output_path, f"cone_mbir_recon.h5")
    # mj.export_recon_hdf5(mbir_path, mbir_recon, recon_dict=None, remove_flash=True)
    # print("FDK recon saved to {}".format(os.path.abspath(fdk_path)))
    # print("MBIR recon saved to {}".format(os.path.abspath(mbir_path)))

    # Display the results
    mj.slice_viewer(jnp.swapaxes(direct_recon, 0, 2), jnp.swapaxes(mbir_recon, 0, 2), slice_axis=1, vmin=0, vmax=0.2,
                    slice_label=['FDK', 'MBIR'],
                    title='Comparison between FDK and MBIR reconstructions')


if __name__ == '__main__':
    main()
