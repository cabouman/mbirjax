### This script is for reconstructing parallel beam and cone beam CT data from Zeiss Ultra scanner and Zeiss Versa scanner

import os
import sys, os
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
    downsample_factor = 2       # Spatial downsampling
    subsample_view_factor = 2   # View  downsampling

    # Determine which dataset to use
    dataset_index = 0       # Index into one of the datasets below
    use_local_data = False   # If True, then use local_data_directory/filename.  Otherwise, use the data_depot path.
    local_data_directory = './data'  # Directory for local testing - you should copy the files below into this directory

    # Path to the dataset
    depot_data_sets = [
        '/depot/bouman/data/ORNL/versa/ParAM-Round-1_Z62.txrm',             # 0: Cylinder with rods and notches
        '/depot/bouman/data/ORNL/versa/SiC-SiC_CompositeFFOV_tomo-A.txrm',  # 1:
        '/depot/bouman/data/Zeiss/purdue/Scan_tomo-A.txrm',                 # 2:
        '/depot/bouman/data/Zeiss/purdue_BGA/17U1-250TC-Normal_Tomo_HART_360_HART.txrm',    # 3: Solder drops, high-angle
        '/depot/bouman/data/Zeiss/purdue_BGA/17U1-250TC-Normal_Tomo_No_HART.txrm',          # 4: Solder drops, equiangle
        '/depot/bouman/data/Zeiss/foam512R1N3000_raw_scan.txrm',                             # 5: Synthetic foam data
        '/depot/bouman/data/AFRL/lipp/Black_Sheep_tomo-B_CS-2.txrm',
        '/depot/bouman/data/AFRL/lipp/Black_Sheep_tomo-C_CS0.txrm',
        '/depot/bouman/data/AFRL/lipp/Black_Sheep_tomo-D_CS-3.8.txrm',
    ]

    dataset_path = depot_data_sets[dataset_index]
    if use_local_data:
        filename = os.path.basename(dataset_path)
        dataset_path = os.path.join(local_data_directory, filename)

    # Output path
    output_path = './output/'  # path to store output recon images

    # Load the sinogram and construct the tomography model.  The model class is auto-selected from the
    # Zeiss scanner type ('ultra' -> parallel-beam, otherwise cone-beam), and the returned model already
    # has its parameters and reconstruction geometry set.
    print("\n********** Load sinogram and construct tomography model **************")
    sinogram, ct_model = mjp.zeiss.get_sino_and_model(dataset_path, downsample_factor=(downsample_factor, downsample_factor),
                                                      subsample_view_factor=subsample_view_factor)

    # Sharpness and snr_db
    ct_model.set_params(sharpness=sharpness, snr_db=snr_db)

    # Display the sinogram
    mj.slice_viewer(sinogram, slice_axis=0, title='Original sinogram')

    # Print out model parameters
    ct_model.print_params()

    # Perform direct reconstruction
    print("\n********** Perform direct reconstruction **************")
    direct_recon = ct_model.direct_recon(sinogram)
    mj.slice_viewer(direct_recon, slice_axis=2, title='Direct reconstruction')

    # Perform sinogram per-view alignment
    # print("\n********** Perform sinogram alignment **************")
    # sinogram = mjp.align_sino_views(ct_model, sinogram, direct_recon)

    # Weights
    weights = mj.gen_weights(sinogram, weight_type='transmission_root')

    # Perform FDK reconstruction
    # print("\n********** Perform FDK reconstruction after alignment **************")
    # direct_recon = ct_model.direct_recon(sinogram)

    # Perform MBIR reconstruction
    print("\n********** Perform MBIR reconstruction **************")
    mbir_recon, recon_dict = ct_model.recon(sinogram, weights=weights)

    # Save recon to hdf5
    print("\n*********** save mbir and fdk recon in h5 format *************")
    os.makedirs(output_path, exist_ok=True)  # mkdir if directory does not exist
    direct_path = os.path.join(output_path, f"direct_recon.h5")
    mj.export_recon_hdf5(direct_path, direct_recon, recon_dict=None)
    mbir_path = os.path.join(output_path, f"mbir_recon.h5")
    mj.export_recon_hdf5(mbir_path, mbir_recon, recon_dict=None, remove_flash=True)
    print("Direct recon saved to {}".format(os.path.abspath(direct_path)))
    print("MBIR recon saved to {}".format(os.path.abspath(mbir_path)))

    # Display the results
    mj.slice_viewer(direct_recon, mbir_recon, slice_axis=2,
                    slice_label=['Direct', 'MBIR'],
                    title='Comparison between Direct and MBIR reconstructions')


if __name__ == '__main__':
    main()
