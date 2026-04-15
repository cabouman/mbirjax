# This script run Translation CT simulation experiment on Purdue BGA sample

import sys
import mbirjax as mj
import mbirjax.preprocess as mjp
import numpy as np
import os
from scipy.ndimage import zoom

def get_experiment_params(experiment_name):
    """Returns experiment-specific parameters."""
    base_config = {
        "source_det_dist_mm": 80,
        "source_iso_dist_mm": 30,
        "det_pixel_pitch_mm": 75 / 1000,
        "x_view_space_mm": 5,
        "z_view_space_mm": 3,
        "num_x_translations": 11,
        "num_z_translations": 9,
        "num_det_rows": 1936,
        "num_det_channels": 3064,
        "qggmrf_nbr_weights": [0.1, 1.0, 1.0],
    }

    if experiment_name == "experiment1":
        return base_config

    elif experiment_name == "experiment2":
        override_config = {
            # Modify only what's different for experiment2
            # Example: "x_view_space_mm": 5,
        }
        return {**base_config, **override_config}

    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")


def main():
    experiment = sys.argv[1] if len(sys.argv) > 1 else "experiment1"
    params = get_experiment_params(experiment)

    # Recon parameters
    downsample_factor = 2  # Spatial downsampling
    subsample_view_factor = 2  # View  downsampling

    # Path to dataset
    dataset_url = '/depot/bouman/data/Zeiss/purdue_BGA/17U1-250TC-Normal_Tomo_No_HART.txrm'

    # Output path
    output_path = './output/'  # path to store TCT simulation recon

    # Load the sinogram and metadata
    print("\n********** Load sinogram and metadata from the data **************")
    sinogram, cone_beam_params, optional_params = mjp.zeiss_cb.compute_sino_and_params(dataset_url, downsample_factor=(downsample_factor, downsample_factor),
                                                                                       subsample_view_factor=subsample_view_factor)

    # Construct cone beam model
    print("\n********** Construct cone beam model **************")
    ct_model = mj.ConeBeamModel(**cone_beam_params)
    ct_model.set_params(**optional_params)

    # Print out cone beam model parameters
    ct_model.print_params()

    # Perform FDK reconstruction to obtain a 3D volume for phantom generation
    print("\n********** Perform FDK reconstruction **************")
    direct_recon = ct_model.direct_recon(sinogram)

    # Extract TCT simulation parameters
    source_det_dist_mm = params["source_det_dist_mm"]
    source_iso_dist_mm = params["source_iso_dist_mm"]
    det_pixel_pitch_mm = params["det_pixel_pitch_mm"]
    x_view_space_mm = params["x_view_space_mm"]
    z_view_space_mm = params["z_view_space_mm"]
    num_x_translations = params["num_x_translations"]
    num_z_translations = params["num_z_translations"]
    num_det_rows = params["num_det_rows"]
    num_det_channels = params["num_det_channels"]
    qggmrf_nbr_weights = params["qggmrf_nbr_weights"]

    # Unit conversion: convert all physical quantities in ALU (1 ALU = 1 mm here)
    unit_conversion = {'um': 1.0, 'mm': 1000.0}

    ALU_unit = 'mm'
    ALU_value = 1

    # Convert physical units to ALU
    delta_det_channel_ALU = det_pixel_pitch_mm * unit_conversion['mm'] / unit_conversion[ALU_unit]
    delta_det_row_ALU = delta_det_channel_ALU
    source_iso_dist_ALU = source_iso_dist_mm * unit_conversion['mm'] / unit_conversion[ALU_unit]
    source_det_dist_ALU = source_det_dist_mm * unit_conversion['mm'] / unit_conversion[ALU_unit]
    x_view_spacing_ALU = x_view_space_mm * unit_conversion['mm'] / unit_conversion[ALU_unit]
    z_view_spacing_ALU = z_view_space_mm * unit_conversion['mm'] / unit_conversion[ALU_unit]
    half_angle_rad = np.arctan2(max(num_det_rows, num_det_channels) / 2.0, source_iso_dist_ALU)

    # Generate translation vectors
    translation_vectors = mj.gen_translation_vectors(num_x_translations, num_z_translations, x_view_spacing_ALU, z_view_spacing_ALU)

    # Compute sinogram shape
    sino_shape = (translation_vectors.shape[0], num_det_rows, num_det_channels)

    # Initialize model for forward projection
    tct_model = mj.TranslationModel(sino_shape, translation_vectors, source_detector_dist=source_det_dist_ALU, source_iso_dist=source_iso_dist_ALU)

    # Calculate recon_shape, delta_voxel, and delta_recon_row parameters
    recon_shape, delta_voxel, delta_recon_row = mj.utilities.calc_tct_recon_params(source_det_dist_ALU,
                                                                     source_iso_dist_ALU,
                                                                     delta_det_row_ALU,
                                                                     delta_det_channel_ALU, sino_shape,
                                                                     translation_vectors)

    delta_recon_row = delta_voxel   # For this simulation, I set row pitch = voxel pitch in (column, slice) plane

    # Generate ground truth phantom
    print("\n********** Generate ground truth phantom from FDK Reconstruction **************")
    gt_phantom = direct_recon.copy()
    gt_phantom = np.maximum(gt_phantom, 0)   # clip negative recon value for valid forward projection

    recon_voxel_pitch = ct_model.get_params('delta_voxel')   # get voxel pitch of cone beam FDK recon

    # Resample phantom to have TCT simulation voxel pitch
    gt_phantom = zoom(gt_phantom, zoom=(recon_voxel_pitch / delta_recon_row, recon_voxel_pitch / delta_voxel, recon_voxel_pitch / delta_voxel), order=0)

    # Pad (column, slice) plane with zeros to compensate for object out of field of view
    pad_h = int(0.5 * gt_phantom.shape[1])
    pad_w = int(0.5 * gt_phantom.shape[2])

    gt_phantom = np.pad(gt_phantom,((0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)))

    # Crop excess empty rows
    gt_phantom = gt_phantom[150:510, :, :]

    # Set parameters for forward projection
    tct_model.set_params(positivity_flag=True)
    tct_model.set_params(partition_sequence=5*[0,] + 100*[1, 3,])
    tct_model.set_params(delta_det_channel=delta_det_channel_ALU)
    tct_model.set_params(delta_det_row=delta_det_row_ALU)
    tct_model.set_params(delta_voxel=delta_voxel)
    tct_model.set_params(delta_recon_row=delta_recon_row)
    tct_model.set_params(recon_shape=gt_phantom.shape)
    tct_model.set_params(qggmrf_nbr_wts=qggmrf_nbr_weights)
    tct_model.set_params(alu_unit=ALU_unit)
    tct_model.set_params(alu_value=ALU_value)

    # Display translation array
    translation_vectors_display = np.asarray(translation_vectors).copy()
    translation_vectors_display[:, 0] /= delta_voxel
    translation_vectors_display[:, 2] /= delta_voxel
    translation_vectors_display[:, 1] /= delta_recon_row
    mj.display_translation_vectors(translation_vectors_display, recon_shape)

    tct_model.print_params()

    # View ground truth phantom
    mj.slice_viewer(gt_phantom.transpose(0, 2, 1), vmin=0, vmax=0.8, title='Ground Truth Recon', slice_label='View', slice_axis=0)

    # Generate synthetic sonogram data
    sino = tct_model.forward_project(gt_phantom)
    sino = np.asarray(sino)

    # View synthetic sinogram
    mj.slice_viewer(sino, slice_axis=0, title='Synthetic sinogram', slice_label='View')

    # Perform MBIR reconstruction
    print("\n********** Perform MBIR reconstruction **************")
    mbir_recon, mbir_dict = tct_model.recon(sino, max_iterations=200, stop_threshold_change_pct=0.2)

    # Save reconstruction results
    print("\n*********** save MBIR recon in h5 format *************")
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, f'TCT_simulation_recon.h5')
    mj.export_recon_hdf5(output_path, mbir_recon, recon_dict=mbir_dict, top_margin=0, bottom_margin=0)

    # Display results
    mj.slice_viewer(gt_phantom.transpose(0, 2, 1), mbir_recon.transpose(0, 2, 1), vmin=0, vmax=0.8,
                    title='Object (left), MBIR reconstruction (right)',
                    slice_axis=0)


if __name__ == '__main__':
    main()