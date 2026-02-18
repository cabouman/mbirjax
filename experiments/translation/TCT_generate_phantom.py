# This script generates simulated phantoms used in Translation CT simulation experiments

import sys
import mbirjax as mj
import mbirjax.preprocess as mjp
import numpy as np
import os

def get_experiment_params(experiment_name):
    """Returns experiment-specific parameters."""
    base_config = {
        "source_det_dist_mm": 190,
        "source_iso_dist_mm": 70,
        "det_pixel_pitch_mm": 75 / 1000,
        "x_view_space_mm": 11.4,
        "z_view_space_mm": 14,
        "num_x_translations": 5,
        "num_z_translations": 3,
        "num_det_rows": 1883,
        "num_det_channels": 3064,
        "num_pixels_in_object": 12,
        "phantom_type": "text",
        "text": ['P'],
        "object_width_mm": 22,
        "object_thickness_mm": 2.15,
        "qggmrf_nbr_weights": [0.1, 1.0, 1.0],
    }

    if experiment_name == "experiment1":
        return base_config

    elif experiment_name == "experiment2":
        override_config = {
            # Modify only what's different for experiment2
            # Example: "phantom_type": "dots",
            'source_det_dist_mm': 120,
            'source_iso_dist_mm': (70/190)*120,
            'x_view_space_mm': 7.25,
            'z_view_space_mm': 7,
            'num_x_translations': 9,
            'num_z_translations': 5,
            'num_pixels_in_object': 7,
            "qggmrf_nbr_weights": [1.0, 1.0, 1.0],
        }
        return {**base_config, **override_config}

    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")

def main():
    experiment = sys.argv[1] if len(sys.argv) > 1 else "experiment1"
    params = get_experiment_params(experiment)

    # Output path
    output_path = './output/'  # path to store ground truth phantom and synthetic sinogram

    # Set parameters for experiment
    source_det_dist_mm = params["source_det_dist_mm"]
    source_iso_dist_mm = params["source_iso_dist_mm"]
    det_pixel_pitch_mm = params["det_pixel_pitch_mm"]
    x_view_space_mm = params["x_view_space_mm"]
    z_view_space_mm = params["z_view_space_mm"]
    num_x_translations = params["num_x_translations"]
    num_z_translations = params["num_z_translations"]
    num_det_rows = params["num_det_rows"]
    num_det_channels = params["num_det_channels"]
    num_pixels_in_object = params["num_pixels_in_object"]
    phantom_type = params["phantom_type"]
    text = params["text"]
    object_width_mm = params["object_width_mm"]
    object_thickness_mm = params["object_thickness_mm"]
    qggmrf_nbr_weights = params["qggmrf_nbr_weights"]

    # Calculate physical parameters in ALU
    # Note: 1 ALU = 1 delta_det_channel_unit
    # Unit conversion table
    unit_conversion = {'um': 1.0, 'mm': 1000.0}

    # Set 1 ALU = 1 delta_det_channel_unit
    ALU_unit = 'um'
    ALU_value = 1

    # Convert physical units to ALU
    delta_det_channel_ALU = det_pixel_pitch_mm * unit_conversion['mm'] / unit_conversion[ALU_unit]
    delta_det_row_ALU = delta_det_channel_ALU
    source_iso_dist_ALU = source_iso_dist_mm * unit_conversion['mm'] / unit_conversion[ALU_unit]
    source_det_dist_ALU = source_det_dist_mm * unit_conversion['mm'] / unit_conversion[ALU_unit]
    x_view_spacing_ALU = x_view_space_mm * unit_conversion['mm'] / unit_conversion[ALU_unit]
    z_view_spacing_ALU = z_view_space_mm * unit_conversion['mm'] / unit_conversion[ALU_unit]
    half_angle_rad = np.arctan2(max(num_det_rows, num_det_channels) / 2.0, source_iso_dist_ALU)
    object_width_ALU = object_width_mm * unit_conversion['mm'] / unit_conversion[ALU_unit]
    object_thickness_ALU = object_thickness_mm * unit_conversion['mm'] / unit_conversion[ALU_unit]
    delta_phantom_row = object_thickness_ALU / num_pixels_in_object

    # Generate translation vectors
    translation_vectors = mj.gen_translation_vectors(num_x_translations, num_z_translations, x_view_spacing_ALU, z_view_spacing_ALU)

    # Compute sinogram shape
    sino_shape = (translation_vectors.shape[0], num_det_rows, num_det_channels)

    # Initialize model for forward projection
    tct_model = mj.TranslationModel(sino_shape, translation_vectors, source_detector_dist=source_det_dist_ALU, source_iso_dist=source_iso_dist_ALU)

    # Calculate recon_shape, delta_voxel, and delta_recon_row parameters
    recon_shape, delta_voxel, _ = mj.utilities.calc_tct_recon_params(source_det_dist_ALU,
                                                                     source_iso_dist_ALU,
                                                                     delta_det_row_ALU,
                                                                     delta_det_channel_ALU, sino_shape,
                                                                     translation_vectors)


    # Set number of pixels in object
    text = text * num_pixels_in_object
    print("Number of pixels in object:", num_pixels_in_object)

    # Set phantom shape
    phantom_shape = (2 * num_pixels_in_object, recon_shape[1], recon_shape[2])

    # Set parameters for forward projection
    tct_model.set_params(positivity_flag=True)
    tct_model.set_params(delta_det_channel=delta_det_channel_ALU)
    tct_model.set_params(delta_det_row=delta_det_row_ALU)
    tct_model.set_params(delta_voxel=delta_voxel)
    tct_model.set_params(recon_shape=phantom_shape)
    tct_model.set_params(delta_recon_row=delta_phantom_row)
    tct_model.set_params(qggmrf_nbr_wts=qggmrf_nbr_weights)

    # Display translation array
    translation_vectors_display = np.asarray(translation_vectors).copy()
    translation_vectors_display[:, 0] /= delta_voxel
    translation_vectors_display[:, 2] /= delta_voxel
    translation_vectors_display[:, 1] /= delta_phantom_row
    mj.display_translation_vectors(translation_vectors_display, phantom_shape)

    # Generate ground truth phantom
    delta_voxel = tct_model.get_params('delta_voxel')
    text_start_index = (phantom_shape[0] - num_pixels_in_object) // 2
    row_indices = list(range(text_start_index, text_start_index + num_pixels_in_object))
    gt_phantom = mj.gen_translation_phantom(recon_shape=phantom_shape, option=phantom_type, text=text,
                                          font_size=object_width_ALU / (delta_voxel * 0.73),
                                          text_row_indices=row_indices)

    # View ground truth phantom
    mj.slice_viewer(gt_phantom.transpose(0, 2, 1), title='Ground Truth Recon', slice_label='View', slice_axis=0)

    # Generate synthetic sonogram data
    sino = tct_model.forward_project(gt_phantom)
    sino = np.asarray(sino)

    # View synthetic sinogram
    mj.slice_viewer(sino, slice_axis=0, title='Synthetic sinogram', slice_label='View')

    # Save the ground truth recon
    os.makedirs(output_path, exist_ok=True)
    gt_phantom_path = os.path.join(output_path, 'TCT_simulated_phantom.npy')
    np.save(gt_phantom_path, gt_phantom)

    # Store the parameters for reconstruction
    translation_params = dict()
    translation_params['sinogram_shape'] = sino_shape
    translation_params['translation_vectors'] = translation_vectors
    translation_params['source_detector_dist'] = source_det_dist_ALU
    translation_params['source_iso_dist'] = source_iso_dist_ALU

    optional_params = dict()
    optional_params['delta_det_channel'] = delta_det_channel_ALU
    optional_params['delta_det_row'] = delta_det_row_ALU
    optional_params['delta_voxel'] = delta_voxel
    optional_params['alu_unit'] = ALU_unit
    optional_params['alu_value'] = ALU_value

    # Save the synthetic sinogram and reconstruction parameters together
    sino_config_path = os.path.join(output_path, 'TCT_synthetic_sino_and_config.npz')
    np.savez(sino_config_path, sino=sino, translation_params=translation_params, optional_params=optional_params)

if __name__ == '__main__':
    main()