# This script is for running general Translation CT simulation experiments

import sys
import mbirjax as mj
import mbirjax.preprocess as mjp
import numpy as np

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
        "num_det_rows": 1936,
        "num_det_channels": 3064,
        "num_pixels_in_object": 3,
        "phantom_type": "text",
        "text": ['P', 'P', 'P'],
        "object_width_mm": 22,
        "object_thickness_mm": 12.15,
        "qggmrf_nbr_weights": [0.1, 1.0, 1.0],
        "sharpness": 1.0,
        "max_iterations": 80,
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
            'text': ['P']*7,
            "qggmrf_nbr_weights": [1.0, 1.0, 1.0],
            'max_iterations': 80,
        }
        return {**base_config, **override_config}

    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")

def main():
    experiment = sys.argv[1] if len(sys.argv) > 1 else "experiment1"
    params = get_experiment_params(experiment)

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
    sharpness = params["sharpness"]
    max_iterations = params["max_iterations"]


    # Calculate physical parameters in ALU
    # Note: 1 ALU = detector pixel pitch at iso
    ALU_per_mm = source_det_dist_mm / (source_iso_dist_mm * det_pixel_pitch_mm)
    source_iso_dist_ALU = source_iso_dist_mm * ALU_per_mm
    source_det_dist_ALU = source_iso_dist_ALU
    x_view_spacing_ALU = x_view_space_mm * ALU_per_mm
    z_view_spacing_ALU = z_view_space_mm * ALU_per_mm
    half_angle_rad = np.arctan2(max(num_det_rows, num_det_channels) / 2.0, source_iso_dist_ALU)
    object_width_ALU = object_width_mm * ALU_per_mm
    object_thickness_ALU = object_thickness_mm * ALU_per_mm
    delta_recon_row = object_thickness_ALU / num_pixels_in_object

    # Generate translation vectors
    translation_vectors = mj.gen_translation_vectors(num_x_translations, num_z_translations, x_view_spacing_ALU, z_view_spacing_ALU)

    # Compute sinogram shape
    sino_shape = (translation_vectors.shape[0], num_det_rows, num_det_channels)

    # Initialize model for reconstruction.
    tct_model = mj.TranslationModel(sino_shape, translation_vectors, source_detector_dist=source_det_dist_ALU, source_iso_dist=source_iso_dist_ALU)
    tct_model.set_params(sharpness=sharpness)
    recon_shape = tct_model.get_params('recon_shape')

    # Set number of rows in recon to match desired object thickness
    if experiment == "experiment1":
        recon_shape = (5*int(object_thickness_ALU / delta_recon_row), recon_shape[1], recon_shape[2])
        tct_model.set_params(recon_shape=recon_shape)
    tct_model.set_params(delta_recon_row=delta_recon_row)
    tct_model.set_params(qggmrf_nbr_wts=qggmrf_nbr_weights)
    tct_model.set_params(partition_sequence=[0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7])


    print("\n*************** Experimental Parameters ***************")
    print("source_detector_dist (ALU):", tct_model.get_params('source_detector_dist'))
    print("source_iso_dist (ALU):", tct_model.get_params('source_iso_dist'))
    print("delta_det_channel (ALU):", tct_model.get_params('delta_det_channel'))
    print("delta_det_row (ALU):", tct_model.get_params('delta_det_row'))
    print("Magnification:", tct_model.get_magnification())
    print("Half cone angle (deg):", np.rad2deg(half_angle_rad))

    print("x,z view spacing (ALU):", x_view_spacing_ALU, z_view_spacing_ALU)
    print("Object width (ALU):", object_width_ALU)
    print("Object thickness (ALU):", object_thickness_ALU)

    print("Recon shape:", recon_shape)
    print("delta_voxel (ALU):", tct_model.get_params('delta_voxel'))
    print("delta_recon_row (ALU):", tct_model.get_params('delta_recon_row'))
    print("************************************************")


    # Print model parameters and display translation array
    #tct_model.print_params()
    mj.display_translation_vectors(translation_vectors, recon_shape)

    # Generate ground truth phantom
    text_start_index = (recon_shape[0] - num_pixels_in_object) // 2
    row_indices = list(range(text_start_index, text_start_index + num_pixels_in_object))
    gt_recon = mj.gen_translation_phantom(recon_shape=recon_shape, option=phantom_type, text=text,
                                          font_size=object_width_ALU/0.73,
                                          text_row_indices=row_indices)

    # View test sample
    mj.slice_viewer(gt_recon.transpose(0, 2, 1), title='Ground Truth Recon', slice_label='View', slice_axis=0)

    # Generate synthetic sonogram data
    sino = tct_model.forward_project(gt_recon)
    sino = np.asarray(sino)

    # View sinogram
    mj.slice_viewer(sino, slice_axis=0, vmin=0, vmax=1, title='Original sinogram', slice_label='View')

    # Perform MBIR reconstruction
    direct_recon = tct_model.direct_recon(sino)
    recon, recon_params = tct_model.recon(sino, stop_threshold_change_pct=0, max_iterations=max_iterations)

    # Display Results
    mj.slice_viewer(gt_recon.transpose(0, 2, 1), direct_recon.transpose(0, 2, 1), recon.transpose(0, 2, 1),
                    vmin=0, vmax=1, title='Object (left), FDK recon (middle), MBIR reconstruction (right)', slice_axis=0)

    # Save as animated gifs
    mj.save_volume_as_gif(gt_recon, "gt_recon.gif", vmin=0, vmax=1)
    mj.save_volume_as_gif(recon, "mbir_recon.gif", vmin=0, vmax=1)


if __name__ == '__main__':
    main()
