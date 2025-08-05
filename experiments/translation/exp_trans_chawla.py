import mbirjax as mj
import mbirjax.preprocess as mjp
import numpy as np

def main():
    # Define geometry
    source_det_dist_mm = 190
    source_iso_dist_mm = 70
    det_pixel_pitch_mm = 75 / 1000
    x_space_mm = 11.4
    z_space_mm = 14
    num_x_translations = 5
    num_z_translations = 3

    # Define detector size
    num_det_rows = 1944
    num_det_channels = 3072

    # Define object parameters
    num_pixels_in_object = 5
    phantom_type = "text"   # Can be "dots" or "text"
    words = ['P'] * num_pixels_in_object     # List of words to render in the text phantom
    object_width_mm = 22
    object_thickness_mm = 2.15

    # Set recon parameters
    sharpness = 1.0

    # Calculate physical parameters in ALU
    # Note: 1 ALU = detector pixel pitch at iso
    ALU_per_mm = source_det_dist_mm / (source_iso_dist_mm * det_pixel_pitch_mm)
    source_iso_dist_ALU = source_iso_dist_mm * ALU_per_mm
    source_det_dist_ALU = source_iso_dist_ALU
    x_spacing_ALU = x_space_mm * ALU_per_mm
    z_spacing_ALU = z_space_mm * ALU_per_mm
    half_angle_rad = np.arctan2(max(num_det_rows, num_det_channels) / 2.0, source_iso_dist_ALU)
    object_width_ALU = object_width_mm * ALU_per_mm
    object_thickness_ALU = object_thickness_mm * ALU_per_mm

    # Print out important values in ALU
    print("ALU per mm:", ALU_per_mm)
    print("Detector height and width (ALU):", num_det_rows, num_det_channels)
    print("Source to iso distance (ALU):", source_iso_dist_ALU)
    print("x,z spacing (ALU):", x_spacing_ALU, z_spacing_ALU)
    print("Half cone angle (deg):", np.rad2deg(half_angle_rad))
    print("Object width (ALU):", object_width_ALU)
    print("Object thickness (ALU):", object_thickness_ALU)

    # Generate translation vectors
    translation_vectors = mj.gen_translation_vectors(num_x_translations, num_z_translations, x_spacing_ALU, z_spacing_ALU)

    # Compute sinogram shape
    sino_shape = (translation_vectors.shape[0], num_det_rows, num_det_channels)

    # Initialize model for reconstruction.
    tct_model = mj.TranslationModel(sino_shape, translation_vectors, source_detector_dist=source_det_dist_ALU, source_iso_dist=source_iso_dist_ALU)
    tct_model.set_params(sharpness=sharpness)
    recon_shape = tct_model.get_params('recon_shape')

    # Change row pitch to be 1/num_pixel_in_object * object thickness
    tct_model.set_params(delta_recon_row=object_thickness_ALU/num_pixels_in_object)
    tct_model.set_params(qggmrf_nbr_wts=[1.0,1.0,0.1])

    # Print model parameters and display translation array
    tct_model.print_params()
    mj.display_translation_vectors(translation_vectors, recon_shape)

    # Generate ground truth phantom
    start_index = (recon_shape[0] - num_pixels_in_object) // 2
    row_indices = list(range(start_index, start_index + num_pixels_in_object))
    gt_recon = mj.gen_translation_phantom(recon_shape=recon_shape, option=phantom_type, words=words,
                                          font_size=object_width_ALU,
                                          row_indices=row_indices)

    # View test sample
    mj.slice_viewer(gt_recon.transpose(0, 2, 1), title='Ground Truth Recon', slice_label='View', slice_axis=0)

    # Generate synthetic sonogram data
    sino = tct_model.forward_project(gt_recon)

    # View sinogram
    mj.slice_viewer(sino, slice_axis=0, vmin=0, vmax=1, title='Original sinogram', slice_label='View')

    # Generate weights array - for an initial reconstruction, use weights = None, then modify if needed.
    weights = None

    # Perform MBIR reconstruction
    recon, recon_params = tct_model.recon(sino, init_recon=0, weights=weights, max_iterations=15)

    # Display Results
    mj.slice_viewer(gt_recon.transpose(0, 2, 1), recon.transpose(0, 2, 1), vmin=0, vmax=1,
                    title='Object (left) vs. MBIR reconstruction (right)', slice_axis=0)

    # Save as animated gifs
    mj.save_volume_as_gif(gt_recon, "gt_recon.gif", vmin=0, vmax=1)
    mj.save_volume_as_gif(recon, "mbir_recon.gif", vmin=0, vmax=1)


if __name__ == '__main__':
    main()
