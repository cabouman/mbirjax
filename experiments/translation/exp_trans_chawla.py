import mbirjax as mj
import mbirjax.preprocess as mjp


def main():
    # Define geometry
    source_det_dist_mm = 190
    source_iso_dist_mm = 70
    det_pixel_pitch_mm = 75 / 1000
    x_space_mm = 11.4
    z_space_mm = 14
    num_x_translations = 5
    num_z_translations = 3
    row_pitch_scaling_factor = 1.0

    # Define detector size
    num_det_rows = 1944
    num_det_channels = 3072

    # Set recon parameters
    sharpness = 1.0
    phantom_type = "text"   # Can be "dots" or "text"
    words = ['P', 'U']     # List of words to render in the text phantom

    # Calculate physical parameters
    mag_physical = source_det_dist_mm / source_iso_dist_mm
    det_pixel_pitch_at_iso_mm = det_pixel_pitch_mm / mag_physical
    det_height_at_iso = det_pixel_pitch_at_iso_mm * num_det_rows
    det_width_at_iso = det_pixel_pitch_at_iso_mm * num_det_channels

    print("Magnification:", mag_physical)
    print("Detector pixel pitch at ISO (mm):", det_pixel_pitch_at_iso_mm)
    print("Detector height and width at ISO in mm:", det_height_at_iso, det_width_at_iso)

    # Compute geometry parameters in ALU
    source_iso_dist = source_iso_dist_mm / det_pixel_pitch_at_iso_mm
    source_det_dist = source_iso_dist
    x_spacing = x_space_mm / det_pixel_pitch_at_iso_mm
    z_spacing = z_space_mm / det_pixel_pitch_at_iso_mm

    # Generate translation vectors
    translation_vectors = mj.gen_translation_vectors(num_x_translations, num_z_translations, x_spacing, z_spacing)

    # Compute sinogram shape
    sino_shape = (translation_vectors.shape[0], num_det_rows, num_det_channels)

    # Initialize model for reconstruction.
    tct_model = mj.TranslationModel(sino_shape, translation_vectors, source_detector_dist=source_det_dist, source_iso_dist=source_iso_dist)
    tct_model.set_params(sharpness=sharpness)
    recon_shape = tct_model.get_params('recon_shape')

    # Change row pitch based on user preference
    delta_recon_row = row_pitch_scaling_factor * tct_model.get_params("delta_recon_row")
    tct_model.set_params(delta_recon_row=delta_recon_row)

    # Print parameters
    tct_model.print_params()

    # Print model parameters
    tct_model.print_params()
    # Display translation array
    mj.display_translation_vectors(translation_vectors, recon_shape)

    # Generate ground truth phantom
    gt_recon = mj.gen_translation_phantom(recon_shape=recon_shape, option=phantom_type, words=words, font_size=800)

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
