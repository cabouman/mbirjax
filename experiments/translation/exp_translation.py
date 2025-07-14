import mbirjax as mj


def main():
    # Define geometry
    source_iso_dist = 64
    source_detector_dist = 64

    # Define detector size
    num_det_rows = 128
    num_det_channels = 128

    # Define view sampling parameters
    num_x_translations = 7
    num_z_translations = 7
    x_spacing = 22
    z_spacing = 22

    # Set recon parameters
    sharpness = 1.0
    phantom_type = "text"   # Can be "dots" or "text"
    words = ["Purdue", "Presents", "Translation", "Tomography"] # List of words to render in the text phantom

    # Generate translation vectors
    translation_vectors = mj.gen_translation_vectors(num_x_translations, num_z_translations, x_spacing, z_spacing)

    # Compute sinogram shape
    sino_shape = (translation_vectors.shape[0], num_det_rows, num_det_channels)

    # Initialize model for reconstruction.
    tct_model = mj.TranslationModel(sino_shape, translation_vectors, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)
    tct_model.set_params(sharpness=sharpness)
    recon_shape = tct_model.get_params('recon_shape')

    # Print model parameters
    tct_model.print_params()
    # Display translation array
    mj.display_translation_vectors(translation_vectors, recon_shape)

    # Generate ground truth phantom
    gt_recon = mj.gen_translation_phantom(recon_shape=recon_shape, option=phantom_type, words=words)

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
