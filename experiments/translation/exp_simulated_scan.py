import mbirjax as mj


def main():
    # Define geometry
    source_iso_dist = 933.33
    source_detector_dist = 2533.33

    # Define detector size
    num_det_rows = 1944
    num_det_channels = 3072

    # Define view sampling parameters
    num_x_translations = 5
    num_z_translations = 3
    x_spacing = 152
    z_spacing = 186.67

    # Define recon object size
    recon_width = 293.33
    recon_height = 293.33

    # Set recon parameters
    sharpness = 1.0
    phantom_type = "dots"   # Can be "dots" or "text"

    # Generate translation vectors
    translation_vectors = mj.gen_translation_vectors(num_x_translations, num_z_translations, x_spacing, z_spacing)

    # Compute sinogram shape
    sino_shape = (translation_vectors.shape[0], num_det_rows, num_det_channels)

    # Initialize model for reconstruction.
    tct_model = mj.TranslationModel(sino_shape, translation_vectors, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)
    tct_model.set_params(sharpness=sharpness)
    recon_shape = tct_model.get_params('recon_shape')
    print("Auto set recon_shape: ", recon_shape)

    # Set recon shape
    delta_voxel = tct_model.get_params('delta_voxel')
    num_recon_cols = int(recon_width/delta_voxel)
    num_recon_slices = int(recon_height/delta_voxel)
    recon_shape = (21, num_recon_cols, num_recon_slices)
    tct_model.set_params(recon_shape=recon_shape)

    # Print model parameters
    tct_model.print_params()
    # Display translation array
    mj.display_translation_vectors(translation_vectors, recon_shape)

    # Generate ground truth phantom
    gt_recon = mj.gen_translation_phantom(option=phantom_type, recon_shape=recon_shape, font_size=100)

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
    mj.save_volume_as_gif(sino.transpose(0, 2, 1), "/home/yang1581/simulated_scan/dots_sino.gif", vmin=0, vmax=1)
    mj.save_volume_as_gif(gt_recon, "/home/yang1581/simulated_scan/dots_gt_recon.gif", vmin=0, vmax=0.5)
    mj.save_volume_as_gif(recon, "/home/yang1581/simulated_scan/dots_mbir_recon.gif", vmin=0, vmax=0.5)


if __name__ == '__main__':
    main()
