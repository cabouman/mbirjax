import numpy as np
import mbirjax as mj


def generate_translation_data(source_iso_dist, source_detector_dist, num_det_rows, num_det_channels,
                              num_x_translations, num_z_translations, x_spacing, z_spacing, verbose=0):

    # Generate all translation vectors
    num_views = num_x_translations * num_z_translations
    translation_vectors = np.zeros((num_views, 3))

    x_center = (num_x_translations - 1) / 2
    z_center = (num_z_translations - 1) / 2

    idx = 0
    for row in range(0, num_z_translations):
        for col in range(0, num_x_translations):
            dx = (col - x_center) * x_spacing
            dz = (row - z_center) * z_spacing
            dy = 0
            translation_vectors[idx] = [dx, dy, dz]
            idx += 1

    # Compute sinogram shape
    sinogram_shape = (num_views, num_det_rows, num_det_channels)

    # Define the model for sinogram generation
    ct_model_for_generation = mj.TranslationModel(sinogram_shape, translation_vectors, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)

    # Check the reconstruction size that is set by auto_set_recon_size
    auto_recon_shape = ct_model_for_generation.get_params('recon_shape')
    print("Auto set recon size = ", auto_recon_shape)

    ### Generate ground truth recon
    np.random.seed(42)
    gt_recon = np.zeros(auto_recon_shape, dtype=np.float32)

    # Define Central rows
    central_start = auto_recon_shape[0] // 3
    central_end = 2 * auto_recon_shape[0] // 3

    # Calculate number of 1s per slice (1% of all points)
    row_size = auto_recon_shape[1] * auto_recon_shape[2]
    num_ones_per_row = int(row_size * 0.01)

    # Fill central rows with random 1s
    for row_idx in range(central_start, central_end):
        flat_row = gt_recon[row_idx].flatten()

        positions_ones = np.random.choice(row_size, num_ones_per_row, replace=False)
        flat_row[positions_ones] = 1.0

        gt_recon[row_idx] = flat_row.reshape(auto_recon_shape[1:])

    # Generate sinogram shape
    sino_shape = (num_views, num_det_rows, num_det_channels)

    if verbose > 0:
        # Print out closest and farthest voxels
        source_iso_dist, delta_recon_row = ct_model_for_generation.get_params(['source_iso_dist', 'delta_recon_row'])
        max_translation = np.amax(translation_vectors, axis=0)
        min_translation = np.amin(translation_vectors, axis=0)
        source_to_closest_pixel = source_iso_dist - (0.5 * gt_recon.shape[0] * delta_recon_row) - max_translation[1]
        source_to_farthest_pixel = source_iso_dist + (0.5 * gt_recon.shape[0] * delta_recon_row) - min_translation[1]
        print("Source to closest pixel distance with translation = ", source_to_closest_pixel)
        print("Source to farthest pixel distance with translation = ", source_to_farthest_pixel)

    return gt_recon, sino_shape, translation_vectors


def main():
    # Define geometry
    source_iso_dist = 32
    source_detector_dist = 64

    # Define detector size
    num_det_rows = 128
    num_det_channels = 128

    # Define view sampling parameters
    num_x_translations = 9
    num_z_translations = 9
    x_spacing = 16
    z_spacing = 16

    # Set recon parameters
    sharpness = 0.0

    # Set sinogram generation parameter values
    gt_recon, sino_shape, translation_vectors = generate_translation_data(source_iso_dist, source_detector_dist,
                                                              num_det_rows, num_det_channels,
                                                              num_x_translations, num_z_translations, x_spacing, z_spacing, verbose=1)

    # View test sample
    mj.slice_viewer(gt_recon, title='Ground Truth Recon', slice_label='View', slice_axis=0)

    # Initialize model for reconstruction.
    tct_model = mj.TranslationModel(sino_shape, translation_vectors, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)
    tct_model.set_params(sharpness=sharpness, recon_shape=gt_recon.shape)

    # Generate synthetic sonogram data
    sino = tct_model.forward_project(gt_recon)

    # View sinogram
    mj.slice_viewer(sino, slice_axis=0, vmin=0, vmax=1, title='Original sinogram', slice_label='View')

    # Generate weights array - for an initial reconstruction, use weights = None, then modify if needed.
    weights = None

    # Print model parameters
    tct_model.print_params()
    # Print out translation array
    print("Translation vectors:\n", translation_vectors)

    # Perform MBIR reconstruction
    recon, recon_params = tct_model.recon(sino, init_recon=0, weights=weights)

    # Display Results
    mj.slice_viewer(gt_recon, recon, vmin=0, vmax=1, title='Object (left) vs. MBIR reconstruction (right)', slice_axis=0)


if __name__ == '__main__':
    main()