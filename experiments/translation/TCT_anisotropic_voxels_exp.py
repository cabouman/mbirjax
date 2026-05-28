"""
This script tests translation reconstruction with anisotropic voxels
"""

import os
import numpy as np
import time
import pprint
import mbirjax as mj

def main():
    # -------------------------
    # Experiment configuration
    # -------------------------
    object_type = 'text'

    x_spacing = 10
    z_spacing = 10
    num_x_translations = 15
    num_z_translations = 15

    num_det_rows = 128
    num_det_channels = 128

    text_word = ['P']

    voxel_row_aspect = 8.0
    voxel_slice_aspect = 0.5

    # Output path
    output_path = './output/'  # path to store output recon images

    # -------------------------
    # Initialize translation model
    # -------------------------
    print("Initialize model...")
    # Compute source to iso and source to detector distance
    source_iso_dist = min(num_det_rows, num_det_channels) / 2
    source_detector_dist = source_iso_dist

    # Generate translation vectors
    translation_vectors = mj.gen_translation_vectors(num_x_translations, num_z_translations, x_spacing, z_spacing)

    # Compute sinogram shape
    num_views = translation_vectors.shape[0]
    sinogram_shape = (num_views, num_det_rows, num_det_channels)

    # Initialize model
    tct_model = mj.TranslationModel(sinogram_shape, translation_vectors, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)

    # -------------------------
    # Generate simulated data
    # -------------------------
    print("Generating demo data...")
    # Set number of pixels in object
    recon_shape = tct_model.get_params('recon_shape')
    num_pixels_in_object = recon_shape[0] // 2
    text = text_word * num_pixels_in_object

    # Generate phantom
    text_start_index = (recon_shape[0] - num_pixels_in_object) // 2
    row_indices = list(range(text_start_index, text_start_index + num_pixels_in_object))
    phantom = mj.gen_translation_phantom(recon_shape=recon_shape, option=object_type, text=text, font_size=70,
                                         text_row_indices=row_indices)

    # Generate synthetic sinogram data
    print('Creating sinogram')
    sinogram = tct_model.forward_project(phantom)
    sinogram = np.asarray(sinogram)

    # View the sinogram
    mj.slice_viewer(sinogram, slice_axis=0, title="Synthetic sinogram", slice_label='View')

    # -------------------------
    # Perform MBIR reconstruction
    # -------------------------
    # Set parameters for reconstruction
    tct_model.set_params(partition_sequence=5*[0,] + 100*[1, 3,])
    tct_model.set_params(voxel_row_aspect=voxel_row_aspect)
    tct_model.set_params(voxel_slice_aspect=voxel_slice_aspect)
    tct_model.auto_set_recon_geometry()

    # Print out model parameters
    tct_model.print_params()

    # Perform MBIR reconstruction
    print('Starting recon')
    mbir_recon, mbir_dict = tct_model.recon(sinogram, max_iterations=200)

    ### Generate a phantom with the same voxel dimensions as the recon:
    # Set number of pixels in object
    recon_shape = tct_model.get_params('recon_shape')
    num_pixels_in_object = recon_shape[0] // 2
    text = text_word * num_pixels_in_object

    # Generate phantom
    text_start_index = (recon_shape[0] - num_pixels_in_object) // 2
    row_indices = list(range(text_start_index, text_start_index + num_pixels_in_object))
    phantom = mj.gen_translation_phantom(recon_shape=recon_shape, option=object_type, text=text,
                                         font_size=70,
                                         text_row_indices=row_indices, voxel_slice_aspect=voxel_slice_aspect)

    # Save reconstruction results
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, f'TCT_demo_recon.h5')
    mj.export_recon_hdf5(output_path, mbir_recon, recon_dict=mbir_dict, top_margin=0, bottom_margin=0)

    # Display results
    mj.slice_viewer(phantom.transpose(0, 2, 1), mbir_recon.transpose(0, 2, 1), slice_axis=0, vmin=0,
                    title="Phantom (left) vs VCD Recon (right)")


if __name__ == '__main__':
    main()