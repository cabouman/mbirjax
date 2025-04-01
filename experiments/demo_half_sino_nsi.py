# -*- coding: utf-8 -*-
"""demo_1_shepp_logan.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1zG_H6CDjuQxeMRQHan3XEyX2YVKcSSNC

**MBIRJAX: Basic Demo**

See the [MBIRJAX documentation](https://mbirjax.readthedocs.io/en/latest/) for an overview and details.

This script demonstrates the basic MBIRJAX code by creating a 3D phantom inspired by Shepp-Logan, forward projecting it to create a sinogram, and then using MBIRJAX to perform a Model-Based, Multi-Granular Vectorized Coordinate Descent reconstruction.

For the demo, we create some synthetic data by first making a phantom, then forward projecting it to obtain a sinogram.

In a real application, you would load your sinogram as a numpy array and use numpy.transpose if needed so that it
has axes in the order (views, rows, channels).  For reference, assuming the rotation axis is vertical, then increasing the row index nominally moves down the rotation axis and increasing the channel index moves to the right as seen from the source.

Select a GPU as runtime type for best performance.
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install mbirjax

import warnings
import time
import pprint
import jax.numpy as jnp
import mbirjax
import os

if __name__ == "__main__":

    """**Data generation:** We load the NSI files.

    Note:  the sliders on the viewer won't work in notebook form.  For that you'll need to run the python code with an interactive matplotlib backend, typcially using the command line or a development environment like Spyder or Pycharm to invoke python.

    """
    """**Set the geometry parameters**"""

    output_path = './results'
    if not os.path.exists(output_path):
        warnings.warn('The output directory {} does not exist. \n'
                      'Create one, preferably with a soft link to scratch, as in ln -s /scratch/gautschi/<username>/.cache/mbirjax {}.'.format(output_path, output_path))
        raise NotADirectoryError(output_path)

    # Choose the geometry type
    geometry_type = 'cone'

    # NSI file path
    dataset_dir = '/depot/bouman/data/nsi_demo_data/demo_nsi_vert_no_metal_all_views'

    downsample_factor = [1, 1]  # downsample factor of scan images along detector rows and detector columns.
    subsample_view_factor = 2  # view subsample factor.

    # #### recon parameters
    sharpness = 1.0
    # ###################### End of parameters

    print("\n*******************************************************",
            "\n************** NSI dataset preprocessing **************",
            "\n*******************************************************")
    time0 = time.time()
    crop_pixels_sides = 32
    crop_pixels_top = 250
    crop_pixels_bottom = 20
    sinogram, cone_beam_params, optional_params = \
        mbirjax.preprocess.nsi.compute_sino_and_params(dataset_dir,
                                                       downsample_factor=downsample_factor,
                                                       subsample_view_factor=subsample_view_factor,
                                                       crop_pixels_sides=crop_pixels_sides,
                                                       crop_pixels_top=crop_pixels_top,
                                                       crop_pixels_bottom=crop_pixels_bottom)

    # mbirjax.slice_viewer(sinogram, slice_axis=0, title='Full sinogram', slice_label='View')

    print("\n*******************************************************",
            "\n***************** Set up MBIRJAX model ****************",
            "\n*******************************************************")
    # ConeBeamModel constructor
    sinogram_shape = sinogram.shape
    angles = cone_beam_params['angles']
    source_detector_dist = cone_beam_params['source_detector_dist']
    source_iso_dist = cone_beam_params['source_iso_dist']
    ct_model_for_full_recon = mbirjax.ConeBeamModel(sinogram_shape, angles,
                                                    source_detector_dist=source_detector_dist,
                                                    source_iso_dist=source_iso_dist)
    # Set additional geometry arguments
    ct_model_for_full_recon.set_params(**optional_params)
    # Set reconstruction parameter values
    ct_model_for_full_recon.set_params(sharpness=sharpness, verbose=1)


    # Take roughly the half of the sinogram and specify roughly half of the volume
    full_recon_shape = ct_model_for_full_recon.get_params('recon_shape')
    num_recon_slices = full_recon_shape[2]
    half_recons = []
    num_extra_rows = 5
    num_det_rows_half = sinogram_shape[1] // 2 + num_extra_rows

    # Initialize the model for reconstruction.
    sinogram_half_shape = (sinogram_shape[0], num_det_rows_half, sinogram_shape[2])

    # ConeBeamModel constructor
    ct_model_for_half_recon = mbirjax.ConeBeamModel(sinogram_half_shape, angles=angles,
                                                    source_detector_dist=source_detector_dist,
                                                    source_iso_dist=source_iso_dist)
    # Set additional geometry arguments
    ct_model_for_half_recon.set_params(**optional_params)
    # Set reconstruction parameter values
    ct_model_for_half_recon.set_params(sharpness=sharpness, verbose=1)

    recon_shape_half = ct_model_for_half_recon.get_params('recon_shape')
    ct_model_for_half_recon.set_params(use_gpu='sinograms')
    num_recon_slices_half = recon_shape_half[2]

    for sinogram_half, sign in zip([sinogram[:, 0:num_det_rows_half], sinogram[:, -num_det_rows_half:]], [1, -1]):

        # View sinogram
        title = 'Half of sinogram \nUse the sliders to change the view or adjust the intensity range.'
        # mbirjax.slice_viewer(sinogram_half, slice_axis=0, title=title, slice_label='View')

        delta_voxel, delta_det_row = ct_model_for_full_recon.get_params(['delta_voxel', 'delta_det_row'])
        recon_slice_offset = sign * (- delta_voxel * ((num_recon_slices-1)/2 - (num_recon_slices_half-1)/2))
        det_row_offset = sign * delta_det_row * ((sinogram_shape[1]-1)/2 - (num_det_rows_half-1)/2)

        ct_model_for_half_recon.set_params(recon_slice_offset=recon_slice_offset, det_row_offset=det_row_offset)

        # Print out model parameters
        ct_model_for_half_recon.print_params()

        """**Do the reconstruction and display the results.**"""

        # ##########################
        # Perform VCD reconstruction
        print('Starting recon')
        time0 = time.time()
        recon, recon_params = ct_model_for_half_recon.recon(sinogram_half)

        recon.block_until_ready()
        elapsed = time.time() - time0
        half_recons.append(recon)
        # ##########################
        # mbirjax.slice_viewer(recon)

    num_overlap_slices = 2 * num_recon_slices_half - num_recon_slices
    recon = mbirjax.stitch_arrays(half_recons, num_overlap_slices)

    mbirjax.preprocess.export_recon_to_hdf5(recon, os.path.join(output_path, "full_size_recon.h5"),
                                            recon_description="MBIRJAX recon of MAR phantom",
                                            alu_description="1 ALU = 0.508 mm")


    # Print parameters used in recon
    pprint.pprint(recon_params._asdict(), compact=True)

    mbirjax.get_memory_stats()
    print('Elapsed time for recon is ' + time.strftime('%H hrs, %M mins, %S secs', time.gmtime(elapsed)))

    # Display results
    title = 'Standard VCD recon (left) and residual with 2 halves stitched VCD Recon (right) \nThe residual is (stitched recon) - (standard recon).'
    mbirjax.slice_viewer(recon, title=title)

    """**Next:** Try changing some of the parameters and re-running or try [some of the other demos](https://mbirjax.readthedocs.io/en/latest/demos_and_faqs.html).  """