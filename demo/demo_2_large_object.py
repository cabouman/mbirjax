import numpy as np
import time
import pprint
import jax
import jax.numpy as jnp
import mbirjax.parallel_beam

if __name__ == "__main__":
    """
    This script demonstrates how to improve your reconstruction when the object does not project completely inside
    the detector.  For simplicity, we show this only for parallel beam, but the same steps apply for cone beam.  
    """
    # For the demo, we create some synthetic data by first making a phantom, then forward projecting it to
    # obtain a sinogram.  In this case, we create a large phantom that does not project fully to the detector.

    # Set parameters for the problem size - you can vary these, but if you make num_det_rows very small relative to
    # channels, then the generated phantom may not have an interior.
    num_views = 120
    num_det_rows = 80
    num_det_channels = 100

    start_angle = - np.pi / 2
    end_angle = np.pi / 2

    # Initialize sinogram
    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

    # ################
    # Data generation: Here we create a phantom and then project it to create a sinogram.

    # In a real application, you would just load your sinogram as a numpy array and use numpy.transpose so that it
    # has axes in the order (views, rows, channels).  Assuming the rotation axis is vertical, then increasing the row
    # index nominally moves down the rotation axis and increasing the channel index moves to the right as seen from
    # the source.

    # The default recon shape for parallel beam is
    # (rows, columns, slices) = (num_det_channels, num_det_channels, num_det_rows),
    # where we assume that the recon voxels are cubes and have the same size as the detector elements.
    # Here we generate a phantom that is bigger than the detector to show how to deal with this case.
    ct_model_for_generation = mbirjax.ParallelBeamModel(sinogram_shape, angles)

    # Generate 3D Shepp Logan phantom
    print('Creating phantom that projects partially outside the detector')
    phantom_row_scale = 2.0
    phantom_col_scale = 1.0
    phantom_rows = int(num_det_channels * phantom_row_scale)
    phantom_cols = int(num_det_channels * phantom_col_scale)
    phantom_slices = num_det_rows
    phantom_shape = (phantom_rows, phantom_cols, phantom_slices)
    phantom = mbirjax.generate_3d_shepp_logan_low_dynamic_range(phantom_shape)

    mbirjax.slice_viewer(phantom, title='Original phantom', slice_axis=2, slice_label='Slice')

    # Generate synthetic sinogram data
    print('Creating sinogram')
    ct_model_for_generation.set_params(recon_shape=phantom_shape)
    sinogram = ct_model_for_generation.forward_project(phantom)
    sinogram = np.array(sinogram)

    # View sinogram
    mbirjax.slice_viewer(sinogram, title='Original sinogram\nChange view to see projections outside detector',
                         slice_axis=0, slice_label='View')

    # ###############
    # Reconstruction: Here we use mbirjax to reconstruct the given sinogram.
    weights = None

    # Set reconstruction parameter values
    ct_model_for_recon = mbirjax.ParallelBeamModel(sinogram_shape, angles)

    # Print out model parameters
    ct_model_for_recon.print_params()

    # ##########################
    # Default VCD reconstruction
    # This recon will have significant artifacts because of the missing projections.
    print('Starting default recon')
    recon, recon_params = ct_model_for_recon.recon(sinogram, weights=weights)

    # Print out parameters used in recon
    pprint.pprint(recon_params._asdict(), compact=True)

    # Display results
    title = 'Default recon: Phantom (left) vs VCD Recon (right)'
    title += '\nAdjust intensity range to [0, 1] to see internal artifacts from projection outside detector.'
    title += '\nAdjust intensity range to [1.5, 2] to see outer ring from projection outside detector.'
    mbirjax.slice_viewer(phantom, recon, title=title)

    # ###########################################
    # Increased regularization VCD reconstruction
    # We can reduce the artifacts by increasing regularization (decreasing sharpness).
    sharpness = -1.5
    ct_model_for_recon.set_params(sharpness=sharpness)
    print('Starting recon with increased regularization')
    recon, recon_params = ct_model_for_recon.recon(sinogram, weights=weights)

    # Print out parameters used in recon
    pprint.pprint(recon_params._asdict(), compact=True)

    # Display results
    title = 'Recon with sharpness = {:.1f}: Phantom (left) vs VCD Recon (right)'.format(sharpness)
    title += '\nAdjust intensity range to [0, 1] to see reduced internal artifacts from projection outside detector.'
    title += '\nOuter ring is still evident in intensity range [1, 2].'
    mbirjax.slice_viewer(phantom, recon, title=title)

    # ###############################
    # Padded recon VCD reconstruction
    # Alternatively, we can pad the recon to allow for a partial reconstruction of the pixels that
    # project outside the detector in some views.  This reduces the artifacts without increasing
    # regularization and greatly reduces the outer ring seen in the non-padded recon.
    sharpness = 0
    recon_row_scale = 1.5
    recon_col_scale = 1.0
    ct_model_for_recon.scale_recon_shape(row_scale=recon_row_scale, col_scale=recon_col_scale)
    ct_model_for_recon.set_params(sharpness=sharpness)
    print('Starting padded recon')
    recon, recon_params = ct_model_for_recon.recon(sinogram, weights=weights)

    # Print out parameters used in recon
    pprint.pprint(recon_params._asdict(), compact=True)

    # Display results
    title = 'Padded recon with sharpness = {:.1f}: Phantom (left) vs VCD Recon (right)'.format(sharpness)
    title += '\nPadding the recon reduces the internal artifacts even with default sharpness.'
    title += '\nOuter ring is gone, and the partially projected pixels are partially recovered.'
    mbirjax.slice_viewer(phantom, recon, title=title)



