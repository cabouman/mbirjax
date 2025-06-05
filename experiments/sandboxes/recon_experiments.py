import numpy as np
import time
import jax.numpy as jnp
import mbirjax

if __name__ == "__main__":
    """
    This is a script to develop, debug, and tune vcd reconstruction
    """
    geometry_type = 'cone'  # 'cone' or 'parallel'

    # Set parameters
    num_views = 64
    num_det_rows = 64
    num_det_channels = 128

    sharpness = 0.0  # This can be used to adjust the regularization
    positivity_flag = True  # Set True to enforce positivity on the recon values

    # These can be adjusted to describe the geometry in the cone beam case.
    # np.Inf is an allowable value, in which case this is essentially parallel beam
    source_detector_dist = 4 * num_det_channels
    source_iso_dist = source_detector_dist

    if geometry_type == 'cone':
        detector_cone_angle = 2 * np.arctan2(num_det_channels / 2, source_detector_dist)
    else:
        detector_cone_angle = 0
    start_angle = -(np.pi + detector_cone_angle) * (1/2)
    end_angle = (np.pi + detector_cone_angle) * (1/2)

    # Initialize sinogram
    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

    # Set up the model
    if geometry_type == 'cone':
        ct_model = mbirjax.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)
    elif geometry_type == 'parallel':
        ct_model = mbirjax.ParallelBeamModel(sinogram_shape, angles)
    else:
        raise ValueError('Invalid geometry type.  Expected cone or parallel, got {}'.format(geometry_type))

    # Here are other things you might want to do

    # Change the recon shape, which has the form (rows, columns, slices)
    # All else being equal, a smaller recon shape will result in a smaller projection on the detector.
    change_recon_shape = False
    if change_recon_shape:
        recon_shape = ct_model.get_params('recon_shape')
        recon_shape = tuple(dim // 2 for dim in recon_shape)  # Or set to whatever you like
        ct_model.set_params(recon_shape=recon_shape)

    # Change the voxel side length (voxels are cubes)
    # The default detector side length is 1.0 (arbitrary units), and delta_voxel has the same units
    # If you change the voxel size, you might want to view the sinogram to see the results.
    # All else being equal, a smaller voxel size will result in a smaller projection on the detector.
    change_voxel_pitch = False
    if change_voxel_pitch:
        ct_model.set_params(delta_voxel=3.0)

    # Change the position of the detector relative to the object
    change_detector_offset = False
    if change_detector_offset:
        ct_model.set_params(det_channel_offset=2.5)  # Positive moves the projected image to the right.
        ct_model.set_params(det_row_offset=5)  # Positive moves the projected image down.

    # Change the granularity as a function of iteration.
    change_partition_sequence = False
    if change_partition_sequence:
        granularity = [1, 4, 8, 64, 256]  # Small granularity means many voxels are updated simultaneously.
        partition_sequence = [0, 1, 2, 3, 4, 2, 3, 2, 3, 3, 3, 3, 3, 3]  # Each entry is an index into granularity.

    # Generate 3D Shepp Logan phantom
    print('Creating phantom')
    phantom = ct_model.gen_modified_3d_sl_phantom()
    mbirjax.slice_viewer(phantom, phantom.transpose((0, 2, 1)),
                         title='Phantom\nLeft: single phantom slice (axial)    Right: single phantom row (coronal)',
                         slice_label='Phantom slice', slice_label2='Phantom row', slice_axis2=0)
    # Generate synthetic sinogram data
    print('Creating sinogram: {} geometry'.format(geometry_type))
    sinogram = ct_model.forward_project(phantom)
    # del phantom  # For large recons, you can delete the phantom to preserve memory.

    # View sinogram
    title = 'Original sinogram ({} geometry, {} views)'.format(geometry_type, num_views)
    mbirjax.slice_viewer(sinogram, title=title, slice_label='View', slice_axis=0)

    # Generate weights array
    weights = ct_model.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')

    # Set reconstruction parameter values
    ct_model.set_params(sharpness=sharpness, verbose=1)
    ct_model.set_params(positivity_flag=positivity_flag)

    # Print out model parameters
    ct_model.print_params()

    # ##########################
    # Perform VCD reconstruction
    print('Starting recon')
    time0 = time.time()
    recon, recon_params = ct_model.recon(sinogram, weights=weights)

    recon.block_until_ready()
    elapsed = time.time() - time0
    print('Elapsed time for recon is {:.3f} seconds'.format(elapsed))
    # ##########################

    # Print the results to screen (and file if desired)
    print_summary_to_file = False
    filename = 'recon timing.txt'

    def print_summary(file=None):
        print('\n-------------------------', file=file)
        print('Current stats:', file=file)
        print('Geometry type: {}'.format(geometry_type), file=file)
        print('Sinogram shape (views, rows, channels) = {}'.format(sinogram.shape), file=file)
        print('Recon shape (rows, columns, slices) = {}'.format(recon.shape), file=file)
        print('Sigma_y = {}'.format(recon_params.regularization_params['sigma_y']))
        print('Sigma_x = {}'.format(recon_params.regularization_params['sigma_x']))
        print('Number of iterations = {}'.format(recon_params.num_iterations))
        print('Granularity = {}'.format(recon_params.granularity))
        print('Partition sequence = {}'.format(recon_params.partition_sequence))
        print('Final RMSE = {:.3f}'.format(recon_params.fm_rmse[-1]))
        print('Final prior loss = {:.3f}'.format(recon_params.prior_loss[-1]))
        print('Elapsed time for recon is {:.3f} seconds'.format(elapsed), file=file)
        mbirjax.get_memory_stats(print_results=True, file=file)
        print('-------------------------', file=file)

    print_summary()

    if print_summary_to_file:
        with open(filename, 'a') as f:
            print_summary(f)

    # Display results
    title = 'VCD recon ({} geometry, {} views)\nLeft: single recon slice (axial)    Right: single recon row (coronal)'.format(geometry_type, num_views)
    mbirjax.slice_viewer(recon, recon.transpose((0, 2, 1)), title=title,
                         slice_label='Recon slice', slice_label2='Recon row', slice_axis2=0)

