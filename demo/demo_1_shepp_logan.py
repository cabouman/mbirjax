import numpy as np
import time
import pprint
import jax
import jax.numpy as jnp
import mbirjax.parallel_beam

if __name__ == "__main__":
    """
    This script demonstrates the basic mbirjax code by creating a 3D phantom inspired by Shepp-Logan, forward
    projecting it to create a sinogram, and then using Model-Based, Multi-Granular Vectorized Coordinate Descent
    to do a reconstruction.  
    """
    # Choose the geometry type
    geometry_type = 'cone'  # 'cone' or 'parallel'

    print('Using {} geometry.'.format(geometry_type))

    # For the demo, we create some synthetic data by first making a phantom, then forward projecting it to
    # obtain a sinogram.

    # Set parameters for the problem size - you can vary these, but if you make num_det_rows very small relative to
    # channels, then the generated phantom may not have an interior.
    num_views = 64
    num_det_rows = 40
    num_det_channels = 128

    # Increase sharpness by 1 or 2 to get clearer edges, possibly with more high-frequency artifacts.
    # Decrease by 1 or 2 to get softer edges and smoother interiors.
    sharpness = 0.0
    
    # For cone beam geometry, we need to describe the distances source to detector and source to rotation axis.
    # np.Inf is an allowable value, in which case this is essentially parallel beam
    source_detector_dist = 4 * num_det_channels
    source_iso_dist = source_detector_dist

    # For cone beam reconstruction, we need a little more than 180 degrees for full coverage.
    if geometry_type == 'cone':
        detector_cone_angle = 2 * np.arctan2(num_det_channels / 2, source_detector_dist)
    else:
        detector_cone_angle = 0
    start_angle = -(np.pi + detector_cone_angle) * (1/2)
    end_angle = (np.pi + detector_cone_angle) * (1/2)

    # Initialize sinogram
    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

    # ################
    # Data generation: Here we create a phantom and then project it to create a sinogram.

    # In a real application, you would just load your sinogram as a numpy array, use numpy.transpose so that it
    # has axes in the order (views, rows, channels).  Assuming the rotation axis is vertical, then increasing the row
    # index nominally moves down the rotation axis and increasing the channel index moves to the right as seen from
    # the source.
    if geometry_type == 'cone':
        ct_model = mbirjax.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)
    elif geometry_type == 'parallel':
        ct_model = mbirjax.ParallelBeamModel(sinogram_shape, angles)
    else:
        raise ValueError('Invalid geometry type.  Expected cone or parallel, got {}'.format(geometry_type))

    # Generate 3D Shepp Logan phantom
    print('Creating phantom')
    phantom = ct_model.gen_modified_3d_sl_phantom()

    # Generate synthetic sinogram data
    print('Creating sinogram')
    sinogram = ct_model.forward_project(phantom)
    sinogram = np.array(sinogram)

    # View sinogram
    mbirjax.slice_viewer(sinogram, title='Original sinogram', slice_axis=0, slice_label='View')

    # ###############
    # Reconstruction: Here we use mbirjax to reconstruct the given sinogram.

    # Generate weights array - for an initial reconstruction, use weights = None, then modify as needed.
    weights = None
    # weights = ct_model.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')

    # Set reconstruction parameter values
    ct_model.set_params(sharpness=sharpness)

    # Print out model parameters
    ct_model.print_params()

    # ##########################
    # Perform VCD reconstruction
    time0 = time.time()
    recon, recon_params = ct_model.recon(sinogram, weights=weights)

    recon.block_until_ready()
    elapsed = time.time() - time0
    # ##########################

    # Print out parameters used in recon
    pprint.pprint(recon_params._asdict(), compact=True)

    max_diff = np.amax(np.abs(phantom - recon))
    print('Geometry = {}'.format(geometry_type))
    nrmse = np.linalg.norm(recon - phantom) / np.linalg.norm(phantom)
    pct_95 = np.percentile(np.abs(recon - phantom), 95)
    print('NRMSE between recon and phantom = {}'.format(nrmse))
    print('Maximum pixel difference between phantom and recon = {}'.format(max_diff))
    print('95% of recon pixels are within {} of phantom'.format(pct_95))

    mbirjax.get_memory_stats()
    print('Elapsed time for recon is {:.3f} seconds'.format(elapsed))

    # Display results
    mbirjax.slice_viewer(phantom, recon, title='Phantom (left) vs VCD Recon (right)')

