import numpy as np
import time
import pprint
import jax.numpy as jnp
import mbirjax.plot_utils as pu
import mbirjax.parallel_beam

if __name__ == "__main__":
    """
    This is a script to develop, debug, and tune the parallel beam mbirjax code.
    """
    # Choose the geometry type
    geometry_type = 'cone'  # 'cone' or 'parallel'

    print('Using {} geometry.'.format(geometry_type))

    # Set parameters
    num_views = 64
    num_det_rows = 64
    num_det_channels = 64
    sharpness = 0.0
    
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

    # Generate 3D Shepp Logan phantom
    print('Creating phantom')
    recon_shape = ct_model.get_params('recon_shape')
    phantom = np.zeros(recon_shape)  # ct_model.gen_modified_3d_sl_phantom()
    phantom[16:48, 16:48, 16:48] = 1

    # Generate synthetic sinogram data
    print('Creating sinogram')
    sinogram = ct_model.forward_project(phantom)

    # View sinogram
    pu.slice_viewer(sinogram, title='Original sinogram', slice_axis=0, slice_label='View')

    # Generate weights array - for an initial reconstruction, use weights = None, then modify as desired.
    weights = None
    # weights = ct_model.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')

    # Set reconstruction parameter values
    ct_model.set_params(sharpness=sharpness, verbose=1)

    # Print out model parameters
    ct_model.print_params()

    # ##########################
    # Perform VCD reconstruction
    time0 = time.time()
    recon, recon_params = ct_model.recon(sinogram, weights=weights, num_iterations=15)

    recon.block_until_ready()
    elapsed = time.time() - time0
    print('Elapsed time for recon is {:.3f} seconds'.format(elapsed))
    # ##########################

    # Print out parameters used in recon
    pprint.pprint(recon_params._asdict())

    max_diff = np.amax(np.abs(phantom - recon))
    print('Geometry = {}'.format(geometry_type))
    nrmse = np.linalg.norm(recon - phantom) / np.linalg.norm(phantom)
    pct_95 = np.percentile(np.abs(recon - phantom), 95)
    print('NRMSE between recon and phantom = {}'.format(nrmse))
    print('Maximum pixel difference between phantom and recon = {}'.format(max_diff))
    print('95% of recon pixels are within {} of phantom'.format(pct_95))

    # Display results
    pu.slice_viewer(phantom, recon, title='Phantom (left) vs VCD Recon (right)')

