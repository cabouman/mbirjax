import numpy as np
import pprint
import jax.numpy as jnp
import mbirjax as mj
import mbirjax.preprocess as mjp

if __name__ == "__main__":
    """
    This is a script to develop, debug, and tune the parallel and cone beam mbirjax code.
    """
    # Choose the geometry type
    geometry_type = 'cone'  # 'cone' or 'parallel'

    print('Using {} geometry.'.format(geometry_type))

    # Set parameters
    num_views = 64
    num_det_rows = 40
    num_det_channels = 128
    sharpness = 0.0
    
    # These can be adjusted to describe the geometry in the cone beam case.
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
        ct_model = mj.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)
    elif geometry_type == 'parallel':
        ct_model = mj.ParallelBeamModel(sinogram_shape, angles)
    else:
        raise ValueError('Invalid geometry type.  Expected cone or parallel, got {}'.format(geometry_type))

    # Generate 3D Shepp Logan phantom
    print('Creating phantom')
    phantom = ct_model.gen_modified_3d_sl_phantom()

    # Generate synthetic sinogram data
    print('Creating sinogram')
    sinogram = ct_model.forward_project(phantom)

    # View sinogram
    mj.slice_viewer(sinogram, title='Original sinogram', slice_axis=0, slice_label='View')

    # Generate weights array - for an initial reconstruction, use weights = None, then modify as desired.
    weights = None
    # weights = ct_model.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')

    # Set reconstruction parameter values
    ct_model.set_params(sharpness=sharpness, verbose=1)

    # Print out model parameters
    ct_model.print_params()

    # Perform MBIR reconstruction
    recon, recon_params = ct_model.recon(sinogram, weights=weights)

    # Print out parameters used in recon
    pprint.pprint(recon_params['recon_params'])

    # Use export/import functions to write out recon to disk
    import os
    os.makedirs('output', exist_ok=True)
    mj.export_recon_hdf5('output/recon_hdf5', recon)
    recon, _ = mj.import_recon_hdf5('output/recon_hdf5')

    # Display results
    mj.slice_viewer(phantom, recon, title='Phantom (left) vs MBIR Recon (right)')
