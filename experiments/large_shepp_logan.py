import jax
import numpy as np
import time
import pprint
import jax.numpy as jnp
import mbirjax
import os

# Set the GPU memory fraction for JAX
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.03'

if __name__ == "__main__":
    """
    This is a script to develop, debug, and tune the parallel beam mbirjax code.
    """
    # Choose the geometry type
    geometry_type = 'cone'  # 'cone' or 'parallel'

    print('Using {} geometry.'.format(geometry_type))

    # Set parameters
    num_views = 1024
    num_det_rows = 1024
    num_det_channels = 1024
    sharpness = 0.0

    print('\tnum_views = {}'.format(num_views))
    print('\tnum_det_rows = {}'.format(num_det_rows))
    print('\tnum_det_channels = {}'.format(num_det_channels))

    # These can be adjusted to describe the geometry in the cone beam case.
    # np.Inf is an allowable value, in which case this is essentially parallel beam
    source_detector_dist = 4 * num_det_channels
    source_iso_dist = source_detector_dist

    if geometry_type == 'cone':
        detector_cone_angle = 2 * np.arctan2(num_det_channels / 2, source_detector_dist)
    else:
        detector_cone_angle = 0
    start_angle = -(np.pi + detector_cone_angle) * (1 / 2)
    end_angle = (np.pi + detector_cone_angle) * (1 / 2)

    # Initialize sinogram
    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

    # Set up the model
    if geometry_type == 'cone':
        ct_model = mbirjax.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist,
                                         source_iso_dist=source_iso_dist)
    elif geometry_type == 'parallel':
        ct_model = mbirjax.ParallelBeamModel(sinogram_shape, angles)
    else:
        raise ValueError('Invalid geometry type.  Expected cone or parallel, got {}'.format(geometry_type))

    # Generate 3D Shepp Logan phantom
    print('Creating phantom')
    phantom = ct_model.gen_modified_3d_sl_phantom()

    # ct_model.pixels_per_batch = 2000
    # ct_model.views_per_batch = 3000
    # ct_model.create_projectors()

    # Generate synthetic sinogram data
    print('Creating sinogram')
    sinogram = ct_model.forward_project(phantom)
    #
    # full_indices = mbirjax.gen_full_indices(phantom.shape)
    # voxel_values = ct_model.get_voxels_at_indices(phantom, full_indices)
    # _ = ct_model.sparse_forward_project(voxel_values, full_indices)
    #
    # _ = ct_model.sparse_back_project(sinogram, full_indices)
    #
    # time0 = time.time()
    # _ = ct_model.sparse_back_project(sinogram, full_indices)
    # _ = ct_model.sparse_back_project(sinogram, full_indices)
    # _ = ct_model.sparse_back_project(sinogram, full_indices)
    # elapsed = time.time() - time0
    # print('Per backward = {:.6f}'.format(elapsed / 3))
    #
    # time0 = time.time()
    # _ = ct_model.sparse_forward_project(voxel_values, full_indices)
    # _ = ct_model.sparse_forward_project(voxel_values, full_indices)
    # _ = ct_model.sparse_forward_project(voxel_values, full_indices)
    # elapsed = time.time() - time0
    # print('Per forward = {:.6f}'.format(elapsed / 3))
    #
    # time0 = time.time()
    # _ = ct_model.sparse_back_project(sinogram, full_indices)
    # _ = ct_model.sparse_back_project(sinogram, full_indices)
    # _ = ct_model.sparse_back_project(sinogram, full_indices)
    # elapsed = time.time() - time0
    # print('Per backward = {:.6f}'.format(elapsed / 3))
    #
    # time0 = time.time()
    # _ = ct_model.sparse_forward_project(voxel_values, full_indices)
    # _ = ct_model.sparse_forward_project(voxel_values, full_indices)
    # _ = ct_model.sparse_forward_project(voxel_values, full_indices)
    # elapsed = time.time() - time0
    # print('Per forward = {:.6f}'.format(elapsed / 3))
    #
    # time0 = time.time()
    # _ = ct_model.sparse_back_project(sinogram, full_indices)
    # _ = ct_model.sparse_back_project(sinogram, full_indices)
    # _ = ct_model.sparse_back_project(sinogram, full_indices)
    # elapsed = time.time() - time0
    # print('Per backward = {:.6f}'.format(elapsed / 3))

    phantom = jax.device_put(phantom, jax.devices('cpu')[0])
    # View sinogram
    # mbirjax.slice_viewer(sinogram, title='Original sinogram', slice_axis=0, slice_label='View')

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
    sinogram = np.asarray(sinogram)
    recon, recon_params = ct_model.recon(sinogram, weights=weights, compute_prior_loss=False, max_iterations=10)

    recon.block_until_ready()
    elapsed = time.time() - time0
    mbirjax.get_memory_stats()
    print('Elapsed time for recon is {:.3f} seconds'.format(elapsed))
    # ##########################
    # mbirjax.get_memory_stats()

    # Print out parameters used in recon
    # pprint.pprint(recon_params._asdict())
    #
    phantom = np.array(phantom)
    recon = np.array(recon)
    max_diff = np.amax(np.abs(recon - phantom))
    print('Geometry = {}'.format(geometry_type))
    nrmse = np.linalg.norm(recon - phantom) / np.linalg.norm(phantom)
    pct_95 = np.percentile(np.abs(recon - phantom), 95)
    print('NRMSE between recon and phantom = {}'.format(nrmse))
    print('Maximum pixel difference between phantom and recon = {}'.format(max_diff))
    print('95% of recon pixels are within {} of phantom'.format(pct_95))

    # Display results
    # mbirjax.slice_viewer(phantom, recon, title='Phantom (left) vs VCD Recon (right)')
