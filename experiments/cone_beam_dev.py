
import numpy as np
import jax.numpy as jnp
import jax
import time
import matplotlib.pyplot as plt
import gc
import warnings
import mbirjax

if __name__ == "__main__":
    """
    This is a script to develop, debug, and tune the cone beam beam projector
    """
    # ##########################
    # Do all the setup
    view_batch_size = 4  # Reduce this for large detector row/channel count, increase for smaller
    pixel_batch_size = 2048

    # Initialize sinogram parameters
    num_views = 128
    num_det_rows = 128
    num_det_channels = 128
    view_step_size = None  # Set to a small, positive integer to subsample the views

    view_indices = np.arange(start=0, stop=num_views, step=view_step_size)

    source_detector_distance = 4 * num_det_channels
    source_iso_distance = source_detector_distance
    delta_voxel = 1
    detector_cone_angle = 2 * np.arctan2(num_det_channels / 2, source_detector_distance)
    start_angle = -(np.pi + detector_cone_angle) * (1/2)
    end_angle = (np.pi + detector_cone_angle) * (1/2)

    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    angles = jnp.linspace(start_angle, jnp.pi, num_views, endpoint=False)

    # Initialize a random key
    seed_value = 0  #np.random.randint(1000000)
    key = jax.random.PRNGKey(seed_value)

    # Set up parallel beam model
    # conebeam_model = mbirjax.ConeBeamModel.from_file('params_conebeam.yaml')
    conebeam_model = mbirjax.ConeBeamModel(sinogram_shape, angles, source_detector_distance, source_iso_distance)
    # conebeam_model.to_file('params_conebeam.yaml')

    # conebeam_model.set_params(delta_voxel=delta_voxel)

    # Generate phantom
    recon_shape = conebeam_model.get_params('recon_shape')
    num_recon_rows, num_recon_cols, num_recon_slices = recon_shape[:3]
    phantom = mbirjax.gen_cube_phantom(recon_shape)

    # Generate indices of pixels
    num_subsets = 1
    full_indices = mbirjax.gen_pixel_partition(recon_shape, num_subsets)
    num_subsets = 5
    subset_indices = mbirjax.gen_pixel_partition(recon_shape, num_subsets)

    # Generate sinogram data
    # mbirjax.slice_viewer(phantom)
    voxel_values = phantom.reshape((-1,) + recon_shape[2:])[full_indices]

    conebeam_model.set_params(view_batch_size=view_batch_size, pixel_batch_size=pixel_batch_size)

    num_projected_views = len(view_indices) if len(view_indices) != 0 else num_views
    print('Starting forward projection with {} views out of {} total'.format(num_projected_views, num_views))
    view_subset_sinogram = conebeam_model.sparse_forward_project(voxel_values[0], full_indices[0], view_indices=view_indices)
    # mbirjax.slice_viewer(view_subset_sinogram, slice_axis=0, slice_label='View')

    # Determine resulting number of views, slices, and channels and image size
    print('View subset sinogram shape: {}'.format(view_subset_sinogram.shape))
    print('Memory stats after forward projection')
    mbirjax.get_memory_stats(print_results=True)

    # Get the vector of indices
    indices = jnp.arange(num_recon_rows * num_recon_cols)
    num_trials = 3
    indices = jnp.mod(np.arange(num_trials, dtype=int).reshape((-1, 1)) + indices.reshape((1, -1)), num_recon_rows * num_recon_cols)

    # Run once to finish compiling
    print('Starting back projection')
    bp = conebeam_model.sparse_back_project(view_subset_sinogram, indices[0], view_indices=view_indices)
    print('Recon shape: ({}, {}, {})'.format(num_recon_rows, num_recon_cols, num_recon_slices))
    print('Memory stats after back projection')
    mbirjax.get_memory_stats(print_results=True)
    # ##########################
    # Test the adjoint property
    # Get a random 3D phantom to test the adjoint property
    key, subkey = jax.random.split(key)
    x = jax.random.uniform(subkey, shape=bp.shape)
    key, subkey = jax.random.split(key)
    y = jax.random.uniform(subkey, shape=view_subset_sinogram.shape)

    # Do a forward projection, then a backprojection
    voxel_values = x.reshape((-1, num_recon_slices))[indices[0]]
    Ax = conebeam_model.sparse_forward_project(voxel_values, indices[0], view_indices=view_indices)
    Aty = conebeam_model.sparse_back_project(y, indices[0], view_indices=view_indices)

    # Calculate <Aty, x> and <y, Ax>
    Aty_x = jnp.sum(Aty * x)
    y_Ax = jnp.sum(y * Ax)

    adjoint_result = np.allclose(Aty_x, y_Ax)
    if adjoint_result:
        print("Adjoint property holds for random x, y <y, Ax> = <Aty, x>: {}".format(adjoint_result))
    else:
        warnings.warn('Adjoint property does not hold.')

    # Clean up before further projections
    del Ax, Aty, bp
    del phantom
    del x, y
    gc.collect()

    # ##########################
    ## Test the hessian against a finite difference approximation ## #
    hessian = conebeam_model.compute_hessian_diagonal(view_indices=view_indices)

    x = jnp.zeros(recon_shape)
    key, subkey = jax.random.split(key)
    i, j = jax.random.randint(subkey, shape=(2,), minval=0, maxval=num_recon_rows)
    key, subkey = jax.random.split(key)
    k = jax.random.randint(subkey, shape=(), minval=0, maxval=num_recon_slices)

    eps = 0.01
    x = x.at[i, j, k].set(eps)
    voxel_values = x.reshape((-1, num_recon_slices))[indices[0]]
    Ax = conebeam_model.sparse_forward_project(voxel_values, indices[0], view_indices=view_indices)
    AtAx = conebeam_model.sparse_back_project(Ax, indices[0], view_indices=view_indices).reshape(x.shape)
    finite_diff_hessian = AtAx[i, j, k] / eps
    hessian_result = jnp.allclose(hessian.reshape(x.shape)[i, j, k], finite_diff_hessian)
    if hessian_result:
        print('Hessian matches finite difference: {}'.format(hessian_result))
    else:
        warnings.warn('Hessian does not match finite difference.')

    # ##########################
    # Check the time taken per forward projection
    #  NOTE: recompiling happens whenever sparse_forward_project is called with a new *length* of input indices
    time_taken = 0

    print('\nStarting multiple forward projections...')
    for j in range(num_trials):
        voxel_values = x.reshape((-1, num_recon_slices))[indices[j]]
        t0 = time.time()
        fp = conebeam_model.sparse_forward_project(voxel_values, indices[j], view_indices=view_indices)
        time_taken += time.time() - t0
        del fp
        gc.collect()

    print('Mean time per call = {}'.format(time_taken / num_trials))
    print('Done')

    # ##########################
    # Check the time taken per backprojection
    #  NOTE: recompiling happens whenever sparse_back_project is called with a new *length* of input indices
    time_taken = 0

    print('\nStarting multiple backprojections...')
    for j in range(num_trials):
        t0 = time.time()
        bp = conebeam_model.sparse_back_project(view_subset_sinogram, indices[j], view_indices=view_indices)
        time_taken += time.time() - t0
        del bp
        gc.collect()

    print('Mean time per call = {}'.format(time_taken / num_trials))
    print('Done')

    print('Memory stats after all multiple projections')
    mbirjax.get_memory_stats(print_results=True)

    # ##########################
    # Show the forward and back projection from a single pixel
    i, j = num_recon_rows // 4, num_recon_cols // 3
    x = jnp.zeros(recon_shape)
    x = x.at[i, j, :].set(1)
    voxel_values = x.reshape((-1, num_recon_slices))[indices[0]]

    Ax = conebeam_model.sparse_forward_project(voxel_values, indices[0], view_indices=view_indices)
    Aty = conebeam_model.sparse_back_project(Ax, indices[0], view_indices=view_indices)
    Aty = conebeam_model.reshape_recon(Aty)

    slice_index = (num_recon_slices + 1) // 2
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    cax = ax[0].imshow(x[:, :, slice_index])
    ax[0].set_title('x = phantom')
    fig.colorbar(cax, ax=ax[0])
    cax = ax[1].imshow(Ax[:, slice_index, :])
    ax[1].set_title('y = Ax')
    fig.colorbar(cax, ax=ax[1])
    cax = ax[2].imshow(Aty[:, :, slice_index])
    ax[2].set_title('Aty = AtAx')
    fig.colorbar(cax, ax=ax[2])
    plt.pause(2)

    print('Final memory stats:')
    mbirjax.get_memory_stats(print_results=True)
    input('Press return to exit')
    a = 0
