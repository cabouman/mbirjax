
import numpy as np
import jax.numpy as jnp
import jax
import time
import matplotlib.pyplot as plt
import gc
import mbirjax

if __name__ == "__main__":
    """
    This is a script to develop, debug, and tune the cone beam beam projector
    """
    # ##########################
    # Do all the setup
    view_batch_size = 1024
    pixel_batch_size = 10000

    # Initialize sinogram
    num_views = 128
    num_det_rows = 20
    num_det_channels = 128
    source_detector_distance = 1000
    source_iso_distance = 500
    delta_voxel = 1
    start_angle = 0
    extra_angle = 0  # jnp.atan2(magnification * num_det_channels / 2, source_detector_distance)
    end_angle = jnp.pi + extra_angle
    sinogram = jnp.zeros((num_views, num_det_rows, num_det_channels))
    angles = jnp.linspace(start_angle, jnp.pi, num_views, endpoint=False)

    # Initialize a random key
    seed_value = np.random.randint(1000000)
    key = jax.random.PRNGKey(seed_value)

    # Set up parallel beam model
    conebeam_model = mbirjax.ConeBeamModel(sinogram.shape, angles, source_detector_distance, source_iso_distance)
    conebeam_model.set_params(delta_voxel=delta_voxel)

    # Generate phantom
    recon_shape = conebeam_model.get_params('recon_shape')
    num_recon_rows, num_recon_cols, num_recon_slices = recon_shape[:3]
    phantom = mbirjax.gen_cube_phantom(recon_shape)

    # Generate indices of pixels
    num_subsets = 1
    full_indices = mbirjax.gen_pixel_partition(recon_shape, num_subsets)
    num_subsets = 5
    subset_indices = mbirjax.gen_pixel_partition(recon_shape, num_subsets)

    ###
    # # Test vertical fan beam
    #
    # geometry_params = conebeam_model.get_geometry_parameters()
    # sinogram_shape, recon_shape = conebeam_model.get_params(['sinogram_shape', 'recon_shape'])
    # projector_params = (tuple(sinogram_shape), tuple(recon_shape), tuple(geometry_params))
    # angle = angles[10]
    # pixel_indices = full_indices[0][12000:12001]
    # voxel_values = np.random.rand(1, recon_shape[2])
    #
    # projection1 = conebeam_model.forward_project_pixel_batch_to_one_view(voxel_values, pixel_indices, angle, projector_params)
    # center = np.where(np.amax(projection1, axis=0) > 0)[0][1]
    # projection1 = projection1[:, center]
    # voxel_values = voxel_values.reshape(-1)
    # projection2 = conebeam_model.forward_project_vertical_fan_beam_one_pixel_one_view(voxel_values, pixel_indices, angle, projector_params)
    # A = np.amax(np.abs(projection2 / projection1 - projection2[0] / projection1[0]))
    # print(A)
    ##

    # ##########################
    # Show the forward and back projection from a single pixel
    i, j = num_recon_rows // 4, num_recon_cols // 3
    x = jnp.zeros(recon_shape)
    x = x.at[i, j, :].set(1)
    voxel_values = x.reshape((-1, num_recon_slices))[full_indices[0]]

    Ax = conebeam_model.sparse_forward_project(voxel_values, full_indices[0])
    Ax = np.array(Ax)
    Aty = conebeam_model.sparse_back_project(Ax, full_indices[0])
    Aty = np.array(Aty)

    # Generate sinogram data
    voxel_values = phantom.reshape((-1,) + recon_shape[2:])[full_indices]

    conebeam_model.set_params(view_batch_size=view_batch_size, pixel_batch_size=pixel_batch_size)

    print('Starting forward projection')
    sinogram = conebeam_model.sparse_forward_project(voxel_values[0][23:27], full_indices[0][23:27])

    # Determine resulting number of views, slices, and channels and image size
    print('Sinogram shape: {}'.format(sinogram.shape))
    print('Memory stats after forward projection')
    mbirjax.get_gpu_memory_stats(print_results=True)

    # Get the vector of indices
    indices = jnp.arange(num_recon_rows * num_recon_cols)
    num_trials = 3
    indices = jnp.mod(np.arange(num_trials, dtype=int).reshape((-1, 1)) + indices.reshape((1, -1)), num_recon_rows * num_recon_cols)

    sinogram = jnp.array(sinogram)
    indices = jnp.array(indices)

    # Run once to finish compiling
    print('Starting back projection')
    bp = conebeam_model.sparse_back_project(sinogram, indices[0])
    print('Recon shape: ({}, {}, {})'.format(num_recon_rows, num_recon_cols, num_recon_slices))
    print('Memory stats after back projection')
    mbirjax.get_gpu_memory_stats(print_results=True)
    # ##########################
    # Test the adjoint property
    # Get a random 3D phantom to test the adjoint property
    key, subkey = jax.random.split(key)
    x = jax.random.uniform(subkey, shape=bp.shape)
    key, subkey = jax.random.split(key)
    y = jax.random.uniform(subkey, shape=sinogram.shape)

    # Do a forward projection, then a backprojection
    voxel_values = x.reshape((-1, num_recon_slices))[indices[0]]
    Ax = conebeam_model.sparse_forward_project(voxel_values, indices[0])
    Aty = conebeam_model.sparse_back_project(y, indices[0])

    # Calculate <Aty, x> and <y, Ax>
    Aty_x = jnp.sum(Aty * x)
    y_Ax = jnp.sum(y * Ax)

    print("Adjoint property holds for random x, y <y, Ax> = <Aty, x>: {}".format(np.allclose(Aty_x, y_Ax)))

    # Clean up before further projections
    del Ax, Aty, bp
    del phantom
    del x, y
    gc.collect()

    # ##########################
    # ## Test the hessian against a finite difference approximation ## #
    hessian = conebeam_model.compute_hessian_diagonal()

    x = jnp.zeros(recon_shape)
    key, subkey = jax.random.split(key)
    i, j = jax.random.randint(subkey, shape=(2,), minval=0, maxval=num_recon_rows)
    key, subkey = jax.random.split(key)
    k = jax.random.randint(subkey, shape=(), minval=0, maxval=num_recon_slices)

    eps = 0.01
    x = x.at[i, j, k].set(eps)
    voxel_values = x.reshape((-1, num_recon_slices))[indices[0]]
    Ax = conebeam_model.sparse_forward_project(voxel_values, indices[0])
    AtAx = conebeam_model.sparse_back_project(Ax, indices[0]).reshape(x.shape)
    finite_diff_hessian = AtAx[i, j, k] / eps
    print('Hessian matches finite difference: {}'.format(jnp.allclose(hessian.reshape(x.shape)[i, j, k], finite_diff_hessian)))

    # ##########################
    # Check the time taken per forward projection
    #  NOTE: recompiling happens whenever sparse_forward_project is called with a new *length* of input indices
    time_taken = 0

    print('\nStarting multiple forward projections...')
    for j in range(num_trials):
        voxel_values = x.reshape((-1, num_recon_slices))[indices[j]]
        t0 = time.time()
        fp = conebeam_model.sparse_forward_project(voxel_values, indices[j])
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
        bp = conebeam_model.sparse_back_project(sinogram, indices[j])
        time_taken += time.time() - t0
        del bp
        gc.collect()

    print('Mean time per call = {}'.format(time_taken / num_trials))
    print('Done')

    # ##########################
    # Show the forward and back projection from a single pixel
    i, j = num_recon_rows // 4, num_recon_cols // 3
    x = jnp.zeros(recon_shape)
    x = x.at[i, j, :].set(1)
    voxel_values = x.reshape((-1, num_recon_slices))[indices[0]]

    Ax = conebeam_model.sparse_forward_project(voxel_values, indices[0])
    Aty = conebeam_model.sparse_back_project(Ax, indices[0])
    Aty = conebeam_model.reshape_recon(Aty)

    y = jnp.zeros_like(sinogram)
    view_index = 30
    y = y.at[view_index].set(sinogram[view_index])
    index = jnp.ravel_multi_index((6, 6), (num_recon_rows, num_recon_cols))
    a1 = conebeam_model.sparse_back_project(y, indices[0])

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
    mbirjax.get_gpu_memory_stats(print_results=True)
    input('Press return to exit')
    a = 0
