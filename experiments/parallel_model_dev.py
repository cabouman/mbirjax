import os
import numpy as np
import jax.numpy as jnp
import jax
import time
import matplotlib.pyplot as plt
import gc
import mbirjax.parallel_beam

if __name__ == "__main__":
    """
    This is a script to develop, debug, and tune the parallel beam projector
    """
    # ##########################
    # Do all the setup
    view_batch_size = 100
    voxel_batch_size = 10000

    # Initialize sinogram
    num_views = 128
    num_det_rows = 128
    num_det_channels = 128
    start_angle = 0
    end_angle = jnp.pi
    sinogram = jnp.zeros((num_views, num_det_rows, num_det_channels))
    angles = jnp.linspace(start_angle, jnp.pi, num_views, endpoint=False)

    # Initialize a random key
    seed_value = np.random.randint(1000000)
    key = jax.random.PRNGKey(seed_value)

    # Set up parallel beam model
    parallel_model = mbirjax.parallel_beam.ParallelBeamModel(angles, sinogram.shape)

    # Generate phantom
    num_recon_rows, num_recon_cols, num_recon_slices = (
        parallel_model.get_params(['num_recon_rows', 'num_recon_cols', 'num_recon_slices']))
    phantom = mbirjax.gen_phantom(num_recon_rows, num_recon_cols, num_recon_slices)

    # Generate indices of voxels
    num_subsets = 1
    full_indices = mbirjax.gen_voxel_partition(num_recon_rows, num_recon_cols, num_subsets)
    num_subsets = 5
    subset_indices = mbirjax.gen_voxel_partition(num_recon_rows, num_recon_cols, num_subsets)

    # Generate sinogram data
    voxel_values = phantom.reshape((-1, num_recon_slices))[full_indices]
    geometry_params = parallel_model.get_geometry_parameters()
    parallel_model.set_params(view_batch_size=view_batch_size, voxel_batch_size=voxel_batch_size)

    print('Starting forward projection')
    sinogram = parallel_model.sparse_forward_project(voxel_values[0], full_indices[0])

    # Determine resulting number of views, slices, and channels and image size
    num_recon_rows, num_recon_cols, num_recon_slices = (
        parallel_model.get_params(['num_recon_rows', 'num_recon_cols', 'num_recon_slices']))
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
    bp = parallel_model.sparse_back_project(sinogram, indices[0])
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
    Ax = parallel_model.sparse_forward_project(voxel_values, indices[0])
    Aty = parallel_model.sparse_back_project(y, indices[0])

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
    hessian = parallel_model.compute_hessian_diagonal()

    x = jnp.zeros((num_recon_rows, num_recon_cols, num_recon_slices))
    key, subkey = jax.random.split(key)
    i, j = jax.random.randint(subkey, shape=(2,), minval=0, maxval=num_recon_rows)
    key, subkey = jax.random.split(key)
    k = jax.random.randint(subkey, shape=(), minval=0, maxval=num_recon_slices)

    eps = 0.01
    x = x.at[i, j, k].set(eps)
    voxel_values = x.reshape((-1, num_recon_slices))[indices[0]]
    Ax = parallel_model.sparse_forward_project(voxel_values, indices[0])
    AtAx = parallel_model.sparse_back_project(Ax, indices[0]).reshape(x.shape)
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
        fp = parallel_model.sparse_forward_project(voxel_values, indices[j])
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
        bp = parallel_model.sparse_back_project(sinogram, indices[j])
        time_taken += time.time() - t0
        del bp
        gc.collect()

    print('Mean time per call = {}'.format(time_taken / num_trials))
    print('Done')

    # ##########################
    # Show the forward and back projection from a single pixel
    i, j = num_recon_rows // 4, num_recon_cols // 3
    x = jnp.zeros((num_recon_rows, num_recon_cols, num_recon_slices))
    x = x.at[i, j, :].set(1)
    voxel_values = x.reshape((-1, num_recon_slices))[indices[0]]

    Ax = parallel_model.sparse_forward_project(voxel_values, indices[0])
    Aty = parallel_model.sparse_back_project(Ax, indices[0])
    Aty = parallel_model.reshape_recon(Aty)

    y = jnp.zeros_like(sinogram)
    view_index = 30
    y = y.at[view_index].set(sinogram[view_index])
    index = jnp.ravel_multi_index((60, 60), (num_recon_rows, num_recon_cols))
    a1 = parallel_model.sparse_back_project(y, indices[0])

    slice_index = 0
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    ax[0].imshow(x[:, :, slice_index])
    ax[0].set_title('x = phantom')
    ax[1].imshow(Ax[:, slice_index, :])
    ax[1].set_title('y = Ax')
    ax[2].imshow(Aty[:, :, slice_index])
    ax[2].set_title('Aty = AtAx')
    plt.pause(2)

    print('Final memory stats:')
    mbirjax.get_gpu_memory_stats(print_results=True)
    input('Press return to exit')
    a = 0
