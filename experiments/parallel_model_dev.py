import os
import numpy as np
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt
import mbirjax


if __name__ == "__main__":
    """
    This is a script to develop, debug, and tune the parallel beam projector
    """
    # ##########################
    # Do all the setup

    # Initialize sinogram
    num_views = 256
    num_det_rows = 5
    num_det_channels = 256
    start_angle = 0
    end_angle = np.pi
    sinogram = jnp.zeros((num_views, num_det_rows, num_det_channels))
    angles = jnp.linspace(start_angle, np.pi, num_views, endpoint=False)

    # Set up parallel beam model
    parallel_model = mbirjax.ParallelBeamModel(angles, sinogram.shape)

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
    cos_sin_angles = parallel_model.get_cos_sin_angles(angles)
    geometry_params = parallel_model.get_geometry_parameters()

    # view = parallel_model.forward_project_voxels_one_view(voxel_values[0], full_indices[0], cos_sin_angles[0], geometry_params, sinogram.shape)
    # in_axes=(None, None, 0, None, None))

    sinogram = parallel_model.forward_project(voxel_values[0], full_indices[0])

    # Determine resulting number of views, slices, and channels and image size
    num_recon_rows, num_recon_cols, num_recon_slices = (
        parallel_model.get_params(['num_recon_rows', 'num_recon_cols', 'num_recon_slices']))
    print('Sinogram shape: {}'.format(sinogram.shape))
    mbirjax.get_gpu_memory_stats(print_results=True)

    # Get the vector of indices
    indices = np.arange(num_recon_rows * num_recon_cols)
    num_trials = 10
    indices = np.mod(np.arange(num_trials, dtype=int).reshape((-1, 1)) + indices.reshape((1, -1)), num_recon_rows * num_recon_cols)

    sinogram = jnp.array(sinogram)
    indices = jnp.array(indices)

    # Run once to finish compiling
    bp = parallel_model.back_project(sinogram, indices[0])  # Using this form takes about 60% the time of the next form for a 128x128 image.
    print('Recon shape: ({}, {}, {})'.format(num_recon_rows, num_recon_cols, num_recon_slices))
    mbirjax.get_gpu_memory_stats(print_results=True)
    # ##########################
    # Test the adjoint property
    # Get a random 3D phantom to test the adjoint property
    x = np.random.random_sample(bp.shape)
    x = jnp.array(x)
    y = np.random.random_sample(sinogram.shape)

    # Do a forward projection, then a backprojection
    voxel_values = x.reshape((-1, num_recon_slices))[indices[0]]
    Ax = parallel_model.forward_project(voxel_values, indices[0])
    Aty = parallel_model.back_project(y, indices[0])

    # Calculate <Aty, x> and <y, Ax>
    Aty_x = np.sum(Aty * x)
    y_Ax = np.sum(y * Ax)

    print("Adjoint property holds for random x, y <y, Ax> = <Aty, x>: {}".format(np.allclose(Aty_x, y_Ax)))

    # ##########################
    # ## Test the hessian against a finite difference approximation ## #
    hessian = parallel_model.compute_hessian_diagonal(None, angles, sinogram_shape=sinogram.shape)

    x = np.zeros((num_recon_rows, num_recon_cols, num_recon_slices))
    i, j = np.random.randint(num_recon_rows, size=2)
    k = np.random.randint(num_recon_slices)
    eps = 0.01
    x[i, j, k] = eps
    voxel_values = x.reshape((-1, num_recon_slices))[indices[0]]
    Ax = parallel_model.forward_project(voxel_values, indices[0])
    AtAx = parallel_model.back_project(Ax, indices[0]).reshape(x.shape)
    finite_diff_hessian = AtAx[i, j, k] / eps
    print('Hessian matches finite difference: {}'.format(np.allclose(hessian.reshape(x.shape)[i, j, k], finite_diff_hessian)))

    # ##########################
    # Check the time taken per forward projection
    #  NOTE: recompiling happens whenever forward_project is called with a new *length* of input indices
    time_taken = 0

    print('\nStarting multiple forward projections...')
    for j in range(num_trials):
        voxel_values = x.reshape((-1, num_recon_slices))[indices[j]]
        t0 = time.time()
        # Using the precomipled forward projector is about 5x faster than using linear_transpose.
        # fp = parallel_model.forward_project_indices(x, indices[j], angles, sinogram.shape)
        fp = parallel_model.forward_project(voxel_values, indices[j])
        time_taken += time.time() - t0

    print('Mean time per call = {}'.format(time_taken / num_trials))
    print('Done')

    # ##########################
    # Check the time taken per backprojection
    #  NOTE: recompiling happens whenever back_project is called with a new *length* of input indices
    time_taken = 0

    print('\nStarting multiple backprojections...')
    for j in range(num_trials):
        t0 = time.time()
        # Using the precompiled back projector is about 25% faster than calling back_project_indices directly.
        bp = parallel_model.back_project(sinogram, indices[j])
        # bp = parallel_model.back_project_indices(sinogram, indices[j], angles)
        time_taken += time.time() - t0

    print('Mean time per call = {}'.format(time_taken / num_trials))
    print('Done')

    # ##########################
    # Show the forward and back projection from a single pixel
    i, j = num_recon_rows // 4, num_recon_cols // 3
    index = np.ravel_multi_index((i, j), (num_recon_rows, num_recon_cols))
    x = np.zeros((num_recon_rows, num_recon_cols)).flatten()
    x[index] = 1
    voxel_values = x.reshape((-1, 1))[indices[0]]
    # x = jnp.array(x)

    Ax = parallel_model.forward_project(voxel_values, indices[0])
    Aty = parallel_model.back_project(Ax, indices[0])

    y = jnp.zeros_like(sinogram)
    view_index = 30
    y = y.at[view_index].set(sinogram[view_index])
    index = np.ravel_multi_index((60, 60), (num_recon_rows, num_recon_cols))
    a1 = parallel_model.back_project(y, indices[0])

    cs = parallel_model.get_cos_sin_angles(angles[view_index])
    a2 = parallel_model.back_project_one_view_to_voxels(sinogram[view_index], indices[0], cs, geometry_params)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    ax[0].imshow(x.reshape((num_recon_rows, num_recon_cols)))
    ax[0].set_title('x = phantom')
    ax[1].imshow(Ax[:, 0, :])
    ax[1].set_title('y = Ax')
    ax[2].imshow(Aty.reshape((num_recon_rows, num_recon_cols)))
    ax[2].set_title('Aty = AtAx')
    plt.pause(0.5)

    input('Press return to exit')
    a = 0
