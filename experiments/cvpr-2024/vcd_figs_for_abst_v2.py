import numpy as np
import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mbirjax
import mbirjax.parallel_beam


def display_slices_for_abstract( recon1, recon2, recon3, labels) :
    # Set global font size
    plt.rcParams.update({'font.size': 15})  # Adjust font size here

    vmin = 0.0
    vmax = phantom.max()
    slice_index = recon1.shape[2] // 2

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    a0 = ax[0].imshow(recon1[:, :, slice_index], vmin=vmin, vmax=vmax, cmap='gray')
    #plt.colorbar(a0, ax=ax[0])
    ax[0].set_title(labels[0])

    a0 = ax[1].imshow(recon2[:, :, slice_index], vmin=vmin, vmax=vmax, cmap='gray')
    #plt.colorbar(a0, ax=ax[1])
    ax[1].set_title(labels[1])

    a2 = ax[2].imshow(recon3[:, :, slice_index], vmin=vmin, vmax=vmax, cmap='gray')
    #plt.colorbar(a2, ax=ax[2])
    ax[2].set_title(labels[2])

    plt.show()


if __name__ == "__main__":
    """
    This is a script to develop, debug, and tune the vcd reconstruction with a parallel beam projector
    """
    # Set parameters
    num_views = 256
    num_det_rows = 10
    num_det_channels = 256
    start_angle = 0
    end_angle = np.pi
    sharpness = 0.0

    # Initialize sinogram
    sinogram = jnp.zeros((num_views, num_det_rows, num_det_channels))
    angles = jnp.linspace(start_angle, np.pi, num_views, endpoint=False)

    # Set up parallel beam model
    parallel_model = mbirjax.ParallelBeamModel(sinogram.shape, angles)

    # Generate 3D Shepp Logan phantom
    phantom = parallel_model.gen_modified_3d_sl_phantom()

    # Generate synthetic sinogram data
    sinogram = parallel_model.forward_project(phantom)

    # Generate weights array
    weights = parallel_model.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')

    # Set reconstruction parameter values
    parallel_model.set_params(sharpness=sharpness, verbose=1)

    # Print out model parameters
    parallel_model.print_params()

    # 'granularity': {'val': [1, 4, 64, 128], 'recompile_flag': False},
    # 'partition_sequence': {'val': [0, 1, 2, 2, 2], 'recompile_flag': False},

    granularity_previous = [1, 8, 64, 256]
    partition_sequence_previous = [0, 1, 2, 3, 1, 2, 3, 2, 3, 3, 0, 1, 2, 3, 1, 2, 3, 2, 3, 3]

    granularity_proposed = [4, 4, 64, 128]
    partition_sequence_proposed = [0, 1, 2, 3]

    # ##########################
    # Perform default VCD reconstruction
    num_iterations = 8
    recon_new_default, recon_params_new_default = parallel_model.recon(sinogram, weights=weights, num_iterations=num_iterations,
                                                       compute_prior_loss=True)
    fm_rmse_new_default = recon_params_new_default.fm_rmse
    prior_loss_new_default = recon_params_new_default.prior_loss
    partition_sequence = recon_params_new_default.partition_sequence
    granularity = np.array(recon_params_new_default.granularity)
    granularity_sequence_new_default = granularity[partition_sequence]

    # Perform previous default reconstruction    #
    parallel_model.set_params(partition_sequence=partition_sequence_previous)
    granularity = np.array(granularity_previous)
    parallel_model.set_params(granularity=granularity)
    recon_prev_default, recon_params_prev_default = parallel_model.recon(sinogram, weights=weights, num_iterations=num_iterations,
                                                       compute_prior_loss=True)
    fm_rmse_prev_default = recon_params_prev_default.fm_rmse
    prior_loss_prev_default = recon_params_prev_default.prior_loss
    partition_sequence = recon_params_prev_default.partition_sequence
    granularity = np.array(recon_params_prev_default.granularity)
    granularity_sequence_prev_default = granularity[partition_sequence]

    # Perform proposed reconstruction
    parallel_model.set_params(partition_sequence=partition_sequence_proposed)
    granularity = np.array(granularity_proposed)
    parallel_model.set_params(granularity=granularity)
    recon_proposed, recon_params_proposed = parallel_model.recon(sinogram, weights=weights, num_iterations=num_iterations,
                                                       compute_prior_loss=True)
    fm_rmse_proposed = recon_params_proposed.fm_rmse
    prior_loss_proposed = recon_params_proposed.prior_loss
    partition_sequence = recon_params_proposed.partition_sequence
    granularity = np.array(recon_params_proposed.granularity)
    granularity_sequence_proposed = granularity[partition_sequence]
    # ##########################

    # Display reconstructions
    labels = ['Previous default', 'New default', 'Extra granularity']
    # display_slices_for_abstract(recon_prev_default, recon_new_default, recon_proposed, labels)

    # Display granularity plots:
    granularity_sequences = [granularity_sequence_prev_default, granularity_sequence_new_default, granularity_sequence_proposed]
    fm_losses = [fm_rmse_prev_default, fm_rmse_new_default, fm_rmse_proposed]
    prior_losses = [prior_loss_prev_default, prior_loss_new_default, prior_loss_proposed]
    # labels = ['Gradient Descent', 'Vectorized Coordinate Descent', 'Coordinate Descent']
    mbirjax.plot_granularity_and_loss(granularity_sequences, fm_losses, prior_losses, labels, granularity_ylim=(0, 256), loss_ylim=(0.1, 15))

    # Generate sequence of partition images for Figure 1
    recon_shape = (32, 32, 1)
    partitions_fig = mbirjax.gen_set_of_pixel_partitions(recon_shape=recon_shape, granularity=[1, 4, 16, 64, 256])

    # Plot the set of partitions
    mbirjax.debug_plot_partitions(partitions=partitions_fig, recon_shape=recon_shape)
