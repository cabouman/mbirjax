import numpy as np
import jax.numpy as jnp
import mbirjax.parallel_beam
import vcd_coimg_utils as vcu


if __name__ == "__main__":
    """
    Create figures for COIMG 2025
    """

    # Set parameters
    num_views = vcu.param_dict['num_views']
    num_det_rows = vcu.param_dict['num_det_rows']
    num_det_channels = vcu.param_dict['num_det_channels']
    start_angle = vcu.param_dict['start_angle']
    end_angle = vcu.param_dict['end_angle']
    granularity = vcu.param_dict['granularity']
    partition_sequence = vcu.param_dict['partition_sequence']
    max_iterations = vcu.param_dict['max_iterations']

    # Initialize sinogram
    sinogram = jnp.zeros((num_views, num_det_rows, num_det_channels))
    angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

    # Set up the model
    ct_model = mbirjax.ParallelBeamModel(sinogram.shape, angles)

    # Generate 3D Shepp Logan phantom
    phantom = ct_model.gen_modified_3d_sl_phantom()

    # Generate synthetic sinogram data
    sinogram = ct_model.forward_project(phantom)

    # Print out model parameters
    ct_model.set_params(granularity=granularity, partition_sequence=partition_sequence)
    ct_model.print_params()

    # Set comparison sequences
    granularity_alt_1 = granularity
    partition_sequence_alt_1 = [0, ]  # Gradient descent

    granularity_alt_2 = granularity
    partition_sequence_alt_2 = [len(granularity) - 1]  # Essentially ICD

    # ##########################
    # Perform default VCD reconstruction
    print('Starting default sequence')
    recon_default, recon_params_default = ct_model.recon(sinogram, max_iterations=max_iterations,
                                                         compute_prior_loss=True)
    fm_rmse_default = recon_params_default.fm_rmse
    prior_loss_default = recon_params_default.prior_loss
    partition_sequence = recon_params_default.partition_sequence
    granularity = np.array(recon_params_default.granularity)
    granularity_sequence_default = granularity[partition_sequence]
    label_default = 'VCD'  # 'Base: ' + str(granularity_sequence_default)

    # Perform alt_1 default reconstruction
    print('Starting alt_1 sequence')
    ct_model.set_params(partition_sequence=partition_sequence_alt_1)
    granularity = np.array(granularity_alt_1)
    ct_model.set_params(granularity=granularity)
    recon_alt_1, recon_params_alt_1 = ct_model.recon(sinogram, max_iterations=max_iterations,
                                                     compute_prior_loss=True)
    fm_rmse_alt_1 = recon_params_alt_1.fm_rmse
    prior_loss_alt_1 = recon_params_alt_1.prior_loss
    partition_sequence = recon_params_alt_1.partition_sequence
    granularity = np.array(recon_params_alt_1.granularity)
    granularity_sequence_alt_1 = granularity[partition_sequence]
    label_alt_1 = 'Gradient descent'  # 'alt_1: ' + str(granularity_sequence_alt_1)

    # Perform alt_2 reconstruction
    print('Starting alt_2 sequence')
    ct_model.set_params(partition_sequence=partition_sequence_alt_2)
    granularity = np.array(granularity_alt_2)
    ct_model.set_params(granularity=granularity)
    recon_alt_2, recon_params_alt_2 = ct_model.recon(sinogram, max_iterations=max_iterations,
                                                     compute_prior_loss=True)
    fm_rmse_alt_2 = recon_params_alt_2.fm_rmse
    prior_loss_alt_2 = recon_params_alt_2.prior_loss
    partition_sequence = recon_params_alt_2.partition_sequence
    granularity = np.array(recon_params_alt_2.granularity)
    granularity_sequence_alt_2 = granularity[partition_sequence]
    label_alt_2 = 'ICD'  # 'alt_2: ' + str(granularity_sequence_alt_2)
    # ##########################

    # Display reconstructions
    fig_title = 'Recons with {} iterations'.format(max_iterations)
    labels = [label_alt_1, label_default, label_alt_2]
    slice_index = recon_default.shape[-1] // 2
    images = [recon[:, :, slice_index] for recon in [recon_alt_1, recon_default, recon_alt_2]]
    vcu.display_images_for_abstract(images[0], images[1], images[2], labels, fig_title=fig_title, vmax=phantom.max())

    # Display granularity plots:
    granularity_sequences = [granularity_sequence_alt_1, granularity_sequence_default, granularity_sequence_alt_2]
    fm_losses = [fm_rmse_alt_1, fm_rmse_default, fm_rmse_alt_2]
    prior_losses = [prior_loss_alt_1, prior_loss_default, prior_loss_alt_2]
    # labels = ['Gradient Descent', 'Vectorized Coordinate Descent', 'Coordinate Descent']
    mbirjax.plot_granularity_and_loss(granularity_sequences, fm_losses, prior_losses, labels, granularity_ylim=(0, 400),
                                      loss_ylim=(0.01, 20), fig_title=fig_title)

    # Generate sequence of partition images for Figure 1
    recon_shape = (32, 32, 1)
    partitions_fig = mbirjax.gen_set_of_pixel_partitions(recon_shape=recon_shape, granularity=[1, 4, 16, 64, 256])

    # Plot the set of partitions
    mbirjax.debug_plot_partitions(partitions=partitions_fig, recon_shape=recon_shape)
