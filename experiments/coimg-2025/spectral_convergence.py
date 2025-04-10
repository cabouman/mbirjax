import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import mbirjax
import vcd_coimg_utils as vcu

if __name__ == "__main__":
    """
    Investigate convergence in the Fourier domain
    """
    # Set parameters
    num_views = vcu.param_dict['num_views']
    num_det_channels = vcu.param_dict['num_det_channels']
    start_angle = vcu.param_dict['start_angle']
    end_angle = vcu.param_dict['end_angle']
    granularity = vcu.param_dict['granularity']
    partition_sequence = vcu.param_dict['partition_sequence']
    max_iterations = vcu.param_dict['max_iterations']
    max_iterations = 10

    num_det_rows = 1  # We use only one detector row for efficiency
    num_phantom_slices = 10  # We use 10 phantom slices to get a non-trivial 3D phantom

    phantom_shape = (num_det_channels, num_det_channels, num_phantom_slices)

    # Initialize sinogram
    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    angles = jnp.linspace(start_angle, np.pi, num_views, endpoint=False)

    # Set up parallel beam model
    parallel_model = mbirjax.ParallelBeamModel(sinogram_shape, angles)

    # Generate 3D Shepp Logan phantom
    phantom = mbirjax.generate_3d_shepp_logan_low_dynamic_range(phantom_shape)
    center_slice = num_phantom_slices // 2
    phantom = phantom[:, :, center_slice:center_slice+1]
    phantom_fft = np.fft.fftn(phantom[:, :, 0])
    phantom_fft = np.fft.fftshift(phantom_fft)

    # Save the phantom and fft
    figure_folder_name = mbirjax.make_figure_folder()
    plt.imshow(phantom, cmap='grey')
    fig_title = 'phantom'
    plt.title(fig_title)
    plt.savefig(os.path.join(figure_folder_name, fig_title + '.png'), bbox_inches='tight')
    plt.show(block=True)
    plt.imshow(20 * np.log10(1e-6 + np.abs(phantom_fft)), cmap='viridis', vmin=0, vmax=60)
    plt.colorbar()
    fig_title = '|FFT(phantom)|'
    plt.title(fig_title)
    plt.savefig(os.path.join(figure_folder_name, fig_title + '.png'), bbox_inches='tight')
    plt.show(block=True)


    # Generate synthetic sinogram data
    print('Creating sinogram')
    sinogram = parallel_model.forward_project(phantom)

    # Set reconstruction parameter values
    parallel_model.set_params(verbose=1)
    parallel_model.set_params(granularity=granularity)

    vcd_partition_sequence = parallel_model.get_params('partition_sequence')
    gd_partition_sequence =  [0, ]
    icd_partition_sequence = [len(granularity) - 1, ]

    sequences = [vcd_partition_sequence, gd_partition_sequence, icd_partition_sequence]
    residual_fft = []
    residual_recon = []
    all_recons = []

    # ##########################
    # Perform reconstructions
    for j, sequence in enumerate(sequences):
        print('Starting reconstruction {}'.format(j+1))
        parallel_model.set_params(partition_sequence=sequence)
        cur_residual = np.zeros((max_iterations,) + phantom.shape[0:2])
        cur_recon = np.zeros((max_iterations,) + phantom.shape[0:2])
        recon = None
        for iteration in range(max_iterations):
            recon, cur_recon_params = parallel_model.recon(sinogram, max_iterations=iteration + 1,
                                                           first_iteration=iteration, init_recon=recon,
                                                           compute_prior_loss=True)
            cur_residual[iteration] = recon[:, :, 0] - phantom[:, :, 0]
            cur_recon[iteration] = recon[:, :, 0]

        all_recons.append(cur_recon)
        residual_recon.append(np.abs(cur_residual))
        cur_residual_fft = np.fft.fftn(cur_residual, axes=(1, 2))
        cur_residual_fft = np.fft.fftshift(cur_residual_fft, axes=(1, 2))
        residual_fft.append(20 * np.log10(1e-6 + np.abs(cur_residual_fft)))

    vcd_recons, gd_recons, icd_recons = all_recons
    vcd_residual, gd_residual, icd_residual = residual_recon
    vcd_residual_fft, gd_residual_fft, icd_residual_fft = residual_fft

    labels = ['Gradient descent', 'VCD', 'ICD']
    for cur_iter in np.arange(max_iterations):
        fig_title = 'Recon after {} iteration(s)'.format(cur_iter + 1)
        vcu.display_images_for_abstract(gd_recons[cur_iter], vcd_recons[cur_iter], icd_recons[cur_iter],
                                        labels=labels, fig_title=fig_title, show_colorbar=True)
        fig_title = '|Recon minus ground truth| after {} iteration(s)'.format(cur_iter + 1)
        vcu.display_images_for_abstract(gd_residual[cur_iter], vcd_residual[cur_iter], icd_residual[cur_iter],
                                        labels=labels, fig_title=fig_title, cmap='viridis', show_colorbar=True)
        fig_title = '|FFT(recon minus ground truth)| after {} iteration(s) in dB'.format(cur_iter + 1)
        vcu.display_images_for_abstract(gd_residual_fft[cur_iter], vcd_residual_fft[cur_iter], icd_residual_fft[cur_iter],
                                        labels=labels, fig_title=fig_title, vmax=60, cmap='viridis', show_colorbar=True)

    # mbirjax.slice_viewer(vcd_recons, gd_recons, slice_axis=0, vmin=0, vmax=1, title='Recon, vcd left, gd right', slice_label='Iteration')
    # mbirjax.slice_viewer(vcd_residual, gd_residual, slice_axis=0, vmin=-1, vmax=1, title='Residual recon, vcd left, gd right', slice_label='Iteration')
    # mbirjax.slice_viewer(vcd_residual_fft, gd_residual_fft, title='Residual |FFT| in dB\nVCD left, GD right', slice_axis=0, slice_label='Iteration', vmin=0, vmax=60)
    #
    # mbirjax.slice_viewer(vcd_recons, icd_recons, slice_axis=0, vmin=0, vmax=1, title='Recon, vcd left, icd right', slice_label='Iteration')
    # mbirjax.slice_viewer(vcd_residual, icd_residual, slice_axis=0, vmin=-1, vmax=1, title='Residual recon, vcd left, icd right', slice_label='Iteration')
    # mbirjax.slice_viewer(vcd_residual_fft, icd_residual_fft, title='Residual |FFT| in dB\nVCD left, ICD right', slice_axis=0, slice_label='Iteration', vmin=0, vmax=60)

    # fm_rmse_vcd = cur_recon_params.fm_rmse
    # prior_loss_vcd = cur_recon_params.prior_loss
    # default_partition_sequence = parallel_model.get_params('partition_sequence')
    # partition_sequence = mbirjax.gen_partition_sequence(default_partition_sequence, max_iterations=max_iterations)
    # granularity_sequence_vcd = granularity[partition_sequence]

    a = 0