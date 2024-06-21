import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mbirjax

if __name__ == "__main__":
    """
    This is a script to develop, debug, and tune the vcd reconstruction with a parallel beam projector
    """
    # Set parameters
    num_views = 256
    num_det_rows = 1
    num_det_channels = 256
    start_angle = 0
    end_angle = np.pi
    sharpness = 0.0
    num_phantom_slices = 10
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

    # Generate synthetic sinogram data
    print('Creating sinogram')
    sinogram = parallel_model.forward_project(phantom)

    # Generate weights array
    weights = parallel_model.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')

    # Set reconstruction parameter values
    parallel_model.set_params(sharpness=sharpness, verbose=1)
    granularity = np.array([1, 2, 64, 512])
    parallel_model.set_params(granularity=granularity)

    # vcd_partition_sequence = [0, 1, 2, 3, 1, 2, 3, 2, 3, 3, 0, 1, 2, 3, 1, 2, 3, 2, 3, 3]
    vcd_partition_sequence = [0, 1, 2, 3, 3, 2, 2, 2, 3, 3]  # 2, 3, 2, 3, 3, 0, 1, 2, 3, 1, 2, 3, 2, 3, 3]
    gd_partition_sequence =  [0, ]
    icd_partition_sequence = [3, ]

    num_iterations = 5

    sequences = [vcd_partition_sequence, gd_partition_sequence, icd_partition_sequence]
    residual_fft = []
    residual_recon = []
    all_recons = []

    # ##########################
    # Perform reconstructions
    for sequence in sequences:
        parallel_model.set_params(partition_sequence=sequence)
        cur_residual = np.zeros((num_iterations,) + phantom.shape[0:2])
        cur_recon = np.zeros((num_iterations,) + phantom.shape[0:2])
        recon = None
        for iteration in range(num_iterations):
            recon, cur_recon_params = parallel_model.recon(sinogram, weights=weights, num_iterations=iteration + 1,
                                                               first_iteration=iteration, init_recon=recon,
                                                               compute_prior_loss=True)
            cur_residual[iteration] = recon[:, :, 0] - phantom[:, :, 0]
            cur_recon[iteration] = recon[:, :, 0]

        all_recons.append(cur_recon)
        residual_recon.append(cur_residual)
        cur_residual_fft = np.fft.fftn(cur_residual, axes=(1, 2))
        cur_residual_fft = np.fft.fftshift(cur_residual_fft, axes=(1, 2))
        residual_fft.append(20 * np.log10(1e-6 + np.abs(cur_residual_fft)))

    vcd_recons, gd_recons, icd_recons = all_recons
    vcd_residual, gd_residual, icd_residual = residual_recon
    vcd_residual_fft, gd_residual_fft, icd_residual_fft = residual_fft

    mbirjax.slice_viewer(vcd_recons, gd_recons, slice_axis=0, vmin=0, vmax=1, title='Recon, vcd left, gd right', slice_label='Iteration')
    mbirjax.slice_viewer(vcd_residual, gd_residual, slice_axis=0, vmin=-1, vmax=1, title='Residual recon, vcd left, gd right', slice_label='Iteration')
    mbirjax.slice_viewer(vcd_residual_fft, gd_residual_fft, title='Residual |FFT| in dB\nVCD left, GD right', slice_axis=0, slice_label='Iteration', vmin=0, vmax=60)

    mbirjax.slice_viewer(vcd_recons, icd_recons, slice_axis=0, vmin=0, vmax=1, title='Recon, vcd left, icd right', slice_label='Iteration')
    mbirjax.slice_viewer(vcd_residual, icd_residual, slice_axis=0, vmin=-1, vmax=1, title='Residual recon, vcd left, icd right', slice_label='Iteration')
    mbirjax.slice_viewer(vcd_residual_fft, icd_residual_fft, title='Residual |FFT| in dB\nVCD left, ICD right', slice_axis=0, slice_label='Iteration', vmin=0, vmax=60)

    # fm_rmse_vcd = cur_recon_params.fm_rmse
    # prior_loss_vcd = cur_recon_params.prior_loss
    # default_partition_sequence = parallel_model.get_params('partition_sequence')
    # partition_sequence = mbirjax.gen_partition_sequence(default_partition_sequence, num_iterations=num_iterations)
    # granularity_sequence_vcd = granularity[partition_sequence]

    a = 0