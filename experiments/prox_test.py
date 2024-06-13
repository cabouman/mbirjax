import numpy as np
import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mbirjax.plot_utils as pu
import mbirjax.parallel_beam

if __name__ == "__main__":
    """
    This is a script to develop, debug, and tune the vcd reconstruction with a parallel beam projector
    """
    # Set parameters
    num_views = 256
    num_det_rows = 20
    num_det_channels = 256
    start_angle = -np.pi*(1/2)
    end_angle = np.pi*(1/2)
    sharpness = 0.0

    # Initialize sinogram
    sinogram = jnp.zeros((num_views, num_det_rows, num_det_channels))
    angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

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
    # ct_model.set_params(positivity_flag=True)

    # Print out model parameters
    parallel_model.print_params()

    # ##########################
    # Test proximal map for fixed point
    # Run auto regularization. If auto_regularize_flag is False, then this will have no effect
    parallel_model.auto_set_regularization_params(sinogram, weights=weights)
    init_recon = phantom + 1.0
    recon, loss_vectors = parallel_model.prox_map(phantom, sinogram, weights=weights, init_recon=init_recon, num_iterations=13)

    # Reshape recon into 3D form
    recon_3d = parallel_model.reshape_recon(recon)

    # Display results
    pu.slice_viewer(phantom, recon_3d, title='Phantom (left) vs VCD Recon (right)')

    # You can also display individual slides with the sinogram
    #pu.display_slices(phantom, sinogram, recon_3d)
