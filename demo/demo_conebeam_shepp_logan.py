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
    num_views = 64
    num_det_rows = 10
    num_det_channels =128
    source_detector_dist = 1000
    magnification = 1.0
    start_angle = -np.pi*(1/2)
    end_angle = np.pi*(1/2)
    sharpness = 0.0

    # Initialize sinogram
    sinogram = jnp.zeros((num_views, num_det_rows, num_det_channels))
    angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

    # Set up parallel beam model
    cone_model = mbirjax.ConeBeamModel(sinogram.shape, angles, source_detector_dist=source_detector_dist, magnification=magnification)

    # Generate 3D Shepp Logan phantom
    phantom = cone_model.gen_3d_sl_phantom()

    # Generate synthetic sinogram data
    sinogram = cone_model.forward_project(phantom)

    # Generate weights array
    weights = cone_model.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')

    # Set reconstruction parameter values
    cone_model.set_params(sharpness=sharpness, verbose=1)
    # cone_model.set_params(positivity_flag=True)

    # Print out model parameters
    cone_model.print_params()

    # ##########################
    # Perform VCD reconstruction
    time0 = time.time()
    recon, fm_rmse = cone_model.recon(sinogram, weights=weights)

    recon.block_until_ready()
    elapsed = time.time() - time0
    print('Elapsed time for recon is {:.3f} seconds'.format(elapsed))
    # ##########################

    # Reshape recon into 3D form
    recon_3d = cone_model.reshape_recon(recon)

    # Display results
    pu.slice_viewer(phantom, recon_3d, title='Phantom (left) vs VCD Recon (right)')

    # You can also display individual slides with the sinogram
    #pu.display_slices(phantom, sinogram, recon_3d)
