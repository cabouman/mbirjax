import numpy as np
import pprint
import jax.numpy as jnp
import mbirjax as mj
import mbirjax.preprocess as mjp

if __name__ == "__main__":
    """
    This is a script demonstrates the reconstruction of a microscopy tilt sequence.
    """
    # Set parameters
    num_views = 64
    num_det_rows = 40
    num_det_channels = 256
    sharpness = 0.0
    num_iterations = 25
    
    # These can be adjusted to describe the geometry in the cone beam case.
    source_detector_dist = 4 * num_det_channels
    source_iso_dist = source_detector_dist

    # assume a tilt sequence of +-60 deg
    start_angle = -np.pi * (1/3)
    end_angle = np.pi * (1/3)

    # Initialize sinogram
    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

    # Set up the model
    ct_model = mj.ParallelBeamModel(sinogram_shape, angles)

    # Reduce region to be thin as is typical for microscopy samples
    recon_shape = ct_model.scale_recon_shape(row_scale=0.25)

    print(recon_shape)

    # Generate 3D Shepp Logan phantom
    phantom = ct_model.gen_modified_3d_sl_phantom()

    # Generate synthetic sinogram data
    sinogram = ct_model.forward_project(phantom)

    # View sinogram
    mj.slice_viewer(sinogram, title='Synthetic Sinogram', slice_axis=0, slice_label='View')

    # Generate weights array - for an initial reconstruction, use weights = None, then modify as desired.
    weights = None
    # weights = ct_model.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')

    # Set reconstruction parameter values
    ct_model.set_params(sharpness=sharpness, verbose=1)

    # Print out model parameters
    ct_model.print_params()

    # Perform MBIR reconstruction
    recon, recon_params = ct_model.recon(sinogram, weights=weights, num_iterations=num_iterations)

    # Print out parameters used in recon
    pprint.pprint(recon_params['recon_params'])

    # Display results
    mj.slice_viewer(phantom, recon, title='Phantom (left) vs MBIR Recon (right)')

