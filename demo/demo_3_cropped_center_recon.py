import numpy as np
import time
import pprint
import jax
import jax.numpy as jnp
import mbirjax.parallel_beam

if __name__ == "__main__":
    """
    This script demonstrates how to do a reconstruction only near the center of rotation.  This is a difficult
    problem for model-based methods, so the reconstruction has some unavoidable artifacts and intensity shifts, but
    it does show features near the center without having to do a full reconstruction, which may be useful for
    large sinograms.  
    
    For this demo, we use a larger detector since the artifacts are more pronounced for very small-scale problems.  
    To maintain a short run-time, we reduce the number of slices. 
    
    Also, we use a phantom that projects fully onto the detector, but you can use the same method when the object
    projects partially outside the detector.  
    
    For simplicity, we show this only for parallel beam, but the same steps apply for cone beam.  
    """

    print('Starting a recon in which only pixels near the center of rotation are estimated.\n')
    num_views = 400
    num_det_rows = 20
    num_det_channels = 400

    start_angle = - np.pi / 2
    end_angle = np.pi / 2

    # Initialize sinogram
    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

    # ################
    # Data generation: Here we create a phantom and then project it to create a sinogram.

    ct_model_for_generation = mbirjax.ParallelBeamModel(sinogram_shape, angles)

    # Generate 3D Shepp Logan phantom
    print('Creating phantom')

    phantom = ct_model_for_generation.gen_modified_3d_sl_phantom()

    mbirjax.slice_viewer(phantom, title='Original phantom - few slices', slice_axis=2, slice_label='Slice')

    # Generate synthetic sinogram data
    print('Creating sinogram')
    sinogram = ct_model_for_generation.forward_project(phantom)
    sinogram = np.array(sinogram)

    # View sinogram
    mbirjax.slice_viewer(sinogram, title='Original sinogram - few rows', slice_axis=0, slice_label='View')

    # Initialize model for reconstruction.
    weights = None
    ct_model_for_recon = mbirjax.ParallelBeamModel(sinogram_shape, angles)

    # Print model parameters
    ct_model_for_recon.print_params()

    # ###############################
    # Cropped center recon VCD reconstruction
    # We can also reduce the size of the recon to reconstruct just the central region.
    sharpness = -0.5
    recon_row_scale = 0.5
    recon_col_scale = 0.5
    ct_model_for_recon.scale_recon_shape(row_scale=recon_row_scale, col_scale=recon_col_scale)
    ct_model_for_recon.set_params(sharpness=sharpness)
    print('Starting cropped center recon')
    recon, recon_params = ct_model_for_recon.recon(sinogram, weights=weights)

    # Print out parameters used in recon
    pprint.pprint(recon_params._asdict(), compact=True)

    # Display results
    title = 'Cropped center recon with sharpness = {:.1f}: Phantom (left) vs VCD Recon (right)'.format(sharpness)
    title += '\nThis recon does not include all pixels used to generate the sinogram.'
    title += '\nThe missing pixels lead to an intensity shift (adjust intensity to [0, 1]) and a bright outer ring.'
    recon_shape = ct_model_for_recon.get_params('recon_shape')
    recon_radius = [length // 2 for length in recon_shape]
    start_inds = [phantom.shape[j] // 2 - recon_radius[j] for j in range(2)]
    end_inds = [start_inds[j] + recon_shape[j] for j in range(2)]
    cropped_phantom = phantom[start_inds[0]:end_inds[0], start_inds[1]:end_inds[1]]
    mbirjax.slice_viewer(cropped_phantom, recon, title=title)
