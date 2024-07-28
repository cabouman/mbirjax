import numpy as np
import time
import pprint
import jax
import jax.numpy as jnp
import mbirjax.parallel_beam

if __name__ == "__main__":
    """
    This script demonstrates what can go wrong if you use the wrong rotation direction in a cone-beam reconstruction.  
    """

    print('\nStarting a cone-beam reconstruction with wrong rotation direction.\n')

    num_views = 64
    num_det_rows = 40
    num_det_channels = 128

    sharpness = 0.0
    
    # For cone beam geometry, we need to describe the distances source to detector and source to rotation axis.
    # np.Inf is an allowable value, in which case this is essentially parallel beam
    source_detector_dist = 4 * num_det_channels
    source_iso_dist = source_detector_dist

    # For cone beam reconstruction, we need a little more than 180 degrees for full coverage.
    detector_cone_angle = 2 * np.arctan2(num_det_channels / 2, source_detector_dist)
    start_angle = -(np.pi + detector_cone_angle) * (1/2)
    end_angle = (np.pi + detector_cone_angle) * (1/2)

    # Initialize sinogram
    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

    # ################
    # Data generation: Here we create a phantom and then project it to create a sinogram.
    ct_model_for_generation = mbirjax.ConeBeamModel(sinogram_shape, angles,
                                                    source_detector_dist=source_detector_dist,
                                                    source_iso_dist=source_iso_dist)

    # Generate 3D Shepp Logan phantom
    print('Creating phantom')
    phantom = ct_model_for_generation.gen_modified_3d_sl_phantom()

    # Generate synthetic sinogram data
    print('Creating sinogram')
    sinogram = ct_model_for_generation.forward_project(phantom)
    sinogram = np.array(sinogram)

    # View sinogram
    title = 'Original sinogram \nUse the sliders to change the view or adjust the intensity range.'
    mbirjax.slice_viewer(sinogram, slice_axis=0, title=title, slice_label='View')

    # ##########################
    # Perform VCD reconstruction

    print('\nStarting recon with correct rotation\n')
    ct_model_for_recon = mbirjax.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist,
                                               source_iso_dist=source_iso_dist)
    ct_model_for_recon.set_params(sharpness=sharpness)

    recon_correct, _ = ct_model_for_recon.recon(sinogram)

    print('\nStarting recon with incorrect rotation\n')
    angles_reversed = angles[::-1]
    ct_model_for_recon = mbirjax.ConeBeamModel(sinogram_shape, angles_reversed, source_detector_dist=source_detector_dist,
                                               source_iso_dist=source_iso_dist)
    recon_incorrect, _ = ct_model_for_recon.recon(sinogram)

    # Display results
    title = 'Correct rotation recon (left) vs incorrect rotation recon (right)'
    title += '\nThe incorrect angle specification leads to shape distortion and top/bottom reflection.'
    mbirjax.slice_viewer(recon_correct, recon_incorrect, title=title)

