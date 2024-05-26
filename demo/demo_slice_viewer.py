import numpy as np
import jax.numpy as jnp
import mbirjax.plot_utils as pu
import mbirjax.parallel_beam

if __name__ == "__main__":
    """
    This is a script to demonstrate the built-in slice viewer
    """
    # Set parameters
    num_views = 128
    num_det_rows = 128
    num_det_channels = 128
    source_detector_dist = 4 * num_det_channels
    source_iso_dist = source_detector_dist

    detector_cone_angle = 2 * np.arctan2(num_det_channels / 2, source_detector_dist)
    start_angle = -(np.pi + detector_cone_angle) * (1/2)
    end_angle = (np.pi + detector_cone_angle) * (1/2)
    sharpness = 0.0

    # Initialize sinogram
    sinogram = jnp.zeros((num_views, num_det_rows, num_det_channels))
    angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

    # Set up parallel beam model
    cone_model = mbirjax.ConeBeamModel(sinogram.shape, angles, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)

    # Generate 3D Shepp Logan phantom
    print('Creating phantom')
    phantom = cone_model.gen_modified_3d_sl_phantom()

    # View the phantom
    pu.slice_viewer(phantom, phantom.transpose((1, 2, 0)), title='Phantom axial and coronal', slice_label='View')

    # Generate synthetic sinogram data
    print('Creating sinogram')
    sinogram = cone_model.forward_project(phantom)

    # View sinogram
    pu.slice_viewer(sinogram, title='Sinogram', slice_axis=0)
