import numpy as np
import jax.numpy as jnp
import mbirjax as mj

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
    cone_model = mj.ConeBeamModel(sinogram.shape, angles, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)

    # Generate 3D Shepp Logan phantom
    print('Creating phantom')
    phantom_shape = cone_model.get_params('recon_shape')
    phantom = mj.generate_3d_shepp_logan_low_dynamic_range(phantom_shape)

    # View the phantom
    mj.slice_viewer(phantom, phantom.transpose((1, 2, 0)), title='Phantom axial and coronal', slice_label='View')

    # Generate synthetic sinogram data
    print('Creating sinogram')
    sinogram = cone_model.forward_project(phantom)

    # View sinogram
    mj.slice_viewer(sinogram, title='Sinogram', slice_axis=0)
