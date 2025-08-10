import numpy as np
import jax.numpy as jnp
import mbirjax as mj

# In utilities.py in main run this at the bottom of the file
# if __name__ == "__main__":
#     phantom = generate_3d_shepp_logan_low_dynamic_range((2000, 2000, 2000), jax.devices('cpu')[0])
#
#     title = 'Phantom Test \nUse the sliders to change the slice or adjust the intensity range.\nRight click an image to see options.'
#     mj.slice_viewer(phantom, title="Phantom Test")
#
#     phantom = np.array(phantom)
#     np.save("../experiments/output/phantom.npy", phantom)

phantom = np.load("../output/phantom.npy")

title = 'Phantom Test \nUse the sliders to change the slice or adjust the intensity range.\nRight click an image to see options.'
mj.slice_viewer(phantom, title="Phantom Test")

# sinogram shape
num_views, num_det_rows, num_det_channels = 2000, 2000, 2000
sinogram_shape = (num_views, num_det_rows, num_det_channels)

# angles
start_angle = -np.pi * (1 / 2)
end_angle = np.pi * (1 / 2)
angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

# recon model
forward_projection_model = mj.ParallelBeamModel(sinogram_shape, angles)
forward_projection_model.set_params(sharpness=0.0, use_gpu='automatic')

pixel_indices = jnp.arange(phantom.shape[0])
voxel_values = forward_projection_model.get_voxels_at_indices(phantom, pixel_indices)
forward_projection = forward_projection_model.sparse_forward_project(voxel_values, pixel_indices,
                                                                     output_device=forward_projection_model.sinogram_device)

