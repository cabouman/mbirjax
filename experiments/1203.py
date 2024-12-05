import numpy as np
import time
import jax.numpy as jnp
import mbirjax
import matplotlib.pyplot as plt
import scipy
num_views = 128
num_det_rows = 1
num_det_channels = 127
start_angle = -(jnp.pi) * (1/2)
end_angle = (jnp.pi) * (1/2)
angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

# Initialize sinogram
sinogram_shape = (num_views, num_det_rows, num_det_channels)
parallel_model = mbirjax.ParallelBeamModel(sinogram_shape, angles)
parallel_model.set_params(delta_det_channel=1.3)
parallel_model.set_params(delta_det_row=1.3)
parallel_model.set_params(delta_voxel=2.3)
# Generate 3D single voxel phantom
print("Creating phantom", end="\n\n")
phantom = jnp.zeros((127, 127))
for i in range(127):
    for j in range(127):
        # Check if the point is inside the disk
        if jnp.sqrt((i - 63)**2 + (j - 63)**2) < 6:
            phantom = phantom.at[i, j].set(1.0)

num_views = 64
num_det_rows = 128
num_det_channels = 128
start_angle = -(jnp.pi) * (1/2)
end_angle = (jnp.pi) * (1/2)
angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

# Initialize sinogram
sinogram_shape = (num_views, num_det_rows, num_det_channels)
parallel_model = mbirjax.ParallelBeamModel(sinogram_shape, angles)
parallel_model.set_params(delta_det_channel=1.4)
parallel_model.set_params(delta_det_row=1.4)
parallel_model.set_params(delta_voxel=1.5)
# Generate 3D Shepp Logan phantom
print("Creating phantom", end="\n\n")
phantom = parallel_model.gen_modified_3d_sl_phantom()

sinogram = parallel_model.forward_project(phantom)
recon = parallel_model.fbp_recon(sinogram, filter_name='ramp')
mbirjax.slice_viewer(phantom, recon, title='FBP result, delta_det_channel=1.4; delta_voxel=1.5')