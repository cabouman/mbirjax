"""
**MBIRJAX: Filtered back projection (FBP) basic demo**

This is a modified version of the demo_1_shepp_logan.py file that is more streamlined and uses filtered back projection insetad of the standard recon.

"""
import numpy as np
import time
import jax.numpy as jnp
import mbirjax
import matplotlib.pyplot as plt


# Set geometry parameters
num_views = 3
num_det_rows = 3
num_det_channels = 3
start_angle = -(jnp.pi) * (1/2)
end_angle = (jnp.pi) * (1/2)
angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

# Initialize sinogram
sinogram_shape = (num_views, num_det_rows, num_det_channels)
parallel_model = mbirjax.ParallelBeamModel(sinogram_shape, angles)
parallel_model.set_params(delta_det_channel=3)
parallel_model.set_params(delta_det_row=3)
parallel_model.set_params(delta_voxel=10)
# Generate 3D single voxel phantom
print("Creating phantom", end="\n\n")
# Create a one-voxel phantom at the center of the volume
# phantom = jnp.zeros((num_det_rows, num_det_rows, num_det_channels))
# phantom = phantom.at[num_det_rows // 2, num_det_rows // 2:num_det_rows // 2 + 2, num_det_channels // 2].set(1.0)  # Set central voxel to 1.0
# phantom = phantom.at[num_det_rows // 2, num_det_rows // 2:num_det_rows // 2 + 2, num_det_channels // 2].set(1.0)
phantom = jnp.zeros((3, 3, 3))
phantom = phantom.at[1, 1:3, 1].set(1.0)

# Generate sinogram from one-voxel phantom
print("Creating sinogram for one-voxel phantom", end="\n\n")
sinogram = parallel_model.forward_project(phantom)
# plt.imshow(sinogram.reshape(3,3))
# View sinogram
title = "Original sinogram \nUse the sliders to change the view or adjust the intensity range."
mbirjax.slice_viewer(sinogram, slice_axis=0, title=title, slice_label="View")

# Print out model parameters
# parallel_model.print_params()

################################################################################
# Reconstruction starts here
################################################################################

# Perform FBP reconstruction
print("Starting recon", end="\n\n")
time0 = time.time()
filter_name = "ramp"
recon = parallel_model.fbp_recon(sinogram, filter_name=filter_name)

recon.block_until_ready()
elapsed = time.time() - time0
print(f"Elapsed time for recon is {elapsed} seconds", end="\n\n")

# Compute descriptive statistics about recon result
max_diff = np.amax(np.abs(phantom - recon))
nrmse = np.linalg.norm(recon - phantom) / np.linalg.norm(phantom)
pct_95 = np.percentile(np.abs(recon - phantom), 95)
print(f"NRMSE between recon and phantom = {nrmse}")
print(f"Maximum pixel difference between phantom and recon = {max_diff}")
print(f"95% of recon pixels are within {pct_95} of phantom")

# Get stats on memory usage
mbirjax.get_memory_stats()

# Display results
title = f"Phantom (left) vs filtered back projection (right). Filter used: {filter}. \nUse the sliders to change the slice or adjust the intensity range."
mbirjax.slice_viewer(phantom, recon, title=title)