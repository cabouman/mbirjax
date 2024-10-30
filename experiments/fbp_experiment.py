"""
**MBIRJAX: Filtered back projection (FBP) basic demo**

This is a modified version of the demo_1_shepp_logan.py file that is more streamlined and uses filtered back projection insetad of the standard recon.

"""
import numpy as np
import time
import jax.numpy as jnp
import mbirjax


# Set geometry parameters
num_views = 64
num_det_rows = 128
num_det_channels = 128
start_angle = -(jnp.pi) * (1/2)
end_angle = (jnp.pi) * (1/2)
angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

# Initialize sinogram
sinogram_shape = (num_views, num_det_rows, num_det_channels)
parallel_model = mbirjax.ParallelBeamModel(sinogram_shape, angles)

# Generate 3D Shepp Logan phantom
print("Creating phantom", end="\n\n")
phantom = parallel_model.gen_modified_3d_sl_phantom()

# Generate sinogram from phantom
print("Creating sinogram", end="\n\n")
sinogram = parallel_model.forward_project(phantom)

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
