# -*- coding: utf-8 -*-
"""demo_fbp_fdk.ipynb

**MBIRJAX: FBP and FDK Reconstruction Demo**

See the [MBIRJAX documentation](https://mbirjax.readthedocs.io/en/latest/) for an overview and details.

This script demonstrates the use of MBIRJAX to perform Filtered Back Projection (FBP) for parallel beam reconstruction and Feldkamp-Davis-Kress reconstruction (FDK) for cone beam reconstruction. We use synthetic data by first creating a Shepp-Logan phantom, then forward-project it to generate a sinogram, and finally reconstruct using the FBP or FDK method.

"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install mbirjax
import os, sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)
import numpy as np
import time
import jax.numpy as jnp
import mbirjax

"""**Set the geometry parameters**"""

# Choose the geometry type
geometry_type = 'cone'  # 'cone' or 'parallel'

# Set parameters for the problem size
num_views = 64
num_det_rows = 128
num_det_channels = 128

# For cone beam geometry, we need to describe the distances source to detector and source to rotation axis.
# np.Inf is an allowable value, in which case this is essentially parallel beam
source_detector_dist = 4 * num_det_channels
source_iso_dist = source_detector_dist

# For cone beam reconstruction, we need a little more than 180 degrees for full coverage.
if geometry_type == 'cone':
    detector_cone_angle = 2 * np.arctan2(num_det_channels / 2, source_detector_dist)
else:
    detector_cone_angle = 0

start_angle = -(np.pi + detector_cone_angle) * (1/2)
end_angle = (np.pi + detector_cone_angle) * (1/2)

"""**Data generation:** For demo purposes, we create a phantom and then project it to create a sinogram.

Note:  the sliders on the viewer won't work in notebook form.  For that you'll need to run the python code with an interactive matplotlib backend, typcially using the command line or a development environment like Spyder or Pycharm to invoke python.

"""

# Initialize sinogram
sinogram_shape = (num_views, num_det_rows, num_det_channels)
angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

if geometry_type == 'cone':
    model = mbirjax.ConeBeamModel(sinogram_shape, angles,
    source_detector_dist=source_detector_dist,
    source_iso_dist=source_iso_dist)
else:
    model = mbirjax.ParallelBeamModel(sinogram_shape, angles)

"""**Data generation:** Create a phantom and generate a sinogram**"""

# Generate 3D Shepp-Logan phantom
print("Creating phantom", end="\n\n")
phantom = model.gen_modified_3d_sl_phantom()

# Generate synthetic sinogram data
print("Creating sinogram", end="\n\n")
sinogram = model.forward_project(phantom)

# View sinogram
title = "Original sinogram \nUse the sliders to change the view or adjust the intensity range."
mbirjax.slice_viewer(sinogram, slice_axis=0, title=title, slice_label="View")

"""**Perform reconstruction and display results**"""

if geometry_type == 'cone':
    print("Starting FDK recon")
    time0 = time.time()
    recon = model.fdk_recon(sinogram, filter_name="ramp")
else:
    print("Starting FBP recon")
    time0 = time.time()
    recon = model.fbp_recon(sinogram, filter_name="ramp")

recon.block_until_ready()
elapsed = time.time() - time0
print(f"Elapsed time for recon is {elapsed:.3f} seconds", end="\n\n")

# Compute descriptive statistics about recon result
nrmse = np.linalg.norm(recon - phantom) / np.linalg.norm(phantom)
max_diff = np.amax(np.abs(phantom - recon))
pct_95 = np.percentile(np.abs(recon - phantom), 95)
print('Geometry = {}'.format(geometry_type))
print(f"NRMSE between recon and phantom = {nrmse}")
print(f"Maximum pixel difference between phantom and recon = {max_diff}")
print(f"95% of recon pixels are within {pct_95} of phantom")

# Get stats on memory usage
mbirjax.get_memory_stats()

# Display results
title = (f"Phantom (left) vs {'FDK' if geometry_type == 'cone' else 'FBP'} Recon (right). "
         f"Filter used: ramp. \nUse the sliders to change the slice or adjust the intensity range.")
mbirjax.slice_viewer(phantom, recon, title=title)
