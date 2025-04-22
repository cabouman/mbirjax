# -*- coding: utf-8 -*-
"""

This code is based on demo_1_shepp_logan.py, but is sandbox code for experiments.

"""

import numpy as np
import time
import pprint
import jax.numpy as jnp
import mbirjax

"""**Set the geometry parameters**"""

# Choose the geometry type
geometry_type = 'cone'  # 'cone' or 'parallel'

# Set parameters for the problem size - you can vary these, but if you make num_det_rows very small relative to
# channels, then the generated phantom may not have an interior.
num_views = 60
num_det_rows = 100
num_det_channels = 100
sharpness = 1.0
snr_db = 30
max_iterations = 20
stop_threshold_change_pct = 0.1

# For cone beam geometry, we need to describe the distances source to detector and source to rotation axis.
# np.Inf is an allowable value, in which case this is essentially parallel beam
source_detector_dist = 4 * num_det_channels
source_iso_dist = source_detector_dist

# For cone beam reconstruction, we need a little more than 180 degrees for full coverage.
if geometry_type == 'cone':
    detector_cone_angle = 2 * np.arctan2(num_det_channels / 2, source_detector_dist)
else:
    detector_cone_angle = 0
start_angle = - np.pi #  -(np.pi + detector_cone_angle) * (1/2)
end_angle = np.pi  #  (np.pi + detector_cone_angle) * (1/2)

"""**Data generation:** For demo purposes, we create a phantom and then project it to create a sinogram.

Note:  the sliders on the viewer won't work in notebook form.  For that you'll need to run the python code with an interactive matplotlib backend, typcially using the command line or a development environment like Spyder or Pycharm to invoke python.  

"""

# Initialize sinogram
sinogram_shape = (num_views, num_det_rows, num_det_channels)
angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

if geometry_type == 'cone':
    ct_model_for_generation = mbirjax.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)
elif geometry_type == 'parallel':
    ct_model_for_generation = mbirjax.ParallelBeamModel(sinogram_shape, angles)
else:
    raise ValueError('Invalid geometry type.  Expected cone or parallel, got {}'.format(geometry_type))

# Generate 3D Shepp Logan phantom
print('Creating phantom')
phantom = ct_model_for_generation.gen_modified_3d_sl_phantom()
mbirjax.slice_viewer(phantom)

num_det_rows = phantom.shape[2]
sinogram_shape = (num_views, num_det_rows, num_det_channels)
ct_model_for_generation = mbirjax.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)

# Generate synthetic sinogram data
print('Creating sinogram')
sinogram = ct_model_for_generation.forward_project(phantom)
sinogram = np.asarray(sinogram)

# View sinogram
title = 'Original sinogram \nUse the sliders to change the view or adjust the intensity range.'
mbirjax.slice_viewer(sinogram, slice_axis=0, title=title, slice_label='View')

"""**Initialize for the reconstruction**"""

# ####################
# Initialize the model for reconstruction.
if geometry_type == 'cone':
    ct_model_for_recon = mbirjax.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)
else:
    ct_model_for_recon = mbirjax.ParallelBeamModel(sinogram_shape, angles)

# Generate weights array - for an initial reconstruction, use weights = None, then modify if needed.
weights = None
# weights = ct_model_for_recon.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')

# Set reconstruction parameter values
# Increase sharpness by 1 or 2 to get clearer edges, possibly with more high-frequency artifacts.
# Decrease by 1 or 2 to get softer edges and smoother interiors.
ct_model_for_recon.set_params(sharpness=sharpness, snr_db=snr_db)

# Print out model parameters
ct_model_for_recon.print_params()

"""**Do the reconstruction and display the results.**"""

# ##########################
# Perform VCD reconstruction
print('Starting recon')
time0 = time.time()
recon, recon_params = ct_model_for_recon.recon(sinogram, weights=weights, max_iterations=max_iterations,
                                               stop_threshold_change_pct=stop_threshold_change_pct)

recon.block_until_ready()
elapsed = time.time() - time0
# ##########################

# Print parameters used in recon
pprint.pprint(recon_params._asdict(), compact=True)

max_diff = np.amax(np.abs(phantom - recon))
print('Geometry = {}'.format(geometry_type))
nrmse = np.linalg.norm(recon - phantom) / np.linalg.norm(phantom)
pct_95 = np.percentile(np.abs(recon - phantom), 95)
print('NRMSE between recon and phantom = {}'.format(nrmse))
print('Maximum pixel difference between phantom and recon = {}'.format(max_diff))
print('95% of recon pixels are within {} of phantom'.format(pct_95))

mbirjax.get_memory_stats()
print('Elapsed time for recon is {:.3f} seconds'.format(elapsed))

# Display results
title = 'Phantom (left) vs Residual (right) \nUse the sliders to change the slice or adjust the intensity range.'
mbirjax.slice_viewer(phantom, (recon-phantom), title=title)

"""**Next:** Try changing some of the parameters and re-running or try [some of the other demos](https://mbirjax.readthedocs.io/en/latest/demos_and_faqs.html).  """