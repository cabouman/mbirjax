# -*- coding: utf-8 -*-
"""

This code is based on demo_1_shepp_logan.py, but is sandbox code for experiments.

"""

import numpy as np
import time
import pprint
import jax.numpy as jnp
import mbirjax as mj

"""**Set the geometry parameters**"""

# Choose the geometry type
model_type = 'parallel'  # 'cone' or 'parallel'
object_type = 'shepp-logan'  # 'shepp-logan' or 'cube'

# Set parameters for the problem size - you can vary these, but if you make num_det_rows very small relative to
# channels, then the generated phantom may not have an interior.
num_views = 60
num_det_rows = 100
num_det_channels = 100
sharpness = 1.0
snr_db = 30
max_iterations = 20
stop_threshold_change_pct = 0.1

"""**Data generation:** For demo purposes, we create a phantom and then project it to create a sinogram.

Note:  the sliders on the viewer won't work in notebook form.  For that you'll need to run the python code with an interactive matplotlib backend, typcially using the command line or a development environment like Spyder or Pycharm to invoke python.  

"""

# Generate simulated data
# In a real application you would not have the phantom, but we include it here for later display purposes
phantom, sinogram, params = mj.generate_demo_data(object_type=object_type, model_type=model_type,
                                                  num_views=num_views, num_det_rows=num_det_rows,
                                                  num_det_channels=num_det_channels)
angles = params['angles']


# View sinogram
title = 'Original sinogram \nUse the sliders to change the view or adjust the intensity range.'
mj.slice_viewer(sinogram, slice_axis=0, title=title, slice_label='View')

"""**Initialize for the reconstruction**"""

# ####################
# Use the parameters to get the data and initialize the model for reconstruction.
if model_type == 'cone':
    source_detector_dist = params['source_detector_dist']
    source_iso_dist = params['source_iso_dist']
    ct_model = mj.ConeBeamModel(sinogram.shape, angles, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)
else:
    ct_model = mj.ParallelBeamModel(sinogram.shape, angles)

# Generate weights array - for an initial reconstruction, use weights = None, then modify if needed.
weights = None
# weights = ct_model.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')

# Set reconstruction parameter values
# Increase sharpness by 1 or 2 to get clearer edges, possibly with more high-frequency artifacts.
# Decrease by 1 or 2 to get softer edges and smoother interiors.
ct_model.set_params(sharpness=sharpness, snr_db=snr_db)

# Print out model parameters
ct_model.print_params()

"""**Do the reconstruction and display the results.**"""

# ##########################
# Perform VCD reconstruction
print('Starting recon')
time0 = time.time()
recon, recon_dict = ct_model.recon(sinogram, weights=weights, max_iterations=max_iterations,
                                               stop_threshold_change_pct=stop_threshold_change_pct)

recon.block_until_ready()
elapsed = time.time() - time0
# ##########################

# Print parameters used in recon
recon_params = recon_dict['recon_params']
pprint.pprint(recon_params, compact=True)

max_diff = np.amax(np.abs(phantom - recon))
print('Geometry = {}'.format(model_type))
nrmse = np.linalg.norm(recon - phantom) / np.linalg.norm(phantom)
pct_95 = np.percentile(np.abs(recon - phantom), 95)
print('NRMSE between recon and phantom = {}'.format(nrmse))
print('Maximum pixel difference between phantom and recon = {}'.format(max_diff))
print('95% of recon pixels are within {} of phantom'.format(pct_95))

mj.get_memory_stats()
print('Elapsed time for recon is {:.3f} seconds'.format(elapsed))

# Display results
title = 'Phantom (left) vs Residual (right) \nUse the sliders to change the slice or adjust the intensity range.'
mj.slice_viewer(phantom, (recon-phantom), title=title)

"""**Next:** Try changing some of the parameters and re-running or try [some of the other demos](https://mbirjax.readthedocs.io/en/latest/demos_and_faqs.html).  """