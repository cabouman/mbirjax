# -*- coding: utf-8 -*-
"""See the notebook version of this demo at:

    https://colab.research.google.com/drive/1zG_H6CDjuQxeMRQHan3XEyX2YVKcSSNC

**MBIRJAX: Basic Demo**

See the [MBIRJAX documentation](https://mbirjax.readthedocs.io/en/latest/) for an overview and details.  

This script demonstrates the basic MBIRJAX code by creating a 3D phantom inspired by Shepp-Logan, forward projecting it to create a sinogram, and then using MBIRJAX to perform a Model-Based, Multi-Granular Vectorized Coordinate Descent reconstruction.

For the demo, we create some synthetic data by first making a phantom, then forward projecting it to obtain a sinogram.  

In a real application, you would load your sinogram as a numpy array and use numpy.transpose if needed so that it
has axes in the order (views, rows, channels).  For reference, assuming the rotation axis is vertical, then increasing the row index nominally moves down the rotation axis and increasing the channel index moves to the right as seen from the source.

Select a GPU as runtime type for best performance.
"""

import numpy as np
import time
import pprint
import mbirjax as mj
import jax.numpy as jnp

def gen_cube_phantom(recon_shape, device=None):
    """Code to generate a simple phantom """
    # Compute phantom height and width
    num_recon_rows, num_recon_cols, num_recon_slices = recon_shape[:3]
    phantom_rows = num_recon_rows // 4  # Phantom height
    phantom_cols = num_recon_cols // 4  # Phantom width

    # Allocate phantom memory
    phantom = np.zeros((num_recon_rows, num_recon_cols, num_recon_slices))

    # Compute start and end locations
    start_rows = (num_recon_rows - phantom_rows) // 2
    stop_rows = (num_recon_rows + phantom_rows) // 2
    start_cols = (num_recon_cols - phantom_cols) // 2
    stop_cols = (num_recon_cols + phantom_cols) // 2
    for slice_index in np.arange(num_recon_slices):
        shift_cols = int(slice_index * phantom_cols / num_recon_slices)
        phantom[start_rows:stop_rows, (shift_cols + start_cols):(shift_cols + stop_cols), slice_index] = 1.0 / max(
            phantom_rows, phantom_cols)

    return jnp.array(phantom, device=device)

"""**Set the geometry parameters**"""

# Choose the geometry type
model_type = 'parallel'  # 'cone' or 'parallel'
object_type = 'cube'  # 'shepp-logan' or 'cube'

# Set parameters for the problem size - you can vary these, but if you make num_det_rows very small relative to
# channels, then the generated phantom may not have an interior.
num_views = 2000
num_det_rows = 2000
num_det_channels = 2000
sinogram_shape = (num_views, num_det_rows, num_det_channels)

# Generate simulated data
# In a real application you would not have the phantom, but we include it here for later display purposes
phantom, sinogram, params = mj.generate_demo_data(object_type=object_type, model_type=model_type,
                                                  num_views=num_views, num_det_rows=num_det_rows,
                                                  num_det_channels=num_det_channels)

mj.get_memory_stats()

angles = params['angles']

# View the sinogram
# title = 'Original sinogram \nUse the sliders to change the view or adjust the intensity range.\nRight click the image to see options.'
# mj.slice_viewer(sinogram, data_dicts=params, slice_axis=0, title=title, slice_label='View')

"""**Initialize for the reconstruction**"""

# ####################
# Use the parameters to get the data and initialize the model for reconstruction.
if model_type == 'cone':
    source_detector_dist = params['source_detector_dist']
    source_iso_dist = params['source_iso_dist']
    ct_model = mj.ConeBeamModel(sinogram.shape, angles, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)
else:
    ct_model = mj.ParallelBeamModel(sinogram.shape, angles)

# Generate weights array - for an initial reconstruction, use weights = None, then modify if needed to reduce the effect of possibly noisy sinogram entries.
weights = None
# weights = mj.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')

# Set reconstruction parameter values
# Sharpness is a float, typically in the range (-1, 2).  The default value is 1.0.
# Sharpness of 1 or 2 will yield clearer edges, with more high-frequency variation.
# Sharnpess of -1 or 0 will yield softer edges and smoother interiors.
# Values of sharpness above 3 may lead to slower convergence with high-frequency artifacts, particularly in the center slices in cone beam.
sharpness = 1.0
ct_model.set_params(sharpness=sharpness, use_gpu='automatic')

# Print out model parameters
ct_model.print_params()

"""**Do the reconstruction and display the results.**"""

# ##########################
# Perform VCD reconstruction
print('Starting recon')
time0 = time.time()

# ct_model.recon returns the estimated object along with a dictionary with entries
# 'recon_params', 'model_params', 'logs', and 'notes'
# recon and recon_dict can be used together for viewing and saving to an hdf5 file.
# Saving can be done either in code or through the viewer, and the hdf5 file can be loaded for viewing
# or to recreate the model if desired.
recon, recon_dict = ct_model.recon(sinogram, weights=weights)

max_diff = np.amax(np.abs(phantom - recon))
nrmse = np.linalg.norm(recon - phantom) / np.linalg.norm(phantom)
pct_95 = np.percentile(np.abs(recon - phantom), 95)

elapsed = time.time() - time0

mj.get_memory_stats()
print('Elapsed time for recon is {:.3f} seconds'.format(elapsed))

# ##########################
# Add some notes to include for display and saving
recon_dict['notes'] += 'NRMSE between recon and phantom = {}'.format(nrmse)
recon_dict['notes'] += 'Maximum pixel difference between phantom and recon = {}'.format(max_diff)
recon_dict['notes'] += '95% of recon pixels are within {} of phantom'.format(pct_95)

mj.get_memory_stats()
print('Elapsed time for recon is {:.3f} seconds'.format(elapsed))

# Display results
title = 'Phantom (left) vs VCD Recon (right) \nUse the sliders to change the slice or adjust the intensity range.\nRight click an image to see options.'
mj.slice_viewer(phantom, recon, data_dicts=[None, recon_dict], title=title)

# recon and recon_dict can be saved from the viewer or directly in code
filepath = './output/demo1_recon.h5'
ct_model.save_recon_hdf5(filepath, recon, recon_dict)

# The recon and recon_dict can be reloaded either here or in the viewer, and the recon_dict can be used to recreate
#  the model if desired. The load function can be used even without an existing instance of a ct model.
new_recon, new_recon_dict, new_model = mj.TomographyModel.load_recon_hdf5(filepath, recreate_model=True)

print('recon and recon_dict loaded from {}'.format(filepath))
print('New model created: {}'.format(new_model.get_params('geometry_type')))

# From this you could view again, restart the recon from the previous iteration, etc.

"""**Next:** Try changing some of the parameters and re-running or try [some of the other demos](https://mbirjax.readthedocs.io/en/latest/demos_and_faqs.html).  """