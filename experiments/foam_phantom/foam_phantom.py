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
import os
# print(os.getcwd())
"""**Set the geometry parameters**"""

# Choose the geometry type
model_type = 'cone'  # 'cone' or 'parallel'
object_type = 'shepp-logan'  # 'shepp-logan' or 'cube'

# Set parameters for the problem size - you can vary these, but if you make num_det_rows very small relative to
# channels, then the generated phantom may not have an interior.
num_views = 256
num_det_rows = 512
num_det_channels = 512

# Generate simulated data
# In a real application you would not have the phantom, but we include it here for later display purposes
# phantom, sinogram, params = mj.generate_demo_data(object_type=object_type, model_type=model_type,
#                                                   num_views=num_views, num_det_rows=num_det_rows,
#                                                   num_det_channels=num_det_channels)
# angles = params['angles']
#
# # View the sinogram
# title = 'Original sinogram \nUse the sliders to change the view or adjust the intensity range.\nRight click the image to see options.'
# mj.slice_viewer(sinogram, data_dicts=params, slice_axis=0, title=title, slice_label='View')

"""**Initialize for the reconstruction**"""
with open("./FDKtest/~XM485AFLT_w512h512s256.bin", 'rb') as f:
    binary_data = f.read()
sinogram_foam = np.frombuffer(binary_data, dtype=np.float32)
sinogram_foam = sinogram_foam.reshape(num_views, num_det_rows, num_det_channels)

with open("./FDKtest/~XM485A_w512h512s512_recon.bin", 'rb') as f:
    binary_data = f.read()
phantom_foam = np.frombuffer(binary_data, dtype=np.float32)
phantom_foam = phantom_foam.reshape(num_det_rows, num_det_channels, num_det_channels)
phantom_foam = phantom_foam[:, : ,::-1]

# # Uncomment to restrict to a small number of center rows for speed
# num_rows = 20
# center_row = num_det_rows // 2
# sinogram_foam = sinogram_foam[:, center_row - (num_rows // 2):center_row + (num_rows // 2)]
# phantom_foam = phantom_foam[center_row - (num_rows // 2):center_row + (num_rows // 2)]

####################
# Use the parameters to get the data and initialize the model for reconstruction.

# mj.slice_viewer(sinogram_foam, slice_axis=0, title='Sinogram')
phantom_foam = phantom_foam.transpose(1, 2, 0)
# mj.slice_viewer(phantom_foam, slice_axis=2, title='Phantom')

det_pixel_pitch = 0.002
det_channel_offset = 0  # 0.2555

if model_type == 'cone':
    source_detector_dist = 200
    source_iso_dist = 100
    ct_model = mj.ConeBeamModel(sinogram_foam.shape, np.linspace(-np.pi, np.pi, 256, endpoint=False), source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)
    ct_model.set_params(det_channel_offset=det_channel_offset, delta_det_channel=det_pixel_pitch, delta_det_row=det_pixel_pitch)
else:
    ct_model = mj.ParallelBeamModel(sinogram_foam.shape, np.linspace(-np.pi, np.pi, 256, endpoint=False))
    ct_model.set_params(det_channel_offset=det_channel_offset, delta_det_channel=det_pixel_pitch, delta_det_row=det_pixel_pitch)

ct_model.auto_set_recon_size(sinogram_foam.shape)

# Generate weights array - for an initial reconstruction, use weights = None, then modify if needed to reduce the effect of possibly noisy sinogram entries.
weights = None
# weights = mj.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')

# Set reconstruction parameter values
# Sharpness is a float, typically in the range (-1, 2).  The default value is 1.0.
# Sharpness of 1 or 2 will yield clearer edges, with more high-frequency variation.
# Sharnpess of -1 or 0 will yield softer edges and smoother interiors.
# Values of sharpness above 3 may lead to slower convergence with high-frequency artifacts, particularly in the center slices in cone beam.
sharpness = 1.0
ct_model.set_params(sharpness=sharpness)

# Print out model parameters
ct_model.print_params()

"""**Do the reconstruction and display the results.**"""

# ##########################
# Perform VCD reconstruction
direct_recon = ct_model.direct_recon(sinogram_foam)

print('Starting recon')
time0 = time.time()

# ct_model.recon returns the estimated object along with a dictionary with entries
# 'recon_params', 'model_params', 'logs', and 'notes'
# recon and recon_dict can be used together for viewing and saving to an hdf5 file.
# Saving can be done either in code or through the viewer, and the hdf5 file can be loaded for viewing
# or to recreate the model if desired.
recon, recon_dict = ct_model.recon(sinogram_foam, init_recon=direct_recon, weights=weights)

max_diff = np.amax(np.abs(phantom_foam - recon))
nrmse = np.linalg.norm(recon - phantom_foam) / np.linalg.norm(phantom_foam)
pct_95 = np.percentile(np.abs(recon - phantom_foam), 95)

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
title = 'Zeiss (left), direct recon(middle), VCD Recon (right)'
mj.slice_viewer(phantom_foam, direct_recon, recon, data_dicts=[None, None, recon_dict], title=title)

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