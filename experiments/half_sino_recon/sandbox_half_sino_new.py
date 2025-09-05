# -*- coding: utf-8 -*-
"""
Code to investigate detector and recon offsets and recon by half sino plus stitching.
"""

import numpy as np
import time
import pprint
import jax.numpy as jnp
import mbirjax as mj

"""**Set the geometry parameters**"""

# Choose the geometry type
geometry_type = 'cone'

# Set parameters for the problem size - you can vary these, but if you make num_det_rows very small relative to
# channels, then the generated phantom may not have an interior.
num_views = 64
num_det_rows = 80
num_det_channels = 128

# For cone beam geometry, we need to describe the distances source to detector and source to rotation axis.
# np.Inf is an allowable value, in which case this is essentially parallel beam
source_detector_dist = 4 * num_det_channels
source_iso_dist = source_detector_dist

# For cone beam reconstruction, we need a little more than 180 degrees for full coverage.
detector_cone_angle = 2 * np.arctan2(num_det_channels / 2, source_detector_dist)
start_angle = -np.pi
end_angle = np.pi

# Initialize a full sinogram
sinogram_shape = (num_views, num_det_rows, num_det_channels)
angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

ct_model = mj.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)
delta_det_row = 0.75
det_row_offset = 9
ct_model.set_params(delta_det_row=delta_det_row, det_row_offset=det_row_offset)

# Generate a simple phantom
print('Creating phantom')
recon_shape = ct_model.get_params('recon_shape')
phantom = np.zeros(recon_shape)
phantom[20:25, 30:35, 20:25] = 1
phantom[20:25, 30:35, 40:45] = 2
phantom[20:25, 30:35, 55:60] = 3

""" 
Generate synthetic sinogram data and project natively and separately with a positive det_row_offset
"""
print('Creating sinogram')
sinogram = ct_model.forward_project(phantom)

# Do the recon and split recon
recon, recon_dict = ct_model.recon(sinogram)
recon_split, recon_split_dict = ct_model.recon_split_sino(sinogram)

max_diff = np.amax(np.abs(phantom - recon))
print('Geometry = {}'.format(geometry_type))
nrmse = np.linalg.norm(recon - phantom) / np.linalg.norm(phantom)
pct_95 = np.percentile(np.abs(recon - phantom), 95)
print('NRMSE between recon and phantom = {}'.format(nrmse))
print('Maximum pixel difference between phantom and recon = {}'.format(max_diff))
print('95% of recon pixels are within {} of phantom'.format(pct_95))

# Display results
title = 'Phantom, (left), standard VCD (middle), abs(split VCD - standard VCD) (right)\nIntensity window small to highlight differences.'
mj.slice_viewer(phantom, recon, np.abs(recon_split-recon), data_dicts=[None, recon_dict, recon_split_dict], title=title,
                vmin=0, vmax=0.01)

"""**Next:** Try changing some of the parameters and re-running or try [some of the other demos](https://mbirjax.readthedocs.io/en/latest/demos_and_faqs.html).  """