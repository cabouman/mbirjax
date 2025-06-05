# -*- coding: utf-8 -*-
"""
Code to investigate detector and recon offsets and recon by half sino plus stitching.
"""

import numpy as np
import time
import pprint
import jax.numpy as jnp
import mbirjax

"""**Set the geometry parameters**"""

# Choose the geometry type
geometry_type = 'cone'

# Set parameters for the problem size - you can vary these, but if you make num_det_rows very small relative to
# channels, then the generated phantom may not have an interior.
num_views = 64
num_det_rows = 81
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

ct_model = mbirjax.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)

# Generate a simple phantom
print('Creating phantom')
recon_shape = ct_model.get_params('recon_shape')
phantom = np.zeros(recon_shape)  # ct_model.gen_modified_3d_sl_phantom()
phantom[20, 30, 20] = 1
phantom[20, 30, 40] = 2
phantom[20, 30, 55] = 3

""" 
Generate synthetic sinogram data and project natively and separately with a positive det_row_offset
"""
print('Creating sinogram')
delta_det_row = 0.75
ct_model.set_params(delta_det_row=delta_det_row)
sinogram = ct_model.forward_project(phantom)
det_row_offset = 9  # This should be an integer multiple of delta_det_row
ct_model.set_params(det_row_offset=det_row_offset)
sinogram_offset = ct_model.forward_project(phantom)

"""
1. Projecting the same phantom with a different detector offset will produce the same sinogram but shifted (same up to 
interpolation when the offset is a partial detector element).

A positive det_row_offset causes a pixel seen in row i of the original sinogram to be seen in row i + index_offset
of the new sinogram (i.e., the detector moves up, the observed pixel moves down).
"""
row_index_offset = int(det_row_offset / delta_det_row)
max_diff = np.amax(np.abs(sinogram[:, :num_det_rows-row_index_offset, :] - sinogram_offset[:, row_index_offset:, :]))
mbirjax.slice_viewer(sinogram, sinogram_offset,
                     slice_axis=0,
                     title='Native projection (left), shifted projection (right)')
mbirjax.slice_viewer(sinogram[:, :num_det_rows-row_index_offset, :], sinogram_offset[:, row_index_offset:, :],
                     slice_axis=0,
                     title='Native projection (left) cropped at top, shifted projection (right) cropped at bottom\nMax pixel difference = {:.3g}'.format(max_diff))

"""
2. Project to the upper half sinogram plus some overlap
"""
num_extra_rows = 5
num_det_rows_half = num_det_rows // 2 + num_extra_rows
sinogram_half_shape = (num_views, num_det_rows_half, num_det_channels)
det_half_offset = delta_det_row * ((num_det_rows-1)/2 - (num_det_rows_half-1)/2)

ct_model = mbirjax.ConeBeamModel(sinogram_half_shape, angles, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)
ct_model.set_params(delta_det_row=delta_det_row)
ct_model.set_params(det_row_offset=det_half_offset, recon_shape=recon_shape)
sinogram_top = ct_model.forward_project(phantom)
max_diff = np.amax(np.abs(sinogram[:, :num_det_rows_half] - sinogram_top))
mbirjax.slice_viewer(sinogram[:, :num_det_rows_half], sinogram_top, slice_axis=0,
                     title='Top part of native sinogram and half sinogram\nMax pixel diff = {:.3g}'.format(max_diff)  )

full_recon, full_recon_params = ct_model.recon(full_sinogram)

mbirjax.slice_viewer(phantom, full_recon, slice_axis=2)

# Get a model for this sinogram
ct_model_for_full_recon = mbirjax.ConeBeamModel(sinogram.shape, angles, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)
full_recon_det_row_offset = ct_model_for_full_recon.get_params('det_row_offset')

# Take roughly the half of the sinogram and specify roughly half of the volume
full_recon_shape = ct_model_for_full_recon.get_params('recon_shape')
num_recon_slices = full_recon_shape[2]
half_recons = []
num_extra_rows = 2
num_det_rows_half = num_det_rows // 2 + num_extra_rows

# Initialize the model for reconstruction.
sinogram_half_shape = (sinogram_shape[0], num_det_rows_half, sinogram_shape[2])
ct_model_for_half_recon = mbirjax.ConeBeamModel(sinogram_half_shape, angles, source_detector_dist=source_detector_dist,
                                                source_iso_dist=source_iso_dist)
recon_shape_half = ct_model_for_half_recon.get_params('recon_shape')
num_recon_slices_half = recon_shape_half[2]

for sinogram_half, sign in zip([sinogram[:, 0:num_det_rows_half], sinogram[:, -num_det_rows_half:]], [1, -1]):

    # View sinogram
    title = 'Half of sinogram \nUse the sliders to change the view or adjust the intensity range.'
    mbirjax.slice_viewer(sinogram_half, slice_axis=0, title=title, slice_label='View')

    delta_voxel, delta_det_row = ct_model_for_generation.get_params(['delta_voxel', 'delta_det_row'])

    # Determine the recon slice and detector row offsets as the difference between the center using the full
    # sinogram and the center using the half sinogram
    recon_slice_offset = sign * (- delta_voxel * ((num_recon_slices-1)/2 - (num_recon_slices_half-1)/2))
    det_row_offset = sign * delta_det_row * ((num_det_rows-1)/2 - (num_det_rows_half-1)/2)
    det_row_offset = det_row_offset + full_recon_det_row_offset

    ct_model_for_half_recon.set_params(recon_slice_offset=recon_slice_offset, det_row_offset=det_row_offset)

    # Print out model parameters
    ct_model_for_half_recon.print_params()

    """**Do the reconstruction and display the results.**"""

    # ##########################
    # Perform VCD reconstruction
    print('Starting recon')
    time0 = time.time()
    recon, recon_params = ct_model_for_half_recon.recon(sinogram_half)

    recon.block_until_ready()
    elapsed = time.time() - time0
    half_recons.append(recon)
    # ##########################
    mbirjax.slice_viewer(recon)

num_overlap_slices = 2 * num_recon_slices_half - num_recon_slices
num_non_overlap_slices = num_recon_slices - num_recon_slices_half
recon = np.zeros(full_recon_shape)
recon_top, recon_bottom = half_recons
recon[:, :, :num_non_overlap_slices] = recon_top[:, :, :num_non_overlap_slices]
recon[:, :, -num_non_overlap_slices:] = recon_bottom[:, :, -num_non_overlap_slices:]
overlap_weights = (np.arange(num_overlap_slices) + 1.0) / (num_overlap_slices + 1.0)
overlap_weights = overlap_weights.reshape((1, 1, -1))
recon[:, :, num_non_overlap_slices:-num_non_overlap_slices] = (1 - overlap_weights) * recon_top[:, :, -num_overlap_slices:]
recon[:, :, num_non_overlap_slices:-num_non_overlap_slices] += overlap_weights * recon_bottom[:, :, :num_overlap_slices]

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

print('Computing full recon for comparison')
full_recon, full_params = ct_model_for_generation.recon(sinogram)

# Display results
title = 'Standard VCD recon (left) and residual with 2 halves stitched VCD Recon (right) \nThe residual is (stitched recon) - (standard recon).'
mbirjax.slice_viewer(full_recon, recon-full_recon, title=title)

"""**Next:** Try changing some of the parameters and re-running or try [some of the other demos](https://mbirjax.readthedocs.io/en/latest/demos_and_faqs.html).  """