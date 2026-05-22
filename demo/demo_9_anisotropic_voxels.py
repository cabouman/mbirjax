# -*- coding: utf-8 -*-
"""
**MBIRJAX: Anisotropic Voxel Reconstruction Demo**

See the [MBIRJAX documentation](https://mbirjax.readthedocs.io/en/latest/) for an overview and details.  

This script demonstrates parallel beam or cone beam reconstruction with anisotropic (non-cubic) voxels.

"""

import numpy as np
import time
import pprint
import mbirjax as mj

# -------------------------
# Demo configuration
# -------------------------
object_type = "shepp-logan"   # "shepp-logan" or "cube"
model_type = 'cone'  # 'cone' or 'parallel'
use_curved_detector = False

num_views = 360   # total number of views in the sinogram
num_det_rows = 40
num_det_channels = 128
voxel_row_aspect = 0.5
if model_type == 'parallel': # setting voxel slice aspect is not supported for parallel beam
    voxel_slice_aspect = 1.0
elif model_type == 'cone':
    voxel_slice_aspect = 1./3.

# Helical controls 
use_helical = False        # set False for circular reconstruction
helical_pitch = 1.0       # dimensionless. helical_pitch = (table travel per rotation) / (det height at iso).
helical_z_range = 80.0    # ALU total travel
helical_z_center = 40.0   # ALU

# -------------------------
# Generate simulated data
# In a real application you would not have the phantom, but we include it here for later display purposes
# -------------------------
print("Generating demo data...")
phantom, sinogram, params = mj.generate_demo_data(
    object_type=object_type,
    model_type=model_type,
    num_views=num_views,
    num_det_rows=num_det_rows,
    num_det_channels=num_det_channels,
    use_helical=use_helical,
    helical_pitch=helical_pitch,
    helical_z_range=helical_z_range,
    helical_z_center=helical_z_center,
    use_curved_detector=use_curved_detector
)

phantom = np.array(phantom)
angles = params["angles"]
helical_z_shifts = params.get("helical_z_shifts", None)

# View the sinogram
title = 'Original sinogram \nUse the sliders to change the view or adjust the intensity range.\nRight click the image to see options.'
mj.slice_viewer(sinogram, data_dicts=params, slice_axis=0, title=title, slice_label='View')

# -------------------------
# Create model for recon
# -------------------------

# ####################
# Use the parameters to get the data and initialize the model for reconstruction.
if model_type == 'cone':
    source_detector_dist = params['source_detector_dist']
    source_iso_dist = params['source_iso_dist']
    ct_model = mj.ConeBeamModel(
        sinogram.shape,
        angles,
        source_detector_dist=source_detector_dist,
        source_iso_dist=source_iso_dist,
        helical_z_shifts=helical_z_shifts,
        use_curved_detector=use_curved_detector
    )
    ct_model.set_params(voxel_row_aspect=voxel_row_aspect)
    ct_model.set_params(voxel_slice_aspect=voxel_slice_aspect)
    ct_model.auto_set_recon_geometry()
else:
    ct_model = mj.ParallelBeamModel(sinogram.shape, angles)
    ct_model.set_params(voxel_row_aspect=voxel_row_aspect)
    ct_model.auto_set_recon_geometry()

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

# -------------------------
# Perform VCD reconstruction
# -------------------------

print('Starting recon')
time0 = time.time()

# ct_model.recon returns the estimated object along with a dictionary with entries
# 'recon_params', 'model_params', 'logs', and 'notes'
# recon and recon_dict can be used together for viewing and saving to an hdf5 file.
# Saving can be done either in code or through the viewer, and the hdf5 file can be loaded for viewing
# or to recreate the model if desired.
recon, recon_dict = ct_model.recon(
    sinogram,
    weights=weights,
    max_iterations=15
)

# Generate a phantom with the same voxel dimensions as the recon:
phantom, _, _ = mj.generate_demo_data(
    object_type=object_type,
    model_type=model_type,
    num_views=num_views,
    num_det_rows=num_det_rows,
    num_det_channels=num_det_channels,
    use_helical=use_helical,
    helical_pitch=helical_pitch,
    helical_z_range=helical_z_range,
    helical_z_center=helical_z_center,
    use_curved_detector=use_curved_detector,
    voxel_row_aspect=voxel_row_aspect,
    voxel_slice_aspect=voxel_slice_aspect
)

phantom = np.array(phantom)

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
new_recon, new_recon_dict = mj.TomographyModel.load_recon_hdf5(filepath)

print('recon and recon_dict loaded from {}'.format(filepath))

# From this you could view again, restart the recon from the previous iteration, etc.

"""**Next:** Try changing some of the parameters and re-running or try [some of the other demos](https://mbirjax.readthedocs.io/en/latest/demos_and_faqs.html).  """