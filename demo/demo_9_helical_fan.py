#!/usr/bin/env python3
"""
**MBIRJAX: Helical Fan Beam Demo** (currently compares Fan to Cone+curved detector)

See the [MBIRJAX documentation](https://mbirjax.readthedocs.io/en/latest/) for an overview and details.

This script demonstrates helical fan beam (curved detector) reconstruction by creating a 3D phantom inspired
by Shepp-Logan, forward projecting it to create a sinogram, and then using MBIRJAX to perform a Model-Based,
Multi-Granular Vectorized Coordinate Descent reconstruction.

"""

import sys
import time
import numpy as np
import mbirjax as mj


# -------------------------
# Demo configuration
# -------------------------
object_type = "shepp-logan"   # "shepp-logan" or "cube"
model_type = "fan"

num_views = 360   # total number of views in the sinogram
num_det_rows = 40
num_det_channels = 128

# Helical controls
use_helical = True        # set False for circular reconstruction
helical_pitch = 1.0       # dimensionless. helical_pitch = (table travel per rotation) / (collimation at iso).
helical_z_range = 80.0    # ALU total travel
helical_z_center = 40.0   # ALU

# Generate weights array - for an initial reconstruction, use weights = None, then modify if needed to reduce the effect of possibly noisy sinogram entries.
weights = None
# weights = mj.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')

# Set reconstruction parameter values
# Sharpness is a float, typically in the range (-1, 2).  The default value is 1.0.
# Sharpness of 1 or 2 will yield clearer edges, with more high-frequency variation.
# Sharnpess of -1 or 0 will yield softer edges and smoother interiors.
# Values of sharpness above 3 may lead to slower convergence with high-frequency artifacts, particularly in the center slices in cone beam.
sharpness = 1.0

# -------------------------
# Generate simulated data
# -------------------------
print("Generating demo data...")
phantom, sinogram_fan, params = mj.generate_demo_data(
    object_type=object_type,
    model_type=model_type,
    num_views=num_views,
    num_det_rows=num_det_rows,
    num_det_channels=num_det_channels,
    use_helical=use_helical,
    helical_pitch=helical_pitch,
    helical_z_range=helical_z_range,
    helical_z_center=helical_z_center,
)

angles = params["angles"]
helical_z_shifts = params.get("helical_z_shifts", None)
source_detector_dist = params.get("source_detector_dist", None)
source_iso_dist = params.get("source_iso_dist", None)

print("\nDemo params summary:")
print(f"  sinogram shape: {sinogram_fan.shape}")
print(f"  angles shape:   {np.asarray(angles).shape}")
if helical_z_shifts is not None:
    z_np = np.asarray(helical_z_shifts)
    print(f"  helical_z_shifts shape: {z_np.shape}  (min={z_np.min():.3f}, max={z_np.max():.3f})")
print(f"  SDD={source_detector_dist}, SID={source_iso_dist}")

# View the sinogram
title = 'Original sinogram \nUse the sliders to change the view or adjust the intensity range.\nRight click the image to see options.'
# mj.slice_viewer(sinogram_fan, data_dicts=params, slice_axis=0, title=title, slice_label='View')

# -------------------------
# Create model for recon
# -------------------------
ct_model = mj.FanBeamModel(
    sinogram_fan.shape,
    angles,
    source_detector_dist=source_detector_dist,
    source_iso_dist=source_iso_dist,
    helical_z_shifts=helical_z_shifts
)

ct_model.set_params(sharpness=sharpness)

print("\nModel parameters:")
ct_model.print_params()

# -------------------------
# Reconstruct
# -------------------------
print("\nStarting recon...")
t0 = time.time()

recon_fan, recon_dict = ct_model.recon(
    sinogram_fan,
    weights=weights,
    max_iterations=15,
    # init_recon=0,  # uncomment to bypass direct_recon
)

max_diff = np.amax(np.abs(phantom - recon_fan))
nrmse = np.linalg.norm(recon_fan - phantom) / np.linalg.norm(phantom)
pct_95 = np.percentile(np.abs(recon_fan - phantom), 95)

elapsed = time.time() - t0
print(f"\nElapsed time for recon: {elapsed:.3f} s")

# ##########################
# Add some notes to include for display and saving
recon_dict['notes'] += 'NRMSE between recon and phantom = {}'.format(nrmse)
recon_dict['notes'] += 'Maximum pixel difference between phantom and recon = {}'.format(max_diff)
recon_dict['notes'] += '95% of recon pixels are within {} of phantom'.format(pct_95)

mj.get_memory_stats()
print('Elapsed time for recon is {:.3f} seconds'.format(elapsed))

# Display results
title = 'Phantom (left) vs VCD Recon (right) \nUse the sliders to change the slice or adjust the intensity range.\nRight click an image to see options.'
# mj.slice_viewer(phantom, recon_fan, data_dicts=[None, recon_dict], title=title)

################# CONE with curved detector #########################
model_type = "cone"

# Detector Type
det_type = 'curved'

# Generate weights array - for an initial reconstruction, use weights = None, then modify if needed to reduce the effect of possibly noisy sinogram entries.
weights = None

# -------------------------
# Generate simulated data
# -------------------------
print("Generating demo data...")
phantom, sinogram_cone, params = mj.generate_demo_data(
    object_type=object_type,
    model_type=model_type,
    num_views=num_views,
    num_det_rows=num_det_rows,
    num_det_channels=num_det_channels,
    use_helical=use_helical,
    helical_pitch=helical_pitch,
    helical_z_range=helical_z_range,
    helical_z_center=helical_z_center,
    detector_type=det_type
)

angles = params["angles"]
helical_z_shifts = params.get("helical_z_shifts", None)
source_detector_dist = params.get("source_detector_dist", None)
source_iso_dist = params.get("source_iso_dist", None)

print("\nDemo params summary:")
print(f"  sinogram shape: {sinogram_cone.shape}")
print(f"  angles shape:   {np.asarray(angles).shape}")
if helical_z_shifts is not None:
    z_np = np.asarray(helical_z_shifts)
    print(f"  helical_z_shifts shape: {z_np.shape}  (min={z_np.min():.3f}, max={z_np.max():.3f})")
print(f"  SDD={source_detector_dist}, SID={source_iso_dist}")

# View the sinogram
title = 'Original sinogram \nUse the sliders to change the view or adjust the intensity range.\nRight click the image to see options.'
# mj.slice_viewer(sinogram_cone, data_dicts=params, slice_axis=0, title=title, slice_label='View')

# -------------------------
# Create model for recon
# -------------------------
ct_model = mj.ConeBeamModel(
    sinogram_cone.shape,
    angles,
    source_detector_dist=source_detector_dist,
    source_iso_dist=source_iso_dist,
    helical_z_shifts=helical_z_shifts,
    detector_type=det_type
)

ct_model.set_params(sharpness=sharpness)

print("\nModel parameters:")
ct_model.print_params()

# -------------------------
# Reconstruct
# -------------------------
print("\nStarting recon...")
t0 = time.time()

recon_cone, recon_dict = ct_model.recon(
    sinogram_cone,
    weights=weights,
    max_iterations=15,
    # init_recon=0,  # uncomment to bypass direct_recon
)

max_diff = np.amax(np.abs(phantom - recon_cone))
nrmse = np.linalg.norm(recon_cone - phantom) / np.linalg.norm(phantom)
pct_95 = np.percentile(np.abs(recon_cone - phantom), 95)

elapsed = time.time() - t0
print(f"\nElapsed time for recon: {elapsed:.3f} s")

# ##########################
# Add some notes to include for display and saving
recon_dict['notes'] += 'NRMSE between recon and phantom = {}'.format(nrmse)
recon_dict['notes'] += 'Maximum pixel difference between phantom and recon = {}'.format(max_diff)
recon_dict['notes'] += '95% of recon pixels are within {} of phantom'.format(pct_95)

mj.get_memory_stats()
print('Elapsed time for recon is {:.3f} seconds'.format(elapsed))

# Display results
title = 'Phantom (left) vs VCD Recon (right) \nUse the sliders to change the slice or adjust the intensity range.\nRight click an image to see options.'
# mj.slice_viewer(phantom, recon_cone, data_dicts=[None, recon_dict], title=title)

mj.slice_viewer(sinogram_fan, sinogram_cone, title='Fan (New class) vs Cone (Using curved flag)')
mj.slice_viewer(recon_fan, recon_cone, title='Fan (New class) vs Cone (Using curved flag)')