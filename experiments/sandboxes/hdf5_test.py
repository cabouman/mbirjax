"""
Python script to test hdf5 export/import routines.
"""
import numpy as np
import time
import jax.numpy as jnp
import mbirjax as mj
import mbirjax.preprocess as mjp
import mbirjax.utilities as mju

"""**Set the geometry parameters**"""

# Choose the geometry type
geometry_type = 'cone'  # 'cone' or 'parallel'

# Set parameters for the problem size - you can vary these, but if you make num_det_rows very small relative to
# channels, then the generated phantom may not have an interior.
num_views = 128
num_det_rows = 128
num_det_channels = 128

# For cone beam geometry, we need to describe the distances source to detector and source to rotation axis.
# np.Inf is an allowable value, in which case this is essentially parallel beam.
source_detector_dist = 4 * num_det_channels
source_iso_dist = source_detector_dist

# Set parameters for viewing angle.
start_angle = -np.pi
end_angle = np.pi

"""**Data generation:** For demo purposes, we create a phantom and then project it to create a sinogram.

Note:  the sliders on the viewer won't work in notebook form.  For that you'll need to run the python code with an interactive matplotlib backend, typcially using the command line or a development environment like Spyder or Pycharm to invoke python.  

"""

# Initialize sinogram
sinogram_shape = (num_views, num_det_rows, num_det_channels)
angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

if geometry_type == 'cone':
    ct_model_for_generation = mj.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)
elif geometry_type == 'parallel':
    ct_model_for_generation = mj.ParallelBeamModel(sinogram_shape, angles)
else:
    raise ValueError('Invalid geometry type.  Expected cone or parallel, got {}'.format(geometry_type))

# Generate 3D Shepp Logan phantom
print('Creating phantom')
phantom = ct_model_for_generation.gen_modified_3d_sl_phantom()

# Generate synthetic sinogram data
print('Creating sinogram')
sinogram = ct_model_for_generation.forward_project(phantom)
sinogram = np.asarray(sinogram)

# View sinogram
title = 'Original sinogram \nUse the sliders to change the view or adjust the intensity range.'
mj.slice_viewer(sinogram, slice_axis=0, title=title, slice_label='View')

"""**Initialize for the reconstruction**"""

# ####################
# Initialize the model for reconstruction.
if geometry_type == 'cone':
    ct_model_for_recon = mj.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)
elif geometry_type == 'parallel':
    ct_model_for_recon = mj.ParallelBeamModel(sinogram_shape, angles)
else:
    raise ValueError('Invalid geometry type.  Expected cone or parallel, got {}'.format(geometry_type))

# Print out model parameters
ct_model_for_recon.print_params()

"""**Do the reconstruction and display the results.**"""

# ##########################
# Perform FBP/FDK reconstruction
if geometry_type == 'cone':
    print("Starting FDK recon")
    time0 = time.time()
    recon = ct_model_for_recon.fdk_recon(sinogram, filter_name="ramp")
else:
    print("Starting FBP recon")
    time0 = time.time()
    recon = ct_model_for_recon.fbp_recon(sinogram, filter_name="ramp")

recon.block_until_ready()
elapsed = time.time() - time0
# ##########################

# ##########################
# Test HDF5 import/export functions
mju.export_recon_to_hdf5(filename='../data/recon.h5', recon=recon, recon_description="Test FDK reconstruction")
data = mju.import_recon_from_hdf5('./data/recon.h5')
recon2 = data['recon']
print(data['recon'].shape)
print(data['recon_description'])
print(data['delta_pixel_image'])
print(data['alu_description'])

# Display results
#mj.slice_viewer(recon, recon2, title="Original Recon (left) vs HDF5 Recon (right)")

mju.HDF5Browser()