import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mbirjax
import time
import pprint

# **Set experiment voxel model**
experiment_input_image = '3D_sl'
# select from "impulse" or "3D_sl" for Shepp Logan phantom

# **Set Geometry Parameters**
geometry_type = 'cone'  # Choose 'parallel' or 'cone'

num_views = 64
num_det_rows = 40
num_det_channels = 128
source_detector_dist = 4 * num_det_channels
source_iso_dist = source_detector_dist

if geometry_type == 'cone':
   detector_cone_angle = 2 * np.arctan2(num_det_channels / 2, source_detector_dist)
else:
   detector_cone_angle = 0

start_angle = -(np.pi + detector_cone_angle) * (1 / 2) * 2
end_angle = (np.pi + detector_cone_angle) * (1 / 2) * 2

print('start and end angles:')
print(start_angle)
print(end_angle)

angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

# **Initialize the CT Model**
if geometry_type == 'cone':
   ct_model = mbirjax.ConeBeamModel(
       (num_views, num_det_rows, num_det_channels),
       angles,
       source_detector_dist=source_detector_dist,
       source_iso_dist=source_iso_dist,
   )
elif geometry_type == 'parallel':
   ct_model = mbirjax.ParallelBeamModel(
       (num_views, num_det_rows, num_det_channels),
       angles
   )
else:
   raise ValueError('Invalid geometry type.')


# **Create Impulse Image**
def create_impulse_image(shape, position):
   """
   Creates a 3D impulse image with one pixel set to 1 and the rest set to 0.

   Parameters:
   - shape (tuple of int): Shape of the 3D image (depth, height, width).
   - position (tuple of int): Position of the impulse (z, y, x).

   Returns:
   - numpy.ndarray: 3D array with an impulse at the specified position.
   """
   # Initialize a 3D array with zeros
   image = np.zeros(shape, dtype=np.float32)

   # Set the specified pixel to 1
   image[position] = 1

   return image

voxel_grid_shape = (128, 128, 40)

# Create the impulse image
impulse_position = tuple(dim // 2 for dim in voxel_grid_shape)  # Center of the grid
#impulse_position = (31,31,20)

print('impulse_position', impulse_position)
epsilon_i = create_impulse_image(voxel_grid_shape, impulse_position)

# Find the coordinates where the value equals 1
coords = np.where(epsilon_i == 1)  # For JAX arrays, use jnp.where
z, y, x = coords[0][0], coords[1][0], coords[2][0]  # Extract the first occurrence

print(f"Coordinates of value 1: z={z}, y={y}, x={x}")

# Display the corresponding slices
# Show the slice along the z-axis (depth)
plt.imshow(epsilon_i[z, :, :], cmap='gray')
plt.title(f'Slice along z-axis at z={z}')
plt.colorbar()
plt.show()

# Convert impulse image to JAX array
epsilon_i = jnp.array(epsilon_i)
mbirjax.slice_viewer(epsilon_i, slice_axis=0, title='epsilon_i', slice_label='View')

# Initialize sinogram
sinogram_shape = (num_views, num_det_rows, num_det_channels)
angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

if geometry_type == 'cone':
   ct_model_for_generation = mbirjax.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)
elif geometry_type == 'parallel':
   ct_model_for_generation = mbirjax.ParallelBeamModel(sinogram_shape, angles)
else:
   raise ValueError('Invalid geometry type.  Expected cone or parallel, got {}'.format(geometry_type))


## **Compute Forward Projection for a Single View**
view_index = z  # Select a single view
y_v = ct_model.forward_project(epsilon_i)  # Forward projection for all views
y_v_single_view = y_v[view_index, :, :]  # Extract the sinogram for the selected view

# Step 3: Find the view_index where the projection is non-zero
view_indices = []
for v in range(y_v.shape[0]):  # Loop through all views
   if jnp.any(y_v[v, :, :] > 0):  # Check if there is any non-zero value in the projection
       view_indices.append(v)

print(f"View indices where the voxel projects: {view_indices}")


# **Plot the Resulting Projection for the Single View**
plt.imshow(y_v_single_view, cmap='gray')
plt.title(f'Projection for View Index {view_index}, {geometry_type.capitalize()} Beam')
plt.colorbar(label='Projection Intensity')
plt.xlabel('Detector Channel')
plt.ylabel('Detector Row')
plt.show()

# **Sum of Sinogram Entries for the View**
sum_v = jnp.sum(y_v_single_view)
print(f'Sum of sinogram entries for view {view_index}: {sum_v}')

# **Compute Forward Projection and Sum of Sinogram Entries**
sinogram_sums = []

for v, angle in enumerate(angles):
   # Forward projection for all views
   y_v = ct_model.forward_project(epsilon_i)

   # Extract the sinogram for the current view
   y_v_single_view = y_v[v, :, :]

   # Sum of sinogram entries for the current view
   sum_v = jnp.sum(y_v_single_view)
   sinogram_sums.append(sum_v)

# **Plot Sum of Sinogram Entries vs View Angle**
plt.plot(angles, sinogram_sums, marker='o')
plt.xlabel('View Angle (radians)')
plt.ylabel('Sum of Sinogram Entries')
plt.title(f'Sum of Sinogram Entries vs View Angle ({geometry_type.capitalize()} Beam)')
plt.grid()
plt.show()





if (experiment_input_image == "impulse"):
    sinogram = y_v
    phantom = epsilon_i

else:
    # Generate 3D Shepp Logan phantom
    print("Creating phantom", end="\n\n")
    phantom = ct_model.gen_modified_3d_sl_phantom()

    # Generate sinogram from phantom
    print("Creating sinogram", end="\n\n")
    sinogram = ct_model.forward_project(phantom)

# View sinogram
title = "Original sinogram \nUse the sliders to change the view or adjust the intensity range."
mbirjax.slice_viewer(sinogram, slice_axis=0, title=title, slice_label="View")


################################################################################
# Reconstruction starts here
################################################################################

# Perform FDK recon for "cone" beam, and FBP for "parallel" beam
print("Starting recon")
time0 = time.time()
filter_name = "ramp"
if(geometry_type=='cone'):
    recon = ct_model_for_generation.fdk_recon(sinogram, filter_name=filter_name)

elif(geometry_type=='parallel'):
    recon = ct_model_for_generation.fbp_recon(sinogram, filter_name=filter_name)

recon.block_until_ready()
elapsed = time.time() - time0
print(f"Elapsed time for recon is {elapsed} seconds", end="\n\n")

# Compute descriptive statistics about recon result
max_diff = np.amax(np.abs(phantom - recon))
nrmse = np.linalg.norm(recon - phantom) / np.linalg.norm(phantom)
pct_95 = np.percentile(np.abs(recon - phantom), 95)
print(f"NRMSE between recon and phantom = {nrmse}")
print(f"Maximum pixel difference between phantom and recon = {max_diff}")
print(f"95% of recon pixels are within {pct_95} of phantom")

# Get stats on memory usage
mbirjax.get_memory_stats()

# Display results
title = f"Phantom (left) vs filtered back projection (FDK, right). Filter used: {filter}. \nUse the sliders to change the slice or adjust the intensity range."
mbirjax.slice_viewer(phantom, recon, title=title)
