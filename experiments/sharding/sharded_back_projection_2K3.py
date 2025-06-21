import hashlib
import numpy as np
import time
import jax.numpy as jnp
import mbirjax as mj
import jax

mj.get_memory_stats()

# sinogram shape
num_views = 2000
num_det_rows = 2000
num_det_channels = 2000
sinogram_shape = (num_views, num_det_rows, num_det_channels)

# angles
start_angle = -np.pi * (1/2)
end_angle = np.pi * (1/2)
angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

# recon model
back_projection_model = mj.ParallelBeamModel(sinogram_shape, angles)
back_projection_model.set_params(sharpness=0.0)

# Print out model parameters
back_projection_model.print_params()

# Generate simulated data
sinogram = jnp.ones_like(2, shape=sinogram_shape, device=back_projection_model.sinogram_device)

# get partition pixel indices
recon_shape, granularity = back_projection_model.get_params(['recon_shape', 'granularity'])
partitions = mj.gen_set_of_pixel_partitions(recon_shape, granularity)
pixel_indices = partitions[0][0]
pixel_indices = jax.device_put(pixel_indices, device=back_projection_model.worker)

############################### SHARDED ###############################
print("Starting sharded back projection")

time0 = time.time()
sharded_back_projection_2k3 = back_projection_model.sparse_back_project(sinogram, pixel_indices,
                                                                    output_device=back_projection_model.main_device)
sharded_back_projection_2k3.block_until_ready()
elapsed = time.time() - time0

mj.get_memory_stats()
print('Elapsed time for back projection is {:.3f} seconds'.format(elapsed))

# view back projection
recon_rows, recon_cols, recon_slices = recon_shape
pixel_indices = jax.device_put(pixel_indices, device=back_projection_model.main_device)
sharded_back_projection_2k3 = jnp.zeros((recon_rows*recon_cols,recon_slices), device=back_projection_model.main_device).at[pixel_indices].add(sharded_back_projection_2k3)
sharded_back_projection_2k3 = back_projection_model.reshape_recon(sharded_back_projection_2k3)
mj.slice_viewer(sharded_back_projection_2k3, slice_axis=0, title='Sharded Back Projection 2K^3')

# # save back projection
# sharded_back_projection_2k3 = np.array(sharded_back_projection_2k3)
# hash_digest = hashlib.sha256(sharded_back_projection_2k3.tobytes()).hexdigest()
# file_path = f"output/sharded_back_projection_2k3{hash_digest[:8]}.npy"
# print(f"Sharded back projection being saved to file {file_path}")
# np.save(file_path, sharded_back_projection_2k3)
