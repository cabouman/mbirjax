import hashlib
import numpy as np
import time
import jax.numpy as jnp
import mbirjax as mj
import jax

mj.get_memory_stats()

# sinogram shape
num_views = 100
num_det_rows = 100
num_det_channels = 100
sinogram_shape = (num_views, num_det_rows, num_det_channels)

# angles
start_angle = -np.pi * (1/2)
end_angle = np.pi * (1/2)

# Generate simulated data
_, sinogram, params = mj.generate_demo_data(object_type='shepp-logan', model_type='parallel',
                                                  num_views=num_views, num_det_rows=num_det_rows,
                                                  num_det_channels=num_det_channels)
angles = params['angles']

# recon model
back_projection_model = mj.ParallelBeamModel(sinogram_shape, angles)
sinogram = jax.device_put(sinogram, device=back_projection_model.sinogram_device)

# Print out model parameters
back_projection_model.print_params()

# get partition pixel indices
recon_shape, granularity = back_projection_model.get_params(['recon_shape', 'granularity'])
pixel_indices = mj.gen_full_indices(recon_shape)
pixel_indices = jax.device_put(pixel_indices, device=back_projection_model.worker)

############################### CONTROL ###############################
print("Starting control back projection")

time0 = time.time()
control_back_projection = back_projection_model.sparse_back_project(sinogram, pixel_indices,
                                                                    output_device=back_projection_model.main_device)
control_back_projection.block_until_ready()
elapsed = time.time() - time0

mj.get_memory_stats()
print('Elapsed time for back projection is {:.3f} seconds'.format(elapsed))

# view back projection
recon_rows, recon_cols, recon_slices = recon_shape
control_back_projection = jax.device_put(control_back_projection, pixel_indices.device)
row_index, col_index = jnp.unravel_index(pixel_indices, recon_shape[:2])
recon = jnp.zeros(recon_shape, device=pixel_indices.device)
recon = recon.at[row_index, col_index].set(control_back_projection)
mj.slice_viewer(recon, slice_axis=2, title='Control Back Projection')

# save back projection
control_back_projection = np.array(control_back_projection)
hash_digest = hashlib.sha256(control_back_projection.tobytes()).hexdigest()
print("hash_digest", hash_digest)
file_path = f"output/control_back_projection_{hash_digest[:8]}.npy"
print(f"Control back projection being saved to file {file_path}")
np.save(file_path, control_back_projection)
