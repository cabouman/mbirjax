import sys
import hashlib
import numpy as np
import time
import jax.numpy as jnp
import mbirjax as mj
import jax

def back_project(size, gpus=None):

    # sinogram shape
    num_views = size
    num_det_rows = size
    num_det_channels = size
    sinogram_shape = (num_views, num_det_rows, num_det_channels)

    # angles
    start_angle = -np.pi * (1/2)
    end_angle = np.pi * (1/2)
    angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

    # recon model
    back_projection_model = mj.ParallelBeamModel(sinogram_shape, angles)
    back_projection_model.set_params(sharpness=0.0)
    back_projection_model.force_num_gpus = gpus
    back_projection_model.set_devices_and_batch_sizes()

    # Print out model parameters
    back_projection_model.print_params()

    # Generate simulated data
    sinogram = jnp.ones_like(2, shape=sinogram_shape, device=back_projection_model.sinogram_device)

    # get partition pixel indices
    recon_shape, granularity = back_projection_model.get_params(['recon_shape', 'granularity'])
    partitions = mj.gen_set_of_pixel_partitions(recon_shape, granularity)
    pixel_indices = partitions[0][0]
    print("pixel_indices.shape", pixel_indices.shape)

    mj.get_memory_stats()

    print("Starting back projection")
    back_projection = back_projection_model.sparse_back_project(sinogram, pixel_indices, output_device=back_projection_model.main_device)
    back_projection.block_until_ready()

    # time the second pass to remove time for jitting the functions
    time0 = time.time()
    back_projection = back_projection_model.sparse_back_project(sinogram, pixel_indices, output_device=back_projection_model.main_device)
    back_projection.block_until_ready()
    elapsed = time.time() - time0

    mj.get_memory_stats()
    print('Elapsed time for back projection is {:.3f} seconds'.format(elapsed))

if __name__ == "__main__":


    size = int(sys.argv[1])
    gpus = int(sys.argv[2])

    print("size:", size, "gpus:", gpus)

    back_project(size, gpus=gpus)