import sys
import hashlib
import numpy as np
import time
import os

import csv


def back_project(size, gpus=None, output_filepath='output.csv'):

    visible_devices = ",".join([str(x) for x in range(gpus)])
    print(visible_devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    import jax.numpy as jnp
    import mbirjax as mj

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

    mem_stats = mj.get_memory_stats()

    # row = [
    #     size,
    #     gpus,
    #     elapsed,
    #     mem_stats[0]['peak_bytes_in_use'],
    #     mem_stats[1]['peak_bytes_in_use'],
    #     mem_stats[2]['peak_bytes_in_use'],
    #     mem_stats[3]['peak_bytes_in_use'],
    #     mem_stats[4]['peak_bytes_in_use'],
    #     mem_stats[5]['peak_bytes_in_use'],
    #     mem_stats[6]['peak_bytes_in_use'],
    #     mem_stats[7]['peak_bytes_in_use'],
    # ]

    # with open(output_filepath, "a", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(row)

    print('Elapsed time for back projection is {:.3f} seconds'.format(elapsed))

if __name__ == "__main__":

    size=256
    gpus=4
    output_filepath = '../output/output.csv'

    # size = int(sys.argv[1])
    # gpus = int(sys.argv[2])
    # output_filepath = sys.argv[3]

    print("size:", size, "gpus:", gpus)

    back_project(size, gpus=gpus, output_filepath=output_filepath)

    print()