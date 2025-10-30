import sys
import os
import time
import numpy as np
import csv
import jax.numpy as jnp
import mbirjax as mj

def forward_project(size, output_filepath='output.csv'):

    # sinogram shape
    num_views, num_det_rows, num_det_channels = size, int(size*3/2), int(size*5/4)
    sinogram_shape = (num_views, num_det_rows, num_det_channels)

    # angles
    start_angle = -np.pi * (1/2)
    end_angle = np.pi * (1/2)
    angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

    # recon model
    source_detector_dist = 4 * num_det_channels
    source_iso_dist = source_detector_dist
    forward_projection_model = mj.ConeBeamModel(sinogram_shape,
                                             angles,
                                             source_detector_dist=source_detector_dist,
                                             source_iso_dist=source_iso_dist)
    forward_projection_model.set_params(sharpness=0.0, use_gpu='automatic')

    # Print out model parameters
    print("MODEL PARAMS:")
    forward_projection_model.print_params()


    # get partition pixel indices
    recon_shape, granularity = forward_projection_model.get_params(['recon_shape', 'granularity'])
    partitions = mj.gen_set_of_pixel_partitions(recon_shape, granularity)
    pixel_indices = partitions[0][0]

    # Generate simulated data
    recon = jnp.ones_like(2, shape=recon_shape, device=forward_projection_model.main_device)

    print("\nTEST PARAMS:")
    print("Device set:", forward_projection_model.sinogram_device.device_set)
    print("Transfer pixel batch size:",  forward_projection_model.transfer_pixel_batch_size)

    print("\nGPU STARTING MEMORY STATS:")
    mj.get_memory_stats()

    print("\nSTARTING FORWARD PROJECTION FIRST PASS:")
    voxel_values = forward_projection_model.get_voxels_at_indices(recon, pixel_indices)
    forward_projection = forward_projection_model.sparse_forward_project(voxel_values, pixel_indices, output_device=forward_projection_model.sinogram_device)
    forward_projection.block_until_ready()

    print("\nGPU FINAL MEMORY STATS:")
    mem_stats = mj.get_memory_stats()

    # if the output file doesn't exist then create it
    if not os.path.exists(output_filepath):
        with open(output_filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(['num_views', 'num_det_rows', 'num_det_channels', 'transfer_pixel_batch_size', 'gpu0_peak_bytes', 'gpu1_peak_bytes', 'gpu2_peak_bytes', 'gpu3_peak_bytes', 'cpu'])

    # append this test data to the output file
    row = [ num_views, num_det_rows, num_det_channels, forward_projection_model.transfer_pixel_batch_size ] + [ mem_stats[i]['peak_bytes_in_use'] for i in range(len(mem_stats)) ]
    with open(output_filepath, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

if __name__ == "__main__":

    size = 1024
    output_filepath = "../output/forward_project_mem.txt"

    # the sinogram and recon will have shape (size, size, size)
    forward_project(size, output_filepath=output_filepath)