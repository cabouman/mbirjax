import sys
import os
import time
import numpy as np
import csv
import jax.numpy as jnp
import mbirjax as mj

def back_project(num_views, num_det_rows, num_det_channels, output_filepath='output.csv'):

    # sinogram shape
    sinogram_shape = (num_views, num_det_rows, num_det_channels)

    # angles
    start_angle = -np.pi * (1/2)
    end_angle = np.pi * (1/2)
    angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

    # recon model
    source_detector_dist = 4 * num_det_channels
    source_iso_dist = source_detector_dist
    back_projection_model = mj.ConeBeamModel(sinogram_shape,
                                             angles,
                                             source_detector_dist=source_detector_dist,
                                             source_iso_dist=source_iso_dist)
    back_projection_model.set_params(sharpness=0.0, use_gpu='automatic')

    # Print out model parameters
    print("MODEL PARAMS:")
    back_projection_model.print_params()

    # Generate simulated data
    sinogram = jnp.ones_like(2, shape=sinogram_shape, device=back_projection_model.sinogram_device)

    # get partition pixel indices
    recon_shape, granularity = back_projection_model.get_params(['recon_shape', 'granularity'])
    partitions = mj.gen_set_of_pixel_partitions(recon_shape, granularity)
    pixel_indices = partitions[0][0]

    print("\nTEST PARAMS:")
    print("Device set:", back_projection_model.sinogram_device.device_set)
    print("Transfer pixel batch size:",  back_projection_model.transfer_pixel_batch_size)

    print("\nGPU STARTING MEMORY STATS:")
    mj.get_memory_stats()

    print("\nSTARTING BACK PROJECTION FIRST PASS:")
    back_projection = back_projection_model.sparse_back_project(sinogram, pixel_indices, output_device=back_projection_model.main_device)
    back_projection.block_until_ready()

    print("\nGPU FINAL MEMORY STATS:")
    mem_stats = mj.get_memory_stats()

    # if the output file doesn't exist then create it
    if not os.path.exists(output_filepath):
        with open(output_filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(['num_views', 'num_det_rows', 'num_det_channels', 'transfer_pixel_batch_size', 'gpu0_peak_bytes', 'gpu1_peak_bytes', 'gpu2_peak_bytes', 'gpu3_peak_bytes', 'cpu'])

    # append this test data to the output file
    row = [ num_views, num_det_rows, num_det_channels, back_projection_model.transfer_pixel_batch_size ] + [ mem_stats[i]['peak_bytes_in_use'] for i in range(len(mem_stats)) ]
    with open(output_filepath, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

if __name__ == "__main__":

    try:
        num_views = int(sys.argv[1])
        num_det_rows = int(sys.argv[2])
        num_det_channels = int(sys.argv[3])
        output_filepath = sys.argv[4]
    except:
        num_views = 256
        num_det_rows = 256
        num_det_channels = 256
        output_filepath = "logs/back_projection_memory.txt"

    back_project(num_views, num_det_rows, num_det_channels, output_filepath=output_filepath)