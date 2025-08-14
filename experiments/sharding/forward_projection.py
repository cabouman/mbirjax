import sys
import os
import time
import numpy as np
import csv

# ignore the TkAgg warning to clean up the output
import warnings
warnings.filterwarnings(
    "ignore",
    message="TkAgg not available. Falling back to Agg."
)

def forward_project(size, num_gpus, model_type, output_filepath='output.csv'):

    # set the visible cuda devices to 0 to num_gpus-1
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in range(num_gpus)])

    # import jax after setting the visible cuda devices
    import jax.numpy as jnp
    import mbirjax as mj

    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.90'

    # sinogram shape
    num_views, num_det_rows, num_det_channels = size, size, size
    sinogram_shape = (num_views, num_det_rows, num_det_channels)

    # angles
    start_angle = -np.pi * (1/2)
    end_angle = np.pi * (1/2)
    angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

    # recon model
    forward_projection_model = mj.ParallelBeamModel(sinogram_shape, angles)
    forward_projection_model.set_params(sharpness=0.0, use_gpu='automatic')

    # recon model
    if model_type == 'cone':
        source_detector_dist = 4 * num_det_channels
        source_iso_dist = source_detector_dist
        forward_projection_model = mj.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist,
                                    source_iso_dist=source_iso_dist)
    else:
        forward_projection_model = mj.ParallelBeamModel(sinogram_shape, angles)

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
    transfer_pixel_batch_size = forward_projection_model.transfer_pixel_batch_size
    pixel_indices_size = pixel_indices.size
    print("Device set:", forward_projection_model.sinogram_device.device_set)
    print("Transfer pixel batch size:", transfer_pixel_batch_size)
    print("Pixel indices size:", pixel_indices_size)

    print("\nGPU STARTING MEMORY STATS:")
    mj.get_memory_stats()

    print("\nSTARTING FORWARD PROJECTION FIRST PASS:")
    voxel_values = forward_projection_model.get_voxels_at_indices(recon, pixel_indices)
    forward_projection = forward_projection_model.sparse_forward_project(voxel_values, pixel_indices, output_device=forward_projection_model.sinogram_device)
    forward_projection.block_until_ready()

    # time the second pass to remove time for jitting the functions
    print("\nSTARTING FORWARD PROJECTION SECOND PASS:")
    time0 = time.time()
    forward_projection = forward_projection_model.sparse_forward_project(voxel_values, pixel_indices, output_device=forward_projection_model.sinogram_device)
    forward_projection.block_until_ready()
    elapsed = time.time() - time0

    print("\nGPU FINAL MEMORY STATS:")
    mem_stats = mj.get_memory_stats()

    # if the output file doesn't exist then create it
    if not os.path.exists(output_filepath):
        with open(output_filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(['size', 'model_type', 'transfer_pixel_batch_size', 'pixel_indices_size', 'num_gpus', 'elapsed', 'gpu0_peak_bytes', 'gpu1_peak_bytes', 'gpu2_peak_bytes', 'gpu3_peak_bytes', 'gpu4_peak_bytes', 'gpu5_peak_bytes', 'gpu6_peak_bytes', 'gpu7_peak_bytes'])

    # append this test data to the output file
    row = [ size, model_type, transfer_pixel_batch_size, pixel_indices_size, num_gpus, elapsed ] + [ mem_stats[i]['peak_bytes_in_use'] for i in range(num_gpus) ]
    with open(output_filepath, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    print('\nELAPSED TIME: {:.3f} seconds'.format(elapsed))

if __name__ == "__main__":

    try:
        size = int(sys.argv[1])
        num_gpus = int(sys.argv[2])
        model_type = sys.argv[3]
        output_filepath = sys.argv[4]
    except:
        size = 256
        num_gpus = 2
        model_type = 'cone'
        output_filepath = "../output/forward_project.csv"

    # the sinogram and recon will have shape (size, size, size)
    forward_project(size, num_gpus, model_type, output_filepath=output_filepath)