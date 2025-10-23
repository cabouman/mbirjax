import sys
import os
import time
import csv
import h5py
import jax

# ignore the TkAgg warning to clean up the output
import warnings
warnings.filterwarnings(
    "ignore",
    message="TkAgg not available. Falling back to Agg."
)

def recon(size, num_gpus, model_type, output_filepath='output.csv'):

    # set the visible cuda devices to 0 to num_gpus-1
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in range(num_gpus)])

    # import jax after setting the visible cuda devices
    import jax.numpy as jnp
    import mbirjax as mj

    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.90'

    output_directory = f"/scratch/gautschi/ncardel/recon_time"
    h5_path = f"{output_directory}/{model_type}_{size}_projection_data.h5"
    with h5py.File(h5_path, "r") as f:
        phantom = f["phantom"][:]
        sinogram = f["sinogram"][:]
        params = f.attrs["params"]

    if model_type == 'cone':
        recon_model = mj.ConeBeamModel.from_file(params)
    else:
        recon_model = mj.ParallelBeamModel.from_file(params)

    print("\nTEST PARAMS:")
    transfer_pixel_batch_size = recon_model.transfer_pixel_batch_size
    print("Device set:", recon_model.sinogram_device.device_set)
    print("Transfer pixel batch size:", transfer_pixel_batch_size)

    print("\nGPU STARTING MEMORY STATS:")
    mj.get_memory_stats()

    print("\nSTARTING RECON FIRST PASS:")
    recon_model.set_params(use_gpu="automatic")
    recon, _ = recon_model.recon(sinogram,
                                 max_iterations=8,
                                 stop_threshold_change_pct=0)
    recon.block_until_ready()
    recon = jax.device_put(recon)

    # time the second pass to remove time for jitting the functions
    print("\nSTARTING RECON SECOND PASS:")
    time0 = time.time()
    recon, _ = recon_model.recon(sinogram,
                                 max_iterations=8,
                                 stop_threshold_change_pct=0)
    recon.block_until_ready()
    elapsed = time.time() - time0
    recon = jax.device_put(recon)

    print("\nGPU FINAL MEMORY STATS:")
    mem_stats = mj.get_memory_stats()

    # if the output file doesn't exist then create it
    if not os.path.exists(output_filepath):
        with open(output_filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(['size', 'model_type', 'transfer_pixel_batch_size', 'num_gpus', 'elapsed', 'gpu0_peak_bytes', 'gpu1_peak_bytes', 'gpu2_peak_bytes', 'gpu3_peak_bytes', 'gpu4_peak_bytes', 'gpu5_peak_bytes', 'gpu6_peak_bytes', 'gpu7_peak_bytes'])

    # append this test data to the output file
    row = [ size, model_type, transfer_pixel_batch_size, num_gpus, elapsed ] + [ mem_stats[i]['peak_bytes_in_use'] for i in range(num_gpus) ]
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
        output_filepath = "../output/recon.csv"

    # the sinogram and recon will have shape (size, size, size)
    recon(size, num_gpus, model_type, output_filepath=output_filepath)