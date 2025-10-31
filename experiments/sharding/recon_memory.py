import sys
import os
import time
import csv
import h5py
import mbirjax as mj
import jax

# ignore the TkAgg warning to clean up the output
import warnings
warnings.filterwarnings(
    "ignore",
    message="TkAgg not available. Falling back to Agg."
)

def recon(size, output_filepath='output.csv'):

    output_directory = f"/scratch/gautschi/ncardel/recon_time"
    h5_path = f"{output_directory}/cone_{size}_projection_data.h5"
    with h5py.File(h5_path, "r") as f:
        phantom = f["phantom"][:]
        sinogram = f["sinogram"][:]
        params = f.attrs["params"]

    recon_model = mj.ConeBeamModel.from_file(params)

    print("\nTEST PARAMS:")
    transfer_pixel_batch_size = recon_model.transfer_pixel_batch_size
    print("Transfer pixel batch size:", transfer_pixel_batch_size)
    try:
        print("Device set:", recon_model.sinogram_device.device_set)
    except:
        pass

    weights = mj.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')

    print("\nGPU STARTING MEMORY STATS:")
    mj.get_memory_stats()

    print("\nSTARTING RECON FIRST PASS:")
    recon_model.set_params(use_gpu="automatic")
    recon, _ = recon_model.recon(sinogram,
                                 weights=weights,
                                 max_iterations=2,
                                 stop_threshold_change_pct=0)
    recon.block_until_ready()

    print("\nGPU FINAL MEMORY STATS:")
    mem_stats = mj.get_memory_stats()

    # if the output file doesn't exist then create it
    print("output_filepath:", output_filepath)
    if not os.path.exists(output_filepath):
        with open(output_filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(['num_views', 'num_det_rows', 'num_det_channels', 'transfer_pixel_batch_size', 'gpu0_peak_bytes', 'gpu1_peak_bytes', 'gpu2_peak_bytes', 'gpu3_peak_bytes', 'cpu'])

    # append this test data to the output file
    num_views, num_det_rows, num_det_channels = size, size, size
    row = [ num_views, num_det_rows, num_det_channels, recon_model.transfer_pixel_batch_size ] + [ mem_stats[i]['peak_bytes_in_use'] for i in range(len(mem_stats)) ]
    with open(output_filepath, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

if __name__ == "__main__":

    try:
        size = int(sys.argv[1])
        output_filepath = sys.argv[2]
    except:
        size = 256
        output_filepath = "../output/recon_mem.csv"

    # the sinogram and recon will have shape (size, size, size)
    recon(size, output_filepath=output_filepath)