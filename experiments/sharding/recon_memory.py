import sys
import os
import time
import csv
import h5py
import mbirjax as mj
import jax
import numpy as np
from mbirjax import generate_3d_shepp_logan_low_dynamic_range

# ignore the TkAgg warning to clean up the output
import warnings
warnings.filterwarnings(
    "ignore",
    message="TkAgg not available. Falling back to Agg."
)

def create_recon_data(num_views, num_det_rows, num_det_channels):

    output_directory = f"/scratch/gautschi/ncardel/recon_mem"
    h5_path = f"{output_directory}/cone_{num_views}_{num_det_rows}_{num_det_channels}_projection_data.h5"
    if os.path.isfile(h5_path):
        return

    start_angle = -np.pi
    end_angle = np.pi
    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    angles = np.linspace(start_angle, end_angle, num_views, endpoint=False)

    source_detector_dist = 4 * num_det_channels
    source_iso_dist = source_detector_dist
    ct_model_for_generation = mj.ConeBeamModel(sinogram_shape, angles,
                                               source_detector_dist=source_detector_dist,
                                               source_iso_dist=source_iso_dist)
    ct_model_for_generation.set_params(use_gpu='projection')

    # Generate phantom
    print('Creating phantom')
    recon_shape = ct_model_for_generation.get_params('recon_shape')
    device = ct_model_for_generation.main_device
    phantom = generate_3d_shepp_logan_low_dynamic_range(recon_shape, device=device)

    # Generate synthetic sinogram data
    print('Creating sinogram')
    sinogram = ct_model_for_generation.forward_project(phantom)
    sinogram = np.asarray(sinogram)

    # save the phantom, sinogram, and params
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("phantom", data=phantom)
        f.create_dataset("sinogram", data=sinogram)
        f.attrs["params"] = ct_model_for_generation.to_file(None)



def recon(num_views, num_det_rows, num_det_channels, output_filepath='output.csv'):

    output_directory = f"/scratch/gautschi/ncardel/recon_mem"
    h5_path = f"{output_directory}/cone_{num_views}_{num_det_rows}_{num_det_channels}_projection_data.h5"
    with h5py.File(h5_path, "r") as f:
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
    row = [ num_views, num_det_rows, num_det_channels, recon_model.transfer_pixel_batch_size ] + [ mem_stats[i]['peak_bytes_in_use'] for i in range(len(mem_stats)) ]
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
        output_filepath = "logs/recon_mem.txt"

    create_recon_data(num_views, num_det_rows, num_det_channels)
    recon(num_views, num_det_rows, num_det_channels, output_filepath=output_filepath)