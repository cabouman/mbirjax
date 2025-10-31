import numpy as np
import pickle
import mbirjax as mj
import time
import h5py
import jax.numpy as jnp

from mbirjax import ModelType, generate_3d_shepp_logan_low_dynamic_range, ObjectType

######################### SETUP ############################

# 1. Set username
USER = "ncardel" # change the user to your username

############################################################

def generate_data(model_type, size, display=False):

    # determine output dir and filepath
    output_directory = f"/scratch/gautschi/{USER}/recon_time"
    h5_path = f"{output_directory}/{model_type}_{size}_projection_data.h5"

    # Set parameters for the problem size
    num_views = size
    num_det_rows = size
    num_det_channels = size

    # Generate demo data
    print('Generating demo data')
    time0 = time.time()

    # Initialize model parameters
    model_type = ModelType(model_type)
    start_angle = -np.pi
    end_angle = np.pi
    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    angles = np.linspace(start_angle, end_angle, num_views, endpoint=False)

    # Initialize model
    if model_type == ModelType.CONE:
        # For cone beam geometry, we need to describe the distances source to detector and source to rotation axis.
        # np.Inf is an allowable value, in which case this is essentially parallel beam
        source_detector_dist = 4 * num_det_channels
        source_iso_dist = source_detector_dist
        ct_model_for_generation = mj.ConeBeamModel(sinogram_shape, angles,
                                                   source_detector_dist=source_detector_dist,
                                                   source_iso_dist=source_iso_dist)
    else:
        ct_model_for_generation = mj.ParallelBeamModel(sinogram_shape, angles)

    # Generate phantom
    print('Creating phantom')
    recon_shape = ct_model_for_generation.get_params('recon_shape')
    device = ct_model_for_generation.main_device
    phantom = generate_3d_shepp_logan_low_dynamic_range(recon_shape, device=device)

    # Generate synthetic sinogram data
    print('Creating sinogram')
    sinogram = ct_model_for_generation.forward_project(phantom)
    sinogram = np.asarray(sinogram)

    elapsed = time.time() - time0
    mj.get_memory_stats()

    print('Elapsed time for generating demo data is {:.3f} seconds'.format(elapsed))

    # save the phantom, sinogram, and params
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("phantom", data=phantom)
        f.create_dataset("sinogram", data=sinogram)
        f.attrs["params"] = ct_model_for_generation.to_file(None)

    if display:
        mj.slice_viewer(phantom, title="Phantom Test")
        mj.slice_viewer(sinogram, title="Sinogram Test")

if __name__ == "__main__":
    for size in [128, 256, 512, 1024]:
        for model_type in ['parallel', 'cone']:
            generate_data(model_type, size)
