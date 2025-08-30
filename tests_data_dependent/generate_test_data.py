import numpy as np
import jax
import jax.numpy as jnp
import mbirjax as mj
import pickle


def generate_test_data(model_type, size=32):
    # sinogram shape
    num_views = size
    num_det_rows = size
    num_det_channels = size

    # Choose the geometry type
    object_type = 'shepp-logan'

    # directory
    USER = "ncardel"
    output_directory = f"/depot/bouman/users/{USER}"

    # generate phantom, sinogram (forward projection), and params
    phantom, sinogram, params = mj.generate_demo_data(object_type=object_type,
                                                                     model_type=model_type,
                                                                     num_views=num_views,
                                                                     num_det_rows=num_det_rows,
                                                                     num_det_channels=num_det_channels)

    # params
    angles = params['angles']

    # create back projection model
    if model_type == 'cone':
        source_detector_dist = params['source_detector_dist']
        source_iso_dist = params['source_iso_dist']
        back_projection_model = mj.ConeBeamModel(sinogram.shape, angles,
                                             source_detector_dist=source_detector_dist,
                                             source_iso_dist=source_iso_dist)
    else:
        back_projection_model = mj.ParallelBeamModel(sinogram.shape, angles)

    # perform back projection and reshape into recon
    recon = back_projection_model.back_project(sinogram)
    recon.block_until_ready()
    recon = jax.device_put(recon)  # move to host

    # save the phantom, sinogram, recon, and params
    phantom = np.array(phantom)
    np.save(f"{output_directory}/{model_type}_phantom_{size}.npy", phantom)

    sinogram = np.array(sinogram)
    np.save(f"{output_directory}/{model_type}_sinogram_{size}.npy", sinogram)

    recon = np.array(recon)
    np.save(f"{output_directory}/{model_type}_recon_{size}.npy", recon)

    with open(f"{output_directory}/{model_type}_params_{size}.pkl", "wb") as f:
        pickle.dump(params, f)

    # view the phantom, sinogram, and recon
    mj.slice_viewer(phantom, title=f"{model_type} phantom {size}")
    mj.slice_viewer(sinogram, title=f"{model_type} sinogram {size}", slice_axis=0)
    mj.slice_viewer(recon, title=f"{model_type} recon {size}")


if __name__ == '__main__':
    generate_test_data('cone', size=32)
    generate_test_data('parallel', size=32)
