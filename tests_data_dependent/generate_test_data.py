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

    # get recon shape and partition pixel indices
    recon_shape, granularity = back_projection_model.get_params(['recon_shape', 'granularity'])
    recon_rows, recon_cols, recon_slices = recon_shape
    partitions = mj.gen_set_of_pixel_partitions(recon_shape, granularity)
    pixel_indices = partitions[0][0]

    # perform back projection and reshape into recon
    back_projection = back_projection_model.sparse_back_project(sinogram, pixel_indices)
    back_projection.block_until_ready()
    recon = jnp.zeros((recon_rows * recon_cols, recon_slices)).at[pixel_indices].add(back_projection)
    recon = back_projection_model.reshape_recon(recon)
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
