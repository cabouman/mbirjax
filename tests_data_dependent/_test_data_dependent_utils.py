import jax
import mbirjax as mj
import h5py
import tarfile
import os
import hashlib

def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""): h.update(b)
    return h.hexdigest()

def generate_projection_test_data(model_type, size=32):
    # sinogram shape
    num_views = size
    num_det_rows = size
    num_det_channels = size

    # Choose the geometry type
    object_type = 'shepp-logan'

    # directory
    output_directory = f"/depot/bouman/data/unit_test_data"

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
    recon = jax.device_get(recon)  # move to host

    h5_path = f"{output_directory}/{model_type}_{size}_projection_data.h5"
    tar_path = f"{output_directory}/{model_type}_{size}_projection_data.tgz"

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("phantom", data=phantom)
        f.create_dataset("sinogram", data=sinogram)
        f.create_dataset("recon", data=recon)
        f.attrs["params"] = back_projection_model.to_file(None)

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(h5_path, arcname=os.path.basename(h5_path))

    # view the phantom, sinogram, and recon
    mj.slice_viewer(phantom, title=f"{model_type} phantom {size}")
    mj.slice_viewer(sinogram, title=f"{model_type} sinogram {size}", slice_axis=0)
    mj.slice_viewer(recon, title=f"{model_type} recon {size}")

def generate_recon_test_data(model_type, size=32):
    # sinogram shape
    num_views = size
    num_det_rows = size
    num_det_channels = size

    # Choose the geometry type
    object_type = 'shepp-logan'

    # directory
    output_directory = f"/depot/bouman/data/unit_test_data"

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
        recon_model = mj.ConeBeamModel(sinogram.shape, angles,
                                             source_detector_dist=source_detector_dist,
                                             source_iso_dist=source_iso_dist)
    else:
        recon_model = mj.ParallelBeamModel(sinogram.shape, angles)

    # perform back projection and reshape into recon
    recon, _ = recon_model.recon(sinogram, max_iterations=15, stop_threshold_change_pct=0)
    recon = jax.device_get(recon)  # move to host

    h5_path = f"{output_directory}/{model_type}_{size}_recon_data.h5"
    tar_path = f"{output_directory}/{model_type}_{size}_recon_data.tgz"

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("phantom", data=phantom)
        f.create_dataset("sinogram", data=sinogram)
        f.create_dataset("recon", data=recon)
        f.attrs["params"] = recon_model.to_file(None)

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(h5_path, arcname=os.path.basename(h5_path))

    # view the phantom, sinogram, and recon
    mj.slice_viewer(phantom, title=f"{model_type} phantom {size}")
    mj.slice_viewer(sinogram, title=f"{model_type} sinogram {size}", slice_axis=0)
    mj.slice_viewer(recon, title=f"{model_type} recon {size}")

if __name__ == '__main__':
    generate_projection_test_data('cone', size=32)
    generate_projection_test_data('parallel', size=32)
    generate_recon_test_data('cone', size=32)
    generate_recon_test_data('parallel', size=32)
