import mbirjax as mj
import numpy as np

if __name__ == "__main__":

    num_views = 64
    num_det_rows = 40
    num_det_channels = 128

    start_angle = -np.pi
    end_angle = np.pi

    # Initialize sinogram
    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    angles = np.linspace(start_angle, end_angle, num_views, endpoint=False)
    ct_model_for_generation = mj.ParallelBeamModel(sinogram_shape, angles)

    phantom = ct_model_for_generation.gen_modified_3d_sl_phantom()
    sinogram = ct_model_for_generation.forward_project(phantom)
    recon_fbp = ct_model_for_generation.direct_recon(sinogram)
    recon, recon_params = ct_model_for_generation.recon(sinogram)
    ct_model_for_generation.save_recon_to_hdf5(filepath='./test_fbp.h5', recon=recon_fbp)
    ct_model_for_generation.save_recon_to_hdf5(filepath='./test_mbir.h5', recon=recon, recon_params=recon_params)

    # Test viewer
    mj.slice_viewer(phantom, phantom + 0.1 * np.random.rand(*phantom.shape) - 0.05, slice_axis=(0, 0))
    mj.slice_viewer(None, None, vmin=0.0, vmax=1.0, slice_axis=(2, 2))
