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
    recon, recon_params = ct_model_for_generation.recon(sinogram, num_iterations=2)
    ct_model_for_generation.save_recon_to_hdf5('./test.h5', recon, recon_params)
    mj.slice_viewer(None, None, slice_axis=(0, 2))
