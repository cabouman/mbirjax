import mbirjax as mj
import numpy as np

if __name__ in {"__main__", "__mp_main__"}:

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
    phantom = phantom[10:-10, :, :]
    a = np.random.rand(2, 200, 300, 250)
    mj.slice_viewer(a[1], 2*a[0], slice_axis=(0, 0))
