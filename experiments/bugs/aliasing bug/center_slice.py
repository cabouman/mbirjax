
import numpy as np
import mbirjax as mj


if __name__ == "__main__":

    # #### recon parameters
    sharpness = 4.0

    # Choose the geometry type
    model_type = 'cone'
    object_type = 'cube'

    # Set parameters for the problem size - you can vary these, but if you make num_det_rows very small relative to
    # channels, then the generated phantom may not have an interior.
    num_views = 128
    num_det_rows = 20
    num_det_channels = 128

    source_detector_dist = 12 * num_det_channels
    source_iso_dist = 4 * num_det_channels
    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
    ct_model = mj.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist,
                                source_iso_dist=source_iso_dist)

    ct_model.set_params(sharpness=sharpness, verbose=1)
    ct_model.set_params(delta_voxel=0.28, det_row_offset=0.4)
    phantom = mj.gen_cube_phantom(ct_model.get_params('recon_shape'))
    print('Creating sinogram')
    sinogram = ct_model.forward_project(phantom)
    sinogram = np.asarray(sinogram)
    recon, recon_dict = ct_model.recon(sinogram, max_iterations=5)
    error_sino = ct_model.forward_project(recon) - sinogram
    mj.slice_viewer(recon.transpose((2, 0, 1)), error_sino, data_dicts=[recon_dict, None],
                    slice_axis=0, vmin=-0.001, vmax=0.001,
                    title='Recon top view (left) and error sinogram (right)\nChange slice to see slice-dependent noise associated \nwith horizontal stripes in error sinogram.')

