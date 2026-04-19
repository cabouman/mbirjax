import os
import matplotlib.pyplot as plt
import numpy as np
import mbirjax as mj


if __name__ == "__main__":

    # Get the data
    data_directory = '/depot/bouman/data/ORNL/hsnt/dinosaur'
    angles = np.load(os.path.join(data_directory, 'angles_radian.npy'))
    sinogram = mj.preprocess.read_tif_stack_dir(data_directory + '/attenuation_data')

    start_slice = 400
    num_slices = 20
    downsample_factor = 1  # Channel downsampling

    end_slice = start_slice + num_slices

    s = sinogram[:, start_slice:end_slice, ::downsample_factor]
    # plt.hist(s.flatten(), bins=400)
    # plt.show()
    mj.slice_viewer(s, slice_axis=0, title='Sinogram, downsampled by {}'.format(downsample_factor))

    ct_model = mj.ParallelBeamModel(s.shape, angles)
    recon0, recon_dict0 = ct_model.recon(s)

    mj.slice_viewer(recon0, slice_axis=2, title='Reconstruction, downsampled by {}'.format(downsample_factor))

