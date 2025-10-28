# -*- coding: utf-8 -*-
"""

This code demonstrates the use of the qggmrf denoiser.

"""
import numpy as np
import mbirjax as mj
import matplotlib.pyplot as plt

a = np.arange(3 * 3 * 3).reshape((3, 3, 3))

b = mj.median_filter3d(a, return_min_max=True)
b0 = np.array([[[3., 3., 4.],
                [6., 6., 7.],
                [6., 7., 8.]],
               [[10., 11., 11.],
                [12., 13., 14.],
                [15., 15., 16.]],
               [[18., 19., 20.],
                [19., 20., 20.],
                [22., 23., 23.]]])
b1 = np.array([[[0., 0., 1.],
                [0., 0., 1.],
                [3., 3., 4.]],

               [[0., 0., 1.],
                [0., 0., 1.],
                [3., 3., 4.]],

               [[9., 9., 10.],
                [9., 9., 10.],
                [12., 12., 13.]]])
b2 = np.array([[[13., 14., 14.],
                [16., 17., 17.],
                [16., 17., 17.]],

               [[22., 23., 23.],
                [25., 26., 26.],
                [25., 26., 26.]],

               [[22., 23., 23.],
                [25., 26., 26.],
                [25., 26., 26.]]])
assert np.allclose(b[0], b0)
assert np.allclose(b[1], b1)
assert np.allclose(b[2], b2)

img_in = np.zeros((1, 128, 128))

img_in[:, 32:96, 32: 96] = 1

img_in += 0.2 * np.random.randn(1, 128, 128)
b = mj.median_filter3d(img_in, return_min_max=True)

mj.slice_viewer(img_in, b[0], b[1], b[2], title='Original, median, min, max', slice_axis=0)

# img_in = strain_comps_recon[2]

denoiser = mj.QGGMRFDenoiser(img_in.shape)
img_out, _ = denoiser.denoise(np.copy(img_in), stop_threshold_change_pct=0, max_iterations=100,
                              print_logs=True)

print('Estimated noise = {}'.format(denoiser.get_params('sigma_noise')))
plt.imshow(img_in[0])
plt.show()

plt.imshow(img_out[0])
plt.show()

import time

import numpy as np
import mbirjax as mj

"""**Set the geometry parameters**"""

# Set the experiment parameters - set one of these to be an empty list to use the default values, then vary the other
sharpness_levels = [2, 1, 0, -1]
sigma_denoise_levels = []  # [0.3, 0.2, 0.1, 0.05]

sigma_noise_added = 0.2
num_tiles = 1  # Note that an even number of tiles will lead to the center slice being all zeros, so use the slice slider to see content.

max_iterations = 300
dc_offset = 0.0

indep_var = 'sigma_denoise' if len(sigma_denoise_levels) > 0 else 'sharpness'

# Set parameters for the problem size - you can vary these, but if you make num_det_rows very small relative to
# channels, then the generated phantom may not have an interior.
num_views = 128
num_det_rows = 128
num_det_channels = 128
stop_threshold_change_pct = 0.1

if indep_var == 'sharpness':
    levels = sharpness_levels
    sharpness = True
else:
    levels = sigma_denoise_levels
    sharpness = False

# Get some noisy data
print('Generating data')
np.random.seed(0)

recon_shape = (num_det_channels, num_det_channels, num_det_rows)
phantom = mj.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
phantom = phantom + dc_offset
phantom = np.tile(phantom, (num_tiles, num_tiles, num_tiles))
#
recon_shape = [num_tiles * size for size in recon_shape]
phantom_noisy = phantom + sigma_noise_added * np.random.randn(*recon_shape)

denoiser = mj.QGGMRFDenoiser(phantom.shape)

"""**Do the reconstruction and display the results.**"""
phantoms = [phantom_noisy]
dicts = [None]
nrmse = np.linalg.norm(phantom_noisy - phantom) / np.linalg.norm(phantom)
slice_labels = ['Noise sigma={}, NRMSE={:.3f}: slice'.format(sigma_noise_added, nrmse)]
init_image = phantom_noisy

# ##########################
# Denoise at various levels
elapsed = 0
for level in levels:
    print('Starting recon with {} = {}'.format(indep_var, level))
    if sharpness:
        sharp_level = level
        denoiser.set_params(sharpness=sharp_level)
        sigma_denoise = denoiser.estimate_image_noise_std(phantom_noisy)
    else:
        sharp_level = denoiser.get_params('sharpness')
        sigma_denoise = level
    time0 = time.time()
    phantom_denoised, recon_dict = denoiser.denoise(phantom_noisy, sigma_noise=sigma_denoise, init_image=init_image,
                                                    max_iterations=max_iterations,
                                                    stop_threshold_change_pct=stop_threshold_change_pct)

    elapsed += time.time() - time0
    phantoms.append(phantom_denoised)
    dicts.append(recon_dict)
    nrmse = np.linalg.norm(phantom_denoised - phantom) / np.linalg.norm(phantom)
    slice_labels.append(
        'Sharpness={}, sigma_denoise={:.2f}, \nNRMSE={:.3f}: slice'.format(sharp_level, sigma_denoise, nrmse))

    init_image = phantom_denoised

# ########################### Print out model parameters
denoiser.print_params()
mj.get_memory_stats()
# Display results
title = 'Noisy phantom and denoising with changing parameters'
print('Elapsed time = {:.3f} sec'.format(elapsed))
mj.slice_viewer(*phantoms, data_dicts=dicts, slice_label=slice_labels, title=title, vmin=-0.2, vmax=1.0)
