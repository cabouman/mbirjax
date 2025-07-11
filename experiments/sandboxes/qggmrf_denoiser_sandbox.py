# -*- coding: utf-8 -*-
"""

This code demonstrates the use of the qggmrf denoiser.

"""
import time

import numpy as np
import mbirjax as mj

"""**Set the geometry parameters**"""

# Set the experiment parameters - set one of these to be an empty list to use the default values, then vary the other
sharpness_levels = [2, 1, 0, -1]
sigma_denoise_levels = [] # [0.3, 0.2, 0.1, 0.05]

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
    phantom_denoised, recon_dict = denoiser.denoise(phantom_noisy, sigma_noise=sigma_denoise, init_image=init_image, max_iterations=max_iterations,
                                                    stop_threshold_change_pct=stop_threshold_change_pct)

    elapsed += time.time() - time0
    phantoms.append(phantom_denoised)
    dicts.append(recon_dict)
    nrmse = np.linalg.norm(phantom_denoised - phantom) / np.linalg.norm(phantom)
    slice_labels.append('Sharpness={}, sigma_denoise={:.2f}, \nNRMSE={:.3f}: slice'.format(sharp_level, sigma_denoise, nrmse))

    init_image = phantom_denoised

# ########################### Print out model parameters
denoiser.print_params()
mj.get_memory_stats()
# Display results
title = 'Noisy phantom and denoising with changing parameters'
print('Elapsed time = {:.3f} sec'.format(elapsed))
mj.slice_viewer(*phantoms, data_dicts=dicts, slice_label=slice_labels, title=title, vmin=-0.2, vmax=1.0)
