# -*- coding: utf-8 -*-
"""

This code demonstrates the use of the qggmrf denoiser.

"""
import time

import numpy as np
import mbirjax as mj

"""**Set the geometry parameters**"""

# Set the experiment parameters
np.random.seed(0)
sharpness_levels = [1, 0, -1, -2]
sigma_noise = 0.1
num_tiles = 1  # Note that an even number of tiles will lead to the center slice being all zeros, so use the slice slider to see content.
max_iterations = 300
dc = 0.0
# phantom = phantom + dc
# Set parameters for the problem size - you can vary these, but if you make num_det_rows very small relative to
# channels, then the generated phantom may not have an interior.
num_views = 128
num_det_rows = 128
num_det_channels = 128
snr_db = 30
stop_threshold_change_pct = 0.1

# Get some noisy data
print('Generating data')
recon_shape = (num_det_channels, num_det_channels, num_det_rows)
phantom = mj.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
phantom = np.tile(phantom, (num_tiles, num_tiles, num_tiles))
#
recon_shape = [num_tiles * size for size in recon_shape]
phantom_noisy = phantom + sigma_noise * np.random.randn(*recon_shape)

denoiser = mj.QGGMRFDenoiser(phantom.shape)
denoiser.set_params(snr_db=snr_db)
denoiser.set_params(partition_sequence=[4])

"""**Do the reconstruction and display the results.**"""
phantoms = [phantom_noisy]
dicts = [None]
nrmse = np.linalg.norm(phantom_noisy - phantom) / np.linalg.norm(phantom)
slice_labels = ['Noise sigma={}, NRMSE={:.3f}: slice'.format(sigma_noise, nrmse)]
init_image = phantom_noisy

# ##########################
# Denoise at various levels
elapsed = 0
for s in sharpness_levels:
    print('Starting recon with sharpness = {}'.format(s))
    denoiser.set_params(sharpness=s)
    time0 = time.time()
    phantom_denoised, recon_dict = denoiser.denoise(phantom_noisy, sigma_noise=sigma_noise, init_image=init_image, max_iterations=max_iterations,
                                                    stop_threshold_change_pct=stop_threshold_change_pct)

    elapsed += time.time() - time0
    phantoms.append(phantom_denoised)
    dicts.append(recon_dict)
    nrmse = np.linalg.norm(phantom_denoised - phantom) / np.linalg.norm(phantom)
    slice_labels.append('Sharpness={}, NRMSE={:.3f}: slice'.format(s, nrmse))

    init_image = phantom_denoised

# ########################### Print out model parameters
denoiser.print_params()
mj.get_memory_stats()
# Display results
title = 'Noisy phantom and denoising with changing sharpness'
print('Elapsed time = {:.3f} sec'.format(elapsed))
mj.slice_viewer(*phantoms, slice_label=slice_labels, title=title, vmin=-0.2, vmax=1.0)
