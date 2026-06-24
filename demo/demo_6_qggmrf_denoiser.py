# -*- coding: utf-8 -*-
"""

This code demonstrates the use of the qggmrf denoiser.

"""

import numpy as np
import mbirjax as mj

"""**Set the geometry parameters**"""

# Set the experiment parameters
sigma_denoise = [0.05, 0.1, 0.15]
sigma_noise_added = 0.1
sharpness = 0

# Set parameters for the problem size - you can vary these, but if you make num_det_rows very small relative to
# channels, then the generated phantom may not have an interior.
num_views = 256
num_det_rows = 256
num_det_channels = 256
max_iterations = 20
stop_threshold_change_pct = 0.0

# Get some noisy data
recon_shape = (num_det_channels, num_det_channels, num_det_rows)
phantom = mj.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
phantom_noisy = phantom + sigma_noise_added * np.random.randn(*recon_shape)

denoiser = mj.QGGMRFDenoiser(phantom.shape)

# Print out model parameters
denoiser.print_params()

"""**Do the reconstruction and display the results.**"""
phantoms = [phantom_noisy]
dicts = [None]
nrmse = np.linalg.norm(phantom_noisy - phantom) / np.linalg.norm(phantom)
slice_labels = ['Input image: noise sigma={}, \nNRMSE={:.3f}: slice'.format(sigma_noise_added, nrmse)]
init_image = phantom_noisy

denoiser.set_params(sharpness=sharpness)

# ##########################
# Denoise at various levels
import time
for s in sigma_denoise:
    print('Starting recon with sigma_denoise = {}'.format(s))
    t0 = time.time()
    phantom_denoised, recon_dict = denoiser.denoise(phantom_noisy, sigma_noise=s,
                                                    stop_threshold_change_pct=stop_threshold_change_pct)
    elapsed = time.time() - t0
    print('Elapsed time = {:.3f} s'.format(elapsed))
    phantoms.append(phantom_denoised)
    dicts.append(recon_dict)
    nrmse = np.linalg.norm(phantom_denoised - phantom) / np.linalg.norm(phantom)
    slice_labels.append('Given sigma_noise={}, \nNRMSE={:.3f}: slice'.format(s, nrmse))


print('Starting recon with sigma_denoise estimated from data')
t0 = time.time()
phantom_denoised, recon_dict = denoiser.denoise(phantom_noisy, sigma_noise=None,
                                                stop_threshold_change_pct=stop_threshold_change_pct)
elapsed = time.time() - t0
print('Elapsed time = {:.3f} s'.format(elapsed))
phantoms.append(phantom_denoised)
dicts.append(recon_dict)
nrmse = np.linalg.norm(phantom_denoised - phantom) / np.linalg.norm(phantom)
sigma_noise = denoiser.get_params('sigma_noise')
print('Estimated sigma_noise = {:.3f} s'.format(sigma_noise))
slice_labels.append('Estimated sigma_noise={:.2f}, \nNRMSE={:.3f}: slice'.format(sigma_noise, nrmse))

# ##########################
mj.get_memory_stats()

# Display results
title = 'Noisy phantom and denoising with changing sigma_noise, sharpness = {}'.format(sharpness)
# mj.slice_viewer(*phantoms, data_dicts=dicts, slice_label=slice_labels, title=title, vmin=-0.2, vmax=1.0)
