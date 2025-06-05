import numpy as np
import mbirjax as mj
import matplotlib.pyplot as plt


sinogram_control = np.load("sinogram_control.npy")
title = 'Control sinogram generated without sharding'
mj.slice_viewer(sinogram_control, slice_axis=0, title=title, slice_label='View')

sinogram_8_gpus = np.load("sinogram_8_gpus.npy")
title = 'Sinogram generated with sharding on 8 GPUs'
mj.slice_viewer(sinogram_8_gpus, slice_axis=0, title=title, slice_label='View')

sonigram_diff = sinogram_control - sinogram_8_gpus
title = 'The difference of the two sinograms'
mj.slice_viewer(sonigram_diff, slice_axis=0, title=title, slice_label='View')

# Plot histogram
plt.hist(sonigram_diff.flatten(), bins=30)
plt.title("Histogram of the values from the difference sinogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()