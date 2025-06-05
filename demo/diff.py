import numpy as np
import mbirjax as mj
import matplotlib.pyplot as plt

import numpy as np


def summarize_array(arr):
    """
    Print summary statistics to help understand the distribution of a NumPy array.
    """

    # Ensure arr is a NumPy array
    arr = np.asarray(arr)

    # Flatten to 1D for global statistics
    flat = arr.ravel()

    # Compute basic statistics
    count = flat.size
    minimum = flat.min()
    maximum = flat.max()
    mean = flat.mean()
    median = np.median(flat)
    std_dev = flat.std()

    # Compute selected percentiles
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    pct_values = np.percentile(flat, percentiles)

    # Print out the results
    print(f"Count of values:       {count}")
    print(f"Minimum:               {minimum:.6f}")
    print(f"1st percentile:        {pct_values[0]:.6f}")
    print(f"5th percentile:        {pct_values[1]:.6f}")
    print(f"25th percentile (Q1):  {pct_values[2]:.6f}")
    print(f"Median (50th pct):     {pct_values[3]:.6f}")
    print(f"75th percentile (Q3):  {pct_values[4]:.6f}")
    print(f"95th percentile:       {pct_values[5]:.6f}")
    print(f"99th percentile:       {pct_values[6]:.6f}")
    print(f"Maximum (100th pct):   {maximum:.6f}")
    print(f"\nMean:                  {mean:.6f}")
    print(f"Standard Deviation:    {std_dev:.6f}")


# Example usage (remove or replace `example_arr` with your actual data):
# example_arr = np.random.randn(100, 100, 100)  # e.g. a (100x100x100) array
# summarize_array(example_arr)


sinogram_control = np.load("sinogram_control.npy")
title = 'Control sinogram generated without sharding'
mj.slice_viewer(sinogram_control, slice_axis=0, title=title, slice_label='View')

sinogram_8_gpus = np.load("sinogram_8_gpus.npy")
title = 'Sinogram generated with sharding on 8 GPUs'
mj.slice_viewer(sinogram_8_gpus, slice_axis=0, title=title, slice_label='View')

sonigram_diff = sinogram_control - sinogram_8_gpus
title = 'The difference of the two sinograms'
mj.slice_viewer(sonigram_diff, slice_axis=0, title=title, slice_label='View')

summarize_array(sonigram_diff.flatten())

# Plot histogram
plt.hist(sonigram_diff.flatten(), bins=30)
plt.title("Histogram of the values from the difference sinogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()