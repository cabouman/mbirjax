import numpy as np
import matplotlib.pyplot as plt

# Function to load data
def load_data(filenames):
    return [np.load(filename) for filename in filenames]

# Function to plot data
def plot_sinograms(angles, sinograms, labels, title, xlabel, ylabel, markers):
    plt.figure(figsize=(10, 6))
    for sinogram, label, marker in zip(sinograms, labels, markers):
        plt.plot(angles, sinogram, marker=marker, label=label)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=12)
    plt.grid()
    plt.show()

# Experiment 1: Vary delta_det_channel
filenames_det_channel = [
    'Sum_of_Sinogram_Entries_vs_View_Angle__Parallel_Beam___delta_voxel__1.00__delta_det_channel__1.00_data.npz',
    'Sum_of_Sinogram_Entries_vs_View_Angle__Parallel_Beam___delta_voxel__1.00__delta_det_channel__2.00_data.npz',
    'Sum_of_Sinogram_Entries_vs_View_Angle__Parallel_Beam___delta_voxel__1.00__delta_det_channel__3.00_data.npz',
    'Sum_of_Sinogram_Entries_vs_View_Angle__Parallel_Beam___delta_voxel__1.00__delta_det_channel__4.00_data.npz',
]

data_det_channel = load_data(filenames_det_channel)
angles = data_det_channel[0]['angles']  # Extract angles from one file
sinograms_det_channel = [data['sinogram_sums'] for data in data_det_channel]
labels_det_channel = ['delta_det_channel = 1', 'delta_det_channel = 2', 'delta_det_channel = 3', 'delta_det_channel = 4']
markers_det_channel = ['o', 'x', '*', 's']

plot_sinograms(angles, sinograms_det_channel, labels_det_channel,
               'Sinogram Sums vs View Angles, delta_voxel = 1',
               'View Angle (radians)', 'Sum of Sinogram Entries',
               markers_det_channel)

# Experiment 2: Vary delta_voxel
filenames_voxel = [
    'Sum_of_Sinogram_Entries_vs_View_Angle__Parallel_Beam___delta_voxel__1.00__delta_det_channel__1.00_data.npz',
    'Sum_of_Sinogram_Entries_vs_View_Angle__Parallel_Beam___delta_voxel__2.00__delta_det_channel__1.00_data.npz',
    'Sum_of_Sinogram_Entries_vs_View_Angle__Parallel_Beam___delta_voxel__3.00__delta_det_channel__1.00_data.npz',
    'Sum_of_Sinogram_Entries_vs_View_Angle__Parallel_Beam___delta_voxel__4.00__delta_det_channel__1.00_data.npz',
]

data_voxel = load_data(filenames_voxel)
sinograms_voxel = [data['sinogram_sums'] for data in data_voxel]
labels_voxel = ['delta_voxel = 1', 'delta_voxel = 2', 'delta_voxel = 3', 'delta_voxel = 4']
markers_voxel = ['o', 'x', '*', 's']

plot_sinograms(angles, sinograms_voxel, labels_voxel,
               'Sinogram Sums vs View Angles, delta_det_channel = 1',
               'View Angle (radians)', 'Sum of Sinogram Entries',
               markers_voxel)
