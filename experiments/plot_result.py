import numpy as np
import matplotlib.pyplot as plt

# Load data from the .npz file
data1 = np.load('Sum_of_Sinogram_Entries_vs_View_Angle__Parallel_Beam___delta_voxel__1.00__delta_det_channel__1.00_data.npz')
data2 = np.load('Sum_of_Sinogram_Entries_vs_View_Angle__Parallel_Beam___delta_voxel__1.00__delta_det_channel__2.00_data.npz')
data3 = np.load('Sum_of_Sinogram_Entries_vs_View_Angle__Parallel_Beam___delta_voxel__1.00__delta_det_channel__3.00_data.npz')
data4 = np.load('Sum_of_Sinogram_Entries_vs_View_Angle__Parallel_Beam___delta_voxel__1.00__delta_det_channel__4.00_data.npz')

angles = data1['angles']
# Plot the data

sinogram_sums1 = data1['sinogram_sums']
sinogram_sums2 = data2['sinogram_sums']
sinogram_sums3 = data3['sinogram_sums']
sinogram_sums4 = data4['sinogram_sums']

plt.plot(angles, sinogram_sums1, marker='o', label='delta_det_channel = 1')
plt.plot(angles, sinogram_sums2, marker='x', label='delta_det_channel = 2')
plt.plot(angles, sinogram_sums3, marker='*', label='delta_det_channel = 3')
plt.plot(angles, sinogram_sums4, marker='s', label='delta_det_channel = 4')

plt.title('Sinogram Sums vs View Angles')
plt.xlabel('View Angle (radians)')
plt.ylabel('Sum of Sinogram Entries')
plt.legend()  # Add a legend to distinguish datasets
plt.grid()
plt.show()



# Load data from the .npz file
data1 = np.load('Sum_of_Sinogram_Entries_vs_View_Angle__Parallel_Beam___delta_voxel__1.00__delta_det_channel__1.00_data.npz')
data2 = np.load('Sum_of_Sinogram_Entries_vs_View_Angle__Parallel_Beam___delta_voxel__2.00__delta_det_channel__1.00_data.npz')
data3 = np.load('Sum_of_Sinogram_Entries_vs_View_Angle__Parallel_Beam___delta_voxel__3.00__delta_det_channel__1.00_data.npz')
data4 = np.load('Sum_of_Sinogram_Entries_vs_View_Angle__Parallel_Beam___delta_voxel__4.00__delta_det_channel__1.00_data.npz')

sinogram_sums1 = data1['sinogram_sums']
sinogram_sums2 = data2['sinogram_sums']
sinogram_sums3 = data3['sinogram_sums']
sinogram_sums4 = data4['sinogram_sums']

plt.plot(angles, sinogram_sums1, marker='o', label='delta_voxel = 1')
plt.plot(angles, sinogram_sums2, marker='x', label='delta_voxel = 2')
plt.plot(angles, sinogram_sums3, marker='*', label='delta_voxel = 3')
plt.plot(angles, sinogram_sums4, marker='s', label='delta_voxel = 4')

plt.title('Sinogram Sums vs View Angles')
plt.xlabel('View Angle (radians)')
plt.ylabel('Sum of Sinogram Entries')
plt.legend()  # Add a legend to distinguish datasets
plt.grid()
plt.show()