# **Create Impulse Image**

import numpy as np

def create_impulse_image(shape, position):
   """
   Creates a 3D impulse image with one pixel set to 1 and the rest set to 0.

   Parameters:
   - shape (tuple of int): Shape of the 3D image (depth, height, width).
   - position (tuple of int): Position of the impulse (z, y, x).

   Returns:
   - numpy.ndarray: 3D array with an impulse at the specified position.
   """
   # Initialize a 3D array with zeros
   image = np.zeros(shape, dtype=np.float32)

   # Set the specified pixel to 1
   image[position] = 1

   return image


def create_impulse_cube(shape, position, cube_size=3):
   """
   Creates a 3D impulse image with a cube of specified size set to 1 and the rest set to 0.

   Parameters:
   - shape (tuple of int): Shape of the 3D image (depth, height, width).
   - position (tuple of int): Center position of the impulse cube (z, y, x).
   - cube_size (int): Length of one side of the cube. Default is 3.

   Returns:
   - numpy.ndarray: 3D array with an impulse cube at the specified position.
   """
   # Initialize a 3D array with zeros
   image = np.zeros(shape, dtype=np.float32)

   # Calculate the ranges for the cube
   half_size = cube_size // 2
   z_start, z_end = position[0] - half_size, position[0] + half_size + 1
   y_start, y_end = position[1] - half_size, position[1] + half_size + 1
   x_start, x_end = position[2] - half_size, position[2] + half_size + 1

   # Ensure ranges are within the bounds of the array
   z_start, z_end = max(0, z_start), min(shape[0], z_end)
   y_start, y_end = max(0, y_start), min(shape[1], y_end)
   x_start, x_end = max(0, x_start), min(shape[2], x_end)

   # Set the specified cube region to 1
   image[z_start:z_end, y_start:y_end, x_start:x_end] = 1

   return image
