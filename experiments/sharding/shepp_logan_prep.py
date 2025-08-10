import numpy as np
import pickle
import mbirjax as mj

"""
**Set the geometry parameters**
    This code is to genreate the phantom and sinogram for the 2000, 2000, 2000 problem size
    with the original sparse projection implementations on the CPU. There are some memory 
    issues with sharding and generating the test phantom and sinogram.
"""

output_directory = "/scratch/gautschi/ncardel"

# Choose the geometry type
model_type = 'parallel'  # 'cone' or 'parallel'
object_type = 'shepp-logan'  # 'shepp-logan' or 'cube'

# Set parameters for the problem size
num_views = 2000
num_det_rows = 2000
num_det_channels = 2000

# Generate simulated data
phantom, sinogram, params = mj.generate_demo_data(object_type=object_type, model_type=model_type,
                                                  num_views=num_views, num_det_rows=num_det_rows,
                                                  num_det_channels=num_det_channels)

mj.slice_viewer(phantom, title="Phantom Test")

mj.slice_viewer(sinogram, title="Sinogram Test")

phantom = np.array(phantom)
np.save(f"{output_directory}/phantom.npy", phantom)

sinogram = np.array(sinogram)
np.save(f"{output_directory}/sinogram.npy", sinogram)

with open(f"{output_directory}/params.pkl", "wb") as f:
    pickle.dump(params, f)
