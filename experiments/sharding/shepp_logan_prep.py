import numpy as np
import pickle
import mbirjax as mj
import time

######################### SETUP ############################

# 1. Set username
USER = "ncardel" # change the user to your username

# 2. Comment out or close the viewers

############################################################

output_directory = f"/scratch/gautschi/{USER}/cube_2000"

# Choose the geometry type
model_type = 'cone'  # 'cone' or 'parallel'
object_type = 'shepp-logan'  # 'shepp-logan' or 'cube'

# Set parameters for the problem size
num_views = 2000
num_det_rows = 2000
num_det_channels = 2000


# Generate demo data
print('Generating demo data')
time0 = time.time()

phantom, sinogram, params = mj.generate_demo_data(object_type=object_type, model_type=model_type,
                                                  num_views=num_views, num_det_rows=num_det_rows,
                                                  num_det_channels=num_det_channels)

elapsed = time.time() - time0

mj.get_memory_stats()

print('Elapsed time for generating demo data is {:.3f} seconds'.format(elapsed))

# save the phantom, sinogram, and params
phantom = np.array(phantom)
np.save(f"{output_directory}/phantom.npy", phantom)

sinogram = np.array(sinogram)
np.save(f"{output_directory}/sinogram.npy", sinogram)

with open(f"{output_directory}/params.pkl", "wb") as f:
    pickle.dump(params, f)

# view the phantom and sinogram
mj.slice_viewer(phantom, title="Phantom Test")

mj.slice_viewer(sinogram, title="Sinogram Test")
