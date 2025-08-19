import numpy as np
import time
import pickle
import mbirjax as mj

# SETUP
# 1. Generate the pho\antom, sinogram, and params
#    On the 'sharing_dev/stable' branch run the script 'expirements/sharding/shepp_logan_prep.py'
#    This generates the phantom and sinogram using the CPU and saves them to the scratch directory.

# 2. Set username
USER = "ncardel" # change the user to your username

# 3. Comment out or close the viewer


###### The

# load phantom, sinogram, and params
output_directory = f"/scratch/gautschi/{USER}/cone"

phantom_filepath = f"{output_directory}/phantom.npy"
params_filepath = f"{output_directory}/params.pkl"

print("phantom_filepath:", phantom_filepath)
print("params_filepath:", params_filepath)

phantom = np.load(phantom_filepath)
with open(params_filepath, "rb") as f:
    params = pickle.load(f)

print("phantom.shape:", phantom.shape)
print("params:", params.items())

recon_filepath = f"{output_directory}/demo1_recon.h5"
print("recon_filepath:", recon_filepath)

# The recon and recon_dict can be reloaded either here or in the viewer, and the recon_dict can be used to recreate
#  the model if desired. The load function can be used even without an existing instance of a ct model.
new_recon, new_recon_dict, _ = mj.TomographyModel.load_recon_hdf5(recon_filepath, recreate_model=True)

# Display results
title = 'Phantom (left) vs Recon using sharding (right)'
mj.slice_viewer(phantom, new_recon, data_dicts=[None, new_recon_dict], title=title)
