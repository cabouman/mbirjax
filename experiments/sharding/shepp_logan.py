import numpy as np
import time
import pickle
import mbirjax as mj

# SETUP
# 1. Generate the phantom, sinogram, and params
#    On the 'sharing_dev/control' branch run the script 'expirements/sharding/shepp_logan_prep.py'
#    This generates the phantom and sinogram using the CPU and saves them to the scratch directory.

# 2. Set username
USER = "ncardel" # change the user to your username

# 3. Comment out or close the viewer


###### The

# load phantom, sinogram, and params
output_directory = f"/scratch/gautschi/{USER}/nsi/cone"

phantom_filepath = f"{output_directory}/phantom.npy"
sinogram_filepath = f"{output_directory}/sinogram.npy"
params_filepath = f"{output_directory}/params.pkl"

print("phantom_filepath:", phantom_filepath)
print("sinogram_filepath:", sinogram_filepath)
print("params_filepath:", params_filepath)

phantom = np.load(phantom_filepath)
sinogram = np.load(sinogram_filepath)
with open(params_filepath, "rb") as f:
    params = pickle.load(f)

print("phantom.shape:", phantom.shape)
print("sinogram.shape:", sinogram.shape)
print("params:", params.items())

if 'source_detector_dist' in params.keys():
    model_type = 'cone'
else:
    model_type = 'parallel'

print("Detected model_type:", model_type)

mj.get_memory_stats()

sinogram_shape = sinogram.shape
angles = params['angles']

# View the sinogram
title = 'Original sinogram \nUse the sliders to change the view or adjust the intensity range.\nRight click the image to see options.'
mj.slice_viewer(sinogram, data_dicts=params, slice_axis=0, title=title, slice_label='View')

"""**Initialize for the reconstruction**"""

# ####################
# Use the parameters to get the data and initialize the model for reconstruction.
if model_type == 'cone':
    source_detector_dist = params['source_detector_dist']
    source_iso_dist = params['source_iso_dist']
    ct_model = mj.ConeBeamModel(sinogram.shape, angles, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)
else:
    ct_model = mj.ParallelBeamModel(sinogram.shape, angles)

# Generate weights array - for an initial reconstruction, use weights = None, then modify if needed to reduce the effect of possibly noisy sinogram entries.
# weights = None
weights = mj.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')

# Set reconstruction parameter values
# Sharpness is a float, typically in the range (-1, 2).  The default value is 1.0.
# Sharpness of 1 or 2 will yield clearer edges, with more high-frequency variation.
# Sharnpess of -1 or 0 will yield softer edges and smoother interiors.
# Values of sharpness above 3 may lead to slower convergence with high-frequency artifacts, particularly in the center slices in cone beam.
sharpness = 1.0
ct_model.set_params(sharpness=sharpness, use_gpu='automatic')

# Print out model parameters
ct_model.print_params()

"""**Do the reconstruction and display the results.**"""

# ##########################
# Perform VCD reconstruction
print('Starting recon')
time0 = time.time()

# ct_model.recon returns the estimated object along with a dictionary with entries
# 'recon_params', 'model_params', 'logs', and 'notes'
# recon and recon_dict can be used together for viewing and saving to an hdf5 file.
# Saving can be done either in code or through the viewer, and the hdf5 file can be loaded for viewing
# or to recreate the model if desired.
recon, recon_dict = ct_model.recon(sinogram, weights=weights)

max_diff = np.amax(np.abs(phantom - recon))
nrmse = np.linalg.norm(recon - phantom) / np.linalg.norm(phantom)
pct_95 = np.percentile(np.abs(recon - phantom), 95)

elapsed = time.time() - time0

mj.get_memory_stats()
print('Elapsed time for recon is {:.3f} seconds'.format(elapsed))

# ##########################
# Add some notes to include for display and saving
recon_dict['notes'] += 'NRMSE between recon and phantom = {}'.format(nrmse)
recon_dict['notes'] += 'Maximum pixel difference between phantom and recon = {}'.format(max_diff)
recon_dict['notes'] += '95% of recon pixels are within {} of phantom'.format(pct_95)

# recon and recon_dict can be saved from the viewer or directly in code
filepath = f"{output_directory}/demo1_recon.h5"
ct_model.save_recon_hdf5(filepath, recon, recon_dict)

# Display results
title = 'Phantom (left) vs VCD Recon (right) \nUse the sliders to change the slice or adjust the intensity range.\nRight click an image to see options.'
mj.slice_viewer(phantom, recon, data_dicts=[None, recon_dict], title=title)

# The recon and recon_dict can be reloaded either here or in the viewer, and the recon_dict can be used to recreate
#  the model if desired. The load function can be used even without an existing instance of a ct model.
new_recon, new_recon_dict, new_model = mj.TomographyModel.load_recon_hdf5(filepath, recreate_model=True)

print('recon and recon_dict loaded from {}'.format(filepath))
print('New model created: {}'.format(new_model.get_params('geometry_type')))

# From this you could view again, restart the recon from the previous iteration, etc.

"""**Next:** Try changing some of the parameters and re-running or try [some of the other demos](https://mbirjax.readthedocs.io/en/latest/demos_and_faqs.html).  """