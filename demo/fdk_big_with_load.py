import numpy as np
import os
import time
import pprint
import jax.numpy as jnp
import pprint
import jax, scipy, pickle
import sys
source_path = "/home/li5273/PycharmProjects/mbirjax/mbirjax"
if source_path not in sys.path:
    sys.path.insert(0, source_path)

import mbirjax

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.98'

pp = pprint.PrettyPrinter(indent=4)

print('This script is a demonstration of the preprocessing module of NSI dataset. Demo functionality includes:\
\n\t * downloading NSI dataset from specified urls;\
\n\t * Loading object scans, blank scan, dark scan, view angles, and MBIRJAX geometry parameters;\
\n\t * Computing sinogram from object scan, blank scan, and dark scan images;\
\n\t * Computing a 3D reconstruction from the sinogram using MBIRJAX;\
\n\t * Displaying the results.\n')

# ###################### User defined params. Change the parameters below for your own use case.
output_path = './output/nsi_demo/'  # path to store output recon images
os.makedirs(output_path, exist_ok=True)  # mkdir if directory does not exist


dataset_dir = '/home/li5273/PycharmProjects/mbirjax_applications/nsi/demo_data/vert_no_metal'
# #### preprocessing parameters
downsample_factor = [1, 1]  # downsample factor of scan images along detector rows and detector columns.
subsample_view_factor = 1  # view subsample factor.

# #### recon parameters
sharpness = 0.0
# ###################### End of parameters

sino = np.load('/home/li5273/PycharmProjects/mbirjax_applications/nsi/demo_data/sino.npy')

with open("/home/li5273/PycharmProjects/mbirjax_applications/nsi/demo_data/optionalparams.pkl", "rb") as f:
    optional_params = pickle.load(f)

with open("/home/li5273/PycharmProjects/mbirjax_applications/nsi/demo_data/conebeamparams.pkl", "rb") as f:
    cone_beam_params = pickle.load(f)


# ConeBeamModel constructor
ct_model = mbirjax.ConeBeamModel(**cone_beam_params)

# Set additional geometry arguments
ct_model.set_params(**optional_params)

# Set reconstruction parameter values
ct_model.set_params(sharpness=sharpness, verbose=1)

# Print out model parameters
ct_model.print_params()

print("\n*******************************************************",
        "\n************** Calculate sinogram weights *************",
        "\n*******************************************************")
weights = ct_model.gen_weights(sino, weight_type='transmission_root')

print("\n*******************************************************",
        "\n************** Perform VCD reconstruction *************",
        "\n*******************************************************")
# ##########################
# Perform VCD reconstruction
time0 = time.time()

init_recon = ct_model.fdk_recon(sino)


mbirjax.preprocess.export_recon_to_hdf5(init_recon, os.path.join(output_path, "recon_fdk11.h5"),
                                        recon_description="MBIRJAX recon of MAR phantom",
                                        alu_description="1 ALU = 0.508 mm")

# change the image data shape to (slices, rows, cols), so that the rotation axis points up when viewing the
# coronal/sagittal slices with slice_viewer
init_recon = np.transpose(init_recon, (2, 1, 0))
init_recon = init_recon[:, :, ::-1]

vmin = 0
vmax = downsample_factor[0] * 0.008
# Display results
mbirjax.slice_viewer(init_recon, vmin=0, vmax=vmax, slice_axis=0, slice_label='Axial Slice', title='MBIRJAX recon')
mbirjax.slice_viewer(init_recon, vmin=0, vmax=vmax, slice_axis=1, slice_label='Coronal Slice', title='MBIRJAX recon')
mbirjax.slice_viewer(init_recon, vmin=0, vmax=vmax, slice_axis=2, slice_label='Sagittal Slice', title='MBIRJAX recon')


recon, recon_params = ct_model.recon(sino, weights=weights, init_recon=init_recon)

recon.block_until_ready()
elapsed = time.time() - time0
print('Elapsed time for recon is {:.3f} seconds'.format(elapsed))
# ##########################

# Print out parameters used in recon
pprint.pprint(recon_params._asdict())

mbirjax.preprocess.export_recon_to_hdf5(recon, os.path.join(output_path, "recon_vcd_11.h5"),
                                        recon_description="MBIRJAX recon of MAR phantom",
                                        alu_description="1 ALU = 0.508 mm")

# change the image data shape to (slices, rows, cols), so that the rotation axis points up when viewing the
# coronal/sagittal slices with slice_viewer
recon = np.transpose(recon, (2, 1, 0))
recon = recon[:, :, ::-1]

vmin = 0
vmax = downsample_factor[0] * 0.008
# Display results
mbirjax.slice_viewer(recon, vmin=0, vmax=vmax, slice_axis=0, slice_label='Axial Slice', title='MBIRJAX recon')
mbirjax.slice_viewer(recon, vmin=0, vmax=vmax, slice_axis=1, slice_label='Coronal Slice', title='MBIRJAX recon')
mbirjax.slice_viewer(recon, vmin=0, vmax=vmax, slice_axis=2, slice_label='Sagittal Slice', title='MBIRJAX recon')





