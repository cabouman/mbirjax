import numpy as np
import os
import time
import pprint
import jax.numpy as jnp
import jax, scipy, pickle
import sys
source_path = "/home/li5273/PycharmProjects/mbirjax/mbirjax"
if source_path not in sys.path:
    sys.path.insert(0, source_path)
import pickle
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
time_start = time.time()
print("\n*******************************************************",
        "\n************** NSI dataset preprocessing **************",
        "\n*******************************************************")
sino, cone_beam_params, optional_params = \
    mbirjax.preprocess.nsi.compute_sino_and_params(dataset_dir,
                                                    downsample_factor=downsample_factor,
                                                    subsample_view_factor=subsample_view_factor)

print(f"Preprocessing time: {time.time() - time_start:.2f} seconds")

