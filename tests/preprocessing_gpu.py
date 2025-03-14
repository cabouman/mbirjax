import numpy as np
import os
import time
import pprint
import jax.numpy as jnp
import sys
import os
source_path = "/home/li5273/Desktop/0313/mbirjax_preprocessing_gpu_testrotate/mbirjax"
import importlib.util
package_name = "mbirjax"
spec = importlib.util.spec_from_file_location(package_name, os.path.join(source_path, "__init__.py"))
mbirjax = importlib.util.module_from_spec(spec)
sys.modules[package_name] = mbirjax
spec.loader.exec_module(mbirjax)

print("mbirjax loaded from:", mbirjax.__file__)
import pprint

pp = pprint.PrettyPrinter(indent=4)

if __name__ == "__main__":

    # ###################### User defined params. Change the parameters below for your own use case.

    # ##### params for dataset downloading. User may change these parameters for their own datasets.
    # An example NSI dataset (tarball) will be downloaded from `dataset_url`, and saved to `download_dir`.
    # url to NSI dataset.
    dataset_dir = '/home/li5273/PycharmProjects/mbirjax_applications/nsi/demo_data/vert_no_metal'
    daraset_dir = '/depot/bouman/data/nsi_demo_data/demo_nsi_vert_no_metal_all_views'

    downsample_factor = [1, 1]  # downsample factor of scan images along detector rows and detector columns.
    subsample_view_factor = 1  # view subsample factor.

    # #### recon parameters
    sharpness = 1.0
    # ###################### End of parameters

    print("\n*******************************************************",
          "\n************** NSI dataset preprocessing **************",
          "\n*******************************************************")
    time0 = time.time()
    sino, cone_beam_params, optional_params = \
        mbirjax.preprocess.nsi.compute_sino_and_params(dataset_dir,
                                                       downsample_factor=downsample_factor,
                                                       subsample_view_factor=subsample_view_factor)

    print("\n*******************************************************",
          "\n***************** Set up MBIRJAX model ****************",
          "\n*******************************************************")
    # ConeBeamModel constructor
    ct_model = mbirjax.ConeBeamModel(**cone_beam_params)

    # Set additional geometry arguments
    ct_model.set_params(**optional_params)

    # Set reconstruction parameter values
    ct_model.set_params(sharpness=sharpness, verbose=1)

    # Print out model parameters
    print(f'Preprocessing finished, model initialized. Total time: {time.time() - time0:.2f} seconds.')