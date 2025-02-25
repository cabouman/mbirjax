import mbirjax
from mbirjax.preprocess.nsi import load_scans_and_params
import time
import pprint as pp
import mbirjax.preprocess as preprocess
import numpy as np
import os
import sys
import warnings
import jax.numpy as jnp
import h5py
from jax import jit
source_path = "/home/li5273/PycharmProjects/mbirjax/mbirjax"
if source_path not in sys.path:
    sys.path.insert(0, source_path)


os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.98'

pp = pp.PrettyPrinter(indent=4)


def compute_sino_and_params_corr_bg(dataset_dir,
                            downsample_factor=(1, 1), crop_region=((0, 1), (0, 1)),
                            subsample_view_factor=1):

    print("\n\n########## Loading object, blank, dark scans, as well as geometry parameters from NSI dataset directory ...")
    obj_scan, blank_scan, dark_scan, cone_beam_params, optional_params, defective_pixel_list = \
            load_scans_and_params(dataset_dir,
                                  downsample_factor=downsample_factor, crop_region=crop_region,
                                  subsample_view_factor=subsample_view_factor)
    mbirjax.get_memory_stats(print_results=True)

    print("MBIRJAX geometry parameters:")
    pp.pprint(cone_beam_params)
    pp.pprint(optional_params)
    print('obj_scan shape = ', obj_scan.shape)
    print('blank_scan shape = ', blank_scan.shape)
    print('dark_scan shape = ', dark_scan.shape)


    sino, defective_pixel_list = preprocess.compute_sino_transmission_jax(obj_scan, blank_scan, dark_scan, defective_pixel_list)
    del obj_scan, blank_scan, dark_scan

    print("\n\n########## Correcting background offset to the sinogram from edge pixels ...")

    background_offset = preprocess.estimate_background_offset_jax(sino)
    print("background_offset = ", background_offset)
    sino = sino - background_offset

    return sino, cone_beam_params, optional_params


def main():

    # ###################### User defined params. Change the parameters below for your own use case.
    output_path = './output/nsi_demo/'  # path to store output recon images
    os.makedirs(output_path, exist_ok=True)  # mkdir if directory does not exist

    dataset_dir = '/home/li5273/PycharmProjects/mbirjax_applications/nsi/demo_data/vert_no_metal'
    # #### preprocessing parameters
    downsample_factor = [1, 1]  # downsample factor of scan images along detector rows and detector columns.
    subsample_view_factor = 1  # view subsample factor.

    # #### recon parameters
    # ###################### End of parameters
    print("\n*******************************************************",
            "\n************** NSI dataset preprocessing **************",
            "\n*******************************************************")
    sino, cone_beam_params, optional_params = \
        compute_sino_and_params_corr_bg(dataset_dir,
                                                        downsample_factor=downsample_factor,
                                                        subsample_view_factor=subsample_view_factor)


    with h5py.File('/home/li5273/PycharmProjects/mbirjax_applications/nsi/demo_data/20slices.h5', 'w') as h5f:
        h5f.create_dataset('sino', data=sino[0:20, :, :])

if __name__ == '__main__':
    main()
