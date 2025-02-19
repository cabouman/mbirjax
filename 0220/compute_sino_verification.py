import mbirjax
from mbirjax.preprocess.nsi import load_scans_and_params
import time
import pprint as pp
import mbirjax.preprocess as preprocess
import numpy as np
import os
import sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
source_path = "/home/li5273/PycharmProjects/mbirjax/mbirjax"
if source_path not in sys.path:
    sys.path.insert(0, source_path)


os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.98'

pp = pp.PrettyPrinter(indent=4)


def compute_sino_and_params_compute_sino(dataset_dir,
                            downsample_factor=(1, 1), crop_region=((0, 1), (0, 1)),
                            subsample_view_factor=1, function='1'):
    """
    Load NSI sinogram data and prepare all needed arrays and parameters for a ConeBeamModel reconstruction.

    This function computes the sinogram and geometry parameters from an NSI scan directory containing scan data and parameters.
    More specifically, the function performs the following operations in a single easy-to-use manner:

    1. Loads the object, blank, and dark scans, as well as the geometry parameters from an NSI dataset directory.

    2. Computes the sinogram from object, blank, and dark scans.

    3. Replaces defective pixels with interpolated values.

    4. Performs background offset correction to the sinogram from the edge pixels.

    5. Corrects sinogram data to account for detector rotation.

    Args:
        dataset_dir (string): Path to an NSI scan directory. The directory is assumed to have the following structure:

            - ``*.nsipro`` (NSI config file)
            - ``Geometry*.rtf`` (geometry report)
            - ``Radiographs*/`` (directory containing all radiograph images)
            - ``**/gain0.tif`` (blank scan image)
            - ``**/offset.tif`` (dark scan image)
            - ``**/*.defect`` (defective pixel information)

        downsample_factor ((int, int), optional) - Down-sample factors along the detector rows and channels respectively.
            If scan size is not divisible by `downsample_factor`, the scans will be first truncated to a size that is divisible by `downsample_factor`.

        crop_region (((float, float),(float, float)), optional) - Values of ((row_start, row_end), (col_start, col_end)) define a bounding box that crops the scan.
            The default of ((0, 1), (0, 1)) retains the entire scan.

        subsample_view_factor (int, optional): View subsample factor. By default no view subsampling will be performed.

    Returns:
        tuple: [sinogram, cone_beam_params, optional_params]

            sino (jax array): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels).

            cone_beam_params (dict): Required parameters for the ConeBeamModel constructor.

            optional_params (dict): Additional ConeBeamModel parameters to be set using set_params().

    Example:
        .. code-block:: python

            # Get data and recon parameters
            sino, cone_beam_params, optional_params = mbirjax.preprocess.NSI.compute_sino_and_params(dataset_dir, downsample_factor=downsample_factor, subsample_view_factor=subsample_view_factor)

            # Create the model and set the parameters
            ct_model = mbirjax.ConeBeamModel(**cone_beam_params)
            ct_model.set_params(**optional_params)
            ct_model.set_params(sharpness=sharpness, verbose=1)

            # Compute sinogram weights and do the reconstruction
            weights = ct_model.gen_weights(sino, weight_type='transmission_root')
            recon, recon_params = ct_model.recon(sino, weights=weights)

    """

    print("\n\n########## Loading object, blank, dark scans, as well as geometry parameters from NSI dataset directory ...")
    time0 = time.time()
    obj_scan, blank_scan, dark_scan, cone_beam_params, optional_params, defective_pixel_list = \
            load_scans_and_params(dataset_dir,
                                  downsample_factor=downsample_factor, crop_region=crop_region,
                                  subsample_view_factor=subsample_view_factor)
    mbirjax.get_memory_stats(print_results=True)
    print(f"time to load scans and params = {time.time()-time0:.2f} seconds")

    print("MBIRJAX geometry parameters:")
    pp.pprint(cone_beam_params)
    pp.pprint(optional_params)
    print('obj_scan shape = ', obj_scan.shape)
    print('blank_scan shape = ', blank_scan.shape)
    print('dark_scan shape = ', dark_scan.shape)

    print("\n\n########## Computing sinogram from object, blank, and dark scans ...")
    time0 = time.time()
    if function == '1':
        sino, defective_pixel_list = \
            preprocess.compute_sino_transmission_jax(obj_scan, blank_scan, dark_scan, defective_pixel_list, batch_size=90)
    elif function == '2':
        sino, defective_pixel_list = \
            preprocess.compute_sino_transmission(obj_scan, blank_scan, dark_scan, defective_pixel_list)
    print(f"time to compute sino = {time.time()-time0:.2f} seconds")
    print(f'sino shape = {sino.shape}')

    return sino, cone_beam_params, optional_params

def main():
    output_path = './output/nsi_demo/'
    os.makedirs(output_path, exist_ok=True)

    dataset_dir = '/home/li5273/PycharmProjects/mbirjax_applications/nsi/demo_data/vert_no_metal'
    downsample_factor = [8, 8]
    subsample_view_factor = 1

    # Get results from optimized function
    sino_opt, cone_beam_params_opt, optional_params_opt = compute_sino_and_params_compute_sino(
        dataset_dir, downsample_factor=downsample_factor, subsample_view_factor=subsample_view_factor, function= '1')

    # Get results from original function
    sino_orig, cone_beam_params_orig, optional_params_orig = compute_sino_and_params_compute_sino(
        dataset_dir, downsample_factor=downsample_factor, subsample_view_factor=subsample_view_factor, function= '2')

    # Check if both sinograms are numerically identical
    if np.array_equal(sino_opt, sino_orig):
        print("Both sinograms are **exactly identical**.")
    else:
        # Check for small numerical differences
        diff = np.abs(sino_opt - sino_orig)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print(f"🔍 Maximum difference: {max_diff:.8f}")
        print(f"🔍 Mean difference: {mean_diff:.8f}")

        if np.allclose(sino_opt, sino_orig, atol=1e-5):  # Adjust tolerance if needed
            print("Both sinograms are **numerically close** (small floating-point differences).")
        else:
            print("Sinograms are **not identical**. Significant differences exist!")


if __name__ == "__main__":
    main()