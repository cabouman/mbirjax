import mbirjax
from mbirjax.preprocess.nsi import load_scans_and_params
import time
import pprint as pp
import mbirjax.preprocess as preprocess
import numpy as np
import os
import sys
import scipy
import dm_pix
import jax.numpy as jnp
import jax
source_path = "/home/li5273/PycharmProjects/mbirjax/mbirjax"
if source_path not in sys.path:
    sys.path.insert(0, source_path)


os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.98'

pp = pp.PrettyPrinter(indent=4)


def compute_sino_and_params_rotat(dataset_dir,
                            downsample_factor=(1, 1), crop_region=((0, 1), (0, 1)),
                            subsample_view_factor=1):
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

    sino, defective_pixel_list = preprocess.compute_sino_transmission_jax(obj_scan, blank_scan, dark_scan, defective_pixel_list)
    del obj_scan, blank_scan, dark_scan

    print("\n\n########## Correcting sinogram data to account for detector rotation ...")
    time0 = time.time()
    mbirjax.get_memory_stats(print_results=True)
    sino = correct_det_rotation_batch_pix(sino, det_rotation=optional_params["det_rotation"])
    del optional_params["det_rotation"]
    mbirjax.get_memory_stats(print_results=True)
    print(f"time to correct detector rotation = {time.time()-time0:.2f} seconds")

    return sino, cone_beam_params, optional_params


def correct_det_rotation_batch(sino, weights=None, det_rotation=0.0, batch_size=180):
    """
    Correct sinogram data to account for detector rotation, using JAX for batch processing and GPU acceleration.
    Weights are not modified.

    Args:
        sino (jax.numpy.ndarray): Sinogram data with 3D shape (num_views, num_det_rows, num_det_channels).
        weights (jax.numpy.ndarray, optional): Sinogram weights, with the same array shape as ``sino`` (kept unchanged).
        det_rotation (optional, float): tilt angle between the rotation axis and the detector columns in radians.
        batch_size (int): Number of views to process in each batch to avoid memory overload.

    Returns:
        - A jax.numpy.ndarray containing the corrected sinogram data if weights is None.
        - A tuple (sino_corrected, weights) if weights is not None.
    """

    num_views = sino.shape[0]  # Total number of views
    sino_corrected = np.empty((0, *sino.shape[1:]), dtype=sino.dtype)
    print(f'before batch:{mbirjax.get_memory_stats()}')
    # Process in batches with looping and progress printing
    for i in range(0, num_views, batch_size):
        print(f"Processing batch {i//batch_size + 1} / {(num_views // batch_size) + 1}")

        # Get the current batch (from i to i + batch_size)
        batch = sino[i : min(i + batch_size, num_views)]

        # Apply the rotation on this batch
        batch_rotated = scipy.ndimage.rotate(batch, np.rad2deg(det_rotation), axes=(1, 2), reshape=False, order=3)

        # Append the rotated batch
        sino_corrected = np.concatenate([sino_corrected, batch_rotated], axis=0)
        print(f'After batch:{mbirjax.get_memory_stats()}')

    if weights is None:
        return sino_corrected
    print("correct_det_rotation: weights provided by the user. Please note that zero weight entries might become non-zero after tilt angle correction.")
    weights = scipy.ndimage.rotate(weights, np.rad2deg(det_rotation), axes=(1,2), reshape=False, order=3)
    print(f'no batch after:{mbirjax.get_memory_stats()}')
    return sino_corrected, weights

def correct_det_rotation_batch_pix(sino, weights=None, det_rotation=0.0, batch_size=60):
    """
    Correct sinogram data to account for detector rotation, using JAX for batch processing and GPU acceleration.
    Weights are not modified.

    Args:
        sino (jax.numpy.ndarray): Sinogram data with 3D shape (num_views, num_det_rows, num_det_channels).
        weights (jax.numpy.ndarray, optional): Sinogram weights, with the same array shape as ``sino`` (kept unchanged).
        det_rotation (optional, float): tilt angle between the rotation axis and the detector columns in radians.
        batch_size (int): Number of views to process in each batch to avoid memory overload.

    Returns:
        - A jax.numpy.ndarray containing the corrected sinogram data if weights is None.
        - A tuple (sino_corrected, weights) if weights is not None.
    """

    num_views = sino.shape[0]  # Total number of views
    sino_batches_list = []

    # Process in batches with looping and progress printing
    for i in range(0, num_views, batch_size):
        print(f"Processing batch {i//batch_size + 1} / {(num_views // batch_size) + 1}")

        # Get the current batch (from i to i + batch_size)
        sino_batch = jax.device_put(sino[i : min(i + batch_size, num_views)], jax.devices('gpu')[0])

        # Apply the rotation on this batch
        sino_batch = dm_pix.rotate(sino_batch, det_rotation, order=1, mode='constant', cval=0.0) # mode and cval are set according to the original code

        # Append the rotated batch
        sino_batches_list.append(sino_batch)

    sino_batches = jnp.concatenate(sino_batches_list, axis=0)
    sino_batches = np.array(sino_batches)
    if weights is None:
        return sino_batches
    print("correct_det_rotation: weights provided by the user. Please note that zero weight entries might become non-zero after tilt angle correction.")
    weights = scipy.ndimage.rotate(weights, np.rad2deg(det_rotation), axes=(1,2), reshape=False, order=3)
    print(f'no batch after:{mbirjax.get_memory_stats()}')
    return sino_batches, weights


def main():


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
        compute_sino_and_params_rotat(dataset_dir,
                                                        downsample_factor=downsample_factor,
                                                        subsample_view_factor=subsample_view_factor)

    print(f"Preprocessing time: {time.time() - time_start:.2f} seconds")

if __name__ == '__main__':
    main()

