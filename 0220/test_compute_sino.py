import mbirjax
from mbirjax.preprocess.nsi import load_scans_and_params
import time
import pprint as pp
import mbirjax.preprocess as preprocess
import numpy as np
import os
import sys
source_path = "/home/li5273/PycharmProjects/mbirjax/mbirjax"
if source_path not in sys.path:
    sys.path.insert(0, source_path)


os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.98'

pp = pp.PrettyPrinter(indent=4)


def compute_sino_and_params_compute_sino(dataset_dir,
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

    print("\n\n########## Computing sinogram from object, blank, and dark scans ...")
    time0 = time.time()
    mbirjax.get_memory_stats(print_results=True)
    sino, defective_pixel_list = \
            compute_sino_transmission_test(obj_scan, blank_scan, dark_scan, defective_pixel_list)
    del obj_scan, blank_scan, dark_scan # delete scan images to save memory
    mbirjax.get_memory_stats(print_results=True)
    print(f"time to compute sino = {time.time()-time0:.2f} seconds")
    print(f'sino shape = {sino.shape}')

    return sino, cone_beam_params, optional_params


def compute_sino_transmission_test(obj_scan, blank_scan, dark_scan, defective_pixel_list=None, correct_defective_pixels=True):
    """
    Compute sinogram from object, blank, and dark scans.

    This function computes sinogram by taking the negative log of the attenuation estimate.
    It can also take in a list of defective pixels and correct those pixel values.
    The invalid sinogram entries are the union of defective pixel entries and sinogram entries with values of inf or Nan.

    Args:
        obj_scan (ndarray, float): 3D object scan with shape (num_views, num_det_rows, num_det_channels).
        blank_scan (ndarray, float): [Default=None] 3D blank scan with shape (num_blank_scans, num_det_rows, num_det_channels). When num_blank_scans>1, the pixel-wise mean will be used as the blank scan.
        dark_scan (ndarray, float): [Default=None] 3D dark scan with shape (num_dark_scans, num_det_rows, num_det_channels). When num_dark_scans>1, the pixel-wise mean will be used as the dark scan.
        defective_pixel_list (optional, list(tuple)): A list of tuples containing indices of invalid sinogram pixels, with the format (view_idx, row_idx, channel_idx) or (detector_row_idx, detector_channel_idx).
            If None, then the invalid pixels will be identified as sino entries with inf or Nan values.
        correct_defective_pixels (optioonal, boolean): [Default=True] If true, the defective sinogram entries will be automatically corrected with `mbirjax.preprocess.interpolate_defective_pixels()`.

    Returns:
        2-element tuple containing:
        - **sino** (*ndarray, float*): Sinogram data with shape (num_views, num_det_rows, num_det_channels).
        - **defective_pixel_list** (list(tuple)): A list of tuples containing indices of invalid sinogram pixels, with the format (view_idx, row_idx, channel_idx) or (detector_row_idx, detector_channel_idx).

    """
    # take average of multiple blank/dark scans, and expand the dimension to be the same as obj_scan.
    blank_scan = 0 * obj_scan + np.mean(blank_scan, axis=0, keepdims=True)
    dark_scan = 0 * obj_scan + np.mean(dark_scan, axis=0, keepdims=True)

    obj_scan = obj_scan - dark_scan
    blank_scan = blank_scan - dark_scan

    #### compute the sinogram.
    # suppress warnings in np.log(), since the defective sino entries will be corrected.
    with np.errstate(divide='ignore', invalid='ignore'):
        sino = -np.log(obj_scan / blank_scan)

    # set the sino pixels corresponding to the provided defective list to 0.0
    if defective_pixel_list is None:
        defective_pixel_list = []
    else:    # if provided list is not None
        for defective_pixel_idx in defective_pixel_list:
            if len(defective_pixel_idx) == 2:
                (r,c) = defective_pixel_idx
                sino[:,r,c] = 0.0
            elif len(defective_pixel_idx) == 3:
                (v,r,c) = defective_pixel_idx
                sino[v,r,c] = 0.0
            else:
                raise Exception("compute_sino_transmission: index information in defective_pixel_list cannot be parsed.")

    # set NaN sino pixels to 0.0
    nan_pixel_list = list(map(tuple, np.argwhere(np.isnan(sino)) ))
    for (v,r,c) in nan_pixel_list:
        sino[v,r,c] = 0.0

    # set Inf sino pixels to 0.0
    inf_pixel_list = list(map(tuple, np.argwhere(np.isinf(sino)) ))
    for (v,r,c) in inf_pixel_list:
        sino[v,r,c] = 0.0

    # defective_pixel_list = union{input_defective_pixel_list, nan_pixel_list, inf_pixel_list}
    defective_pixel_list = list(set().union(defective_pixel_list,nan_pixel_list,inf_pixel_list))

    if correct_defective_pixels:
        print("Interpolate invalid sinogram entries.")
        sino, defective_pixel_list = interpolate_defective_pixels_test(sino, defective_pixel_list)
    else:
        if defective_pixel_list:
            print("Invalid sino entries detected! Please correct then manually or with function `mbirjax.preprocess.interpolate_defective_pixels()`.")
    return sino, defective_pixel_list


def interpolate_defective_pixels_test(sino, defective_pixel_list):
    """
    Interpolates defective sinogram entries with the mean of neighboring pixels.

    Args:
        sino (ndarray, float): Sinogram data with 3D shape (num_views, num_det_rows, num_det_channels).
        defective_pixel_list (list(tuple)): A list of tuples containing indices of invalid sinogram pixels, with the format (detector_row_idx, detector_channel_idx) or (view_idx, detector_row_idx, detector_channel_idx).
    Returns:
        2-element tuple containing:
        - **sino** (*ndarray, float*): Corrected sinogram data with shape (num_views, num_det_rows, num_det_channels).
        - **defective_pixel_list** (*list(tuple)*): Updated defective_pixel_list with the format (detector_row_idx, detector_channel_idx) or (view_idx, detector_row_idx, detector_channel_idx).
    """
    defective_pixel_list_new = []
    num_views, num_det_rows, num_det_channels = sino.shape
    weights = np.ones((num_views, num_det_rows, num_det_channels), dtype=np.float32) # could even be int since it's a binary mask

    for defective_pixel_idx in defective_pixel_list:
        if len(defective_pixel_idx) == 2:
            (r,c) = defective_pixel_idx
            weights[:,r,c] = 0.0
        elif len(defective_pixel_idx) == 3:
            (v,r,c) = defective_pixel_idx
            weights[v,r,c] = 0.0
        else:
            raise Exception("replace_defective_with_mean: index information in defective_pixel_list cannot be parsed.")

    for defective_pixel_idx in defective_pixel_list:
        if len(defective_pixel_idx) == 2:
            v_list = list(range(num_views))
            (r,c) = defective_pixel_idx
        elif len(defective_pixel_idx) == 3:
            (v,r,c) = defective_pixel_idx
            v_list = [v,]

        r_min, r_max = max(r-1, 0), min(r+2, num_det_rows)
        c_min, c_max = max(c-1, 0), min(c+2, num_det_channels)
        for v in v_list:
            # Perform interpolation when there are non-defective pixels in the neighborhood
            if np.sum(weights[v,r_min:r_max,c_min:c_max]) > 0:
                sino[v,r,c] = np.average(sino[v,r_min:r_max,c_min:c_max],
                                         weights=weights[v,r_min:r_max,c_min:c_max])
            # Corner case: all the neighboring pixels are defective
            else:
                print(f"Unable to correct sino entry ({v},{r},{c})! All neighborhood values are defective!")
                defective_pixel_list_new.append((v,r,c))
    return sino, defective_pixel_list_new

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
    compute_sino_and_params_compute_sino(dataset_dir,
                                                    downsample_factor=downsample_factor,
                                                    subsample_view_factor=subsample_view_factor)

print(f"Preprocessing time: {time.time() - time_start:.2f} seconds")


