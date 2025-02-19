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
from jax import jit
from compute_sino import compute_sino_and_params_compute_sino
source_path = "/home/li5273/PycharmProjects/mbirjax/mbirjax"
if source_path not in sys.path:
    sys.path.insert(0, source_path)


os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.98'

pp = pp.PrettyPrinter(indent=4)


def compute_sino_and_params_corr_bg(dataset_dir,
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


    sino, cone_beam_params, optional_params = \
    compute_sino_and_params_compute_sino(dataset_dir,
                                                    downsample_factor=downsample_factor,
                                                    subsample_view_factor=subsample_view_factor)


    print("\n\n########## Correcting background offset to the sinogram from edge pixels ...")
    time0 = time.time()
    mbirjax.get_memory_stats(print_results=True)
    background_offset = estimate_background_offset_jax(sino)
    print("background_offset = ", background_offset)
    sino = sino - background_offset
    mbirjax.get_memory_stats(print_results=True)
    print(f"time to correct background offset = {time.time()-time0:.2f} seconds")

    return sino, cone_beam_params, optional_params

@jit
def estimate_background_offset_jax(sino, edge_width=9):
    """
    Estimate background offset of a sinogram using JAX for GPU acceleration.

    Args:
        sino (jax.numpy.ndarray): Sinogram data with shape (num_views, num_det_rows, num_det_channels).
        edge_width (int, optional): Width of the edge regions in pixels. Must be an odd integer >= 3.

    Returns:
        offset (float): Background offset value.
    """
    sino = jnp.asarray(sino)
    print('1')
    mbirjax.get_memory_stats(print_results=True)
    if (edge_width % 2 == 0):
        edge_width += 1
        warnings.warn(f"edge_width of background regions should be an odd number! Setting edge_width to {edge_width}.")
    if (edge_width < 3):
        edge_width = 3
        warnings.warn("edge_width of background regions should be >= 3! Setting edge_width to 3.")

    _, _, num_det_channels = sino.shape
    # Extract edge regions directly from the sinogram (without computing a full median)
    median_top = jnp.median(jnp.median(sino[:, :edge_width, :], axis=0), axis=0) # Top edge
    median_left = jnp.median(jnp.median(sino[:, :, :edge_width], axis=0), axis=0)  # Left edge
    median_right = jnp.median(jnp.median(sino[:, :, num_det_channels-edge_width:], axis=0), axis=0)  # Right edge
    print('2')
    mbirjax.get_memory_stats(print_results=True)
    # Compute final offset as median of the three regions
    offset = jnp.median(jnp.array([median_top, median_left, median_right]))

    return offset

def estimate_background_offset(sino, option=0, edge_width=9):
    """
    Estimate background offset of a sinogram from the edge pixels.

    This function estimates the background offset when no object is present by computing a robust centroid estimate using `edge_width` pixels along the edge of the sinogram across views.
    Typically, this estimate is subtracted from the sinogram so that air is reconstructed as approximately 0.

    Args:
        sino (float, ndarray): Sinogram data with 3D shape (num_views, num_det_rows, num_det_channels).
        option (int, optional): [Default=0] Option of algorithm used to calculate the background offset.
        edge_width(int, optional): [Default=9] Width of the edge regions in pixels. It must be an odd integer >= 3.
    Returns:
        offset (float): Background offset value.
    """

    # Check validity of edge_width value
    assert(isinstance(edge_width, int)), "edge_width must be an integer!"
    if (edge_width % 2 == 0):
        edge_width = edge_width+1
        warnings.warn(f"edge_width of background regions should be an odd number! Setting edge_width to {edge_width}.")

    if (edge_width < 3):
        warnings.warn("edge_width of background regions should be >= 3! Setting edge_width to 3.")
        edge_width = 3

    _, _, num_det_channels = sino.shape

    print('3')
    mbirjax.get_memory_stats(print_results=True)

    # calculate mean sinogram
    sino_median=np.median(sino, axis=0)

    print('4')
    mbirjax.get_memory_stats(print_results=True)
    # offset value of the top edge region.
    # Calculated as median([median value of each horizontal line in top edge region])
    median_top = np.median(np.median(sino_median[:edge_width], axis=1))

    # offset value of the left edge region.
    # Calculated as median([median value of each vertical line in left edge region])
    median_left = np.median(np.median(sino_median[:, :edge_width], axis=0))

    # offset value of the right edge region.
    # Calculated as median([median value of each vertical line in right edge region])
    median_right = np.median(np.median(sino_median[:, num_det_channels-edge_width:], axis=0))

    # offset = median of three offset values from top, left, right edge regions.
    offset = np.median([median_top, median_left, median_right])
    return offset


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
        compute_sino_and_params_corr_bg(dataset_dir,
                                                        downsample_factor=downsample_factor,
                                                        subsample_view_factor=subsample_view_factor)

    print(f"Preprocessing time: {time.time() - time_start:.2f} seconds")

if __name__ == '__main__':
    main()
