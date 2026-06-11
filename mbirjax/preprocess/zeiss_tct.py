import os, sys
from operator import itemgetter
import numpy as np
import jax
import jax.numpy as jnp
import warnings
import mbirjax as mj
import mbirjax.preprocess as mjp
import pprint
import logging
import olefile
import struct
from pathlib import Path
from scipy.ndimage import binary_erosion
pp = pprint.PrettyPrinter(indent=4)
logger = logging.getLogger(__name__)


def compute_sino_and_params(dataset_dir, crop_pixels_sides=0, crop_pixels_top=0, crop_pixels_bottom=0, alu_unit='mm', verbose=1):
    """
    Load Zeiss sinogram data and prepare arrays ana parameters for TranslationModel reconstruction.

    This function computes the sinogram and geometry parameters from a Zeiss scan directory. It performs the following:

    1. Load object, blank, and dark scans, and geometry parameters from the dataset.
    2. Computes the sinogram from the scan images.
    3. Applies background offset correction.

    Args:
        dataset_dir (str): Path to the Zeiss scan directory. Expected structure.
            - "obj_scan" (a subfolder containing the object scan)
            - "blank_scan" (a subfolder containing the blank scan)
            - "dark_scan" (a subfolder containing the dark scan)
        crop_pixels_sides (int, optional): Pixels to crop from each side of the sinogram. Defaults to None.
        crop_pixels_top (int, optional): Pixels to crop from top of the sinogram. Defaults to None.
        crop_pixels_bottom (int, optional): Pixels to crop from bottom of the sinogram. Defaults to None.
        alu_unit (str, optional): The physical unit used to define 1 ALU (Arbitrary Length Unit). Defaults to 'mm'.
            Supported units input: 'um', 'mm', 'cm', 'm'.
        verbose (int, optional): Verbosity level. Defaults to 1.

    Returns:
        tuple: (sino, translation_params, optional_params)
            - ``sino`` (jax.numpy.ndarray): Sinogram of shape (num_views, num_det_rows, num_channels)
            - ``translation_params`` (dict): Parameters for initializing TranslationModel.
            - ``optional_params`` (dict): Parameters to be passed via ``set_params``.
            -  ``weights`` (numpy.ndarray): A 3D array of weights with the same shape as the sinogram.

    Example:
        .. code-block:: python

            # Get data and reconstruction parameters
            sino, translation_params, optional_params = mbirjax.preprocess.zeiss.compute_sino_and_params(dataset_dir)

            # Create the model and set parameters
            tct_model = mbirjax.TranslationModel(**translation_params)
            tct_model.set_params(**optional_params)
            tct_model.set_params(sharpness=sharpness, verbose=1)

            # Run reconstruction
            recon, recon_dict = tct_model.recon(sino)
    """
    if verbose > 0:
        print("\n\n########## Loading object, blank, dark scans, and geometry parameters from Zeiss dataset directory")
    obj_scan, blank_scan, dark_scan, zeiss_params = load_scans_and_params(dataset_dir, verbose=verbose)

    translation_params, optional_params = convert_zeiss_to_mbirjax_params(zeiss_params,
                                                                          crop_pixels_sides=crop_pixels_sides,
                                                                          crop_pixels_top=crop_pixels_top,
                                                                          crop_pixels_bottom=crop_pixels_bottom,
                                                                          alu_unit=alu_unit)

    if verbose > 0:
        print("\n\n########## Cropping scans")
    ### crop the scans based on input params
    obj_scan, blank_scan, dark_scan, defective_pixel_array = mjp.crop_view_data(obj_scan, blank_scan, dark_scan,
                                                                                crop_pixels_sides=crop_pixels_sides,
                                                                                crop_pixels_top=crop_pixels_top,
                                                                                crop_pixels_bottom=crop_pixels_bottom)

    # Generate weights that exclude dark boundary regions
    weights = compute_weight(blank_scan, obj_scan)

    if verbose > 0:
        print("\n\n########## Computing sinogram from object, blank, and dark scans")
    sino = mjp.compute_sino_transmission(obj_scan, blank_scan, dark_scan, defective_pixel_array)

    if verbose > 0:
        print("\n\n########## Correcting sinogram data to account for background offset and detector rotation")
    sino = mjp.correct_background_offset(sino)

    if verbose > 0:
        print('obj_scan shape = ', obj_scan.shape)
        print('blank_scan shape = ', blank_scan.shape)
        print('dark_scan shape = ', dark_scan.shape)

    return sino, translation_params, optional_params, weights


def load_scans_and_params(dataset_dir, verbose=1):
    """
    Load the object scan, blank scan, dark scan, and geometry from a Zeiss scan directory.

    Args:
        dataset_dir (str): Path to a Zeiss scan directory. Expected structure:

            - ``obj_scan`` — subfolder containing the object scan
            - ``blank_scan`` — subfolder containing the blank scan
            - ``dark_scan`` — subfolder containing the dark scan

        verbose (int, optional): Verbosity level. Defaults to ``1``.

    Returns:
        tuple: ``(obj_scan, blank_scan, dark_scan, zeiss_params)``

            - ``obj_scan`` (numpy.ndarray): 3D object scan with shape ``(num_views, num_det_rows, num_channels)``.
            - ``blank_scan`` (numpy.ndarray): 3D blank scan with shape ``(1, num_det_rows, num_channels)``.
            - ``dark_scan`` (numpy.ndarray): 3D dark scan with shape ``(1, num_det_rows, num_channels)``.
            - ``zeiss_params`` (dict): Required parameters for ``convert_zeiss_to_mbirjax_params`` (e.g., geometry vectors, spacings, and angles).
    """
    ### automatically parse the paths to Zeiss scans from dataset—dir
    obj_scan_dir, blank_scan_dir, dark_scan_dir = _parse_filenames_from_dataset_dir(dataset_dir)

    if verbose > 0:
        print("The following files will be used to compute the Zeiss reconstruction:\n",
              f"    - Object scan directory: {obj_scan_dir}\n",
              f"    - Blank scan directory: {blank_scan_dir}\n",
              f"    - Dark scan directory: {dark_scan_dir}\n",)

    # Read object scans and metadata
    obj_scan, zeiss_params = read_xrm_dir(obj_scan_dir)

    # Read blank scans
    blank_scan, _ = read_xrm_dir(blank_scan_dir)

    # Read dark scans
    # TODO: If there is no dark scan available, using an array of all 0s.
    if os.path.isdir(dark_scan_dir) and len(os.listdir(dark_scan_dir)) > 0:
        dark_scan, _ = read_xrm_dir(dark_scan_dir)
    else:
        dark_scan = np.zeros(blank_scan.shape)

    # Flip the scans vertically
    # TODO: It seems that we need to flip the scan to get the correct object orientation
    if verbose > 0:
        print("Flipping scans vertically")
    obj_scan = np.flip(obj_scan, axis=1)
    blank_scan = np.flip(blank_scan, axis=1)
    dark_scan = np.flip(dark_scan, axis=1)

    try:
        # Get the list of measurement axis names and units
        axis_names = zeiss_params.get("axis_names")  # This is a list of names: ['Sample X', 'Sample Y', ..., 'CCD_X', ...]
        axis_units = zeiss_params.get("axis_units")  # This is a list of units: ['um', 'um', ..., ]

        # Get source to iso distance
        source_iso_dist = zeiss_params["source_iso_dist"]
        source_iso_dist = float(np.abs(source_iso_dist))
        source_iso_dist_index = get_index_in_list(axis_names, 'Source Z')
        source_iso_dist_unit = axis_units[source_iso_dist_index] if source_iso_dist_index > -1 else 'mm'

        # Get iso to detector distance
        iso_det_dist = zeiss_params["iso_det_dist"]
        iso_det_dist = float(np.abs(iso_det_dist)) if iso_det_dist is not None else 0.0
        iso_det_dist_unit = source_iso_dist_unit

        # Get detector pixel pitch
        det_pixel_pitch = zeiss_params["det_pixel_pitch"]
        det_pixel_pitch = float(np.abs(det_pixel_pitch))

        # Zeiss detector pixel has equal width and height
        delta_det_row = det_pixel_pitch
        delta_det_channel = det_pixel_pitch
        delta_det_index = get_index_in_list(axis_names, 'CCD_X')
        delta_det_row_unit = axis_units[delta_det_index] if delta_det_index > -1 else 'um'
        delta_det_channel_unit = delta_det_row_unit

        # Get pixel pitch at iso
        iso_pixel_pitch = zeiss_params["iso_pixel_pitch"]
        iso_pixel_pitch = float(np.abs(iso_pixel_pitch))
        iso_pixel_pitch_index = get_index_in_list(axis_names, 'Sample X')
        iso_pixel_pitch_unit = axis_units[iso_pixel_pitch_index] if iso_pixel_pitch_index > -1 else 'um'

        # Get object positions in x, y, z axis
        # TODO: It seems that purdue p dataset and purdue BGA dataset use the different coordinate system
        # The scanner uses a coordinate system different from MBIRJAX
        # Axis mapping:
        #   Scanner z-axis -> MBIRJAX x-axis or -x-axis not sure
        #   Scanner x-axis -> MBIRJAX y-axis or -y-axis not sure
        #   Scanner y-axis -> MBIRJAX z-axis or -z-axis not sure
        object_x_positions = -np.array(zeiss_params['x_positions'], dtype=float).ravel()
        object_y_positions = np.array(zeiss_params['z_positions'], dtype=float).ravel()
        object_z_positions = np.array(zeiss_params['y_positions'], dtype=float).ravel()

        object_x_position_index = get_index_in_list(axis_names, 'Sample X')
        object_y_position_index = get_index_in_list(axis_names, 'Sample Y')
        object_z_position_index = get_index_in_list(axis_names, 'Sample Z')

        object_x_position_unit = axis_units[object_x_position_index] if object_x_position_index > -1 else 'um'
        object_y_position_unit = axis_units[object_y_position_index] if object_y_position_index > -1 else 'um'
        object_z_position_unit = axis_units[object_z_position_index] if object_z_position_index > -1 else 'um'

        # Get sinogram per-view shifts
        x_shifts = zeiss_params["x_shifts"]
        y_shifts = zeiss_params["y_shifts"]

    except ValueError as e:
        print("Unable to determine units for geometry parameters; cannot safely convert to mbirjax format.")
        raise e

    # Get optical Magnification
    opt_mag = zeiss_params["opt_mag"]
    opt_mag = 1 if opt_mag is None else opt_mag

    # Get dimensions of radiograph
    num_views = zeiss_params["num_views"]
    num_det_channels = zeiss_params["num_det_channels"]
    num_det_rows = zeiss_params["num_det_rows"]

    # Get detector offset
    # MBIRJAX has the reverse convention for the channel shift
    center_shift = zeiss_params.get("center_shift")

    if center_shift is None:
        det_channel_offset = 0.0
    else:
        det_channel_offset = -center_shift

    det_row_offset = 0.0  # There doesn't appear to be a Zeiss parameter for detector row offset

    zeiss_params = {
        'source_iso_dist': source_iso_dist,
        'iso_det_dist': iso_det_dist,
        'delta_det_channel': delta_det_channel,
        'delta_det_row': delta_det_row,
        'iso_pixel_pitch': iso_pixel_pitch,
        'opt_mag': opt_mag,
        'num_views': num_views,
        'num_det_channels': num_det_channels,
        'num_det_rows': num_det_rows,
        'det_row_offset': det_row_offset,
        'det_channel_offset': det_channel_offset,
        'object_x_positions': object_x_positions,
        'object_y_positions': object_y_positions,
        'object_z_positions': object_z_positions,
        'source_iso_dist_unit': source_iso_dist_unit,
        'iso_det_dist_unit': iso_det_dist_unit,
        'delta_det_row_unit': delta_det_row_unit,
        'delta_det_channel_unit': delta_det_channel_unit,
        'iso_pixel_pitch_unit': iso_pixel_pitch_unit,
        'object_x_position_unit': object_x_position_unit,
        'object_y_position_unit': object_y_position_unit,
        'object_z_position_unit': object_z_position_unit,
        'x_shifts': x_shifts,
        'y_shifts': y_shifts,
    }

    if verbose > 0:
        print("############ Zeiss geometry parameters ############")
        print(f"Source to iso distance: {source_iso_dist} [{source_iso_dist_unit}]")
        print(f"Iso to detector distance: {iso_det_dist} [{iso_det_dist_unit}]")
        print(f"Detector pixel pitch: {det_pixel_pitch:.3f} [{delta_det_row_unit}]")
        print(f"Pixel pitch at iso: {iso_pixel_pitch} [{iso_pixel_pitch_unit}]")
        print(f"Optical magnification: {opt_mag}")
        print(f"Number of views: {num_views}")
        print(f"Detector size: (num_det_rows, num_det_channels) = ({num_det_rows}, {num_det_channels})")
        print("############ End Zeiss geometry parameters ############")
    ### END load Zeiss parameters from scan data

    return obj_scan, blank_scan, dark_scan, zeiss_params


def convert_zeiss_to_mbirjax_params(zeiss_params, crop_pixels_sides=0, crop_pixels_top=0, crop_pixels_bottom=0, alu_unit='mm'):
    """
    Convert geometry parameters from zeiss into mbirjax format, including modifications to reflect crop.

    Args:
        zeiss_params (dict): Required Zeiss geometry parameters for reconstruction.
        crop_pixels_sides (int, optional): The number of pixels to crop from each side of the sinogram. Defaults to 0.
        crop_pixels_top (int, optional): The number of pixels to crop from top of the sinogram. Defaults to 0.
        crop_pixels_bottom (int, optional): The number of pixels to crop from bottom of the sinogram. Defaults to 0.
        alu_unit (str, optional): The physical unit used to define 1 ALU (Arbitrary Length Unit). Defaults to 'mm'.
            Supported units input: 'um', 'mm', 'cm', 'm'.

    Returns:
        translation_params (dict): Required parameters for the TranslationModel constructor.
        optional_params (dict): Additional TranslationModel parameters to be set using set_params()
    """
    # Get zeiss parameters
    source_iso_dist, iso_det_dist, source_iso_dist_unit, iso_det_dist_unit = itemgetter('source_iso_dist', 'iso_det_dist', 'source_iso_dist_unit', 'iso_det_dist_unit')(zeiss_params)
    delta_det_channel, delta_det_row, delta_det_channel_unit, delta_det_row_unit = itemgetter('delta_det_channel', 'delta_det_row', 'delta_det_channel_unit', 'delta_det_row_unit')(zeiss_params)
    iso_pixel_pitch, iso_pixel_pitch_unit = itemgetter('iso_pixel_pitch', 'iso_pixel_pitch_unit')(zeiss_params)
    opt_mag = itemgetter('opt_mag')(zeiss_params)
    num_det_rows, num_det_channels = itemgetter('num_det_rows', 'num_det_channels')(zeiss_params)
    object_x_positions, object_x_position_unit = itemgetter('object_x_positions', 'object_x_position_unit')(zeiss_params)
    object_y_positions, object_y_position_unit = itemgetter('object_y_positions', 'object_y_position_unit')(zeiss_params)
    object_z_positions, object_z_position_unit = itemgetter('object_z_positions', 'object_z_position_unit')(zeiss_params)
    det_row_offset, det_channel_offset = itemgetter('det_row_offset', 'det_channel_offset')(zeiss_params)

    # Create unit conversion table for all units used in the xrm files
    unit_conversion = {'um': 1.0, 'mm': 1000.0, 'cm': 1e4, 'm': 1e6}

    # Define 1 ALU as 1 unit of alu_unit
    alu_value = 1

    # Convert physical units to ALU
    source_iso_dist *= unit_conversion[source_iso_dist_unit] / unit_conversion[alu_unit]
    iso_det_dist *= unit_conversion[iso_det_dist_unit] / unit_conversion[alu_unit]
    delta_det_channel *= unit_conversion[delta_det_channel_unit] / unit_conversion[alu_unit]
    delta_det_row *= unit_conversion[delta_det_row_unit] / unit_conversion[alu_unit]
    iso_pixel_pitch *= unit_conversion[iso_pixel_pitch_unit] / unit_conversion[alu_unit]
    object_x_positions *= unit_conversion[object_x_position_unit] / unit_conversion[alu_unit]
    object_y_positions *= unit_conversion[object_y_position_unit] / unit_conversion[alu_unit]
    object_z_positions *= unit_conversion[object_z_position_unit] / unit_conversion[alu_unit]

    # Compute default value of source to detector distance
    source_detector_dist = source_iso_dist + iso_det_dist

    # Compute translation vectors
    translation_vectors = calc_translation_vec_params(object_x_positions, object_y_positions, object_z_positions)

    # Make conversions for optical magnification
    # In this case, the "detector" is actually a scintillator
    if opt_mag is not None:
        # Compute total magnification = (optical magnification) * (magnification to scintillator)
        scintillator_mag = source_detector_dist / source_iso_dist
        magnification = opt_mag * scintillator_mag
    else:
        magnification = 1.0

    # Compute source to equivalent quantities accounting for total magnification
    source_detector_dist = magnification * source_iso_dist
    delta_det_channel = magnification * iso_pixel_pitch
    delta_det_row = magnification * iso_pixel_pitch

    # Adjust detector size params w.r.t. cropping arguments
    num_det_rows = num_det_rows - (crop_pixels_top + crop_pixels_bottom)
    num_det_channels = num_det_channels - 2 * crop_pixels_sides

    # Convert offset parameters to ALU: This assumes that the det_channel_offset and det_row_offset have units of pixels.
    det_channel_offset *= delta_det_channel
    det_row_offset *= delta_det_row

    # Calculate recon_shape, delta_voxel, and voxel_row_aspect parameters
    num_views = len(translation_vectors)
    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    recon_shape, delta_voxel, voxel_row_aspect = mj.utilities.calc_tct_recon_params(source_detector_dist, source_iso_dist, delta_det_row, delta_det_channel, sinogram_shape, translation_vectors)

    # Create a dictionary to store MBIR parameters
    translation_params = dict()
    translation_params['sinogram_shape'] = sinogram_shape
    translation_params['translation_vectors'] = translation_vectors
    translation_params['source_detector_dist'] = source_detector_dist
    translation_params['source_iso_dist'] = source_iso_dist

    optional_params = dict()
    optional_params['delta_det_channel'] = delta_det_channel
    optional_params['delta_det_row'] = delta_det_row
    optional_params['delta_voxel'] = delta_voxel
    optional_params['voxel_row_aspect'] = voxel_row_aspect
    optional_params['recon_shape'] = recon_shape
    optional_params['det_row_offset'] = det_row_offset
    optional_params['det_channel_offset'] = det_channel_offset
    optional_params['alu_unit'] = alu_unit
    optional_params['alu_value'] = alu_value

    return translation_params, optional_params


######## subroutines for parsing Zeiss object scan, blank scan, and dark scan
def _parse_filenames_from_dataset_dir(dataset_dir):
    """
    Given a path to a Zeiss scan directory, automatically parse the paths to the following files and directories：
        - object scan directory
        - blank scan directory
        - dark scan directory

    Args:
        dataset_dir (string): Path to the directory containing the Zeiss scan files.

    Returns:
        4-element tuple containing:
            - obj_scan_dir (string): Path to the object scan directory
            - blank_scan_dir (string): Path to the blank scan directory
            - dark_scan_dir (string): Path to the dark scan directory
    """
    # Object scan directory
    obj_scan_dir = os.path.join(dataset_dir, "obj_scan")

    # Blank scan
    blank_scan_dir = os.path.join(dataset_dir, "blank_scan")

    # Dark scan
    dark_scan_dir = os.path.join(dataset_dir, "dark_scan")

    return obj_scan_dir, blank_scan_dir, dark_scan_dir


def _check_read(fname):
    """
    Validate the file path and ensure it has a recognized extension.

    Notes:
        Portions of this code are adapted from the DXchange library: https://github.com/data-exchange/dxchange

    Args:
        fname (str) : Path to the file to be read. Must be a string and have one of the recognized file extensions:
        ['.edf', '.tiff', '.tif', '.h5', '.hdf', '.npy', '.nc', '.xrm', '.txrm', '.txm', '.xmt', '.nxs'].


    Returns:
        str: Absolute path to the file.
    """
    known_extensions = {
        '.edf', '.tiff', '.tif', '.h5', '.hdf', '.npy', '.nc',
        '.xrm', '.txrm', '.txm', '.xmt', '.nxs'
    }

    if not isinstance(fname, str):
        logger.error('File name must be a string')
    else:
        _, ext = os.path.splitext(fname)
        ext = ext.lower()
        if ext not in known_extensions:
            logger.error('Unknown file extension')

    return os.path.abspath(fname)


def read_xrm(fname):
    """
    Read data from xrm file.

    Notes:
        Portions of this code are adapted from the DXchange library: https://github.com/data-exchange/dxchange

    Args:
        fname (str): String defining the path of file or file name.

    Returns:
        np.ndarray: Output 2D image with shape (num_det_rows, num_det_channels).
        dict: Output metadata.
    """
    fname = _check_read(fname)
    try:
        ole = olefile.OleFileIO(fname)
    except IOError:
        print('No such file or directory: %s', fname)
        return False

    # Read metadata from xrm file
    metadata = read_metadata(ole)

    # Read scan data from xrm file
    stream = ole.openstream("ImageData1/Image1")
    data = stream.read()

    # Get the data type of scan data
    data_type = _get_ole_data_type(metadata)
    data_type = data_type.newbyteorder('<')

    # Reshape the scan data into 2D array
    arr = np.reshape(
        np.frombuffer(data, data_type),
        (
            metadata["num_det_rows"],
            metadata["num_det_channels"]
        )
    )

    _log_imported_data(fname, arr)

    # Normalize the scan data
    # arr = mjp.utilities._normalize_to_float32(arr)

    ole.close()
    return arr, metadata


def read_xrm_dir(dir_path):
    """
    Read all .xrm files in a directory (filesystem order), stack into (num_views, num_det_rows, num_det_cols),
    and concatenate selected metadata.

    Notes:
        Portions of this code are adapted from the DXchange library: https://github.com/data-exchange/dxchange

    Args:
        dir_path (str) : Path to the directory to be read.

    Returns:
        np.ndarray: Output 3D image with shape (num_views, num_det_rows, num_det_channels).
        dict: Output metadata
    """
    dir_path = Path(dir_path)
    files = [p for p in dir_path.iterdir() if p.is_file()]

    # Load the scan data and metadata from first file
    proj0, md0 = read_xrm(str(files[0]))
    num_views = len(files)
    num_det_rows, num_det_channels = proj0.shape
    arr = np.empty((num_views, num_det_rows, num_det_channels), dtype=proj0.dtype)
    arr[0] = proj0

    # Load the x, y, z object positions of the first file
    x0 = md0['x_positions'][0]
    y0 = md0['y_positions'][0]
    z0 = md0['z_positions'][0]

    metadata = dict(md0)
    metadata['num_views'] = num_views
    metadata['x_positions'] = [x0]
    metadata['y_positions'] = [y0]
    metadata['z_positions'] = [z0]

    # Load the remaining files and stack them together
    for i, p in enumerate(files[1:], start=1):
        proj, md = read_xrm(str(p))
        arr[i] = proj
        metadata['x_positions'].append(md['x_positions'][0])
        metadata['y_positions'].append(md['y_positions'][0])
        metadata['z_positions'].append(md['z_positions'][0])

    _log_imported_data(str(dir_path), arr)

    # Normalize the scan data
    # arr = mjp.utilities._normalize_to_float32(arr)

    return arr, metadata


def read_txrm(file_name):
    """
    Read data from a .txrm file, a compilation of .xrm files.

    This code is adapted from the DXchange library:
    https://github.com/data-exchange/dxchange

    Reference:
    [1] DXchange library: https://github.com/data-exchange/dxchange

    Args:
        file_name (str): String defining the path of file or file name.

    Returns:
        np.ndarray: Output 3D image with shape (num_views, num_det_rows, num_det_channels).
        dict: Output metadata
    """
    file_name = _check_read(file_name)
    try:
        ole = olefile.OleFileIO(file_name)
    except IOError:
        print('No such file or directory: %s', file_name)
        return False

    metadata = read_metadata(ole)

    array_of_images = np.empty(
        (
            metadata["num_views"],
            metadata["num_det_rows"],
            metadata["num_det_channels"],
        ),
        dtype=_get_ole_data_type(metadata)
    )

    for i, idx in enumerate(range(metadata["num_views"])):
        img_string = "ImageData{}/Image{}".format(
            int(np.ceil((idx + 1) / 100.0)), int(idx + 1))
        array_of_images[i] = _read_ole_image(ole, img_string, metadata)

    _log_imported_data(file_name, array_of_images)

    ole.close()
    return array_of_images, metadata


def read_metadata(ole):
    """
    Read metadata from an xradia OLE file (.xrm, .txrm, .txm).

    Notes:
        Portions of this code are adapted from the DXchange library: https://github.com/data-exchange/dxchange

    Args:
        ole (OleFileIO instance) : An ole file to read from.

    Returns:
        dict: A dictionary of image metadata.
    """

    number_of_images = _read_ole_value(ole, "ImageInfo/NoOfImages", "<I")

    metadata = {
        'num_det_channels': _read_ole_value(ole, 'ImageInfo/ImageWidth', '<I'),
        'num_det_rows': _read_ole_value(ole, 'ImageInfo/ImageHeight', '<I'),
        'data_type': _read_ole_value(ole, 'ImageInfo/DataType', '<1I'),
        'num_views': number_of_images,
        'iso_pixel_pitch': _read_ole_value(ole, 'ImageInfo/PixelSize', '<f'),
        'det_pixel_pitch': _read_ole_value(ole, 'ImageInfo/CamPixelSize', '<f'),
        'iso_det_dist': _read_ole_value(ole, 'ImageInfo/DtoRADistance', "<f"),
        'source_iso_dist': _read_ole_value(ole,'ImageInfo/StoRADistance', "<{0}f".format(number_of_images)),
        'thetas': _read_ole_arr(
            ole, 'ImageInfo/Angles', "<{0}f".format(number_of_images)),
        'x_positions': _read_ole_arr(
            ole, 'ImageInfo/XPosition', "<{0}f".format(number_of_images)),
        'y_positions': _read_ole_arr(
            ole, 'ImageInfo/YPosition', "<{0}f".format(number_of_images)),
        'z_positions': _read_ole_arr(
            ole, 'ImageInfo/ZPosition', "<{0}f".format(number_of_images)),
        'x_shifts':_read_ole_arr(
            ole, 'alignment/x-shifts', "<{0}f".format(number_of_images)),
        'y_shifts': _read_ole_arr(
            ole, 'alignment/y-shifts', "<{0}f".format(number_of_images)),
        'current': _read_ole_value(
            ole, "ImageInfo/XrayCurrent", "<{0}f".format(number_of_images)),
        'voltage': _read_ole_value(
            ole, "ImageInfo/XrayVoltage", "<{0}f".format(number_of_images)),
        'ExpTimes': _read_ole_value(
            ole, "ImageInfo/ExpTimes", "<{0}f".format(number_of_images)),
        'center_shift': _read_ole_value(ole, "ReconSettings/CenterShift", '<f'),
        'opt_mag': _read_ole_value(ole, 'ImageInfo/OpticalMagnification', '<f'),
        'fan_angle': _read_ole_value(ole, 'ImageInfo/FanAngle', '<f'),
        'cone_angle': _read_ole_value(ole, 'ImageInfo/ConeAngle', '<f'),
        'axis_names': _read_ole_str(ole, 'PositionInfo/AxisNames'),
        'axis_units': _read_ole_str(ole, 'PositionInfo/AxisUnits')
    }

    return metadata


def _log_imported_data(fname, arr):
    """
    Log information about imported data.

    Notes:
        Portions of this code are adapted from the DXchange library: https://github.com/data-exchange/dxchange

    Args:
        fname (str) : Path of the file from which data was imported.
        arr (np.ndarray) : Array containing the image data.
    """
    logger.debug('Data shape & type: %s %s', arr.shape, arr.dtype)
    logger.info('Data successfully imported: %s', fname)


def get_index_in_list(input_list, target):
    """
    Find the index of target in the given list.
    Return -1 if not present.
    """
    if target in input_list:
        idx = input_list.index(target)
    else:
        idx = -1  # or None

    return idx


def _get_ole_data_type(metadata, datatype=None):
    """
    Determine the Numpy data type for image data stored in a Zeiss OLE (.xrm, .txrm, .txm) file.

    Notes:
        Portions of this code are adapted from the DXchange library: https://github.com/data-exchange/dxchange

    Args:
        metadata (dict) : Dictionary containing metadata extracted from the OLE file.
                          Must include the key "data_type" which is an integer code indicating the pixel data format.
        datatype (int, optional): Integer code for the data type. If None, the function uses `metadata["data_type"]`.

    Returns:
        np.dtype: The data type of the image data.
    """
    # 10 float; 5 uint16 (unsigned 16-bit (2-byte) integers)
    if datatype is None:
        datatype = metadata["data_type"]
    if datatype == 10:
        return np.dtype(np.float32)
    elif datatype == 5:
        return np.dtype(np.uint16)
    else:
        raise Exception("Unsupport XRM datatype: %s" % str(datatype))


def _read_ole_struct(ole, label, struct_fmt):
    """
    Reads the struct associated with label in an ole file

    Notes:
        Portions of this code are adapted from the DXchange library: https://github.com/data-exchange/dxchange

    Args:
        ole (OleFileIO) : An ole file to read from.
        label (str) : Label associated with the OLE file.
        struct_fmt (str) : Format of the OLE file.

    Returns:
        tuple or None: A tuple of unpacked values from the binary stream if the label exists.
    """
    value = None
    if ole.exists(label):
        stream = ole.openstream(label)
        data = stream.read()
        value = struct.unpack(struct_fmt, data)
    return value


def _read_ole_value(ole, label, struct_fmt):
    """
    Reads the value associated with label in an ole file

    Notes:
        Portions of this code are adapted from the DXchange library: https://github.com/data-exchange/dxchange

    Args:
        ole (OleFileIO) : An ole file to read from.
        label (str) : Label associated with the OLE file.
        struct_fmt (str) : Format of the OLE file.

    Returns:
        int or float : The unpacked scalar value from the binary stream if the label exists,
    """
    value = _read_ole_struct(ole, label, struct_fmt)
    if value is not None:
        value = value[0]
    return value


def _read_ole_arr(ole, label, struct_fmt):
    """
    Reads the numpy array associated with label in an ole file

    Notes:
        Portions of this code are adapted from the DXchange library: https://github.com/data-exchange/dxchange

    Args:
        ole (OleFileIO) : An ole file to read from.
        label (str) : Label associated with the OLE file.
        struct_fmt (str) : Format of the OLE file.

    Returns:
        np.ndarray: The unpacked numpy array from the binary stream if the label exists.
    """
    arr = _read_ole_struct(ole, label, struct_fmt)
    if arr is not None:
        arr = np.array(arr)
    return arr


def _read_ole_image(ole, label, metadata, datatype=None):
    """
    Reads the image data associated with label in an ole file

    Notes:
        Portions of this code are adapted from the DXchange library: https://github.com/data-exchange/dxchange

    Args:
        ole (OleFileIO) : An ole file to read from.
        label (str) : Label associated with the OLE file.
        metadata (dict) : Dictionary containing metadata extracted from the OLE file.
        datatype: Data type of the image data. Defaults to None.

    Returns:
        np.ndarray: Output 2D image with shape (num_det_rows, num_det_channels).
    """
    stream = ole.openstream(label)
    data = stream.read()
    data_type = _get_ole_data_type(metadata, datatype)
    data_type = data_type.newbyteorder('<')
    image = np.reshape(
        np.frombuffer(data, data_type),
        (metadata["num_det_rows"], metadata["num_det_channels"], )
    )
    return image


def _read_ole_str(ole, label):
    """
    Reads the string associated with label in an ole file

    Args:
        ole (OleFileIO) : An ole file to read from.
        label (str) : Label associated with the OLE file.

    Returns:
        list: A list contain all the strings from the binary stream if the label exists
    """
    str = None
    if ole.exists(label):
        stream = ole.openstream(label)
        data = stream.read()
        str = [name.decode('utf-8') for name in data.split(b'\x00') if name]
    return str


######## END subroutines for parsing Zeiss object scan, blank scan, and dark scan


######## subroutines for Zeiss-MBIR parameter conversion
def calc_translation_vec_params(obj_x_positions, obj_y_positions, obj_z_positions):
    """
    Calculate the translation geometry parameters: translation_vectors

    Args:
        obj_x_positions (tuple): The object positions of all views in x-axis with shape (num_views,)
        obj_y_positions (tuple): The object positions of all views in y-axis with shape (num_views,)
        obj_z_positions (tuple): The object positions of all views in z-axis with shape (num_views,)

    Returns:
        translation_vectors (np.ndarray): Array of shape (num_views, 3) with translation vectors [dx, dy, dz]
    """
    # Stack the object positions of all views in x, y, z axis into a 3D array of shape (number of views, 3)
    obj_xyz_positions = np.stack([obj_x_positions, obj_y_positions, obj_z_positions], axis=1)

    # Calculate the max and min of object positions along the x, y, z axis
    max_obj_xyz_positions = np.max(obj_xyz_positions, axis=0)
    min_obj_xyz_positions = np.min(obj_xyz_positions, axis=0)

    # Set the object position at the center to be the midpoint of the extremes along the x, y, z axis
    center_xyz_position = (max_obj_xyz_positions + min_obj_xyz_positions) / 2

    # Compute the translation vectors in um
    translation_vectors = center_xyz_position - obj_xyz_positions

    return translation_vectors
######## END subroutines for Zeiss-MBIR parameter conversion


def correct_sino_shifts(sino, zeiss_params):
    """
    Align each sinogram view based on the per-view projection offset.

    The xrm file stores the horizontal (x-shift) and vertical (y-shift) offsets for each projection.
    This function compensates the object's vibration by shifting each view of the sinogram accordingly.

    Coordinate convention (view from source):
      • x-shift: shift should be applied in the horizontal direction. Positive x-shift means the view should be shifted to the right
      • y-shift: shift should be applied in the vertical direction. Positive y-shift means the view should be shifted down

    For each view, the function:
      1. Reads the corresponding offset (x-shift, y-shift)
      2. Translate the view based on the (x-shift, y-shift)

    Padding is added before shifting to handle image boundary,
    and the padding is removed afterward.

    Args:
        sino (numpy array or jax array): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels).
        zeiss_params (dict): parameters stored in Zeiss xrm file.

    Returns:
        corrected_sino (numpy array or jax array): 3D sinogram data after alignment
    """
    # Get sinogram view offset
    # TODO: Currrently I assume that the view offset has units of pixels
    #   I test it and think that this assumption is correct
    sino_x_offset = zeiss_params["x_shifts"]
    sino_y_offset = zeiss_params["y_shifts"]

    ### Pad the sinogram to handle boundaries
    # Set pad size as the largest shift in pixels across views
    max_x_offset = np.max(sino_x_offset) - np.min(sino_x_offset)
    max_y_offset = np.max(sino_y_offset) - np.min(sino_y_offset)

    pad_size = int(np.ceil(np.maximum(max_x_offset, max_y_offset)))

    if pad_size > 0:
        sino_pad = np.pad(sino, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), mode='edge')
    else:
        sino_pad = sino

    # Apply per-view translation
    corrected_sino = np.zeros_like(sino_pad)
    for view in range(sino.shape[0]):
        corrected_sino[view] = jax.image.scale_and_translate(sino_pad[view],
                                      shape=sino_pad[view].shape,
                                      spatial_dims=(0, 1),
                                      scale=jnp.array([1.0, 1.0]),
                                      translation=jnp.array([sino_y_offset[view], sino_x_offset[view]]),
                                      method='linear',
                                      antialias=False)

    # Remove padding
    if pad_size > 0:
        corrected_sino = corrected_sino[:, pad_size:-pad_size, pad_size:-pad_size]

    return corrected_sino


def compute_weight(blank_scan, obj_scan, dark_region_ratio=0.6, safety_buffer=20):
    """
    Create a 3D binary weight mask to zero out black boundary regions.

    Args:
        blank_scan (numpy array): A 3D blank scan of shape (num_blank_scans, num_det_rows, num_det_channels).
        obj_scan (numpy array): A 3D object scan of shape (num_views, num_det_rows, num_det_channels).
        dark_region_ratio (float): ratio used to detect dark boundary region.
            Pixels below dark_region_ratio * median(blank_scan) are assigned zero weight. Defaults to 0.6.
        safety_buffer (int, optional): Safety buffer (in pixels) to add around detected dark regions.
            Defaults to 20.

    Returns:
        weights: A 3D array of weights with the same shape as the input obj_scan.
    """
    # Get object scan shape
    num_views, num_rows, num_cols = obj_scan.shape

    # Detect dark boundary regions using the blank scan
    # Assume that dark boundary region intensity <= 0.5 * median blank scan intensity
    weight_mask_2d = blank_scan[0] >= dark_region_ratio * np.median(blank_scan[0])

    # Add a safety buffer around dark boundary regions
    if safety_buffer > 0:
        padded = np.pad(weight_mask_2d, safety_buffer, mode='constant', constant_values=True)
        padded = binary_erosion(padded, iterations=safety_buffer)
        weight_mask_2d = padded[safety_buffer:-safety_buffer, safety_buffer:-safety_buffer]

    # Broadcast the same mask to all views
    weight_mask = np.broadcast_to(
        weight_mask_2d[np.newaxis, :, :],
        (num_views, num_rows, num_cols),
    ).astype(np.float32)

    return weight_mask