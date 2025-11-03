import os, sys
from operator import itemgetter
import numpy as np
import jax.numpy as jnp
import warnings
import mbirjax as mj
import mbirjax.preprocess as mjp
import pprint
import logging
import olefile
import struct
from pathlib import Path
pp = pprint.PrettyPrinter(indent=4)
logger = logging.getLogger(__name__)


def compute_sino_and_params(dataset_dir, crop_pixels_sides=0, crop_pixels_top=0, crop_pixels_bottom=0, verbose=1):
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
        verbose (int, optional): Verbosity level. Defaults to 1.

    Returns:
        tuple: (sino, translation_params, optional_params)
            - ``sino`` (jax.numpy.ndarray): Sinogram of shape (num_views, num_det_rows, num_channels)
            - ``translation_params`` (dict): Parameters for initializing TranslationModel.
            - ``optional_params`` (dict): Parameters to be passed via ``set_params``.

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
    obj_scan, blank_scan, dark_scan, zeiss_params = \
        load_scans_and_params(dataset_dir, verbose=verbose)

    translation_params, optional_params = convert_zeiss_to_mbirjax_params(zeiss_params,
                                                                          crop_pixels_sides=crop_pixels_sides,
                                                                          crop_pixels_top=crop_pixels_top,
                                                                          crop_pixels_bottom=crop_pixels_bottom)

    if verbose > 0:
        print("\n\n########## Cropping scans")
    ### crop the scans based on input params
    obj_scan, blank_scan, dark_scan, defective_pixel_array = mjp.crop_view_data(obj_scan, blank_scan, dark_scan,
                                                                                crop_pixels_sides=crop_pixels_sides,
                                                                                crop_pixels_top=crop_pixels_top,
                                                                                crop_pixels_bottom=crop_pixels_bottom)

    if verbose > 0:
        print("\n\n########## Computing sinogram from object, blank, and dark scans")
    sino = mjp.compute_sino_transmission(obj_scan, blank_scan, dark_scan, defective_pixel_array)
    scan_shapes = obj_scan.shape, blank_scan.shape, dark_scan.shape
    del obj_scan, blank_scan, dark_scan  # delete scan images to save memory

    if verbose > 0:
        print("\n\n########## Correcting sinogram data to account for background offset and detector rotation")
    background_offset = mjp.estimate_background_offset(sino)
    sino = sino - background_offset
    if verbose > 0:
        print("background_offset = ", background_offset)

    if verbose > 0:
        print('obj_scan shape = ', scan_shapes[0])
        print('blank_scan shape = ', scan_shapes[1])
        print('dark_scan shape = ', scan_shapes[2])

    return sino, translation_params, optional_params


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
    obj_scan_dir, blank_scan_dir, dark_scan_dir = \
        _parse_filenames_from_dataset_dir(dataset_dir)

    if verbose > 0:
        print("The following files will be used to compute the Zeiss reconstruction:\n",
              f"    - Object scan directory: {obj_scan_dir}\n",
              f"    - Blank scan directory: {blank_scan_dir}\n",
              f"    - Dark scan directory: {dark_scan_dir}\n",)

    _, Zeiss_params = read_xrm_dir(obj_scan_dir) # Zeiss parameters of all the object scans

    # source to iso distance
    source_iso_dist = Zeiss_params["source_iso_dist"]
    source_iso_dist = float(np.abs(source_iso_dist))

    # iso to detector distance
    iso_det_dist = Zeiss_params["iso_det_dist"]
    iso_det_dist = float(np.abs(iso_det_dist))

    # detector pixel pitch
    # Zeiss detector pixel has equal width and height
    det_pixel_pitch = Zeiss_params["det_pixel_pitch"]
    iso_pixel_pitch = Zeiss_params["iso_pixel_pitch"]
    delta_det_row = det_pixel_pitch
    delta_det_channel = det_pixel_pitch

    # dimensions of radiograph
    num_views = Zeiss_params["num_views"]
    num_det_channels = Zeiss_params["num_det_channels"]
    num_det_rows = Zeiss_params["num_det_rows"]

    # Detector offset
    # TODO: Need to check whether the detector offset parameter is correctly read from the file
    #   Since I can only decoded one single float from the directory I found in the file,
    #   I am assuming that this is the detector channel offset, and I am setting the detector row offset to 0.0
    detector_offset = Zeiss_params["det_offset"]
    det_channel_offset = detector_offset
    det_row_offset = 0.0

    # object positions in x, y, z axis
    # The scanner uses a coordinate system different from MBIRJAX
    # ToDo: Perform experiments to determine the Zeiss coordinates
    # Axis mapping:
    #   Scanner z-axis -> MBIRJAX x-axis or -x-axis not sure
    #   Scanner x-axis -> MBIRJAX y-axis or -y-axis not sure
    #   Scanner y-axis -> MBIRJAX z-axis or -z-axis not sure
    obj_x_positions = np.array(Zeiss_params['z_positions'], dtype=float).ravel()
    obj_y_positions = np.array(Zeiss_params['x_positions'], dtype=float).ravel()
    obj_z_positions = np.array(Zeiss_params['y_positions'], dtype=float).ravel()

    # Unit of parameters
    axis_names = Zeiss_params["axis_names"]
    axis_units = Zeiss_params["axis_units"]

    source_iso_dist_unit = None
    iso_det_dist_unit = None
    delta_det_row_unit = None
    delta_det_channel_unit = None
    obj_x_position_unit = None
    obj_y_position_unit = None
    obj_z_position_unit = None

    if axis_names is not None and axis_units is not None:
        for name, unit in zip(axis_names, axis_units):
            # TODO: Need to verify whether the geometry parameter units actually match the specified axis names.
            if name == "Source Z":
                source_iso_dist_unit = unit
                iso_det_dist_unit = unit

            elif name == "CCD_X":
                delta_det_row_unit = unit
                delta_det_channel_unit = unit

            elif name == "Sample X":
                obj_x_position_unit = unit

            elif name == "Sample Y":
                obj_y_position_unit = unit

            elif name == "Sample Z":
                obj_z_position_unit = unit

    else:
        raise ValueError("Unknown units for geometry parameters; cannot safely convert to mbirjax format.")

    if verbose > 0:
        print("############ Zeiss geometry parameters ############")
        print(f"Source to iso distance: {source_iso_dist} [{source_iso_dist_unit}]")
        print(f"Iso to detector distance: {iso_det_dist} [{iso_det_dist_unit}]")
        print(f"Detector pixel pitch: (delta_det_row, delta_det_channel) = ({det_pixel_pitch:.3f} [{delta_det_row_unit}], {det_pixel_pitch:.3f} [{delta_det_channel_unit}])")
        print(f"Detector size: (num_det_rows, num_det_channels) = ({num_det_rows}, {num_det_channels})")
        print("############ End Zeiss geometry parameters ############")
    ### END load Zeiss parameters from scan data

    ### read blank scans and dark scans
    blank_scan, _ = read_xrm_dir(blank_scan_dir)
    if dark_scan_dir is not None:
        dark_scan, _ = read_xrm_dir(dark_scan_dir)
    else:
        dark_scan = np.zeros(blank_scan.shape)

    ### read object scans
    obj_scan, _ = read_xrm_dir(obj_scan_dir)
    if verbose > 0:
        print("Scans loaded.")

    ### flip the scans vertically
    if verbose > 0:
        print("Flipping scans vertically")
    obj_scan = np.flip(obj_scan, axis=1)
    blank_scan = np.flip(blank_scan, axis=1)
    dark_scan = np.flip(dark_scan, axis=1)

    zeiss_params = {
        'source_iso_dist': source_iso_dist,
        'iso_det_dist': iso_det_dist,
        'delta_det_channel': delta_det_channel,
        'delta_det_row': delta_det_row,
        'num_views': num_views,
        'num_det_channels': num_det_channels,
        'num_det_rows': num_det_rows,
        'det_row_offset': det_row_offset,
        'det_channel_offset': det_channel_offset,
        'obj_x_positions': obj_x_positions,
        'obj_y_positions': obj_y_positions,
        'obj_z_positions': obj_z_positions,
        'source_iso_dist_unit': source_iso_dist_unit,
        'iso_det_dist_unit': iso_det_dist_unit,
        'delta_det_row_unit': delta_det_row_unit,
        'delta_det_channel_unit': delta_det_channel_unit,
        'obj_x_position_unit': obj_x_position_unit,
        'obj_y_position_unit': obj_y_position_unit,
        'obj_z_position_unit': obj_z_position_unit,
    }

    return obj_scan, blank_scan, dark_scan, zeiss_params


def convert_zeiss_to_mbirjax_params(zeiss_params, crop_pixels_sides=0, crop_pixels_top=0, crop_pixels_bottom=0):
    """
    Convert geometry parameters from zeiss into mbirjax format, including modifications to reflect crop.

    Args:
        zeiss_params (dict): Required Zeiss geometry parameters for reconstruction.
        crop_pixels_sides (int, optional): The number of pixels to crop from each side of the sinogram. Defaults to 0.
        crop_pixels_top (int, optional): The number of pixels to crop from top of the sinogram. Defaults to 0.
        crop_pixels_bottom (int, optional): The number of pixels to crop from bottom of the sinogram. Defaults to 0.

    Returns:
        translation_params (dict): Required parameters for the TranslationModel constructor.
        optional_params (dict): Additional TranslationModel parameters to be set using set_params()
    """
    # Get zeiss parameters and convert them
    source_iso_dist, iso_det_dist, source_iso_dist_unit, iso_det_dist_unit = itemgetter('source_iso_dist', 'iso_det_dist', 'source_iso_dist_unit', 'iso_det_dist_unit')(zeiss_params)
    delta_det_channel, delta_det_row, delta_det_channel_unit, delta_det_row_unit = itemgetter('delta_det_channel', 'delta_det_row', 'delta_det_channel_unit', 'delta_det_row_unit')(zeiss_params)
    num_views, num_det_rows, num_det_channels = itemgetter('num_views', 'num_det_rows', 'num_det_channels')(zeiss_params)
    obj_x_positions, obj_y_positions, obj_z_positions, obj_x_position_unit, obj_y_position_unit, obj_z_position_unit = itemgetter('obj_x_positions', 'obj_y_positions', 'obj_z_positions', 'obj_x_position_unit', 'obj_y_position_unit', 'obj_z_position_unit')(zeiss_params)
    det_row_offset, det_channel_offset = itemgetter('det_row_offset', 'det_channel_offset')(zeiss_params)

    source_detector_dist = source_iso_dist + iso_det_dist
    translation_vectors = calc_translation_vec_params(obj_x_positions, obj_y_positions, obj_z_positions)

    # Adjust detector size params w.r.t. cropping arguments
    num_det_rows = num_det_rows - (crop_pixels_top + crop_pixels_bottom)
    num_det_channels = num_det_channels - 2 * crop_pixels_sides

    sinogram_shape = (num_views, num_det_rows, num_det_channels)

    # Unit conversion table (relative to um)
    # TODO: Need to include other possible unit conversions to ensure all geometry parameters can be safely converted to ALU.
    #   For now, I assume that only um and mm appear in the xrm file
    unit_conversion = {'um': 1.0, 'mm': 1000.0}

    # Set 1 ALU = 1 delta_det_channel_unit
    ALU_unit = delta_det_channel_unit

    # Convert physical units to ALU
    source_iso_dist = source_iso_dist * unit_conversion[source_iso_dist_unit] / unit_conversion[ALU_unit]
    source_detector_dist = source_detector_dist * unit_conversion[source_iso_dist_unit] / unit_conversion[ALU_unit]

    if obj_x_position_unit == obj_y_position_unit == obj_z_position_unit:
        translation_vectors = translation_vectors * unit_conversion[obj_x_position_unit] / unit_conversion[ALU_unit]
    else:
        translation_vectors[:, 0] = translation_vectors[:, 0] * unit_conversion[obj_x_position_unit] / unit_conversion[ALU_unit]
        translation_vectors[:, 1] = translation_vectors[:, 1] * unit_conversion[obj_y_position_unit] / unit_conversion[ALU_unit]
        translation_vectors[:, 2] = translation_vectors[:, 2] * unit_conversion[obj_z_position_unit] / unit_conversion[ALU_unit]

    delta_det_row = delta_det_row * unit_conversion[delta_det_row_unit] / unit_conversion[ALU_unit]

    # ToDo: Need to check the units of detector offset
    #  For now, we assume that the det_channel_offset have units of pixels.
    det_channel_offset *= delta_det_channel # pixels to ALU

    # Calculate recon_shape, delta_voxel, and delta_recon_row parameters
    recon_shape, delta_voxel, delta_recon_row = mj.utilities.calc_tct_recon_params(source_detector_dist, source_iso_dist, delta_det_row, delta_det_channel, sinogram_shape, translation_vectors)

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
    optional_params['delta_recon_row'] = delta_recon_row
    optional_params['recon_shape'] = recon_shape
    optional_params['det_row_offset'] = det_row_offset
    optional_params['det_channel_offset'] = det_channel_offset

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

    This code is adapted from the DXchange library:
    https://github.com/data-exchange/dxchange

    Reference:
    [1] DXchange library: https://github.com/data-exchange/dxchange

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

    This code is adapted from the DXchange library:
    https://github.com/data-exchange/dxchange

    Reference:
    [1] DXchange library: https://github.com/data-exchange/dxchange

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

    metadata = read_ole_metadata(ole)

    stream = ole.openstream("ImageData1/Image1")
    data = stream.read()

    data_type = _get_ole_data_type(metadata)
    data_type = data_type.newbyteorder('<')

    arr = np.reshape(
        np.frombuffer(data, data_type),
        (
            metadata["num_det_rows"],
            metadata["num_det_channels"]
        )
    )

    _log_imported_data(fname, arr)

    if np.issubdtype(arr.dtype, np.integer):
        # make float and normalize integer types
        maxval = np.iinfo(arr.dtype).max
        arr = arr.astype(np.float32) / maxval

    arr.astype(np.float32)

    ole.close()
    return arr, metadata


def read_xrm_dir(dir_path):
    """
    Read all .xrm files in a directory (filesystem order), stack into (num_views, num_det_rows, num_det_cols),
    and concatenate x/y/z position metadata.

    This code is adapted from the DXchange library:
    https://github.com/data-exchange/dxchange

    Reference:
    [1] DXchange library: https://github.com/data-exchange/dxchange

    Args:
        dir_path (str) : Path to the directory to be read.

    Returns:
        np.ndarray: Output 3D image with shape (num_views, num_det_rows, num_det_channels).
        dict: Output metadata
    """
    dir_path = Path(dir_path)
    files = [p for p in dir_path.iterdir() if p.is_file()]

    # Load the first file
    proj0, md0 = read_xrm(str(files[0]))
    num_views = len(files)
    num_det_rows, num_det_channels = proj0.shape
    arr = np.empty((num_views, num_det_rows, num_det_channels), dtype=proj0.dtype)
    arr[0] = proj0

    # Load the x, y, z positions of the first file
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

    if np.issubdtype(arr.dtype, np.integer):
        # make float and normalize integer types
        maxval = np.iinfo(arr.dtype).max
        arr = arr.astype(np.float32) / maxval

    arr.astype(np.float32)

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

    metadata = read_ole_metadata(ole)

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


def read_ole_metadata(ole):
    """
    Read metadata from an xradia OLE file (.xrm, .txrm, .txm).

    This code is adapted from the DXchange library:
    https://github.com/data-exchange/dxchange

    Reference:
    [1] DXchange library: https://github.com/data-exchange/dxchange

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
        # TODO: Need to check whether we read the correct detector offset parameter from the file
        'det_offset': _read_ole_value(ole, 'DetAssemblyInfo/CameraOffset', '<f'),
        'iso_det_dist': _read_ole_arr(
            ole, 'ImageInfo/DtoRADistance', "<{0}f".format(number_of_images)),
        'source_iso_dist': _read_ole_arr(
            ole, 'ImageInfo/StoRADistance', "<{0}f".format(number_of_images)),
        'thetas': _read_ole_arr(
            ole, 'ImageInfo/Angles', "<{0}f".format(number_of_images)) * np.pi / 180.,
        'x_positions': _read_ole_arr(
            ole, 'ImageInfo/XPosition', "<{0}f".format(number_of_images)),
        'y_positions': _read_ole_arr(
            ole, 'ImageInfo/YPosition', "<{0}f".format(number_of_images)),
        'z_positions': _read_ole_arr(
            ole, 'ImageInfo/ZPosition', "<{0}f".format(number_of_images)),
        'x-shifts': _read_ole_arr(
            ole, 'alignment/x-shifts', "<{0}f".format(number_of_images)),
        'y-shifts': _read_ole_arr(
            ole, 'alignment/y-shifts', "<{0}f".format(number_of_images)),
        'axis_names': _read_ole_str(ole, 'PositionInfo/AxisNames'),
        'axis_units': _read_ole_str(ole, 'PositionInfo/AxisUnits')
    }

    return metadata


def _log_imported_data(fname, arr):
    """
    Log information about imported data.

    This code is adapted from the DXchange library:
    https://github.com/data-exchange/dxchange

    Reference:
    [1] DXchange library: https://github.com/data-exchange/dxchange

    Args:
        fname (str) : Path of the file from which data was imported.
        arr (np.ndarray) : Array containing the image data.
    """
    logger.debug('Data shape & type: %s %s', arr.shape, arr.dtype)
    logger.info('Data successfully imported: %s', fname)


def _get_ole_data_type(metadata, datatype=None):
    """
    Determine the Numpy data type for image data stored in a Zeiss OLE (.xrm, .txrm, .txm) file.

    This code is adapted from the DXchange library:
    https://github.com/data-exchange/dxchange

    Reference:
    [1] DXchange library: https://github.com/data-exchange/dxchange

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

    This code is adapted from the DXchange library:
    https://github.com/data-exchange/dxchange

    Reference:
    [1] DXchange library: https://github.com/data-exchange/dxchange

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

    This code is adapted from the DXchange library:
    https://github.com/data-exchange/dxchange

    Reference:
    [1] DXchange library: https://github.com/data-exchange/dxchange

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

    This code is adapted from the DXchange library:
    https://github.com/data-exchange/dxchange

    Reference:
    [1] DXchange library: https://github.com/data-exchange/dxchange

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

    This code is adapted from the DXchange library:
    https://github.com/data-exchange/dxchange

    Reference:
    [1] DXchange library: https://github.com/data-exchange/dxchange

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
    NOTICE: THIS FUNCTION IS STILL UNDER DEVELOPMENT AND MAY CONTAIN BUGS OR NOT WORK AS EXPECTED

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