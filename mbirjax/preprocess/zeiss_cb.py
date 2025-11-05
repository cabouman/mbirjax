import os, sys
from operator import itemgetter
import numpy as np
import warnings
import mbirjax.preprocess as mjp
import pprint
import logging
import olefile
import struct
from pathlib import Path
pp = pprint.PrettyPrinter(indent=4)
logger = logging.getLogger(__name__)


def compute_sino_and_params(dataset_dir, downsample_factor=(1, 1), subsample_view_factor=1, crop_pixels_sides=0, crop_pixels_top=0, crop_pixels_bottom=0, verbose=1, is_preprocessed=True):
    """
    NOTICE: THIS FUNCTION IS STILL UNDER DEVELOPMENT AND MAY CONTAIN BUGS OR NOT WORK AS EXPECTED

    Compute Zeiss sinogram and MBIR-JAX geometry parameters.

    This reads object/blank/dark scans and Zeiss geometry from a dataset directory, builds a
    sinogram, and applies a background-offset correction.

    Steps:
        1. Load object, blank, and dark scans and geometry.
        2. Compute the sinogram from the scans.
        3. Apply background offset correction.

    Args:
        dataset_dir (str): Path to the Zeiss dataset. Accepts a ``.txrm`` file with:
            - ``ImageData*/Image*`` (scan data)
            - Zeiss OLE metadata streams
        downsample_factor (Tuple[int, int], optional): Downsample factors for detector rows and channels. Defaults to (1, 1).
        subsample_view_factor (int, optional): Factor by which to subsample views. Defaults to 1.
        crop_pixels_sides (int, optional): Pixels to crop from each lateral side of the detector. Defaults to ``0``.
        crop_pixels_top (int, optional): Pixels to crop from the top of the detector. Defaults to ``0``.
        crop_pixels_bottom (int, optional): Pixels to crop from the bottom of the detector. Defaults to ``0``.
        verbose (int, optional): Verbosity level. Defaults to ``1``.

        is_preprocessed (bool, optional):
            If ``True``, assume the scan data read from the .txrm file is already preprocessed
            and can be used directly as the sinogram. If ``False``, compute the sinogram
            from the raw object, blank, and dark scans. Defaults to ``True``.

    Returns:
        tuple: ``(sino, cone_beam_params, optional_params, metadata)``

            - ``sino`` (numpy.ndarray): Sinogram of shape ``(num_views, num_det_rows, num_channels)``.
            - ``cone_beam_params`` (dict): Parameters for initializing ``ConeBeamModel``.
            - ``optional_params`` (dict): Additional parameters to be set via ``ConeBeamModel.set_params``.
            - ``metadata`` (dict): Zeiss metadata parsed from ``.txrm``.

    Example:
        .. code-block:: python

            from mbirjax.preprocess.zeiss_cb import compute_sino_and_params
            sino, cone_beam_params, optional_params, metadata = compute_sino_and_params(
                dataset_dir, verbose=1
            )
            ct_model = mbirjax.ConeBeamModel(**cone_beam_params)
            ct_model.set_params(**optional_params)
            recon, recon_dict = ct_model.recon(sino)
    """
    if verbose > 0:
        print("\n\n########## Loading object, blank, dark scans, and geometry parameters from Zeiss dataset directory")
    obj_scan, blank_scan, dark_scan, zeiss_params, metadata = load_scans_and_params(dataset_dir, subsample_view_factor, is_preprocessed)

    cone_beam_params, optional_params, metadata = convert_zeiss_to_mbirjax_params(zeiss_params, metadata, downsample_factor=downsample_factor,
                                                                          crop_pixels_sides=crop_pixels_sides,
                                                                          crop_pixels_top=crop_pixels_top,
                                                                          crop_pixels_bottom=crop_pixels_bottom)

    if verbose > 0:
        print("\n\n########## Cropping and downsampling scans")
    ### crop the scans based on input params
    obj_scan, blank_scan, dark_scan, defective_pixel_array = mjp.crop_view_data(obj_scan, blank_scan, dark_scan,
                                                                                crop_pixels_sides=crop_pixels_sides,
                                                                                crop_pixels_top=crop_pixels_top,
                                                                                crop_pixels_bottom=crop_pixels_bottom)

    ### downsample the scans with block-averaging
    if downsample_factor[0] * downsample_factor[1] > 1:
        obj_scan, blank_scan, dark_scan, defective_pixel_array = mjp.downsample_view_data(obj_scan, blank_scan, dark_scan,
                                                                                          downsample_factor=downsample_factor,
                                                                                          defective_pixel_array=defective_pixel_array)

    if verbose > 0:
        print("\n\n########## Computing sinogram from object, blank, and dark scans")
    if is_preprocessed:
        sino = obj_scan
    else:
        sino = mjp.compute_sino_transmission(obj_scan, blank_scan, dark_scan, defective_pixel_array)
    scan_shapes = obj_scan.shape, blank_scan.shape, dark_scan.shape

    if verbose > 0:
        print("\n\n########## Correcting sinogram data to account for background offset and sino offset")
    background_offset = mjp.estimate_background_offset(sino)
    sino = sino - background_offset
    if verbose > 0:
        print("background_offset = ", background_offset)

    if verbose > 0:
        print('obj_scan shape = ', scan_shapes[0])
        print('blank_scan shape = ', scan_shapes[1])
        print('dark_scan shape = ', scan_shapes[2])

    return sino, cone_beam_params, optional_params, metadata


def load_scans_and_params(dataset_dir, subsample_view_factor, is_preprocessed, verbose=1):
    """
    NOTICE: THIS FUNCTION IS STILL UNDER DEVELOPMENT AND MAY CONTAIN BUGS OR NOT WORK AS EXPECTED

    Load the scan data and geometry from a Zeiss scan directory.

    Args:
        dataset_dir (str): Path to a Zeiss scan directory (expect a `.txrm` file). Expected structure:
            - ``ImageData*/Image*`` (scan data)
            - ``**/**`` (Zeiss metadata)
        subsample_view_factor (int, optional): view subsample factor.
        is_preprocessed (bool):
            If ``True``, assume the scan data read from the .txrm file is already preprocessed
            and can be used directly as the sinogram. If ``False``, compute the sinogram
            from the raw object, blank, and dark scans.
        verbose (int, optional): Verbosity level. Defaults to 1.

    Returns:
        tuple: (obj_scan, blank_scan, dark_scan, zeiss_params, Zeiss_params)

            - obj_scan (numpy.ndarray): 3D object scan with shape ``(num_views, num_det_rows, num_channels)``.
            - blank_scan (numpy.ndarray): 3D blank scan with shape ``(1, num_det_rows, num_channels)``.
            - dark_scan (numpy.ndarray): 3D dark scan with shape ``(1, num_det_rows, num_channels)``.
                If no dark scan is available, returns a zero array of the same shape.
            - zeiss_params (dict): Required parameters for ``convert_zeiss_to_mbirjax_params`` (e.g., geometry vectors, spacings, and angles).
            - Zeiss_params (dict): Metadata stored in Zeiss `.txrm` files.
    """
    ### automatically parse the paths to Zeiss scans from dataset_dir
    data_dir = _parse_filenames_from_dataset_dir(dataset_dir)

    if verbose > 0:
        print("The following files will be used to compute the Zeiss reconstruction:\n",
              f"    - txrm file: {data_dir}\n")

    # Read object scans and metadata
    obj_scan, Zeiss_params = read_txrm(data_dir)

    # Subsample the views
    obj_scan = obj_scan[::subsample_view_factor, :, :]

    # Read blank scans
    if is_preprocessed:
        blank_scan = np.zeros(obj_scan.shape, dtype=obj_scan.dtype)
    else:
        if Zeiss_params.get("reference") is not None:
            blank_scan = Zeiss_params["reference"]
            blank_scan = blank_scan[None, :, :]
        else:
            raise ValueError("Missing blank scan; unable to compute sinogram for reconstruction.")

    # Read dark scans
    # TODO: Currently we assume that there is no dark scan for txrm file
    dark_scan = np.zeros(obj_scan.shape, dtype=obj_scan.dtype)

    if verbose > 0:
        print("Scans loaded.")

    # source to iso distance
    source_iso_dist = Zeiss_params["source_iso_dist"][0]
    source_iso_dist = float(np.abs(source_iso_dist))

    # iso to detector distance
    iso_det_dist = Zeiss_params["iso_det_dist"][0]
    iso_det_dist = float(np.abs(iso_det_dist))

    # detector pixel pitch
    # Zeiss detector pixel has equal width and height
    # TODO: The value of the stored detector pixel pitch seems to be wrong;
    #  Read it directly from metadata to keep the issue visible for future correction
    det_pixel_pitch = Zeiss_params["det_pixel_pitch"]
    delta_det_row = det_pixel_pitch
    delta_det_channel = det_pixel_pitch

    # dimensions of radiograph
    num_views = Zeiss_params["num_views"]
    num_det_channels = Zeiss_params["num_det_channels"]
    num_det_rows = Zeiss_params["num_det_rows"]

    # Rotation angles
    angles = np.array(Zeiss_params['thetas'], dtype=float).ravel()
    angles = angles[::subsample_view_factor]

    # Detector offset
    # TODO: Need to check whether the detector offset parameter is correctly read from the file
    #   Since I can only decoded one single float from the directory I found in the file,
    #   I am assuming that this is the detector channel offset, and I am setting the detector row offset to 0.0
    detector_offset = Zeiss_params["det_offset"]
    det_channel_offset = detector_offset
    det_row_offset = 0.0

    # Unit of parameters
    axis_names = Zeiss_params["axis_names"]
    axis_units = Zeiss_params["axis_units"]

    source_iso_dist_unit = None
    iso_det_dist_unit = None
    delta_det_row_unit = None
    delta_det_channel_unit = None
    angle_unit = None

    if axis_names is not None and axis_units is not None:
        for name, unit in zip(axis_names, axis_units):
            # TODO: Need to verify whether the geometry parameter units actually match the specified axis names.
            if "Source Z" in name:
                source_iso_dist_unit = unit
                iso_det_dist_unit = unit

            elif "CCD" in name and "X" in name:
                delta_det_row_unit = unit
                delta_det_channel_unit = unit

            elif "Sample Theta" in name:
                angle_unit = unit

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

    zeiss_params = {
        'source_iso_dist': source_iso_dist,
        'iso_det_dist': iso_det_dist,
        'delta_det_channel': delta_det_channel,
        'delta_det_row': delta_det_row,
        'num_views': num_views,
        'num_det_channels': num_det_channels,
        'num_det_rows': num_det_rows,
        'angles': angles,
        'det_row_offset': det_row_offset,
        'det_channel_offset': det_channel_offset,
        'source_iso_dist_unit': source_iso_dist_unit,
        'iso_det_dist_unit': iso_det_dist_unit,
        'delta_det_row_unit': delta_det_row_unit,
        'delta_det_channel_unit': delta_det_channel_unit,
        'angle_unit': angle_unit,
    }

    return obj_scan, blank_scan, dark_scan, zeiss_params, Zeiss_params


def convert_zeiss_to_mbirjax_params(zeiss_params, Zeiss_metadata, downsample_factor=(1, 1), crop_pixels_sides=0, crop_pixels_top=0, crop_pixels_bottom=0):
    """
    NOTICE: THIS FUNCTION IS STILL UNDER DEVELOPMENT AND MAY CONTAIN BUGS OR NOT WORK AS EXPECTED

    Convert geometry parameters from zeiss into mbirjax format, including modifications to reflect crop.

    Args:
        zeiss_params (dict): Required Zeiss geometry parameters for reconstruction.
        Zeiss_metadata (dict): metadata stored in Zeiss txrm file.
        downsample_factor ((int, int), optional) - Down-sample factors along the detector rows and channels respectively.
            If scan size is not divisible by `downsample_factor`, the scans will be first truncated to a size that is divisible by `downsample_factor`.
        crop_pixels_sides (int, optional): The number of pixels to crop from each side of the sinogram. Defaults to 0.
        crop_pixels_top (int, optional): The number of pixels to crop from top of the sinogram. Defaults to 0.
        crop_pixels_bottom (int, optional): The number of pixels to crop from bottom of the sinogram. Defaults to 0.

    Returns:
        cone_beam_params (dict): Required parameters for the ConeBeamModel constructor.
        optional_params (dict): Additional ConeBeamModel parameters to be set using set_params()
        metadata (dict): metadata stored in Zeiss txrm file.
    """
    # Get zeiss parameters and convert them
    source_iso_dist, iso_det_dist, source_iso_dist_unit, iso_det_dist_unit = itemgetter('source_iso_dist', 'iso_det_dist', 'source_iso_dist_unit', 'iso_det_dist_unit')(zeiss_params)
    delta_det_channel, delta_det_row, delta_det_channel_unit, delta_det_row_unit = itemgetter('delta_det_channel', 'delta_det_row', 'delta_det_channel_unit', 'delta_det_row_unit')(zeiss_params)
    num_det_rows, num_det_channels = itemgetter('num_det_rows', 'num_det_channels')(zeiss_params)
    angles, angle_unit = itemgetter('angles', 'angle_unit')(zeiss_params)
    det_row_offset, det_channel_offset = itemgetter('det_row_offset', 'det_channel_offset')(zeiss_params)

    source_detector_dist = source_iso_dist + iso_det_dist

    # Adjust detector size params w.r.t. cropping arguments
    num_det_rows = num_det_rows - (crop_pixels_top + crop_pixels_bottom)
    num_det_channels = num_det_channels - 2 * crop_pixels_sides

    # Adjust detector size and pixel pitch params w.r.t. downsampling arguments
    num_det_rows = num_det_rows // downsample_factor[0]
    num_det_channels = num_det_channels // downsample_factor[1]

    delta_det_row *= downsample_factor[0]
    delta_det_channel *= downsample_factor[1]

    # Unit conversion table (relative to um)
    # TODO: Need to include other possible unit conversions to ensure all geometry parameters can be safely converted to ALU.
    #   For now, I assume that only um and mm appear in the txrm file
    unit_conversion = {'um': 1.0, 'mm': 1000.0}

    # Set 1 ALU = 1 delta_det_channel_unit
    ALU_unit = delta_det_channel_unit
    Zeiss_metadata['ALU_unit'] = ALU_unit
    Zeiss_metadata['ALU_value'] = 1

    # Convert physical units to ALU
    source_iso_dist = source_iso_dist * unit_conversion[source_iso_dist_unit] / unit_conversion[ALU_unit]
    source_detector_dist = source_detector_dist * unit_conversion[source_iso_dist_unit] / unit_conversion[ALU_unit]

    if angle_unit == 'deg':
        angles = np.deg2rad(angles)
    else:
        pass

    delta_det_row = delta_det_row * unit_conversion[delta_det_row_unit] / unit_conversion[ALU_unit]

    # ToDo: Need to check the units of detector offset
    #  For now, we assume that the det_channel_offset have units of pixels.
    det_channel_offset *= delta_det_channel  # pixels to ALU

    # Create a dictionary to store MBIR parameters
    num_views = len(angles)
    cone_beam_params = dict()
    cone_beam_params['sinogram_shape'] = (num_views, num_det_rows, num_det_channels)
    cone_beam_params["angles"] = angles
    cone_beam_params['source_detector_dist'] = source_detector_dist
    cone_beam_params['source_iso_dist'] = source_iso_dist

    optional_params = dict()
    optional_params['delta_det_channel'] = delta_det_channel
    optional_params['delta_det_row'] = delta_det_row
    optional_params['delta_voxel'] = delta_det_channel * (source_iso_dist / source_detector_dist)
    optional_params['det_row_offset'] = det_row_offset
    optional_params['det_channel_offset'] = det_channel_offset

    return cone_beam_params, optional_params, Zeiss_metadata


######## subroutines for parsing Zeiss object scan, blank scan, and dark scan
def _parse_filenames_from_dataset_dir(dataset_dir):
    """
    NOTICE: THIS FUNCTION IS STILL UNDER DEVELOPMENT AND MAY CONTAIN BUGS OR NOT WORK AS EXPECTED

    Given a path to a Zeiss scan directory, automatically parse the paths to the following files and directories：
        - the txrm file store the projection data

    Args:
        dataset_dir (string): Path to the directory containing the Zeiss scan files.

    Returns:
        Path to the txrm file storing the projection data
    """
    if os.path.isfile(dataset_dir):
        if dataset_dir.endswith(".txrm"):
            return dataset_dir
        else:
            raise ValueError(f"Unsupported file type {dataset_dir}; only .txrm files are supported.")
    else:
        raise ValueError("This function currently supports only direct paths to .txrm files; please specify the exact .txrm file path.")


def _check_read(fname):
    """
    NOTICE: THIS FUNCTION IS STILL UNDER DEVELOPMENT AND MAY CONTAIN BUGS OR NOT WORK AS EXPECTED

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
    NOTICE: THIS FUNCTION IS STILL UNDER DEVELOPMENT AND MAY CONTAIN BUGS OR NOT WORK AS EXPECTED

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
    arr = mjp.utilities._normalize_to_float32(arr)

    ole.close()
    return arr, metadata


def read_xrm_dir(dir_path):
    """
    NOTICE: THIS FUNCTION IS STILL UNDER DEVELOPMENT AND MAY CONTAIN BUGS OR NOT WORK AS EXPECTED

    Read all .xrm files in a directory (filesystem order), stack into (num_views, num_det_rows, num_det_cols),
    and concatenate selected metadata.

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

    # Load the rotation angle of the object of the first file
    angle0 = md0['thetas'][0]

    metadata = dict(md0)
    metadata['num_views'] = num_views
    metadata['x_positions'] = [x0]
    metadata['y_positions'] = [y0]
    metadata['z_positions'] = [z0]
    metadata['thetas'] = [angle0]

    # Load the remaining files and stack them together
    for i, p in enumerate(files[1:], start=1):
        proj, md = read_xrm(str(p))
        arr[i] = proj
        metadata['x_positions'].append(md['x_positions'][0])
        metadata['y_positions'].append(md['y_positions'][0])
        metadata['z_positions'].append(md['z_positions'][0])
        metadata['thetas'].append(md['thetas'][0])

    _log_imported_data(str(dir_path), arr)

    # Normalize the scan data
    arr = mjp.utilities._normalize_to_float32(arr)

    return arr, metadata



def read_txrm(file_name):
    """
    NOTICE: THIS FUNCTION IS STILL UNDER DEVELOPMENT AND MAY CONTAIN BUGS OR NOT WORK AS EXPECTED

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

    # Read metadata from txrm file
    metadata = read_metadata(ole)

    # Create an empty array to store the scan data
    array_of_images = np.empty(
        (
            metadata["num_views"],
            metadata["num_det_rows"],
            metadata["num_det_channels"],
        ),
        dtype=_get_ole_data_type(metadata)
    )

    # Read scan data from txrm file
    for i, idx in enumerate(range(metadata["num_views"])):
        img_string = "ImageData{}/Image{}".format(
            int(np.ceil((idx + 1) / 100.0)), int(idx + 1))
        array_of_images[i] = _read_ole_image(ole, img_string, metadata)

    _log_imported_data(file_name, array_of_images)

    # Normalize the scan data
    # array_of_images = mjp.utilities._normalize_to_float32(array_of_images)

    ole.close()
    return array_of_images, metadata


def read_metadata(ole):
    """
    NOTICE: THIS FUNCTION IS STILL UNDER DEVELOPMENT AND MAY CONTAIN BUGS OR NOT WORK AS EXPECTED

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
        'reference_data_type': _read_ole_value(ole, 'referencedata/DataType', '<1I'),
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
            ole, 'ImageInfo/Angles', "<{0}f".format(number_of_images)),
        'x_positions': _read_ole_arr(
            ole, 'ImageInfo/XPosition', "<{0}f".format(number_of_images)),
        'y_positions': _read_ole_arr(
            ole, 'ImageInfo/YPosition', "<{0}f".format(number_of_images)),
        'z_positions': _read_ole_arr(
            ole, 'ImageInfo/ZPosition', "<{0}f".format(number_of_images)),
        'axis_names': _read_ole_str(ole, 'PositionInfo/AxisNames'),
        'axis_units': _read_ole_str(ole, 'PositionInfo/AxisUnits')
    }

    if ole.exists('referencedata/image'):
        reference = _read_ole_image(ole, 'referencedata/image', metadata, metadata['reference_data_type'])
    else:
        reference = None
    metadata['reference'] = reference

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