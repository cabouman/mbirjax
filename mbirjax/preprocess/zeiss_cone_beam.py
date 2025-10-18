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


def compute_sino_and_params(dataset_dir, downsample_factor=(1, 1),
                            crop_pixels_sides=0, crop_pixels_top=0, crop_pixels_bottom=0, verbose=1):
    """
    NOTICE: THIS FUNCTION IS STILL UNDER DEVELOPMENT AND MAY CONTAIN BUGS OR NOT WORK AS EXPECTED

    Load Zeiss sinogram data and prepare arrays ana parameters for ConeBeamModel reconstruction.

    This function computes the sinogram and geometry parameters from a Zeiss scan directory. It performs the following:

    1. Load object, blank, and dark scans, and geometry parameters from the dataset.
    2. Computes the sinogram from the scan images.
    3. Applies background offset correction.

    Args:
        dataset_dir (str): Path to the Zeiss scan directory (expect to be a txrm or xrm file). Expected structure.
            - ``ImageData*/Image*`` (scan data)
            - ``**/**`` (Zeiss metadata)
        downsample_factor (Tuple[int, int], optional): Downsampling factor for detector rows and channels. Defaults to (1, 1).
        crop_pixels_sides (int, optional): Pixels to crop from each side of the sinogram. Defaults to None.
        crop_pixels_top (int, optional): Pixels to crop from top of the sinogram. Defaults to None.
        crop_pixels_bottom (int, optional): Pixels to crop from bottom of the sinogram. Defaults to None.
        verbose (int, optional): Verbosity level. Defaults to 1.

    Returns:
        tuple: (sino, coen_beam_params, optional_params)
            - ``sino`` (jax.numpy.ndarray): Sinogram of shape (num_views, num_det_rows, num_channels)
            - ``cone_beam_params`` (dict): Parameters for initializing ConeBeamModel.
            - ``optional_params`` (dict): Parameters to be passed via ``set_params``.
            - ``metadata`` (dict): Metadata stored in Zeiss txrm file.

    Example:
        .. code-block:: python

            # Get data and reconstruction parameters
            sino, cone_beam_params, optional_params = mbirjax.preprocess.zeiss.compute_sino_and_params(
            dataset_dir, downsample_factor=(1, 1))

            # Create the model and set parameters
            ct_model = mbirjax.ConeBeamModel(**cone_beam_params)
            ct_model.set_params(**optional_params)
            ct_model.set_params(sharpness=sharpness, verbose=1)

            # Run reconstruction
            recon, recon_dict = ct_model.recon(sino)
    """
    if verbose > 0:
        print("\n\n########## Loading scan data and geometry parameters from Zeiss dataset directory")
    obj_scan, blank_scan, dark_scan, zeiss_params, metadata = load_scans_and_params(dataset_dir)

    cone_beam_params, optional_params = convert_zeiss_to_mbirjax_params(zeiss_params, downsample_factor=downsample_factor,
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
        obj_scan, blank_scan, dark_scan, defective_pixel_array = mjp.downsample_view_data(obj_scan, blank_scan,
                                                                                          dark_scan,
                                                                                          downsample_factor=downsample_factor,
                                                                                          defective_pixel_array=defective_pixel_array)

    if verbose > 0:
        print("\n\n########## Computing sinogram from object, blank, and dark scans")
    # For now, we assume the data we read from .txrm file to be the sinogram served as input to our model.
    sino = obj_scan
    scan_shapes = obj_scan.shape, blank_scan.shape, dark_scan.shape

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

    return sino, cone_beam_params, optional_params, metadata


def load_scans_and_params(dataset_dir, verbose=1):
    """
    NOTICE: THIS FUNCTION IS STILL UNDER DEVELOPMENT AND MAY CONTAIN BUGS OR NOT WORK AS EXPECTED

    Load the scan data, and geometry from a Zeiss scan directory.
    Args:
        dataset_dir (string): Path to a Zeiss scan directory (expect to be a txrm or xrm file). The directory is assumed to have the following structure:

            - ``ImageData*/Image*`` (scan data)
            - ``**/**`` (Zeiss metadata)

        verbose (int, optional): Verbosity level. Defaults to 1.

    Returns:
        tuple: (sino, zeiss_params)

            - obj_scan (numpy.ndarray): 3D object scan with shape (num_views, num_det_rows, num_channels).
            - blank_scan (numpy.ndarray): 3D blank scan with shape (1, num_det_rows, num_det_channels).
            - dark_scan (numpy.ndarray): 3D dark scan with shape (1, num_det_rows, num_det_channels).
            - zeiss_params (dict): Required parameters needed for "convert_zeiss_to_mbirjax_params()" (e.g., geometry vectors, spacings, and angles).
            - Zeiss_metadata (dict): metadata stored in Zeiss txrm file.
    """
    ### automatically parse the paths to Zeiss scans from dataset_dir
    data_dir = _parse_filenames_from_dataset_dir(dataset_dir)

    if isinstance(data_dir, tuple) and len(data_dir) == 3:
        obj_scan_dir, blank_scan_dir, dark_scan_dir = data_dir

        if verbose > 0:
            print("The following files will be used to compute the Zeiss reconstruction:\n",
                  f"    - Object scan directory: {obj_scan_dir}\n",
                  f"    - Blank scan directory: {blank_scan_dir}\n",
                  f"    - Dark scan directory: {dark_scan_dir}\n")

        obj_scan, Zeiss_metadata = read_xrm_dir(obj_scan_dir)
        blank_scan, _ = read_xrm_dir(blank_scan_dir)

        if dark_scan_dir is not None:
            dark_scan, _ = read_xrm_dir(dark_scan_dir)
        else:
            dark_scan = np.zeros(blank_scan.shape)

        if verbose > 0:
            print("Scans loaded.")

    else:
        if verbose > 0:
            print("The following files will be used to compute the Zeiss reconstruction:\n",
                  f"    - Projection data directory: {data_dir}\n")

        # For now, we assume the data we read from .txrm file to be the sinogram served as input to our model.
        obj_scan, Zeiss_metadata = read_txrm(data_dir)

        # Currently we do not have blank scan and dark scan for txrm file
        blank_scan = np.zeros((1, obj_scan.shape[1], obj_scan.shape[2]))
        dark_scan = np.zeros((1, obj_scan.shape[1], obj_scan.shape[2]))

        if verbose > 0:
            print("Scans loaded.")

    # source to iso distance (in mm)
    source_iso_dist = Zeiss_metadata["source_iso_dist"][0] # mm
    source_iso_dist = float(np.abs(source_iso_dist))

    # iso to detector distance (in mm)
    iso_det_dist = Zeiss_metadata["iso_det_dist"][0] # mm
    iso_det_dist = float(np.abs(iso_det_dist))

    # detector pixel pitch (in um)
    # Zeiss detector pixel has equal width and height
    det_pixel_pitch = 2.0 # um
    delta_det_row = det_pixel_pitch
    delta_det_channel = det_pixel_pitch

    # dimensions of radiograph
    num_det_channels = Zeiss_metadata["num_det_channels"]
    num_det_rows = Zeiss_metadata["num_det_rows"]

    # Rotation angles (in radians)
    angles = np.array(Zeiss_metadata['thetas'], dtype=float).ravel()

    if verbose > 0:
        print("############ Zeiss geometry parameters ############")
        print(f"Source to iso distance: {source_iso_dist} [mm]")
        print(f"Iso to detector distance: {iso_det_dist} [mm]")
        print(f"Detector pixel pitch: (delta_det_row, delta_det_channel) = ({det_pixel_pitch:.3f}, {det_pixel_pitch:.3f}) [um]")
        print(f"Detector size: (num_det_rows, num_det_channels) = ({num_det_rows}, {num_det_channels})")
        print("############ End Zeiss geometry parameters ############")
    ### END load Zeiss parameters from scan data

    zeiss_params = {
        'source_iso_dist': source_iso_dist,
        'iso_det_dist': iso_det_dist,
        'delta_det_channel': delta_det_channel,
        'delta_det_row': delta_det_row,
        'num_det_channels': num_det_channels,
        'num_det_rows': num_det_rows,
        'angles': angles,
    }

    return obj_scan, blank_scan, dark_scan, zeiss_params, Zeiss_metadata


def convert_zeiss_to_mbirjax_params(zeiss_params, downsample_factor=(1, 1), crop_pixels_sides=0, crop_pixels_top=0, crop_pixels_bottom=0):
    """
    NOTICE: THIS FUNCTION IS STILL UNDER DEVELOPMENT AND MAY CONTAIN BUGS OR NOT WORK AS EXPECTED

    Convert geometry parameters from zeiss into mbirjax format, including modifications to reflect crop and downsample.

    Args:
        zeiss_params (dict): Required Zeiss geometry parameters for reconstruction.
        downsample_factor ((int, int), optional) - Down-sample factors along the detector rows and channels respectively.
            If scan size is not divisible by `downsample_factor`, the scans will be first truncated to a size that is divisible by `downsample_factor`.
        crop_pixels_sides (int, optional): The number of pixels to crop from each side of the sinogram. Defaults to 0.
        crop_pixels_top (int, optional): The number of pixels to crop from top of the sinogram. Defaults to 0.
        crop_pixels_bottom (int, optional): The number of pixels to crop from bottom of the sinogram. Defaults to 0.

    Returns:
        cone_beam_params (dict): Required parameters for the ConeBeamModel constructor.
        optional_params (dict): Additional ConeBeamModel parameters to be set using set_params()
    """
    # Get zeiss parameters and convert them
    source_iso_dist, iso_det_dist = itemgetter('source_iso_dist', 'iso_det_dist')(zeiss_params)
    delta_det_channel, delta_det_row = itemgetter('delta_det_channel', 'delta_det_row')(zeiss_params)
    num_det_rows, num_det_channels, angles = itemgetter('num_det_rows', 'num_det_channels', 'angles')(zeiss_params)

    source_detector_dist = calc_source_det_params(source_iso_dist, iso_det_dist)
    det_row_offset = calc_row_params(crop_pixels_top, crop_pixels_bottom)

    # Adjust detector size params w.r.t. cropping arguments
    num_det_rows = num_det_rows - (crop_pixels_top + crop_pixels_bottom)
    num_det_channels = num_det_channels - 2 * crop_pixels_sides

    # Adjust detector size and pixel pitch params w.r.t. downsampling arguments
    num_det_rows = num_det_rows // downsample_factor[0]
    num_det_channels = num_det_channels // downsample_factor[1]

    delta_det_row *= downsample_factor[0]
    delta_det_channel *= downsample_factor[1]

    # Set 1 ALU = delta_det_channel
    source_iso_dist *= 1000 # mm to um
    source_detector_dist *= 1000 # mm to um
    source_iso_dist /= delta_det_channel # um to ALU
    source_detector_dist /= delta_det_channel # um to ALU
    delta_det_row /= delta_det_channel
    delta_det_channel = 1.0

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

    return cone_beam_params, optional_params


######## subroutines for parsing Zeiss object scan, blank scan, and dark scan
def _parse_filenames_from_dataset_dir(dataset_dir):
    """
    NOTICE: THIS FUNCTION IS STILL UNDER DEVELOPMENT AND MAY CONTAIN BUGS OR NOT WORK AS EXPECTED

    Given a path to a Zeiss scan directory, automatically parse the paths to the following files and directories：

    if the files are in xrm format：
        - object scan directory
        - blank scan directory
        - dark scan directory (If exists)

    if the files are in txrm format:
        - the txrm file store the projection data

    Args:
        dataset_dir (string): Path to the directory containing the Zeiss scan files.

    Returns:
        Path to the txrm file storing the projection data
    """
    # If the data is saved in txrm format
    if os.path.isfile(dataset_dir):
        if dataset_dir.endswith(".txrm"):
            return dataset_dir

    # If the data is saved in xrm format
    for root, dirs, files in os.walk(dataset_dir):
        for filename in files:
            if filename.endswith(".xrm"):
                # Object scan directory
                obj_scan_dir = os.path.join(dataset_dir, "obj_scan")

                # Blank scan
                blank_scan_dir = os.path.join(dataset_dir, "blank_scan")

                # Dark scan
                dark_scan_dir = os.path.join(dataset_dir, "dark_scan")
                if not os.path.exists(dark_scan_dir):
                    dark_scan_dir = None

                return obj_scan_dir, blank_scan_dir, dark_scan_dir

    return dataset_dir


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

    # Load the first file
    proj0, md0 = read_xrm(str(files[0]))
    num_views = len(files)
    num_det_rows, num_det_channels = proj0.shape
    arr = np.empty((num_views, num_det_rows, num_det_channels), dtype=proj0.dtype)
    arr[0] = proj0

    # Load the x, y, z object positions of the first file
    x0 = md0['x_positions'][0]
    y0 = md0['y_positions'][0]
    z0 = md0['z_positions'][0]

    # Load the stage angle of the object of the first file
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
        'num_views': number_of_images,
        'iso_pixel_pitch': _read_ole_value(ole, 'ImageInfo/PixelSize', '<f'),
        'det_pixel_pitch': _read_ole_value(ole, 'ImageInfo/CamPixelSize', '<f'),
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
    Reads the string associated with label in an ole file

    This code is adapted from the DXchange library:
    https://github.com/data-exchange/dxchange

    Reference:
    [1] DXchange library: https://github.com/data-exchange/dxchange

    Args:
        ole (OleFileIO) : An ole file to read from.
        label (str) : Label associated with the OLE file.

    Returns:
        list: A list contain all the strings from the binary stream if the label exists
    """
    stream = ole.openstream(label)
    data = stream.read()
    str = [name.decode('utf-8') for name in data.split(b'\x00') if name]
    return str

######## END subroutines for parsing Zeiss object scan, blank scan, and dark scan


######## subroutines for Zeiss-MBIR parameter conversion
def calc_source_det_params(source_iso_dist, iso_det_dist):
    """
    NOTICE: THIS FUNCTION IS STILL UNDER DEVELOPMENT AND MAY CONTAIN BUGS OR NOT WORK AS EXPECTED

    Calculate MBIRJAX geometry parameters: source_det_dist

    Args:
        source_iso_dist (float): Distance between the X-ray source and iso
        iso_det_dist (float): Distance between the detector and iso

    Returns:
        source_detector_dist (float): Distance between the detector and source
    """
    source_det_dist = source_iso_dist + iso_det_dist
    return source_det_dist


def calc_row_params(crop_pixels_top, crop_pixels_bottom):
    """
    NOTICE: THIS FUNCTION IS STILL UNDER DEVELOPMENT AND MAY CONTAIN BUGS OR NOT WORK AS EXPECTED

    Calculate the MBIRJAX geometry parameters: det_row_offset

    Args:
        crop_pixels_top (int): The number of pixels to crop from top of the sinogram.
        crop_pixels_bottom (int): The number of pixels to crop from bottom of the sinogram.

    Returns:
        det_row_offset (float): Distance from center of detector to the source-detector line along a column.
    """
    det_row_offset = - (crop_pixels_top + crop_pixels_bottom) / 2
    return det_row_offset

######## END subroutines for Zeiss-MBIR parameter conversion