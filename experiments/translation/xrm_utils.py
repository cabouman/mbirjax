import numpy as np
import os
import logging
import struct
import olefile
import mbirjax.preprocess as mjp
from pathlib import Path

logger = logging.getLogger(__name__)


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
        np.ndarray: Output 2D image.
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
    )[::-1, :]

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
        np.ndarray: Output 3D image.
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
        'iso_pixel_pitch': _read_ole_value(ole, 'ImageInfo/pixelsize', '<f'),
        'det_pixel_pitch': _read_ole_value(ole, 'ImageInfo/CamPixelSize', '<f'),
        'iso_det_dist': _read_ole_value(ole, 'ImageInfo/DtoRADistance', '<f'),
        'iso_source_dist': _read_ole_value(ole, 'ImageInfo/StoRADistance', '<f'),
        'thetas': _read_ole_arr(
            ole, 'ImageInfo/Angles', "<{0}f".format(number_of_images)),
        'x_positions': _read_ole_arr(
            ole, 'ImageInfo/XPosition', "<{0}f".format(number_of_images)),
        'y_positions': _read_ole_arr(
            ole, 'ImageInfo/YPosition', "<{0}f".format(number_of_images)),
        'z_positions': _read_ole_arr(
            ole, 'ImageInfo/ZPosition', "<{0}f".format(number_of_images)),
        'x-shifts': _read_ole_arr(
            ole, 'alignment/x-shifts', "<{0}f".format(number_of_images)),
        'y-shifts': _read_ole_arr(
            ole, 'alignment/y-shifts', "<{0}f".format(number_of_images))
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


def create_params_dict(obj_scan_path):
    """
    Compute translation parameters from a directory of xrm files.

    Args:
        obj_scan_path (str) : Directory containing xrm files.

    Returns:
        source_det_dist_ALU (float): Distance from the source to the detector in ALU.
        source_iso_dist_ALU (float): Distance from the source to the iso in ALU.
        translation_vectors (np.ndarray): Array of shape (num_views, 3) with translation vectors [dx, dy, dz]
    """

    # Load physical parameters from metadata
    center_path = Path(obj_scan_path) / "MC.xrm"
    _, metadata_center = read_xrm(str(center_path))
    x_center = float(np.asarray(metadata_center['z_positions']).ravel()[0])
    y_center = float(np.asarray(metadata_center['x_positions']).ravel()[0])
    z_center = float(np.asarray(metadata_center['y_positions']).ravel()[0])
    det_pixel_pitch = float(metadata_center['det_pixel_pitch'])
    iso_det_dist = float(metadata_center['iso_det_dist'])
    iso_source_dist = float(metadata_center['iso_source_dist'])

    # Calculate physical parameters in mm
    det_pixel_pitch_mm = det_pixel_pitch / 1000
    source_det_dist_mm = np.abs(iso_det_dist) + np.abs(iso_source_dist)
    source_iso_dist_mm = np.abs(iso_source_dist)

    # Calculate physical parameters in ALU
    ALU_per_mm = source_det_dist_mm / (source_iso_dist_mm * det_pixel_pitch_mm)
    source_iso_dist_ALU = source_iso_dist_mm * ALU_per_mm
    source_det_dist_ALU = source_iso_dist_ALU

    # Load x, y, z positions
    _, metadata = read_xrm_dir(obj_scan_path)

    # x, y, z positions for all files
    x_positions = np.asarray(metadata['z_positions'], dtype=float).ravel()
    y_positions = np.asarray(metadata['x_positions'], dtype=float).ravel()
    z_positions = np.asarray(metadata['y_positions'], dtype=float).ravel()

    # Compute translation vectors in um
    xyz_positions = np.stack([x_positions, y_positions, z_positions], axis=1)
    xyz_center = np.array([x_center, y_center, z_center], float)
    translation_vectors = xyz_center - xyz_positions

    # For now, assume that there is no shift in y direction
    # translation_vectors[:, 1] = 0.0

    # Compute translation vectors in ALU
    translation_vectors = translation_vectors * ALU_per_mm / 1000

    sino_shape = (metadata['num_views'], metadata['num_det_rows'], metadata['num_det_channels'])

    translation_params = dict()
    translation_params['sinogram_shape'] = sino_shape
    translation_params['translation_vectors'] = translation_vectors
    translation_params['source_detector_dist'] = source_det_dist_ALU
    translation_params['source_iso_dist'] = source_iso_dist_ALU

    return translation_params


def compute_sino_and_params(obj_scan_path, blank_scan_path, dark_scan_path):
    """

    Args:
        obj_scan_path:
        blank_scan_path:
        dark_scan_path:

    Returns:

    """
    # Compute and load required parameters for translation reconstruction
    translation_params = create_params_dict(obj_scan_path)

    # Load obj scan, blank scan, and dark scan
    obj_scan, _ = read_xrm_dir(obj_scan_path)
    blank_scan, _ = read_xrm_dir(blank_scan_path)
    dark_scan, _ = read_xrm_dir(dark_scan_path)

    # Crop out defective rows in obj scan, blank scan, and dark scan
    crop_pixels_bottom = 53
    obj_scan, blank_scan, dark_scan, _ = mjp.crop_view_data(
        obj_scan, blank_scan, dark_scan,
        crop_pixels_sides=0, crop_pixels_top=0, crop_pixels_bottom=crop_pixels_bottom,
        defective_pixel_array=())

    # Compute sinogram
    sino = mjp.compute_sino_transmission(obj_scan, blank_scan, dark_scan)
    translation_params['sinogram_shape'] = sino.shape

    return sino, translation_params