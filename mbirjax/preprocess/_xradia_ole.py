"""
Shared private helpers for reading Xradia/Zeiss OLE-based scan files (``.xrm``, ``.txrm``, ``.txm``).

These low-level readers are used by both :mod:`mbirjax.preprocess.zeiss` (Ultra/Versa rotational CT) and
:mod:`mbirjax.preprocess.zeiss_tct` (translation CT).  They were extracted verbatim from those two
modules, which previously carried byte-identical copies, so the OLE-parsing layer lives in one place.

``read_metadata`` is intentionally NOT shared here: the two readers parse metadata differently (zeiss
resolves ReferenceData/MultiReferenceData; zeiss_tct is scalar), so each module keeps its own.
``read_xrm`` / ``read_xrm_dir`` therefore take that ``read_metadata`` as an injected argument.

Notes:
    Portions of this code are adapted from the DXchange library: https://github.com/data-exchange/dxchange
"""
import os
import struct
import logging
import numpy as np
import olefile
from pathlib import Path
from .utilities import _normalize_to_float32

logger = logging.getLogger(__name__)


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


def read_xrm(fname, read_metadata, *, normalize_to_float32=True):
    """
    Read a single Xradia ``.xrm`` radiograph and its metadata.

    Notes:
        Portions of this code are adapted from the DXchange library: https://github.com/data-exchange/dxchange

    Args:
        fname (str): String defining the path of file or file name.
        read_metadata (callable): The per-reader metadata parser, called as ``read_metadata(ole) -> dict``.
            It is injected because the zeiss and zeiss_tct readers extract metadata differently (zeiss
            resolves ReferenceData/MultiReferenceData; zeiss_tct is scalar), so each passes its own.
        normalize_to_float32 (bool, optional): If True, convert the image to float32 and normalize integer
            data to ``[0, 1]`` (see :func:`mbirjax.preprocess.utilities._normalize_to_float32`).  If False,
            return the raw image; ``read_xrm_dir`` uses False and normalizes once at the stack level.
            Defaults to True.

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

    # Optionally normalize the scan data to float32.
    if normalize_to_float32:
        arr = _normalize_to_float32(arr)

    ole.close()
    return arr, metadata


def read_xrm_dir(dir_path, read_metadata):
    """
    Read all .xrm files in a directory (filesystem order), stack into (num_views, num_det_rows, num_det_cols),
    and concatenate selected metadata.

    Notes:
        Portions of this code are adapted from the DXchange library: https://github.com/data-exchange/dxchange

    Args:
        dir_path (str): Path to the directory to be read.
        read_metadata (callable): The per-reader metadata parser, called as ``read_metadata(ole) -> dict``
            (injected; see :func:`read_xrm`).

    Returns:
        np.ndarray: Output 3D image with shape (num_views, num_det_rows, num_det_channels).
        dict: Output metadata
    """
    dir_path = Path(dir_path)
    files = [p for p in dir_path.iterdir() if p.is_file()]

    # Load the scan data and metadata from the first file.  Each projection is read RAW
    # (normalize_to_float32=False) and the whole stack is normalized once at the end below.  This is
    # numerically identical to normalizing each projection first (the float32 conversion is a fixed
    # per-dtype scaling, so a second pass on already-float32 data is a no-op) and avoids the redundant
    # per-file normalization.
    proj0, md0 = read_xrm(str(files[0]), read_metadata, normalize_to_float32=False)
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
    # Per-view alignment shifts: accumulated like the positions (metadata starts as a copy of
    # the FIRST file's dict, whose shift entries cover only that file).
    metadata['x_shifts'] = [md0['x_shifts'][0]]
    metadata['y_shifts'] = [md0['y_shifts'][0]]
    # Per-view rotation angles: collect them when the OLE 'Angles' stream is present.  Some acquisitions
    # (e.g. pure translation scans) have no Angles stream, in which case read_metadata returns 'thetas'
    # as None; we then leave metadata['thetas'] as None rather than index into None.
    collect_thetas = md0.get('thetas') is not None
    metadata['thetas'] = [md0['thetas'][0]] if collect_thetas else None

    # Load the remaining files and stack them together
    for i, p in enumerate(files[1:], start=1):
        proj, md = read_xrm(str(p), read_metadata, normalize_to_float32=False)
        arr[i] = proj
        metadata['x_positions'].append(md['x_positions'][0])
        metadata['y_positions'].append(md['y_positions'][0])
        metadata['z_positions'].append(md['z_positions'][0])
        metadata['x_shifts'].append(md['x_shifts'][0])
        metadata['y_shifts'].append(md['y_shifts'][0])
        if collect_thetas and md.get('thetas') is not None:
            metadata['thetas'].append(md['thetas'][0])

    _log_imported_data(str(dir_path), arr)

    # Normalize the whole stack to float32 (see the per-projection note above).
    arr = _normalize_to_float32(arr)

    return arr, metadata
