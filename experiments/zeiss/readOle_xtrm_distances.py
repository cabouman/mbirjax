# Amir Koushyar Ziabari, ziabariak@ORNL.gov 2021-2022

import olefile
import numpy as np
import struct
import logging
import math

def _read_ole_struct(ole, label, struct_fmt):
    """
    Reads the struct associated with label in an ole file
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
    """
    value = _read_ole_struct(ole, label, struct_fmt)
    if value is not None:
        value = value[0]
    return value

def calc_XrayDet_distances(data_path):
    ole = olefile.OleFileIO(data_path)
    ole.listdir('ImageInfo')

    ref_dir, _ = _get_ole_reference_directory(ole)

    RefD2RADistance = _read_ole_value(ole, f'{ref_dir}/RefD2RADistance', '<f')
    RefS2RADistance = _read_ole_value(ole, f'{ref_dir}/RefS2RADistance', '<f')
    RefS2RADistance = -RefS2RADistance

    return RefD2RADistance, RefS2RADistance

logger = logging.getLogger(__name__)

def _read_ole_arr(ole, label, struct_fmt):
    """
    Reads the numpy array associated with label in an ole file
    """
    arr = _read_ole_struct(ole, label, struct_fmt)
    if arr is not None:
        arr = np.array(arr)
    return arr

def _get_ole_data_type(metadata, datatype=None):
    # 10 float; 5 uint16 (unsigned 16-bit (2-byte) integers)
    if datatype is None:
        datatype = metadata["data_type"]
    if datatype == 10:
        return np.dtype(np.float32)
    elif datatype == 5:
        return np.dtype(np.uint16)
    else:
        raise Exception("Unsupport XRM datatype: %s" % str(datatype))

def _read_ole_image(ole, label, metadata, datatype=None):
    stream = ole.openstream(label)
    data = stream.read()
    data_type = _get_ole_data_type(metadata, datatype)
    data_type = data_type.newbyteorder('<')
    image = np.reshape(
        np.frombuffer(data, data_type),
        (metadata["image_height"], metadata["image_width"], )
    )
    return image

def _get_ole_reference_directory(ole):
    """
    Returns the ole directory name for reference image(s) and the number of images
    Args:
        ole:

    Returns:
        ole directory name
        number of images
    """
    if ole.exists('referencedata'):
        ref_dir_name = 'referencedata'
        num_ref_images = 1
    elif ole.exists('multireferencedata'):
        ref_dir_name = 'multireferencedata'
        num_ref_images = _read_ole_value(ole, 'multireferencedata/totalrefimages', '<i')
    else:
        print("No reference directory found among")
        print(ole.listdir())
        raise Exception("No reference directory found")

    return ref_dir_name, num_ref_images


def read_ole_metadata(ole):
    """
    Read metadata from an xradia OLE file (.xrm, .txrm, .txm).

    Parameters
    ----------
    ole : OleFileIO instance
        An ole file to read from.

    Returns
    -------
    tuple
        A tuple of image metadata.
    """
    # metadata["center_shift"],
    # metadata["current"],
    # metadata["voltage"],
    # metadata["exposure_time"],
    # metadata["binning"],
    # metadata["filter"],
    number_of_images = _read_ole_value(ole, "ImageInfo/NoOfImages", "<I")
    ## to get all the labels:
    # labels = ole.listdir()
    #center_shift = _read_ole_value(ole, "SampleInfo/center_shift", '<f')
    #['ReferenceData', 'ImageInfo', 'FanAngle']#_read_ole_value(ole, 'ReferenceData/ImageInfo/FanAngle','<f')

    ref_dir, num_ref_images = _get_ole_reference_directory(ole)

    metadata = {
        'facility': _read_ole_value(ole, 'SampleInfo/Facility', '<50s'),
        'image_width': _read_ole_value(ole, 'ImageInfo/ImageWidth', '<I'),
        'image_height': _read_ole_value(ole, 'ImageInfo/ImageHeight', '<I'),
        'data_type': _read_ole_value(ole, 'ImageInfo/DataType', '<1I'),
        'number_of_images': number_of_images,
        'pixel_size': _read_ole_value(ole, 'ImageInfo/pixelsize', '<f'),
        'reference_filename': _read_ole_value(ole, 'ImageInfo/referencefile', '<260s'),
        'reference_data_type': _read_ole_value(ole, f'{ref_dir}/DataType', '<1I'),
        # NOTE: converting theta to radians from degrees
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
        'current':_read_ole_value(
            ole, "ImageInfo/XrayCurrent", "<{0}f".format(number_of_images)),
        'voltage':_read_ole_value(
            ole, "ImageInfo/XrayVoltage", "<{0}f".format(number_of_images)),
        'ExpTimes':_read_ole_value(
            ole, "ImageInfo/ExpTimes", "<{0}f".format(number_of_images)),
        'center_shift':_read_ole_value(ole, "ReconSettings/CenterShift", '<f'),
        'opt_mag': _read_ole_value(ole, f'{ref_dir}/ImageInfo/OpticalMagnification', '<f'),
        'fan_angle':_read_ole_value(ole, f'{ref_dir}/ImageInfo/FanAngle', f'<{num_ref_images}f'),
    }
    # special case to remove trailing null characters
    reference_filename = _read_ole_value(ole, 'ImageInfo/referencefile', '<260s')
    if reference_filename is not None:
        for i in range(len(reference_filename)):
            if reference_filename[i] == '\x00':
                #null terminate
                reference_filename = reference_filename[:i]
                break
    metadata['reference_filename'] = reference_filename
    if num_ref_images == 1 and ole.exists(f'{ref_dir}/image'):
        reference = _read_ole_image(ole, f'{ref_dir}/image', metadata, metadata['reference_data_type'])
    elif num_ref_images > 1 and ole.exists(f'{ref_dir}/image1'):
        reference = 0
        for ind in np.arange(1, num_ref_images + 1):
            reference += _read_ole_image(ole, f'{ref_dir}/image{ind}', metadata, metadata['reference_data_type'])
        reference = reference / num_ref_images
    else:
        raise FileNotFoundError('Reference image not found')
    metadata['reference'] = reference
    return metadata

def _make_slice_object_a_tuple(slc):
    """
    Fix up a slc object to be tuple of slices.
    slc = None returns None
    slc is container and each element is converted into a slice object

    Parameters
    ----------
    slc : None or sequence of tuples
        Range of values for slicing data in each axis.
        ((start_1, end_1, step_1), ... , (start_N, end_N, step_N))
        defines slicing parameters for each axis of the data matrix.
    """
    if slc is None:
        return None  # need arr shape to create slice
    fixed_slc = list()
    for s in slc:
        if not isinstance(s, slice):
            # create slice object
            if s is None or isinstance(s, int):
                # slice(None) is equivalent to np.s_[:]
                # numpy will return an int when only an int is passed to
                # np.s_[]
                s = slice(s)
            else:
                s = slice(*s)
        fixed_slc.append(s)
    return tuple(fixed_slc)

def _shape_after_slice(shape, slc):
    """
    Return the calculated shape of an array after it has been sliced.
    Only handles basic slicing (not advanced slicing).

    Parameters
    ----------
    shape : tuple of ints
        Tuple of ints defining the ndarray shape
    slc : tuple of slices
        Object representing a slice on the array.  Should be one slice per
        dimension in shape.

    """
    if slc is None:
        return shape
    new_shape = list(shape)
    slc = _make_slice_object_a_tuple(slc)
    for m, s in enumerate(slc):
        # indicies will perform wrapping and such for the shape
        start, stop, step = s.indices(shape[m])
        new_shape[m] = int(math.ceil((stop - start) / float(step)))
        if new_shape[m] < 0:
            new_shape[m] = 0
    return tuple(new_shape)

def _log_imported_data(fname, arr):
    logger.debug('Data shape & type: %s %s', arr.shape, arr.dtype)
    logger.info('Data successfully imported: %s', fname)

def read_txrm(file_name, slice_range=None):
    """
    Read data from a .txrm file, a compilation of .xrm files.

    Parameters
    ----------
    file_name : str
        String defining the path of file or file name.
    slice_range : sequence of tuples, optional
        Range of values for slicing data in each axis.
        ((start_1, end_1, step_1), ... , (start_N, end_N, step_N))
        defines slicing parameters for each axis of the data matrix.

    Returns
    -------
    ndarray
        Array of 2D images.

    dictionary
        Dictionary of metadata.
    """
    try:
        ole = olefile.OleFileIO(file_name)
    except IOError:
        print('No such file or directory: %s', file_name)
        return False

    metadata = read_ole_metadata(ole)

    array_of_images = np.empty(
        _shape_after_slice(
            (
                metadata["number_of_images"],
                metadata["image_height"],
                metadata["image_width"]#,
                # metadata["center_shift"],
                # metadata["current"],
                # metadata["voltage"],
                # metadata["exposure_time"],
                # metadata["binning"],
                # metadata["filter"],
            ),
            slice_range
        ),
        dtype=_get_ole_data_type(metadata)
    )

    if slice_range is None:
        slice_range = (slice(None), slice(None), slice(None))
    else:
        slice_range = _make_slice_object_a_tuple(slice_range)

    for i, idx in enumerate(range(*slice_range[0].indices(metadata["number_of_images"]))):
        img_string = "ImageData{}/Image{}".format(
            int(np.ceil((idx + 1) / 100.0)), int(idx + 1))
        array_of_images[i] = _read_ole_image(ole, img_string, metadata)[slice_range[1:]]

    reference = metadata['reference']
    if reference is not None:
        metadata['reference'] = reference[slice_range[1:]]

    _log_imported_data(file_name, array_of_images)

    ole.close()
    return array_of_images, metadata
