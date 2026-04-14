"""
convert_txrm_scan.py
---------------
Utility to read a .txrm file, apply exp(-scan) to the image data,
and write the result to a new .txrm file.

Usage:
    Set FILE_PATH (required) and OUTPUT_PATH (optional) below, then run:
        python convert_txrm.py

"""

# -- Configuration -------------------------------------------------------------
# Required: set this to your input .txrm file path.
FILE_PATH = "/depot/bouman/data/Zeiss/foam512R1N3000.txrm"

# Optional: set this to your desired output path including filename.
# Leave as "" to auto-save as <input>_raw_scan.txrm in the same folder as FILE_PATH.
OUTPUT_PATH = "/depot/bouman/data/Zeiss/foam512R1N3000_raw_scan.txrm"
# ------------------------------------------------------------------------------

import os
import shutil
import struct
import warnings
import numpy as np
import olefile


# -- OLE helper functions (adapted from DXchange library) ---------------------

def _read_ole_struct(ole, label, struct_fmt):
    value = None
    if ole.exists(label):
        stream = ole.openstream(label)
        data = stream.read()
        value = struct.unpack(struct_fmt, data)
    return value


def _read_ole_value(ole, label, struct_fmt):
    value = _read_ole_struct(ole, label, struct_fmt)
    if value is not None:
        value = value[0]
    return value


def _read_ole_arr(ole, label, struct_fmt):
    arr = _read_ole_struct(ole, label, struct_fmt)
    if arr is not None:
        arr = np.array(arr)
    return arr


def _read_ole_str(ole, label):
    result = None
    if ole.exists(label):
        stream = ole.openstream(label)
        data = stream.read()
        result = [name.decode('utf-8') for name in data.split(b'\x00') if name]
    return result


def _read_ole_reference(ole, labels, struct_fmt):
    for label in labels:
        if ole.exists(label):
            try:
                return _read_ole_value(ole, label, struct_fmt)
            except Exception:
                try:
                    return _read_ole_arr(ole, label, struct_fmt)
                except Exception:
                    pass
    return None


def _get_ole_data_type(metadata, datatype=None):
    """
    Determine the Numpy data type for image data stored in a Zeiss OLE
    (.xrm, .txrm, .txm) file.
    Notes:
        Portions of this code are adapted from the DXchange library:
        https://github.com/data-exchange/dxchange
    """
    # 10 -> float32;  5 -> uint16
    if datatype is None:
        datatype = metadata["data_type"]
    if datatype == 10:
        return np.dtype(np.float32)
    elif datatype == 5:
        return np.dtype(np.uint16)
    else:
        raise Exception("Unsupported XRM datatype: %s" % str(datatype))


def _read_ole_image(ole, label, metadata, datatype=None):
    """Read a single 2D image from an OLE stream."""
    stream = ole.openstream(label)
    data = stream.read()
    data_type = _get_ole_data_type(metadata, datatype)
    data_type = data_type.newbyteorder('<')
    image = np.reshape(
        np.frombuffer(data, data_type),
        (metadata["num_det_rows"], metadata["num_det_channels"])
    )
    return image


def read_metadata(ole):
    """Read image dimensions and data type metadata from a .txrm OLE file."""
    number_of_images = _read_ole_value(ole, "ImageInfo/NoOfImages", "<I")

    metadata = {
        'num_det_channels': _read_ole_value(ole, 'ImageInfo/ImageWidth', '<I'),
        'num_det_rows':     _read_ole_value(ole, 'ImageInfo/ImageHeight', '<I'),
        'data_type':        _read_ole_value(ole, 'ImageInfo/DataType', '<1I'),
        'num_views':        number_of_images,
    }
    return metadata


# -- core conversion ----------------------------------------------------------

def convert_txrm(input_path: str, output_path: str):
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_path}")

    if not olefile.isOleFile(input_path):
        raise ValueError(f"'{input_path}' does not appear to be a valid .txrm / OLE2 file.")

    # 1. Copy the entire file to preserve ALL metadata streams untouched
    shutil.copy2(input_path, output_path)
    print("  Copied original file -> will patch image streams in-place.")

    # 2. Read all projections into a 3D array (num_views, num_det_rows, num_det_channels)
    with olefile.OleFileIO(input_path) as ole:
        metadata = read_metadata(ole)
        num_views      = metadata["num_views"]
        num_det_rows   = metadata["num_det_rows"]
        num_det_channels = metadata["num_det_channels"]
        dtype_in       = _get_ole_data_type(metadata)

        print(f"  Dimensions : {num_views} views x {num_det_rows} rows x {num_det_channels} channels")
        print(f"  Data type  : {dtype_in}")

        # Allocate 3D array for all projections
        view_indices = np.arange(num_views)
        obj_scan = np.zeros((num_views, num_det_rows, num_det_channels), dtype=dtype_in)

        print("  Reading projections...")
        for i, idx in enumerate(view_indices):
            img_string = "ImageData{}/Image{}".format(
                int(np.ceil((idx + 1) / 100.0)), int(idx + 1))
            obj_scan[i] = _read_ole_image(ole, img_string, metadata)

    # 3. Apply exp(-scan) to the entire 3D array at once
    print("  Applying exp(-scan)...")
    obj_scan_exp = np.exp(-obj_scan.astype(np.float64)).astype(np.float32)

    print(f"  src range    : [{obj_scan.min():.4f}, {obj_scan.max():.4f}]")
    print(f"  exp(-x) range: [{obj_scan_exp.min():.4f}, {obj_scan_exp.max():.4f}]")

    # 4. Write converted projections back into the copied file
    print("  Writing converted projections...")
    with olefile.OleFileIO(output_path, write_mode=True) as ole_out:
        for idx in range(num_views):
            img_string = "ImageData{}/Image{}".format(
                int(np.ceil((idx + 1) / 100.0)), int(idx + 1))
            ole_out.write_stream(img_string, obj_scan_exp[idx].tobytes())

    print(f"\n  Done! Converted file saved to:\n    {output_path}")


# -- entry point --------------------------------------------------------------

def main():
    # Input path
    if not FILE_PATH.strip():
        raise ValueError(
            "FILE_PATH is empty. Please set it to your .txrm file path at the top of this script."
        )
    input_path = FILE_PATH.strip()

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: '{input_path}'")

    # Output path
    if OUTPUT_PATH.strip():
        output_path = OUTPUT_PATH.strip()
    else:
        # Save in the same folder as the input file
        input_dir      = os.path.dirname(os.path.abspath(input_path))
        input_filename = os.path.basename(input_path)
        base, ext      = os.path.splitext(input_filename)
        output_path    = os.path.join(input_dir, base + "_raw_scan" + ext)

    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError("FILE_PATH and OUTPUT_PATH are the same. Please set a different OUTPUT_PATH.")

    print("\n=== convert_txrm : exp(-scan) ===")
    convert_txrm(input_path, output_path)


if __name__ == '__main__':
    main()
