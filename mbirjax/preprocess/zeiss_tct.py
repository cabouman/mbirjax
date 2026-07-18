import os, sys
from operator import itemgetter
import numpy as np
import jax
import jax.numpy as jnp
import warnings
import mbirjax as mj
import mbirjax.preprocess as mjp
import pprint
import olefile
from scipy.ndimage import binary_erosion
from . import _xradia_ole
from ._xradia_ole import (_check_read, _get_ole_data_type, _log_imported_data, _read_ole_struct,
                          _read_ole_value, _read_ole_arr, _read_ole_image, _read_ole_str,
                          get_index_in_list)
pp = pprint.PrettyPrinter(indent=4)


def get_sino_and_model(dataset_dir, *, crop_pixels_sides=0, crop_pixels_top=0, crop_pixels_bottom=0,
                       alu_unit='mm', verbose=1):
    """
    Load a Zeiss translation-CT (TCT) dataset, compute its sinogram, and return a ready-to-reconstruct
    model together with a data-specific weight mask.

    One-call replacement for the ``compute_sino_and_params -> TranslationModel(...) -> set_params ->
    auto_set_recon_geometry`` sequence: it constructs the TranslationModel, applies the detector
    parameters, and computes the reconstruction geometry, so the returned model can never be left with a
    stale (default-pitch) reconstruction grid.

    Unlike the other scanner readers, this one returns ``weights``: a data-specific mask (from
    :func:`compute_weight`) that zeros the dark detector-boundary regions of the TCT detector.  Pass it
    to ``model.recon(sino, weights=weights)``.

    Args:
        dataset_dir (str): Path to the Zeiss TCT scan directory (``obj_scan`` / ``blank_scan`` /
            ``dark_scan`` subfolders).
        crop_pixels_sides (int, optional): Pixels to crop from each lateral side of the detector. Defaults to 0.
        crop_pixels_top (int, optional): Pixels to crop from the top of the detector. Defaults to 0.
        crop_pixels_bottom (int, optional): Pixels to crop from the bottom of the detector. Defaults to 0.
        alu_unit (str, optional): Physical unit for 1 ALU (``'um'``, ``'mm'``, ``'cm'``, ``'m'``). Defaults to ``'mm'``.
        verbose (int, optional): Verbosity level. Defaults to 1.

    Returns:
        tuple: ``(sino, model, weights)`` where

            - ``sino`` (jax array): the computed sinogram, shape (num_views, num_det_rows, num_channels).
            - ``model`` (TranslationModel): a model with its reconstruction geometry already set.
            - ``weights`` (numpy.ndarray): a 3D weight mask (same shape as ``sino``) that excludes the
              dark detector boundary.

    Example:
        .. code-block:: python

            sino, model, weights = mbirjax.preprocess.zeiss_tct.get_sino_and_model(dataset_dir)
            recon, recon_dict = model.recon(sino, weights=weights)
    """
    sino, required_params, optional_params, weights = _compute_sino_and_params(
        dataset_dir, crop_pixels_sides=crop_pixels_sides, crop_pixels_top=crop_pixels_top,
        crop_pixels_bottom=crop_pixels_bottom, alu_unit=alu_unit, verbose=verbose)
    sino, model = mjp.finalize_model(sino, required_params, optional_params)
    return sino, model, weights


def _compute_sino_and_params(dataset_dir, crop_pixels_sides=0, crop_pixels_top=0, crop_pixels_bottom=0, alu_unit='mm', verbose=1):
    """
    Load Zeiss TCT scans and compute the sinogram, build_model-ready parameters, and a weight mask.

    Private helper for :func:`get_sino_and_model`.  Loads object/blank/dark scans and geometry, computes
    the sinogram (transmission + background-offset correction), and builds a dark-boundary weight mask.

    Args:
        dataset_dir (str): Path to the Zeiss TCT scan directory (``obj_scan`` / ``blank_scan`` /
            ``dark_scan`` subfolders).
        crop_pixels_sides (int, optional): Pixels to crop from each side of the detector. Defaults to 0.
        crop_pixels_top (int, optional): Pixels to crop from the top of the detector. Defaults to 0.
        crop_pixels_bottom (int, optional): Pixels to crop from the bottom of the detector. Defaults to 0.
        alu_unit (str, optional): The physical unit used to define 1 ALU. Defaults to ``'mm'``.
        verbose (int, optional): Verbosity level. Defaults to 1.

    Returns:
        tuple: ``(sino, required_params, optional_params, weights)`` where ``required_params`` holds the
        TranslationModel constructor arguments plus a ``geometry_type`` entry so ``build_model`` can
        resolve the class, ``optional_params`` holds the ``set_params`` arguments, and ``weights`` is the
        dark-boundary weight mask (same shape as ``sino``).
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
    # Transmission via the shared, view-sharded core (no downsample or rotation for translation CT).
    sino = mjp.scan_to_sino(obj_scan, blank_scan, dark_scan, defective_pixel_array,
                            downsample_factor=(1, 1), det_rotation=0.0)

    if verbose > 0:
        print("\n\n########## Correcting sinogram data to account for background offset and detector rotation")
    sino = mjp.correct_background_offset(sino)

    if verbose > 0:
        print('obj_scan shape = ', obj_scan.shape)
        print('blank_scan shape = ', blank_scan.shape)
        print('dark_scan shape = ', dark_scan.shape)

    # Normalize for build_model: tag the constructor dict with the geometry class identity.
    translation_params['geometry_type'] = str(mj.TranslationModel)
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
        # The scanner uses a coordinate system different from MBIRJAX
        # Axis mapping:
        #   Scanner x-axis -> MBIRJAX -x-axis
        #   Scanner z-axis -> MBIRJAX y-axis
        #   Scanner y-axis -> MBIRJAX z-axis
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

    # Define 1 ALU as 1 unit of alu_unit
    alu_value = 1

    # Convert physical units to ALU
    source_iso_dist = mjp.to_alu(source_iso_dist, source_iso_dist_unit, alu_unit)
    iso_det_dist = mjp.to_alu(iso_det_dist, iso_det_dist_unit, alu_unit)
    delta_det_channel = mjp.to_alu(delta_det_channel, delta_det_channel_unit, alu_unit)
    delta_det_row = mjp.to_alu(delta_det_row, delta_det_row_unit, alu_unit)
    iso_pixel_pitch = mjp.to_alu(iso_pixel_pitch, iso_pixel_pitch_unit, alu_unit)
    object_x_positions = mjp.to_alu(object_x_positions, object_x_position_unit, alu_unit)
    object_y_positions = mjp.to_alu(object_y_positions, object_y_position_unit, alu_unit)
    object_z_positions = mjp.to_alu(object_z_positions, object_z_position_unit, alu_unit)

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

    # Convert offset parameters to ALU (using the raw pitch) BEFORE cropping, so the configuration crop
    # can be routed through the shared detector-plane primitive.  Assumes det_channel_offset /
    # det_row_offset have units of pixels.
    det_channel_offset *= delta_det_channel
    det_row_offset *= delta_det_row

    # Apply the configuration crop through the shared primitive: it reduces the shape and, for an
    # asymmetric top/bottom crop, shifts det_row_offset (symmetric crops are a no-op).  Translation CT
    # has no downsampling, so the raw pitch is the final pitch.
    num_views = len(translation_vectors)
    num_det_rows, num_det_channels, det_row_offset, det_channel_offset = mjp.apply_config_crop(
        num_det_rows, num_det_channels, det_row_offset, det_channel_offset, delta_det_row, delta_det_channel,
        crop_pixels_top=crop_pixels_top, crop_pixels_bottom=crop_pixels_bottom, crop_pixels_sides=crop_pixels_sides)

    # Calculate recon_shape, delta_voxel, and voxel_row_aspect parameters from the cropped sinogram.
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


def read_xrm(fname, normalize_to_float32=False):
    """
    Read a single Xradia ``.xrm`` radiograph and its metadata (zeiss_tct reader).

    Thin wrapper over :func:`mbirjax.preprocess._xradia_ole.read_xrm` that binds this module's
    :func:`read_metadata`.  Defaults to ``normalize_to_float32=False`` (translation CT reads raw here and
    normalizes once at the stack level in :func:`read_xrm_dir`).  See the shared function for details.
    """
    return _xradia_ole.read_xrm(fname, read_metadata, normalize_to_float32=normalize_to_float32)


def read_xrm_dir(dir_path):
    """
    Read all ``.xrm`` files in a directory and stack them (zeiss_tct reader).

    Thin wrapper over :func:`mbirjax.preprocess._xradia_ole.read_xrm_dir` that binds this module's
    :func:`read_metadata`.  See that function for argument and return details.
    """
    return _xradia_ole.read_xrm_dir(dir_path, read_metadata)


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


# NOTE: view-alignment shift correction (correct_sino_shifts) was REMOVED from this module:
# it had no callers, and whether the .xrm shift fields carry the same meaning for a
# translation-CT acquisition (where the per-view x/y POSITIONS are the geometry itself) is
# unvalidated.  If TCT view alignment is needed, adapt zeiss.correct_sino_shifts -- the
# per-view x_shifts/y_shifts are already collected by read_xrm_dir above.


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
    # Assume that dark boundary region intensity <= dark_region_ratio * median blank scan intensity
    weight_mask_2d = blank_scan[0] >= dark_region_ratio * np.median(blank_scan[0])

    # Add a safety buffer around dark boundary regions
    if safety_buffer > 0:
        padded_mask = np.pad(weight_mask_2d, safety_buffer, mode='constant', constant_values=True)
        padded_mask = binary_erosion(padded_mask, iterations=safety_buffer)
        weight_mask_2d = padded_mask[safety_buffer:-safety_buffer, safety_buffer:-safety_buffer]

    # Broadcast the same mask to all views
    weight_mask = np.broadcast_to(
        weight_mask_2d[np.newaxis, :, :],
        (num_views, num_rows, num_cols),
    ).astype(np.float32)

    return weight_mask