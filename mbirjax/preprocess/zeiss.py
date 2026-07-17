import os, sys
from operator import itemgetter
import numpy as np
import jax
import jax.numpy as jnp
import warnings
import mbirjax
import mbirjax.preprocess as mjp
import pprint
import olefile
from . import _xradia_ole
from ._xradia_ole import (_check_read, _get_ole_data_type, _log_imported_data, _read_ole_struct,
                          _read_ole_value, _read_ole_arr, _read_ole_image, _read_ole_str,
                          get_index_in_list)
pp = pprint.PrettyPrinter(indent=4)


def get_sino_and_model(dataset_dir, *, downsample_factor=(1, 1), subsample_view_factor=1,
                       crop_pixels_sides=0, crop_pixels_top=0, crop_pixels_bottom=0, alu_unit='mm',
                       bg_option="global", zinger_correction=True, auto_crop=False, verbose=1):
    """
    Load a Zeiss Ultra/Versa scan dataset, compute its sinogram, and return a ready-to-reconstruct model.

    One-call replacement for the ``compute_sino_and_params -> (ParallelBeamModel | ConeBeamModel)(...) ->
    set_params -> auto_set_recon_geometry`` sequence.  It selects the model class from the Zeiss scanner
    type -- ``'ultra'`` (Xradia Ultra) is treated as parallel-beam and ``'versa'`` (Xradia Versa) as
    cone-beam -- constructs the model, applies the detector parameters, and computes the reconstruction
    geometry, so the returned model can never be left with a stale (default-pitch) reconstruction grid.

    Thanks to contributions of Amir Koushyar Ziabari of Oak Ridge National Laboratory (ORNL).

    Args:
        dataset_dir (str): Path to the Zeiss ``.txrm`` dataset (``ImageData*/Image*`` scan data and
            Zeiss OLE metadata streams).
        downsample_factor (tuple[int, int], optional): Detector row/channel downsampling. Defaults to ``(1, 1)``.
        subsample_view_factor (int, optional): Keep every n-th view. Defaults to ``1``.
        crop_pixels_sides (int, optional): Pixels to crop from each lateral side of the detector. Defaults to ``0``.
        crop_pixels_top (int, optional): Pixels to crop from the top of the detector. Defaults to ``0``.
        crop_pixels_bottom (int, optional): Pixels to crop from the bottom of the detector. Defaults to ``0``.
        alu_unit (str, optional): Physical unit for 1 ALU (``'um'``, ``'mm'``, ``'cm'``, ``'m'``). Defaults to ``'mm'``.
        bg_option (str or None, optional): Background offset correction: ``None``, ``'global'``, or
            ``'per_view'``. Defaults to ``'global'``.
        zinger_correction (bool, optional): Detect and interpolate zinger pixels. Defaults to ``True``.
        auto_crop (bool, optional): If True, detect and remove blank sinogram margins after the sinogram
            is computed, shrinking the reconstruction. Defaults to False.
        verbose (int, optional): Verbosity level. Defaults to ``1``.

    Returns:
        tuple: ``(sino, model)`` where

            - ``sino`` (jax array): the computed sinogram, shape (num_views, num_det_rows, num_channels).
            - ``model`` (ParallelBeamModel or ConeBeamModel): a CT model with its reconstruction geometry already set.
            (``'ultra'`` -> parallel; ``'versa'`` -> cone; ``'unknown'`` -> cone)

    Example:
        .. code-block:: python

            sino, model = mbirjax.preprocess.zeiss.get_sino_and_model(dataset_dir)
            recon, recon_dict = model.recon(sino)

    Note:
        Reconstruction weights are not returned; generate them with ``mbirjax.gen_weights``.
    """
    sino, required_params, optional_params = _compute_sino_and_params(
        dataset_dir, downsample_factor=downsample_factor, subsample_view_factor=subsample_view_factor,
        crop_pixels_sides=crop_pixels_sides, crop_pixels_top=crop_pixels_top,
        crop_pixels_bottom=crop_pixels_bottom, alu_unit=alu_unit, bg_option=bg_option,
        zinger_correction=zinger_correction, verbose=verbose)
    return mjp.finalize_model(sino, required_params, optional_params, auto_crop=auto_crop)


def _compute_sino_and_params(dataset_dir, downsample_factor=(1, 1), subsample_view_factor=1, crop_pixels_sides=0, crop_pixels_top=0, crop_pixels_bottom=0, alu_unit='mm', bg_option="global", zinger_correction=True, verbose=1):
    """
    Compute the sinogram and build_model-ready parameters from a Zeiss Ultra/Versa ``.txrm`` dataset.

    Private helper for :func:`get_sino_and_model`.  Loads scans and geometry, computes the sinogram
    (downsample -> transmission -> optional zinger, then background-offset and sinogram-shift
    corrections), and resolves the geometry class from the scanner type.

    Thanks to contributions of Amir Koushyar Ziabari of Oak Ridge National Laboratory (ORNL).
    Portions of this code are adapted from the DXchange library: https://github.com/data-exchange/dxchange

    Args:
        dataset_dir (str): Path to the Zeiss ``.txrm`` dataset.
        downsample_factor (tuple[int, int], optional): Downsample factors for detector rows and channels. Defaults to ``(1, 1)``.
        subsample_view_factor (int, optional): Factor by which to subsample views. Defaults to ``1``.
        crop_pixels_sides (int, optional): Pixels to crop from each lateral side of the detector. Defaults to ``0``.
        crop_pixels_top (int, optional): Pixels to crop from the top of the detector. Defaults to ``0``.
        crop_pixels_bottom (int, optional): Pixels to crop from the bottom of the detector. Defaults to ``0``.
        alu_unit (str, optional): The physical unit used to define 1 ALU. Defaults to ``'mm'``.
        bg_option (str or None): Background offset correction (``None``, ``'global'``, ``'per_view'``). Defaults to ``'global'``.
        zinger_correction (bool, optional): Detect and interpolate zinger pixels. Defaults to ``True``.
        verbose (int, optional): Verbosity level. Defaults to ``1``.

    Returns:
        tuple: ``(sino, required_params, optional_params)`` where ``required_params`` holds the model
        constructor arguments plus a ``geometry_type`` entry (``ParallelBeamModel`` for an ``'ultra'``
        scan, ``ConeBeamModel`` otherwise) so ``build_model`` can resolve the class, and
        ``optional_params`` holds the ``set_params`` arguments.
    """
    if verbose > 0:
        print("\n\n########## Loading object, blank, dark scans, and geometry parameters from Zeiss dataset directory")
    obj_scan, blank_scan, dark_scan, zeiss_params = load_scans_and_params(dataset_dir, subsample_view_factor)

    geometry_params, optional_params, zeiss_metadata = convert_zeiss_to_mbirjax_params(zeiss_params, downsample_factor=downsample_factor,
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

    if verbose > 0:
        print("\n\n########## Computing sinogram (downsample -> transmission -> zinger, fused and view-sharded)")
    # Downsample + transmission (+ zinger) via the shared, view-sharded core (zeiss has no detector
    # rotation).  Zinger correction is folded into this single pass when enabled -- no extra host
    # round-trip -- and runs before the offset/shift passes below, so a zinger is removed before a
    # sub-pixel detector shift could interpolate it into its neighbors.
    sino = mjp.scan_to_sino(obj_scan, blank_scan, dark_scan, defective_pixel_array,
                            downsample_factor=downsample_factor, det_rotation=0.0,
                            zinger_pixel_ratio=0.1 if zinger_correction else None)

    if verbose > 0:
        print("\n\n########## Correcting sinogram data to account for background offset and sino offset")
    sino = mjp.correct_background_offset(sino, option=bg_option)

    # Correct sino offset
    sino = correct_sino_shifts(sino, zeiss_params, downsample_factor, subsample_view_factor)

    if verbose > 0:
        print('obj_scan shape = ', obj_scan.shape)
        print('blank_scan shape = ', blank_scan.shape)
        print('dark_scan shape = ', dark_scan.shape)

    # Normalize for build_model: resolve the geometry class from the Zeiss scanner type, mirroring
    # convert_zeiss_to_mbirjax_params -- 'ultra' -> parallel-beam, everything else (versa, or an
    # undetermined 'unknown' for which classify_zeiss_system already warns it is assuming cone) -> cone.
    scanner_type = zeiss_metadata['scanner_type']
    if scanner_type == 'ultra':
        geometry_params['geometry_type'] = str(mbirjax.ParallelBeamModel)
    else:
        geometry_params['geometry_type'] = str(mbirjax.ConeBeamModel)
    return sino, geometry_params, optional_params


def load_scans_and_params(dataset_dir, subsample_view_factor, verbose=1):
    """
    Load the scan data and geometry from a Zeiss scan directory.

    Notes:
        Thanks to contributions of Amir Koushyar Ziabari of Oak Ridge National Laboratory (ORNL).
        Portions of this code are adapted from the DXchange library: https://github.com/data-exchange/dxchange

    Args:
        dataset_dir (str): Path to a Zeiss scan directory (expect a `.txrm` file). Expected structure:
            - ``ImageData*/Image*`` (scan data)
            - ``**/**`` (Zeiss metadata/parameters)
        subsample_view_factor (int, optional): view subsample factor.
        verbose (int, optional): Verbosity level. Defaults to 1.

    Returns:
        tuple: (obj_scan, blank_scan, dark_scan, zeiss_params, zeiss_params)

            - obj_scan (numpy.ndarray): 3D object scan with shape ``(num_views, num_det_rows, num_channels)``.
            - blank_scan (numpy.ndarray): 3D blank scan with shape ``(1, num_det_rows, num_channels)``.
            - dark_scan (numpy.ndarray): 3D dark scan with shape ``(1, num_det_rows, num_channels)``.
                If no dark scan is available, returns a zero array of the same shape.
    """
    ### automatically parse the paths to Zeiss scans from dataset_dir
    data_dir = _parse_filenames_from_dataset_dir(dataset_dir)

    if verbose > 0:
        print("The following files will be used to compute the Zeiss reconstruction:\n",
              f"    - txrm file: {data_dir}\n")

    # Read object scans and metadata
    file_name = _check_read(data_dir)
    try:
        ole = olefile.OleFileIO(file_name)
    except IOError as e:
        print('No such file or directory: %s', file_name)
        raise e

    # Get scanner type
    scanner_type = classify_zeiss_system(file_name)

    # Read metadata from txrm file
    zeiss_params = read_metadata(ole)

    # Create an empty array to store the scan data
    obj_scan = np.empty(
        (
            zeiss_params["num_views"],
            zeiss_params["num_det_rows"],
            zeiss_params["num_det_channels"],
        ),
        dtype=_get_ole_data_type(zeiss_params)
    )

    # Read the (subsampled) scan data from txrm file
    view_indices = np.arange(zeiss_params["num_views"], step=subsample_view_factor)
    obj_scan = np.zeros((len(view_indices), zeiss_params["num_det_rows"], zeiss_params["num_det_channels"]),
                        dtype=_get_ole_data_type(zeiss_params))
    for i, idx in enumerate(view_indices):
        img_string = "ImageData{}/Image{}".format(
            int(np.ceil((idx + 1) / 100.0)), int(idx + 1))
        obj_scan[i] = _read_ole_image(ole, img_string, zeiss_params)

    # Read blank scans
    blank_scan = zeiss_params["reference"]

    # Read dark scans
    # TODO: Currently we assume that there is no dark scan for txrm file
    dark_scan = np.zeros(obj_scan.shape, dtype=obj_scan.dtype)

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

        angle_index = get_index_in_list(axis_names, 'Sample Theta')
        angle_unit = axis_units[angle_index] if angle_index > -1 else 'deg'

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

    # Rotation angles
    angles = -np.array(zeiss_params['thetas'], dtype=float).ravel()
    angles = angles[view_indices]

    # Detector offset parameters
    # MBIRJAX has the reverse convention for the channel shift
    det_channel_offset = -zeiss_params["center_shift"]
    det_row_offset = 0.0    # There doesn't appear to be a Zeiss parameter for detector row offset

    zeiss_params = {
        'scanner_type': scanner_type,
        'source_iso_dist': source_iso_dist,
        'iso_det_dist': iso_det_dist,
        'delta_det_channel': delta_det_channel,
        'delta_det_row': delta_det_row,
        'iso_pixel_pitch': iso_pixel_pitch,
        'opt_mag': opt_mag,
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
        'iso_pixel_pitch_unit': iso_pixel_pitch_unit,
        'angle_unit': angle_unit,
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


def convert_zeiss_to_mbirjax_params(zeiss_params, downsample_factor=(1, 1), crop_pixels_sides=0, crop_pixels_top=0, crop_pixels_bottom=0, alu_unit='mm'):
    """
    Convert geometry parameters from zeiss into mbirjax format, including modifications to reflect crop.

    Notes:
        Thanks to contributions of Amir Koushyar Ziabari of Oak Ridge National Laboratory (ORNL).

    Args:
        zeiss_params (dict): Required Zeiss geometry parameters for reconstruction.
        downsample_factor ((int, int), optional) - Down-sample factors along the detector rows and channels respectively.
            If scan size is not divisible by `downsample_factor`, the scans will be first truncated to a size that is divisible by `downsample_factor`.
        crop_pixels_sides (int, optional): The number of pixels to crop from each side of the sinogram. Defaults to 0.
        crop_pixels_top (int, optional): The number of pixels to crop from top of the sinogram. Defaults to 0.
        crop_pixels_bottom (int, optional): The number of pixels to crop from bottom of the sinogram. Defaults to 0.
        alu_unit (str, optional): The physical unit used to define 1 ALU (Arbitrary Length Unit). Defaults to 'mm'.
            Supported units input: 'um', 'mm', 'cm', 'm'.

    Returns:
        geometry_params (dict): Required parameters for the tomography model constructor.
        optional_params (dict): Additional geometry model parameters to be set using set_params()
        zeiss_metadata (dict): some metadata stored in Zeiss txrm file.
    """
    # Get zeiss parameters
    source_iso_dist, iso_det_dist, source_iso_dist_unit, iso_det_dist_unit = itemgetter('source_iso_dist', 'iso_det_dist', 'source_iso_dist_unit', 'iso_det_dist_unit')(zeiss_params)
    delta_det_channel, delta_det_row, delta_det_channel_unit, delta_det_row_unit = itemgetter('delta_det_channel', 'delta_det_row', 'delta_det_channel_unit', 'delta_det_row_unit')(zeiss_params)
    iso_pixel_pitch, iso_pixel_pitch_unit = itemgetter('iso_pixel_pitch', 'iso_pixel_pitch_unit')(zeiss_params)
    opt_mag = itemgetter('opt_mag')(zeiss_params)
    num_det_rows, num_det_channels = itemgetter('num_det_rows', 'num_det_channels')(zeiss_params)
    angles, angle_unit = itemgetter('angles', 'angle_unit')(zeiss_params)
    det_row_offset, det_channel_offset = itemgetter('det_row_offset', 'det_channel_offset')(zeiss_params)

    scanner_type = zeiss_params['scanner_type']

    # Define 1 ALU as 1 unit of alu_unit
    alu_value = 1

    # Convert physical units to ALU
    source_iso_dist = mjp.to_alu(source_iso_dist, source_iso_dist_unit, alu_unit)
    iso_det_dist = mjp.to_alu(iso_det_dist, iso_det_dist_unit, alu_unit)
    delta_det_channel = mjp.to_alu(delta_det_channel, delta_det_channel_unit, alu_unit)
    delta_det_row = mjp.to_alu(delta_det_row, delta_det_row_unit, alu_unit)
    iso_pixel_pitch = mjp.to_alu(iso_pixel_pitch, iso_pixel_pitch_unit, alu_unit)

    # Compute default value of source to detector distance
    source_detector_dist = source_iso_dist + iso_det_dist

    # Convert angles to radians
    if angle_unit == 'deg':
        angles = np.deg2rad(angles)
    else:
        pass

    # Make conversions for optical magnification
    # In this case, the "detector" is actually a scintillator
    if opt_mag is not None and not np.isinf(opt_mag):
        # Compute total magnification = (optical magnification) * (magnification to scintillator)
        scintillator_mag = source_detector_dist/source_iso_dist
        magnification = opt_mag * scintillator_mag
    else:
        magnification = 1.0

    # Compute source to equivalent quantities accounting for total magnification
    source_detector_dist = magnification * source_iso_dist
    delta_det_channel = magnification * iso_pixel_pitch
    delta_det_row = magnification * iso_pixel_pitch

    # Convert to ALU: This assumes that the det_channel_offset and det_row_offset have units of pixels.
    det_channel_offset *= delta_det_channel
    det_row_offset *= delta_det_row

    # Apply the configuration crop through the shared primitive: it reduces the shape and, for an
    # asymmetric top/bottom crop, shifts det_row_offset (symmetric crops are a no-op).  Offsets are
    # already in ALU and the crop is in raw detector pixels (matched by the raw pitch); downsampling is
    # applied afterward.
    num_det_rows, num_det_channels, det_row_offset, det_channel_offset = mjp.apply_config_crop(
        num_det_rows, num_det_channels, det_row_offset, det_channel_offset, delta_det_row, delta_det_channel,
        crop_pixels_top=crop_pixels_top, crop_pixels_bottom=crop_pixels_bottom, crop_pixels_sides=crop_pixels_sides)

    # Adjust detector size and pixel pitch params w.r.t. downsampling arguments
    num_det_rows = num_det_rows // downsample_factor[0]
    num_det_channels = num_det_channels // downsample_factor[1]

    delta_det_row *= downsample_factor[0]
    delta_det_channel *= downsample_factor[1]

    iso_pixel_pitch *= downsample_factor[0]

    # Create a dictionary to store MBIR parameters
    # 'ultra' is treated as parallel-beam, and 'versa' is treated as cone-beam.
    if scanner_type == "ultra":
        num_views = len(angles)
        geometry_params = dict()
        geometry_params['sinogram_shape'] = (num_views, num_det_rows, num_det_channels)
        geometry_params["angles"] = angles

        optional_params = dict()
        optional_params['delta_det_channel'] = delta_det_channel
        optional_params['delta_det_row'] = delta_det_row
        optional_params['delta_voxel'] = delta_det_channel    # For parallel-beam geometry, there is no geometric magnification
        optional_params['det_row_offset'] = det_row_offset
        optional_params['det_channel_offset'] = det_channel_offset
        optional_params['alu_unit'] = alu_unit
        optional_params['alu_value'] = alu_value
    else:
        num_views = len(angles)
        geometry_params = dict()
        geometry_params['sinogram_shape'] = (num_views, num_det_rows, num_det_channels)
        geometry_params["angles"] = angles
        geometry_params['source_detector_dist'] = source_detector_dist
        geometry_params['source_iso_dist'] = source_iso_dist

        optional_params = dict()
        optional_params['delta_det_channel'] = delta_det_channel
        optional_params['delta_det_row'] = delta_det_row
        optional_params['delta_voxel'] = iso_pixel_pitch
        optional_params['det_row_offset'] = det_row_offset
        optional_params['det_channel_offset'] = det_channel_offset
        optional_params['alu_unit'] = alu_unit
        optional_params['alu_value'] = alu_value

    # Create a dictionary for other zeiss metadata
    # TODO: More metadata will be added in the future based on users' interest
    zeiss_metadata = {"scanner_type": scanner_type}

    return geometry_params, optional_params, zeiss_metadata


######## subroutines for parsing Zeiss object scan, blank scan, and dark scan
def _parse_filenames_from_dataset_dir(dataset_dir):
    """
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
        raise FileNotFoundError(dataset_dir)


def read_xrm(fname, normalize_to_float32=True):
    """
    Read a single Xradia ``.xrm`` radiograph and its metadata (zeiss reader).

    Thin wrapper over :func:`mbirjax.preprocess._xradia_ole.read_xrm` that binds this module's
    :func:`read_metadata`.  See that function for argument and return details.
    """
    return _xradia_ole.read_xrm(fname, read_metadata, normalize_to_float32=normalize_to_float32)


def read_xrm_dir(dir_path):
    """
    Read all ``.xrm`` files in a directory and stack them (zeiss reader).

    Thin wrapper over :func:`mbirjax.preprocess._xradia_ole.read_xrm_dir` that binds this module's
    :func:`read_metadata`.  See that function for argument and return details.
    """
    return _xradia_ole.read_xrm_dir(dir_path, read_metadata)

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
    number_of_reference = _read_ole_reference(
        ole, ["ReferenceData/ImageInfo/NoOfImages", "MultiReferenceData/ImageInfo/NoOfImages"], "<I")

    metadata = {
        'num_det_channels': _read_ole_value(ole, 'ImageInfo/ImageWidth', '<I'),
        'num_det_rows': _read_ole_value(ole, 'ImageInfo/ImageHeight', '<I'),
        'data_type': _read_ole_value(ole, 'ImageInfo/DataType', '<1I'),
        'reference_data_type': _read_ole_reference(
            ole, ['ReferenceData/DataType', 'MultiReferenceData/DataType'], '<1I'),
        'num_views': number_of_images,
        'num_reference': number_of_reference,
        'iso_pixel_pitch': _read_ole_value(ole, 'ImageInfo/PixelSize', '<f'),
        'det_pixel_pitch': _read_ole_value(ole, 'DetAssemblyInfo/CamPixelSize', '<f'),
        'iso_det_dist': _read_ole_value(ole, 'ImageInfo/DtoRADistance', "<{0}f".format(number_of_images)),
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
        'center_shift': _read_ole_reference(
            ole, ["AutoRecon/CenterShift", "ReconSettings/CenterShift"], '<f'),
        'opt_mag': _read_ole_value(ole, "DetAssemblyInfo/OpticalMagnification", '<f'),
        'fan_angle': _read_ole_reference(
            ole, ['ReferenceData/ImageInfo/FanAngle', 'MultiReferenceData/ImageInfo/FanAngle'], '<{0}f'.format(number_of_reference)),
        'cone_angle': _read_ole_reference(
            ole,['ReferenceData/ImageInfo/ConeAngle', 'MultiReferenceData/ImageInfo/ConeAngle'], '<{0}f'.format(number_of_reference)),
        'axis_names': _read_ole_str(ole, 'PositionInfo/AxisNames'),
        'axis_units': _read_ole_str(ole, 'PositionInfo/AxisUnits')
    }


    reference = None
    if ole.exists('ReferenceData'):
        reference = _read_ole_image(ole, 'ReferenceData/Image', metadata, metadata['reference_data_type'])
    elif ole.exists('MultiReferenceData'):
        array_of_reference = []
        for idx in range(metadata['num_reference']):
            img_string = f"MultiReferenceData/Image{idx + 1}"
            if ole.exists(img_string):
                array_of_reference.append(
                    _read_ole_image(ole, img_string, metadata, metadata['reference_data_type'])
                )

        if len(array_of_reference) > 0:
            reference = np.stack(array_of_reference, axis=0)
    else:
        warnings.warn('No reference data available.  Using an array of all 1s.')
        reference = np.ones((metadata['num_det_rows'], metadata['num_det_channels']))

    if reference.ndim == 2:
        reference = reference[None, :, :]

    metadata['reference'] = reference

    return metadata


def _read_ole_reference(ole, labels, struct_fmt):
    """
    Reads the reference-related value from any matching label in an ole file

    First tries to unpack the data as a scalar value. If that does not work,
    it attempts to unpack it as a NumPy array.

    Args:
        ole (OleFileIO): An ole file to read from.
        labels (list of str): List of possible labels associated with the reference data parameters.
        struct_fmt (str): Format of the OLE file.

    Returns:
        int, float, or ndarray: The unpacked value from a matching label. Return None if no matching label exists.
    """
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


def classify_zeiss_system(fname):
    """
    Determine the Zeiss microscope system type from a .txrm file.

    The classification is based on the presence of system-specific metadata:
    Ultra files contain "ConfigZonePlates", while Versa files contain "ConfigXrayFilters/Filter0/Name".

    Args:
        fname (str): Path to a Zeiss .txrm file.

    Returns:
        str: 'versa'   -- cone-beam micro-CT (Xradia Versa)
             'ultra'   -- parallel-beam nano-CT (Xradia Ultra)
             'unknown'  -- The system type cannot be determined
    """

    if not olefile.isOleFile(str(fname)):
        raise ValueError(f"'{fname}' is not a valid Zeiss .txrm file.")

    with olefile.OleFileIO(str(fname)) as ole:
        paths = ["/".join(entry) for entry in ole.listdir()]

    if any("ConfigZonePlates" in path for path in paths):
        return 'ultra'

    if any(
        path == "ConfigXrayFilters/Filter0/Name"
        or path.endswith("/ConfigXrayFilters/Filter0/Name")
        for path in paths
    ):
        return 'versa'

    warnings.warn(
        f"Unable to determine the Zeiss system type from '{fname}'. "
        "Neither Ultra nor Versa metadata was found."
        "Assuming the file using cone beam geometry."
    )

    return 'unknown'

######## END subroutines for parsing Zeiss object scan, blank scan, and dark scan


def correct_sino_shifts(sino, zeiss_params, downsample_factor, subsample_view_factor):
    """
    Align each sinogram view based on the per-view projection offset.

    The txrm file stores the horizontal (x-shift) and vertical (y-shift) offsets for each projection.
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
        zeiss_params (dict): parameters stored in Zeiss txrm file.
        downsample_factor (Tuple[int, int], optional): Downsample factors for detector rows and channels. Defaults to (1, 1).
        subsample_view_factor (int, optional): Factor by which to subsample views. Defaults to 1.

    Returns:
        corrected_sino (numpy array or jax array): 3D sinogram data after alignment
    """
    # Get sinogram view offset
    # OUT-OF-PLACE scaling: the slice of a numpy array is a VIEW, so an in-place /= here would
    # silently rescale the caller's zeiss_params (and a second call would double-divide).
    sino_x_offset = np.asarray(zeiss_params["x_shifts"][::subsample_view_factor],
                               dtype=np.float64) / downsample_factor[1]
    sino_y_offset = np.asarray(zeiss_params["y_shifts"][::subsample_view_factor],
                               dtype=np.float64) / downsample_factor[0]

    ### Pad the sinogram to handle boundaries
    # The translation below applies each view's ABSOLUTE offset, so the padding must cover the
    # largest absolute shift (padding by the across-view RANGE under-pads whenever the shifts
    # share a common offset, corrupting the boundary region).
    pad_size = int(np.ceil(np.maximum(np.max(np.abs(sino_x_offset)),
                                      np.max(np.abs(sino_y_offset)))))

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