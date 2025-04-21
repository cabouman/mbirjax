import os, sys
import re
from operator import itemgetter
import numpy as np
import warnings
import mbirjax.preprocess as preprocess
import glob
import pprint
pp = pprint.PrettyPrinter(indent=4)


def compute_sino_and_params(dataset_dir, downsample_factor=(1, 1), subsample_view_factor=1,
                            crop_pixels_sides=None, crop_pixels_top=None, crop_pixels_bottom=None):
    """
    Load NSI sinogram data and prepare all needed arrays and parameters for a ConeBeamModel reconstruction.

    This function computes the sinogram and geometry parameters from an NSI scan directory containing scan data and parameters.
    More specifically, the function performs the following operations in a single easy-to-use manner:

    1. Loads the object, blank, and dark scans, as well as the geometry parameters from an NSI dataset directory.

    2. Computes the sinogram from object, blank, and dark scans.

    3. Replaces defective pixels with interpolated values.

    4. Performs background offset correction to the sinogram from the edge pixels.

    5. Corrects sinogram data to account for detector rotation.

    Args:
        dataset_dir (string): Path to an NSI scan directory. The directory is assumed to have the following structure:

            - ``*.nsipro`` (NSI config file)
            - ``Geometry*.rtf`` (geometry report)
            - ``Radiographs*/`` (directory containing all radiograph images)
            - ``**/gain0.tif`` (blank scan image)
            - ``**/offset.tif`` (dark scan image)
            - ``**/*.defect`` (defective pixel information)

        downsample_factor ((int, int), optional) - Down-sample factors along the detector rows and channels respectively.
            If scan size is not divisible by `downsample_factor`, the scans will be first truncated to a size that is divisible by `downsample_factor`.
        subsample_view_factor (int, optional): View subsample factor. By default no view subsampling will be performed.
        crop_pixels_sides (int, optional): The number of pixels to crop from each side of the sinogram. Defaults to None, in which case the NSI config file is used.
        crop_pixels_top (int, optional): The number of pixels to crop from top of the sinogram. Defaults to None, in which case the NSI config file is used.
        crop_pixels_bottom (int, optional): The number of pixels to crop from bottom of the sinogram. Defaults to None, in which case the NSI config file is used.

    Returns:
        tuple: [sinogram, cone_beam_params, optional_params]

            sino (jax array): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels).
            cone_beam_params (dict): Required parameters for the ConeBeamModel constructor.
            optional_params (dict): Additional ConeBeamModel parameters to be set using set_params().

    Example:
        .. code-block:: python

            # Get data and recon parameters
            sino, cone_beam_params, optional_params = mbirjax.preprocess.NSI.compute_sino_and_params(dataset_dir, downsample_factor=downsample_factor, subsample_view_factor=subsample_view_factor)

            # Create the model and set the parameters
            ct_model = mbirjax.ConeBeamModel(**cone_beam_params)
            ct_model.set_params(**optional_params)
            ct_model.set_params(sharpness=sharpness, verbose=1)

            # Compute sinogram weights and do the reconstruction
            weights = ct_model.gen_weights(sino, weight_type='transmission_root')
            recon, recon_params = ct_model.recon(sino, weights=weights)

    """

    print("\n\n########## Loading object, blank, dark scans, and geometry parameters from NSI dataset directory")
    obj_scan, blank_scan, dark_scan, nsi_params, defective_pixel_array = \
            load_scans_and_params(dataset_dir, subsample_view_factor=subsample_view_factor)

    # Get the crops from the config file if not provided and make sure they are symmetric
    # TODO:  adjust detector offsets for asymmetric crops
    max_crop = nsi_params['max_crop']
    if crop_pixels_sides is None:
        crop_pixels_sides = max_crop
    if crop_pixels_top is None:
        crop_pixels_top = max_crop
    if crop_pixels_bottom is None:
        crop_pixels_bottom = max_crop
    if crop_pixels_bottom != crop_pixels_top:
        warnings.warn('Only symmetric cropping is allowed in this release.  Replacing top and bottom crops with their max.')
        crop_pixels_top = max(crop_pixels_top, crop_pixels_bottom)
        crop_pixels_bottom = max(crop_pixels_bottom, crop_pixels_top)

    cone_beam_params, optional_params = convert_nsi_to_mbirjax_params(nsi_params, downsample_factor=downsample_factor,
                                                                      crop_pixels_sides=crop_pixels_sides,
                                                                      crop_pixels_top=crop_pixels_top,
                                                                      crop_pixels_bottom=crop_pixels_bottom)

    print("\n\n########## Cropping and downsampling scans")
    ### crop the scans based on input params
    obj_scan, blank_scan, dark_scan, defective_pixel_array = preprocess.crop_scans(obj_scan, blank_scan, dark_scan,
                                                                                   crop_pixels_sides=crop_pixels_sides,
                                                                                   crop_pixels_top=crop_pixels_top,
                                                                                   crop_pixels_bottom=crop_pixels_bottom,
                                                                                   defective_pixel_array=defective_pixel_array)

    ### downsample the scans with block-averaging
    if downsample_factor[0]*downsample_factor[1] > 1:
        obj_scan, blank_scan, dark_scan, defective_pixel_array = preprocess.downsample_scans(obj_scan, blank_scan, dark_scan,
                                                                                             downsample_factor=downsample_factor,
                                                                                             defective_pixel_array=defective_pixel_array)

    print("\n\n########## Computing sinogram from object, blank, and dark scans")
    sino = preprocess.compute_sino_transmission(obj_scan, blank_scan, dark_scan, defective_pixel_array)
    scan_shapes = obj_scan.shape, blank_scan.shape, dark_scan.shape
    del obj_scan, blank_scan, dark_scan  # delete scan images to save memory

    print("\n\n########## Correcting sinogram data to account for background offset and detector rotation")
    background_offset = preprocess.estimate_background_offset(sino)
    print("background_offset = ", background_offset)

    det_rotation = optional_params["det_rotation"]
    sino = preprocess.correct_det_rotation_and_background(sino, det_rotation=det_rotation, background_offset=background_offset)
    del optional_params["det_rotation"]  # We delete this since it's not an allowed parameter in TomographyModel.

    # print("MBIRJAX geometry parameters:")
    # pp.pprint(cone_beam_params)
    # pp.pprint(optional_params)
    print('obj_scan shape = ', scan_shapes[0])
    print('blank_scan shape = ', scan_shapes[1])
    print('dark_scan shape = ', scan_shapes[2])

    return sino, cone_beam_params, optional_params


def load_scans_and_params(dataset_dir, view_id_start=0, view_id_end=None, subsample_view_factor=1):
    """
    Load the object scan, blank scan, dark scan, view angles, defective pixel information, and geometry parameters from an NSI scan directory.

    This function loads the sinogram data and parameters from an NSI scan directory for users who would prefer to implement custom preprocessing of the data.

    Args:
        dataset_dir (string): Path to an NSI scan directory. The directory is assumed to have the following structure:

            - ``*.nsipro`` (NSI config file)
            - ``Geometry*.rtf`` (geometry report)
            - ``Radiographs*/`` (directory containing all radiograph images)
            - ``**/gain0.tif`` (blank scan image)
            - ``**/offset.tif`` (dark scan image)
            - ``**/*.defect`` (defective pixel information)

        view_id_start (int, optional): view index corresponding to the first view.
        view_id_end (int, optional): view index corresponding to the last view. If None, this will be equal to the total number of object scan images in ``obj_scan_dir``.
        subsample_view_factor (int, optional): view subsample factor.

    Returns:
        tuple: [obj_scan, blank_scan, dark_scan, cone_beam_params, optional_params, defective_pixel_list]

            obj_scan (jax array): 3D object scan with shape (num_views, num_det_rows, num_det_channels).
            blank_scan (jax array): 3D blank scan with shape (1, num_det_rows, num_det_channels).
            dark_scan (jax array): 3D dark scan with shape (1, num_det_rows, num_det_channels).
            nsi_params (dict): Required parameters needed for convert_nsi_to_mbirjax_params().
            defective_pixel_array (ndarray): An nx2 array containing indices of invalid sinogram pixels, with the format (detector_row_idx, detector_channel_idx).
    """
    ### automatically parse the paths to NSI metadata and scans from dataset_dir
    config_file_path, geom_report_path, obj_scan_dir, blank_scan_path, dark_scan_path, defective_pixel_path = \
        _parse_filenames_from_dataset_dir(dataset_dir)

    print("The following files will be used to compute the NSI reconstruction:\n",
          f"    - NSI config file: {config_file_path}\n",
          f"    - Geometry report: {geom_report_path}\n",
          f"    - Radiograph directory: {obj_scan_dir}\n",
          f"    - Blank scan image: {blank_scan_path}\n",
          f"    - Dark scan image: {dark_scan_path}\n",
          f"    - Defective pixel information: {defective_pixel_path}\n")
    ### NSI param tags in nsipro file
    tag_section_list = [['source', 'Result'],                           # vector from origin to source
                        ['reference', 'Result'],                        # vector from origin to first row and column of the detector
                        ['pitch', 'Object Radiograph'],                 # detector pixel pitch
                        ['width pixels', 'Detector'],                   # number of detector rows
                        ['height pixels', 'Detector'],                  # number of detector channels
                        ['number', 'Object Radiograph'],                # number of views
                        ['Rotation range', 'CT Project Configuration'], # Range of rotation angle (usually 360)
                        ['rotate', 'Correction'],                       # rotation of radiographs
                        ['flipH', 'Correction'],                        # Horizontal flip (boolean)
                        ['flipV', 'Correction'],                        # Vertical flip (boolean)
                        ['angleStep', 'Object Radiograph'],             # step size of adjacent view angles
                        ['clockwise', 'Processed'],                     # rotation direction (boolean)
                        ['axis', 'Result'],                             # unit vector in direction ofrotation axis
                        ['normal', 'Result'],                           # unit vector in direction of source-detector line
                        ['horizontal', 'Result'],                       # unit vector in direction of detector rows
                        ['crop', 'Radiograph']                          # 4-tuple of pixels to crop from each view
                       ]
    assert(os.path.isfile(config_file_path)), f'Error! NSI config file does not exist. Please check whether {config_file_path} is a valid file.'
    NSI_params = _read_str_from_config(config_file_path, tag_section_list)

    # vector from origin to source
    r_s = NSI_params[0].split(' ')
    r_s = np.array([np.single(elem) for elem in r_s])

    # vector from origin to reference, where reference is the center of first row and column of the detector
    r_r = NSI_params[1].split(' ')
    r_r = np.array([np.single(elem) for elem in r_r])

    # correct the coordinate of (0,0) detector pixel based on "Geometry Report.rtf"
    x_r, y_r = _read_detector_location_from_geom_report(geom_report_path)
    r_r[0] = x_r
    r_r[1] = y_r
    print("Corrected coordinate of (0,0) detector pixel (from Geometry Report) = ", r_r)

    # detector pixel pitch
    pixel_pitch_det = NSI_params[2].split(' ')
    delta_det_channel = np.single(pixel_pitch_det[0])
    delta_det_row = np.single(pixel_pitch_det[1])

    # dimension of radiograph
    num_det_channels = int(NSI_params[3])
    num_det_rows = int(NSI_params[4])

    # total number of radiograph scans
    num_acquired_scans = int(NSI_params[5])

    # total angles (usually 360 for 3D data, and (360*number_of_full_rotations) for 4D data
    total_angles = int(NSI_params[6])

    # Radiograph rotation (degree)
    scan_rotate = int(NSI_params[7])
    if (scan_rotate == 180) or (scan_rotate == 0):
        print('scans are in portrait mode!')
    elif (scan_rotate == 270) or (scan_rotate == 90):
        print('scans are in landscape mode!')
        num_det_channels, num_det_rows = num_det_rows, num_det_channels
    else:
        warnings.warn("Picture mode unknown! Should be either portrait (0 or 180 deg rotation) or landscape (90 or 270 deg rotation). Automatically setting picture mode to portrait.")
        scan_rotate = 180

    # Radiograph horizontal & vertical flip
    if NSI_params[8] == "True":
        flipH = True
    else:
        flipH = False
    if NSI_params[9] == "True":
        flipV = True
    else:
        flipV = False

    # Detector rotation angle step (degree)
    angle_step = np.single(NSI_params[10])

    # Detector rotation direction
    if NSI_params[11] == "True":
        print("clockwise rotation.")
    else:
        print("counter-clockwise rotation.")
        # counter-clockwise rotation
        angle_step = -angle_step

    # Rotation axis
    r_a = NSI_params[12].split(' ')
    r_a = np.array([np.single(elem) for elem in r_a])
    # make sure rotation axis points down
    if r_a[1] > 0:
        r_a = -r_a

    # Detector normal vector
    r_n = NSI_params[13].split(' ')
    r_n = np.array([np.single(elem) for elem in r_n])

    # Detector horizontal vector
    r_h = NSI_params[14].split(' ')
    r_h = np.array([np.single(elem) for elem in r_h])
    crops = NSI_params[15].split(' ')
    crops = np.array([np.int32(elem) for elem in crops])
    max_crop = np.amax(crops)

    print("############ NSI geometry parameters ############")
    print("vector from origin to source = ", r_s, " [mm]")
    print("vector from origin to (0,0) detector pixel = ", r_r, " [mm]")
    print("Unit vector of rotation axis = ", r_a)
    print("Unit vector of normal = ", r_n)
    print("Unit vector of horizontal = ", r_h)
    print(f"Detector pixel pitch: (delta_det_row, delta_det_channel) = ({delta_det_row:.3f},{delta_det_channel:.3f}) [mm]")
    print(f"Detector size: (num_det_rows, num_det_channels) = ({num_det_rows},{num_det_channels})")
    print(f"Pixels to crop from the border of each view = {max_crop}")
    print("############ End NSI geometry parameters ############")
    ### END load NSI parameters from an nsipro file

    ### read blank scans and dark scans
    blank_scan = np.expand_dims(preprocess.read_scan_img(blank_scan_path), axis=0)
    if dark_scan_path is not None:
        dark_scan = np.expand_dims(preprocess.read_scan_img(dark_scan_path), axis=0)
    else:
        dark_scan = np.zeros(blank_scan.shape)

    ### read object scans
    if view_id_end is None:
        view_id_end = num_acquired_scans
    view_ids = np.arange(start=view_id_start, stop=view_id_end, step=subsample_view_factor, dtype=np.int32)
    print('Loading {} object scans from disk.'.format(len(view_ids)))
    obj_scan = preprocess.read_scan_dir(obj_scan_dir, view_ids)
    print('Scans loaded.')

    ### Load defective pixel information
    if defective_pixel_path is not None:
        tag_section_list = [['Defect', 'Defective Pixels']]
        defective_loc = _read_str_from_config(defective_pixel_path, tag_section_list)
        defective_pixel_array = np.array([defective_pixel_ind.split()[1::-1] for defective_pixel_ind in defective_loc ]).astype(int)
    else:
        defective_pixel_array = ()
    num_defective_pixels = len(defective_pixel_array)

    ### flip the scans according to flipH and flipV information from nsipro file
    if flipV:
        print("Flipping scans vertically")
        obj_scan = np.flip(obj_scan, axis=1)
        blank_scan = np.flip(blank_scan, axis=1)
        dark_scan = np.flip(dark_scan, axis=1)
        # adjust the defective pixel information: vertical flip
        if num_defective_pixels > 0:
            defective_pixel_array[:, 0] = blank_scan.shape[1] - defective_pixel_array[:, 0] - 1

    if flipH:
        print("Flipping scans horizontally")
        obj_scan = np.flip(obj_scan, axis=2)
        blank_scan = np.flip(blank_scan, axis=2)
        dark_scan = np.flip(dark_scan, axis=2)
        # adjust the defective pixel information: horizontal flip
        if num_defective_pixels > 0:
            defective_pixel_array[:, 1] = blank_scan.shape[2] - defective_pixel_array[:, 1] - 1

    ### rotate the scans according to scan_rotate param
    rot_count = scan_rotate // 90
    for n in range(rot_count):
        obj_scan = np.rot90(obj_scan, 1, axes=(2,1))
        blank_scan = np.rot90(blank_scan, 1, axes=(2,1))
        dark_scan = np.rot90(dark_scan, 1, axes=(2,1))
        # adjust the defective pixel information: rotation (clockwise)
        if num_defective_pixels > 0:
            defective_pixel_array = np.fliplr(defective_pixel_array)
            defective_pixel_array[:, 1] = blank_scan.shape[2] - defective_pixel_array[:, 1] - 1

    ### compute projection angles based on angle_step and view_ids
    angles = np.deg2rad(np.array([(view_idx*angle_step) % 360.0 for view_idx in view_ids]))

    nsi_params = {
        'r_a': r_a,
        'r_n': r_n,
        'r_h': r_h,
        'r_s': r_s,
        'r_r': r_r,
        'delta_det_channel': delta_det_channel,
        'delta_det_row': delta_det_row,
        'num_det_channels': num_det_channels,
        'num_det_rows': num_det_rows,
        'angles': angles,
        'max_crop': max_crop
    }

    return obj_scan, blank_scan, dark_scan, nsi_params, defective_pixel_array


def convert_nsi_to_mbirjax_params(nsi_params, downsample_factor=(1, 1), crop_pixels_sides=0, crop_pixels_top=0, crop_pixels_bottom=0):
    """
    Convert geometry parameters from nsi into mbirjax format, including modification to reflect crop and downsample.

    Args:
        nsi_params (dict):
        downsample_factor ((int, int), optional) - Down-sample factors along the detector rows and channels respectively.
            If scan size is not divisible by `downsample_factor`, the scans will be first truncated to a size that is divisible by `downsample_factor`.
        crop_pixels_sides (int, optional): The number of pixels to crop from each side of the sinogram. Defaults to 0.
        crop_pixels_top (int, optional): The number of pixels to crop from top of the sinogram. Defaults to 0.
        crop_pixels_bottom (int, optional): The number of pixels to crop from bottom of the sinogram. Defaults to 0.

    Returns:
        cone_beam_params (dict): Required parameters for the ConeBeamModel constructor.
        optional_params (dict): Additional ConeBeamModel parameters to be set using set_params().
    """
    # Get the nsi parameters and convert them
    r_a, r_n, r_h, r_s, r_r = itemgetter('r_a', 'r_n', 'r_h', 'r_s', 'r_r')(nsi_params)
    delta_det_channel, delta_det_row = itemgetter('delta_det_channel', 'delta_det_row')(nsi_params)
    num_det_channels, num_det_rows, angles = itemgetter('num_det_channels', 'num_det_rows', 'angles')(nsi_params)

    source_detector_dist, source_iso_dist, magnification, det_rotation = calc_source_detector_params(r_a, r_n, r_h, r_s, r_r)
    det_channel_offset, det_row_offset = calc_row_channel_params(r_a, r_n, r_h, r_s, r_r, delta_det_channel, delta_det_row, num_det_channels, num_det_rows, magnification)

    # Adjust detector size params w.r.t. cropping arguments
    num_det_rows = num_det_rows - (crop_pixels_top + crop_pixels_bottom)
    num_det_channels = num_det_channels - 2 * crop_pixels_sides

    # Adjust detector size and pixel pitch params w.r.t. downsampling arguments
    num_det_rows = num_det_rows // downsample_factor[0]
    num_det_channels = num_det_channels // downsample_factor[1]

    delta_det_row *= downsample_factor[0]
    delta_det_channel *= downsample_factor[1]

    # Set 1 ALU = delta_det_channel
    source_detector_dist /= delta_det_channel # mm to ALU
    source_iso_dist /= delta_det_channel # mm to ALU
    det_channel_offset /= delta_det_channel # mm to ALU
    det_row_offset /= delta_det_channel # mm to ALU
    delta_det_row /= delta_det_channel
    delta_det_channel = 1.0

    # Create a dictionary to store MBIR parameters
    num_views = len(angles)
    cone_beam_params = dict()
    cone_beam_params["sinogram_shape"] = (num_views, num_det_rows, num_det_channels)
    cone_beam_params["angles"] = angles
    cone_beam_params["source_detector_dist"] = source_detector_dist
    cone_beam_params["source_iso_dist"] = source_iso_dist

    optional_params = dict()
    optional_params["delta_det_channel"] = delta_det_channel
    optional_params["delta_det_row"] = delta_det_row
    optional_params['delta_voxel'] = delta_det_channel * (source_iso_dist/source_detector_dist)
    optional_params["det_channel_offset"] = det_channel_offset
    optional_params["det_row_offset"] = det_row_offset
    optional_params["det_rotation"] = det_rotation # tilt angle of rotation axis

    return cone_beam_params, optional_params


######## subroutines for parsing NSI metadata
def _parse_filenames_from_dataset_dir(dataset_dir):
    """ Given the path to an NSI dataset directory, automatically parse the paths to the following files and directories:
            - NSI config file (nsipro file),
            - geometry report (Geometry Report.rtf),
            - object scan directory (Radiographs/),
            - blank scan (Corrections/gain0.tif),
            - dark scan (Corrections/offset.tif),
            - defective pixel information (Corrections/defective_pixels.defect),
        If multiple files with the same patterns are found, then the user will be prompted to select the correct file.

    Args:
        dataset_dir (string): Path to the directory containing the NSI scans and metadata.
    Returns:
        6-element tuple containing:
            - config_file_path (string): Path to the NSI config file (nsipro file).
            - geom_report_path (string): Path to the geometry report file (Geometry Report.rtf)
            - obj_scan_dir (string): Path to the directory containing the object scan images (radiographs).
            - blank_scan_path (string): Path to the blank scan image.
            - dark_scan_path (string): Path to the dark scan image.
            - defective_pixel_path (string): Path to the file containing defective pixel information.
    """
    # NSI config file
    config_file_path_list = glob.glob(os.path.join(dataset_dir, "*.nsipro"))
    config_file_path = _prompt_user_choice("NSI config files", config_file_path_list)

    # geometry report
    geom_report_path_list = glob.glob(os.path.join(dataset_dir, "Geometry*.rtf"))
    geom_report_path = _prompt_user_choice("geometry report files", geom_report_path_list)

    # Radiograph directory
    obj_scan_dir_list = glob.glob(os.path.join(dataset_dir, "Radiographs*"))
    obj_scan_dir = _prompt_user_choice("radiograph directories", obj_scan_dir_list)

    # blank scan
    blank_scan_path_list = glob.glob(os.path.join(dataset_dir, "**/gain0.tif"))
    blank_scan_path = _prompt_user_choice("blank scans", blank_scan_path_list)

    # dark scan
    dark_scan_path_list = glob.glob(os.path.join(dataset_dir, "**/offset.tif"))
    dark_scan_path = _prompt_user_choice("dark scans", dark_scan_path_list)

    # defective pixel file
    defective_pixel_path_list = glob.glob(os.path.join(dataset_dir, "**/*.defect"))
    defective_pixel_path = _prompt_user_choice("defective pixel files", defective_pixel_path_list)

    return config_file_path, geom_report_path, obj_scan_dir, blank_scan_path, dark_scan_path, defective_pixel_path


def _prompt_user_choice(file_description, file_path_list):
    """ Given a list of candidate files, prompt the user to select the desired one.
        If only one candidate exists, the function will return the name of that file without any user prompts.
    """
    # file_path_list should contain at least one element
    assert(len(file_path_list) > 0), f"No {file_description} found!! Please make sure you provided a valid NSI scan path."

    # if only file_path_list contains only one file, then return it without user prompt.
    if len(file_path_list) == 1:
        return file_path_list[0]

    # file_path_list contains multiple files. Prompt the user to select the desired one.
    choice_min = 0
    choice_max = len(file_path_list)-1
    question = f"Multiple {file_description} detected. Please select the desired one from the following candidates "
    prompt = ":\n"
    for i in range(len(file_path_list)):
        prompt += f"\n    {i}: {file_path_list[i]}"
    prompt += f"\n[{choice_min}-{choice_max}]"
    while True:
        sys.stdout.write(question + prompt)
        try:
            choice = int(input())
            if choice in range(len(file_path_list)):
                return file_path_list[choice]
            else:
                sys.stdout.write(f"Please respond with a number between {choice_min} and {choice_max}.\n")
        except:
            sys.stdout.write(f"Please respond with a number between {choice_min} and {choice_max}.\n")
    return


def _read_detector_location_from_geom_report(geom_report_path):
    """ Give the path to "Geometry Report.rtf", returns the X and Y coordinates of the first row and first column of the detector.
        It is observed that the coordinates given in "Geometry Report.rtf" is more accurate than the coordinates given in the <reference> field in nsipro file.
        Specifically, this function parses the information of "Image center" from "Geometry Report.rtf".
        Example:
            - content in "Geometry Report.rtf": Image center    (95.707, 123.072) [mm]  / (3.768, 4.845) [in]
            - Returns: (95.707, 123.072)
    Args:
        geom_report_path (string): Path to "Geometry Report.rtf" file. This file contains more accurate information regarding the coordinates of the first detector row and column.
    Returns:
        (x_r, y_r): A tuple containing the X and Y coordinates of center of the first detector row and column.
    """
    rtf_file = open(geom_report_path, 'r')
    rtf_raw = rtf_file.read()
    rtf_file.close()
    # Find the image center in mm
    start_index = rtf_raw.find('Image center')
    end_index = start_index + rtf_raw[start_index:].find('[mm]')
    line = rtf_raw[start_index:end_index+1]
    data = re.findall(r"(\d+\.*\d*, \d+\.*\d*)", line)

    data = data[0].split(",")
    x_r = float(data[0])
    y_r = float(data[1])
    return x_r, y_r


def _read_str_from_config(filepath, tags_sections):
    """Returns strings about dataset information read from NSI configuration file.

    Args:
        filepath (string): Path to NSI configuration file. The filename extension is '.nsipro'.
        tags_sections (list[string,string]): Given tags and sections to locate the information we want to read.
    Returns:
        list[string], a list of strings have needed dataset information for reconstruction.

    """
    tag_strs = ['<' + tag + '>' for tag, section in tags_sections]
    section_starts = ['<' + section + '>' for tag, section in tags_sections]
    section_ends = ['</' + section + '>' for tag, section in tags_sections]
    NSI_params = []

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except IOError:
        print("Could not read file:", filepath)

    # TODO: Replace with more efficient code that doesn't use a nested loop
    for tag_str, section_start, section_end in zip(tag_strs, section_starts, section_ends):
        section_start_inds = [ind for ind, match in enumerate(lines) if section_start in match]
        section_end_inds = [ind for ind, match in enumerate(lines) if section_end in match]
        section_start_ind = section_start_inds[0]
        section_end_ind = section_end_inds[0]

        for line_ind in range(section_start_ind + 1, section_end_ind):
            line = lines[line_ind]
            if tag_str in line:
                tag_ind = line.find(tag_str, 1) + len(tag_str)
                if tag_ind == -1:
                    NSI_params.append("")
                else:
                    NSI_params.append(line[tag_ind:].strip('\n'))

    return NSI_params
######## END subroutines for parsing NSI metadata


######## subroutines for NSI-MBIR parameter conversion
def calc_det_rotation(r_a, r_n, r_h, r_v):
    """ Calculate the tilt angle between the rotation axis and the detector columns in unit of radians. User should call `preprocess.correct_det_rotation()` to rotate the sinogram images w.r.t. to the tilt angle.

    Args:
        r_a: 3D real-valued unit vector in direction of rotation axis pointing down.
        r_n: 3D real-valued unit vector perpendicular to the detector plan pointing from source to detector.
        r_h: 3D real-valued unit vector in direction parallel to detector rows pointing from left to right.
        r_v: 3D real-valued unit vector in direction parallel to detector columns pointing down.
    Returns:
        float number specifying the angle between the rotation axis and the detector columns in units of radians.
    """
    # project the rotation axis onto the detector plane
    r_a_p = preprocess.unit_vector(r_a - preprocess.project_vector_to_vector(r_a, r_n))
    # calculate angle between the projected rotation axis and the horizontal detector vector
    det_rotation = -np.arctan(np.dot(r_a_p, r_h)/np.dot(r_a_p, r_v))
    return det_rotation


def calc_source_detector_params(r_a, r_n, r_h, r_s, r_r):
    """ Calculate the MBIRJAX geometry parameters: source_detector_dist, magnification, and rotation axis tilt angle.
    Args:
        r_a (tuple): 3D real-valued unit vector in direction of rotation axis pointing down.
        r_n (tuple): 3D real-valued unit vector perpendicular to the detector plan pointing from source to detector.
        r_h (tuple): 3D real-valued unit vector in direction parallel to detector rows pointing from left to right.
        r_s (tuple): 3D real-valued vector from origin to the source location.
        r_r (tuple): 3D real-valued vector from origin to the center of pixel on first row and colum of detector.
    Returns:
        4-element tuple containing:
        - **source_detector_dist** (float): Distance between the X-ray source and the detector.
        - **source_iso_dist** (float): Distance between the X-ray source and the center of rotation.
        - **det_rotation (float)**: Angle between the rotation axis and the detector columns in units of radians.
        - **magnification** (float): Magnification of the cone-beam geometry defined as
            (source to detector distance)/(source to center-of-rotation distance).
    """
    r_n = preprocess.unit_vector(r_n)      # make sure r_n is normalized
    r_v = np.cross(r_n, r_h)    # r_v = r_n x r_h

    #### vector pointing from source to center of rotation along the source-detector line.
    r_s_r = preprocess.project_vector_to_vector(-r_s, r_n) # project -r_s to r_n

    #### vector pointing from source to detector along the source-detector line.
    r_s_d = preprocess.project_vector_to_vector(r_r-r_s, r_n)

    source_detector_dist = np.linalg.norm(r_s_d) # ||r_s_d||
    source_iso_dist = np.linalg.norm(r_s_r) # ||r_s_r||
    magnification = source_detector_dist/source_iso_dist
    det_rotation = calc_det_rotation(r_a, r_n, r_h, r_v) # rotation axis tilt angle
    return source_detector_dist, source_iso_dist, magnification, det_rotation


def calc_row_channel_params(r_a, r_n, r_h, r_s, r_r, delta_det_channel, delta_det_row, num_det_channels, num_det_rows, magnification):
    """ Calculate the MBIRJAX geometry parameters: det_channel_offset, det_row_offset.
    Args:
        r_a (tuple): 3D real-valued unit vector in direction of rotation axis pointing down.
        r_n (tuple): 3D real-valued unit vector perpendicular to the detector plan pointing from source to detector.
        r_h (tuple): 3D real-valued unit vector in direction parallel to detector rows pointing from left to right.
        r_s (tuple): 3D real-valued vector from origin to the source location.
        r_r (tuple): 3D real-valued vector from origin to the center of pixel on first row and colum of detector.
        delta_det_channel (float): spacing between detector columns.
        delta_det_row (float): spacing between detector rows.
        num_det_channels (int): Number of detector channels.
        num_det_rows (int): Number of detector rows.
        magnification (float): Magnification of the cone-beam geometry.
    Returns:
        2-element tuple containing:
        - **det_channel_offset** (float): Distance from center of detector to the source-detector line along a row.
        - **det_row_offset** (float): Distance from center of detector to the source-detector line along a column.
    """
    r_n = preprocess.unit_vector(r_n) # make sure r_n is normalized
    r_h = preprocess.unit_vector(r_h) # make sure r_h is normalized
    r_v = np.cross(r_n, r_h) # r_v = r_n x r_h

    # vector pointing from center of detector to the first row and column of detector along detector columns.
    c_v = -(num_det_rows-1)/2*delta_det_row*r_v
    # vector pointing from center of detector to the first row and column of detector along detector rows.
    c_h = -(num_det_channels-1)/2*delta_det_channel*r_h
    # vector pointing from source to first row and column of detector.
    r_s_r = r_r - r_s
    # vector pointing from source-detector line to center of detector.
    r_delta = r_s_r - preprocess.project_vector_to_vector(r_s_r, r_n) - c_v - c_h
    # detector row and channel offsets
    det_channel_offset = -np.dot(r_delta, r_h)
    det_row_offset = -np.dot(r_delta, r_v)
    # rotation offset
    delta_source = r_s - preprocess.project_vector_to_vector(r_s, r_n)
    delta_rot = delta_source - preprocess.project_vector_to_vector(delta_source, r_a)# rotation offset vector (perpendicular to rotation axis)
    rotation_offset = np.dot(delta_rot, np.cross(r_n, r_a))
    det_channel_offset += rotation_offset*magnification
    return det_channel_offset, det_row_offset

######## END subroutines for NSI-MBIR parameter conversion
