import os
import re
from glob import glob
import numpy as np
from PIL import Image
import warnings
import math
import scipy
import striprtf.striprtf as striprtf

######## API functions for MBIRJAX users
def NSI_load_scans_and_params_jax(config_file_path, geom_report_path, 
                                  obj_scan_path, blank_scan_path, dark_scan_path,
                                  defective_pixel_path,
                                  downsample_factor=[1, 1], crop_region=[(0, 1), (0, 1)],
                                  view_id_start=0, view_angle_start=0.,
                                  view_id_end=None, subsample_view_factor=1):
    """ Load the object scan, blank scan, dark scan, view angles, defective pixel information, and geometry parameters from an NSI dataset directory.

    The scan images will be (optionally) cropped and downsampled.

    A subset of the views may be selected based on user input. In that case, the object scan images and view angles corresponding to the subset of the views will be returned.

    This function is specific to NSI datasets.

    **Arguments specific to file paths**:

        - config_file_path (string): Path to NSI configuration file. The filename extension is '.nsipro'.
        - geom_report_path (string): Path to "Geometry Report.rtf" file. This file contains more accurate information regarding the coordinates of the first detector row and column.
        - obj_scan_path (string): Path to an NSI radiograph directory.
        - blank_scan_path (string): Path to a blank scan image, e.g. 'dataset_path/Corrections/gain0.tif'
        - dark_scan_path (string): Path to a dark scan image, e.g. 'dataset_path/Corrections/offset.tif'
        - defective_pixel_path (string): Path to the file containing defective pixel information, e.g. 'dataset_path/Corrections/defective_pixels.defect'

    **Arguments specific to radiograph downsampling and cropping**:

        - downsample_factor ([int, int]): [Default=[1,1]] Down-sample factors along the detector rows and channels respectively. By default no downsampling will be performed.

            In case where the scan size is not divisible by `downsample_factor`, the scans will be first truncated to a size that is divisible by `downsample_factor`, and then downsampled.

        - crop_region ([(float, float),(float, float)] or [float, float, float, float]): [Default=[(0, 1), (0, 1)]]. Two fractional points [(row0, row1), (col0, col1)] defining the bounding box that crops the scans, where 0<=row0<=row1<=1 and 0<=col0<=col1<=1. By default no cropping will be performed.

            row0 and row1 defines the cropping factors along the detector rows. col0 and col1 defines the cropping factors along the detector channels. ::

            :       (0,0)--------------------------(0,1)
            :         |  (row0,col0)---------------+     |
            :         |     |                  |     |
            :         |     | (Cropped Region) |     |
            :         |     |                  |     |
            :         |     +---------------(row1,col1)  |
            :       (1,0)--------------------------(1,1)

            For example, ``crop_region=[(0.25,0.75), (0,1)]`` will crop out the middle half of the scan image along the vertical direction.

    **Arguments specific to view subsampling**:

        - view_id_start (int): [Default=0] view id corresponding to the first view.
        - view_angle_start (float): [Default=0.0] view angle in radian corresponding to the first view.
        - view_id_end (int): [Default=None] view id corresponding to the last view. If None, this will be equal to the total number of object scan images in ``obj_scan_path``.
        - subsample_view_factor (int): [Default=1]: view subsample factor. By default no view subsampling will be performed.

            For example, with ``subsample_view_factor=2``, every other view will be loaded.

    Returns:
        6-element tuple containing:

        - **obj_scan** (*ndarray, float*): 3D object scan with shape (num_views, num_det_rows, num_det_channels)

        - **blank_scan** (*ndarray, float*): 3D blank scan with shape (1, num_det_rows, num_det_channels)

        - **dark_scan** (*ndarray, float*): 3D dark scan with shape (1, num_det_rows, num_det_channels)

        - **angles** (*ndarray, double*): 1D view angles array in radians in the interval :math:`[0,2\pi)`.

        - **geo_params**: MBIRJAX format geometric parameters containing the following entries:
            
            - sinogram_shape (tuple): Shape of the sinogram as a tuple in the form (views, rows, channels).
            - source_detector_dist (float): Distance between the X-ray source and the detector in units of :math:`ALU`.
            - source_iso_dist (float): Distance between the X-ray source and the center of rotation in units of :math:`ALU`.
            - delta_det_channel (float): Detector channel spacing in :math:`ALU`.
            - delta_det_row (float): Detector row spacing in :math:`ALU`.
            - delta_voxel (float): Spacing between voxels in ALU.
            - det_channel_offset (float): Distance = (detector iso channel) - (center of detector channels) in ALU.
            - det_row_offset (float): Distance = (detector iso row) - (center of detector rows) in ALU.
            - det_rotation: Angle in radians between the projection of the object rotation axis and the detector vertical axis, where positive describes a clockwise rotation of the detector as seen from the source.
        - **defective_pixel_list** (list(tuple)): A list of tuples containing indices of invalid sinogram pixels, with the format (detector_row_idx, detector_channel_idx).
    """
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
                        ['horizontal', 'Result']                        # unit vector in direction of detector rows
                       ]
    assert(os. path. isfile(config_file_path)), f'Error! NSI config file does not exist. Please check whether {config_file_path} is a valid file.'
    NSI_params = _NSI_read_str_from_config(config_file_path, tag_section_list)

    # vector from origin to source
    r_s = NSI_params[0].split(' ')
    r_s = np.array([np.single(elem) for elem in r_s])
    
    # vector from origin to reference, where reference is the center of first row and column of the detector
    r_r = NSI_params[1].split(' ')
    r_r = np.array([np.single(elem) for elem in r_r])

    # correct the coordinate of (0,0) detector pixel based on "Geometry Report.rtf"
    x_r, y_r = _NSI_read_detector_location_from_geom_report(geom_report_path)
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

    print("############ NSI geometry parameters ############")
    print("vector from origin to source = ", r_s, " [mm]")
    print("vector from origin to (0,0) detector pixel = ", r_r, " [mm]")
    print("Unit vector of rotation axis = ", r_a)
    print("Unit vector of normal = ", r_n)
    print("Unit vector of horizontal = ", r_h)
    print(f"Detector pixel pitch: (delta_det_row, delta_det_channel) = ({delta_det_row:.3f},{delta_det_channel:.3f}) [mm]")
    print(f"Detector size: (num_det_rows, num_det_channels) = ({num_det_rows},{num_det_channels})")
    print("############ End NSI geometry parameters ############")
    ### END load NSI parameters from an nsipro file
    
    
    ### Convert NSI geometry parameters to MBIR parameters
    source_detector_dist, source_iso_dist, magnification, det_rotation = calc_source_detector_params(r_a, r_n, r_h, r_s, r_r)
    
    det_channel_offset, det_row_offset = calc_row_channel_params(r_a, r_n, r_h, r_s, r_r, delta_det_channel, delta_det_row, num_det_channels, num_det_rows, magnification)
    
    ### END Convert NSI geometry parameters to MBIR parameters
    
    ### Adjust geometry NSI_params according to crop_region and downsample_factor
    if isinstance(crop_region[0], (list, tuple)):
        (row0, row1), (col0, col1) = crop_region
    else:
        row0, row1, col0, col1 = crop_region

    ### Adjust detector size and pixel pitch params w.r.t. downsampling arguments
    num_det_rows = num_det_rows // downsample_factor[0]
    num_det_channels = num_det_channels // downsample_factor[1]

    delta_det_row *= downsample_factor[0]
    delta_det_channel *= downsample_factor[1]

    ### Adjust detector size params w.r.t. cropping arguments
    num_det_rows_shift0 = np.round(num_det_rows * row0)
    num_det_rows_shift1 = np.round(num_det_rows * (1 - row1))
    num_det_rows = num_det_rows - (num_det_rows_shift0 + num_det_rows_shift1)

    num_det_channels_shift0 = np.round(num_det_channels * col0)
    num_det_channels_shift1 = np.round(num_det_channels * (1 - col1))
    num_det_channels = num_det_channels - (num_det_channels_shift0 + num_det_channels_shift1)

    ### read blank scans and dark scans
    blank_scan = np.expand_dims(_read_scan_img(blank_scan_path), axis=0)
    if dark_scan_path is not None:
        dark_scan = np.expand_dims(_read_scan_img(dark_scan_path), axis=0)
    else:
        dark_scan = np.zeros(blank_scan.shape)

    if view_id_end is None:
        view_id_end = num_acquired_scans
    view_ids = list(range(view_id_start, view_id_end, subsample_view_factor))
    obj_scan = _read_scan_dir(obj_scan_path, view_ids)

    ### Load defective pixel information
    if defective_pixel_path is not None:
        tag_section_list = [['Defect', 'Defective Pixels']]
        defective_loc = _NSI_read_str_from_config(defective_pixel_path, tag_section_list)
        defective_pixel_list = np.array([defective_pixel_ind.split()[1::-1] for defective_pixel_ind in defective_loc ]).astype(int)
        defective_pixel_list = list(map(tuple, defective_pixel_list))
    else:
        defective_pixel_list = None


    ### flip the scans according to flipH and flipV information from nsipro file
    if flipV:
        print("Flip scans vertically!")
        obj_scan = np.flip(obj_scan, axis=1)
        blank_scan = np.flip(blank_scan, axis=1)
        dark_scan = np.flip(dark_scan, axis=1)
        # adjust the defective pixel information: vertical flip
        if defective_pixel_list is not None:
            for i in range(len(defective_pixel_list)):
                (r,c) = defective_pixel_list[i]
                defective_pixel_list[i] = (blank_scan.shape[1]-r-1, c)
    if flipH:
        print("Flip scans horizontally!")
        obj_scan = np.flip(obj_scan, axis=2)
        blank_scan = np.flip(blank_scan, axis=2)
        dark_scan = np.flip(dark_scan, axis=2)
        # adjust the defective pixel information: horizontal flip
        if defective_pixel_list is not None:
            for i in range(len(defective_pixel_list)):
                (r,c) = defective_pixel_list[i]
                defective_pixel_list[i] = (r, blank_scan.shape[2]-c-1)

    ### rotate the scans according to scan_rotate param
    rot_count = scan_rotate // 90
    for n in range(rot_count):
        obj_scan = np.rot90(obj_scan, 1, axes=(2,1))
        blank_scan = np.rot90(blank_scan, 1, axes=(2,1))
        dark_scan = np.rot90(dark_scan, 1, axes=(2,1))
        # adjust the defective pixel information: rotation (clockwise)
        if defective_pixel_list is not None:
            for i in range(len(defective_pixel_list)):
                (r,c) = defective_pixel_list[i]
                defective_pixel_list[i] = (c, blank_scan.shape[2]-r-1)

    ### crop the scans based on input params
    obj_scan, blank_scan, dark_scan, defective_pixel_list = _crop_scans(obj_scan, blank_scan, dark_scan,
                                                                        crop_region=crop_region,
                                                                        defective_pixel_list=defective_pixel_list)

    ### downsample the scans with block-averaging
    if downsample_factor[0]*downsample_factor[1] > 1:
        obj_scan, blank_scan, dark_scan, defective_pixel_list = _downsample_scans(obj_scan, blank_scan, dark_scan,
                                                                                  downsample_factor=downsample_factor,
                                                                                  defective_pixel_list=defective_pixel_list)

    ### compute projection angles based on angle_step and rotation direction
    view_angle_start_deg = np.rad2deg(view_angle_start)
    angle_step *= subsample_view_factor
    angles = np.deg2rad(np.array([(view_angle_start_deg+n*angle_step) % 360.0 for n in range(len(view_ids))]))

    ### Set 1 ALU = delta_det_channel
    source_detector_dist /= delta_det_channel # mm to ALU
    source_iso_dist /= delta_det_channel # mm to ALU 
    det_channel_offset /= delta_det_channel # mm to ALU
    det_row_offset /= delta_det_row # mm to ALU
    delta_det_channel = 1.0
    delta_det_row = 1.0
    
    # Create a dictionary to store MBIR parameters 
    num_views = len(angles)
    geo_params = dict()
    geo_params["sinogram_shape"] = (num_views, num_det_rows, num_det_channels)
    geo_params["source_detector_dist"] = source_detector_dist
    geo_params["source_iso_dist"] = source_iso_dist
    geo_params["delta_det_channel"] = delta_det_channel
    geo_params["delta_det_row"] = delta_det_row
    geo_params['delta_voxel'] = delta_det_channel * (source_iso_dist/source_detector_dist)
    geo_params["det_channel_offset"] = det_channel_offset
    geo_params["det_row_offset"] = det_row_offset
    geo_params["det_rotation"] = det_rotation # tilt angle of rotation axis

    return obj_scan, blank_scan, dark_scan, angles, geo_params, defective_pixel_list


def transmission_CT_compute_sino(obj_scan, blank_scan, dark_scan, defective_pixel_list=None):
    """Given a set of object scans, blank scan, and dark scan, compute the sinogram data with the steps below:

        1. ``sino = -numpy.log((obj_scan-dark_scan) / (blank_scan-dark_scan))``.
        2. Identify the invalid sinogram entries. The invalid sinogram entries are indentified as the union of defective pixel entries (speicified by ``defective_pixel_list``) and sinogram entries with values of inf or Nan.

    Args:
        obj_scan (ndarray, float): 3D object scan with shape (num_views, num_det_rows, num_det_channels).
        blank_scan (ndarray, float): [Default=None] 3D blank scan with shape (num_blank_scans, num_det_rows, num_det_channels). When num_blank_scans>1, the pixel-wise mean will be used as the blank scan.
        dark_scan (ndarray, float): [Default=None] 3D dark scan with shape (num_dark_scans, num_det_rows, num_det_channels). When num_dark_scans>1, the pixel-wise mean will be used as the dark scan.
        defective_pixel_list (optional, list(tuple)): A list of tuples containing indices of invalid sinogram pixels, with the format (view_idx, row_idx, channel_idx) or (detector_row_idx, detector_channel_idx).
            If None, then the defective pixels will be identified as sino entries with inf or Nan values.
    Returns:
        2-element tuple containing:
        - **sino** (*ndarray, float*): Sinogram data with shape (num_views, num_det_rows, num_det_channels).
        - **defective_pixel_list** (list(tuple)): A list of tuples containing indices of invalid sinogram pixels, with the format (view_idx, row_idx, channel_idx) or (detector_row_idx, detector_channel_idx).

    """
    # take average of multiple blank/dark scans, and expand the dimension to be the same as obj_scan.
    blank_scan = 0 * obj_scan + np.mean(blank_scan, axis=0, keepdims=True)
    dark_scan = 0 * obj_scan + np.mean(dark_scan, axis=0, keepdims=True)

    obj_scan = obj_scan - dark_scan
    blank_scan = blank_scan - dark_scan
    sino = -np.log(obj_scan / blank_scan)

    # set the sino pixels corresponding to the provided defective list to 0.0
    if defective_pixel_list is None:
        defective_pixel_list = []
    else:    # if provided list is not None
        for defective_pixel_idx in defective_pixel_list:
            if len(defective_pixel_idx) == 2:
                (r,c) = defective_pixel_idx
                sino[:,r,c] = 0.0
            elif len(defective_pixel_idx) == 3:
                (v,r,c) = defective_pixel_idx
                sino[v,r,c] = 0.0
            else:
                raise Exception("transmission_CT_compute_sino: index information in defective_pixel_list cannot be parsed.")

    # set NaN sino pixels to 0.0
    nan_pixel_list = list(map(tuple, np.argwhere(np.isnan(sino)) ))
    for (v,r,c) in nan_pixel_list:
        sino[v,r,c] = 0.0

    # set Inf sino pixels to 0.0
    inf_pixel_list = list(map(tuple, np.argwhere(np.isinf(sino)) ))
    for (v,r,c) in inf_pixel_list:
        sino[v,r,c] = 0.0

    # defective_pixel_list = union{input_defective_pixel_list, nan_pixel_list, inf_pixel_list}
    defective_pixel_list = list(set().union(defective_pixel_list,nan_pixel_list,inf_pixel_list))

    return sino, defective_pixel_list

def interpolate_defective_pixels(sino, defective_pixel_list):
    """ This function interpolates defective sinogram entries with the mean of neighboring pixels.
     
    Args:
        sino (ndarray, float): Sinogram data with 3D shape (num_views, num_det_rows, num_det_channels).
        defective_pixel_list (list(tuple)): A list of tuples containing indices of invalid sinogram pixels, with the format (detector_row_idx, detector_channel_idx) or (view_idx, detector_row_idx, detector_channel_idx).
    Returns:    
        2-element tuple containing:
        - **sino** (*ndarray, float*): Corrected sinogram data with shape (num_views, num_det_rows, num_det_channels).
        - **defective_pixel_list** (*list(tuple)*): Updated defective_pixel_list with the format (detector_row_idx, detector_channel_idx) or (view_idx, detector_row_idx, detector_channel_idx). 
    """
    defective_pixel_list_new = []
    num_views, num_det_rows, num_det_channels = sino.shape
    weights = np.ones((num_views, num_det_rows, num_det_channels))

    for defective_pixel_idx in defective_pixel_list:
        if len(defective_pixel_idx) == 2:
            (r,c) = defective_pixel_idx
            weights[:,r,c] = 0.0
        elif len(defective_pixel_idx) == 3:
            (v,r,c) = defective_pixel_idx
            weights[v,r,c] = 0.0
        else:
            raise Exception("replace_defective_with_mean: index information in defective_pixel_list cannot be parsed.")

    for defective_pixel_idx in defective_pixel_list:
        if len(defective_pixel_idx) == 2:
            v_list = list(range(num_views))
            (r,c) = defective_pixel_idx
        elif len(defective_pixel_idx) == 3:
            (v,r,c) = defective_pixel_idx
            v_list = [v,]

        r_min, r_max = max(r-1, 0), min(r+2, num_det_rows)
        c_min, c_max = max(c-1, 0), min(c+2, num_det_channels)
        for v in v_list:
            # Perform interpolation when there are non-defective pixels in the neighborhood
            if np.sum(weights[v,r_min:r_max,c_min:c_max]) > 0:
                sino[v,r,c] = np.average(sino[v,r_min:r_max,c_min:c_max],
                                         weights=weights[v,r_min:r_max,c_min:c_max])
            # Corner case: all the neighboring pixels are defective
            else:
                print(f"Unable to correct sino entry ({v},{r},{c})! All neighborhood values are defective!")
                defective_pixel_list_new.append((v,r,c)) 
    return sino, defective_pixel_list_new

def correct_det_rotation(sino, weights=None, det_rotation=0.0):
    """ Correct the sinogram data (and sinogram weights if provided) according to the rotation axis tilt.

    Args:
        sino (float, ndarray): Sinogram data with 3D shape (num_views, num_det_rows, num_det_channels).
        weights (float, ndarray): Sinogram weights, with the same array shape as ``sino``.
        det_rotation (optional, float): tilt angle between the rotation axis and the detector columns in unit of radians.
    
    Returns:
        - A numpy array containing the corrected sinogram data if weights is None. 
        - A tuple (sino, weights) if weights is not None
    """
    sino = scipy.ndimage.rotate(sino, np.rad2deg(det_rotation), axes=(1,2), reshape=False, order=5)
    # weights not provided
    if weights is None:
        return sino
    # weights provided
    print("correct_det_rotation: weights provided by the user. Please note that zero weight entries might become non-zero after tilt angle correction.") 
    weights = scipy.ndimage.rotate(weights, np.rad2deg(det_rotation), axes=(1,2), reshape=False, order=5)
    return sino, weights

def calc_background_offset(sino, option=0, edge_width=9):
    """ Given a sinogram, automatically calculate the background offset based on the selected option. Available options are:

        **Option 0**: Calculate the background offset using edge_width pixels along the upper, left, and right edges of a median sinogram view.

    Args:
        sino (float, ndarray): Sinogram data with 3D shape (num_views, num_det_rows, num_det_channels).
        option (int, optional): [Default=0] Option of algorithm used to calculate the background offset.
        edge_width(int, optional): [Default=9] Width of the edge regions in pixels. It must be an odd integer >= 3.
    Returns:
        offset (float): Background offset value.
    """

    # Check validity of edge_width value
    assert(isinstance(edge_width, int)), "edge_width must be an integer!"
    if (edge_width % 2 == 0):
        edge_width = edge_width+1
        warnings.warn(f"edge_width of background regions should be an odd number! Setting edge_width to {edge_width}.")

    if (edge_width < 3):
        warnings.warn("edge_width of background regions should be >= 3! Setting edge_width to 3.")
        edge_width = 3

    _, _, num_det_channels = sino.shape

    # calculate mean sinogram
    sino_median=np.median(sino, axis=0)

    # offset value of the top edge region.
    # Calculated as median([median value of each horizontal line in top edge region])
    median_top = np.median(np.median(sino_median[:edge_width], axis=1))

    # offset value of the left edge region.
    # Calculated as median([median value of each vertical line in left edge region])
    median_left = np.median(np.median(sino_median[:, :edge_width], axis=0))

    # offset value of the right edge region.
    # Calculated as median([median value of each vertical line in right edge region])
    median_right = np.median(np.median(sino_median[:, num_det_channels-edge_width:], axis=0))

    # offset = median of three offset values from top, left, right edge regions.
    offset = np.median([median_top, median_left, median_right])
    return offset

######## subroutines for loading scan images
def _read_scan_img(img_path):
    """Reads a single scan image from an image path. This function is a subroutine to the function `_read_scan_dir`.

    Args:
        img_path (string): Path object or file object pointing to an image. 
            The image type must be compatible with `PIL.Image.open()`. See `https://pillow.readthedocs.io/en/stable/reference/Image.html` for more details.
    Returns:
        ndarray (float): 2D numpy array. A single scan image.
    """

    img = np.asarray(Image.open(img_path))

    if np.issubdtype(img.dtype, np.integer):
        # make float and normalize integer types
        maxval = np.iinfo(img.dtype).max
        img = img.astype(np.float32) / maxval

    return img.astype(np.float32)


def _read_scan_dir(scan_dir, view_ids=[]):
    """Reads a stack of scan images from a directory. This function is a subroutine to `NSI_load_scans_and_params`.

    Args:
        scan_dir (string): Path to a ConeBeam Scan directory. 
            Example: "<absolute_path_to_dataset>/Radiographs"
        view_ids (list[int]): List of view indices to specify which scans to read.
    Returns:
        ndarray (float): 3D numpy array, (num_views, num_det_rows, num_det_channels). A stack of scan images.
    """

    if view_ids == []:
        warnings.warn("view_ids should not be empty.")

    img_path_list = sorted(glob(os.path.join(scan_dir, '*')))
    img_path_list = [img_path_list[idx] for idx in view_ids]
    img_list = [_read_scan_img(img_path) for img_path in img_path_list]

    # return shape = num_views x num_det_rows x num_det_channels
    return np.stack(img_list, axis=0)
######## END subroutines for loading scan images

######## subroutines for image cropping and down-sampling
def _downsample_scans(obj_scan, blank_scan, dark_scan,
                      downsample_factor,
                      defective_pixel_list=None):
    """Performs Down-sampling to the scan images in the detector plane.

    Args:
        obj_scan (float): A stack of sinograms. 3D numpy array, (num_views, num_det_rows, num_det_channels).
        blank_scan (float): A blank scan. 2D numpy array, (num_det_rows, num_det_channels).
        dark_scan (float): A dark scan. 3D numpy array, (num_det_rows, num_det_channels).
        downsample_factor ([int, int]): Default=[1,1]] Two numbers to define down-sample factor.
    Returns:
        Downsampled scans
        - **obj_scan** (*ndarray, float*): A stack of sinograms. 3D numpy array, (num_views, num_det_rows, num_det_channels).
        - **blank_scan** (*ndarray, float*): A blank scan. 3D numpy array, (num_det_rows, num_det_channels).
        - **dark_scan** (*ndarray, float*): A dark scan. 3D numpy array, (num_det_rows, num_det_channels).
    """

    assert len(downsample_factor) == 2, 'factor({}) needs to be of len 2'.format(downsample_factor)
    assert (downsample_factor[0]>=1 and downsample_factor[1]>=1), 'factor({}) along each dimension should be greater or equal to 1'.format(downsample_factor)

    good_pixel_mask = np.ones((blank_scan.shape[1], blank_scan.shape[2]), dtype=int)
    if defective_pixel_list is not None:
        for (r,c) in defective_pixel_list:
            good_pixel_mask[r,c] = 0

    # crop the scan if the size is not divisible by downsample_factor.
    new_size1 = downsample_factor[0] * (obj_scan.shape[1] // downsample_factor[0])
    new_size2 = downsample_factor[1] * (obj_scan.shape[2] // downsample_factor[1])

    obj_scan = obj_scan[:, 0:new_size1, 0:new_size2]
    blank_scan = blank_scan[:, 0:new_size1, 0:new_size2]
    dark_scan = dark_scan[:, 0:new_size1, 0:new_size2]
    good_pixel_mask = good_pixel_mask[0:new_size1, 0:new_size2]

    ### Compute block sum of the high res scan images. Defective pixels are excluded.
    # filter out defective pixels
    good_pixel_mask = good_pixel_mask.reshape(good_pixel_mask.shape[0] // downsample_factor[0], downsample_factor[0],
                                              good_pixel_mask.shape[1] // downsample_factor[1], downsample_factor[1])
    obj_scan = obj_scan.reshape(obj_scan.shape[0],
                                obj_scan.shape[1] // downsample_factor[0], downsample_factor[0],
                                obj_scan.shape[2] // downsample_factor[1], downsample_factor[1]) * good_pixel_mask

    blank_scan = blank_scan.reshape(blank_scan.shape[0],
                                    blank_scan.shape[1] // downsample_factor[0], downsample_factor[0],
                                    blank_scan.shape[2] // downsample_factor[1], downsample_factor[1]) * good_pixel_mask
    dark_scan = dark_scan.reshape(dark_scan.shape[0],
                                  dark_scan.shape[1] // downsample_factor[0], downsample_factor[0],
                                  dark_scan.shape[2] // downsample_factor[1], downsample_factor[1]) * good_pixel_mask

    # compute block sum
    obj_scan = obj_scan.sum((2,4))
    blank_scan = blank_scan.sum((2, 4))
    dark_scan = dark_scan.sum((2, 4))
    # number of good pixels in each down-sampling block
    good_pixel_count = good_pixel_mask.sum((1,3))

    # new defective pixel list = {indices of pixels where the downsampling block contains all bad pixels}
    defective_pixel_list = np.argwhere(good_pixel_count < 1)

    # compute block averaging by dividing block sum with number of good pixels in the block
    obj_scan = obj_scan / good_pixel_count
    blank_scan = blank_scan / good_pixel_count
    dark_scan = dark_scan / good_pixel_count

    return obj_scan, blank_scan, dark_scan, defective_pixel_list


def _crop_scans(obj_scan, blank_scan, dark_scan,
                crop_region=[(0, 1), (0, 1)],
                defective_pixel_list=None):
    """Crop obj_scan, blank_scan, and dark_scan images by decimal factors, and update defective_pixel_list accordingly. 
    Args:
        obj_scan (float): A stack of sinograms. 3D numpy array, (num_views, num_det_rows, num_det_channels).
        blank_scan (float) : A blank scan. 3D numpy array, (1, num_det_rows, num_det_channels).
        dark_scan (float): A dark scan. 3D numpy array, (1, num_det_rows, num_det_channels).
        crop_region ([(float, float),(float, float)] or [float, float, float, float]):
            [Default=[(0, 1), (0, 1)]] Two points to define the bounding box. Sequence of [(row0, row1), (col0, col1)] or
            [row0, row1, col0, col1], where 0<=row0 <= row1<=1 and 0<=col0 <= col1<=1.
        
            The scan images will be cropped using the following algorithm:
                obj_scan <- obj_scan[:,Nr_lo:Nr_hi, Nc_lo:Nc_hi], where 
                    - Nr_lo = round(row0 * obj_scan.shape[1])
                    - Nr_hi = round(row1 * obj_scan.shape[1])
                    - Nc_lo = round(col0 * obj_scan.shape[2])
                    - Nc_hi = round(col1 * obj_scan.shape[2])

    Returns:
        Cropped scans
        - **obj_scan** (*ndarray, float*): A stack of sinograms. 3D numpy array, (num_views, num_det_rows, num_det_channels).
        - **blank_scan** (*ndarray, float*): A blank scan. 3D numpy array, (1, num_det_rows, num_det_channels).
        - **dark_scan** (*ndarray, float*): A dark scan. 3D numpy array, (1, num_det_rows, num_det_channels).
    """
    if isinstance(crop_region[0], (list, tuple)):
        (row0, row1), (col0, col1) = crop_region
    else:
        row0, row1, col0, col1 = crop_region

    assert 0 <= row0 <= row1 <= 1 and 0 <= col0 <= col1 <= 1, 'crop_region should be sequence of [(row0, row1), (col0, col1)] ' \
                                                      'or [row0, row1, col0, col1], where 1>=row1 >= row0>=0 and 1>=col1 >= col0>=0.'
    assert math.isclose(col0, 1 - col1), 'horizontal crop limits must be symmetric'

    Nr_lo = round(row0 * obj_scan.shape[1])
    Nc_lo = round(col0 * obj_scan.shape[2])

    Nr_hi = round(row1 * obj_scan.shape[1])
    Nc_hi = round(col1 * obj_scan.shape[2])

    obj_scan = obj_scan[:, Nr_lo:Nr_hi, Nc_lo:Nc_hi]
    blank_scan = blank_scan[:, Nr_lo:Nr_hi, Nc_lo:Nc_hi]
    dark_scan = dark_scan[:, Nr_lo:Nr_hi, Nc_lo:Nc_hi]

    # adjust the defective pixel information: any down-sampling block containing a defective pixel is also defective
    i = 0
    while i < len(defective_pixel_list):
        (r,c) = defective_pixel_list[i]
        (r_new, c_new) = (r-Nr_lo, c-Nc_lo)
        # delete the index tuple if it falls outside the cropped region
        if (r_new<0 or r_new>=obj_scan.shape[1] or c_new<0 or c_new>=obj_scan.shape[2]):
            del defective_pixel_list[i]
        else:
            i+=1
    return obj_scan, blank_scan, dark_scan, defective_pixel_list
######## END subroutines for image cropping and down-sampling

######## subroutines for parsing NSI metadata
def _NSI_read_detector_location_from_geom_report(geom_report_path):
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
    # convert rft file content to plain text.
    rtf_converted = striprtf.rtf_to_text(rtf_raw).split("\n")
    for line in rtf_converted:
        if "Image center" in line:
            # read the two floating numbers immediately following the keyword "Image center". 
            # This is the X and Y coordinates of (0,0) detector pixel in units of mm.
            data = re.findall(r"(\d+\.*\d*, \d+\.*\d*)", line)
            break
    data = data[0].split(",")
    x_r = float(data[0])
    y_r = float(data[1])
    return x_r, y_r

def _NSI_read_str_from_config(filepath, tags_sections):
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
def unit_vector(v):
    """ Normalize v. Returns v/||v|| """
    return v / np.linalg.norm(v)

def project_vector_to_vector(u1, u2):
    """ Projects the vector u1 onto the vector u2. Returns the vector <u1|u2>.
    """
    u2 = unit_vector(u2)
    u1_proj = np.dot(u1, u2)*u2
    return u1_proj

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
    r_a_p = unit_vector(r_a - project_vector_to_vector(r_a, r_n))
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
        3-element tuple containing:
        - **source_detector_dist** (float): Distance between the X-ray source and the detector. 
        - **magnification** (float): Magnification of the cone-beam geometry defined as
            (source to detector distance)/(source to center-of-rotation distance).
        - **det_rotation (float)**: Angle between the rotation axis and the detector columns in units of radians.
    """
    r_n = unit_vector(r_n)      # make sure r_n is normalized
    r_v = np.cross(r_n, r_h)    # r_v = r_n x r_h

    #### vector pointing from source to center of rotation along the source-detector line.
    r_s_r = project_vector_to_vector(-r_s, r_n) # project -r_s to r_n 
    
    #### vector pointing from source to detector along the source-detector line.
    r_s_d = project_vector_to_vector(r_r-r_s, r_n)
    
    source_detector_dist = np.linalg.norm(r_s_d) # ||r_s_d||
    source_iso_dist = np.linalg.norm(r_s_r) # ||r_s_r||
    magnification = source_detector_dist/source_iso_dist 
    det_rotation = calc_det_rotation(r_a, r_n, r_h, r_v) # rotation axis tilt angle
    return source_detector_dist, source_iso_dist, magnification, det_rotation

def calc_row_channel_params(r_a, r_n, r_h, r_s, r_r, delta_det_channel, delta_det_row, num_det_channels, num_det_rows, magnification):
    """ Calculate the MBIRJAX geometry parameters: det_channel_offset, det_row_offset, rotation_offset. 
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
    Returns:
        3-element tuple containing:
        - **det_channel_offset** (float): Distance from center of detector to the source-detector line along a row. 
        - **det_row_offset** (float): Distance from center of detector to the source-detector line along a column. 
        - **rotation_offset** (float): Distance from source-detector line to axis of rotation.
    """
    r_n = unit_vector(r_n) # make sure r_n is normalized
    r_h = unit_vector(r_h) # make sure r_h is normalized
    r_v = np.cross(r_n, r_h) # r_v = r_n x r_h
    
    # vector pointing from center of detector to the first row and column of detector along detector columns.
    c_v = -(num_det_rows-1)/2*delta_det_row*r_v 
    # vector pointing from center of detector to the first row and column of detector along detector rows.
    c_h = -(num_det_channels-1)/2*delta_det_channel*r_h
    # vector pointing from source to first row and column of detector.
    r_s_r = r_r - r_s 
    # vector pointing from source-detector line to center of detector. 
    r_delta = r_s_r - project_vector_to_vector(r_s_r, r_n) - c_v - c_h
    # detector row and channel offsets
    det_channel_offset = -np.dot(r_delta, r_h)
    det_row_offset = -np.dot(r_delta, r_v)
    # rotation offset
    delta_source = r_s - project_vector_to_vector(r_s, r_n)
    delta_rot = delta_source - project_vector_to_vector(delta_source, r_a)# rotation offset vector (perpendicular to rotation axis)
    rotation_offset = np.dot(delta_rot, np.cross(r_n, r_a))
    det_channel_offset += rotation_offset*magnification
    return det_channel_offset, det_row_offset

######## END subroutines for NSI-MBIR parameter conversion
