import numpy as np
import warnings
import math
import scipy

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
    
    #### compute the sinogram. 
    # warning handler for sinogram calculation
    def sino_warning_handler(type, flag):
        print("mbirjax.preprocess.transmission_CT_compute_sino(): Invalid sinogram entries encountered. Please use mbirjax.preprocess.interpolate_defective_pixels() to correct the invalid entries.")
    
    np.seterrcall(sino_warning_handler) 
    # If warning encountered during sinogram computation, then print out customized warning message defined in sino_warning_handler()
    with np.errstate(invalid='call'):
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
    sino = scipy.ndimage.rotate(sino, np.rad2deg(det_rotation), axes=(1,2), reshape=False, order=3)
    # weights not provided
    if weights is None:
        return sino
    # weights provided
    print("correct_det_rotation: weights provided by the user. Please note that zero weight entries might become non-zero after tilt angle correction.") 
    weights = scipy.ndimage.rotate(weights, np.rad2deg(det_rotation), axes=(1,2), reshape=False, order=3)
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

######## subroutines for image cropping and down-sampling
def downsample_scans(obj_scan, blank_scan, dark_scan,
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


def crop_scans(obj_scan, blank_scan, dark_scan,
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
