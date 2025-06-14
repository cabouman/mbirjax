import warnings
import numpy as np
import jax.numpy as jnp
import h5py


def compute_sino_and_params(filename, bh_correction=True):
    """
    Load ORNL sinogram data from an HDF5 file and prepare all required parameters for cone‐beam reconstruction.

    This function performs the following steps in one call:

      1. Extracts geometry parameters and defaults via `create_proj_params_dict_ornl`.
      2. Reads the raw sinogram data via `load_projection_data_ornl`.
      3. Optionally applies beam hardening correction to the sinogram via `apply_bh_correction`.

    Args:
        filename (str):
            Path to the ORNL HDF5 file containing projection data and geometry attributes.
        bh_correction (bool, optional):
            If True, apply beam hardening correction using the file’s stored parameters. Defaults to True.

    Returns:
        tuple:
            sino (numpy.ndarray):
                sinogram data of shape (num_views, num_det_rows, num_det_channels).
            cone_beam_params (dict):
                Dictionary of mandatory parameters for mbirjax.ConeBeamModel, including:
                  - "sinogram_shape"
                  - "angles"
                  - "source_detector_dist"
                  - "source_iso_dist"
            optional_params (dict):
                Additional model settings for set_params(), including:
                  - "delta_det_channel"
                  - "delta_det_row"
                  - "delta_voxel"
                  - "det_channel_offset"
                  - "det_row_offset"

    Example:
        .. code-block:: python

            sino, cone_beam_params, optional_params = compute_sino_and_params("scan.h5", bh_correction=True)
            ct_model = mbirjax.ConeBeamModel(**cone_beam_params)
            ct_model.set_params(**optional_params)
            recon, recon_dict = ct_model.recon(sino, weights=weights)
    """
    import mbirjax.preprocess as mjp
    with h5py.File(filename, 'r') as h5_file:
        cone_beam_params, optional_params, det_rotation = create_proj_params_dict_ornl(h5_file)

        sinogram = load_projection_data_ornl(h5_file)
        if bh_correction:
            BHCN_params = h5_file.attrs['BHC_params']
            sinogram = apply_bh_correction(sinogram, BHCN_params)

        if np.abs(det_rotation) > 1e-6:
            # Correct the sinogram for detector rotation
            sinogram = mjp.correct_det_rotation_and_background(sinogram, det_rotation=det_rotation)
            warnings.warn('TODO: Verify the direction of sinogram rotation.')

    return sinogram, cone_beam_params, optional_params


def create_proj_params_dict_ornl(h5_file):
    """
    Build cone‐beam geometry dictionaries from an open ORNL HDF5 file.

    This function reads the HDF5 attributes and computes all distances and offsets
    normalized to detector‐pixel units, and converts angles to radians.

    Args:
        h5_file (h5py.File):
            Opened NSI HDF5 file with required attributes:
            - 'distance unit', 'det_pixel_size', 'iso_det_dist', 'src_iso_dist'
            - 'voxel_size', 'det_column_offset', 'det_row_offset'
            - 'angle unit', 'angles', 'det_angle'
            - Dataset group 'projection' with at least one entry

    Returns:
        tuple:
            cone_beam_params (dict):
                Required parameters for ConeBeamModel constructor:
                  - "sinogram_shape" (tuple): shape of the raw projection array.
                  - "angles" (ndarray): projection angles in radians.
                  - "source_detector_dist" (float): source to detector distance (in pixel units).
                  - "source_iso_dist" (float): source to isocenter distance (in pixel units).
            optional_params (dict):
                Additional reconstruction settings for `set_params()`:
                  - "delta_det_channel" (float)
                  - "delta_det_row" (float)
                  - "delta_voxel" (float)
                  - "det_channel_offset" (float)
                  - "det_row_offset" (float)
    """
    # Unit conversion to 'ALU' pixel units if necessary
    det_pixel_size = h5_file.attrs['det_pixel_size']
    iso_det_dist = h5_file.attrs['iso_det_dist'] / det_pixel_size
    source_iso_dist = h5_file.attrs['src_iso_dist'] / det_pixel_size
    source_detector_dist = source_iso_dist + iso_det_dist
    delta_voxel = h5_file.attrs['voxel_size'] / det_pixel_size
    det_channel_offset = -h5_file.attrs['det_column_offset'] / det_pixel_size
    det_row_offset = -h5_file.attrs['det_row_offset'] / det_pixel_size
    delta_det_channel = 1.0
    delta_det_row = 1.0

    # Angle conversion to radians
    if h5_file.attrs['angle unit'] == 'radian':
        angles = h5_file.attrs['angles']
        det_rotation = h5_file.attrs['det_angle']
    else:
        angles = h5_file.attrs['angles'] * np.pi / 180
        det_rotation = h5_file.attrs['det_angle'] * np.pi / 180

    sinogram_shape = h5_file['projection']['NegativeLogNorm_Proj'].shape

    cone_beam_params = dict()
    cone_beam_params["sinogram_shape"] = sinogram_shape
    cone_beam_params["angles"] = angles
    cone_beam_params["source_detector_dist"] = source_detector_dist
    cone_beam_params["source_iso_dist"] = source_iso_dist

    optional_params = dict()
    optional_params["delta_det_channel"] = delta_det_channel
    optional_params["delta_det_row"] = delta_det_row
    optional_params['delta_voxel'] = delta_voxel
    optional_params["det_channel_offset"] = det_channel_offset
    optional_params["det_row_offset"] = det_row_offset

    return cone_beam_params, optional_params, det_rotation


def load_projection_data_ornl(h5_file):
    """
    Read and return the raw sinogram array from an ORNL HDF5 file.

    This function assumes the first entry in the 'projection' group
    is the object scan used to build –log(I/I0) sinogram data.

    Args:
        h5_file (h5py.File):
            Open NSI HDF5 file containing a 'projection' group.

    Returns:
        numpy.ndarray:
            Raw sinogram data as a float32 array.
    """
    sinogram = h5_file['projection']['NegativeLogNorm_Proj'][()].astype(np.float32)
    return sinogram


def apply_bh_correction(sinogram, BHCN_params):
    """
    Perform beam hardening correction on a sinogram.

    Uses a polynomial fit derived from ORNL‐stored beam hardening parameters.

    Args:
        sinogram (numpy.ndarray):
            Uncorrected sinogram data.
        BHCN_params (array-like of length 4):
            Beam hardening parameters [alpha, mu1, mu2, max_thickness].

    Returns:
        numpy.ndarray:
            Sinogram after beam hardening linearization.
    """
    alpha, mu1, mu2, max_thickness = BHCN_params
    poly_coefs = find_linearization_fit(alpha, mu1, mu2, max_thick=max_thickness)
    corrected_sinogram = jnp.polyval(poly_coefs, sinogram)
    corrected_sinogram = np.asarray(corrected_sinogram)
    return corrected_sinogram


def find_linearization_fit(alpha, density1, density2, poly_order=8, max_thick=30.25, step_size=0.25):
    """
    Fit a polynomial that linearizes the beam‐hardened transmission curve.


    Args:
        alpha (float):
            Ratio parameter for two‐material model.
        density1 (float):
            Linear attenuation coefficient of material 1.
        density2 (float):
            Linear attenuation coefficient of material 2.
        poly_order (int, optional):
            Order of the polynomial fit. Defaults to 8.
        max_thick (float, optional):
            Maximum thickness to sample (in same units as sinogram). Defaults to 30.25.
        step_size (float, optional):
            Sampling interval for thickness. Defaults to 0.25.

    Returns:
        numpy.ndarray:
            Coefficients for a polynomial that correct BH effect in measured attenuation (–log I/I0).
    """
    uavg0 = (alpha * density1 + density2) / (1 + alpha)
    t_l = np.arange(0, max_thick, step_size)
    dut = (density2 - density1) * t_l
    prfit_l = density2 * t_l + np.log((1 + alpha) / (1 + alpha * np.exp(dut)))
    xxi_l = np.concatenate(([0], prfit_l))
    yyi_l = np.concatenate(([0], uavg0 * t_l))
    coefs = np.polyfit(xxi_l, yyi_l, poly_order)
    return coefs  # np.poly1d(np.polyfit(xxi_l, yyi_l, poly_order))



