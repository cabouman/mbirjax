import numpy as np
import h5py
import warnings
import matplotlib.pyplot as plt
from sklearn.decomposition import non_negative_factorization
from sklearn.utils.extmath import randomized_svd


# -----------------------------------------------------------------------
# Hyperspectral Neutron Radiographic/Tomographic Data Denoising Functions
# -----------------------------------------------------------------------


def hyper_denoise(data, dataset_type='attenuation', subspace_dimension=None, subspace_basis=None, safety_factor=2,
                  batch_size=2 ** 24, beta_loss='frobenius', max_iter=300, tolerance=1e-6, verbose=1):
    """
    Denoise a hyperspectral dataset using dehydration and rehydration.

    Dehydration:
        Learns (or accepts) a set of :math:`N_s` basis spectra and then projects the full dataset onto that subspace. Typically,
        :math:`N_s << N_k`, where :math:`N_k` is the number of spectral bins. Significant spectral noise is removed while fitting the data
        into lower dimensional subspace.

    Rehydration:
        Projects the subspace domain data back to the hyperspectral domain and retrieves original dimensions.

    Args:
        data: Hyperspectral data array with arbitrary axes and a spectral axis of length :math:`N_k` in the last position.
        dataset_type: 'attenuation' or 'transmission' where attenuation = -log(transmission). Defaults to 'attenuation'.
        subspace_dimension: Desired dimension of the subspace :math:`N_s`. If None, the dimension is either set from the
            provided subspace basis matrix or estimated automatically from the data. Defaults to None.
        subspace_basis: Pre-computed subspace basis spectra of shape :math:`(N_s, N_k)`. If None, the basis spectra are
            estimated directly from the data. Defaults to None.
        safety_factor: Multiplicative factor ≥ 1 used to scale the initial estimate of subspace dimension and ensure
            safer final choice. Defaults to 2.
        batch_size: Size of data processed per batch. Useful for large datasets to limit memory usage. Defaults to 2**24.
        beta_loss: Beta divergence minimized in NMF. Can be 'frobenius' or 'kullback-leibler'. Defaults to 'frobenius'.
        max_iter: Maximum iterations for the NMF solver. Defaults to 300.
        tolerance: Convergence tolerance for the NMF solver. Defaults to 1e-6.
        verbose: Verbosity level. If 0, prints nothing; if 1, prints details; if >1, also generates plots. Defaults to 1.

    Returns:
        Denoised hyperspectral data with the same shape as the input data.

    Example:
        >>> denoised_data = hyper_denoise(data, subspace_dimension=10)
        >>> data.shape, denoised_data.shape
        ((N_x, N_y, N_z, ..., N_k), (N_x, N_y, N_z, ..., N_k))

    Note:
        The function works with hyperspectral data in either sinogram or space domain.
    """
    # Get the data dimensions
    data_shape = data.shape
    num_bands = data_shape[-1]  # Number of wavelength bands
    num_points = data.size // num_bands  # Number of pixels or voxels

    # Reshape the hyperspectral data to 2D (number of data points x number of spectral bands)
    data = data.reshape(num_points, num_bands)

    # Determine number of batches
    num_points_batch = batch_size // num_bands  # Number of hyperspectral pixels or voxels per batch
    num_batches = int(np.ceil(num_points / num_points_batch))  # Number of batches

    if verbose >= 1:
        print("hyper_denoise(): ")
        print("   -Number of batches: ", num_batches)

    # Estimate subspace basis (if not provided) from sampled data if the dataset is too big
    if num_batches > 1 and subspace_basis is None:
        # Sample rows without replacement
        rng = np.random.default_rng()
        row_idx = rng.choice(num_points, size=num_points_batch, replace=False)
        sample_data = data[row_idx, :]

        _, subspace_basis, _ = dehydrate(sample_data,
                                         dataset_type=dataset_type,
                                         subspace_dimension=subspace_dimension,
                                         subspace_basis=subspace_basis,
                                         safety_factor=safety_factor,
                                         beta_loss=beta_loss,
                                         max_iter=max_iter,
                                         tolerance=tolerance,
                                         verbose=verbose)
        verbose = 0  # dehydrate already printed verbose info, so we won't do it in the loop below

    # Denoise data in batches
    denoised_data = np.zeros_like(data)
    for batch in range(num_batches):
        batch_start = batch * num_points_batch
        batch_stop = min((batch + 1) * num_points_batch, num_points)
        batch_data = data[batch_start: batch_stop]

        dehydrated_data = dehydrate(batch_data,
                                    dataset_type=dataset_type,
                                    subspace_dimension=subspace_dimension,
                                    subspace_basis=subspace_basis,
                                    safety_factor=safety_factor,
                                    beta_loss=beta_loss,
                                    max_iter=max_iter,
                                    tolerance=tolerance,
                                    verbose=verbose)  # verbose may be 0 here if the info was printed above
        verbose = 0  # dehydrate already printed verbose info, so we won't do it the next time through

        denoised_data[batch_start: batch_stop] = rehydrate(dehydrated_data)

    # Reshape back to original dimensions
    denoised_data = denoised_data.reshape(data_shape)

    return denoised_data


def dehydrate(data, dataset_type='attenuation', subspace_dimension=None, subspace_basis=None, safety_factor=2,
              beta_loss='frobenius', max_iter=300, tolerance=1e-6, verbose=1):
    """
    Dehydrate/compress a hyperspectral dataset onto a low-dimensional subspace.

    The function learns (or accepts) a set of :math:`N_s` basis spectra, projects the full dataset onto that subspace, and
    returns the low-dimensional subspace data along with the basis spectra. Typically, :math:`N_s << N_k`, where :math:`N_k` is
    the number of spectral bins.

    Args:
        data: Hyperspectral data array with arbitrary axes and a spectral axis of length :math:`N_k` in the last position.
        dataset_type: 'attenuation' or 'transmission' where attenuation = -log(transmission). Defaults to 'attenuation'.
        subspace_dimension: Desired dimension of the subspace :math:`N_s`. If None, the dimension is either set from the
            provided subspace basis matrix or estimated automatically from the data. Defaults to None.
        subspace_basis: Pre-computed subspace basis spectra of shape :math:`(N_s, N_k)`. If None, the basis spectra are
            estimated directly from the data. Defaults to None.
        safety_factor: Multiplicative factor ≥ 1 used to scale the initial estimate of subspace dimension and ensure
            safer final choice. Defaults to 2.
        beta_loss: Beta divergence minimized in NMF. Can be 'frobenius' or 'kullback-leibler'. Defaults to 'frobenius'.
        max_iter: Maximum iterations for the NMF solver. Defaults to 300.
        tolerance: Convergence tolerance for the NMF solver. Defaults to 1e-6.
        verbose: Verbosity level. If 0, prints nothing; if 1, prints details; if >1, also generates plots. Defaults to 1.

    Returns:
        A list containing the dehydrated hyperspectral dataset in the form [subspace_data, subspace_basis, dataset_type].
            - subspace_data: ndarray with same shape as input data except the last axis length is :math:`N_s`.
            - subspace_basis: ndarray of shape :math:`(N_s, N_k)`, where rows are subspace basis spectra.
            - dataset_type: Can be 'attenuation' or 'transmission' where attenuation = -log(transmission).

    Example:
        >>> [subspace_data, subspace_basis, dataset_type] = dehydrate(data, subspace_dimension=10)
        >>> data.shape, subspace_data.shape, subspace_basis.shape
        ((N_x, N_y, N_z, ..., N_k), (N_x, N_y, N_z, ..., 10), (10, N_k))

    Note:
        The function works with hyperspectral data in either sinogram or space domain.
    """
    # define epsilon
    epsilon = 1e-8

    # Check that the dataset_type is either 'attenuation' or 'transmission'
    if dataset_type not in ('attenuation', 'transmission'):
        raise ValueError("'dataset_type' must be either 'attenuation' or 'transmission'.")

    # Get the data dimensions
    data_shape = data.shape

    # Reshape the hyperspectral data to 2D (number of data points x number of spectral bins)
    data = data.reshape(-1, data_shape[-1])

    # Convert data to attenuation if it is transmission
    if dataset_type == 'transmission':
        data[data < epsilon] = epsilon
        data = - np.log(data)

    # Replace values less than zero with zeros as NMF expects all the matrices to be non-negative
    data[data < 0] = 0

    # Cast to float64 for numerical stability in svd
    data = np.asarray(data, dtype=np.float64)
    if subspace_basis is not None:
        subspace_basis = np.asarray(subspace_basis, dtype=np.float64)

    # Set options based on subspace basis availability
    if subspace_basis is None:
        init = 'nndsvd'  # Initialize NMF basis (H) and data (W) using Non-Negative Double Singular Value Decomposition
        update_basis = True
    else:
        init = 'custom'  # Initialize NMF basis (H) and data (W) using given input and an array of zeros, respectively
        update_basis = False
        subspace_dimension = subspace_basis.shape[0]

    # Automatically estimate the dimension of the subspace if required
    if subspace_dimension is None:
        subspace_dimension = _estimate_subspace_dimension(data, safety_factor=safety_factor, verbose=verbose)

    # Set NMF solver based on the selected beta loss
    if beta_loss == 'frobenius':
        solver = 'cd'
    elif beta_loss == 'kullback-leibler':
        solver = 'mu'
    else:
        warnings.warn(f"Invalid beta_loss '{beta_loss}' specified: falling back to 'frobenius'.")
        beta_loss = 'frobenius'
        solver = 'cd'

    # Perform NMF using Scikit-learn
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        subspace_data, subspace_basis, _ = non_negative_factorization(data,
                                                                      n_components=subspace_dimension,
                                                                      init=init,
                                                                      W=None,  # If None, initialized via init
                                                                      H=subspace_basis,  # If None, initialized via init
                                                                      beta_loss=beta_loss,
                                                                      solver=solver,
                                                                      tol=tolerance,
                                                                      max_iter=max_iter,
                                                                      update_H=update_basis)

    # Reshape subspace data to original dimensions (except the spectral axis)
    subspace_data = subspace_data.reshape(*data_shape[:-1], -1)

    if verbose >= 1:
        print("dehydrate(): ")
        print("   -Original spectral dimension: ", data.shape[-1])
        print("   -Estimated/given subspace dimension: ", subspace_data.shape[-1])

    # Cast back to float32 for memory efficiency
    subspace_data = np.asarray(subspace_data, dtype=np.float32)
    subspace_basis = np.asarray(subspace_basis, dtype=np.float32)
    dehydrated_data = [subspace_data, subspace_basis, dataset_type]
    return dehydrated_data


def rehydrate(dehydrated_data, hyperspectral_idx=None):
    """
    Rehydrate/decompress selected spectral bins from dehydrated hyperspectral data.

    Args:
        dehydrated_data: Dehydrated hyperspectral data in the form [subspace_data, subspace_basis, dataset_type]:

            - subspace_data: ndarray with arbitrary axes and a subspace axis of length :math:`N_s` in the last position.
            - subspace_basis: ndarray of shape :math:`(N_s, N_k)`, where rows are subspace basis spectra.
            - dataset_type: 'attenuation' or 'transmission' where attenuation = -log(transmission).
        hyperspectral_idx: A list of :math:`N_h` indices along the original spectral axis to rehydrate. If None, all :math:`N_k`
            spectral bins are rehydrated. Defaults to None.

    Returns:
        Rehydrated/decompressed hyperspectral data with the same shape as the input subspace_data except the last axis
        length is :math:`N_h (N_h <= N_k)`.

    Example:
        >>> hyper_data = rehydrate([subspace_data, subspace_basis, dataset_type], hyperspectral_idx=[5, 10, 15])
        >>> subspace_data.shape, subspace_basis.shape, hyper_data.shape
        ((N_x, N_y, N_z, ..., N_s), (N_s, N_k), (N_x, N_y, N_z, ..., 3))

    Note:
        The function works with hyperspectral data in either sinogram or space domain.
    """
    [subspace_data, subspace_basis, dataset_type] = dehydrated_data
    if hyperspectral_idx is None:
        rehydrated_data = subspace_data @ subspace_basis
    else:
        rehydrated_data = subspace_data @ subspace_basis[:, hyperspectral_idx]

    if dataset_type == 'transmission':
        rehydrated_data = np.exp(-rehydrated_data)

    return rehydrated_data


def _estimate_subspace_dimension(data, safety_factor=2, noise_fit_window=[25.0, 75.0], threshold=1.5, random_state=None,
                                 verbose=1):
    """
    Estimate the signal subspace dimension using a log-linear fit to singular values.

    Args:
        data: 2D array of shape (num_samples, :math:`N_k`). Values should be real.
        safety_factor: Multiplicative factor ≥ 1 used to scale the initial estimate of subspace dimension and ensure
            safer final choice. Defaults to 2.
        noise_fit_window: Two-element list or tuple [start_percent, stop_percent] indicating the percentile window (0–100)
            over which the singular value fitting is performed. Defaults to [25.0, 75.0].
        threshold: Multiplicative factor to define the cutoff relative to the predicted singular values. Defaults to 1.5.
        random_state: Random seed for reproducibility of row sampling and SVD. Defaults to None.
        verbose: Verbosity level. If >1, plots singular values, fit, and threshold curves. Defaults to 1.

    Returns:
        Estimated dimension of the signal subspace (positive integer).
    """
    if data.ndim != 2:
        raise ValueError("`data` must be a 2D array shaped (samples, N_k).")

    n_points, n_bands = data.shape

    # Decide how many rows to sample for speed/robustness
    sample_size = min(n_points, n_bands)

    # Sample rows without replacement
    rng = np.random.default_rng(random_state)
    row_idx = rng.choice(n_points, size=sample_size, replace=False)

    # Cast to float64 for numerical stability in svd
    Y = np.asarray(data[row_idx, :], dtype=np.float64)

    # Compute singular values via randomized SVD
    _, s, _ = randomized_svd(Y, n_components=sample_size, random_state=random_state)

    # Guard against degenerate cases
    s = np.asarray(s, dtype=float)
    if s.size == 0:
        return 0

    # Extract start and stop percent from noise_fit_window
    start_percent, stop_percent = noise_fit_window
    # Fit window around percentile: [percentile-10, percentile+10], in s-index space
    start_idx = int(np.floor((start_percent / 100.0) * s.size))
    stop_idx = int(np.ceil((stop_percent / 100.0) * s.size))

    # Clip and ensure at least 2 points
    start_idx = max(0, min(start_idx, s.size - 2))
    stop_idx = max(start_idx + 2, min(stop_idx, s.size))

    # Fit log(s) ≈ a*n + b on [start_idx:stop_idx]
    n = np.arange(s.size)
    a, b = np.polyfit(n[start_idx:stop_idx], np.log(s[start_idx:stop_idx] + 1e-12), 1)

    # Predicted singular values for all indices
    s_pred = np.exp(a * n + b)

    # Compute tau by scaling the predicted singular values with the threshold
    tau = threshold * s_pred

    # Consider singular values > the corresponding tau values to be associated with signals
    signal_flag = s > tau
    subspace_dimension = int(np.sum(signal_flag[:start_idx]))

    if verbose > 1:
        plt.figure()
        plt.semilogy(s, label='s: actual singular values from data (signal + noise)')
        plt.semilogy(s_pred, label='s_pred: predicted singular values from noise model')
        plt.semilogy(tau, label='tau: noise and signal discriminator (threshold x s_pred)')
        plt.title("Modeling noise singular values for subspace dimension estimation")
        plt.xlabel("singular value index")
        plt.ylabel("singular value")
        plt.legend()

    # Multiply by safety factor
    subspace_dimension = int(np.ceil(safety_factor * subspace_dimension))

    return max(1, subspace_dimension)


# -----------------------------------------------------------------------
# HDF5 Import/Export Utilities for Hyperspectral Neutron Data/Metadata
# -----------------------------------------------------------------------


# Description of the allowed keys
KEY_DESCRIPTIONS = {
    "dataset_name": "Character string with the name of the dataset.",
    "dataset_type": "'attenuation' or 'transmission'.",
    "dataset_modality": "'hyperspectral neutron'.",
    "wavelengths": "Array of wavelength values in Angstroms.",
    "alu_unit": "Character string defining geometry unit (e.g., 'mm' or 'cm').",
    "alu_value": "Float that represents the value of 1 ALU in the defined unit.",
    "delta_det_channel": "Detector channel spacing in ALU.",
    "delta_det_row": "Detector row spacing in ALU.",
    "dataset_geometry": "'parallel' or 'cone'.",
    "angles": "Array of view angles in degrees.",
    "det_channel_offset": "Assumed offset between center of rotation and center of detector in ALU.",
    "source_detector_dist": "Distance from source to detector in ALU.",
    "source_iso_dist": "Distance from source to iso in ALU."
}

# Acceptable input options for certain keys
VALIDATION_RULES = {
    "dataset_type": (None, "attenuation", "transmission"),
    "dataset_modality": (None, "hyperspectral neutron"),
    "dataset_geometry": (None, "parallel", "cone"),
}

# Allowed keys derived from the KEY_DESCRIPTIONS
ALLOWED_KEYS = list(KEY_DESCRIPTIONS.keys())


def _validate_key(key, value):
    """Validate categorical keys according to VALIDATION_RULES."""
    if key in VALIDATION_RULES and value not in VALIDATION_RULES[key]:
        valid_options = [v for v in VALIDATION_RULES[key] if v is not None]
        warnings.warn(f"Invalid '{key}': should be one of {valid_options}.")


def _with_key_docstring(style):
    """Function to insert key descriptions into docstrings."""
    indent = "\t- " if style == "dict" else "\t"
    text = "\n".join(f"{indent}{k}: {v}" for k, v in KEY_DESCRIPTIONS.items())

    def decorator(func):
        if func.__doc__:
            func.__doc__ = func.__doc__.replace("{_KEY_DOCS}", text)
        return func

    return decorator


def import_hsnt_list_hdf5(filename):
    """
    Returns a list of all datasets in the HDF5 file.

    Args:
        filename: Path to the HDF5 file containing the datasets.

    Returns:
        A list of dataset names available in the HDF5 file.
    """
    with h5py.File(filename, "r") as f:
        dataset_names = list(f.keys())

    return dataset_names


@_with_key_docstring("dict")
def import_hsnt_data_hdf5(filename, dataset_name):
    """
    Import a hyperspectral dataset and metadata from an HDF5 file.

    Args:
        filename: Path to the HDF5 file.
        dataset_name: Character string with the name of the dataset.

    Returns:
        A list containing hyperspectral data and parameters in the form [data, metadata].
            - data: ndarray with spectral last axis (hyperspectral form) or a list (dehydrated form).
            - metadata: A dictionary with the keys shown below.

    Keys:
    {_KEY_DOCS}

    Note:
        Multiple datasets can coexist in the same HDF5 file, each stored under a unique dataset name.
    """
    with h5py.File(filename, "r") as f:
        if dataset_name not in f:
            raise ValueError(f"Dataset '{dataset_name}' not found in file '{filename}'.")
        group = f[dataset_name]

        # Check if data is dehydrated/compressed
        dehydrated = all(k in group for k in ["subspace_data", "subspace_basis", "dataset_type"])

        # Importing data
        if dehydrated:
            data = [group["subspace_data"][()],
                    group["subspace_basis"][()],
                    group["dataset_type"][()].decode()]
        else:
            data = group["data"][()]

        # Importing metadata
        metadata = {"dataset_name": dataset_name}
        for key in ALLOWED_KEYS:
            if key == "dataset_name":
                continue
            if key in group:
                value = group[key][()]
                if isinstance(value, (bytes, np.bytes_)):
                    value = value.decode()
                elif isinstance(value, np.ndarray) and value.shape == ():
                    value = value.item()
                metadata[key] = value
            else:
                metadata[key] = None

    # Validate categorical keys
    for key, value in metadata.items():
        _validate_key(key, value)

    return [data, metadata]


@_with_key_docstring("arg")
def create_hsnt_metadata(**kwargs):
    """
    Create a dictionary of parameters (metadata) associated with a hyperspectral neutron dataset.

    Args:
    {_KEY_DOCS}

    Returns:
        dict: Dictionary containing hyperspectral neutron dataset parameters (metadata).

    Example:
        >>> metadata = create_hsnt_metadata(
        ...     dataset_name="sample1",
        ...     dataset_type="attenuation",
        ...     dataset_modality="hyperspectral neutron",
        ...     wavelengths=np.linspace(1.0, 5.0, 50),
        ...     alu_unit="mm",
        ...     alu_value=1.0,
        ...     dataset_geometry="parallel",
        ...     angles=np.linspace(0, 180, 10)
        ... )
        >>> print(metadata["dataset_name"])
        sample1
    """
    # Warn for unexpected keyword arguments
    for key in kwargs.keys():
        if key not in ALLOWED_KEYS:
            warnings.warn(f"Ignoring invalid key '{key}' in arguments.")

    metadata = {k: kwargs.get(k, None) for k in ALLOWED_KEYS}
    if not metadata.get("dataset_name"):
        raise ValueError("'dataset_name' is required.")

    # Validation
    for key, value in metadata.items():
        _validate_key(key, value)

    return metadata


@_with_key_docstring("dict")
def export_hsnt_data_hdf5(filename, data, metadata):
    """
    Export a hyperspectral dataset and metadata to an HDF5 file.

    Args:
        filename: Path to the HDF5 file.
        data: ndarray with spectral last axis (hyperspectral form) or a list (dehydrated form).
        metadata: A dictionary with the keys shown below. Use create_hsnt_metadata to create a metadata dictionary.

    Keys:
    {_KEY_DOCS}

    Returns:
        None. Creates or appends to an HDF5 file with the corresponding structure.

    Note:
        - Multiple datasets can coexist in the same HDF5 file, each is stored under a unique dataset name.
        - If a dataset with the same name already exists in the file, it will be overwritten with a warning.
    """
    dataset_name = metadata.get("dataset_name")
    if not dataset_name:
        raise ValueError("'dataset_name' is required in metadata.")

    # Check if data is dehydrated/compressed
    dehydrated = (isinstance(data, list)
                  and len(data) == 3
                  and isinstance(data[2], str)
                  and data[2] in VALIDATION_RULES["dataset_type"][1:])

    # Validate categorical keys before writing
    for key, value in metadata.items():
        _validate_key(key, value)

    with h5py.File(filename, "a") as f:
        if dataset_name in f:
            warnings.warn(f"Overwriting existing dataset '{dataset_name}' in the HDF5 file.")
            del f[dataset_name]
        group = f.create_group(dataset_name)

        # Exporting data
        if dehydrated:
            group.create_dataset("subspace_data", data=data[0])
            group.create_dataset("subspace_basis", data=data[1])
            group.create_dataset("dataset_type", data=np.bytes_(data[2]))
        else:
            group.create_dataset("data", data=data)

        # Exporting metadata
        for key, value in metadata.items():
            if key not in ALLOWED_KEYS:
                warnings.warn(f"Ignoring invalid key '{key}' in metadata.")
                continue
            if key == "dataset_name" or value is None or (key == "dataset_type" and dehydrated):
                continue
            if isinstance(value, str):
                group.create_dataset(key, data=np.bytes_(value))
            else:
                group.create_dataset(key, data=value)


# -----------------------------------------------------------------------
# Noisy Hyperspectral Neutron Data Simulation Function (Ni, Cu, and Al)
# -----------------------------------------------------------------------


def generate_hyper_data(material_basis, detector_rows=64, detector_columns=64, dosage_rate=300, material_thickness=None,
                        verbose=1):
    """
    Simulate noisy hyperspectral neutron attenuation data for :math:`N_m=3` materials (Ni, Cu, Al) and :math:`N_k` wavelength bins.

    Args:
        material_basis: ndarray of shape :math:`(N_m, N_k)`, where rows are material linear attenuation coefficient spectra.
        detector_rows: Number of rows in the detector :math:`(N_r)`. Defaults to 64.
        detector_columns: Number of columns in the detector :math:`(N_c)`. Defaults to 64.
        dosage_rate: Neutron dosage rate during hyperspectral data collection. Defaults to 300.
        material_thickness: Material thicknesses (cm) for Ni, Cu, and Al. Defaults to {"Ni": 2.0, "Cu": 2.0, "Al": 10.0}.
        verbose: Verbosity level. If 0, prints nothing; if 1, prints details; if >1, also generates plots. Defaults to 1.

    Returns:
        A list in the form [hsnt_data, gt_hyper_projection].
            - hsnt_data: Simulated noisy hyperspectral data of shape :math:`(N_r, N_c, N_k)`.
            - gt_hyper_projection: Ground truth noiseless hyperspectral data of same shape.

    """
    # Ensure material_basis has exactly 3 rows
    if material_basis.shape[0] != 3:
        raise ValueError("material_basis must have exactly 3 rows (Ni, Cu, Al).")

    # Validate geometry and inputs
    if detector_rows < 3 or detector_columns < 2:
        raise ValueError("detector_rows must be ≥3 and detector_columns ≥2.")
    if dosage_rate <= 0:
        raise ValueError("dosage_rate must be positive.")

    # Handle default material_thickness and verify required keys
    if material_thickness is None:
        material_thickness = {"Ni": 2.0, "Cu": 2.0, "Al": 10.0}
    required = {"Ni", "Cu", "Al"}
    missing = required - set(material_thickness)
    if missing:
        raise KeyError(f"material_thickness missing keys: {sorted(missing)}")

    # Basic sanity on basis values
    if np.any(material_basis < 0):
        raise ValueError("material_basis should be non-negative attenuation coefficients.")

    # Set variable values
    epsilon = 1e-8
    number_of_materials = material_basis.shape[0]
    number_of_wavelengths = material_basis.shape[1]

    # Generate simulated projection data for 3 materials (Ni, Cu, and Al)
    height = detector_rows // 3
    width = detector_columns // 2
    material_projection = np.zeros((detector_rows, detector_columns, number_of_materials)).astype(np.float32)
    material_projection[:height, width // 2:width + width // 2, 0] = material_thickness["Ni"]
    material_projection[2 * height:, width // 2:width + width // 2, 1] = material_thickness["Cu"]
    material_projection[height:2 * height, width // 2:width + width // 2, 2] = material_thickness["Al"]

    # Generate noiseless hyperspectral projection data using rehydrate function
    gt_hyper_projection = rehydrate([material_projection, material_basis, 'attenuation'])

    # Generate noiseless hyperspectral open beam data using the given dosage rate
    noiseless_open_beam = dosage_rate * np.ones((detector_rows, detector_columns, number_of_wavelengths)).astype(
        np.float32)

    # Generate noiseless raw hyperspectral neutron counts
    noiseless_object_scan = np.exp(-gt_hyper_projection) * noiseless_open_beam
    noiseless_object_scan = np.nan_to_num(noiseless_object_scan, nan=0, posinf=0, neginf=0)

    # Generate noisy neutron counts from Poisson distribution
    noisy_open_beam = np.random.poisson(noiseless_open_beam).astype(np.float32)
    noisy_object_scan = np.random.poisson(noiseless_object_scan).astype(np.float32)

    # Generate noisy hyperspectral projection data
    ratio = noisy_object_scan / noisy_open_beam
    ratio[ratio < epsilon] = epsilon
    noisy_hyper_projection = -np.log(ratio)

    if verbose >= 1:
        print("generate_hyper_data(): ")
        print("   -Shape of material_basis (linear attenuation coefficients for Ni, Cu, and Al):", material_basis.shape)
        print("   -Shape of material_projection (density of Ni, Cu, and Al):", material_projection.shape)
        print("   -Shape of hyperspectral data: ", noisy_hyper_projection.shape)

    if verbose > 1:
        plt.figure()
        plt.plot(material_basis.T)  # each column is a basis function
        plt.xlabel("wavelength index")
        plt.ylabel("linear attenuation ($cm^{-1}$)")
        plt.title("Material basis functions (Ni, Cu, Al)")
        plt.legend(["Ni", "Cu", "Al"])

    return [noisy_hyper_projection, gt_hyper_projection]
