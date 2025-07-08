import jax
import jax.numpy as jnp
import pywt
from scipy import interpolate
from scipy.ndimage import uniform_filter1d, median_filter, binary_dilation
from concurrent.futures import ThreadPoolExecutor


def generate_column_index_matrix(num_rows, num_cols):
    """
    Create a 2D array of indexes used for the sorting technique.

    This code is based on the method described in:
    [Nghia T. Vo et al., 2018] - "Superior techniques for eliminating ring artifacts in x-ray micro-tomography"

    This code is adapted from the Tomopy library:
    https://github.com/tomopy/tomopy.git

    References:
    [1] Vo N, and Atwood RC, and Drakopoulos M. Superior techniques for eliminating ring artifacts in x-ray micro-tomography. Optics Express, 26(22):28396–28412, 2018.
    [2] Tomopy library https://github.com/tomopy/tomopy.git

    Args:
        num_rows (int): number of detector rows in the sinogram
        num_cols (int): number of detector channels in the sinogram

    Returns:
        index_matrix(jax array): a 2D jax array of indexes
    """
    list_index = jnp.arange(0.0, num_cols, 1.0)
    index_matrix = jnp.tile(list_index, (num_rows, 1))
    return index_matrix


def remove_small_stripes_sorting(sino, filter_size, index_matrix):
    """
    Remove small-to-medium partial and fulll stripes using the sorting technique.

    This code is based on the method described in:
    [Nghia T. Vo et al., 2018] - "Superior techniques for eliminating ring artifacts in x-ray micro-tomography"

    This code is adapted from the Tomopy library:
    https://github.com/tomopy/tomopy.git

    References:
    [1] Vo N, and Atwood RC, and Drakopoulos M. Superior techniques for eliminating ring artifacts in x-ray micro-tomography. Optics Express, 26(22):28396–28412, 2018.
    [2] Tomopy library https://github.com/tomopy/tomopy.git

    Args:
        sino (jax array): a 2D slice of the sinogram data with shape (num_views, num_det_channels)
        filter_size (int): window size of the median filter
        index_matrix (jax array): a 2D array of indexes used for the sorting technique

    Return:
        corrected_sino (jax array): corrected 2D slice of the sinogram data after stripes removal
    """

    # Sort each column of the sinogram by its grayscale values
    sino = jnp.transpose(sino)
    stacked_matrix = jnp.stack([index_matrix, sino], axis=2)
    sorted_indices = jnp.argsort(stacked_matrix[:, :, 1], axis=1)
    sorted_indices_expanded = sorted_indices[:, :, None]
    sorted_stacked_matrix = jnp.take_along_axis(stacked_matrix, sorted_indices_expanded, axis=1)

    # Apply the median filter on teh sorted sinogram along each row
    sorted_stacked_matrix = sorted_stacked_matrix.at[:, :, 1].set(median_filter(sorted_stacked_matrix[:, :, 1], (filter_size, 1)))

    # Re-sort the smoothed image columns to the original rows to get the corrected sinogram
    sorted_indices = jnp.argsort(sorted_stacked_matrix[:, :, 0], axis=1)
    sorted_indices_expanded = sorted_indices[:, :, None]
    sort_back_matrix = jnp.take_along_axis(sorted_stacked_matrix, sorted_indices_expanded, axis=1)
    corrected_sino = sort_back_matrix[:, :, 1]
    corrected_sino = jnp.transpose(corrected_sino)

    return corrected_sino

def detect_stripe(list_data, snr):
    """
    Used to locate stripes.
    A segmentation algorithm to separate the extremely positive and negative defects from the normal values in the
    sinogram.

    This code is based on the method described in:
    [Nghia T. Vo et al., 2018] - "Superior techniques for eliminating ring artifacts in x-ray micro-tomography"

    This code is adapted from the Tomopy library:
    https://github.com/tomopy/tomopy.git

    Args:
        list_data (jax array): a normalized 1D array
        snr (float): a ratio between the defective value and the background value. a reasonable choice of snr should be around 3.0 or above

    Returns:
        list_mask (jax array): a 2D binary array denoting the stripes detected
    """

    num_data = list_data.shape[0]

    # Sort the 1D array
    sorted_list = jnp.sort(list_data)[::-1]
    x_list = jnp.arange(0, num_data, 1.0)

    # Apply a linear fit to values around the middle of the sorted array
    # Calculating the noise level to avoid false positives caused by minor background variations
    num_data_drop = jnp.int16(0.25 * num_data)
    (_slope, _intercept) = jnp.polyfit(x_list[num_data_drop:-num_data_drop - 1], sorted_list[num_data_drop:-num_data_drop - 1], 1)
    fitted_value_last_index = _intercept + _slope * x_list[-1]
    noise_level = jnp.abs(fitted_value_last_index - _intercept)
    noise_level = jnp.clip(noise_level, 1e-6, None)
    val1 = jnp.abs(sorted_list[0] - _intercept) / noise_level
    val2 = jnp.abs(sorted_list[-1] - fitted_value_last_index) / noise_level

    # Calculate the upper threshold and the lower threshold
    # Binarize the array by replacing all values between lower and upper threshold with 0 and others with 1.
    list_mask = jnp.zeros_like(list_data)
    if (val1 >= snr):
        upper_threshold = _intercept + noise_level * snr * 0.5
        list_mask = jnp.where(list_data > upper_threshold, 1.0, list_mask)
    if (val2 >= snr):
        lower_threshold = fitted_value_last_index - noise_level * snr * 0.5
        list_mask = jnp.where(list_data <= lower_threshold, 1.0, list_mask)

    return list_mask


def remove_large_stripes_sorting(sino, snr, filter_size, index_matrix, drop_ratio=0.1):
    """
    Remove large partial and full stripes using the sorting technique.

    This code is based on the method described in:
    [Nghia T. Vo et al., 2018] - "Superior techniques for eliminating ring artifacts in x-ray micro-tomography"

    This code is adapted from the Tomopy library:
    https://github.com/tomopy/tomopy.git

    References:
    [1] Vo N, and Atwood RC, and Drakopoulos M. Superior techniques for eliminating ring artifacts in x-ray micro-tomography. Optics Express, 26(22):28396–28412, 2018.
    [2] Tomopy library https://github.com/tomopy/tomopy.git

    Args:
        sino (jax array): a 2D slice of the sinogram data with shape (num_views, num_det_channels)
        snr (float): a ratio between the defective value and the background value
        filter_size (int): window size of the median filter
        index_matrix (jnp array): a 2D array of indexes used for the sorting technique
        drop_ratio (float, optional): ratio of pixels at the top and the bottom of the sinogram to be removed. Defaults to 0.1.

    Returns:
        sino (jax array): corrected 2D slice of the sinogram data after stripes removal
    """

    drop_ratio = jnp.clip(drop_ratio, 0.0, 0.8)
    (num_rows, num_cols) = sino.shape
    num_rows_drop = jnp.int16(0.5 * drop_ratio * num_rows)

    # Sorting the columns of the sinogram and apply the median filter on the sorted image along each row
    sorted_sino = jnp.sort(sino, axis=0)
    smoothed_sino = jnp.array(median_filter(sorted_sino, (1, filter_size)))

    # Compute the column-wise average of the sorted and smoothed sinogram
    # Compute the normalized 1D array
    raw_list = jnp.mean(sorted_sino[num_rows_drop:num_rows - num_rows_drop], axis=0)
    smoothed_list = jnp.mean(smoothed_sino[num_rows_drop:num_rows - num_rows_drop], axis=0)
    normalized_list = jnp.where(
        smoothed_list != 0,
        jnp.divide(raw_list, smoothed_list),
        jnp.ones_like(raw_list)
    )

    # Locate the large stripes
    list_mask = detect_stripe(normalized_list, snr)
    list_mask = jnp.array(binary_dilation(list_mask, iterations=1).astype(list_mask.dtype))
    normalized_factor = jnp.tile(normalized_list, (num_rows, 1))

    # Apply pre-correction to the original sinogram
    sino = sino / normalized_factor

    # Apply the sorting-based algorithm again to get the corrected columns
    sino_T = jnp.transpose(sino)
    stacked_matrix = jnp.stack([index_matrix, sino_T], axis=2)

    sorted_indices = jnp.argsort(stacked_matrix[:, :, 1], axis=1)
    sorted_indices_expanded = sorted_indices[:, :, None]
    sorted_matrix = jnp.take_along_axis(stacked_matrix, sorted_indices_expanded, axis=1)

    sorted_matrix = sorted_matrix.at[:, :, 1].set(jnp.transpose(smoothed_sino))

    sorted_indices = jnp.argsort(sorted_matrix[:, :, 0], axis=1)
    sorted_indices_expanded = sorted_indices[:, :, None]
    sort_back_matrix = jnp.take_along_axis(sorted_matrix, sorted_indices_expanded, axis=1)

    corrected_sino = jnp.transpose(sort_back_matrix[:, :, 1])

    list_x_miss = jnp.where(list_mask > 0.0)[0]

    # Selective Replacement of Defective Columns with corrected columns
    sino = sino.at[:, list_x_miss].set(corrected_sino[:, list_x_miss])
    return sino


def remove_dead_fluctuating_stripes_interpolation(sino, snr, filter_size, index_matrix):
    """
    Remove unresponsive and fluctuating stripes using the interpolation technique.
    Sorting approach does not work here because the rankings of the grayscales are significantly different between
    pixels inside the stripes and outside the stripes. Instead, interpolation is an appropriate choice.

    This code is based on the method described in:
    [Nghia T. Vo et al., 2018] - "Superior techniques for eliminating ring artifacts in x-ray micro-tomography"

    This code is adapted from the Tomopy library:
    https://github.com/tomopy/tomopy.git

    References:
    [1] Vo N, and Atwood RC, and Drakopoulos M. Superior techniques for eliminating ring artifacts in x-ray micro-tomography. Optics Express, 26(22):28396–28412, 2018.
    [2] Tomopy library https://github.com/tomopy/tomopy.git

    Args:
        sino (jax array): a 2D slice of the sinogram data with shape (num_views, num_det_channels)
        snr (float): a ratio between the defective value and the background value
        filter_size (int): window size of the median filter
        index_matrix (jax array): a 2D array of indexes used for the sorting technique

    Returns:
        sino (jax array): corrected 2D slice of the sinogram data after stripes removal

    """
    num_rows = sino.shape[0]

    # Compute the column-wise absolute difference between the original sinogram and the smoothed sinogram
    # Help detect the unresponsive and fluctuating stripes (large diff -> fluctuating stripes; small diff -> unresponsive stripes)
    smoothed_sino = jnp.array(uniform_filter1d(sino, 10, axis=0))
    difference_list = jnp.sum(jnp.abs(sino - smoothed_sino), axis=0)

    # Compute normalized 1D array where the large value correspond to defective pixels
    difference_list_filtered = jnp.array(median_filter(difference_list, size=filter_size))
    normalized_list = jnp.where(
        difference_list_filtered != 0,
        jnp.divide(difference_list, difference_list_filtered),
        jnp.ones_like(difference_list_filtered)
    )

    # Generate binary mask
    list_mask = detect_stripe(normalized_list, snr)
    list_mask = jnp.array(binary_dilation(list_mask, iterations=1).astype(list_mask.dtype))
    list_mask = list_mask.at[0:2].set(0.0)
    list_mask = list_mask.at[-2:].set(0.0)

    # Interpolation
    x_list = jnp.where(list_mask < 1.0)[0]
    y_list = jnp.arange(num_rows)
    z_matrix = sino[:, x_list]
    fit_function = interpolate.RectBivariateSpline(y_list, x_list, z_matrix, kx=1, ky=1)

    # Apply interpolation to defective columns
    list_x_miss = jnp.where(list_mask > 0.0)[0]
    if len(list_x_miss) > 0:
        matrix_x_miss, matrix_y = jnp.meshgrid(list_x_miss, y_list)
        estimate_output = fit_function.ev(jnp.ravel(matrix_y), jnp.ravel(matrix_x_miss))
        sino = sino.at[:, list_x_miss].set(jnp.reshape(estimate_output, matrix_x_miss.shape))

    # Remove residual large stripes
    corrected_sino = remove_large_stripes_sorting(sino, snr, filter_size, index_matrix)

    return corrected_sino

def remove_all_stripe(sino, snr=3, large_filter_size=61, small_filter_size=21):
    """
    Removes all types of stripe artifacts from a sinogram using a combination of three algorithms:
    1. Interpolation-based removal of unresponsive and fluctuating stripes.
    2. Sorting-based removal of large partial and full stripes.
    3. Sorting-based removal of small to medium partial and full stripes.

    This method is adapted from `tomopy.remove_all_stripes()` and is based on:
    Vo N, Atwood RC, Drakopoulos M. "Superior techniques for eliminating ring artifacts in x-ray micro-tomography."
    Optics Express, 26(22):28396–28412, 2018.

    Args:
        sino (jax.Array): A 3D sinogram array with shape (num_views, num_det_rows, num_det_channels).
        snr (float, optional): Signal-to-noise ratio used for stripe detection. A typical value is 3.0. Defaults to 3.
        large_filter_size (int, optional): Median filter window size for removing large stripes. Defaults to 61.
        small_filter_size (int, optional): Median filter window size for removing small-to-medium stripes. Defaults to 21.

    Returns:
        jax.Array: Corrected 3D sinogram array after removing all stripe artifacts.

    Example:
        >>> import jax.numpy as jnp
        >>> import mbirjax.preprocess as mjp
        >>> sino = jnp.ones((180, 128, 256))  # Simulated 3D sinogram
        >>> cleaned_sino = mjp.remove_all_stripe(sino)
    """
    index_matrix = generate_column_index_matrix(sino.shape[2], sino.shape[0])
    index_matrix_cpu = jax.device_put(index_matrix, device=jax.devices("cpu")[0])

    sino_cpu = jax.device_put(sino, device=jax.devices("cpu")[0])

    result = jnp.zeros_like(sino)

    def process_slice(m):
        sino_slice = sino_cpu[:, m, :]
        sino_slice = remove_dead_fluctuating_stripes_interpolation(sino_slice, snr, large_filter_size, index_matrix_cpu)
        sino_slice = remove_small_stripes_sorting(sino_slice, small_filter_size, index_matrix_cpu)
        return sino_slice

    with ThreadPoolExecutor() as executor:
        processed_slices = list(executor.map(process_slice, range(sino.shape[1])))

    for m, processed_slice in enumerate(processed_slices):
        result = result.at[:, m, :].set(processed_slice)

    return jax.device_put(result)


def remove_stripe_fw(sino, wavelet_filter_name="db5", sigma=2):
    """
    Removes vertical stripe artifacts from a 3D sinogram using a combined wavelet-Fourier filtering technique.

    This method uses a 2D Discrete Wavelet Transform followed by a 2D Fourier transform to suppress vertical stripes,
    as described in:
    Beat Münch et al., "Stripe and ring artifact removal with combined wavelet—Fourier filtering", Optics Express, 2009.

    This implementation is adapted from the Tomopy library's `remove_stripe_fw()`:
    https://github.com/tomopy/tomopy.git

    Args:
        sino (jax.Array): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels).
        wavelet_filter_name (str, optional): Wavelet filter type (e.g., 'db5', 'haar'). Defaults to 'db5'.
        sigma (float, optional): Damping parameter in the Fourier domain. Controls the strength of stripe suppression.
            Defaults to 2.

    Returns:
        jax.Array: Corrected sinogram data with reduced vertical stripe artifacts.

    Example:
        >>> import jax.numpy as jnp
        >>> import mbirjax.preprocess as mjp
        >>> sino = jnp.ones((180, 128, 256))  # Simulated sinogram
        >>> cleaned_sino = mjp.remove_stripe_fw(sino)
    """
    # Determine decomposition level L
    level = int(jnp.ceil(jnp.log2(jnp.max(jnp.array(sino.shape)))))
    views, num_rows, num_columns = sino.shape
    padded_views = views + views // 8
    shift_val = views // 16

    for m in range(sino.shape[1]):
        sino_slice = jnp.zeros((padded_views, num_columns), dtype=jnp.float32)
        sino_slice = sino_slice.at[shift_val:views + shift_val].set(sino[:, m, :])

        # 2D Discrete Wavelte Transform
        cH, cV, cD = {}, {}, {}
        for n in range(level):
            sino_slice, (cH[n], cV[n], cD[n]) = pywt.dwt2(sino_slice, wavelet_filter_name)

        # FFT transform of horizontal frequency bands
        for n in range(level):
            # FFT
            fcV = jnp.fft.fftshift(jnp.fft.fft(cV[n], axis=0))
            my, mx = fcV.shape

            # Damping of vertical stripe information
            y_hat = (jnp.arange(-my, my, 2, dtype='float32') + 1) / 2
            damp = -jnp.expm1(-jnp.square(y_hat) / (2 * jnp.square(sigma)))
            fcV *= jnp.transpose(jnp.tile(damp, (mx, 1)))

            # Inverse FFT
            cV[n] = jnp.real(jnp.fft.ifft(jnp.fft.ifftshift(fcV), axis=0))

        # 2D inverse discrete wavelet transform
        for n in range(level)[::-1]:
            sino_slice = sino_slice[0:cH[n].shape[0], 0:cH[n].shape[1]]
            sino_slice = pywt.idwt2((sino_slice, (cH[n], cV[n], cD[n])), wavelet_filter_name)

        sino = sino.at[:, m, :].set(sino_slice[shift_val:views + shift_val, 0:num_columns])

    return sino