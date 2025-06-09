from jax import numpy as jnp

import mbirjax


def gen_huber_weights(weights, sino_error, T=1.0, delta=1.0, epsilon=1e-6):
    """
    This function generates generalized Huber weights based on the method described in the referenced notes.
    It adds robustness by treating any element where ``|sino_error / weights| > T`` as an outlier,
    down-weighting it according to the generalized Huber function.

    The function returns new `ghuber_weights`.

    Typically, to obtain the final robust weights, the `ghuber_weights` should be multiplied by the original `weights`:

        final_weights = weights * ghuber_weights

    Args:
        weights: jnp.ndarray of shape (views, rows, cols):
            Initial weights, typically derived from inverse variance estimates.
        sino_error: jnp.ndarray of shape (views, rows, cols):
            Sinogram error array representing deviations from the model.
        T: float, optional (default=1.0):
            Threshold parameter; values greater than T are treated as outliers.
        delta: float, optional (default=1.0):
            Controls the strength of the generalized Huber function (delta=1 corresponds to the conventional Huber).
        epsilon: float, optional (default=1e-6):
            Small number to avoid division by zero.

    Returns:
        huber_weights: jnp.ndarray of shape (views, rows, cols)
            The computed generalized Huber weights.

    Notes:
        The generalized Huber function used in this function is based on:
        Venkatakrishnan, S. V., Drummy, L. F., Jackson, M., De Graef, M., Simmons, J. P., and Bouman, C. A.,
        "Model-Based Iterative Reconstruction for Bright-Field Electron Tomography,"
        IEEE Transactions on Computational Imaging, vol. 1, no. 1, pp. 1–15, 2015. DOI: 10.1109/TCI.2014.2371751

    Example:
        >>> from mbirjax import gen_huber_weights
        >>> huber_weights = gen_huber_weights(weights, sino_error)
        >>> final_weights = weights * huber_weights
    """
    if not (0.0 <= delta <= 1.0):
        raise ValueError("delta must be between 0 and 1.")

    weights = jnp.asarray(weights)
    sino_error = jnp.asarray(sino_error)

    # Compute std and global alpha
    std = 1.0 / jnp.maximum(jnp.sqrt(weights), epsilon)
    alpha = jnp.linalg.norm(sino_error) / (jnp.linalg.norm(std) + epsilon)
    std_norm = alpha * std

    # Compute normalized error
    normalized_error = sino_error / std_norm
    abs_norm_error = jnp.abs(normalized_error)

    # Apply generalized Huber function
    huber_weights = jnp.where(abs_norm_error <= T, 1.0, (delta * T) / (abs_norm_error + epsilon))

    return huber_weights


def BH_correction(sino, alpha, batch_size=64):
    """
    Apply a polynomial beam hardening correction to a sinogram.

    This function applies a polynomial correction to each view of the sinogram
    by evaluating powers of the sinogram values and weighting them by the coefficients in `alpha`,
    while also including the original linear term (the sinogram itself).

    The corrected sinogram is computed as:

        corrected_sino = sino + alpha[0] * sino**2 + alpha[1] * sino**3 + ...

    It processes the sinogram in batches of views for memory efficiency.

    Args:
        sino (jnp.ndarray or np.ndarray of shape (views, rows, cols)):
            Input sinogram to correct.
        alpha (list or array of floats):
            Coefficients for the polynomial correction. The k-th term corresponds to sino^(k+1).
        batch_size (int, optional, default=16):
            Number of views to process in a single batch.

    Returns:
        corrected_sino: jnp.ndarray of shape (views, rows, cols)
            Beam hardening corrected sinogram.

    Example:
        >>> from mbirjax.preprocess import BH_correction
        >>> alpha = [1.0, 0.2, 0.1]  # Correction: sino + 0.2 * sino^2 + 0.1 * sino^3
        >>> corrected_sino = BH_correction(sino, alpha)
    """
    # Ensure inputs are JAX arrays
    sino = jnp.asarray(sino)
    alpha = jnp.asarray(alpha)

    views, rows, cols = sino.shape
    corrected = []

    for i in range(0, views, batch_size):
        sino_batch = sino[i:i+batch_size]

        # Initialize corrected batch to the linear term (sino_batch)
        corrected_batch = jnp.array(sino_batch)

        # Apply polynomial terms
        for k in range(len(alpha)):
            corrected_batch += alpha[k] * jnp.power(sino_batch, k + 1)

        corrected.append(corrected_batch)

    corrected_sino = jnp.concatenate(corrected, axis=0)

    return corrected_sino
