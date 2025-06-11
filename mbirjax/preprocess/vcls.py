import os
import random
import tempfile
import warnings

import numpy as np
import matplotlib.pyplot as plt

import mbirjax as mj
import jax.numpy as jnp
import tqdm  # Included in mbirjax


def subsample_R_gamma(R, gamma, selected_indices):
    """
    Extract a submatrix of R and subvector of gamma corresponding to the selected indices.

    Args:
        R (ndarray): Full covariance matrix of shape (N, N).
        gamma (ndarray): Full gamma vector of shape (N, 1).
        selected_indices (ndarray): 1D array of indices to select.

    Returns:
        tuple: A tuple (R_sub, gamma_sub) where:
            R_sub (ndarray): Submatrix of shape (K, K).
            gamma_sub (ndarray): Subvector of shape (K, 1).
    """
    R_sub = R[selected_indices[:, None], selected_indices]
    gamma_sub = gamma[selected_indices, :]
    return R_sub, gamma_sub


def get_ct_model(geometry_type, sinogram_shape, angles, source_detector_dist=None, source_iso_dist=None):
    """
    Create an instance of TomographyModel with the given parameters

    Args:
        geometry_type (str): 'parallel' or 'cone'
        sinogram_shape (tuple list of int): (num_views, num_rows, num_channels)
        angles (ndarray of float): 1D vector of projection angles in radians
        source_detector_dist (float or None, optional): Distance in ALU from source to detector.  Defaults to None for geometries that don't need this.
        source_iso_dist (float or None, optional): Distance in ALU from source to iso.  Defaults to None for geometries that don't need this.

    Returns:
        An instance of ConeBeamModel or ParallelBeam model
    """
    if geometry_type == 'cone':
        model = mj.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist,
                                 source_iso_dist=source_iso_dist)
    elif geometry_type == 'parallel':
        model = mj.ParallelBeamModel(sinogram_shape, angles)
    else:
        raise ValueError('Invalid geometry type.  Expected cone or parallel, got {}'.format(geometry_type))

    return model


def copy_ct_model(ct_model, new_angles):
    """
    Create a TomographyModel with the same type and parameters as the given ct_model except with the new input angles
    and a corresponding sinogram shape.

    Args:
        ct_model (TomographyModel): The model to copy.
        new_angles (ndarray of float): 1D vector of projection angles in radians

    Returns:
        An instance of ConeBeamModel or ParallelBeam model
    """
    required_param_names = ct_model.get_required_param_names()
    required_params, other_params = ct_model.get_required_params_from_dict(ct_model.params,
                                                                           required_param_names=required_param_names,
                                                                           values_only=True)

    #  Get the shape of the old sinogram
    old_shape = ct_model.get_params('sinogram_shape')

    # Set the new sinogram shape and angles
    required_params['sinogram_shape'] = (len(new_angles), old_shape[1], old_shape[2])
    required_params['angles'] = new_angles
    new_model = type(ct_model)(**required_params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new_model.set_params(**other_params)

    return new_model


def max_abs_neighbor_diff(arr):
    padded = np.pad(arr, pad_width=1, mode='reflect')
    center = arr
    max_diff = np.zeros_like(arr)

    # Define the directional offsets: (di, dj)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    for di, dj in directions:
        neighbor = padded[1 + di : 1 + di + arr.shape[0],
                          1 + dj : 1 + dj + arr.shape[1]]
        diff = np.abs(center - neighbor)
        np.maximum(max_diff, diff, out=max_diff)

    return max_diff



def get_opt_views(ct_model, reference_object, num_selected_views, r_1=0.002, r_2=0.5, verbose=0, seed=None):
    """
    Compute the optimal view angles by minimizing the View Covariance Loss (VCL) using a stochastic greedy optimization algorithm.
    The VCL is defined in the following paper:
    
    J. Lin, A. Ziabari, S. V. Venkatakrishnan, O. Rahman, G. T. Buzzard, C. A.Bouman, "Tomographic Sparse View Selection
    using the View Covariance Loss", to appear in the IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

    Args:
        ct_model (TomographyModel): A CT model instance (e.g., ParallelBeamModel or ConeBeamModel) containing the system geometry and angles.
        reference_object (ndarray): 3D array representing the reference volume (e.g., ground truth).
        num_selected_views (int): Number of view angles to select.
        r_1 (float, optional): Voxel sampling rate in the reference object (default is 0.001).
        r_2 (float, optional): View sampling rate for stochastic minimization (default is 0.01).
        verbose (int, optional): Verbosity level. If > 0, visualizations of the covariance matrix and gamma vector will be shown.
        seed (int, optional): Random seed for deterministic behavior. If set, results will be reproducible.

    Returns:
        Tuple[ndarray, float]: A tuple containing:
            - A 1D NumPy array of the indices into the ct_model angles for the optimal view angles of shape (K,).
            - The scalar VCL value for the selected subset.

    Example:
        >>> angles = np.linspace(0, np.pi, num=180, endpoint=False)
        >>> sinogram_shape = (180, 128, 1)
        >>> ct_model = mj.ParallelBeamModel(sinogram_shape, angles)
        >>> ref_obj = np.random.rand(128, 128, 1)
        >>> selected_angles = get_opt_views(ct_model, ref_obj, num_selected_views=10)
        >>> print(selected_angles.shape)
        (10,)
    """
    num_views = ct_model.get_params('sinogram_shape')[0]
    angle_candidates = np.asarray(ct_model.get_params('angles'))
    recon_shape = ct_model.get_params('recon_shape')
    if recon_shape != reference_object.shape:
        raise ValueError("The recon shape from ct_model and reference_object.shape must match.\n Got ct_model recon_shape = {}, reference_shape = {}.".format(recon_shape, reference_object.shape))

    with tempfile.TemporaryDirectory() as data_store_dir:
        # Compute recon bases
        gamma = compute_view_basis_functions(ct_model, reference_object, r_1=r_1, data_store_dir=data_store_dir, seed=seed)

        # Compute inner product between recon bases
        R = compute_cov_matrix(num_views, data_store_dir)

    if verbose > 0:
        # plot the the covariance matrix and gamma
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(R)
        axes[0].set_title(r'$R$ View-Covariance Matrix')
        axes[1].imshow(np.linalg.inv(R))
        axes[1].set_title(r'$B=R^{-1}$ View-Precision Matrix')
        axes[2].plot(gamma)
        axes[2].set_ylim([0, 1])
        axes[2].set_title('Gamma Vector')
        plt.tight_layout()
        plt.show()

    # Compute optimal view angles
    optimal_angle_inds, vcl_value = compute_opt_angle_subset(R, gamma, angle_candidates, num_selected_views, r_2, seed=seed)

    return optimal_angle_inds, vcl_value



def compute_view_basis_functions(ct_model, ref_object, r_1, data_store_dir, seed=None):
    """
    Compute the view basis functions and inner product vector (gamma) used in the VCLS algorithm.

    Args:
        ct_model (TomographyModel): CT model specifying the system geometry.
        ref_object (ndarray): 3D reference object with shape (rows, cols, slices).
        r_1 (float): Voxel sampling rate in the reference object (fraction of total voxels).
        data_store_dir (str): Directory where the computed reconstructions will be stored as .npy files.
        seed (int, optional): Random seed for deterministic behavior. Default is None.

    Returns:
        ndarray: A 2D array of shape (num_views, 1) representing the gamma column vector.

    Example:
        >>> gamma = compute_view_basis_functions(ct_model, ref_object, 0.001, "/tmp/recons")
        >>> print(gamma.shape)
        (180, 1)
    """
    eps = 1e-12

    # Forward project the reference object
    print('Creating sinogram for reference object of shape {}'.format(ref_object.shape))
    ref_sino = ct_model.forward_project(ref_object)
    print('Done')

    # Create ROI mask and subsample the indices
    mask = mj.get_2d_ror_mask(ref_object[:, :, 0].shape)
    sparse_indices, row_col_indices = get_2d_subsampling_indices(mask, r_1, seed=seed)
    ref_object_flat = ref_object.reshape(ref_object.shape[0] * ref_object.shape[1], ref_object.shape[2])
    sparse_ref_object = jnp.asarray(ref_object_flat[sparse_indices, :])
    norm_sparse_ref = jnp.linalg.norm(sparse_ref_object)
    sparse_ref_object /= (norm_sparse_ref + eps)

    # Get number of views and angles
    num_views = ct_model.get_params('sinogram_shape')[0]

    print('Creating recon bases')

    # Filter the sinogram in a single call
    filtered_sinogram = ct_model.direct_filter(ref_sino, view_batch_size=None)
    del ref_sino  # Free up space in case the sino is large

    # Compute recon bases individually for each view
    gamma = np.zeros((num_views, 1))
    for i in tqdm.trange(num_views, desc='Computing and storing view basis functions'):
        view_sino = filtered_sinogram[i:i + 1]
        recon_i = ct_model.sparse_back_project(view_sino, sparse_indices, view_indices=jnp.array([i]))

        norm_i = jnp.linalg.norm(recon_i) + eps
        recon_i /= (norm_i + eps)

        # Save view basis function
        with open(os.path.join(data_store_dir, f'view_basis_function{i}.npy'), 'wb') as f:
            np.save(f, recon_i)

        gamma[i, 0] = jnp.tensordot(recon_i, sparse_ref_object)
    return gamma


def compute_cov_matrix(num_views, data_store_dir, batch_size=100):
    """
    Compute the covariance matrix of view basis functions in parallel.

    This function utilizes multiprocessing to efficiently compute the symmetric covariance matrix
    by loading precomputed basis functions stored as `.npy` files.

    Args:
        num_views (int): Total number of view basis functions.
        data_store_dir (str): Directory containing the stored `.npy` files for each view basis function.
        batch_size (int, optional): The number of views to use for computing the symmetric covariance matrix.  Defaults to 100.

    Returns:
        ndarray: A symmetric covariance matrix of shape `(num_views, num_views)`.

    Example:
        >>> cov_matrix = compute_cov_matrix(180, "/tmp/recons")
        >>> print(cov_matrix.shape)
        (180, 180)
    """

    cov_matrix = np.zeros((num_views, num_views))
    num_batches = int(np.ceil(num_views / batch_size))
    batches = np.array_split(np.arange(num_views), num_batches)
    recon0 = np.load(os.path.join(data_store_dir, f'view_basis_function{0}.npy'))
    recon_size = recon0.size

    for batch_index, batch in enumerate(tqdm.tqdm(batches, desc='Computing covariance matrix')):
        recons_batch = np.zeros((len(batch), recon_size))
        # Load the recons for the current batch
        for i in batch:
            recons_batch[i - batch[0]] = np.load(os.path.join(data_store_dir, f'view_basis_function{i}.npy')).flatten()
        # Find the inner products for the block diagonal for this batch
        recons_batch = jnp.array(recons_batch)
        batch_start, batch_stop = batch[0], batch[0] + len(batch)
        dot_products = recons_batch @ recons_batch.T
        cov_matrix[batch_start:batch_stop, batch_start:batch_stop] = dot_products

        # Loop over the higher index batches
        for batch2_index, batch2 in enumerate(batches[batch_index+1:]):
            # Load a batch
            recons_batch2 = np.zeros((len(batch2), recon_size))
            for j in batch2:
                recons_batch2[j - batch2[0]] = np.load(
                    os.path.join(data_store_dir, f'view_basis_function{j}.npy')).flatten()
            # Compute the inner product with the outer loop batch
            recons_batch2 = jnp.array(recons_batch2)
            batch2_start, batch2_stop = batch2[0], batch2[0] + len(batch2)
            dot_products = recons_batch @ recons_batch2.T

            # Store the inner product in the two symmetric blocks
            cov_matrix[batch_start:batch_stop, batch2_start:batch2_stop] = dot_products
            cov_matrix[batch2_start:batch2_stop, batch_start:batch_stop] = dot_products.T


    return cov_matrix


def compute_vcl(sub_R, sub_gamma):
    """
    Compute the View Correlation Loss (VCL) for a subset of views.

    This function evaluates the VCL metric, defined as:
        VCL = 1 - γᵀ R⁻¹ γ,
    where R is the submatrix of the covariance matrix and γ is the inner product vector.
    Lower values of VCL indicate a more informative and less redundant view subset.

    Args:
        sub_R (ndarray): A square (K, K) covariance matrix corresponding to a subset of K views.
        sub_gamma (ndarray): A column vector of shape (K, 1) representing the inner products
                              between the reconstruction bases and the reference object.

    Returns:
        float: The scalar VCL value for the selected subset of views.
    """
    loss_value = 1 - sub_gamma.T @ np.linalg.solve(sub_R, sub_gamma)
    return loss_value


def compute_opt_angle_subset(R, gamma, candidate_angles, K, r_2, search_min=30, max_iterations = 100, seed=None):
    """
    Select a subset of view angles that minimize the View Correlation Loss (VCL) using stochastic greedy optimization.

    This function performs an iterative stochastic search over candidate view indices to minimize the VCL,
    defined as VCL = -γᵀ R⁻¹ γ, where R is a covariance matrix of reconstructions and γ is the inner product vector.
    At each step, it considers random replacements of the current selection and keeps changes that improve the loss.

    Args:
        R (ndarray): Covariance matrix of shape (num_views, num_views).
        gamma (ndarray): Column vector of shape (num_views, 1), representing the inner product between reconstructions and reference.
        candidate_angles (ndarray): 1D array of view angles (shape (num_views,)) corresponding to R and gamma.
        K (int): Number of view angles to select.
        r_2 (float): Fraction of unchosen candidates to sample per view per iteration.
        search_min (int, optional): Minimum number of angles that are searched per iteration. Defaults to 30.
        max_iterations (int, optional): Maximum allowed number of iterations. Defaults to 100.
        seed (int, optional): Random seed for deterministic behavior. Default is None.

    Returns:
        Tuple[ndarray, float]: A tuple containing:
            - A 1D NumPy array of selected view angles of shape (K,).
            - The scalar VCL value for the selected subset.

    Example:
        >>> selected = compute_opt_angle_subset(R, gamma, candidate_angles, K=10, r_2=0.01)
        >>> print(selected.shape)
        (10,)

    Raises:
        ValueError: If K <= 0.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Determine the number of candidate views for the stochastic search
    num_candidate_angles = len(candidate_angles)
    num_unselected_angles = num_candidate_angles - K

    if K <= 0:
        raise ValueError("K must be positive. Received K={}".format(K))

    # If there are no available angles, just return the full set of angle candidates
    if num_unselected_angles <= 0:
        import warnings
        warnings.warn(f"Requested {K} views, but only {num_candidate_angles} available. Returning all candidates.")
        sorted_angle_inds = np.arange(len(candidate_angles))
        return sorted_angle_inds, float(compute_vcl(*subsample_R_gamma(R, gamma, sorted_angle_inds)))

    # Compute the number of candidates to search
    num_search_candidates = np.minimum(np.maximum(int(r_2 * num_unselected_angles), search_min), num_unselected_angles)

    # Initialize with uniformly spaced angles across candidate list.
    selected_angle_inds = np.linspace(0, num_candidate_angles, K, endpoint=False, dtype=int)

    # Subsample R and gamma to form smaller submatrix and subvector
    R_chosen, gamma_chosen = subsample_R_gamma(R, gamma, selected_angle_inds)

    # Compute the vcl loss
    vcl_current_best = compute_vcl(R_chosen, gamma_chosen)

    for i in range(max_iterations):
        prev_selected_angle_inds = np.copy(selected_angle_inds)
        for j in range(K):
            candidate_indices = np.setdiff1d(np.arange(num_candidate_angles), selected_angle_inds, assume_unique=True).tolist()
            random.shuffle(candidate_indices)
            candidate_indices = candidate_indices[:num_search_candidates]

            for k in candidate_indices:
                selected_angle_inds_tmp = np.copy(selected_angle_inds)
                selected_angle_inds_tmp[j] = k
                R_temp, gamma_temp = subsample_R_gamma(R, gamma, selected_angle_inds_tmp)
                vcl_temp = compute_vcl(R_temp, gamma_temp)

                if vcl_temp < vcl_current_best:
                    vcl_current_best = np.copy(vcl_temp)
                    selected_angle_inds = np.copy(selected_angle_inds_tmp)

        # Early stopping: exit if no change in selected angles during this iteration.
        if np.array_equal(selected_angle_inds, prev_selected_angle_inds):
            print(f'Early stopping at iteration {i}, no change in indices')
            break

    # Read-out and sort set of best angles
    best_view_angle_inds = np.sort(selected_angle_inds)
    return best_view_angle_inds, float(vcl_current_best)


def get_2d_subsampling_indices(mask, r_1, seed=None, blue_noise=False):
    """
    Perform 2D subsampling of voxel indices within a masked region.

    If `blue_noise` is False, then the function samples with a uniformly distributed random sampling pattern.
    Otherwise, a blue noise pattern is used to select points using a stored blue noise mask.
    However, the blue noise doesn't work with a ramp filtered signal, so it is probably not a good choice in this applications.

    Args:
        mask (ndarray): A 2D binary array indicating the region of interest (ROI).
        r_1 (float): Fraction of voxels to sample from the ROI. Must be in (0, 1].
        seed (int, optional): Random seed for reproducibility in random mode.
        blue_noise (bool, optional): If False (default), use uniform random sampling.
                                     If True, use a blue noise pattern for sampling.

    Returns:
        Tuple:
            random_indices_2d (jnp.ndarray): Flattened 1D array of selected voxel indices.
            (row_inds, col_inds) (Tuple[ndarray, ndarray]): Arrays of row and column indices
                corresponding to the selected voxels.
    """
    # Math is needed for ceiling operations used in tiling the blue noise pattern.
    import math

    # Validate that r_1 is a valid fraction.
    if r_1 <= 0 or r_1 > 1:
        raise ValueError("r_1 must be in the range (0, 1].")

    # Extract dimensions of the mask and compute number of samples to select.
    num_rows, num_cols = mask.shape
    mask_flat = mask.ravel()
    num_total = np.sum(mask_flat)
    num_samples = min(int(num_total * r_1), int(num_total))

    # Blue noise-based voxel sampling.
    if not blue_noise:
        # Uniform random voxel sampling.
        if seed is not None:
            np.random.seed(seed)
        # Identify eligible voxel indices from the flattened mask.
        eligible_indices = np.where(mask_flat > 0)[0]
        flat_indices = np.random.choice(eligible_indices, size=num_samples, replace=False)
    else:
        # Load the precomputed blue noise pattern from mbirjax.
        bn_pattern = mj.bn256
        # Determine how many times to tile the blue noise pattern to cover the mask.
        tile_rows = math.ceil(num_rows / bn_pattern.shape[0])
        tile_cols = math.ceil(num_cols / bn_pattern.shape[1])
        tiled_pattern = np.tile(bn_pattern, (tile_rows, tile_cols))
        tiled_pattern = tiled_pattern[:num_rows, :num_cols]

        # Mask out non-ROI regions with infinity to exclude them from sampling.
        masked_values = np.where(mask, tiled_pattern, np.inf)
        # Select the lowest blue noise values within the mask.
        flat_indices = np.argsort(masked_values.ravel())[:num_samples]

    # Convert flat indices back to 2D row/column indices and linear indices.
    row_inds, col_inds = np.unravel_index(flat_indices, (num_rows, num_cols))
    random_indices_2d = row_inds * num_cols + col_inds
    random_indices_2d = jnp.array(random_indices_2d)

    # Return the flattened and row/column indices.
    return random_indices_2d, (row_inds, col_inds)


def show_image_with_projection_rays(
    image: np.ndarray,
    *,
    rotation_angles_deg: np.ndarray = None,
    rotation_angles_rad: np.ndarray = None,
    title: str = None
) -> None:
    """
    Display an image and overlay arrows pointing along the projection from source to detector for the given 
    rotation angles.  The angles are rotation angles using the convention of mbirjax objects:  
    Looking down at the object with the detector at the top of the FoV, 0 degrees points from bottom to top of the 
    object.  As the rotation angle increases, the object rotates clockwise, which means that if the object is kept 
    in a fixed view, then the projection angle rotates counterclockwise.

    Exactly one of `rotation_angles_deg` or `rotation_angles_rad` must be provided (not both).

    Args:
        image (np.ndarray): A 2D NumPy array representing the image.
        rotation_angles_deg (np.ndarray, optional): A 1D array of angles in degrees. Each angle is visualized
            as an arrow through the image center.
        rotation_angles_rad (np.ndarray, optional): A 1D array of angles in radians. Each angle is visualized
            as an arrow through the image center.
        title (str, optional): Optional title to display above the plot.

    Returns:
        None
    """
    if image.ndim != 2:
        raise ValueError("Image must be a 2D array")
    if (rotation_angles_deg is None and rotation_angles_rad is None) or (rotation_angles_deg is not None and rotation_angles_rad is not None):
        raise ValueError("Exactly one of rotation_angles_deg or rotation_angles_rad must be None, and the other must be an array of floats")

    if rotation_angles_rad is None:
        rotation_angles_rad = np.deg2rad(rotation_angles_deg)

    # Convert from projection angles to angles in the standard representation
    rotation_angles_rad = np.pi / 2 + rotation_angles_rad

    rows, cols = image.shape
    center_x, center_y = cols / 2, rows / 2
    radius = min(rows, cols) / 2  # Use shortest dimension to ensure arrows fit within the image

    # Plot the image
    plt.imshow(image, cmap='gray', origin='upper', extent=[0, cols, rows, 0])
    plt.gca().set_aspect('equal')

    # Overlay arrows for each angle
    colors = plt.cm.tab10(np.arange(len(rotation_angles_rad)) % 10)

    for i, theta in enumerate(rotation_angles_rad):
        dx = 0.95*radius * np.cos(theta)
        dy = -0.95*radius * np.sin(theta)

        plt.arrow(center_x, center_y, dx, dy, color=colors[i], linewidth=1.75,
                  head_width=min(rows, cols) * 0.02, length_includes_head=True)

    plt.title(title or "Image with Overlaid Angles")
    plt.axis('off')
    plt.show()
