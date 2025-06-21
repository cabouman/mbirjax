import numpy as np
import mbirjax as mj
import matplotlib.pyplot as plt


# Utility function to display translation vectors' x and z components using a scatter plot
def display_translation_vectors(translation_vectors):
    """Display the x and z components of translation vectors using a scatter plot.

    Args:
        translation_vectors (np.ndarray): Array of shape (N, 3) containing [dx, dy, dz] vectors.
    """
    dx = translation_vectors[:, 0]
    dz = translation_vectors[:, 2]

    plt.figure(figsize=(6, 6))
    plt.scatter(dx, dz, c='blue', marker='o')
    plt.title("Translation Grid (dx vs dz)")
    plt.xlabel("dx")
    plt.ylabel("dz")
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def generate_translation_vectors(num_x_translations, num_z_translations, x_spacing, z_spacing):
    """
    Generate translation vectors for lateral (x) and axial (z) displacements.

    Args:
        num_x_translations (int): Number of x-direction translations
        num_z_translations (int): Number of z-direction translations
        x_spacing (float): Spacing between x translations in ALU
        z_spacing (float): Spacing between z translations in ALU

    Returns:
        np.ndarray: Array of shape (num_views, 3) with translation vectors [dx, dy, dz]
    """
    num_views = num_x_translations * num_z_translations
    translation_vectors = np.zeros((num_views, 3))

    x_center = (num_x_translations - 1) / 2
    z_center = (num_z_translations - 1) / 2

    idx = 0
    for row in range(num_z_translations):
        for col in range(num_x_translations):
            dx = (col - x_center) * x_spacing
            dz = (row - z_center) * z_spacing
            dy = 0
            translation_vectors[idx] = [dx, dy, dz]
            idx += 1

    return translation_vectors


def generate_ground_truth_recon(recon_shape):
    """
    Generate a synthetic ground truth reconstruction volume.

    Args:
        recon_shape (tuple[int, int, int]): Shape of the reconstruction volume.

    Returns:
        np.ndarray: Ground truth reconstruction volume with sparse binary features.
    """
    np.random.seed(42)
    gt_recon = np.zeros(recon_shape, dtype=np.float32)
    fill_rate = 0.05

    y_pad = recon_shape[0] // 6
    central_start = y_pad
    central_end = recon_shape[0] - y_pad

    row_size = recon_shape[1] * recon_shape[2]
    num_ones_per_row = int(row_size * fill_rate)

    for row_idx in range(central_start, central_end):
        flat_row = gt_recon[row_idx].flatten()
        positions_ones = np.random.choice(row_size, num_ones_per_row, replace=False)
        flat_row[positions_ones] = 1.0
        gt_recon[row_idx] = flat_row.reshape(recon_shape[1:])

    return gt_recon


def main():
    # Define geometry
    source_iso_dist = 64
    source_detector_dist = 64

    # Define detector size
    num_det_rows = 128
    num_det_channels = 128

    # Define view sampling parameters
    num_x_translations = 7
    num_z_translations = 7
    x_spacing = 32
    z_spacing = 32

    # Set recon parameters
    sharpness = 0.0

    # Generate translation vectors
    translation_vectors = generate_translation_vectors(num_x_translations, num_z_translations, x_spacing, z_spacing)

    # Compute sinogram shape
    sino_shape = (translation_vectors.shape[0], num_det_rows, num_det_channels)

    # Initialize model for reconstruction.
    tct_model = mj.TranslationModel(sino_shape, translation_vectors, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)
    tct_model.set_params(sharpness=sharpness)

    # Generate ground truth phantom
    recon_shape = tct_model.get_params('recon_shape')
    gt_recon = generate_ground_truth_recon(recon_shape)

    # View test sample
    mj.slice_viewer(gt_recon, title='Ground Truth Recon', slice_label='View', slice_axis=0)

    # Generate synthetic sonogram data
    sino = tct_model.forward_project(gt_recon)

    # View sinogram
    mj.slice_viewer(sino, slice_axis=0, vmin=0, vmax=1, title='Original sinogram', slice_label='View')

    # Generate weights array - for an initial reconstruction, use weights = None, then modify if needed.
    weights = None

    # Print model parameters
    tct_model.print_params()
    # Display translation array
    display_translation_vectors(translation_vectors)

    # Perform MBIR reconstruction
    recon, recon_params = tct_model.recon(sino, init_recon=0, weights=weights)

    # Display Results
    mj.slice_viewer(gt_recon, recon, vmin=0, vmax=1, title='Object (left) vs. MBIR reconstruction (right)', slice_axis=0)


if __name__ == '__main__':
    main()
