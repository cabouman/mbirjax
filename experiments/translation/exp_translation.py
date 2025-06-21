import numpy as np
from PIL import Image, ImageDraw, ImageFont

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


def gen_translation_vectors(num_x_translations, num_z_translations, x_spacing, z_spacing):
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


def gen_dot_phantom(recon_shape):
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



# --- 3D Text Phantom Generator ---
def gen_text_phantom(recon_shape, words, positions, font_path="DejaVuSans.ttf"):
    """
    Generate a 3D text phantom with binary word patterns embedded in specific slices.

    Args:
        recon_shape (tuple[int, int, int]): Shape of the phantom volume (num_rows, num_cols, num_slices).
        words (list[str]): List of ASCII words to render.
        positions (list[tuple[int, int, int]]): List of (row, col, slice) positions corresponding to each word.
        array_size (int, optional): Size of the 2D square image used to render each word. Default is 256.
        font_path (str, optional): Path to the TrueType font file. Default is "DejaVuSans.ttf".

    Returns:
        np.ndarray: A 3D numpy array of shape `recon_shape` containing the text phantom.
    """
    assert len(words) == len(positions), "Number of words must match number of positions."

    array_size = np.minimum(recon_shape[1], recon_shape[2])

    phantom = np.zeros(recon_shape, dtype=np.float32)
    try:
        font = ImageFont.truetype(font_path, size=20)
    except OSError:
        from pathlib import Path
        fallback_paths = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS fallback
            "/Library/Fonts/Arial.ttf",  # Additional macOS path
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux fallback
        ]
        for fallback in fallback_paths:
            if Path(fallback).exists():
                font = ImageFont.truetype(fallback, size=20)
                break
        else:
            raise FileNotFoundError(
                f"Could not find a usable font. Tried the following paths:\n"
                + "\n".join(fallback_paths)
                + "\nPlease install one of these fonts or specify a valid font_path."
            )

    for word, (r, c, s) in zip(words, positions):
        img = Image.new('L', (array_size, array_size), 0)
        draw = ImageDraw.Draw(img)

        text_box = draw.textbbox((0, 0), word, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]

        x = (array_size - text_width) // 2
        y = (array_size - text_height) // 2
        draw.text((x, y), word, fill=1, font=font)

        word_array = np.array(img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT))
        word_array = (word_array > 0).astype(np.float32)

        # Crop or pad word_array to fit in the recon volume
        r_start = max(0, r)
        r_end = min(recon_shape[0], r + 1)
        c_start = max(0, c - array_size // 2)
        c_end = min(recon_shape[1], c_start + array_size)
        s_start = max(0, s - array_size // 2)
        s_end = min(recon_shape[2], s_start + array_size)

        word_crop = word_array[:(c_end - c_start), :(s_end - s_start)]
        phantom[r_start:r_end, c_start:c_end, s_start:s_end] = word_crop

    return phantom



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
    x_spacing = 22
    z_spacing = 22

    # Set recon parameters
    sharpness = 0.0

    # Generate translation vectors
    translation_vectors = gen_translation_vectors(num_x_translations, num_z_translations, x_spacing, z_spacing)

    # Compute sinogram shape
    sino_shape = (translation_vectors.shape[0], num_det_rows, num_det_channels)

    # Initialize model for reconstruction.
    tct_model = mj.TranslationModel(sino_shape, translation_vectors, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)
    tct_model.set_params(sharpness=sharpness)
    recon_shape = tct_model.get_params('recon_shape')

    # Generate ground truth phantom
    words = ["Hello", "Goodbye"]
    positions = [
        (5, recon_shape[1] // 2, recon_shape[2] // 2),
        (10, recon_shape[1] // 2, recon_shape[2] // 2)
    ]
    gt_recon = gen_text_phantom(recon_shape, words, positions)

    # View test sample
    mj.slice_viewer(gt_recon.transpose(0, 2, 1), title='Ground Truth Recon', slice_label='View', slice_axis=0)

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
    recon, recon_params = tct_model.recon(sino, init_recon=0, weights=weights, max_iterations=20)

    # Display Results
    mj.slice_viewer(gt_recon.transpose(0, 2, 1), recon.transpose(0, 2, 1), vmin=0, vmax=1,
                    title='Object (left) vs. MBIR reconstruction (right)', slice_axis=0)


if __name__ == '__main__':
    main()
