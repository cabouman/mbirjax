import jax
import jax.numpy as jnp
import dm_pix as pix
import numpy as np

def create_test_image(size=64, white_ratio=0.2, seed=0):
    """
    Create a test image with some random points in a certain region

    Args:
        size (int): Size of the square image.
        white_ratio (float): Fraction of pixels in the square region to set to 1.0.
        seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Generated test image with shape (size, size, 1)
    """
    # Initialize a black image
    image = np.zeros((size, size), dtype=np.float32)

    # Define white square region
    x_start, x_end = 20, 40
    y_start, y_end = 20, 40
    square_h, square_w = x_end - x_start, y_end - y_start
    total_square_pixels = square_h * square_w
    num_white_pixels = int(total_square_pixels * white_ratio)

    # Generate random coordinates within the square
    rng = np.random.default_rng(seed)
    indices = rng.choice(total_square_pixels, size=num_white_pixels, replace=False)
    coords = np.unravel_index(indices, (square_h, square_w))

    # Set selected pixels to 1.0
    image[x_start + coords[0], y_start + coords[1]] = 1.0

    return image[..., None]


def apply_translation(original_image, dy, dx):
    """
    Apply a translation on an image using dm_pix.affine_transform
    Args:
        original_image (jax.array): Image to be transformed
        dx (float): translation along x_axis
        dy (float): translation along y_axis

    Returns:
        jax.array: Transformed image
    """
    matrix = jnp.eye(3)
    offset = jnp.array([dy, dx, 0.0])
    return pix.affine_transform(original_image, matrix, offset=offset, order=1)


def loss_fn(shift, original_image, translated_image):
    """
    Compute the MSE between the fixed image and the moving image.
    Args:
        shift (jax.array): Shift of the image in x and y axis
        original_image (jax.array): Fixed image, assumed to be the reference position
        translated_image (jax.array): the image to be aligned, assumed to be the shifted version of the fixed image

    Returns:
        jax.array: MSE between image_fixed and image_moving
    """
    dy, dx = shift
    offset = jnp.array([-dy, -dx, 0.0])
    matrix = jnp.eye(3)
    adjusted_image = pix.affine_transform(translated_image, matrix, offset=offset, order=1)
    return jnp.mean((original_image - adjusted_image) ** 2)