import jax
import jax.numpy as jnp
import numpy as np

def create_reference_image(option, size=64, sigma=4.0):
    """
    Create a reference image based on the selected option

    Args:
        size (integer): the size of the test image
        sigma (float): the standard deviation of the Gaussian
        option (str): Image type to generate. Options are 'gaussian' or 'constant'.

    Returns:
        np.ndarray: Generated reference image.
    """
    if option == 'gaussian':
        return create_gaussian_square_image(size=size, sigma=sigma)
    elif option == 'constant':
        return create_constant_square_image(size=size)
    else:
        raise ValueError(f"Unsupported image option: {option}")


def create_gaussian_square_image(size=64, sigma=4.0):
    """
    Create an image with a 2D Gaussian centered in a square region.

    Args:
        size (integer): the size of the test image
        sigma (float): the standard deviation of the Gaussian

    Returns:
        np.ndarray: (size, size, 1)
    """
    image = np.zeros((size, size), dtype=np.float32)

    # Define square region
    square_size = size // 3
    x_start = (size - square_size) // 2
    x_end = x_start + square_size
    y_start = (size - square_size) // 2
    y_end = y_start + square_size

    # Create 2D Gaussian grid
    x = np.linspace(-1, 1, square_size)
    y = np.linspace(-1, 1, square_size)
    xv, yv = np.meshgrid(x, y)
    gaussian = np.exp(-(xv ** 2 + yv ** 2) / (2 * (sigma / 10.0) ** 2))  # normalize sigma to [0, 1] scale

    # Normalize to [0, 1]
    gaussian = (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min())

    # Insert into image
    image[x_start:x_end, y_start:y_end] = gaussian

    return image[..., None]


def create_constant_square_image(size=64):
    """
    Create a test image containing a white square.

    Args:
        size (integer): the size of the test image

    Returns:
        jax.array: Generated test image
    """
    image = np.zeros((size, size))

    # Define square region
    square_size = size // 3
    x_start = (size - square_size) // 2
    x_end = x_start + square_size
    y_start = (size - square_size) // 2
    y_end = y_start + square_size

    image[x_start:x_end, y_start:y_end] = 1.0
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
    translated_image = jax.image.scale_and_translate(original_image,
                                                     shape=original_image.shape,
                                                     spatial_dims=(0, 1),
                                                     scale=jnp.array([1.0, 1.0]),
                                                     translation=jnp.array([dy, dx]),
                                                     method="linear",
                                                     antialias=False)
    return translated_image


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
    adjusted_image = jax.image.scale_and_translate(translated_image,
                                                   shape=translated_image.shape,
                                                   spatial_dims=(0, 1),
                                                   scale=jnp.array([1.0, 1.0]),
                                                   translation=jnp.array([-dy, -dx]),
                                                   method="linear",
                                                   antialias=False)
    return jnp.mean((adjusted_image - original_image) ** 2)
