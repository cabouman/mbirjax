import jax
import jax.numpy as jnp
import numpy as np

def create_reference_image(option, size=64, sigma=4.0, num_slices=1):
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
        return create_gaussian_square_image(size=size, sigma=sigma, num_slices=num_slices)
    elif option == 'constant':
        return create_constant_square_image(size=size, num_slices=num_slices)
    else:
        raise ValueError(f"Unsupported image option: {option}")


def create_gaussian_square_image(size=64, sigma=4.0, num_slices=1):
    """
    Create an image with a 2D Gaussian centered in a square region.

    Args:
        size (integer): the size of the test image
        sigma (float): the standard deviation of the Gaussian
        num_slices (int): number of slices

    Returns:
        np.ndarray: (size, size, 1)
    """
    gaussian_volume = np.zeros((num_slices, size, size), dtype=np.float32)

    for slice_idx in range(num_slices):
        image = np.zeros((size, size), dtype=np.float32)

        # Add slight variation to each slice
        slice_sigma = sigma * (0.8 + 0.4 * slice_idx / max(1, num_slices - 1))

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
        gaussian = np.exp(-(xv ** 2 + yv ** 2) / (2 * (slice_sigma / 10.0) ** 2))

        # Normalize to [0, 1]
        gaussian = (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min())

        # Insert into image
        image[x_start:x_end, y_start:y_end] = gaussian
        gaussian_volume[slice_idx] = image

    return jnp.array(gaussian_volume)


def create_constant_square_image(size=64, num_slices=1):
    """
    Create a test image containing a white square.

    Args:
        size (integer): the size of the test image
        num_slices (int): number of slices

    Returns:
        jax.array: Generated test image
    """
    square_volume = np.zeros((num_slices, size, size))

    for slice_idx in range(num_slices):
        image = np.zeros((size, size))

        # Define square region
        square_size = size // 3
        x_start = (size - square_size) // 2
        x_end = x_start + square_size
        y_start = (size - square_size) // 2
        y_end = y_start + square_size

        image[x_start:x_end, y_start:y_end] = 1.0
        square_volume[slice_idx] = image

    return jnp.array(square_volume)


def shift_image(imgs, shifts):
    """
    Apply a translations to an array of images using jax.image.scale_and_translate.

    Args:
        imgs (jax.array): Image to be transformed with size (num_slices, num_rows, num_columns)
        shifts (jax.Array): 2D array of shape (num_slices, 2) containing (vertical shift, horizontal shift) for each slice

    Returns:
        jax.array: shifted image
    """
    def translate_slice(img_slice, shift):
        dy, dx = shift
        shifted_imgs = jax.image.scale_and_translate(img_slice,
                                                     shape=img_slice.shape,
                                                     spatial_dims=(0, 1),
                                                     scale=jnp.array([1.0, 1.0]),
                                                     translation=jnp.array([dy, dx]),
                                                     method="linear",
                                                     antialias=False)
        return shifted_imgs

    return jax.vmap(translate_slice, in_axes=(0, 0))(imgs, shifts)


def loss_fn(shifts_flatten, reference_imgs, shifted_imgs):
    """
    Compute the MSE between an array of reference images and an array of shifted image.
    
    Args:
        shifts_flatten (jax.Array): Flattened shifts array of shape (num_slices * 2,).
        reference_imgs (jax.array): Array of reference images with shape (num_slices, num_rows, num_columns)
        shifted_imgs (jax.array): Array of shifted images with shape (num_slices, num_rows, num_columns)

    Returns:
        jax.array: MSE between image_fixed and image_moving
    """
    num_slices = reference_imgs.shape[0]
    shifts = shifts_flatten.reshape((num_slices, 2))

    def loss_per_slice(shift, original_slice, translated_slice):
        dy, dx = shift
        adjusted_slice = jax.image.scale_and_translate(translated_slice,
                                                       shape=translated_slice.shape,
                                                       spatial_dims=(0, 1),
                                                       scale=jnp.array([1.0, 1.0]),
                                                       translation=jnp.array([-dy, -dx]),
                                                       method="linear",
                                                       antialias=False)
        return jnp.mean((adjusted_slice - original_slice) ** 2)

    per_slice_mse = jax.vmap(loss_per_slice, in_axes=(0, 0, 0))(shifts, reference_imgs, shifted_imgs)
    return jnp.sum(per_slice_mse)