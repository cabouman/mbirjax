import jax
import jax.numpy as jnp


def create_test_image(size=64):
    """
    Create a test image containing a white square.

    Args:
        size (integer): the size of the test image

    Returns:
        jax.array: Generated test image
    """
    image = jnp.zeros((size, size))
    image = image.at[20:40, 20:40].set(1.0)
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
