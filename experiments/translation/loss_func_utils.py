import jax
import jax.numpy as jnp
import numpy as np


def loss_fn(shifts_flatten, original_image, translated_image):
    """
    Compute the MSE between the fixed image and the moving image.
    Args:
        shifts_flatten (jax.Array): Flattened shifts array of shape (num_slices * 2,).
        original_image (jax.array): Fixed image with shape (num_slices, num_rows, num_columns), assumed to be the reference position
        translated_image (jax.array): the image to be aligned with shape (num_slices, num_rows, num_columns), assumed to be the shifted version of the fixed image

    Returns:
        jax.array: MSE between image_fixed and image_moving
    """
    num_slices = original_image.shape[0]
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

    per_slice_mse = jax.vmap(loss_per_slice, in_axes=(0, 0, 0))(shifts, original_image, translated_image)
    return jnp.sum(per_slice_mse)