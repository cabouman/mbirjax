import jax
import jax.numpy as jnp
import dm_pix as pix
import matplotlib.pyplot as plt
import jax.scipy.optimize
import mbirjax as mj
from img_registration_utils import *


def main():
    # Define the size and type of test images
    size = 64
    image_type = "gaussian"  # Can be "gaussian" or "constant"

    # Define the ground truth shift in x and y axis
    true_dy, true_dx = 5.0, -3.0

    # Initialize the shift parameters
    initial_shift = jnp.array([0.0, 0.0])

    # Generate the reference image and the image to be aligned with it
    reference_image = create_reference_image(size=size, option=image_type)
    translated_image = apply_translation(reference_image, true_dy, true_dx)

    # Visualize the fixed and moving images
    mj.slice_viewer(reference_image, translated_image, title='Reference Image (Left) and Translated Image (Right)', slice_axis=2)

    # Test gradients
    test_shift = jnp.array([0.0, 0.0])
    grads = jax.grad(loss_fn)(test_shift, reference_image, translated_image)
    print("===============Test Gradients============")
    print("Gradient of loss function with respect to shift:", grads)

    # Compute the optimization using scipy.optimize.minimize
    result = jax.scipy.optimize.minimize(
        fun=loss_fn,
        x0=initial_shift,
        args=(reference_image, translated_image),
        method='BFGS'
    )

    shift = result.x
    print("\n===============Optimization Results=============")
    print(f"Optimization status: {result.status}")
    print(f"Success: {result.success}")
    print(f"Number of iterations: {result.nit}")
    print(f"Estimated shift: dx = {shift[1]:.2f}, dy = {shift[0]:.2f}")
    print(f"True shift: dx = {true_dx}, dy = {true_dy}")
    print(f"Final loss: {result.fun:.6f}")

    # Translate the moving image using final estimated shift
    final_offset = jnp.array([-shift[0], -shift[1], 0.0])
    registered_image = pix.affine_transform(translated_image, jnp.eye(3), offset=final_offset, order=1)

    # Show registered image vs. original
    mj.slice_viewer(reference_image, registered_image, title='Reference Image (Left) and Registered Image (Right)', slice_axis=2)


if __name__ == '__main__':
    main()