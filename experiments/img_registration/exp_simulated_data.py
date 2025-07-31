import sys
import importlib.util
import subprocess

# Install necessary packages
required_packages = ['optax']
for package in required_packages:
    if importlib.util.find_spec(package) is None:
        print(f"{package} not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}. Please install it manually.")
            sys.exit(1)
    else:
        print(f"{package} is already installed.")

import jax
import jax.numpy as jnp
import jax.scipy.optimize
import mbirjax as mj
import time
import optax
from img_registration_utils import *


def main():
    # Define the size and type of test images
    size = 64
    num_slices = 5
    image_type = "gaussian" # Can be "gaussian" or "constant"

    # Define the maximum shift applied to the reference image
    maximum_shift = 5

    # Define the optimization parameters
    num_iterations = 10
    learning_rate = 0.7

    # Define the ground truth shift in x and y axis
    true_shifts = np.random.randint(-maximum_shift, maximum_shift, (num_slices, 2)).astype(np.float32)
    print("===============True Shifts============")
    for i in range(num_slices):
        print(f"Slice {i}: dy = {true_shifts[i, 0]}, dx = {true_shifts[i, 1]}")

    # Initialize the shift parameters
    initial_shift = jnp.zeros((num_slices, 2))

    # Generate the reference image and the image to be aligned with it
    reference_image = create_reference_image(option=image_type, size=size, num_slices=num_slices)
    translated_image = apply_translation(reference_image, true_shifts)

    # Visualize the fixed and moving images
    mj.slice_viewer(reference_image, translated_image, title='Reference Image (Left) and Translated Image (Right)', slice_axis=0)

    # Test gradients
    test_shift = jnp.zeros((num_slices, 2)).flatten()
    grads = jax.grad(loss_fn)(test_shift, reference_image, translated_image)
    print(f"\n===============Test Gradients============")
    print("Gradient of loss function with respect to shift:", grads)

    # Perform the optimization using optax
    print(f"\n===============Starting Optimization=============")
    time0 = time.time()
    params = initial_shift.flatten()
    optimizer = optax.rmsprop(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    grad_fn = jax.grad(loss_fn)

    min_loss = jnp.inf
    best_params = params

    for step in range(num_iterations):
        loss = loss_fn(params, reference_image, translated_image)
        gradients = grad_fn(params, reference_image, translated_image)
        updates, opt_state = optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)

        if loss < min_loss:
            min_loss = loss
            best_params = params

    elapsed = time.time() - time0
    print(f'Elapsed time for optimization is {elapsed:.3f} seconds')

    print(f"\n===============Final Estimated Shift============")
    shifts_estimated = best_params.reshape((num_slices, 2))
    for i in range(num_slices):
        print(f"Slice {i}: estimated [dy={shifts_estimated[i, 0]:.3f}, dx={shifts_estimated[i, 1]:.3f}] | "
              f"true [dy={true_shifts[i, 0]}, dx={true_shifts[i, 1]}]")


if __name__ == '__main__':
    main()