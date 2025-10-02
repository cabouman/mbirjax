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
import matplotlib.pyplot as plt
from img_registration_utils import *
from skimage.registration import phase_cross_correlation


def main():
    # Define the size and type of test images
    size = 64
    num_slices = 8
    image_type = "gaussian" # Can be "gaussian" or "constant"

    # Define the maximum shift applied to the reference image
    maximum_shift = 10

    # Define the optimization parameters
    num_iterations = 150
    learning_rate = 0.1

    # Define the ground truth shift in x and y axis
    np.random.seed(42)
    true_shifts = np.random.uniform(-maximum_shift, maximum_shift, (num_slices, 2)).astype(np.float32)
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
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    grad_fn = jax.grad(loss_fn)

    min_loss = jnp.inf
    best_params = params
    loss_val = []

    for step in range(num_iterations):
        loss = loss_fn(params, reference_image, translated_image)
        loss_val.append(loss)
        gradients = grad_fn(params, reference_image, translated_image)
        updates, opt_state = optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)

        if step % 10 == 0:
            print(f"Iteration {step}, loss = {loss}")

        if loss < min_loss:
            min_loss = loss
            best_params = params

    elapsed = time.time() - time0
    print(f'Elapsed time for optimization is {elapsed:.3f} seconds')

    print(f"\n===============Final Estimated Shift============")
    shifts_estimated = best_params.reshape((num_slices, 2))
    for i in range(num_slices):
        print(f"Slice {i}: estimated [dy={shifts_estimated[i, 0]:.3f}, dx={shifts_estimated[i, 1]:.3f}] | "
              f"true [dy={true_shifts[i, 0]:.3f}, dx={true_shifts[i, 1]:.3f}] | "
              f"error [dy={np.abs(shifts_estimated[i, 0] - true_shifts[i, 0]):.3f}, dx={np.abs(shifts_estimated[i, 1] - true_shifts[i, 1]):.3f}]")

    plt.semilogy(range(num_iterations), loss_val)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Gradient Descent Convergence (Loss vs Iterations)")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()