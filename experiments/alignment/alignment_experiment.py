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
from alignment_utils import *


def main():
    # Define the size and type of test images
    size = 64
    num_slices = 8
    image_type = "gaussian" # Can be "gaussian" or "constant"

    # Define the maximum shift applied to the reference image
    maximum_shift = 10

    # Define the optimization parameters
    num_iterations = 10
    learning_rate = 1.0

    # Define the ground truth shift in x and y axis
    # np.random.seed(42)
    true_shifts = np.random.uniform(-maximum_shift, maximum_shift, (num_slices, 2)).astype(np.float32)
    print("===============Ground Truth Shifts============")
    for i in range(num_slices):
        # dy means vertical shift. Positive dy shifts image down.
        # dx means horizontal shift. PPositive dx shifts image right.
        print(f"Slice {i}: dy = {true_shifts[i, 0]:.3f}, dx = {true_shifts[i, 1]:.3f}")

    # Generate the reference image and the shifted image
    print("\n===============Create synthetic Phantom and its shifted verison============")
    reference_image = create_reference_image(option=image_type, size=size, num_slices=num_slices)
    shifted_image = shift_image(reference_image, true_shifts)

    # Visualize the fixed and shifted images
    mj.slice_viewer(reference_image, shifted_image, title='Synthetic Reference Image (Left) and Shifted Image (Right)', slice_axis=0)

    # Perform the gradient descent optimization using optax
    print(f"\n===============Starting Gradient Descent Optimization=============")
    time0 = time.time()
    initial_shift = jnp.zeros((num_slices, 2))
    params = initial_shift.flatten()
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    grad_fn = jax.grad(loss_fn)

    min_loss = jnp.inf
    best_params = params
    loss_val = []

    for step in range(num_iterations):
        loss = loss_fn(params, reference_image, shifted_image)
        loss_val.append(loss)
        gradients = grad_fn(params, reference_image, shifted_image)
        updates, opt_state = optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)

        # if step % 10 == 0:
        #     print(f"Iteration {step}, loss = {loss}")

        if loss < min_loss:
            min_loss = loss
            best_params = params

    elapsed = time.time() - time0
    print(f'Elapsed time for optimization is {elapsed:.3f} seconds')
    print(f'Number of iterations: {num_iterations}')

    print(f"\n===============Final Estimated Shift============")
    shifts_estimated = best_params.reshape((num_slices, 2))
    for i in range(num_slices):
        print(f"\nSlice {i}: "
              f"\nEstimated shift: [dy={shifts_estimated[i, 0]:.3f}, dx={shifts_estimated[i, 1]:.3f}] | "
              f"\nGround truth shift: [dy={true_shifts[i, 0]:.3f}, dx={true_shifts[i, 1]:.3f}] | "
              f"\nDifference between estimated shift and ground truth shift: [dy={np.abs(shifts_estimated[i, 0] - true_shifts[i, 0]):.3f}, dx={np.abs(shifts_estimated[i, 1] - true_shifts[i, 1]):.3f}]")

    # Correct the shifted images
    shifts_estimated = -shifts_estimated
    corrected_image = shift_image(shifted_image, shifts_estimated)

    # Visualize the reference image and corrected image
    mj.slice_viewer(reference_image, corrected_image, title='Synthetic Reference Image (Left) and Corrected Image (Right)', slice_axis=0)

if __name__ == '__main__':
    main()