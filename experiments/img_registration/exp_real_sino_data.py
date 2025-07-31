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
import mbirjax.preprocess as mjp
import h5py
import optax
import time
from img_registration_utils import *


def main():
    # Define the size and type of test images
    dataset_url = '/depot/bouman/data/nersc/demo_nersc_permafrost.tgz'
    num_slices = 4

    # Define the maximum shift applied to the reference image
    maximum_shift = 5

    # Define the optimization parameters
    num_iterations = 100
    learning_rate = 0.7

    # Define the ground truth shift in x and y axis
    true_shifts = np.random.randint(-maximum_shift, maximum_shift, (num_slices, 2)).astype(np.float32)
    print("===============True Shifts============")
    for i in range(num_slices):
        print(f"Slice {i}: dy = {true_shifts[i, 0]}, dx = {true_shifts[i, 1]}")

    # Initialize the shift parameters
    initial_shift = jnp.zeros((num_slices, 2))

    # Download data
    download_dir = '/home/yang1581/Github/mbirjax_applications/nersc/demo_data/'
    dataset_dir = mj.download_and_extract(dataset_url, download_dir)

    # Load reconstruction parameters from data.
    with h5py.File(dataset_dir, "r") as data:
        # Get pixel size in units of cm
        pixel_size = data['/measurement/instrument/detector/pixel_size'][0] / 10.0
        angles = -np.deg2rad(data['exchange/theta'])
        obj_scan = data['exchange/data'][:]
        blank_scan = data['exchange/data_white'][:]
        dark_scan = data['exchange/data_dark'][:]

    # Select out a typically small number of slices from the important region of the sample.
    num_views, num_det_rows, num_det_channels = obj_scan.shape
    num_slices = np.minimum(num_det_rows, num_slices)
    crop_pixels_top = (num_det_rows - num_slices) // 2
    crop_pixels_bottom = (num_det_rows - num_slices) // 2


    print("\n********** Crop out desired rows from each view **************")
    obj_scan, blank_scan, dark_scan, _ = mjp.crop_view_data(
        obj_scan, blank_scan, dark_scan,
        crop_pixels_sides=0, crop_pixels_top=crop_pixels_top, crop_pixels_bottom=crop_pixels_bottom,
        defective_pixel_array=()
    )

    print("\n********** Compute sinogram **************")
    sino = mjp.compute_sino_transmission(obj_scan, blank_scan, dark_scan)
    sino = sino.transpose((1, 0, 2))
    sino = jnp.array(sino)

    # Generate the shifted sinogram
    translated_sino = apply_translation(sino, true_shifts)

    # Visualize the fixed and moving sinogram
    mj.slice_viewer(sino, translated_sino, title='Reference Sinogram (Left) and Translated Sinogram (Right)', slice_axis=0)

    # Test gradients
    test_shift = jnp.zeros((num_slices, 2)).flatten()
    grads = jax.grad(loss_fn)(test_shift, sino, translated_sino)
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
        loss = loss_fn(params, sino, translated_sino)
        gradients = grad_fn(params, sino, translated_sino)
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