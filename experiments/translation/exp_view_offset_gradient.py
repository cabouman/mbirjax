# This script is for reconstructing Translation CT data from Zeiss scanner

import mbirjax as mj
import mbirjax.preprocess as mjp
import numpy as np
import os
import pprint
import sys
import subprocess
import importlib.util
import optax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from loss_func_utils import *

def main():
    # Download and extract data
    download_dir = "./purdue_p_data"
    dataset_url = "/depot/bouman/data/Translation/purdue_p_xrm.tgz"
    dataset_dir = mj.download_and_extract(dataset_url, download_dir)

    # Define the number of epoch
    num_epochs = 100

    # Define the optimization parameters
    num_iterations = 50
    learning_rate = 0.01

    # Load and preprocess data
    sino, translation_params, optional_params = mjp.zeiss.compute_sino_and_params(dataset_dir, crop_pixels_bottom=53, verbose=0)

    translation_vectors = translation_params["translation_vectors"]
    for epoch in range(num_epochs):
        print(f"\n===============Epoch {epoch}============")
        # Initialize model for reconstruction.
        translation_params["translation_vectors"] = translation_vectors
        tct_model = mj.TranslationModel(**translation_params)
        tct_model.set_params(**optional_params)
        tct_model.set_params(sharpness=0.0)
        # recon_shape = tct_model.get_params('recon_shape')

        # View sinogram
        # mj.slice_viewer(sino, slice_axis=0, title='Original sinogram', slice_label='View')

        # Generate weights array - for an initial reconstruction, use weights = None, then modify if needed.
        weights = None

        # Perform MBIR reconstruction
        recon, recon_params = tct_model.recon(sino, init_recon=0, weights=weights, max_iterations=80)

        # Save reconstruction results
        output_path = './output/'  # path to store output recon
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, f'purdue_p_recon.h5')
        mj.export_recon_hdf5(output_path, recon, recon_dict=recon_params, top_margin=0, bottom_margin=0)

        sino_from_recon = tct_model.forward_project(recon)
        mj.slice_viewer(sino, sino_from_recon, title=f'Original sinogram and forward projected recon (Epoch {epoch})', slice_axis=0)
        sino_scale = sino * 100
        sino_from_recon_scale = sino_from_recon * 100

        # Initialize the shift parameters
        initial_shift = jnp.zeros((sino.shape[0], 2))

        # Perform the optimization using optax
        params = initial_shift.flatten()
        optimizer = optax.adam(learning_rate=learning_rate)
        opt_state = optimizer.init(params)

        grad_fn = jax.grad(loss_fn)

        min_loss = jnp.inf
        best_params = params
        loss_val = []

        print(f"\n===============Starting Optimization=============")
        for step in range(num_iterations):
            loss = loss_fn(params, sino_scale, sino_from_recon_scale)
            loss_val.append(loss)
            gradients = grad_fn(params, sino_scale, sino_from_recon_scale)
            updates, opt_state = optimizer.update(gradients, opt_state)
            params = optax.apply_updates(params, updates)

            if step % 10 == 0:
                print(f"Iteration {step}, loss = {loss}")

            if loss < min_loss:
                min_loss = loss
                best_params = params

        shifts_estimated = best_params.reshape((sino.shape[0], 2))
        print("\n===============Estimated Shifts============")
        for view in range(sino.shape[0]):
            print(f"Slice {view}: estimated [dy={shifts_estimated[view, 0]}, dx={shifts_estimated[view, 1]}]")

        print("Translation vectors: ", translation_vectors)
        translation_vectors[:, 0] = translation_vectors[:, 0] + shifts_estimated[:, 1]
        translation_vectors[:, 2] = translation_vectors[:, 2] + shifts_estimated[:, 0]
        print("Translation vectors: ", translation_vectors)

        plt.semilogy(range(num_iterations), loss_val)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title(f"Gradient Descent Convergence (Loss vs Iterations) Epoch {epoch}")
        plt.grid(True)
        plt.show()

        # Display Results
        mj.slice_viewer(recon.transpose(0, 2, 1), title=f'MBIR reconstruction (Epoch {epoch})', slice_axis=0)



if __name__ == '__main__':
    main()
