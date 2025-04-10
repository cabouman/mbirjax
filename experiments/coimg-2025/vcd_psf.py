import os
import numpy as np
import jax.numpy as jnp
import jax
import time
import matplotlib.pyplot as plt
import gc
import mbirjax.parallel_beam
import vcd_coimg_utils

if __name__ == "__main__":
    """
    Investigate the psf
    """
    # ##########################
    # Do all the setup

    # Initialize sinogram
    num_views = 128
    num_det_rows = 20
    num_det_channels = 128
    start_angle = 0
    end_angle = jnp.pi
    sinogram = jnp.zeros((num_views, num_det_rows, num_det_channels))
    angles = jnp.linspace(start_angle, jnp.pi, num_views, endpoint=False)

    # Initialize a random key
    seed_value = np.random.randint(1000000)
    key = jax.random.PRNGKey(seed_value)

    # Set up parallel beam model
    # parallel_model = mbirjax.ParallelBeamModel.from_file('params_parallel.yaml')
    parallel_model = mbirjax.ParallelBeamModel(sinogram.shape, angles)
    # parallel_model.to_file('params_parallel.yaml')

    # Generate phantom
    recon_shape = parallel_model.get_params('recon_shape')
    num_recon_rows, num_recon_cols, num_recon_slices = recon_shape[:3]
    phantom = mbirjax.gen_cube_phantom(recon_shape)

    # Generate indices of pixels
    num_subsets = 1
    full_indices = mbirjax.gen_pixel_partition(recon_shape, num_subsets)
    num_subsets = 5
    subset_indices = mbirjax.gen_pixel_partition(recon_shape, num_subsets)

    # Get the vector of indices
    indices = jnp.arange(num_recon_rows * num_recon_cols)
    num_trials = 3
    indices = jnp.mod(np.arange(num_trials, dtype=int).reshape((-1, 1)) + indices.reshape((1, -1)), num_recon_rows * num_recon_cols)

    sinogram = jnp.array(sinogram)
    indices = jnp.array(indices)

    # ##########################
    # Show the forward and back projection from a single pixel
    i, j = num_recon_rows // 2, num_recon_cols // 2
    x = jnp.zeros(recon_shape)
    x = x.at[i, j, :].set(1)
    voxel_values = x.reshape((-1, num_recon_slices))[indices[0]]

    Ax = parallel_model.sparse_forward_project(voxel_values, indices[0])
    Aty = parallel_model.sparse_back_project(Ax, indices[0])
    Aty = parallel_model.reshape_recon(Aty)
    slice_index = num_recon_slices // 2
    Aty_normalized = Aty / Aty[i, j, slice_index]
    reference = 0.45 / np.clip(np.abs(np.arange(num_recon_rows) - (num_recon_cols / 2)), 0.45, None)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax[0].imshow(np.log10(Aty_normalized[:, :, slice_index]))
    ax[0].set_title('log10(AtAx)')
    ax[1].semilogy(Aty_normalized[:, num_recon_cols // 2, 0])
    ax[1].semilogy(reference)
    ax[1].legend(['AtAx / max, restricted to a line', 'C / |v - v0|, clipped'])
    ax[1].set_ylim(top=10)
    ax[1].set_title('Slice of AtAx')
    fig_folder = mbirjax.make_figure_folder()
    plt.savefig(os.path.join(fig_folder, 'AtAx_normalized.png'), bbox_inches='tight')
    plt.show()

    a = 0
