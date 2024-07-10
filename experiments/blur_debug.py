
import numpy as np
import jax.numpy as jnp
import jax
import time
import matplotlib.pyplot as plt
import gc
import mbirjax


if __name__ == "__main__":

    indices = np.arange(20)

    def mosaic(x):
        y = jnp.zeros(x.shape)
        y = y.at[1::2].set(x[1::2])
        return y


    x = jax.device_put(np.r_[:10].astype(np.float32))
    fx = mosaic(x)
    trans_fun = jax.linear_transpose(mosaic, x)
    b = trans_fun(fx)  # Raises exception:

    def mosaic(x):
        blurred_image_shape, sigma = ((128, 128, 10), 2.0)  # self.get_params(['sinogram_shape', 'sigma_psf'])
        blurred_image = jnp.zeros(blurred_image_shape).reshape((-1, blurred_image_shape[2]))
        blurred_image = blurred_image.at[indices].set(x)
        blurred_image = blurred_image.reshape(blurred_image_shape)
        return blurred_image

    x = jnp.ones((len(indices), 10))
    fx = mosaic(x)

    vjp_fun = jax.vjp(mosaic, x)[1]
    a = vjp_fun(fx)  # works fine!

    # trans_fun = jax.linear_transpose(mosaic, x)
    # b = trans_fun(fx)  # Raises exception:

    """
    This is a script to develop, debug, and tune the blur model
    """
    # Initialize the image
    image_shape = (64, 64, 5)
    sigma = 0.5
    sharpness = -2
    noise_std = 0.04

    # Set up blur model
    blur_model = mbirjax.blur.Blur(image_shape, sigma)

    # Generate phantom
    recon_shape = blur_model.get_params('recon_shape')
    num_recon_rows, num_recon_cols, num_recon_slices = recon_shape[:3]
    phantom = mbirjax.generate_3d_shepp_logan_low_dynamic_range((image_shape[0], image_shape[1], image_shape[0]))
    center = phantom.shape[2] // 2
    start = center - image_shape[2] // 2
    phantom = phantom[:, :, start:start+image_shape[2]]
    blurred_phantom = blur_model.forward_project(phantom)
    blurred_phantom += noise_std * np.random.randn(*blurred_phantom.shape)
    mbirjax.slice_viewer(phantom, blurred_phantom)

    blur_model.set_params(sharpness=sharpness)
    # blur_model.set_params(sigma_y=noise_std/2)
    blur_model.set_params(partition_sequence=(0, 0, 0, 1, 2, 2, ))
    recon, recon_params = blur_model.recon(blurred_phantom, weights=None, compute_prior_loss=True, num_iterations=20)
    mbirjax.slice_viewer(phantom, recon)

    # Generate indices of pixels
    num_subsets = 1
    full_indices = mbirjax.gen_pixel_partition(recon_shape, num_subsets)[0]
    num_subsets = 5
    subset_indices = mbirjax.gen_pixel_partition(recon_shape, num_subsets)
    voxel_values = phantom.reshape((-1,) + recon_shape[2:])[full_indices]

    x = np.random.rand(*voxel_values.shape)
    Ax = blur_model.sparse_forward_project(x, full_indices)
    y = np.random.rand(*Ax.shape)
    Aty = blur_model.sparse_back_project(y, full_indices)
    yt_Ax = np.sum(y * Ax)
    xt_Aty = np.sum(x * Aty)
    assert(np.allclose(yt_Ax, xt_Aty))
    print("Adjoint property holds for random x, y <y, Ax> = <Aty, x>: {}".format(np.allclose(yt_Ax, xt_Aty)))

    # ##########################
    # ## Test the hessian against a finite difference approximation ## #
    hessian = blur_model.compute_hessian_diagonal()

    x = jnp.zeros(recon_shape)
    key = jax.random.key(np.random.randint(100000))
    key, subkey = jax.random.split(key)
    i, j = jax.random.randint(subkey, shape=(2,), minval=0, maxval=num_recon_rows)
    key, subkey = jax.random.split(key)
    k = jax.random.randint(subkey, shape=(), minval=0, maxval=num_recon_slices)

    eps = 0.01
    x = x.at[i, j, k].set(eps)
    indices = jnp.arange(num_recon_rows * num_recon_cols)
    voxel_values = x.reshape((-1, num_recon_slices))[indices]
    Ax = blur_model.sparse_forward_project(voxel_values, indices)
    AtAx = blur_model.sparse_back_project(Ax, indices).reshape(x.shape)
    finite_diff_hessian = AtAx[i, j, k] / eps
    print('Hessian matches finite difference: {}'.format(jnp.allclose(hessian.reshape(x.shape)[i, j, k], finite_diff_hessian)))

    # ##########################
    # Show the forward and back projection from a single pixel
    i, j = num_recon_rows // 4, num_recon_cols // 3
    x = jnp.zeros(recon_shape)
    x = x.at[i, j, :].set(1)
    voxel_values = x.reshape((-1, num_recon_slices))[indices]

    Ax = blur_model.sparse_forward_project(voxel_values, indices)
    Aty = blur_model.sparse_back_project(Ax, indices)
    Aty = blur_model.reshape_recon(Aty)

    y = jnp.zeros_like(Ax)
    view_index = 30
    y = y.at[view_index].set(Ax[view_index])
    index = jnp.ravel_multi_index((6, 6), (num_recon_rows, num_recon_cols))

    slice_index = (num_recon_slices + 1) // 2
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    cax = ax[0].imshow(x[:, :, slice_index])
    ax[0].set_title('x = phantom')
    fig.colorbar(cax, ax=ax[0])
    cax = ax[1].imshow(Ax[:, :, slice_index])
    ax[1].set_title('y = Ax')
    fig.colorbar(cax, ax=ax[1])
    cax = ax[2].imshow(Aty[:, :, slice_index])
    ax[2].set_title('Aty = AtAx')
    fig.colorbar(cax, ax=ax[2])
    plt.pause(1)

    a = 0
