import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mbirjax
import mbirjax.parallel_beam
from scipy.sparse.linalg import svds, eigsh, aslinearoperator, LinearOperator
import jax

if __name__ == "__main__":
    """
    This is a script to develop, debug, and tune the vcd reconstruction with a parallel beam projector
    """
    view_batch_size = None
    pixel_batch_size = None
    with jax.experimental.enable_x64(True):  # Finite difference requires 64 bit arithmetic

        # Initialize sinogram
        num_views = 32
        num_det_rows = 1
        num_det_channels = 32
        start_angle = 0
        end_angle = jnp.pi
        sinogram = jnp.zeros((num_views, num_det_rows, num_det_channels))
        angles = jnp.linspace(start_angle, jnp.pi, num_views, endpoint=False)

        # Set up parallel beam model
        parallel_model = mbirjax.ParallelBeamModel(sinogram.shape, angles)

        # Generate phantom
        recon_shape = parallel_model.get_params('recon_shape')
        num_recon_rows, num_recon_cols, num_recon_slices = recon_shape[:3]
        phantom = mbirjax.generate_3d_shepp_logan_low_dynamic_range((num_det_channels,num_det_channels,num_det_channels))
        phantom = phantom[:, :, num_det_channels // 2]

        # Generate indices of pixels
        num_subsets = 4
        full_indices = mbirjax.gen_pixel_partition(recon_shape, num_subsets)[0]

        # Generate sinogram data
        voxel_values = phantom.reshape((-1,) + recon_shape[2:])[full_indices]

        parallel_model.set_params(view_batch_size=view_batch_size, pixel_batch_size=pixel_batch_size)

        print('Starting forward projection')
        sinogram = parallel_model.sparse_forward_project(voxel_values, full_indices)

        # Determine resulting number of views, slices, and channels and image size
        print('Sinogram shape: {}'.format(sinogram.shape))

        # Get the vector of indices
        all_indices = jnp.arange(num_recon_rows * num_recon_cols)

        sinogram = jnp.array(sinogram)
        all_indices = jnp.array(all_indices)

        hess = parallel_model.compute_hessian_diagonal().flatten()

        # Run once to finish compiling
        print('Starting back projection')
        bp = parallel_model.sparse_back_project(sinogram, all_indices)
        print('Recon shape: ({}, {}, {})'.format(num_recon_rows, num_recon_cols, num_recon_slices))

        input_size = all_indices.size
        input_shape = (all_indices.size, 1)
        output_size = sinogram.size
        output_shape = sinogram.shape

        # Set up for svd
        def Ax_full(local_x):
            local_x = np.reshape(local_x, input_shape)
            ax_flat = parallel_model.sparse_forward_project(local_x, all_indices)
            ax_flat = np.array(ax_flat.flatten())
            return ax_flat

        def Aty_full(local_y):
            local_y = np.reshape(local_y, output_shape)
            aty_flat = parallel_model.sparse_back_project(local_y, all_indices)
            aty_flat = np.array(aty_flat.flatten())
            return aty_flat

        def precond_AtAx(local_x):
            atax = Aty_full(Ax_full(local_x))
            precond_atax = atax / hess
            return precond_atax

        def precond_AtAx_T(local_x):
            local_x = local_x / hess
            atax = Aty_full(Ax_full(local_x))
            return atax

        AtAx_linear_operator = LinearOperator(matvec=precond_AtAx, rmatvec=precond_AtAx_T, shape=(input_size, input_size))

        operator_shape = (sinogram.size, input_size)
        num_sing_values = np.amin(operator_shape) - 20
        eig_vects = True
        print('Computing full AtAx / H eigen-decomposition')
        if eig_vects:
            u, s, vh = svds(AtAx_linear_operator, k=num_sing_values, tol=1e-6, return_singular_vectors=True, solver='propack')
            vh = vh[::-1, :]
            u = u[:, ::-1]
            # mbirjax.slice_viewer(vh.reshape(num_sing_values, num_det_channels, num_det_channels), slice_axis=0)
            # mbirjax.slice_viewer(u.reshape((num_det_channels, num_det_channels, num_sing_values)), slice_axis=2)
        else:
            s = svds(AtAx_linear_operator, k=num_sing_values, tol=1e-6, return_singular_vectors=False, solver='propack')

        s = s[::-1]

        # Get a mask
        g = 1 / num_subsets
        mask = np.array(np.random.rand(*phantom.shape) < g)
        mask = mask.reshape((-1, 1))
        mask_indices = np.where(mask)[0]
        hess_m = hess[mask_indices]

        # Define the masked operators for svd
        def Ax_masked(local_x):
            local_x = local_x.reshape((-1, 1))
            ax_flat = parallel_model.sparse_forward_project(local_x, mask_indices)
            ax_flat = np.array(ax_flat.flatten())
            return ax_flat

        def Aty_masked(local_y):
            local_y = local_y.reshape(output_shape)
            aty_flat = parallel_model.sparse_back_project(local_y, mask_indices)
            aty_flat = np.array(aty_flat.flatten())
            return aty_flat

        def precond_AtAx_masked(local_x):
            atax = Aty_masked(Ax_masked(local_x))
            precond_atax = atax / hess_m
            return precond_atax

        def precond_AtAx_T_masked(local_x):
            local_x = local_x / hess_m
            atax = Aty_masked(Ax_masked(local_x))
            return atax

        # Get the svd for the masked operator
        masked_operator_shape = (len(mask_indices), len(mask_indices))
        Ax_masked_linear_operator = LinearOperator(matvec=precond_AtAx_masked, rmatvec=precond_AtAx_T_masked, shape=masked_operator_shape)

        print('Computing masked AtA / H eigen-decomposition')
        num_masked_sing_values = np.amin(masked_operator_shape) - 1
        num_masked_sing_values = np.minimum(num_masked_sing_values, num_sing_values)
        u_m, s_m, vh_m = svds(Ax_masked_linear_operator, k=num_masked_sing_values, tol=1e-6, return_singular_vectors=True, solver='propack')
        vh_m = vh_m[::-1, :]
        u_m = u_m[:, ::-1]
        s_m = s_m[::-1]

        vm = np.zeros((vh_m.shape[0], np.prod(phantom.shape)))
        vm[:, mask_indices] = vh_m
        vm = vm.reshape((vm.shape[0],) + phantom.shape)
        mbirjax.slice_viewer(vm, slice_axis=0, title='Eigenimages for masked A matrix, {} subsets'.format(num_subsets))

        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        plt.suptitle('recon_shape={}; masking with {} subsets'.format(recon_shape, num_subsets))

        axs[0, 0].semilogy(s, '.')
        axs[0, 0].set_title('Singular values of AtA / H')
        axs[0, 1].semilogy(s_m, '.')
        axs[0, 1].set_title('Singular values of masked AtA / H')
        y_limits = list(axs[0, 0].get_ylim())
        y_limits[0] = 1e-2
        axs[0, 0].set_ylim(y_limits)
        axs[0, 1].set_ylim(y_limits)

        im0 = axs[1, 0].imshow(u @ (np.diag(s) @ vh))
        fig.colorbar(im0, ax=axs[1, 0])
        axs[1, 0].set_title('Full AtA / H')

        im1 = axs[1, 1].imshow(u_m @ (np.diag(s_m) @ vh_m))
        fig.colorbar(im1, ax=axs[1, 1])
        axs[1, 1].set_title('Masked AtA / H')
        plt.show()
        a = 0