import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mbirjax
import mbirjax.parallel_beam
from scipy.sparse.linalg import svds, aslinearoperator, LinearOperator
import jax

if __name__ == "__main__":
    """
    This is a script to develop, debug, and tune the vcd reconstruction with a parallel beam projector
    """
    view_batch_size = None
    pixel_batch_size = None
    with jax.experimental.enable_x64(True):  # Finite difference requires 64 bit arithmetic

        # Initialize sinogram
        num_views = 128
        num_det_rows = 1
        num_det_channels = 128
        start_angle = 0
        end_angle = jnp.pi
        sinogram = jnp.zeros((num_views, num_det_rows, num_det_channels))
        angles = jnp.linspace(start_angle, jnp.pi, num_views, endpoint=False)

        # Initialize a random key
        seed_value = np.random.randint(1000000)
        key = jax.random.PRNGKey(seed_value)

        # Set up parallel beam model
        parallel_model = mbirjax.ParallelBeamModel(sinogram.shape, angles)

        # Generate phantom
        recon_shape = parallel_model.get_params('recon_shape')
        num_recon_rows, num_recon_cols, num_recon_slices = recon_shape[:3]
        phantom = mbirjax.gen_cube_phantom(recon_shape)

        # Generate indices of pixels
        num_subsets = 1
        full_indices = mbirjax.gen_pixel_partition(recon_shape, num_subsets)[0]

        # Generate sinogram data
        voxel_values = phantom.reshape((-1,) + recon_shape[2:])[full_indices]

        parallel_model.set_params(view_batch_size=view_batch_size, pixel_batch_size=pixel_batch_size)

        print('Starting forward projection')
        sinogram = parallel_model.sparse_forward_project(voxel_values, full_indices)

        # Determine resulting number of views, slices, and channels and image size
        print('Sinogram shape: {}'.format(sinogram.shape))

        # Get the vector of indices
        indices = jnp.arange(num_recon_rows * num_recon_cols)

        sinogram = jnp.array(sinogram)
        indices = jnp.array(indices)

        # Run once to finish compiling
        print('Starting back projection')
        bp = parallel_model.sparse_back_project(sinogram, indices)
        print('Recon shape: ({}, {}, {})'.format(num_recon_rows, num_recon_cols, num_recon_slices))

        # ##########################
        # Test the adjoint property
        # Get a random 3D phantom to test the adjoint property
        key, subkey = jax.random.split(key)
        x = jax.random.uniform(subkey, shape=bp.shape)
        key, subkey = jax.random.split(key)
        y = jax.random.uniform(subkey, shape=sinogram.shape)

        # Do a forward projection, then a backprojection
        x = x.reshape((-1, num_recon_slices))[indices]
        Ax = parallel_model.sparse_forward_project(x, indices)
        Aty = parallel_model.sparse_back_project(y, indices)

        # Calculate <Aty, x> and <y, Ax>
        Aty_x = jnp.sum(Aty * x)
        y_Ax = jnp.sum(y * Ax)

        assert(np.allclose(Aty_x, y_Ax))
        print("Adjoint property holds for random x, y <y, Ax> = <Aty, x>: {}".format(np.allclose(Aty_x, y_Ax)))

        def Ax_flat(local_x):
            local_x = np.reshape(local_x, x.shape)
            ax_flat = parallel_model.sparse_forward_project(local_x, indices)
            ax_flat = np.array(ax_flat.flatten())
            return ax_flat

        assert(np.allclose(Ax.flatten(), Ax_flat(x.flatten())))
        print('Ax_flat matches forward projection')

        def Aty_flat(y):
            local_y = np.reshape(y, Ax.shape)
            aty_flat = parallel_model.sparse_back_project(local_y, indices)
            aty_flat = np.array(aty_flat.flatten())
            return aty_flat

        assert(np.allclose(Aty.flatten(), Aty_flat(y.flatten())))
        print('Aty_flat matches back projection')

        Ax_linear_operator = LinearOperator(matvec=Ax_flat, rmatvec=Aty_flat, shape=(Ax.size, x.size))

        Ax_lo = Ax_linear_operator(x.flatten())
        Aty_lo = Ax_linear_operator.rmatvec(np.array(y).flatten())

        assert(np.allclose(Ax.flatten(), Ax_lo))
        assert(np.allclose(Aty.flatten(), Aty_lo))
        print('Linear operator matches known projectors')

        num_sing_values = 15  # num_views * num_det_channels
        sing_vects = True
        if sing_vects:
            u, s, vh = svds(Ax_linear_operator, k=num_sing_values, tol=1e-6, return_singular_vectors=True, solver='propack')
            vh = vh.reshape(num_sing_values, num_det_channels, num_det_channels)
            vh = vh[::-1, :, :]
            mbirjax.slice_viewer(vh, slice_axis=0)
            u = u.reshape((num_det_channels, num_det_channels, num_sing_values))
            u = u[:, :, ::-1]
            mbirjax.slice_viewer(u, slice_axis=2)
        else:
            s = svds(Ax_linear_operator, k=num_sing_values, tol=1e-6, return_singular_vectors=False,
                     solver='propack')
        plt.plot(np.sort(s)[::-1], '.')
        plt.show()
        a = 0