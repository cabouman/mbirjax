import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mbirjax
import mbirjax.parallel_beam
from scipy.sparse.linalg import svds, eigsh, aslinearoperator, LinearOperator
import jax


def create_deltas(diam):
    # Create an array of images, each with a single nonzero entry, corresponding to increasing Fourier frequencies,
    # where low frequencies are tin the center.
    # Each image has size diam x diam if diam is even or (diam-1) x (diam-1) otherwise.
    # Image 0 has a 1 in the approximate center of the image, corresponding to the constant image after FFT.
    # The subsequent images fill in concentric square rings centered at the first point: 3x3, then 5x5, etc.
    # Taking the FFT of these images produces a set of complex images in space domain with increasing frequency.
    # These delta images will also be used in space to illustrate the AtA psf.
    n = diam // 2
    size = 2 * n
    total_layers = 4 * (n ** 2)
    deltas = np.zeros((size, size, total_layers))
    ordered_indices = np.zeros((total_layers,), dtype=int)

    # Center point
    deltas[n, n, 0] = 1
    ordered_indices[0] = np.ravel_multi_index((n, n), (size, size))
    index = 1

    # Iterate over increasing square rings
    for radius in range(1, n+1):
        # Get the top right point
        i = n - radius
        j = n + radius
        # Move left to right along the top
        for dj in range(0, 2*radius + 1):
            if j - dj < size:
                deltas[i, j - dj, index] = 1
                ordered_indices[index] = np.ravel_multi_index([i, j - dj], (size, size))
                index += 1
        # Then the left side
        j = j - 2*radius
        for di in range(1, 2*radius + 1):
            if i + di < size:
                deltas[i + di, j, index] = 1
                ordered_indices[index] = np.ravel_multi_index([i + di, j], (size, size))
                index += 1
        # Bottom
        i = i + 2*radius
        for dj in range(1, 2*radius + 1):
            if np.maximum(j + dj, i) < size:
                deltas[i, j + dj, index] = 1
                ordered_indices[index] = np.ravel_multi_index([i, j + dj], (size, size))
                index += 1
        # Right side
        j = j + 2 * radius
        for di in range(1, 2 * radius):
            if np.maximum(j, i - di) < size:
                deltas[i - di, j, index] = 1
                ordered_indices[index] = np.ravel_multi_index([i - di, j], (size, size))
                index += 1

    return deltas, ordered_indices


def neighbor_mean(image_stack):
    # Determine the mean over 4 adjacent nearest neighbors in xy directions only
    output = np.zeros_like(image_stack)

    # Pad using reflected boundaries
    m0, m1 = image_stack.shape[:2]
    padded_stack = np.zeros((m0+2, m1+2, image_stack.shape[2]))
    padded_stack[1:-1, 1:-1] = image_stack
    padded_stack[0, 1:-1] = image_stack[1]
    padded_stack[-1, 1:-1] = image_stack[-2]
    padded_stack[1:-1, 0] = image_stack[:, 1]
    padded_stack[1:-1, -1] = image_stack[:, -2]
    for i in [0, 1]:
        for j in [0, 1]:
            padded_stack[(m0+1)*i, (m1+1)*j] = image_stack[(m0-1)*i, (m1-1)*j]

    # Sum over the xy neighbor differences
    for a0 in [0, 1]:
        for a1 in [0, 1]:
            output[a0:m0-1+a0, a1:m1-1+a1] += image_stack[1-a0:m0-a0, 1-a1:m1-a1]

    output /= 4
    return output


if __name__ == "__main__":
    """
    This is a script to investigate the Fourier response of the forward and prior models, with and without masking.
    """
    # Set the gray level (0-1) and whether the subset is random or a grid
    g = 0.5
    grid = False

    view_batch_size = None
    pixel_batch_size = None
    with jax.experimental.enable_x64(True):  # Finite difference requires 64 bit arithmetic

        # Initialize sinogram
        num_views = 32
        num_det_rows = 1024
        num_det_channels = 32
        start_angle = 0
        end_angle = jnp.pi
        sinogram = jnp.zeros((num_views, num_det_rows, num_det_channels))
        angles = jnp.linspace(start_angle, jnp.pi, num_views, endpoint=False)

        # Set up parallel beam model
        parallel_model = mbirjax.ParallelBeamModel(sinogram.shape, angles)
        recon_shape = parallel_model.get_params('recon_shape')
        hess = parallel_model.compute_hessian_diagonal()[:, :, 0].reshape((-1, 1))

        # Generate indices of pixels
        flat_recon_shape = (recon_shape[0] * recon_shape[1], recon_shape[2])
        if grid:
            linear_subsample = np.ceil(np.sqrt(1 / g)).astype(int)
            subset_mask = np.zeros(recon_shape[:2], dtype=int)
            subset_mask[::linear_subsample] += 1
            subset_mask[:, ::linear_subsample] += 1
            subset_mask = np.clip(subset_mask - 1, 0, None)
            subset_mask = subset_mask.flatten()
            subset_indices = np.where(subset_mask > 0)[0]
        else:
            full_indices = np.arange(np.prod(recon_shape[:2]))
            subset_indices = np.where(np.random.rand(*full_indices.shape) <= g)[0]
            subset_mask = np.zeros(flat_recon_shape[0])
            subset_mask[subset_indices] = 1

        clip_min = 0.001
        # Generate single point images, starting from the centering and spiralling out
        deltas, ordered_inds = create_deltas(num_det_channels)

        ##################
        # Get psf in space
        voxel_values = deltas.reshape((-1,) + recon_shape[2:])

        print('Starting forward projection of spatial deltas')
        sinogram = parallel_model.sparse_forward_project(voxel_values[subset_indices], subset_indices)
        bp_subset = parallel_model.sparse_back_project(sinogram, subset_indices)
        bp = np.zeros(flat_recon_shape)
        bp[subset_indices] = bp_subset
        bp = bp / hess
        bp = bp.reshape(recon_shape)
        bp_norm = np.linalg.norm(bp, axis=(0, 1))
        scale = np.amax(bp)
        title = 'AtA PSF in space: output scaled by 1 / {:.1f}'.format(scale)
        title += '\nLeft: single point in space, Right: AtA of that point, g={}'.format(g)
        mbirjax.slice_viewer(deltas, bp / scale, title=title)
        title = 'AtA PSF in space: output in log10'.format(scale)
        title += '\nLeft: single point in space, Right: AtA of that point, g={}'.format(g)
        mbirjax.slice_viewer(deltas, np.log10(np.clip(bp / scale, clip_min, 1)), title=title)

        ######################
        # Get psf in frequency
        deltas_shift = np.fft.fftshift(deltas, axes=(0, 1))
        fourier_images = np.fft.fft2(deltas_shift, axes=(0, 1))
        title = 'fftshift Fourier frequency and corresponding real(FFT)'
        title += '\nLeft: single point in frequency, Right: real(FFT) of that point, g={}'.format(g)
        mbirjax.slice_viewer(deltas, np.real(fourier_images), slice_axis=2, title=title)
        fft_images_phantom = fourier_images[:, :, :num_det_rows]

        # Generate sinogram data
        voxel_values = fft_images_phantom.reshape((-1,) + recon_shape[2:])

        print('Starting forward projection of frequency deltas')
        sinogram_real = parallel_model.sparse_forward_project(np.real(voxel_values[subset_indices]), subset_indices)
        sinogram_imag = parallel_model.sparse_forward_project(np.imag(voxel_values[subset_indices]), subset_indices)

        bp_real_subset = parallel_model.sparse_back_project(sinogram_real, subset_indices)
        bp_imag_subset = parallel_model.sparse_back_project(sinogram_imag, subset_indices)
        bp_complex_subset = bp_real_subset + 1j * bp_imag_subset
        bp_complex = np.zeros(flat_recon_shape, dtype=np.complex64)
        bp_complex[subset_indices] = bp_complex_subset
        bp_complex = bp_complex / hess
        bp_complex = bp_complex.reshape(recon_shape)
        bp_fft = np.fft.ifft2(bp_complex, axes=(0, 1))
        bp_fft = np.fft.ifftshift(bp_fft, axes=(0, 1))
        scale = np.amax(np.abs(bp_fft))
        title = '|AtA frequency transfer function|: output scaled by 1 / {:.1f}'.format(scale)
        title += '\nLeft: single point in frequency, Right: |IFFT(AtA(FFT))| of that point, g={}'.format(g)
        mbirjax.slice_viewer(deltas, np.abs(bp_fft) / scale, title=title,
                             vmin=0, vmax=1, cmap='viridis')
        title = '|AtA frequency transfer function|: output in log10'
        title += '\nLeft: single point in frequency, Right: |IFFT(AtA(FFT))| of that point, g={}'.format(g)
        mbirjax.slice_viewer(deltas, np.log10(np.clip(np.abs(bp_fft), clip_min, None)),
                             title=title, cmap='viridis')

        ######################################
        # Get psf in frequency for prior model
        print('Computing prior update of frequency deltas')
        prior_space_real = 2 * (np.real(fft_images_phantom) - neighbor_mean(np.real(fft_images_phantom)))
        prior_space_imag = 2 * (np.imag(fft_images_phantom) - neighbor_mean(np.imag(fft_images_phantom)))
        prior_space_real *= subset_mask
        prior_space_imag *= subset_mask
        prior_fft = np.fft.ifft2(prior_space_real + 1j * prior_space_imag, axes=(0, 1))
        prior_fft = np.fft.ifftshift(prior_fft, axes=(0, 1))
        scale = np.amax(np.abs(prior_fft))
        title = '|Prior step frequency transfer function|: output scaled by 1 / {:.1f}'.format(scale)
        title += '\nLeft: single point in frequency, Right: |FFT(prior step(FFT))| of that point, g={}'.format(g)
        mbirjax.slice_viewer(deltas, np.abs(prior_fft) / scale, title=title,
                             vmin=0, vmax=1, cmap='viridis')
        title = '|Prior step frequency transfer function|: output in log10'.format(scale)
        title += '\nLeft: single point in frequency, Right: log10|FFT(prior step(FFT))| of that point, g={}'.format(g)
        mbirjax.slice_viewer(deltas, np.log10(np.clip(np.abs(prior_fft), clip_min, None)),
                             title=title, cmap='viridis')

        bp_fft_flat = bp_fft.reshape((-1, bp_fft.shape[2]))
        bp_fft_flat = bp_fft_flat[ordered_inds]
        bp_fft_flat_log10 = np.log10(np.clip(np.abs(bp_fft_flat), clip_min, None))

        prior_fft_flat = prior_fft.reshape((-1, prior_fft.shape[2]))
        prior_fft_flat = prior_fft_flat[ordered_inds]
        prior_fft_flat_log10 = np.log10(np.clip(np.abs(prior_fft_flat), clip_min, None))

        plt.plot(np.diag(bp_fft_flat_log10), '.')
        plt.plot(np.diag(prior_fft_flat_log10), '.')
        plt.title('Diagonal elements of log10 of |freq transfer function|')
        plt.legend(['AtA', 'Prior'])
        title = 'log10 of |freq transfer function|\nLeft: AtA, Right: Prior'
        title += '\nEach row is one input frequency, each column one ouptut frequency'
        mbirjax.slice_viewer(bp_fft_flat_log10, prior_fft_flat_log10, cmap='viridis',title=title)

        gammas = np.linspace(0, 2, 20)
        weighted_sum = bp_fft_flat[:, :, None] + prior_fft_flat[:, :, None] * gammas[None, None, :]
        joint_transfer_log10 = np.log10(np.clip(np.abs(weighted_sum), clip_min, None))
        title = 'log10 of |freq trans func| of AtA + gamma * prior'
        title += '\nAdjust the slider to change gamma'
        mbirjax.slice_viewer(joint_transfer_log10, title=title, slice_label='10 * gamma =', cmap='viridis')

        mbirjax.slice_viewer(np.log10(np.clip(np.abs(bp_fft), clip_min, None)),
                             np.log10(np.clip(np.abs(prior_fft), clip_min, None)),
                             title='Forward (left) and prior (right) PSF in frequency with output in log10')
        a = 0
