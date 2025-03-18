import os
import numpy as np
import jax.numpy as jnp
import jax
import time
from functools import partial

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.98'

def set_sinogram_parameters():
    # Specify sinogram info
    num_views = 600
    num_det_rows = 1500
    num_det_channels = 2000
    return num_views, num_det_rows, num_det_channels

def sparse_forward_project(voxel_values, indices, sinogram_shape, recon_shape, angles, output_device, worker):
    """
    Batch the views (angles) and voxels/indices, send batches to the GPU to project, and collect the results.
    """
    max_views_per_batch = 200
    max_pixels_per_batch = 8000
    num_pixels_to_exclude = 1000

    indices = indices[:len(indices)-num_pixels_to_exclude]
    angles = jax.device_put(angles, device=worker)

    # Batch the views and pixels
    num_views = len(angles)
    view_batch_indices = jnp.arange(num_views, step=max_views_per_batch)
    view_batch_indices = jnp.concatenate([view_batch_indices, num_views * jnp.ones(1, dtype=int)])

    num_pixels = len(indices)
    pixel_batch_indices = jnp.arange(num_pixels, step=max_pixels_per_batch)
    pixel_batch_indices = jnp.concatenate([pixel_batch_indices, num_pixels * jnp.ones(1, dtype=int)])

    # Create the output sinogram
    sinogram = []

    # Loop over the view batches
    for j, view_index_start in enumerate(view_batch_indices[:-1]):
        # Send a batch of views to worker
        view_index_end = view_batch_indices[j+1]
        cur_view_batch = jnp.zeros([view_index_end-view_index_start, sinogram_shape[1], sinogram_shape[2]],
                                   device=worker)
        cur_view_params_batch = angles[view_index_start:view_index_end]
        if j == 0:
            get_memory_stats()
        print('Starting view block {} of {}.'.format(j+1, view_batch_indices.shape[0]-1))

        # Loop over pixel batches
        for k, pixel_index_start in enumerate(pixel_batch_indices[:-1]):
            # Send a batch of pixels to worker
            pixel_index_end = pixel_batch_indices[k+1]
            cur_voxel_batch, cur_index_batch = jax.device_put([voxel_values[pixel_index_start:pixel_index_end],
                                                              indices[pixel_index_start:pixel_index_end]],
                                                              worker)
            if len(cur_index_batch) < max_pixels_per_batch:
                cur_voxel_batch = jnp.concatenate([cur_voxel_batch, jnp.zeros([max_pixels_per_batch-len(cur_index_batch), sinogram_shape[1],], device=worker)])
                cur_index_batch = jnp.concatenate([cur_index_batch, jnp.zeros([max_pixels_per_batch-len(cur_index_batch)], device=worker)])

            def forward_project_pixel_batch_local(view, angle):
                # Add the forward projection to the given existing view
                return forward_project_pixel_batch_to_one_view(cur_voxel_batch, cur_index_batch, angle, view,
                                                               sinogram_shape, recon_shape)

            view_map = jax.vmap(forward_project_pixel_batch_local)
            # print(jax.make_jaxpr(view_map)(cur_view_batch, cur_view_params_batch))
            # input('Enter to continue')
            # a = jax.jit(view_map).lower(cur_view_batch, cur_view_params_batch).compiler_ir('hlo')
            # with open("outfile.dot", "w") as f:
            #     f.write(a.as_hlo_dot_graph())
            # dot outfile.dot  -Tpng > outfile.png
            # or
            # dot -Tps outfile.dot -o outfile.ps
            # ps2pdf outfile.ps
            # print(jax.jit(view_map).lower(cur_view_batch, cur_view_params_batch).compile().as_text())
            cur_view_batch = view_map(cur_view_batch, cur_view_params_batch)

        sinogram.append(jax.device_put(cur_view_batch, output_device))
    sinogram = jnp.concatenate(sinogram)
    return sinogram


@partial(jax.jit, static_argnames=['sinogram_shape', 'recon_shape'], donate_argnames='sinogram_view')
def forward_project_pixel_batch_to_one_view(voxel_values, pixel_indices, angle, sinogram_view,
                                            sinogram_shape, recon_shape):
    """
    Apply a parallel beam transformation to a set of voxel cylinders. These cylinders are assumed to have
    slices aligned with detector rows, so that a parallel beam maps a cylinder slice to a detector row.
    This function returns the resulting sinogram view.

    """
    # Get all the geometry parameters - we use gp since geometry parameters is a named tuple and we'll access
    # elements using, for example, gp.delta_det_channel, so a longer name would be clumsy.
    num_views, num_det_rows, num_det_channels = sinogram_shape
    psf_radius = 1
    delta_voxel = 1

    # Get the data needed for horizontal projection
    n_p, n_p_center, W_p_c, cos_alpha_p_xy = compute_proj_data(pixel_indices, angle, sinogram_shape, recon_shape)
    L_max = jnp.minimum(1, W_p_c)

    # Do the projection
    for n_offset in jnp.arange(start=-psf_radius, stop=psf_radius+1):
        n = n_p_center + n_offset
        abs_delta_p_c_n = jnp.abs(n_p - n)
        L_p_c_n = jnp.clip((W_p_c + 1) / 2 - abs_delta_p_c_n, 0, L_max)
        A_chan_n = delta_voxel * L_p_c_n / cos_alpha_p_xy
        A_chan_n *= (n >= 0) * (n < num_det_channels)
        sinogram_view = sinogram_view.at[:, n].add(A_chan_n.reshape((1, -1)) * voxel_values.T)

    return sinogram_view

def compute_proj_data(pixel_indices, angle, sinogram_shape, recon_shape):
    """
    Compute the quantities n_p, n_p_center, W_p_c, cos_alpha_p_xy needed for vertical projection.
    """

    cosine = jnp.cos(angle)
    sine = jnp.sin(angle)

    delta_voxel = 1.0
    dvc = delta_voxel
    dvs = delta_voxel
    dvc *= cosine
    dvs *= sine

    num_views, num_det_rows, num_det_channels = sinogram_shape

    # Convert the index into (i,j,k) coordinates corresponding to the indices into the 3D voxel array
    row_index, col_index = jnp.unravel_index(pixel_indices, recon_shape[:2])

    y_tilde = dvs * (row_index - (recon_shape[0] - 1) / 2.0)
    x_tilde = dvc * (col_index - (recon_shape[1] - 1) / 2.0)

    x_p = x_tilde - y_tilde

    det_center_channel = (num_det_channels - 1) / 2.0  # num_of_cols

    # Calculate indices on the detector grid
    n_p = x_p + det_center_channel
    n_p_center = jnp.round(n_p).astype(int)

    # Compute cos alpha for row and columns
    cos_alpha_p_xy = jnp.maximum(jnp.abs(cosine), jnp.abs(sine))

    # Compute projected voxel width along columns and rows (in fraction of detector size)
    W_p_c = cos_alpha_p_xy

    proj_data = (n_p, n_p_center, W_p_c, cos_alpha_p_xy)

    return proj_data


def get_memory_stats(print_results=True, file=None):
    # Get all GPU devices
    gpus = [device for device in jax.devices() if 'cpu' not in device.device_kind.lower()]

    # Collect memory info for gpus
    for gpu in gpus:
        # Memory info returns total_memory and available_memory in bytes
        gpu_stats = gpu.memory_stats()
        memory_stats = dict()
        memory_stats['id'] = 'GPU ' + str(gpu.id)
        memory_stats['bytes_in_use'] = gpu_stats['bytes_in_use']
        memory_stats['peak_bytes_in_use'] = gpu_stats['peak_bytes_in_use']
        memory_stats['bytes_limit'] = gpu_stats['bytes_limit']

        print(memory_stats['id'], file=file)
        for tag in ['bytes_in_use', 'peak_bytes_in_use', 'bytes_limit']:
            cur_value = memory_stats[tag] / (1024 ** 3)
            extra_space = ' ' * (21 - len(tag) - len(str(int(cur_value))))
            print(f'  {tag}:{extra_space}{cur_value:.3f}GB', file=file)



def main():
    """
    This is a script to develop, debug, and tune the parallel beam projector
    """

    # Specify sinogram info
    num_views, num_det_rows, num_det_channels = set_sinogram_parameters()
    start_angle = 0
    end_angle = jnp.pi
    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

    output_device = jax.devices('cpu')[0]
    try:
        worker = jax.devices('gpu')[0]
        use_gpu = True
    except RuntimeError:
        worker = jax.devices('cpu')[0]
        use_gpu = False

    # Generate phantom - all zero except a small cube
    recon_shape = (num_det_channels, num_det_channels, num_det_rows)
    num_recon_rows, num_recon_cols, num_recon_slices = recon_shape[:3]
    with jax.default_device(output_device):
        phantom = jnp.zeros(recon_shape)  #mbirjax.gen_cube_phantom(recon_shape)
        i, j, k = recon_shape[0]//3, recon_shape[1]//2, recon_shape[2]//2
        phantom = phantom.at[i:i+5, j:j+5, k:k+5].set(1.0)

        # Generate indices of pixels and sinogram data
        # Determine the 2D indices within the RoR
        max_index_val = num_recon_rows * num_recon_cols
        indices = np.arange(max_index_val, dtype=np.int32)
        indices = jnp.array(indices)
        voxel_values = phantom.reshape((-1,) + recon_shape[2:])[indices]

    print('Starting forward projection')
    voxel_values, indices = jax.device_put([voxel_values, indices], output_device)
    t0 = time.time()
    sinogram = sparse_forward_project(voxel_values, indices, sinogram_shape, recon_shape, angles,
                                      output_device=output_device, worker=worker)
    print('Elapsed time:', time.time() - t0)

    # Determine resulting number of views, slices, and channels and image size
    print('Sinogram shape: {}'.format(sinogram.shape))
    if use_gpu:
        print('Memory stats after forward projection')
        get_memory_stats(print_results=True)

    # import mbirjax
    # mbirjax.slice_viewer(sinogram, slice_axis=0)


if __name__ == "__main__":
    main()