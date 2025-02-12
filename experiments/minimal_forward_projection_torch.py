import time
import torch

def set_sinogram_parameters():
    # Specify sinogram info
    num_views = 2000
    num_det_rows = 1000
    num_det_channels = 1000
    return num_views, num_det_rows, num_det_channels

@torch.compile
def sparse_forward_project(voxel_values, indices, sinogram_shape, recon_shape, angles, output_device, worker):
    """
    Batch the views (angles) and voxels/indices, send batches to the GPU to project, and collect the results.
    """
    max_views = 200
    max_pixels = 8000
    num_to_exclude = 0

    indices = indices[:len(indices)-num_to_exclude]
    angles = angles.to(worker)

    # Batch the views and pixels
    num_views = len(angles)
    view_batch_indices = torch.arange(start=0, end=num_views, step=max_views)
    view_batch_indices = torch.concatenate([view_batch_indices, num_views * torch.ones(1, dtype=torch.int32)])

    num_pixels = len(indices)
    pixel_batch_indices = torch.arange(start=0, end=num_pixels, step=max_pixels)
    pixel_batch_indices = torch.concatenate([pixel_batch_indices, num_pixels * torch.ones(1, dtype=torch.int32)])

    # Create the output sinogram
    sinogram = []

    # Loop over the view batches
    for j, view_index_start in enumerate(view_batch_indices[:2]):
        # Send a batch of views to worker
        view_index_end = view_batch_indices[j+1]
        cur_view_batch = torch.zeros([view_index_end-view_index_start, sinogram_shape[1], sinogram_shape[2]],
                                     device=worker)
        cur_view_params_batch = angles[view_index_start:view_index_end]

        print('Starting view block {} of {}.'.format(j+1, view_batch_indices.shape[0]-1))

        # Loop over pixel batches
        for k, pixel_index_start in enumerate(pixel_batch_indices[:-1]):
            # Send a batch of pixels to worker
            pixel_index_end = pixel_batch_indices[k+1]
            cur_voxel_batch = voxel_values[pixel_index_start:pixel_index_end].to(worker)
            cur_index_batch = indices[pixel_index_start:pixel_index_end].to(worker)

            def forward_project_pixel_batch_local(view, angle):
                # Add the forward projection to the given existing view
                return forward_project_pixel_batch_to_one_view(cur_voxel_batch, cur_index_batch, angle, view,
                                                               sinogram_shape, recon_shape)

            view_map = torch.vmap(forward_project_pixel_batch_local)
            cur_view_batch = view_map(cur_view_batch, cur_view_params_batch)

        sinogram.append(cur_view_batch.to(output_device))
    sinogram = torch.concatenate(sinogram)
    return sinogram

@torch.compile
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
    L_max = torch.clip(W_p_c, None, 1)

    # Do the projection
    for n_offset in torch.arange(start=-psf_radius, end=psf_radius+1):
        n = n_p_center + n_offset
        abs_delta_p_c_n = torch.abs(n_p - n)
        L_p_c_n = torch.clamp((W_p_c + 1) / 2 - abs_delta_p_c_n, torch.zeros(1, device=torch.device('cuda')), L_max)
        A_chan_n = delta_voxel * L_p_c_n / cos_alpha_p_xy
        A_chan_n *= (n >= 0) * (n < num_det_channels)
        n = torch.clip(n, 0, num_det_channels - 1)  # n to a valid range and then not add anything.
        update = A_chan_n.reshape((1, -1)) * voxel_values.T
        # Scatter add the values into the sinogram_view tensor
        indices = n.expand(num_det_rows, -1).type(torch.int64)  # Expand to match batch dim
        sinogram_view = sinogram_view.scatter_add(1, indices, update)

    return sinogram_view


def compute_proj_data(pixel_indices, angle, sinogram_shape, recon_shape):
    """
    Compute the quantities n_p, n_p_center, W_p_c, cos_alpha_p_xy needed for vertical projection.
    """

    cosine = torch.cos(angle)
    sine = torch.sin(angle)

    delta_voxel = 1.0
    dvc = delta_voxel
    dvs = delta_voxel
    dvc *= cosine
    dvs *= sine

    num_views, num_det_rows, num_det_channels = sinogram_shape

    # Convert the index into (i,j,k) coordinates corresponding to the indices into the 3D voxel array
    row_index, col_index = torch.unravel_index(pixel_indices, recon_shape[:2])

    y_tilde = dvs * (row_index - (recon_shape[0] - 1) / 2.0)
    x_tilde = dvc * (col_index - (recon_shape[1] - 1) / 2.0)

    x_p = x_tilde - y_tilde

    det_center_channel = (num_det_channels - 1) / 2.0  # num_of_cols

    # Calculate indices on the detector grid
    n_p = x_p + det_center_channel
    n_p_center = torch.round(n_p).type(torch.int32)

    # Compute cos alpha for row and columns
    cos_alpha_p_xy = torch.maximum(torch.abs(cosine), torch.abs(sine))

    # Compute projected voxel width along columns and rows (in fraction of detector size)
    W_p_c = cos_alpha_p_xy

    proj_data = (n_p, n_p_center, W_p_c, cos_alpha_p_xy)

    return proj_data

#
# def get_memory_stats(print_results=True, file=None):
#     # Get all GPU devices
#     gpus = [device for device in jax.devices() if 'cpu' not in device.device_kind.lower()]
#
#     # Collect memory info for gpus
#     for gpu in gpus:
#         # Memory info returns total_memory and available_memory in bytes
#         gpu_stats = gpu.memory_stats()
#         memory_stats = dict()
#         memory_stats['id'] = 'GPU ' + str(gpu.id)
#         memory_stats['bytes_in_use'] = gpu_stats['bytes_in_use']
#         memory_stats['peak_bytes_in_use'] = gpu_stats['peak_bytes_in_use']
#         memory_stats['bytes_limit'] = gpu_stats['bytes_limit']
#
#         print(memory_stats['id'], file=file)
#         for tag in ['bytes_in_use', 'peak_bytes_in_use', 'bytes_limit']:
#             cur_value = memory_stats[tag] / (1024 ** 3)
#             extra_space = ' ' * (21 - len(tag) - len(str(int(cur_value))))
#             print(f'  {tag}:{extra_space}{cur_value:.3f}GB', file=file)
#


def main():
    """
    This is a script to develop, debug, and tune the parallel beam projector
    """
    num_views, num_det_rows, num_det_channels = set_sinogram_parameters()
    start_angle = 0
    end_angle = torch.pi
    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    step_size = (end_angle - start_angle) / num_views
    angles = torch.linspace(start=start_angle, end=end_angle - step_size, steps=num_views)

    output_device = torch.device("cpu")
    worker = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate phantom - all zero except a small cube
    recon_shape = (num_det_channels, num_det_channels, num_det_rows)
    num_recon_rows, num_recon_cols, num_recon_slices = recon_shape[:3]
    phantom = torch.zeros(recon_shape, device=output_device)  #mbirjax.gen_cube_phantom(recon_shape)
    i, j, k = recon_shape[0]//3, recon_shape[1]//2, recon_shape[2]//2
    phantom[i:i+5, j:j+5, k:k+5] = 1.0

    # Generate indices of pixels and sinogram data
    # Determine the 2D indices within the RoR
    max_index_val = num_recon_rows * num_recon_cols
    indices = torch.arange(max_index_val, dtype=torch.int32)
    voxel_values = phantom.reshape((-1,) + recon_shape[2:])[indices]

    print('Starting forward projection')
    voxel_values.to(output_device)
    indices.to(output_device)
    t0 = time.time()
    sinogram = sparse_forward_project(voxel_values, indices, sinogram_shape, recon_shape, angles,
                                      output_device=output_device, worker=worker)
    print('Elapsed time:', time.time() - t0)

    # Determine resulting number of views, slices, and channels and image size
    print('Sinogram shape: {}'.format(sinogram.shape))
    if torch.cuda.is_available():
        print('Memory stats after forward projection')
        device = torch.device('cuda:0')
        free, total = torch.cuda.mem_get_info(device)
        total_gb = total / (1024 ** 3)
        free_gb = free / (1024 ** 3)
        mem_used_gb = total_gb - free_gb
        print('Used {:.2f}GB of {:.2f}GB total memory'.format(mem_used_gb, total_gb))

    # import mbirjax
    # mbirjax.slice_viewer(sinogram.detach().cpu().numpy(), slice_axis=0)


if __name__ == "__main__":
    main()