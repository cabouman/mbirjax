import mbirjax
import numpy as np
import jax.numpy as jnp
import time
import sys


def evaluate_over_indices(filename, nv, nc, nr):

    voxel_batch_size = None

    if mbirjax.get_gpu_memory_stats() is None:
        raise EnvironmentError('This script is for gpu only.')

    # Load the existing data
    data = np.load(filename, allow_pickle=True)
    mem_values = data['mem_values']
    time_values = data['time_values']
    eval_type_index = data['eval_type_index']
    max_percent_used_gb = data['max_percent_used_gb']
    max_avail_gb = data['max_avail_gb']
    num_views = data['num_views']
    num_channels = data['num_channels']
    num_det_rows = data['num_det_rows']
    num_indices = data['num_indices']

    # Set up for projections
    start_angle = 0.0
    end_angle = jnp.pi
    sinogram = None
    bp = None

    i = np.where(num_views == nv)
    j = np.where(num_channels == nc)
    k = np.where(num_det_rows == nr)
    for l, ni in enumerate(num_indices):
        angles = jnp.linspace(start_angle, end_angle, nv, endpoint=False)

        # Set up parallel beam model
        sinogram_shape = (nv, nr, nc)
        parallel_model = mbirjax.parallel_beam.ParallelBeamModel(angles, sinogram_shape)

        # Generate phantom for forward projection
        num_recon_rows, num_recon_cols, num_recon_slices = (
            parallel_model.get_params(['num_recon_rows', 'num_recon_cols', 'num_recon_slices']))
        phantom = mbirjax.gen_phantom(num_recon_rows, num_recon_cols, num_recon_slices)

        # Get a subset of the given size
        indices = np.arange(ni, dtype=int)
        voxel_values = phantom.reshape((-1, num_recon_slices))[indices]

        if eval_type_index == 0:
            print(
                'Initial forward projection for memory: nv={}, nc={}, nr={}, ni={}'.format(nv, nc, nr, ni))
            try:
                sinogram = parallel_model.forward_project(voxel_values, indices)
                m1 = mbirjax.get_gpu_memory_stats()
                peak_mem_gb = m1[0]['peak_bytes_in_use'] / (1024 ** 3)
            except:
                print('Out of memory')
                peak_mem_gb = 1000 * max_avail_gb
            mem_values[i, j, k, l] = peak_mem_gb
            print('Peak GB used = {}'.format(peak_mem_gb))

            print('Forward projection for speed')
            t0 = time.time()
            try:
                sinogram = parallel_model.forward_project(voxel_values, indices)
            except:
                print('Out of memory on pass 2')
            t1 = time.time()
            time_diff_secs = t1 - t0
            time_values[i, j, k, l] = time_diff_secs
            print('Elapsed time = {}'.format(time_diff_secs))

        else:
            print('Initial back projection for memory: nv={}, nc={}, nr={}, ni={}'.format(nv, nc, nr, ni))
            sinogram = np.ones((nv, nr, nc))
            try:
                bp = parallel_model.back_project(sinogram, indices, voxel_batch_size=voxel_batch_size)
                m1 = mbirjax.get_gpu_memory_stats()
                peak_mem_gb = m1[0]['peak_bytes_in_use'] / (1024 ** 3)
            except:
                print('Out of memory')
                peak_mem_gb = 1000 * max_avail_gb

            mem_values[i, j, k, l] = peak_mem_gb
            print('Peak GB used = {}'.format(peak_mem_gb))

            print('Back projection for speed')
            t0 = time.time()
            try:
                bp = parallel_model.back_project(sinogram, indices, voxel_batch_size=voxel_batch_size)
                t1 = time.time()
                time_diff_secs = t1 - t0
            except:
                print('Out of memory on pass 2')
                t1 = time.time()
                time_diff_secs = 1000
            time_values[i, j, k, l] = time_diff_secs
            print('Elapsed time = {}'.format(time_diff_secs))
        max_percent_used_gb = max(max_percent_used_gb, 100 * peak_mem_gb / max_avail_gb)

    print('Max percentage GB used = {}%'.format(max_percent_used_gb))

    np.savez(filename, mem_values=mem_values, time_values=time_values, eval_type_index=np.array(eval_type_index),
             max_percent_used_gb=np.array(max_percent_used_gb), max_avail_gb=np.array(max_avail_gb),
             num_views=np.array(num_views), num_channels=np.array(num_channels),
             num_det_rows=np.array(num_det_rows), num_indices=np.array(num_indices))


if __name__ == '__main__':
    # filename, nv, nc, nr
    filename = sys.argv[1]
    nv = int(sys.argv[2])
    nc = int(sys.argv[3])
    nr = int(sys.argv[4])

    evaluate_over_indices(filename, nv, nc, nr)
