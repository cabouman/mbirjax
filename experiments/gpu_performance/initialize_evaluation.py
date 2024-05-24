import sys
import numpy as np
import mbirjax


def initialize_evaluation(eval_type_index, pixel_batch_size, num_views, num_channels, num_det_rows, num_indices):

    eval_types = ['Forward_projection', 'Backward_projection']

    # Make room for the data
    mem_values = np.zeros((len(num_views), len(num_channels), len(num_det_rows), len(num_indices)))
    time_values = np.zeros_like(mem_values)
    max_percent_used_gb = 0

    m1 = mbirjax.get_memory_stats()
    max_avail_gb = m1[0]['bytes_limit'] / (1024 ** 3)

    filename = 'GB_secs_used_for_' + eval_types[eval_type_index]
    np.savez(filename, mem_values=mem_values, time_values=time_values, eval_type_index=np.array(eval_type_index),
             pixel_batch_size=np.array(pixel_batch_size),
             max_percent_used_gb=np.array(max_percent_used_gb), max_avail_gb=np.array(max_avail_gb),
             num_views=np.array(num_views), num_channels=np.array(num_channels),
             num_det_rows=np.array(num_det_rows), num_indices=np.array(num_indices), allow_pickle=True)
    return filename + '.npz'


if __name__ == '__main__':
    eval_type = int(sys.argv[1])
    vox_batch_size = int(sys.argv[2])
    n_views = eval(sys.argv[3])
    n_channels = eval(sys.argv[4])
    n_det_rows = eval(sys.argv[5])
    n_indices = eval(sys.argv[6])

    filename = initialize_evaluation(eval_type, vox_batch_size, n_views, n_channels, n_det_rows, n_indices)
    print(filename)  # Pass back to the calling script
