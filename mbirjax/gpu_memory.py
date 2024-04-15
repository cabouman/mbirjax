import jax

"""
Memory usage from back_project (which takes 5x-30x more memory than forward_project):

Changing number of rows with 256 views by 256 channels
10: {'num_allocs': 63, 'bytes_in_use': 14983168, 'peak_bytes_in_use': 2431167744, 'largest_alloc_size': 2415922432, 'bytes_limit': 12701761536, 'bytes_reserved': 0, 'peak_bytes_reserved': 0, 'largest_free_block_bytes': 0, 'pool_bytes': 3228616192, 'peak_pool_bytes': 3228616192}
20: {'num_allocs': 63, 'bytes_in_use': 22846976, 'peak_bytes_in_use': 4251233024, 'largest_alloc_size': 4228123904, 'bytes_limit': 12701761536, 'bytes_reserved': 0, 'peak_bytes_reserved': 0, 'largest_free_block_bytes': 0, 'pool_bytes': 12701761536, 'peak_pool_bytes': 12701761536}
25: {'num_allocs': 63, 'bytes_in_use': 28498432, 'peak_bytes_in_use': 5263517440, 'largest_alloc_size': 5234756864, 'bytes_limit': 12701761536, 'bytes_reserved': 0, 'peak_bytes_reserved': 0, 'largest_free_block_bytes': 0, 'pool_bytes': 12701761536, 'peak_pool_bytes': 12701761536}
30: {'num_allocs': 63, 'bytes_in_use': 34761984, 'peak_bytes_in_use': 6276413952, 'largest_alloc_size': 6241389824, 'bytes_limit': 12701761536, 'bytes_reserved': 0, 'peak_bytes_reserved': 0, 'largest_free_block_bytes': 0, 'pool_bytes': 12701761536, 'peak_pool_bytes': 12701761536}
40: {'num_allocs': 63, 'bytes_in_use': 42659328, 'peak_bytes_in_use': 8297577216, 'largest_alloc_size': 8254655744, 'bytes_limit': 12701761536, 'bytes_reserved': 0, 'peak_bytes_reserved': 0, 'largest_free_block_bytes': 0, 'pool_bytes': 12701761536, 'peak_pool_bytes': 12701761536}

Same but with only half the indices:
40: {'num_allocs': 69, 'bytes_in_use': 37415936, 'peak_bytes_in_use': 4164947712, 'largest_alloc_size': 4127329536, 'bytes_limit': 12701761536, 'bytes_reserved': 0, 'peak_bytes_reserved': 0, 'largest_free_block_bytes': 0, 'pool_bytes': 12701761536, 'peak_pool_bytes': 12701761536}
One quarter the indices:
40: {'num_allocs': 69, 'bytes_in_use': 34794496, 'peak_bytes_in_use': 2098526464, 'largest_alloc_size': 2063666432, 'bytes_limit': 12701761536, 'bytes_reserved': 0, 'peak_bytes_reserved': 0, 'largest_free_block_bytes': 0, 'pool_bytes': 12701761536, 'peak_pool_bytes': 12701761536}

"""


def get_gpu_memory_stats(print_results=False):
    # Get all GPU devices
    gpus = [device for device in jax.devices() if 'cpu' not in device.device_kind.lower()]

    # If there are no GPUs, return None
    if not gpus:
        return "No GPU devices found."

    # Collect memory info
    memory_info = []
    for gpu in gpus:
        # Memory info returns total_memory and available_memory in bytes
        stats = gpu.memory_stats()
        memory_info.append(stats)
        if print_results:
            print('GPU: {}'.format(gpu.id))
            print('bytes_in_use:      {:.3f}GB'.format(stats['bytes_in_use'] / (1024 ** 3)))
            print('peak_bytes_in_use: {:.3f}GB'.format(stats['peak_bytes_in_use'] / (1024 ** 3)))
            print('bytes_limit      : {:.3f}GB'.format(stats['bytes_limit'] / (1024 ** 3)))

    return memory_info

