import jax
import os
import psutil


def get_memory_stats(print_results=True, file=None):
    # Get all GPU devices
    gpus = [device for device in jax.devices() if 'cpu' not in device.device_kind.lower()]

    memory_stats_per_processor = []

    # Collect memory info for gpus
    for gpu in gpus:
        # Memory info returns total_memory and available_memory in bytes
        gpu_stats = gpu.memory_stats()
        memory_stats = dict()
        memory_stats['id'] = 'GPU ' + str(gpu.id)
        memory_stats['bytes_in_use'] = gpu_stats['bytes_in_use']
        memory_stats['peak_bytes_in_use'] = gpu_stats['peak_bytes_in_use']
        memory_stats['bytes_limit'] = gpu_stats['bytes_limit']

        memory_stats_per_processor.append(memory_stats)

    # Then add info for the CPU
    memory_stats = dict()
    # Get the current process ID
    pid = os.getpid()

    # Create a process object using the current PID
    current_process = psutil.Process(pid)

    # Get the memory usage
    memory_info = current_process.memory_full_info()
    memory_stats['id'] = 'CPU'

    # memory_info.rss will return the Resident Set Size (RSS)
    # which is the non-swapped physical memory the process has used
    memory_stats['peak_bytes_in_use'] = memory_info.rss
    memory_stats['bytes_in_use'] = memory_info.uss  # This is the 'Unique Set Size' used by the process
    # Get the virtual memory statistics
    mem = psutil.virtual_memory()

    # Available physical memory (excluding swap)
    memory_stats['bytes_limit'] = mem.available
    memory_stats_per_processor.append(memory_stats)

    if print_results:
        for memory_stats in memory_stats_per_processor:
            print(memory_stats['id'], file=file)
            for tag in ['bytes_in_use', 'peak_bytes_in_use', 'bytes_limit']:
                cur_value = memory_stats[tag] / (1024 ** 3)
                extra_space = ' ' * (21 - len(tag) - len(str(int(cur_value))))
                print(f'  {tag}:{extra_space}{cur_value:.3f}GB', file=file)

    return memory_stats_per_processor
