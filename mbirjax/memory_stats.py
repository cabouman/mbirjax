import jax
import os
import psutil


def _nbytes(a):
    try:
        return int(a.nbytes)
    except Exception:       # noqa: BLE001 — some live objects may not expose nbytes
        return 0


def memory_report(label="", device=None, top_n=15):
    """Print current + peak device bytes and the largest live jax arrays.

    Call at a phase boundary or from the debugger.  Returns peak MB.  On CPU
    `memory_stats()` may be empty (cur/peak show 0) but the live-array inventory
    still works.
    """
    if device is None:
        device = jax.devices()[0]
    try:
        stats = device.memory_stats() or {}
    except Exception:       # noqa: BLE001 — CPU backend may not implement memory_stats
        stats = {}
    cur = stats.get("bytes_in_use", 0) / 1e6
    peak = stats.get("peak_bytes_in_use", 0) / 1e6

    arrays = sorted(jax.live_arrays(), key=_nbytes, reverse=True)
    total = sum(_nbytes(a) for a in arrays) / 1e6

    print(f"\n=== mem_report [{label}] ===")
    print(f"  device bytes_in_use={cur:10.1f} MB   peak_bytes_in_use={peak:10.1f} MB")
    # Full allocator stats: pool_bytes / bytes_reserved / peak_pool_bytes /
    # largest_free_block_bytes / num_allocs distinguish a genuinely-large LIVE working
    # set from a BFC POOL that grew via many alloc/free cycles and stuck (pool ≫ in_use).
    if stats:
        print("  full memory_stats (bytes->MB):")
        for k in sorted(stats):
            v = stats[k]
            if isinstance(v, (int, float)) and "bytes" in k:
                print(f"    {k:<30} = {v / 1e6:12.1f} MB")
            else:
                print(f"    {k:<30} = {v}")
    print(f"  live jax arrays: {len(arrays)}   sum={total:10.1f} MB   (top {top_n}):")
    for a in arrays[:top_n]:
        mb = _nbytes(a) / 1e6
        shp = tuple(getattr(a, "shape", ()))
        dt = getattr(a, "dtype", "?")
        # NamedSharding (a sharded intermediate) vs SingleDeviceSharding (gathered/plain).
        shd = type(getattr(a, "sharding", None)).__name__
        print(f"    {mb:10.1f} MB  shape={shp}  dtype={dt}  sharding={shd}")
    print("=== end ===\n")
    return peak, cur


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
