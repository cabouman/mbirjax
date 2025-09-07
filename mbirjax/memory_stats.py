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


from collections.abc import Iterable

def _human(n):
    for u in ('B','KB','MB','GB','TB'):
        if n < 1024 or u == 'TB':
            return f"{n:.2f} {u}"
        n /= 1024

def _buffer_pointer(arr):
    try:
        return arr._arrays[0].unsafe_buffer_pointer()   # private, may change
    except Exception:
        try:
            return id(arr._arrays[0])
        except Exception:
            return id(arr)

def _normalize_kinds(kinds):
    if kinds in (None, 'all'):
        # detect available platforms dynamically: 'cpu', 'gpu', 'tpu', etc.
        return ['cpu', 'gpu']
    if isinstance(kinds, str):
        return [kinds]
    if isinstance(kinds, Iterable):
        return list(kinds)
    raise ValueError("kinds must be 'all', str, or iterable of str")

def live_unique_buffers(kinds: str | Iterable[str] | None = 'all', top_k: int | None = 10):
    """
    Summarize unique live device buffers for JAX arrays across one or more backends.

    kinds: 'cpu', 'gpu', 'tpu', list like ['cpu','gpu'], or 'all' (default).
    top_k: show the top-K largest unique buffers; None shows all.
    """
    uniq = {}  # (device, ptr) -> row

    def _add(device, arr, logical_shape, dtype, shard_index):
        nbytes = arr.size * arr.dtype.itemsize
        key = (device, _buffer_pointer(arr))
        if key not in uniq:
            uniq[key] = {
                "device": str(device),
                "ptr": key[1],
                "bytes": nbytes,
                "shape": tuple(logical_shape),
                "dtype": str(dtype),
                "shard_index": shard_index,
            }

    arrays_to_scan = []
    for k in _normalize_kinds(kinds):
        arrays_to_scan.extend(jax.live_arrays(k))

    # Ensure pending work is realized before accounting
    for a in arrays_to_scan:
        try:
            a.block_until_ready()
        except Exception:
            pass

    for a in arrays_to_scan:
        shards = getattr(a, "addressable_shards", None)
        if shards:
            for sh in shards:
                _add(sh.device, sh.data, a.shape, a.dtype, getattr(sh, "index", None))
        else:
            devs = getattr(a, "devices", lambda: [])()
            dev = devs[0] if devs else None
            _add(dev, a, a.shape, a.dtype, None)

    by_device = {}
    for row in uniq.values():
        by_device[row["device"]] = by_device.get(row["device"], 0) + row["bytes"]

    rows = sorted(uniq.values(), key=lambda r: r["bytes"], reverse=True)
    if top_k is not None:
        rows = rows[:top_k]

    print("Per-device unique buffer totals:")
    for dev, b in by_device.items():
        print(f"  {dev:>12}: {_human(b)}")

    print("\nTop unique buffers:")
    for r in rows:
        idx = "None" if r["shard_index"] is None else str(r["shard_index"])
        print(f"  {r['device']:>12}  {_human(r['bytes']):>10}  {r['dtype']:<8}  {r['shape']}  shard_index={idx}")

    return {"by_device": by_device, "top": rows}
