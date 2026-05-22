import jax
import os
import sys
import psutil
import resource


def get_memory_stats(print_results=True, file=None):
    """Return memory statistics for all GPUs and the CPU process.

    Each entry in the returned list is a dict with keys:
        id                  -- device label, e.g. 'GPU 0' or 'CPU'
        bytes_in_use        -- current allocation (GPU: JAX pool; CPU: USS or RSS)
        peak_bytes_in_use   -- high-watermark (GPU: JAX pool peak since start;
                               CPU: peak RSS from OS resource accounting)
        bytes_limit         -- capacity ceiling (GPU: JAX pool limit = VRAM × mem_fraction;
                               CPU: total physical RAM)

    Args:
        print_results (bool): print a formatted summary when True (default).
        file: file-like object for print output; None → stdout.

    Returns:
        list[dict]: one dict per device (GPUs first, then CPU).
    """
    memory_stats_per_processor = []

    # ── GPU devices ──────────────────────────────────────────────────────────
    gpus = [d for d in jax.devices() if 'cpu' not in d.device_kind.lower()]

    for gpu in gpus:
        try:
            gpu_stats = gpu.memory_stats()
            memory_stats = {
                'id':                'GPU ' + str(gpu.id),
                'bytes_in_use':      gpu_stats['bytes_in_use'],
                'peak_bytes_in_use': gpu_stats['peak_bytes_in_use'],
                'bytes_limit':       gpu_stats['bytes_limit'],
            }
        except Exception as exc:
            print(f'Warning: could not read memory stats for GPU {gpu.id}: {exc}',
                  file=file)
            memory_stats = {
                'id':                'GPU ' + str(gpu.id),
                'bytes_in_use':      0,
                'peak_bytes_in_use': 0,
                'bytes_limit':       0,
            }
        memory_stats_per_processor.append(memory_stats)

    # ── CPU process ───────────────────────────────────────────────────────────
    current_process = psutil.Process(os.getpid())

    # bytes_in_use: Unique Set Size (memory this process uniquely owns).
    # memory_full_info() is needed for uss but is slower and may fail on some
    # platforms; fall back to RSS if unavailable.
    try:
        mem_full = current_process.memory_full_info()
        bytes_in_use = mem_full.uss
    except Exception:
        bytes_in_use = current_process.memory_info().rss

    # peak_bytes_in_use: true OS-level peak RSS via resource accounting.
    # ru_maxrss is in bytes on macOS and kilobytes on Linux.
    try:
        ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = ru_maxrss if sys.platform == 'darwin' else ru_maxrss * 1024
    except Exception:
        # Fallback: current RSS is a lower bound on the true peak.
        peak_bytes = current_process.memory_info().rss

    # bytes_limit: total physical RAM — a stable capacity number.
    # (Previously this was mem.available, which is a changing snapshot, not a limit.)
    bytes_limit = psutil.virtual_memory().total

    memory_stats_per_processor.append({
        'id':                'CPU',
        'bytes_in_use':      bytes_in_use,
        'peak_bytes_in_use': peak_bytes,
        'bytes_limit':       bytes_limit,
    })

    # ── Print ─────────────────────────────────────────────────────────────────
    if print_results:
        for stats in memory_stats_per_processor:
            print(stats['id'], file=file)
            for tag in ['bytes_in_use', 'peak_bytes_in_use', 'bytes_limit']:
                gb = stats[tag] / 1024 ** 3
                print(f'  {tag:<24}{gb:.3f} GB', file=file)

    return memory_stats_per_processor
