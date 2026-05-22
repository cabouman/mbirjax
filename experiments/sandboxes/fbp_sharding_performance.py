"""
FBP filter sharding: correctness and performance across sizes and device counts.

Toggle between virtual CPU devices and real GPUs with USE_VIRTUAL_CPU below.
All configuration lives in the block immediately below this docstring.
"""

# ── Configuration ────────────────────────────────────────────────────────────

USE_VIRTUAL_CPU = True   # True → virtual CPU devices via XLA_FLAGS
                         # False → use real GPUs (comment out XLA_FLAGS line)

# Sizes to sweep.  num_views = num_det_rows = num_det_channels = size.
# Every size must be divisible by every entry in N_DEVICES_LIST.
SIZES = [64, 128, 256]

# Device counts to test.  Entries larger than the number of available devices
# are skipped automatically.
N_DEVICES_LIST = [1, 2, 4]

N_TIMING_RUNS = 3   # timed runs per (size, n_devices); report the minimum

MAX_DIFF_THRESHOLD = 1e-4  # warn if sharded vs unsharded max diff exceeds this

# ── Environment setup (must happen before any JAX import) ────────────────────

import os
if USE_VIRTUAL_CPU:
    _max_devs = max(N_DEVICES_LIST)
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={_max_devs}"

# ── Imports ───────────────────────────────────────────────────────────────────

import jax
import jax.numpy as jnp
import numpy as np
import time
import mbirjax as mj

# ── Validation ────────────────────────────────────────────────────────────────

available_devices = jax.devices()
print(f"Available devices ({len(available_devices)}): {available_devices}")

for size in SIZES:
    for n in N_DEVICES_LIST:
        if size % n != 0:
            raise ValueError(f"size={size} is not divisible by n_devices={n}. "
                             f"Adjust SIZES or N_DEVICES_LIST.")

# ── Main sweep ────────────────────────────────────────────────────────────────

# results[(size, n_devices)] = {'time': float, 'max_diff': float}
results = {}

for size in SIZES:
    print(f"\n{'=' * 60}")
    print(f"Size: {size}  (sinogram shape: {size} × {size} × {size})")

    sino_shape = (size, size, size)
    angles = np.linspace(0, np.pi, size)

    # Random sinogram — same seed every size for reproducibility.
    rng = np.random.default_rng(42)
    sinogram_np = rng.standard_normal(sino_shape).astype(np.float32)

    # Unsharded baseline: single-device, original code path.
    model = mj.ParallelBeamModel(sino_shape, angles)
    model.configure_sharding(None)
    # Warm up JIT before measuring.
    _ = model.fbp_filter(sinogram_np)
    baseline = np.array(model.fbp_filter(sinogram_np))

    # Time the unsharded path and store under key (size, 0) so the table can
    # show it alongside the sharded results.
    times_base = []
    for _ in range(N_TIMING_RUNS):
        t0 = time.time()
        out = model.fbp_filter(sinogram_np)
        jax.block_until_ready(out)
        times_base.append(time.time() - t0)
    mem_stats_base = mj.get_memory_stats(print_results=False)
    gpu_mem_base = {s['id']: s['bytes_in_use'] for s in mem_stats_base
                    if s['id'].startswith('GPU')}
    results[(size, 0)] = {'time': min(times_base), 'max_diff': 0.0, 'gpu_mem': gpu_mem_base}
    mem_str = '  '.join(f"{dev}: {b/1024**3:.2f}GB" for dev, b in gpu_mem_base.items())
    print(f"  unsharded baseline:  {min(times_base):.3f} s"
          + (f"  [{mem_str}]" if mem_str else ''))

    # Sharded paths.
    for n_devices in N_DEVICES_LIST:
        if n_devices > len(available_devices):
            print(f"  n_devices={n_devices}: skipped "
                  f"(only {len(available_devices)} available)")
            continue

        devices = available_devices[:n_devices]
        model.configure_sharding(devices)

        # Warm up (triggers shard_map JIT compilation for this device count).
        _ = model.fbp_filter(sinogram_np)

        # Correctness: compare sharded output to unsharded baseline.
        mj.get_memory_stats()
        sharded_result = model.fbp_filter(sinogram_np)
        max_diff = float(np.max(np.abs(np.array(sharded_result) - baseline)))

        # Timing.
        times = []
        for _ in range(N_TIMING_RUNS):
            t0 = time.time()
            out = model.fbp_filter(sinogram_np)
            jax.block_until_ready(out)
            times.append(time.time() - t0)

        min_time = min(times)
        # Memory snapshot: current bytes_in_use per GPU after the timed runs.
        mem_stats = mj.get_memory_stats(print_results=False)
        gpu_mem = {s['id']: s['bytes_in_use'] for s in mem_stats
                   if s['id'].startswith('GPU')}

        results[(size, n_devices)] = {
            'time': min_time,
            'max_diff': max_diff,
            'gpu_mem': gpu_mem,
        }
        speedup = results[(size, 0)]['time'] / min_time
        diff_flag = '  *** WARNING: max_diff exceeds threshold ***' if max_diff > MAX_DIFF_THRESHOLD else ''
        mem_str = '  '.join(f"{dev}: {b/1024**3:.2f}GB" for dev, b in gpu_mem.items())
        print(f"  n_devices={n_devices}: {min_time:.3f} s  "
              f"({speedup:.2f}x vs unsharded)  max_diff={max_diff:.2e}"
              + (f"  [{mem_str}]" if mem_str else '')
              + diff_flag)

# ── Summary table ─────────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print("Summary — time in seconds (speedup vs unsharded baseline)\n")

col_labels = ["unsharded"] + [f"{n}dev" for n in N_DEVICES_LIST]
col_w = 18
header = f"{'size':>6}  " + "  ".join(f"{lbl:>{col_w}}" for lbl in col_labels)
print(header)
print("-" * len(header))

for size in SIZES:
    row = f"{size:>6}  "
    cells = []

    base_time = results.get((size, 0), {}).get('time')

    # Unsharded column
    if base_time is not None:
        cells.append(f"{base_time:.3f}s {'(baseline)':>{col_w - 8}}")
    else:
        cells.append(f"{'—':>{col_w}}")

    for n in N_DEVICES_LIST:
        r = results.get((size, n))
        if r is None:
            cells.append(f"{'skipped':>{col_w}}")
        else:
            t = r['time']
            diff = r['max_diff']
            speedup = base_time / t if base_time else float('nan')
            cells.append(f"{t:.3f}s {speedup:.2f}x d={diff:.1e}")

    row += "  ".join(f"{c:>{col_w}}" for c in cells)
    print(row)

print()

# ── Plot: speedup vs number of devices ───────────────────────────────────────

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 5))

all_n = sorted(set(n for n in N_DEVICES_LIST if n <= len(available_devices)))

for size in SIZES:
    base_time = results.get((size, 0), {}).get('time')
    if base_time is None:
        continue
    xs, ys = [], []
    for n in all_n:
        r = results.get((size, n))
        if r is not None:
            xs.append(n)
            ys.append(base_time / r['time'])
    if xs:
        ax.loglog(xs, ys, marker='o', label=f'size={size}')

# Ideal linear-scaling reference line
if all_n:
    ax.loglog(all_n, all_n, 'k--', linewidth=1, label='ideal')

ax.set_xlabel('Number of devices')
ax.set_ylabel('Speedup vs unsharded baseline')
ax.set_title('fbp_filter sharding speedup')
ax.set_xticks(all_n)
ax.set_xticklabels(all_n)
ax.set_yticks(all_n)
ax.set_yticklabels(all_n)
ax.legend()
ax.grid(True, which='both', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
