"""
FBP filter sharding: correctness and performance across sizes and device counts.

GPUs are used automatically when available; virtual CPU devices are created
otherwise so the script runs on any machine.  All other configuration lives in
the block immediately below this docstring.

What this script measures
─────────────────────────
Each timed call measures ONLY the filter computation — the sinogram is already
pre-sharded across devices before the clock starts.  This matches the real VCD
pipeline, where the sinogram comes from a previous step already resident on the
device(s).

Baseline choice: 1-device sharded (threading path), NOT unsharded else-branch
──────────────────────────────────────────────────────────────────────────────
Both the baseline (n=1) and the N-device runs use configure_sharding(devices),
so they go through the same threading code path in fbp_filter.  Comparing the
unsharded else-branch to the sharded threading path is apples-to-oranges: the
two paths have different JIT compilation profiles and can show spurious speedup
or slowdown that has nothing to do with scaling.  Using the 1-device sharded
run as baseline (same approach as sharding_scaling.py) isolates pure compute
scaling — the only thing that matters.

Why NOT timing with numpy sinogram input
─────────────────────────────────────────
Passing a numpy sinogram to fbp_filter forces a CPU→GPU scatter on every call.
On GPU, that copy dominates and masks the compute speedup completely.  On CPU
(Mac virtual devices) the copy is free, so speedup appears even without pre-
sharding.  Pre-sharding removes the asymmetry and reveals true compute scaling
on both platforms.
"""

import os


def _count_available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))   # Linux — process CPU set
    except AttributeError:
        return os.cpu_count() or 1             # macOS / Windows fallback


# Must be set before JAX initialises.  The flag only affects the CPU backend;
# on GPU machines it is harmless.  setdefault leaves cluster-set values alone.
os.environ.setdefault(
    "XLA_FLAGS",
    f"--xla_force_host_platform_device_count={_count_available_cpus()}"
)

# ── Configuration ────────────────────────────────────────────────────────────

# Device counts to test.  Entries larger than the number of available devices
# are skipped automatically.
N_DEVICES_LIST = [1, 2, 4, 8]

N_WARMUP_RUNS = 2   # discarded; warms up JIT and GPU caches
N_TIMING_RUNS = 5   # timed runs per (size, n_devices); report the minimum
                    # Use ≥3 on GPU — single-trial GPU timing is noisy.

MAX_DIFF_THRESHOLD = 1e-4  # warn if max diff between device counts exceeds this

# ── Imports ───────────────────────────────────────────────────────────────────

import jax
import jax.numpy as jnp
import numpy as np
import time
import mbirjax as mj

# ── Device selection ──────────────────────────────────────────────────────────
# Prefer real GPUs; fall back to virtual CPU devices when none are present.

try:
    available_devices = jax.devices('gpu')
except RuntimeError:
    available_devices = []

if available_devices:
    device_type = 'GPU'
    # Sizes to sweep.  num_views = num_det_rows = num_det_channels = size.
    # Every size must be divisible by every entry in N_DEVICES_LIST.
    SIZES = [512, 1024]
else:
    available_devices = jax.devices('cpu')
    device_type = 'CPU (virtual)'
    # Sizes to sweep.  num_views = num_det_rows = num_det_channels = size.
    # Every size must be divisible by every entry in N_DEVICES_LIST.
    SIZES = [128, 256, 512]

print(f"Using {device_type} — {len(available_devices)} device(s): {available_devices}")

# ── Validation ────────────────────────────────────────────────────────────────

for size in SIZES:
    for n in N_DEVICES_LIST:
        if size % n != 0:
            raise ValueError(f"size={size} is not divisible by n_devices={n}. "
                             f"Adjust SIZES or N_DEVICES_LIST.")

# ── Helper ───────────────────────────────────────────────────────────────────

def _time_fn(fn, n_warmup, n_trials):
    """Run fn() for n_warmup+n_trials iterations; return (min_s, results_list)."""
    last = None
    times = []
    for i in range(n_warmup + n_trials):
        t0 = time.perf_counter()
        last = fn()
        jax.block_until_ready(last)
        dt = time.perf_counter() - t0
        if i >= n_warmup:
            times.append(dt)
    return min(times), last

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

    # ── Per-device-count sweep ──────────────────────────────────────────────
    # IMPORTANT: the baseline is 1-device sharded (threading path), NOT the
    # unsharded else-branch.  Both the baseline and the N-device runs go
    # through configure_sharding(devices), so they use the same threading
    # code path.  Comparing else-branch vs threading-path is apples-to-
    # oranges and can show spurious speedup or slowdown unrelated to scaling.
    #
    # The sinogram is pre-sharded once per (size, n_devices) combination
    # (outside timing) so every timed call is pure compute — zero PCIe on
    # GPU, zero scatter/gather overhead.  This matches sharding_scaling.py.

    model = mj.ParallelBeamModel(sino_shape, angles)

    # Correctness reference: compute once with 1 device.
    model.configure_sharding(available_devices[:1])
    sino_sharding_1 = jax.sharding.NamedSharding(
        model.mesh,
        jax.sharding.PartitionSpec(None, 'slices', None))
    baseline_np = np.array(
        model.fbp_filter(jax.device_put(np.asarray(sinogram_np), sino_sharding_1)))

    min_base = None   # set on first (1-device) iteration below

    for n_devices in N_DEVICES_LIST:
        if n_devices > len(available_devices):
            print(f"  n_devices={n_devices}: skipped "
                  f"(only {len(available_devices)} available)")
            continue

        devices = available_devices[:n_devices]
        model.configure_sharding(devices)

        # Pre-shard the sinogram once (outside timing).
        # This simulates data arriving from a previous pipeline step already
        # distributed across devices — exactly what happens in the VCD loop.
        # Every timed call sees a pre-sharded input so fbp_filter does zero
        # scatter/gather; all measured time is pure compute.
        #
        # We construct the sharding directly (public JAX API) rather than
        # calling model._shard_sinogram so this script works with any installed
        # version of mbirjax.  The parallel-beam native sinogram sharding is:
        #   axis 0 (views)    → not sharded
        #   axis 1 (det_rows) → sharded across devices  (='slices' mesh axis)
        #   axis 2 (channels) → not sharded
        sino_sharding = jax.sharding.NamedSharding(
            model.mesh,
            jax.sharding.PartitionSpec(None, 'slices', None))
        sharded_sino = jax.device_put(np.asarray(sinogram_np), sino_sharding)
        jax.block_until_ready(sharded_sino)

        # Correctness: N-device result must match 1-device reference.
        sharded_result_np = np.array(model.fbp_filter(sharded_sino))
        max_diff = float(np.max(np.abs(sharded_result_np - baseline_np)))

        min_t, _ = _time_fn(lambda: model.fbp_filter(sharded_sino),
                             N_WARMUP_RUNS, N_TIMING_RUNS)

        # First passing entry is the 1-device baseline for speedup computation.
        if min_base is None:
            min_base = min_t

        mem_stats = mj.get_memory_stats(print_results=False)
        gpu_mem = {s['id']: s['peak_bytes_in_use'] for s in mem_stats
                   if s['id'].startswith('GPU')}

        results[(size, n_devices)] = {
            'time': min_t,
            'max_diff': max_diff,
            'gpu_mem': gpu_mem,
        }
        speedup = min_base / min_t
        diff_flag = ('  *** WARNING: max_diff exceeds threshold ***'
                     if max_diff > MAX_DIFF_THRESHOLD else '')
        mem_str = '  '.join(f"{dev}: {b/1024**3:.2f}GB" for dev, b in gpu_mem.items())
        baseline_tag = '  (1-dev baseline)' if n_devices == N_DEVICES_LIST[0] else ''
        print(f"  n_devices={n_devices}: {min_t:.3f} s  "
              f"({speedup:.2f}x vs 1-dev)  max_diff={max_diff:.2e}"
              + (f"  [{mem_str}]" if mem_str else '')
              + baseline_tag
              + diff_flag)

# ── Summary table ─────────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print("Summary — time in seconds (speedup vs 1-device baseline)")
print("  [sinogram pre-sharded; timing = pure compute, zero PCIe]\n")

col_labels = [f"{n}dev" for n in N_DEVICES_LIST]
col_w = 22
header = f"{'size':>6}  " + "  ".join(f"{lbl:>{col_w}}" for lbl in col_labels)
print(header)
print("-" * len(header))

for size in SIZES:
    # Find the baseline: smallest n_devices that actually has a result.
    base_n = next((n for n in N_DEVICES_LIST if (size, n) in results), None)
    base_time = results[(size, base_n)]['time'] if base_n is not None else None

    cells = []
    for n in N_DEVICES_LIST:
        r = results.get((size, n))
        if r is None:
            cells.append(f"{'skipped':>{col_w}}")
        elif n == base_n:
            cells.append(f"{r['time']:.3f}s  (baseline)  d={r['max_diff']:.1e}")
        else:
            speedup = base_time / r['time']
            cells.append(f"{r['time']:.3f}s  {speedup:.2f}x  d={r['max_diff']:.1e}")

    row = f"{size:>6}  " + "  ".join(f"{c:>{col_w}}" for c in cells)
    print(row)

print()

# ── Plot: speedup vs number of devices ───────────────────────────────────────

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 5))

all_n = sorted(n for n in N_DEVICES_LIST if n <= len(available_devices))

for size in SIZES:
    # Baseline = smallest n_devices that ran.
    base_n = next((n for n in all_n if (size, n) in results), None)
    if base_n is None:
        continue
    base_time = results[(size, base_n)]['time']

    xs, ys = [], []
    for n in all_n:
        if (size, n) in results:
            xs.append(n)
            ys.append(base_time / results[(size, n)]['time'])
    if xs:
        ax.loglog(xs, ys, marker='o', label=f'size={size}')

# Ideal linear-scaling reference line: n devices → n× speedup vs 1 device
xs_ideal = [n for n in all_n if n >= (all_n[0] if all_n else 1)]
if xs_ideal:
    ys_ideal = [n / xs_ideal[0] for n in xs_ideal]
    ax.loglog(xs_ideal, ys_ideal, 'k--', linewidth=1, label='ideal')

ax.set_xlabel('Number of devices')
ax.set_ylabel(f'Speedup vs {all_n[0] if all_n else 1}-device baseline')
ax.set_title('fbp_filter sharding speedup\n(sinogram pre-sharded; pure compute, zero PCIe)')
ax.set_xticks(all_n)
ax.set_xticklabels(all_n)
ax.legend()
ax.grid(True, which='both', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
