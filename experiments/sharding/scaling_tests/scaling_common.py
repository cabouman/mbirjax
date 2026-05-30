"""
experiments/sharding/scaling_tests/scaling_common.py
────────────────────────────────────────────────────
Shared helpers for the per-operation scaling scripts (fbp_filter_scaling.py,
and later sparse_forward_project_scaling.py / sparse_back_project_scaling.py).

Each op driver is thin: it declares how to build a model + input and how to run
the op, then calls the engine here for the two sweeps (by device count, and by
problem size at a fixed device count), correctness vs a stored prerelease
baseline, text tables, and plots.

Design notes
────────────
- Import mbirjax BEFORE jax (device-setup-first ordering).  These scripts are
  consumers, so they follow the rule users should follow.
- which_mbirjax() prints the resolved mbirjax path at startup.  Because both
  worktrees share one conda env and one editable install, the *working
  directory* decides whether beta or research code is loaded — so we surface it
  loudly rather than let a mis-set cwd give silently wrong results.
- Correctness is reported as the percent of elements whose abs error exceeds an
  fp32 threshold (plus max error and count), so a handful of bad points among
  millions reads as "0.0003% above threshold", not a total failure.  This is
  deliberate given the known lax.map/scatter rounding bug.
- Results are written as YAML (ruamel) for readability and saved under results/
  (gitignored).  Plots go to the same place.
"""

import os
import time
import resource
import platform as _platform

# Import mbirjax before jax so _device_setup configures virtual devices first.
import mbirjax

import numpy as np
import jax

from ruamel.yaml import YAML

import matplotlib
matplotlib.use("Agg")   # file output only; no interactive backend needed
import matplotlib.pyplot as plt


# ── Paths ───────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "results")
BASELINES_DIR = os.path.join(_HERE, "baselines")


def _ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(BASELINES_DIR, exist_ok=True)


# ── Which mbirjax am I running? ───────────────────────────────────────────────
def which_mbirjax(beta_substr="mbirjax_sharding"):
    """Print a clear beta/RESEARCH banner for the resolved mbirjax.

    Both worktrees share one conda env and one editable install, so the working
    directory / sys.path decides which code is loaded.  This prints an
    unmissable role label ('beta' or 'RESEARCH') plus a loud warning when the
    research code is loaded, so a mis-set working directory can't silently give
    wrong results.

    Args:
        beta_substr (str): substring identifying the beta worktree path.

    Returns:
        str: the mbirjax package path.
    """
    path = os.path.dirname(mbirjax.__file__)
    is_beta = beta_substr in path
    print("=" * 72)
    if is_beta:
        print("  mbirjax code in use:  *** beta ***  (mbirjax_sharding)")
    else:
        print("  mbirjax code in use:  ###  RESEARCH  ###   <-- NOT the beta code")
        print("  !! WARNING: scaling/correctness results will reflect RESEARCH code,")
        print("  !! not the beta sharding code.  Fix the working directory / PYTHONPATH")
        print("  !! so the beta worktree is on sys.path before running.")
    print(f"  path: {path}")
    print("=" * 72)
    return path


# ── Device / platform detection ───────────────────────────────────────────────
def gpus():
    """List of GPU devices, or [] if no GPU backend."""
    try:
        return jax.devices("gpu")
    except RuntimeError:
        return []


def detect_platform():
    """Return (platform_str, max_devices) where platform_str is 'gpu' or 'cpu'."""
    g = gpus()
    if g:
        return "gpu", len(g)
    return "cpu", len(jax.devices("cpu"))


def pick_devices(n):
    """Return n devices (GPUs preferred, virtual CPUs otherwise), or None."""
    g = gpus()
    if len(g) >= n:
        return g[:n]
    c = jax.devices("cpu")
    if len(c) >= n:
        return c[:n]
    return None


def default_device_counts(max_devices):
    """Powers-of-two-ish device-count ladder up to max_devices, always incl. 1."""
    counts = [1]
    k = 2
    while k <= max_devices:
        counts.append(k)
        k *= 2
    if max_devices not in counts:
        counts.append(max_devices)
    return sorted(set(c for c in counts if c <= max_devices))


# ── Timing ────────────────────────────────────────────────────────────────────
def time_op(run_fn, warmup=1, trials=3):
    """Time run_fn() over warmup + trials iterations (blocking each result).

    Args:
        run_fn (callable): zero-arg; returns a JAX array (or pytree).
        warmup (int): untimed iterations (compile + caches).
        trials (int): timed iterations.

    Returns:
        (stats, last_result): stats is a dict of min/mean/std in ms; last_result
        is the final returned value (for correctness checking).
    """
    result = None
    times = []
    for i in range(warmup + trials):
        t0 = time.perf_counter()
        result = run_fn()
        jax.block_until_ready(result)
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
    arr = np.array(times) * 1e3
    stats = {"min_ms": float(arr.min()),
             "mean_ms": float(arr.mean()),
             "std_ms": float(arr.std())}
    return stats, result


# ── Memory ────────────────────────────────────────────────────────────────────
def peak_memory_mb(devices):
    """Best-effort peak memory in MB.

    GPU: max over the given devices of memory_stats()['peak_bytes_in_use'] (a
    cumulative high-water mark since process start — meaningful when sizes are
    swept in increasing order within a fresh process).
    CPU: process RSS via getrusage (process-level, approximate; per-device CPU
    memory stats are not available from JAX).

    Returns:
        (value_mb, kind) where kind is 'gpu_peak_per_device' or 'cpu_rss'.
    """
    plat, _ = detect_platform()
    if plat == "gpu":
        peak = 0
        for d in devices:
            try:
                peak = max(peak, int(d.memory_stats().get("peak_bytes_in_use", 0)))
            except Exception:
                pass
        return peak / (1024 ** 2), "gpu_peak_per_device"
    # CPU fallback: maxrss (bytes on macOS, KiB on Linux).
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if _platform.system() == "Darwin":
        rss_mb = rss / (1024 ** 2)   # macOS reports bytes
    else:
        rss_mb = rss / 1024          # Linux reports KiB
    return rss_mb, "cpu_rss"


# ── Correctness ───────────────────────────────────────────────────────────────
def correctness_metrics(reference, output, threshold=1e-4):
    """Compare output to a reference array; tolerate a few bad points.

    Args:
        reference (np.ndarray): the prerelease (or single-device) reference.
        output: a JAX or numpy array of the same shape.
        threshold (float): abs-error level above which a point "differs".

    Returns:
        dict with max_abs_diff, pct_above_threshold, n_above, n_total, threshold.
    """
    ref = np.asarray(reference)
    out = np.asarray(output)
    if ref.shape != out.shape:
        return {"error": f"shape mismatch ref {ref.shape} vs out {out.shape}"}
    diff = np.abs(out - ref)
    n_total = int(diff.size)
    n_above = int(np.count_nonzero(diff > threshold))
    return {"max_abs_diff": float(diff.max()),
            "pct_above_threshold": 100.0 * n_above / n_total,
            "n_above": n_above,
            "n_total": n_total,
            "threshold": float(threshold)}


# ── YAML I/O ──────────────────────────────────────────────────────────────────
_yaml = YAML()
_yaml.default_flow_style = False


def save_yaml(path, data):
    _ensure_dirs()
    with open(path, "w") as f:
        _yaml.dump(_to_plain(data), f)
    print(f"  wrote {path}")


def load_yaml(path):
    with open(path, "r") as f:
        return _yaml.load(f)


def _to_plain(obj):
    """Recursively convert numpy scalars/arrays to plain Python for YAML."""
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_device_sweep(op_name, device_counts, min_ms, mem_mb, mem_kind, out_path):
    """Strong-scaling plot: speedup vs devices, and memory vs devices."""
    device_counts = list(device_counts)
    base = min_ms[0]
    speedup = [base / t for t in min_ms]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(device_counts, speedup, "o-", label="measured")
    ax1.plot(device_counts, device_counts, "k--", alpha=0.5, label="ideal linear")
    ax1.set_xlabel("number of devices")
    ax1.set_ylabel("speedup vs 1 device")
    ax1.set_title(f"{op_name}: strong scaling")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(device_counts, mem_mb, "s-", color="tab:red")
    ax2.set_xlabel("number of devices")
    ax2.set_ylabel(f"peak memory (MB) [{mem_kind}]")
    ax2.set_title(f"{op_name}: memory vs devices")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_size_sweep(op_name, sizes_label, min_ms, mem_mb, mem_kind, out_path):
    """Fixed-device plot: time vs size and peak memory vs size (log-y)."""
    x = list(range(len(sizes_label)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(x, min_ms, "o-")
    ax1.set_xticks(x); ax1.set_xticklabels(sizes_label, rotation=30, ha="right")
    ax1.set_ylabel("min time (ms)")
    ax1.set_yscale("log")
    ax1.set_title(f"{op_name}: time vs size")
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, mem_mb, "s-", color="tab:red")
    ax2.set_xticks(x); ax2.set_xticklabels(sizes_label, rotation=30, ha="right")
    ax2.set_ylabel(f"peak memory (MB) [{mem_kind}]")
    ax2.set_title(f"{op_name}: memory vs size")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ── Problem-size presets ──────────────────────────────────────────────────────
# (n_views, n_rows, n_channels).  n_views and n_rows must be divisible by the
# device counts used.  'quick' keeps a default run short; bigger sets probe
# limits and are opt-in.
SIZE_PRESETS = {
    "cpu": {
        "quick":  [(64, 64, 64), (64, 128, 128)],
        "medium": [(128, 128, 128), (128, 256, 256)],
        "large":  [(180, 512, 512)],
    },
    "gpu": {
        "quick":  [(256, 256, 256), (512, 512, 512)],
        "medium": [(512, 512, 512), (1024, 1024, 1024)],
        "large":  [(1024, 1024, 1024), (2048, 2048, 2048)],
    },
}


def size_label(size):
    v, r, c = size
    return f"{v}x{r}x{c}"
