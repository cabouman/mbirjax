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
import sys
import json
import time
import resource
import tempfile
import subprocess
import platform as _platform

import numpy as np

from ruamel.yaml import YAML

import matplotlib
matplotlib.use("Agg")   # file output only; no interactive backend needed
import matplotlib.pyplot as plt

# NOTE: jax and mbirjax are imported LAZILY, inside the functions that need them
# (gpus / detect_platform / pick_devices / time_op / which_mbirjax).  This keeps
# `import scaling_common` JAX-free so the orchestrator (see the op driver) can
# use the pure helpers — paths, YAML, plots, annotate_*, size_label,
# default_device_counts, run_worker — WITHOUT initializing a JAX backend.  That
# matters on GPU: only the isolated worker subprocesses touch JAX, so the
# orchestrator never holds GPU memory while a worker measures peak usage.
# Workers still import mbirjax before jax (device-setup-first) — they do
# `import mbirjax` at the top of the worker entry, before any sc call that
# triggers the lazy `import jax`.


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
    import mbirjax
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


# ── Subprocess orchestration (worker isolation) ───────────────────────────────
def run_worker(script_path, worker_args, extra_env=None):
    """Run an op driver in --worker mode as an isolated subprocess.

    Each JAX-touching task (device probe, correctness, one size's measurement)
    runs in its own fresh process so the orchestrator never holds a JAX backend
    while a worker measures peak memory.  The worker writes its result as JSON to
    a temp file (passed via --out-file) and may rewrite it incrementally, so a
    worker that dies partway (e.g. GPU OOM at the largest config) still returns
    whatever it completed.  The child inherits the current environment plus
    extra_env; the caller is responsible for putting the beta worktree on
    PYTHONPATH so the worker's `import mbirjax` resolves to beta.

    Args:
        script_path (str): absolute path to the op driver (its own __file__).
        worker_args (list[str]): args after the script, e.g.
            ['--worker', '--mode', 'measure', '--size', '256x256x256', ...].
            '--out-file <tmp>' is appended automatically.
        extra_env (dict|None): environment overrides for the child.

    Returns:
        (result, returncode): result is the parsed JSON (or None if the worker
        wrote nothing parseable); returncode is the subprocess exit status.
    """
    # Flush any pending orchestrator output first so the worker's live stdout
    # interleaves in the right order even when stdout is a pipe (PyCharm console).
    sys.stdout.flush()
    fd, out_path = tempfile.mkstemp(suffix=".json", prefix="scaling_worker_")
    os.close(fd)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    cmd = [sys.executable, script_path, *worker_args, "--out-file", out_path]
    proc = subprocess.run(cmd, env=env)
    result = None
    try:
        with open(out_path) as f:
            result = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        result = None
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)
    return result, proc.returncode


def write_worker_result(out_file, data):
    """Worker side: atomically (re)write a JSON result to out_file.

    Written via a temp file + os.replace so a reader (the orchestrator) never
    sees a half-written file even if the worker is killed mid-write.  Safe to
    call repeatedly to publish partial progress.
    """
    tmp = out_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, out_file)


# ── Device / platform detection ───────────────────────────────────────────────
def gpus():
    """List of GPU devices, or [] if no GPU backend."""
    import jax
    try:
        return jax.devices("gpu")
    except RuntimeError:
        return []


def detect_platform():
    """Return (platform_str, max_devices) where platform_str is 'gpu' or 'cpu'."""
    import jax
    g = gpus()
    if g:
        return "gpu", len(g)
    return "cpu", len(jax.devices("cpu"))


def pick_devices(n):
    """Return n devices (GPUs preferred, virtual CPUs otherwise), or None."""
    import jax
    g = gpus()
    if len(g) >= n:
        return g[:n]
    c = jax.devices("cpu")
    if len(c) >= n:
        return c[:n]
    return None


def device_label():
    """Human-readable device label for plot titles.

    Returns e.g. 'CPU (cpu)' on a CPU backend or 'GPU (NVIDIA H100 80GB HBM3)'
    on a GPU backend, using the first available device's reported kind.
    """
    plat, _ = detect_platform()
    devs = pick_devices(1)
    kind = devs[0].device_kind if devs else "?"
    return f"{plat.upper()} ({kind})"


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
    import jax
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


# ── Speedup / scaling ─────────────────────────────────────────────────────────
def annotate_speedups(rows, time_key="min_ms", base_key="n_devices", base_val=1):
    """Add a 'speedup' field to each row, relative to the 1-device run.

    speedup = base_time / row_time, where the baseline is the row whose
    base_key equals base_val (the 1-device run by default).  If no such row is
    present (e.g. a custom device sweep that omits 1 device), fall back to the
    row with the smallest base_key value and print a one-line note, so the
    reported factor is never silently mislabeled as "vs 1 device".

    Args:
        rows (list[dict]): sweep rows, each containing base_key and time_key.
        time_key (str): timing field to ratio (default 'min_ms', the best time).
        base_key (str): field identifying the baseline row (default 'n_devices').
        base_val: baseline value to look for (default 1 = single device).

    Returns:
        The base_key value actually used as the reference (base_val, or the
        smallest present if base_val is absent), or None if rows is empty.
    """
    if not rows:
        return None
    base_row = next((r for r in rows if r.get(base_key) == base_val), None)
    if base_row is None:
        base_row = min(rows, key=lambda r: r[base_key])
        print(f"  (note: no {base_key}={base_val} run; reporting speedup "
              f"relative to {base_key}={base_row[base_key]})")
    base_time = base_row[time_key]
    for r in rows:
        r["speedup"] = base_time / r[time_key]
    return base_row[base_key]


def annotate_mem_fraction(rows, mem_key="mem_mb", base_key="n_devices", base_val=1):
    """Add a 'mem_frac' field: peak memory relative to the 1-device run.

    mem_frac = row_mem / base_mem, same baseline-selection rule as
    annotate_speedups (fall back to the smallest device count if base_val is
    absent).

    CAVEAT: the underlying peak is a *process-cumulative high-water mark* (CPU
    RSS via getrusage; single-process GPU peak_bytes_in_use), so within one
    process it does not reset between configs — device 0 participates in every
    run, so its lifetime peak tends to equal the largest (1-device) run.  This
    fraction is therefore only a rough indicator and will often read ~1.0/flat;
    a faithful per-device measurement needs a fresh subprocess per config.
    """
    if not rows:
        return None
    base_row = next((r for r in rows if r.get(base_key) == base_val), None)
    if base_row is None:
        base_row = min(rows, key=lambda r: r[base_key])
    base_mem = base_row[mem_key]
    for r in rows:
        r["mem_frac"] = (r[mem_key] / base_mem) if base_mem else float("nan")
    return base_row[base_key]


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


# ── Baselines (reference output: array → .npy, metadata → .yaml) ──────────────
def save_baseline(op_name, array, meta):
    """Store a single reference output for op_name.

    The numeric array goes to ``<op>.npy`` (compact, exact binary); human-
    readable metadata (capture platform, prerelease path, size, seed, shape,
    dtype, timing) goes to ``<op>.yaml``.  There is ONE baseline per op, not one
    per platform: every beta run (CPU or GPU) compares against this same
    reference, so a significant CPU/GPU divergence shows up as a real difference
    instead of being hidden by per-platform self-comparison.  The capture
    platform is kept in the metadata so cross-platform comparisons are labeled.

    Args:
        op_name (str): operation name, used for the file stem.
        array: the reference output (JAX or numpy); stored at its own dtype.
        meta (dict): metadata to record alongside (platform, seed, size, ...).

    Returns:
        (npy_path, yaml_path)
    """
    _ensure_dirs()
    arr = np.asarray(array)
    npy_path = os.path.join(BASELINES_DIR, f"{op_name}.npy")
    yaml_path = os.path.join(BASELINES_DIR, f"{op_name}.yaml")
    np.save(npy_path, arr)
    full = {**meta,
            "npy_file": os.path.basename(npy_path),
            "output_shape": list(arr.shape),
            "output_dtype": str(arr.dtype)}
    save_yaml(yaml_path, full)
    print(f"  wrote baseline array {npy_path}  shape={arr.shape} dtype={arr.dtype}")
    return npy_path, yaml_path


def load_baseline(op_name):
    """Load the reference output saved by save_baseline.

    Returns:
        (array, meta): array is a numpy ndarray, meta is the YAML metadata dict.
        Returns (None, None) if either file is missing.
    """
    npy_path = os.path.join(BASELINES_DIR, f"{op_name}.npy")
    yaml_path = os.path.join(BASELINES_DIR, f"{op_name}.yaml")
    if not (os.path.exists(npy_path) and os.path.exists(yaml_path)):
        return None, None
    meta = load_yaml(yaml_path)
    arr = np.load(npy_path)
    return arr, meta


# ── Plotting ──────────────────────────────────────────────────────────────────
def _grid_lookup(grid, size_label):
    """Return {n_devices: row} for one size from the measurement grid."""
    return {r["n_devices"]: r for r in grid.get(size_label, [])}


def plot_device_sweep(op_name, grid, device_counts, sizes, dev_label,
                      mem_kind, out_path):
    """Device sweep: speedup and fractional memory vs device count, per size.

    One curve per problem size.  Left: speedup vs devices (with the ideal-linear
    reference).  Right: peak memory as a FRACTION of the 1-device value (ideal
    sharding drives per-device memory toward 1/n).  See annotate_mem_fraction's
    caveat: the underlying peak is a process-cumulative high-water mark, so this
    fraction is only a rough indicator (often ~1.0) until measured with
    per-config subprocess isolation.

    Args:
        grid (dict): size_label -> list of row dicts (n_devices, speedup,
            mem_frac, ...), as produced by the driver.
        device_counts (list[int]): x-axis device counts (ascending).
        sizes (list[str]): size labels, in legend order.
        dev_label (str): device type for the suptitle (see device_label()).
    """
    device_counts = list(device_counts)
    base_dev = device_counts[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    for size_label in sizes:
        rows = _grid_lookup(grid, size_label)
        xs = [n for n in device_counts if n in rows]
        ax1.plot(xs, [rows[n]["speedup"] for n in xs], "o-", label=size_label)
        ax2.plot(xs, [rows[n].get("mem_frac", float("nan")) for n in xs],
                 "s-", label=size_label)
    ax1.plot(device_counts, device_counts, "k--", alpha=0.5, label="ideal linear")

    ax1.set_xlabel("number of devices")
    ax1.set_ylabel(f"speedup vs {base_dev} device" + ("s" if base_dev != 1 else ""))
    ax1.set_title("speedup vs devices")
    ax1.legend(title="size (v×r×c)")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("number of devices")
    if mem_kind == "gpu_peak_per_device":
        # GPU peak_bytes_in_use is per-device → fraction is physically meaningful.
        ax2.set_ylabel(f"peak mem/device ÷ {base_dev}-device")
        ax2.set_title("per-device memory vs devices (fraction of 1-device)")
    else:
        # CPU RSS is whole-process and shares one RAM pool across virtual
        # devices, so this fraction is NOT per-device savings — label honestly.
        ax2.set_ylabel(f"process RSS ÷ {base_dev}-device  [{mem_kind}]")
        ax2.set_title("memory vs devices (process RSS — not per-device)")
    ax2.legend(title="size (v×r×c)")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"{op_name} — device sweep — {dev_label}", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_size_sweep(op_name, grid, device_counts, sizes, dev_label,
                    mem_kind, out_path):
    """Size sweep: time and peak memory vs problem size, one curve per device count.

    Left: min time vs size (log-y).  Right: peak memory vs size.

    Args:
        grid (dict): size_label -> list of row dicts, as produced by the driver.
        device_counts (list[int]): one curve per count.
        sizes (list[str]): size labels along the x-axis, in order.
        dev_label (str): device type for the suptitle.
    """
    x = list(range(len(sizes)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    for n in device_counts:
        tms, mem = [], []
        for size_label in sizes:
            row = _grid_lookup(grid, size_label).get(n)
            tms.append(row["min_ms"] if row else float("nan"))
            mem.append(row["mem_mb"] if row else float("nan"))
        ax1.plot(x, tms, "o-", label=f"{n} dev")
        ax2.plot(x, mem, "s-", label=f"{n} dev")

    ax1.set_xticks(x); ax1.set_xticklabels(sizes, rotation=30, ha="right")
    ax1.set_ylabel("min time (ms)")
    ax1.set_yscale("log")
    ax1.set_title("time vs size")
    ax1.legend(title="devices")
    ax1.grid(True, alpha=0.3)

    ax2.set_xticks(x); ax2.set_xticklabels(sizes, rotation=30, ha="right")
    ax2.set_ylabel(f"peak memory (MB) [{mem_kind}]")
    ax2.set_yscale("log")
    ax2.set_title("memory vs size")
    ax2.legend(title="devices")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"{op_name} — size sweep — {dev_label}", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ── Problem-size label ────────────────────────────────────────────────────────
# Problem-size *sets* now live at the top of each op driver (different ops want
# different sizes); scaling_common only provides the shared label formatter.
def size_label(size):
    v, r, c = size
    return f"{v}x{r}x{c}"
