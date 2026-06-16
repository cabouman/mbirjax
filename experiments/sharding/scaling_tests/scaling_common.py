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
import gc
import sys
import time
import resource
import tempfile
import traceback
import subprocess
import platform as _platform

import numpy as np

from ruamel.yaml import YAML, YAMLError

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
def mbirjax_git_branch(pkg_path):
    """Git branch of the checkout containing pkg_path, or None if undetermined.

    Returns None when pkg_path is not a git checkout, git is unavailable, or HEAD
    is detached (rev-parse returns 'HEAD').
    """
    try:
        out = subprocess.run(
            ["git", "-C", pkg_path, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            branch = out.stdout.strip()
            return branch if branch and branch != "HEAD" else None
    except Exception:
        pass
    return None


def pyproject_version(root):
    """Project version string from ``<root>/pyproject.toml`` (or None).

    ``root`` is the package root (one dir up from the ``mbirjax/`` package), so this matches the
    LOADED mbirjax — including a PYTHONPATH override to a main worktree, where it reads main's
    version (e.g. 0.6.17.1) rather than the editable install's.
    """
    import re
    try:
        with open(os.path.join(root, "pyproject.toml")) as f:
            m = re.search(r'^\s*version\s*=\s*["\']([^"\']+)["\']', f.read(), re.M)
        return m.group(1) if m else None
    except Exception:
        return None


def beta_status(pkg_path):
    """Identify whether the loaded mbirjax is the beta sharding code.

    Detection is by GIT BRANCH, not directory name, so it works regardless of
    where the worktree lives — locally the beta worktree is 'mbirjax_sharding',
    but on the cluster the beta branch may be checked out into a plain 'mbirjax'
    directory.  Beta == a branch whose name contains 'sharding' (the beta branch
    is greg/parallel_sharding; research is greg/parallel_tests; prerelease is
    'prerelease').

    Returns:
        (state, branch): state is 'beta', 'not-beta', or 'unknown' (branch could
        not be determined — e.g. detached HEAD or no git).
    """
    branch = mbirjax_git_branch(pkg_path)
    if branch is None:
        return "unknown", None
    return ("beta" if "sharding" in branch else "not-beta"), branch


def which_mbirjax():
    """Print a clear beta/not-beta banner for the resolved mbirjax (by branch).

    Returns:
        str: the mbirjax package path.
    """
    import mbirjax
    path = os.path.dirname(mbirjax.__file__)
    state, branch = beta_status(path)
    print("=" * 72)
    if state == "beta":
        print(f"  mbirjax code in use:  *** beta ***  (branch {branch})")
    elif state == "not-beta":
        print(f"  mbirjax code in use:  ### NOT beta ###  (branch {branch})")
        print("  !! WARNING: results will reflect NON-beta code.  Check PYTHONPATH /")
        print("  !! working directory so the beta sharding branch is loaded.")
    else:
        print("  mbirjax code in use:  (branch undetermined — verify path manually)")
    print(f"  path: {path}")
    print("=" * 72)
    return path


# ── Subprocess orchestration (worker isolation) ───────────────────────────────
def run_worker(script_path, worker_args, extra_env=None):
    """Run an op driver in --worker mode as an isolated subprocess.

    Each JAX-touching task (device probe, correctness, one size's measurement)
    runs in its own fresh process so the orchestrator never holds a JAX backend
    while a worker measures peak memory.  The worker writes its result as YAML to
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
        (result, returncode): result is the parsed YAML (or None if the worker
        wrote nothing parseable); returncode is the subprocess exit status.
    """
    # Flush any pending orchestrator output first so the worker's live stdout
    # interleaves in the right order even when stdout is a pipe (PyCharm console).
    sys.stdout.flush()
    fd, out_path = tempfile.mkstemp(suffix=".yaml", prefix="scaling_worker_")
    os.close(fd)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    cmd = [sys.executable, script_path, *worker_args, "--out-file", out_path]
    proc = subprocess.run(cmd, env=env)
    result = None
    try:
        with open(out_path) as f:
            result = _yaml.load(f)   # None for an empty/never-written file
    except (FileNotFoundError, YAMLError, ValueError):
        result = None
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)
    return result, proc.returncode


def write_worker_result(out_file, data):
    """Worker side: atomically (re)write a YAML result to out_file.

    Written via a temp file + os.replace so a reader (the orchestrator) never
    sees a half-written file even if the worker is killed mid-write.  Safe to
    call repeatedly to publish partial progress.  Uses the same ruamel YAML
    instance as the rest of the harness (readability + consistency); numpy
    scalars are converted to plain Python first via _to_plain so they serialize
    cleanly.  (_yaml and _to_plain are module-level, defined below and resolved
    at call time.)
    """
    tmp = out_file + ".tmp"
    with open(tmp, "w") as f:
        _yaml.dump(_to_plain(data), f)
    os.replace(tmp, out_file)


# ── Shared scaling-driver harness ─────────────────────────────────────────────
# The pieces every scaling driver (fbp_filter, sparse_back/forward_project,
# direct_recon, vcd_recon) shares.  A driver supplies only the op-specific shims
# (make_model / make_input / run_op / a correctness check) plus its size/device
# knobs; the orchestration, isolated-subprocess discipline, and measurement loop
# live here so all ops behave identically.

# Substrings (upper-cased) marking a caught failure as memory exhaustion.  Beyond
# the clean allocator tokens, GPU FBP hits cuFFT OOM, which XLA surfaces as
# "Failed to create cuFFT batched plan with scratch allocator" / "Failed to
# allocate work area" -- none of the usual OOM tokens (confirmed H100 1624^3/1dev).
OOM_MARKERS = ("RESOURCE_EXHAUSTED", "OUT OF MEMORY", "OOM", "BAD_ALLOC",
               "FAILED TO ALLOCATE", "WORK AREA", "SCRATCH ALLOCATOR",
               "FAILED TO CREATE CUFFT")


def is_oom(text):
    """True if ``text`` names a known out-of-memory marker.

    Prefer passing the full traceback rather than ``str(e)``: an OOM often
    surfaces as an unrelated-looking error (e.g. a numpy "setting an array element
    with a sequence") with the real RESOURCE_EXHAUSTED only visible deeper in the
    stack.
    """
    up = text.upper()
    return any(k in up for k in OOM_MARKERS)


def beta_root():
    """Beta worktree root, derived from this file's location.

    scaling_common.py lives at <beta>/experiments/sharding/scaling_tests/, so the
    worktree root is three directories up from its directory.  The drivers sit in
    the same directory, so this is the same root they used to derive themselves.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        os.pardir, os.pardir, os.pardir))


def build_worker_env(mem_fraction=0.9, preallocate=True, lib_root=None):
    """Orchestrator side: the environment every worker subprocess inherits.

    Forces a mbirjax checkout onto PYTHONPATH so ``import mbirjax`` resolves to it
    regardless of how the orchestrator was launched (PyCharm or CLI), and sets the
    JAX allocator knobs.  ``lib_root`` selects WHICH checkout: pass it to measure a
    DIFFERENT branch's library (e.g. the nightly points it at a per-branch worktree,
    so the same harness — which may live in mbirjax_metrics, not next to mbirjax —
    measures main / prerelease / a dev branch).  Default (None) keeps the historical
    behavior: ``beta_root()``, the checkout this harness lives in.  Preallocating the
    pool up front avoids per-call cudaMalloc growth (clean timing); peak_bytes_in_use
    still tracks in-use tensors so memory stays accurate.  Lower ``mem_fraction`` to
    probe the OOM threshold.  Warns if no mbirjax/ is found under the chosen root.
    """
    root = lib_root or beta_root()
    if not os.path.isdir(os.path.join(root, "mbirjax")):
        print(f"  WARNING: no mbirjax/ under derived beta root {root}")
    existing = os.environ.get("PYTHONPATH", "")
    return {
        "PYTHONPATH": root + (os.pathsep + existing if existing else ""),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true" if preallocate else "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": str(mem_fraction),
    }


def build_setup_result(plat, max_dev, dev_label, corr):
    """Worker side: assemble + print the standard setup-result dict.

    Records which mbirjax / git branch is loaded, and on GPU snapshots the
    topology (physical GPUs / interconnect / NUMA -- the allocation-quality
    variable behind run-to-run multi-device scaling surprises) and the
    device-to-device safety probe.  ``corr`` is the op-specific correctness dict
    (already computed and printed by the caller).  Returns the dict to write.
    """
    import mbirjax
    topology = gpu_topology() if plat == "gpu" else {}
    dev2dev_safe = None
    if plat == "gpu":
        try:
            import jax
            import mbirjax._sharding as mjs
            g = jax.devices("gpu")
            if len(g) > 1:
                dev2dev_safe = bool(mjs.is_dev2dev_safe(g))
        except Exception:   # noqa: BLE001 — best effort, never abort setup
            pass
    pkg_path = os.path.dirname(mbirjax.__file__)
    state, branch = beta_status(pkg_path)   # by git branch, not dir name
    result = {"platform": plat, "max_devices": max_dev, "device_label": dev_label,
              "mbirjax_path": pkg_path, "beta_state": state, "branch": branch,
              "correctness": corr, "topology": topology, "dev2dev_safe": dev2dev_safe}
    print(f"[setup] platform={plat}  max_devices={max_dev}  ({dev_label})")
    if dev2dev_safe is not None:
        print(f"[setup] dev2dev_safe={dev2dev_safe}"
              + ("" if dev2dev_safe else "  <-- HOST-BOUNCE active (slow d2d!)"))
    if topology.get("devices"):
        print("[setup] GPUs:\n    " + topology["devices"].replace("\n", "\n    "))
    return result


def print_setup_banner(setup):
    """Orchestrator side: print the mbirjax / beta / platform / correctness /
    topology banner from a setup-worker result, and return the common fields
    ``(plat, max_dev, dev_label, corr, mpath)``.  Topology / dev2dev lines print
    only when the worker recorded them (GPU runs).
    """
    plat = setup["platform"]
    max_dev = setup["max_devices"]
    dev_label = setup["device_label"]
    corr = setup["correctness"]
    mpath = setup.get("mbirjax_path", "?")
    state = setup.get("beta_state", "unknown")
    branch = setup.get("branch")
    if state == "beta":
        label = f"*** beta ***  (branch {branch})"
    elif state == "not-beta":
        label = f"### NOT beta — branch {branch} — check PYTHONPATH ###"
    else:
        label = "(branch undetermined — verify path manually)"
    print(f"  mbirjax: {label}   {mpath}")
    print(f"  platform: {plat}   max devices: {max_dev}   ({dev_label})")
    if corr.get("baseline_present") and corr.get("max_abs_diff") is not None:
        print(f"  correctness: max_abs_diff={corr['max_abs_diff']:.3e}  "
              f"pct_above={corr['pct_above_threshold']:.6f}%"
              + ("   <-- CROSS-PLATFORM" if corr.get("cross_platform") else ""))
    elif corr.get("baseline_present"):
        print("  correctness: baseline present (see setup log for details)")
    else:
        print("  correctness: no baseline present")
    dev2dev_safe = setup.get("dev2dev_safe")
    if dev2dev_safe is not None:
        print(f"  dev2dev_safe: {dev2dev_safe}"
              + ("" if dev2dev_safe else "  <-- HOST-BOUNCE active (slow d2d!)"))
    topology = setup.get("topology") or {}
    if topology.get("topo"):
        print("  GPU topology (nvidia-smi topo -m):")
        print("    " + topology["topo"].replace("\n", "\n    "))
    return plat, max_dev, dev_label, corr, mpath


def run_measure_loop(size_label, device_counts, out_file, build_and_time,
                     header_extra="", print_traceback=True):
    """Worker side: the shared device-count descent for one problem size.

    Owns what every op's measure shares: iterate device counts DESCENDING
    (8->4->2->1, so per-device allocation is ascending within this fresh process
    and the cumulative peak_bytes_in_use equals each config's own allocation when
    read right after it), catch/classify failures (an OOM stops the descent --
    fewer-device configs need MORE per-device memory and would also OOM), sample
    per-GPU clocks/temps and flag throttling (which silently caps multi-device
    scaling), publish partial results incrementally (so even a hard crash returns
    the completed configs), and free device buffers between configs.

    The op supplies ``build_and_time(n, devs)``, which builds its model for this
    device count, prepares the timed input, times the op, and returns
    ``(stats, mem_mb, mem_kind)`` -- or ``None`` to skip this device count (e.g.
    the count does not evenly divide the sharded axes).  Anything it raises is
    treated as a measurement failure, classified for OOM from the full traceback.

    Returns ``(rows, failures)``; both are also written to ``out_file`` as they grow.
    """
    desc = sorted(set(device_counts), reverse=True)
    print(f"\n[measure {size_label}{header_extra}]  "
          f"device counts (descending): {desc}")
    rows, failures = [], []
    mem_kind = "n/a"

    def _publish():
        write_worker_result(out_file, {"size": size_label, "mem_kind": mem_kind,
                                       "rows": rows, "failures": failures})

    for n in desc:
        devs = pick_devices(n)
        if devs is None:
            print(f"  n_devices={n}: not enough devices, skipping")
            continue
        try:
            timed = build_and_time(n, devs)
        except Exception as e:   # noqa: BLE001 — harness: never abort the sweep
            msg = str(e).replace("\n", " ")
            tb = traceback.format_exc()
            oom = is_oom(tb)   # classify from the FULL stack, not just str(e)
            failures.append({"n_devices": n, "oom": oom, "error": msg[:300],
                             "traceback": tb})
            print(f"  n_devices={n:2d}  {'OOM' if oom else 'ERROR'}: {msg[:120]}")
            if not oom and print_traceback:
                # Full traceback for a real failure (don't truncate to one line).  A caller that
                # EXPECTS failures (e.g. performance_tracking's known cone-padding cells) passes
                # print_traceback=False for a clean one-line report; the full tb is still stored
                # in the failure dict above, so nothing is lost.
                print(tb)
            _publish()
            if oom:
                print(f"  stopping descent at {size_label}: fewer-device configs "
                      f"need more per-device memory and would also OOM")
                break
            continue
        if timed is None:   # op signalled "skip this device count"
            continue
        stats, mem_mb, mem_kind = timed
        gpu_state = sample_gpu_state()
        hot = throttled_gpus(gpu_state)
        rows.append({"n_devices": n, **stats, "mem_mb": mem_mb,
                     "gpu_state": gpu_state, "throttled": bool(hot)})
        print(f"  n_devices={n:2d}  min={stats['min_ms']:9.1f} ms  "
              f"mean={stats['mean_ms']:9.1f} ms  mem={mem_mb:8.1f} MB ({mem_kind})")
        if hot:
            print("  !! THROTTLING — this timing is UNRELIABLE: "
                  + ", ".join(f"GPU{g['index']}={g['sm_mhz']}MHz@{g['temp_c']}C"
                              for g in hot))
        _publish()
        gc.collect()   # release this config's device buffers before the next
    _publish()
    return rows, failures


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


def gpu_topology():
    """Best-effort GPU topology snapshot for reproducibility.

    Records which physical GPUs the scheduler handed us (UUIDs, via
    ``nvidia-smi -L``) and how they interconnect (``nvidia-smi topo -m``).
    Cross-allocation performance can hinge on this -- e.g. all GPUs on one NUMA
    socket vs split across two changes host-side launch latency, which hits the
    launch-heavy multi-device paths most -- so we log it next to every result.
    Returns ``{}`` when nvidia-smi is unavailable (e.g. CPU runs).
    """
    out = {}
    for key, cmd in (("devices", ["nvidia-smi", "-L"]),
                     ("topo", ["nvidia-smi", "topo", "-m"])):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if r.returncode == 0:
                out[key] = r.stdout.strip()
        except Exception:   # nvidia-smi missing / CPU node — best effort
            pass
    return out


def sample_gpu_state():
    """Per-GPU SM clock (MHz) and temperature (C) via nvidia-smi.

    A throttled card under load shows a collapsed SM clock at high temperature
    (e.g. 345 MHz @ 86 C vs a healthy 1980 MHz @ 40 C), which silently caps
    multi-device scaling.  Sampling this with each measurement lets a bad card
    auto-flag itself in the result instead of masquerading as a code regression.
    Returns a list of ``{index, sm_mhz, temp_c}`` (one per GPU), or ``[]`` when
    nvidia-smi is unavailable (CPU runs).
    """
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,clocks.sm,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return []
        out = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                out.append({"index": int(parts[0]), "sm_mhz": int(parts[1]),
                            "temp_c": int(parts[2])})
        return out
    except Exception:   # nvidia-smi missing / CPU node — best effort
        return []


def throttled_gpus(state, clock_below=1500, temp_above=80):
    """GPUs in ``state`` that look thermally throttled (low SM clock + high temp).

    The clock+temp pair discriminates throttling from a normal idle low-clock:
    an idle card is also low-clock but COOL, while a throttling card is low-clock
    and HOT.  Returns the sublist of throttled GPU dicts.
    """
    return [g for g in state
            if g["sm_mhz"] < clock_below and g["temp_c"] > temp_above]


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

    Memory note: we drop the PREVIOUS iteration's result before allocating the
    next one, so the device peak reflects a single call (input + output), not two
    outputs alive at once.  Without this, peak_bytes_in_use over-reports by a full
    output (one shard): the loop holds the prior result while the next run_fn
    allocates its output, on top of the persistent input.  Freeing is by refcount
    when the name is dropped; gc.collect() is belt-and-suspenders and sits outside
    the timed region so it cannot perturb the timing.
    """
    import gc
    import jax
    result = None
    times = []
    for i in range(warmup + trials):
        result = None      # free the prior output before the next allocation
        gc.collect()       # insurance for any lingering ref; outside the timed region
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


def _label_volume(size_label):
    """Total voxels (v·r·c) for a 'VxRxC' size label — the cost variable that
    fbp_filter time and memory scale with, used as the size-sweep x-axis."""
    v, r, c = (int(x) for x in size_label.split("x"))
    return v * r * c


def _label_proj_cost(size_label):
    """Projection compute cost for a 'VxRxC' size: num_voxels × num_views.

    Tomographic forward/back projection touches each voxel once per view, so its
    cost scales as voxels × views, NOT voxels alone.  For the cubic sweep sizes
    (N×N×N) this is N⁴ vs the volume's N³ — the extra factor of N (the views axis,
    the first label component) is why doubling the linear size raises projection
    time ~16×, not 8×.  Used for the size-sweep TIME ideal curve; MEMORY still
    scales with volume (resident sino+recon), so it keeps using _label_volume.
    """
    v, r, c = (int(x) for x in size_label.split("x"))
    return (v * r * c) * v   # voxels × views (v is the views axis)


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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    oom_notes = []
    for size_label in sizes:
        rows = _grid_lookup(grid, size_label)
        xs = [n for n in device_counts if n in rows]
        if not xs:
            continue
        size_base = xs[0]   # smallest device count measured for THIS size
        # Anchor each curve so its first measured point sits on the ideal line.
        # The stored speedup is 1.0 at size_base, so ×size_base puts it at
        # (size_base, size_base).  For size_base==1 this is the ordinary speedup
        # vs 1 device; for size_base>1 (1-device OOM'd) the curve starts on the
        # ideal line — clearer than implying 2 devices gave no speedup — and we
        # add an OOM note rather than a misleading "1.0x at 2 devices".
        ax1.plot(xs, [rows[n]["speedup"] * size_base for n in xs], "o-",
                 label=size_label)
        # Memory as a multiple of the per-device data shard: peak / shard, where
        # shard = float32 sinogram bytes / n_devices.  This shows the filter's
        # memory overhead above the data it holds (ideal ≈ a small constant — the
        # input + output shards + bounded FFT work area), independent of size.
        vol_bytes = _label_volume(size_label) * 4   # float32 sinogram, total bytes
        ax2.plot(xs, [rows[n]["mem_mb"] / (vol_bytes / n / (1024 ** 2)) for n in xs],
                 "s-", label=size_label)
        if size_base > 1:
            oom_notes.append(f"{size_label}: 1-device OOM "
                             f"(anchored to ideal at {size_base} dev)")
    ax1.plot(device_counts, device_counts, "k--", alpha=0.5, label="ideal linear")

    ax1.set_xlabel("number of devices")
    ax1.set_ylabel("speedup vs 1 device")
    ax1.set_title("speedup vs devices")
    ax1.legend(title="size (v×r×c)", loc="upper left")
    ax1.grid(True, alpha=0.3)
    if oom_notes:
        ax1.text(0.98, 0.02, "\n".join(oom_notes), transform=ax1.transAxes,
                 va="bottom", ha="right", fontsize=7.5, color="dimgray",
                 bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))

    ax2.set_xlabel("number of devices")
    # Ideal is a flat ~2× (input shard + output shard, negligible work area),
    # independent of device count; a kernel whose work area is geometry-bound
    # (per_view) rises above it as the shard shrinks.  Shown on both platforms.
    ax2.axhline(2.0, ls="--", color="gray", alpha=0.7,
                label="ideal (2× = read+write)")
    ax2.set_title("per-device memory ÷ sino shard size")
    if mem_kind == "gpu_peak_per_device":
        ax2.set_ylabel("peak mem/device ÷ sino shard")
    else:
        # CPU RSS is whole-process / shared RAM, so the ratio is not truly
        # per-device — the y-label flags the metric (the title is kept uniform).
        ax2.set_ylabel(f"process RSS ÷ shard  [{mem_kind}]")
    ax2.legend(title="size (v×r×c)")
    ax2.grid(True, alpha=0.3)

    # Tick only at the integer device counts present in the data (1, 2, 4, …),
    # not matplotlib's auto 1.0/1.5/2.0/… floats.
    for ax in (ax1, ax2):
        ax.set_xticks(device_counts)
        ax.set_xticklabels([str(n) for n in device_counts])

    fig.suptitle(f"{op_name} — device sweep — {dev_label}", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_size_sweep(op_name, grid, device_counts, sizes, dev_label,
                    mem_kind, out_path, time_ideal="voxels_views"):
    """Size sweep: time and peak memory vs problem size, one curve per device count.

    The x-axis is the true problem size in voxels (v·r·c) on a LOG scale, so the
    spacing reflects the real size ratios (e.g. ×8 then ×4) instead of equal
    categorical steps; both panels are then log-log and the scaling slope is
    readable.  Ticks are labeled with the size strings at their true positions.

    Time is plotted in minutes, per-device memory in GB.

    TIME panel: the y-range is chosen per run — top = the smallest power of 10 that
    holds the largest measured time, bottom = top / 1e4 (four decades) — and the
    ideal line is anchored at that bottom-left corner, so the data rides above a
    reference of the EXPECTED slope.  ``time_ideal`` picks that slope:
      - "voxels_views" (default): projection cost ∝ voxels × views (N⁴ for cubic
        sizes) — correct for the projectors (back/forward/direct) and VCD, which
        touch each voxel once per view.
      - "voxels": ∝ voxels (N³) — correct for fbp_filter, a per-view filter whose
        cost is the sinogram size, not a projection.

    MEMORY panel: fixed y-range 0.1 .. 100 GB (per-device memory is comparable
    across runs), with the ∝voxels (N³, resident sino+recon) ideal anchored at
    0.1 GB at the smallest size.

    Args:
        grid (dict): size_label -> list of row dicts, as produced by the driver.
        device_counts (list[int]): one curve per count.
        sizes (list[str]): size labels, in order.
        dev_label (str): device type for the suptitle.
        time_ideal (str): "voxels_views" (default) or "voxels"; the slope of the
            time-panel ideal line (see above).
    """
    if time_ideal not in ("voxels", "voxels_views"):
        raise ValueError(
            f"time_ideal must be 'voxels' or 'voxels_views', got {time_ideal!r}")

    # Stored results use ms and MB; plot in the more intuitive minutes and GB.
    MS_PER_MIN = 60_000.0
    MB_PER_GB = 1024.0

    vols = [_label_volume(s) for s in sizes]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    all_tmin = []
    for n in device_counts:
        tmin, memgb = [], []
        for size_label in sizes:
            row = _grid_lookup(grid, size_label).get(n)
            tmin.append(row["min_ms"] / MS_PER_MIN if row else float("nan"))
            memgb.append(row["mem_mb"] / MB_PER_GB if row else float("nan"))
        all_tmin.extend(tmin)
        ax1.plot(vols, tmin, "o-", label=f"{n} dev")
        ax2.plot(vols, memgb, "s-", label=f"{n} dev")

    # TIME y-range: four decades ending at the smallest power of 10 that holds the
    # largest measured time.  floor(log10)+1 (not ceil) so a value that is itself a
    # power of 10 still clears the top spine; fall back to 0.1 .. 1000 if nothing
    # was measured (all-OOM run).
    finite_t = [t for t in all_tmin if np.isfinite(t)]
    t_top = 10.0 ** (np.floor(np.log10(max(finite_t))) + 1) if finite_t else 1000.0
    t_bottom = t_top / 1e4

    # Ideal references: a fixed-slope line anchored at each panel's BOTTOM-LEFT, so
    # the data rides above a reference of the expected slope (identical across runs).
    # TIME slope per `time_ideal`; MEMORY ∝ voxels (resident sino+recon, N³).
    if time_ideal == "voxels_views":
        tcost = [_label_proj_cost(s) for s in sizes]   # ∝ voxels × views (N⁴)
        tlabel = "ideal (∝ voxels·views)"
    else:
        tcost = vols                                   # ∝ voxels (N³)
        tlabel = "ideal (∝ voxels)"
    ax1.plot(vols, [t_bottom * c / tcost[0] for c in tcost], "k--", alpha=0.5,
             label=tlabel)
    ax2.plot(vols, [0.1 * v / vols[0] for v in vols], "k--", alpha=0.5,
             label="ideal (∝ voxels)")

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xticks(vols)
        ax.set_xticklabels(sizes, rotation=30, ha="right")
        ax.set_xlabel("problem size (voxels v·r·c, log scale)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    ax1.set_ylim(t_bottom, t_top)
    ax1.set_ylabel("min execution time (minutes)")
    ax1.set_title("time vs size")
    ax1.legend(title="devices")

    ax2.set_ylim(0.1, 100)
    ax2.set_ylabel("peak memory (GB)")
    ax2.set_title("per-device memory vs size")
    ax2.legend(title="devices")

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
