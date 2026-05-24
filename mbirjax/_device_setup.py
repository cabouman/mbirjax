"""
mbirjax._device_setup
─────────────────────
Runs automatically when mbirjax is imported.  Must be imported as the very
first line of mbirjax/__init__.py, before any JAX import reaches this process.

What this module does
─────────────────────
On CPU-only machines it sets XLA_FLAGS so JAX sees one virtual device per
available CPU core, enabling multi-device sharding tests without real GPUs.
On GPU machines the flag is set but ignored by the GPU backend, so it is
always safe to apply.

If JAX was already imported before mbirjax (only a problem on CPU-only
machines), a warning is issued with the exact environment variable to set.
"""

import os
import sys
import warnings


def _count_available_cpus() -> int:
    """Return the number of CPU cores available to this process.

    On Linux, os.sched_getaffinity(0) returns the set of logical CPUs the
    process is actually allowed to use, respecting taskset, cgroup CPU quotas,
    and SLURM --cpus-per-task.  This is the right number inside containers
    and cluster jobs.

    On macOS (and as a fallback on any platform where sched_getaffinity is
    unavailable), os.cpu_count() returns the total logical core count.  On
    Apple Silicon there is no hyperthreading, so logical == physical.
    """
    try:
        return len(os.sched_getaffinity(0))   # Linux — process CPU set
    except AttributeError:
        return os.cpu_count() or 1             # macOS / Windows fallback


def _setup_devices() -> None:
    """Configure JAX virtual CPU devices, or warn if the window has passed.

    Key insight: --xla_force_host_platform_device_count only affects the CPU
    backend.  JAX's GPU backend is a completely separate runtime and is
    unaffected by this flag, so setting it is always safe regardless of
    whether GPUs are present.

    os.environ.setdefault() is used so that any value already set by the user
    or by a cluster job scheduler is left untouched.
    """
    n_cpus = _count_available_cpus()
    flag = f"--xla_force_host_platform_device_count={n_cpus}"

    if "jax" in sys.modules:
        # JAX has already initialised its backends.  Setting XLA_FLAGS now
        # has no effect.  On GPU machines this is harmless; on CPU-only
        # machines the user will be stuck with however many virtual devices
        # JAX created at startup (usually 1).
        import jax
        try:
            if jax.devices("gpu"):
                return   # GPU backend found — nothing to do
        except RuntimeError:
            pass

        # CPU-only machine and JAX already imported.  Warn only if the user
        # did not already configure the virtual device count themselves.
        if "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
            n_current = len(jax.devices("cpu"))
            warnings.warn(
                f"mbirjax was imported after JAX on a CPU-only system. "
                f"JAX has already initialised with {n_current} CPU device(s); "
                f"{n_cpus} could be available. "
                f"For optimal multi-device performance, import mbirjax before "
                f"JAX, or set this environment variable before any Python "
                f"import:\n\n"
                f"    XLA_FLAGS='{flag}'\n",
                stacklevel=2,
            )
        return

    # JAX not yet imported — safe to set the flag now.
    os.environ.setdefault("XLA_FLAGS", flag)


_setup_devices()
