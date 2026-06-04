"""
mbirjax._device_setup
─────────────────────
Runs automatically when mbirjax is imported.  Must be imported as the very
first line of mbirjax/__init__.py, before any JAX import reaches this process.

What this module does
─────────────────────
On CPU-only machines it sets XLA_FLAGS so JAX sees a sensible number of virtual
CPU devices, enabling multi-device sharding (and sharding tests) without real
GPUs.  On GPU machines the flag is set but ignored by the GPU backend, so it is
always safe to apply.

If JAX was already imported before mbirjax (only a problem on CPU-only
machines), a warning is issued with the exact environment variable to set.

Choosing the virtual device count
─────────────────────────────────
Empirically (Apple M3 Max, 10 P + 4 E cores) throughput on CPU plateaus and
then regresses past ~6-8 devices: work is split evenly, so the slow efficiency
cores and memory-bandwidth saturation make extra devices a net loss.  So the
default is capped (DEFAULT_MAX_CPU_DEVICES) rather than "one per core".

Resolution order (first that applies wins):
  1. Env var MBIRJAX_NUM_CPU_DEVICES - explicit user/cluster override.
  2. macOS: min(performance_cores, cap)   [P-cores via sysctl]
  3. Linux: min(len(sched_getaffinity(0)), cap)   [respects job allocation]
  4. Fallback: min(os.cpu_count(), cap)
"""

import os
import sys
import warnings
from typing import Optional

# Default upper bound on virtual CPU devices.  See module docstring: throughput
# plateaus/regresses past this on tested hardware.  Override with the env var
# MBIRJAX_NUM_CPU_DEVICES for deliberate performance tuning.
DEFAULT_MAX_CPU_DEVICES = 8


def _performance_core_count() -> Optional[int]:
    """Return the number of macOS performance (P) cores, or None if unknown.

    On Apple Silicon, sysctl hw.perflevel0.physicalcpu reports the P-core count
    (perflevel1 is the slower E-cores).  Splitting work across the E-cores hurts
    because every device gets an equal share and the E-cores straggle, so we
    prefer to size by P-cores on macOS.  Returns None on non-macOS or if the
    sysctl is unavailable, so the caller can fall back.
    """
    if sys.platform != "darwin":
        return None
    try:
        import subprocess
        out = subprocess.run(
            ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
            capture_output=True, text=True, timeout=2.0,
        )
        n = int(out.stdout.strip())
        return n if n > 0 else None
    except Exception:
        return None


def _count_available_cpus() -> int:
    """Return the number of CPU cores available to this process.

    On Linux, os.sched_getaffinity(0) returns the set of logical CPUs the
    process is actually allowed to use, respecting taskset, cgroup CPU quotas,
    and SLURM --cpus-per-task.  This is the right number inside containers and
    cluster jobs.

    On macOS (and as a fallback), os.cpu_count() returns the total logical core
    count.  On Apple Silicon there is no hyperthreading, so logical == physical.
    """
    try:
        return len(os.sched_getaffinity(0))   # Linux - process CPU set
    except AttributeError:
        return os.cpu_count() or 1             # macOS / Windows fallback


def _resolve_num_cpu_devices() -> int:
    """Decide how many virtual CPU devices to request (see module docstring)."""
    # 1. Explicit override.
    override = os.environ.get("MBIRJAX_NUM_CPU_DEVICES")
    if override:
        try:
            n = int(override)
            if n >= 1:
                return n
        except ValueError:
            warnings.warn(
                f"MBIRJAX_NUM_CPU_DEVICES={override!r} is not a positive "
                f"integer; ignoring.", stacklevel=2,
            )

    cap = DEFAULT_MAX_CPU_DEVICES

    # 2. macOS: size by performance cores.
    p_cores = _performance_core_count()
    if p_cores is not None:
        return min(p_cores, cap)

    # 3/4. Linux affinity set, or generic fallback.
    return min(_count_available_cpus(), cap)


def _setup_devices() -> None:
    """Configure JAX virtual CPU devices, or warn if the window has passed.

    Key insight: --xla_force_host_platform_device_count only affects the CPU
    backend.  JAX's GPU backend is a completely separate runtime and is
    unaffected by this flag, so setting it is always safe regardless of whether
    GPUs are present.

    os.environ.setdefault() is used so that any value already set by the user or
    by a cluster job scheduler is left untouched.
    """
    n_devices = _resolve_num_cpu_devices()
    flag = f"--xla_force_host_platform_device_count={n_devices}"

    if "jax" in sys.modules:
        # JAX has already initialised its backends.  Setting XLA_FLAGS now has
        # no effect.  On GPU machines this is harmless; on CPU-only machines the
        # user is stuck with however many virtual devices JAX created (usually 1).
        import jax
        try:
            if jax.devices("gpu"):
                return   # GPU backend found - nothing to do
        except RuntimeError:
            pass

        # CPU-only machine and JAX already imported.  Warn only if the user did
        # not already configure the virtual device count themselves.
        if "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
            n_current = len(jax.devices("cpu"))
            warnings.warn(
                f"mbirjax was imported after JAX on a CPU-only system. "
                f"JAX has already initialised with {n_current} CPU device(s); "
                f"{n_devices} could be available. "
                f"For optimal multi-device performance, import mbirjax before "
                f"JAX, or set this environment variable before any Python "
                f"import:\n\n"
                f"    XLA_FLAGS='{flag}'\n",
                stacklevel=2,
            )
        return

    # JAX not yet imported - safe to set the flag now.
    os.environ.setdefault("XLA_FLAGS", flag)


_setup_devices()
