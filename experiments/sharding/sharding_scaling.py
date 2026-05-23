"""
experiments/sharding/sharding_scaling.py
─────────────────────────────────────────
Measure multi-device speedup for key mbirjax operations.

Prints a table of wall-clock time and speedup for each (operation, n_devices)
combination, along with a correctness check (max absolute diff vs the
single-device reference).

Operations tested (enabled as each step is implemented):
  ✓ fbp_filter      — Step 1
  ○ forward_project — Step 2  (uncomment when implemented)
  ○ back_project    — Step 3  (uncomment when implemented)
  ○ fbp_recon       — Step 4  (uncomment when implemented)

Usage:
  # on any machine — uses virtual CPU devices when no GPUs are available
  python experiments/sharding/sharding_scaling.py

  # force a specific set of device counts to test
  python experiments/sharding/sharding_scaling.py --device-counts 1 2 4

  # change problem size (number of detector rows — must be divisible by max n_devices)
  python experiments/sharding/sharding_scaling.py --n-rows 256

  # change number of warmup / timed trials
  python experiments/sharding/sharding_scaling.py --warmup 3 --trials 10
"""
import argparse
import os
import time
import textwrap

# Must be set before JAX initialises.  On a machine with real GPUs this is a
# no-op (setdefault won't override an env var that is already set or that JAX
# has already processed).  On a CPU-only machine it creates 4 virtual devices
# so the script can demonstrate 1-vs-2-vs-4 scaling without real hardware.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import numpy as np
import jax
import jax.numpy as jnp
import mbirjax

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _gpus():
    """Return list of GPU devices, or [] if no GPU backend is available."""
    try:
        return jax.devices('gpu')
    except RuntimeError:
        return []


def _pick_devices(n_requested: int):
    """Return a list of n_requested devices, preferring GPUs over virtual CPUs."""
    gpus = _gpus()
    if len(gpus) >= n_requested:
        return gpus[:n_requested]
    cpus = jax.devices('cpu')
    if len(cpus) >= n_requested:
        return cpus[:n_requested]
    return None   # not enough devices


def _make_model(n_views, n_rows, n_channels, n_devices=1):
    """Build a ParallelBeamModel configured for n_devices (GPU preferred)."""
    angles = np.linspace(0, np.pi, n_views, endpoint=False)
    model = mbirjax.ParallelBeamModel((n_views, n_rows, n_channels), angles)
    devs = _pick_devices(n_devices)
    if devs is None:
        return None
    model.configure_sharding(devs)
    return model


def _timed_run(fn, warmup: int, trials: int):
    """Run fn() for warmup+trials iterations; return (times_list, last_result)."""
    result = None
    times = []
    for i in range(warmup + trials):
        t0 = time.perf_counter()
        result = fn()
        jax.block_until_ready(result)
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
    return times, result


def _summarise(times):
    """Return (mean_ms, min_ms, max_ms)."""
    arr = np.array(times) * 1e3
    return arr.mean(), arr.min(), arr.max()


def _correctness(ref_np: np.ndarray, out) -> float:
    """Max absolute difference between reference and output (both as numpy)."""
    return float(np.max(np.abs(np.asarray(out) - ref_np)))


# ──────────────────────────────────────────────────────────────────────────────
# Per-operation benchmark definitions
# ──────────────────────────────────────────────────────────────────────────────

def bench_fbp_filter(model, sino_jax, warmup, trials):
    """Return (times, result_jax) for fbp_filter."""
    fn = lambda: model.fbp_filter(sino_jax)
    return _timed_run(fn, warmup, trials)


# ── Add new operations here as steps are completed ────────────────────────────
# def bench_forward_project(model, recon_jax, warmup, trials):
#     fn = lambda: model.forward_project(recon_jax)
#     return _timed_run(fn, warmup, trials)
#
# def bench_back_project(model, sino_jax, warmup, trials):
#     fn = lambda: model.back_project(sino_jax)
#     return _timed_run(fn, warmup, trials)
#
# def bench_fbp_recon(model, sino_jax, warmup, trials):
#     fn = lambda: model.fbp_recon(sino_jax)
#     return _timed_run(fn, warmup, trials)
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = [
    # (name, bench_fn, input_key)
    # input_key matches a key in the `inputs` dict built in main()
    ('fbp_filter', bench_fbp_filter, 'sinogram'),
    # ('forward_project', bench_forward_project, 'recon'),   # Step 2
    # ('back_project',   bench_back_project,   'sinogram'),  # Step 3
    # ('fbp_recon',      bench_fbp_recon,      'sinogram'),  # Step 4
]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Measure multi-device speedup for mbirjax sharding.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__))
    parser.add_argument('--n-views',        type=int, default=180,
                        help='Number of projection views (default 180)')
    parser.add_argument('--n-rows',         type=int, default=128,
                        help='Number of detector rows / recon slices (default 128, '
                             'must be divisible by max device count)')
    parser.add_argument('--n-channels',     type=int, default=256,
                        help='Number of detector channels / recon cols (default 256)')
    parser.add_argument('--device-counts',  type=int, nargs='+', default=None,
                        help='Device counts to benchmark (default: 1 plus all '
                             'available GPUs or up to 4 virtual CPUs)')
    parser.add_argument('--warmup',         type=int, default=2,
                        help='Number of warmup iterations (default 2)')
    parser.add_argument('--trials',         type=int, default=5,
                        help='Number of timed iterations (default 5)')
    args = parser.parse_args()

    n_views, n_rows, n_channels = args.n_views, args.n_rows, args.n_channels

    # Determine device counts to test
    n_gpus = len(_gpus())
    n_cpus = len(jax.devices('cpu'))
    max_avail = n_gpus if n_gpus > 0 else n_cpus

    if args.device_counts is not None:
        device_counts = sorted(set(args.device_counts))
    else:
        device_counts = sorted({1, 2, max_avail}) if max_avail > 1 else [1]

    print(f"\n{'='*62}")
    print(f"  mbirjax sharding scaling benchmark")
    print(f"{'='*62}")
    print(f"  Sinogram:  ({n_views}, {n_rows}, {n_channels})")
    device_type = 'GPU' if n_gpus > 0 else 'CPU (virtual)'
    single_dev  = _pick_devices(1)
    print(f"  Devices:   {device_counts}  ({device_type})")
    print(f"  Device[0]: {single_dev[0] if single_dev else 'none'}")
    print(f"  Warmup / trials: {args.warmup} / {args.trials}")
    print(f"{'='*62}\n")

    # Build raw numpy inputs once.  Each n_dev iteration will pre-shard or
    # pre-upload the relevant array before timing so that fbp_filter (and
    # future operations) never perform a GPU→CPU→GPU scatter inside the
    # timed loop.  Passing a plain JAX array to fbp_filter triggers
    # _shard_sinogram → np.asarray (GPU→CPU) + device_put (CPU→GPU scatter)
    # on every call — O(sinogram bytes) of PCIe traffic that swamps the
    # compute speedup on GPU.
    rng = np.random.default_rng(42)
    sino_np = rng.standard_normal((n_views, n_rows, n_channels)).astype(np.float32)

    # recon_np = rng.standard_normal((n_rows, n_channels, n_rows)).astype(np.float32)

    # Map input_key → raw numpy array (sharded per n_dev inside the loop).
    inputs_np = {
        'sinogram': sino_np,
        # 'recon': recon_np,   # uncomment for Steps 2/3
    }

    # ── Run each operation ────────────────────────────────────────────────────
    for op_name, bench_fn, input_key in OPERATIONS:
        print(f"  {'─'*58}")
        print(f"  Operation: {op_name}")
        print(f"  {'─'*58}")
        print(f"  {'n_dev':>5}  {'mean_ms':>9}  {'min_ms':>9}  {'speedup':>8}  {'max_diff':>10}")
        print(f"  {'─'*5}  {'─'*9}  {'─'*9}  {'─'*8}  {'─'*10}")

        baseline_mean = None
        ref_np = None

        for n_dev in device_counts:
            model = _make_model(n_views, n_rows, n_channels, n_devices=n_dev)
            if model is None:
                print(f"  {n_dev:>5}  {'skip (not enough devices)':>40}")
                continue

            # Pre-shard / pre-upload the input once outside timing.
            # model.mesh is always set (configure_sharding is called with ≥1
            # real device), so we always take the sharded path here.
            raw_np = inputs_np[input_key]
            if model.mesh is not None:
                # Sinogram: shard along det_rows (axis 1 = 'slices' mesh axis).
                # Update the PartitionSpec when adding recon operations.
                sharding = jax.sharding.NamedSharding(
                    model.mesh,
                    jax.sharding.PartitionSpec(None, 'slices', None))
                inp = jax.device_put(raw_np, sharding)
            else:
                inp = jnp.array(raw_np)
            jax.block_until_ready(inp)

            try:
                times, result = bench_fn(model, inp, args.warmup, args.trials)
            except Exception as e:
                print(f"  {n_dev:>5}  ERROR: {e}")
                continue

            mean_ms, min_ms, _ = _summarise(times)

            # First run = baseline; subsequent = compute speedup
            if baseline_mean is None:
                baseline_mean = mean_ms
                ref_np = np.asarray(result)
                speedup_str = '  1.00×'
                diff_str    = '      —'
            else:
                speedup = baseline_mean / mean_ms
                diff    = _correctness(ref_np, result)
                speedup_str = f'{speedup:>7.2f}×'
                diff_str    = f'{diff:>10.2e}'

            print(f"  {n_dev:>5}  {mean_ms:>9.1f}  {min_ms:>9.1f}  {speedup_str}  {diff_str}")

        print()

    print(f"{'='*62}\n")


if __name__ == '__main__':
    main()
