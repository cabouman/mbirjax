"""
experiments/sharding/sharding_scaling.py
─────────────────────────────────────────
Measure multi-device speedup for key mbirjax operations.

Prints a table of wall-clock time and speedup for each (operation, n_devices)
combination, along with a correctness check (max absolute diff vs the
single-device reference).

Operations tested (enabled as each step is implemented):
  ✓ fbp_filter      — Step 1 (pre-sharded input; threading in fbp_filter)
  ✓ forward_project — Step 2 (plain input; threading internal to sparse_forward_project)
  ○ back_project    — deferred (limited CPU scaling; GPU scaling good but not yet integrated)
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
import warnings


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

def bench_fbp_filter(model, inp, warmup, trials):
    """Return (times, result_jax) for fbp_filter."""
    fn = lambda: model.fbp_filter(inp)
    return _timed_run(fn, warmup, trials)


def bench_sparse_forward_project(model, inp, warmup, trials):
    """Return (times, result_jax) for sparse_forward_project.

    inp is a tuple (sharded_flat_recon, pixel_indices) pre-computed outside
    the timing loop so that the shard-scatter is not included in the timing.
    """
    sharded_flat_recon, pixel_indices = inp
    fn = lambda: model.sparse_forward_project(sharded_flat_recon, pixel_indices)
    return _timed_run(fn, warmup, trials)


def bench_sparse_back_project(model, inp, warmup, trials):
    """Return (times, result_jax) for sparse_back_project with pre-sharded sinogram.

    inp is a tuple (sharded_sino, pixel_indices) pre-computed outside
    the timing loop so that the shard-scatter is not included in the timing.
    """
    sharded_sino, pixel_indices = inp
    fn = lambda: model.sparse_back_project(sharded_sino, pixel_indices)
    return _timed_run(fn, warmup, trials)


# def bench_fbp_recon(model, inp, warmup, trials):
#     fn = lambda: model.fbp_recon(inp)
#     return _timed_run(fn, warmup, trials)

OPERATIONS = [
    # (name, bench_fn, input_key, shard_axis)
    # input_key matches a key in the inputs_np dict built in main().
    # shard_axis: axis of the primary data array to pre-shard across 'slices'.
    #   tuple entries carry a (primary_np, aux_np) pair; primary is sharded
    #   along shard_axis, aux (pixel_indices) is uploaded as a plain array.
    ('fbp_filter',             bench_fbp_filter,             'sinogram',   1),
    # ('sparse_forward_project', bench_sparse_forward_project, 'flat_recon', 1),
    # ('sparse_back_project',    bench_sparse_back_project,    'sino_bp',    1),
    # ('fbp_recon', bench_fbp_recon, 'sinogram', 1),  # Step 4
]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    out_file = 'sharding_baseline_ref.npz'
    ref_dict = None
    if os.path.exists(out_file):
        ref_dict = np.load(out_file)
        print('Using saved baseline.')
    else:
        warnings.warn('No existing baseline found. Comparing to current code.')

    # Determine device counts to test
    n_gpus = len(_gpus())
    n_cpus = len(jax.devices('cpu'))
    n_cpus = np.round(n_cpus / 2).astype(int) + 1
    max_avail = n_gpus if n_gpus > 0 else n_cpus

    if n_gpus > 0:
        n_views = 1024
        n_rows = 1024
        n_channels = 1024
    else:
        n_views = 180
        n_rows = 128
        n_channels = 256

    num_warmup = 1
    num_trials = 3

    parser = argparse.ArgumentParser(
        description='Measure multi-device speedup for mbirjax sharding.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__))
    parser.add_argument('--n-views',        type=int, default=n_views,
                        help='Number of projection views (default {})'.format(n_views))
    parser.add_argument('--n-rows',         type=int, default=n_rows,
                        help='Number of detector rows / recon slices (default {}), '
                             'must be divisible by max device count)'.format(n_rows))
    parser.add_argument('--n-channels',     type=int, default=n_channels,
                        help='Number of detector channels / recon cols (default {})'.format(n_channels))
    parser.add_argument('--device-counts',  type=int, nargs='+', default=None,
                        help='Device counts to benchmark (default: 1 plus all '
                             'available GPUs or up to 4 virtual CPUs)')
    parser.add_argument('--warmup',         type=int, default=num_warmup,
                        help='Number of warmup iterations (default {})'.format(num_warmup))
    parser.add_argument('--trials',         type=int, default=num_trials,
                        help='Number of timed iterations (default {})'.format(num_trials))
    args = parser.parse_args()

    n_views, n_rows, n_channels = args.n_views, args.n_rows, args.n_channels

    if args.device_counts is not None:
        device_counts = sorted(set(args.device_counts))
    else:
        device_counts = 1 + np.arange(max_avail)

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
    # pre-upload the relevant array before timing so that the benchmarked
    # operations never perform a GPU→CPU→GPU scatter inside the timed loop.
    rng = np.random.default_rng(42)
    sino_np = rng.standard_normal((n_views, n_rows, n_channels)).astype(np.float32)

    # Build a 1-device model just to get geometry-derived shapes.
    _ref_model = _make_model(n_views, n_rows, n_channels, n_devices=1)
    recon_shape = _ref_model.get_params('recon_shape')
    recon_np = rng.standard_normal(recon_shape).astype(np.float32)

    # Pre-compute flat recon for the sparse_forward_project benchmark.
    # pixel_indices selects all in-ROR voxels; flat_recon_np is (num_pixels, n_slices).
    _all_indices = mbirjax.gen_full_indices(recon_shape, use_ror_mask=_ref_model.use_ror_mask)
    _all_indices_np = np.asarray(_all_indices)
    _flat_recon_np = np.asarray(
        _ref_model.get_voxels_at_indices(jnp.array(recon_np), jnp.array(_all_indices_np)))

    # Map input_key → raw data for pre-sharding.
    # Tuple entries are (data_np, aux_np) pairs; data_np is sharded along
    # shard_axis, aux_np (pixel_indices) is uploaded as a plain array.
    inputs_np = {
        'sinogram':   sino_np,
        'flat_recon': (_flat_recon_np, _all_indices_np),
        'sino_bp':    (sino_np, _all_indices_np),
    }

    # ── Run each operation ────────────────────────────────────────────────────
    for op_name, bench_fn, input_key, shard_axis in OPERATIONS:
        print(f"  {'─'*58}")
        print(f"  Operation: {op_name}")
        print(f"  {'─'*58}")
        print(f"  {'n_dev':>5}  {'mean_ms':>9}  {'min_ms':>9}  {'speedup':>8}  {'max_diff':>10}")
        print(f"  {'─'*5}  {'─'*9}  {'─'*9}  {'─'*8}  {'─'*10}")

        baseline_mean = None
        ref_np = None

        for n_dev in device_counts:
            shape = (n_views, n_rows, n_channels)
            if shape[shard_axis] % n_dev != 0:
                # print(f"n_dev={n_dev} does not divide n_rows={n_rows}: skipping")
                continue
            model = _make_model(n_views, n_rows, n_channels, n_devices=n_dev)
            if model is None:
                print(f"  {n_dev:>5}  {'skip (not enough devices)':>40}")
                continue

            # Pre-upload / pre-shard input(s) once outside timing.
            #
            # Plain arrays: shard along shard_axis when mesh is configured.
            # Tuple entries (data_np, aux_np): shard data_np along shard_axis,
            #   upload aux_np (pixel_indices) as a plain array.  The bench
            #   function receives the assembled tuple as `inp`.
            raw = inputs_np[input_key]
            if isinstance(raw, tuple):
                data_np, aux_np = raw
                if model.mesh is not None:
                    spec = [None] * data_np.ndim
                    spec[shard_axis] = 'slices'
                    sharding = jax.sharding.NamedSharding(
                        model.mesh, jax.sharding.PartitionSpec(*spec))
                    data_jax = jax.device_put(data_np, sharding)
                else:
                    data_jax = jnp.array(data_np)
                jax.block_until_ready(data_jax)
                inp = (data_jax, jnp.array(aux_np))
            else:
                if model.mesh is not None and shard_axis is not None:
                    spec = [None] * raw.ndim
                    spec[shard_axis] = 'slices'
                    sharding = jax.sharding.NamedSharding(
                        model.mesh, jax.sharding.PartitionSpec(*spec))
                    inp = jax.device_put(raw, sharding)
                else:
                    inp = jnp.array(raw)
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
                if ref_dict is not None:
                    ref_saved = ref_dict[op_name]
                    if not np.allclose(ref_np, ref_saved):
                        warnings.warn('Saved reference does not match new reference.  Max diff = {}'.format(np.amax(np.abs(ref_np - ref_saved))))
                    diff_str = _correctness(ref_np, ref_saved)
                    ref_np = ref_saved
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
