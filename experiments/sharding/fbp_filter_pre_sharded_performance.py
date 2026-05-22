"""
fbp_filter_pre_sharded_performance.py
──────────────────────────────────────
Measure fbp_filter timing when the sinogram is *already* distributed across
GPUs before the call — the situation that will arise naturally in the VCD loop.

Four paths:
  A  Unsharded baseline   — single-device production fbp_filter (reference)
  B  Current threading    — pre-sharded sinogram; fbp_filter still pulls the
                            full sinogram to CPU and re-uploads it (current code)
  C  Fast threading       — pre-sharded sinogram; addressable_shards used so
                            each shard is read directly on its device (zero input PCIe)
  D  Fully sharded        — no host memory at all; each thread keeps its result
                            on-device; output is a NamedSharding JAX array

Expected PCIe traffic (2-GPU, size=1024, float32 → 4 GB total):
  A  0 GB  (already on GPU 0)
  B  16 GB  (pull 4 GB in, push 2+2 GB shards, pull 2+2 GB results, push 4 GB out)
  C  8 GB   (pull 2+2 GB results, push 4 GB out — input already on correct devices)
  D  0 GB   (no host involvement at all)

Usage
-----
    conda activate mbirjax
    python experiments/sharding/fbp_filter_pre_sharded_performance.py
"""

import concurrent.futures
import time
import numpy as np
import jax
import jax.numpy as jnp

import mbirjax
from mbirjax import tomography_utils
from mbirjax.parallel_beam import _apply_fbp_filter_to_shard
from mbirjax.memory_stats import get_memory_stats

# ── Configuration ─────────────────────────────────────────────────────────────

SIZES           = [512, 1024]
VIEW_BATCH_SIZE = 100
N_RUNS          = 3       # min over this many timed calls


# ── Path C: fast threading (skip input GPU→CPU) ───────────────────────────────

def _fbp_filter_fast(model, sinogram, view_batch_size=VIEW_BATCH_SIZE):
    """fbp_filter using addressable_shards — input never leaves GPU.

    The sinogram must already be sharded along axis 1 (rows) with the same
    NamedSharding that model's mesh produces.  Each shard.data is a
    single-device array that lives on its assigned GPU; we read it directly
    without any host copy.  The output still comes back to the host for
    concatenation before a final upload.
    """
    num_views, num_rows, num_channels = sinogram.shape
    delta_voxel    = model.get_params('delta_voxel')
    scaling_factor = 1.0 / (delta_voxel ** 2)
    recon_filter   = tomography_utils.generate_direct_recon_filter(
        num_channels, filter_name='ramp')
    recon_filter  *= scaling_factor          # in-place → stays float32
    filter_np      = np.asarray(recon_filter)

    n_devices  = model.mesh.devices.size
    devices    = list(model.mesh.devices.flat)
    body_views = (num_views // view_batch_size) * view_batch_size
    tail_views = num_views - body_views

    # Build device→shard map without touching host memory.
    dev_to_shard = {s.device: s.data for s in sinogram.addressable_shards}
    results      = [None] * n_devices

    def _worker(i):
        dev = devices[i]
        with jax.default_device(dev):
            shard_jax  = dev_to_shard[dev]       # already on dev — zero PCIe
            filter_jax = jnp.array(filter_np)
            out = _apply_fbp_filter_to_shard(
                shard_jax, filter_jax,
                body_views=body_views, tail_views=tail_views,
                view_batch_size=view_batch_size)
            jax.block_until_ready(out)
        results[i] = np.asarray(out)             # D→H for gather step

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_devices) as ex:
        list(ex.map(_worker, range(n_devices)))

    filtered = jnp.array(np.concatenate(results, axis=1))
    filtered = filtered * (np.pi / num_views)
    return filtered


# ── Path D: fully sharded (no host memory at all) ─────────────────────────────

def _fbp_filter_fully_sharded(model, sinogram, view_batch_size=VIEW_BATCH_SIZE):
    """fbp_filter with zero PCIe traffic — output is a NamedSharding JAX array.

    Both input and output stay on their respective GPUs.  The returned array
    has the same sharding as the input sinogram, so it can be passed directly
    to any downstream operation that accepts a sharded array.

    Note: fftconvolve('valid') with a (2*channels-1,) filter on a (channels,)
    row produces a (channels,) output, so the output sharding is identical to
    the input sharding.
    """
    num_views, num_rows, num_channels = sinogram.shape
    delta_voxel    = model.get_params('delta_voxel')
    scaling_factor = 1.0 / (delta_voxel ** 2)
    recon_filter   = tomography_utils.generate_direct_recon_filter(
        num_channels, filter_name='ramp')
    recon_filter  *= scaling_factor
    filter_np      = np.asarray(recon_filter)
    pi_scale       = float(np.pi / num_views)   # plain float — no device placement

    n_devices  = model.mesh.devices.size
    devices    = list(model.mesh.devices.flat)
    body_views = (num_views // view_batch_size) * view_batch_size
    tail_views = num_views - body_views

    dev_to_shard = {s.device: s.data for s in sinogram.addressable_shards}
    results      = [None] * n_devices

    def _worker(i):
        dev = devices[i]
        with jax.default_device(dev):
            shard_jax  = dev_to_shard[dev]
            filter_jax = jnp.array(filter_np)
            out = _apply_fbp_filter_to_shard(
                shard_jax, filter_jax,
                body_views=body_views, tail_views=tail_views,
                view_batch_size=view_batch_size)
            out = out * pi_scale             # scale on-device; plain float avoids
                                             # device-placement mismatch
            jax.block_until_ready(out)
        results[i] = out                     # keep on device — no D→H

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_devices) as ex:
        list(ex.map(_worker, range(n_devices)))

    # Assemble a global sharded array from per-device single-device arrays.
    # jax.make_array_from_single_device_arrays does not touch host memory.
    filtered = jax.make_array_from_single_device_arrays(
        shape=(num_views, num_rows, num_channels),
        sharding=sinogram.sharding,
        arrays=results)
    return filtered


# ── Utilities ─────────────────────────────────────────────────────────────────

def make_model(size, gpus=None):
    angles = jnp.linspace(0, np.pi, size, endpoint=False)
    model  = mbirjax.ParallelBeamModel((size, size, size), angles)
    model.set_params(delta_voxel=1.0, delta_det_channel=1.0)
    model.auto_set_recon_geometry()
    if gpus:
        model.configure_sharding(gpus)
    return model


def time_fn(fn, n_runs=N_RUNS):
    """Return (min_time, last_result).  Calls fn() n_runs times."""
    times, result = [], None
    for _ in range(n_runs):
        t0     = time.perf_counter()
        result = fn()
        jax.block_until_ready(result)
        times.append(time.perf_counter() - t0)
    return min(times), result


def gpu_peak_str():
    stats = get_memory_stats(print_results=False)
    parts = [f"{s['id']}: {s['peak_bytes_in_use']/1024**3:.4f}GB"
             for s in stats if 'cpu' not in s['id'].lower()]
    return '  '.join(parts)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_gpus = [d for d in jax.devices() if 'cpu' not in d.device_kind.lower()]
    n_gpu    = len(all_gpus)
    print(f"GPUs available: {n_gpu}")
    if n_gpu < 2:
        print("WARNING: fewer than 2 GPUs — multi-device paths will be skipped.")
    print(f"Sizes: {SIZES}  view_batch_size={VIEW_BATCH_SIZE}\n")

    summary = []

    for size in SIZES:
        print(f"{'='*64}")
        print(f"Size: {size}  (sinogram shape: {size} × {size} × {size})")

        rng     = np.random.default_rng(42)
        sino_np = rng.standard_normal((size, size, size)).astype(np.float32)

        # ── Path A: unsharded single-device baseline ──────────────────
        model_1 = make_model(size)
        sino_1  = jnp.array(sino_np)

        # warmup
        jax.block_until_ready(model_1.fbp_filter(sino_1, view_batch_size=VIEW_BATCH_SIZE))

        t_a, out_a = time_fn(
            lambda: model_1.fbp_filter(sino_1, view_batch_size=VIEW_BATCH_SIZE))
        ref = np.asarray(out_a)
        print(f"  A. unsharded baseline:                {t_a:.3f} s"
              f"  [{gpu_peak_str()}]")

        if n_gpu < 2:
            summary.append((size, t_a, None, None, None))
            continue

        # ── Build 2-GPU model and pre-shard sinogram ──────────────────
        model_n     = make_model(size, gpus=all_gpus[:2])
        sino_sharded = model_n._maybe_shard(sino_1, axis=1)
        jax.block_until_ready(sino_sharded)
        print(f"  (sinogram pre-sharded: {sino_sharded.sharding})")

        # ── Path B: current fbp_filter (always pulls to CPU even if pre-sharded)
        jax.block_until_ready(                        # warmup
            model_n.fbp_filter(sino_sharded, view_batch_size=VIEW_BATCH_SIZE))

        t_b, out_b = time_fn(
            lambda: model_n.fbp_filter(sino_sharded, view_batch_size=VIEW_BATCH_SIZE))
        diff_b = float(np.max(np.abs(np.asarray(out_b) - ref)))
        print(f"  B. current  (CPU roundtrip):          {t_b:.3f} s"
              f"  ({t_a/t_b:.2f}x vs A)  diff={diff_b:.1e}"
              f"  [{gpu_peak_str()}]")

        # ── Path C: fast threading (addressable_shards, zero input PCIe) ─
        jax.block_until_ready(_fbp_filter_fast(model_n, sino_sharded))  # warmup

        t_c, out_c = time_fn(lambda: _fbp_filter_fast(model_n, sino_sharded))
        diff_c = float(np.max(np.abs(np.asarray(out_c) - ref)))
        print(f"  C. fast     (skip input pull):        {t_c:.3f} s"
              f"  ({t_a/t_c:.2f}x vs A)  diff={diff_c:.1e}"
              f"  [{gpu_peak_str()}]")

        # ── Path D: fully sharded (zero PCIe, output stays distributed) ──
        jax.block_until_ready(                        # warmup
            _fbp_filter_fully_sharded(model_n, sino_sharded))

        t_d, out_d = time_fn(
            lambda: _fbp_filter_fully_sharded(model_n, sino_sharded))
        diff_d = float(np.max(np.abs(np.asarray(out_d) - ref)))
        print(f"  D. sharded  (no host memory at all):  {t_d:.3f} s"
              f"  ({t_a/t_d:.2f}x vs A)  diff={diff_d:.1e}"
              f"  [{gpu_peak_str()}]")

        summary.append((size, t_a, t_b, t_c, t_d))

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("Summary  (min over 3 runs, speedup vs A unsharded baseline)\n")
    hdr = (f"  {'size':>6}  {'A baseline':>13}  {'B current':>13}"
           f"  {'C fast':>13}  {'D fully sharded':>16}")
    print(hdr)
    print("-" * len(hdr))

    for size, ta, tb, tc, td in summary:
        def fmt(t):
            return f"{t:.3f}s {ta/t:.2f}x" if t is not None else "     skipped"
        print(f"  {size:>6}  {ta:.3f}s base  {fmt(tb):>13}"
              f"  {fmt(tc):>13}  {fmt(td):>16}")

    print()


if __name__ == '__main__':
    main()
