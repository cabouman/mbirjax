"""
experiments/sharding/scaling_tests/fbp_filter_flatten_experiment.py
────────────────────────────────────────────────────────────────────
Throwaway investigation: does FLATTENING the FBP-filter kernel help CPU
scaling under view-axis sharding?

Background
──────────
The FBP filter is per detector row: each (channels,) row is convolved with the
filter independently of every other row and view.  Two ways to organize the
per-device work:

  per_view : lax.map over VIEWS, vmap the convolution over rows inside each view
             (the current beta/library kernel).  When sharded by view, a device
             may hold only a few views, which may not expose enough independent
             work to parallelize.
  flat     : reshape (views, rows, channels) -> (views*rows, channels) and
             lax.map the row convolution over that single flat axis, exposing
             views*rows independent rows regardless of view count.

This script times both, single-device and threaded across N devices, at a fixed
total problem size, so the only thing that changes with device count is how the
views are split.  It does NOT modify the library; it defines both kernels
locally so you can compare them directly.

Run from the beta worktree root (banner confirms which mbirjax is loaded):
    PYTHONPATH="$PWD" python experiments/sharding/scaling_tests/fbp_filter_flatten_experiment.py
    #  --size 256 128 256       (views rows channels)
    #  --device-counts 1 2 4 8
    #  --warmup 2 --trials 6
"""
import argparse
import os
import time
from functools import partial

# Import mbirjax before jax (device-setup-first ordering).
import mbirjax

import numpy as np
import jax
import jax.numpy as jnp

import mbirjax._sharding as mjs


# ── The two kernels (defined here so the library is untouched) ────────────────
@partial(jax.jit, static_argnames=("body", "vbs"))
def kernel_per_view(views, filt, body, vbs):
    """lax.map over views; vmap the row convolution within each view."""
    def conv_row(row):
        return jax.scipy.signal.fftconvolve(row, filt, mode="valid")
    def conv_view(v):
        return jax.vmap(conv_row)(v)
    n = views.shape[0]
    parts = []
    if body > 0:
        parts.append(jax.lax.map(conv_view, views[:body], batch_size=vbs))
    if n - body > 0:
        parts.append(jax.lax.map(conv_view, views[body:]))
    return jnp.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]


@partial(jax.jit, static_argnames=("body", "vbs"))
def kernel_flat(views, filt, body, vbs):
    """Flatten to (views*rows, channels); lax.map the row convolution over all rows."""
    v, r, c = views.shape
    rows = views.reshape(v * r, c)
    def conv_row(row):
        return jax.scipy.signal.fftconvolve(row, filt, mode="valid")
    nr = rows.shape[0]
    parts = []
    if body > 0:
        parts.append(jax.lax.map(conv_row, rows[:body], batch_size=vbs))
    if nr - body > 0:
        parts.append(jax.lax.map(conv_row, rows[body:]))
    out = jnp.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
    return out.reshape(v, r, c)


# ── Banner ────────────────────────────────────────────────────────────────────
def banner():
    path = os.path.dirname(mbirjax.__file__)
    is_beta = "mbirjax_sharding" in path
    print("=" * 72)
    print("  mbirjax code in use:  " +
          ("*** beta ***  (mbirjax_sharding)" if is_beta
           else "###  RESEARCH  ###   <-- NOT the beta code"))
    print(f"  path: {path}")
    print("=" * 72)


# ── Timing ────────────────────────────────────────────────────────────────────
def bench(run, warmup, trials):
    for _ in range(warmup):
        jax.block_until_ready(run())
    ts = []
    for _ in range(trials):
        t0 = time.perf_counter()
        jax.block_until_ready(run())
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


def pick_devices(n):
    try:
        g = jax.devices("gpu")
        if len(g) >= n:
            return g[:n]
    except RuntimeError:
        pass
    c = jax.devices("cpu")
    return c[:n] if len(c) >= n else None


# ── Build a view-sharded sinogram and time both kernels, threaded ─────────────
def run_threaded(model_sino, devices, kernel, vbs_in_rows):
    """Run `kernel` once per device on its view-shard, assemble, return callable."""
    sino = model_sino
    dev_to_shard = {s.device: s.data for s in sino.addressable_shards}

    def worker_process(i, device):
        shard = dev_to_shard[device]
        v, r, c = shard.shape
        if kernel is kernel_flat:
            total = v * r
            vbs = min(vbs_in_rows, total)
            body = (total // vbs) * vbs
        else:  # per_view: batch size is in views
            vbs = min(vbs_in_rows // r if vbs_in_rows >= r else 1, v)
            vbs = max(vbs, 1)
            body = (v // vbs) * vbs
        return kernel(shard, jnp.asarray(FILTER_NP), body, vbs)

    def run():
        results = mjs.run_per_device(devices, worker_process)
        return mjs.assemble_sharded(results, sino.shape, sino.sharding)
    return run


FILTER_NP = None


def main():
    global FILTER_NP
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, nargs=3, default=[256, 128, 256],
                   metavar=("VIEWS", "ROWS", "CHANNELS"))
    p.add_argument("--device-counts", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--trials", type=int, default=6)
    p.add_argument("--row-batch-size", type=int, default=128,
                   help="lax.map batch size, in ROWS (flat) / converted to views (per_view)")
    args = p.parse_args()

    banner()
    n_views, n_rows, n_ch = args.size
    print(f"  size = ({n_views}, {n_rows}, {n_ch})   warmup={args.warmup} "
          f"trials={args.trials}   row_batch_size={args.row_batch_size}")

    rng = np.random.default_rng(0)
    sino_np = rng.random((n_views, n_rows, n_ch), dtype=np.float32)
    FILTER_NP = rng.random(2 * n_ch - 1, dtype=np.float32)

    print(f"\n  {'n_dev':>5} {'per_view_ms':>12} {'flat_ms':>10} "
          f"{'pv_speedup':>11} {'fl_speedup':>11}")
    pv_base = fl_base = None
    for n in args.device_counts:
        devs = pick_devices(n)
        if devs is None or n_views % n != 0:
            print(f"  {n:>5}   (skip: need {n} devices and n_views%{n}==0)")
            continue
        model = mbirjax.ParallelBeamModel(
            (n_views, n_rows, n_ch),
            np.linspace(0, np.pi, n_views, endpoint=False))
        model.configure_sharding(devs)
        sino = model._shard_sinogram(sino_np)

        pv = bench(run_threaded(sino, devs, kernel_per_view, args.row_batch_size),
                   args.warmup, args.trials)
        fl = bench(run_threaded(sino, devs, kernel_flat, args.row_batch_size),
                   args.warmup, args.trials)
        if pv_base is None:
            pv_base, fl_base = pv, fl
        print(f"  {n:>5} {pv:>12.2f} {fl:>10.2f} "
              f"{pv_base/pv:>10.2f}x {fl_base/fl:>10.2f}x")

    # Correctness: both kernels agree (single device, full array).
    devs = pick_devices(1)
    model = mbirjax.ParallelBeamModel(
        (n_views, n_rows, n_ch), np.linspace(0, np.pi, n_views, endpoint=False))
    model.configure_sharding(devs)
    sino = model._shard_sinogram(sino_np)
    a = np.asarray(run_threaded(sino, devs, kernel_per_view, args.row_batch_size)())
    b = np.asarray(run_threaded(sino, devs, kernel_flat, args.row_batch_size)())
    print(f"\n  max|per_view - flat| = {np.max(np.abs(a - b)):.2e}  (should be ~0)")
    print("\nDone.")


if __name__ == "__main__":
    main()
