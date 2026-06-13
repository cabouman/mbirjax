"""
experiments/sharding/scaling_tests/sparse_back_project_phase_ablation.py
────────────────────────────────────────────────────────────────────────
Phase-split ablation for sharded back projection scaling.

The full-sweep driver (sparse_back_project_scaling.py) shows the sharded back
projection caps at ~1.4x on CPU and regresses at 8 devices.  Two hypotheses
explain the cap, and they imply different fixes:

  (A) reduce-scatter overhead — Phase 2 (cross-device sum/scatter of the
      per-device partials) is the drag; Phase 1 (per-device compute) scales fine.
      => pipelining / a smarter reduction could help.
  (B) memory-bandwidth saturation — Phase 1 itself doesn't scale, because the
      virtual CPU devices all share one memory bus and back projection is
      bandwidth-bound.  => no reduce-scatter cleverness helps on CPU.

This script DISCRIMINATES between them by timing the two phases separately,
across device counts, for one size.  It does NOT modify production code: it
reconstructs the exact Phase-1 and Phase-2 closures from
TomographyModel._sparse_back_project_sharded, calling the same primitives
(the jitted projector, run_per_device, move_shard) on the same data layout, so
the measurement is faithful by construction.

  Phase 1 = each device back-projects its view-shard into a full-cylinder
            partial (num_pixels, num_slices); the per-device compute.
  Phase 2 = for each slice-owner, sum the n_dev partials over its band
            (the all-to-all reduce-scatter), assembled to a slice-sharded array.

Timing note: separating the phases inserts a block between them that the
production path does not (there Phase 1 dispatches unblocked so its compute can
overlap Phase 2's transfers).  So phase1 + phase2 may exceed the fused total;
that is expected.  The decomposition is for reading how each phase SCALES with
device count, not for reproducing the fused wall-clock.

Single process (timing, not memory, so no subprocess isolation needed).  Run
from the beta worktree root:

    python experiments/sharding/scaling_tests/sparse_back_project_phase_ablation.py
"""
import os
import sys

# Resolve `mbirjax` to THIS beta worktree, not the editable install (which points
# at the research worktree and has no _sharding).  Prepend the beta root (three
# dirs up from this file) before importing mbirjax.  This must precede the import.
_BETA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          os.pardir, os.pardir, os.pardir))
sys.path.insert(0, _BETA_ROOT)

# Import mbirjax before jax (device-setup-first: establishes the virtual CPU
# device count before any JAX backend init).
import mbirjax
import mbirjax._sharding as mjs

import time
import numpy as np
import jax
import jax.numpy as jnp


# ── Run configuration (edit here) ─────────────────────────────────────────────
SIZES = [(256, 256, 256)]      # (views, rows, channels); add more to taste
DEVICE_COUNTS = [1, 2, 4, 8]
WARMUP = 1
TRIALS = 3
SEED = 0


def pick_devices(n):
    """First n CPU (or GPU) devices, or None if not enough."""
    try:
        g = jax.devices("gpu")
    except RuntimeError:
        g = []
    devs = g if len(g) >= n else jax.devices("cpu")
    return devs[:n] if len(devs) >= n else None


def timeit(fn, warmup=WARMUP, trials=TRIALS):
    """Min wall-clock (ms) of fn() over trials, blocking each result."""
    for _ in range(warmup):
        jax.block_until_ready(fn())
    ts = []
    for _ in range(trials):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


def build(size, devices):
    """Sharded ParallelBeamModel + pre-sharded sinogram + full-FOV indices."""
    n_views, n_rows, n_channels = size
    angles = np.linspace(0, np.pi, n_views, endpoint=False)
    model = mbirjax.ParallelBeamModel((n_views, n_rows, n_channels), angles)
    model.configure_sharding(devices)
    sino_np = np.random.default_rng(SEED).random(size, dtype=np.float32)
    sino_sharded = model._shard_sinogram(sino_np)
    idx = mbirjax.gen_full_indices(model.get_params('recon_shape'),
                                   use_ror_mask=model.get_params('use_ror_mask'))
    return model, sino_sharded, idx


def make_phase_fns(model, sino_sharded, idx):
    """Reconstruct the Phase-1 and Phase-2 closures verbatim from
    _sparse_back_project_sharded, plus the fused total, for one configuration.
    Returns (phase1, phase2, total, n_dev)."""
    devices = model.shard_devices
    n_dev = len(devices)
    num_views = sino_sharded.shape[0]
    num_slices = model.get_params('recon_shape')[2]
    slices_per_dev = num_slices // n_dev
    bands = [slice(t * slices_per_dev, (t + 1) * slices_per_dev) for t in range(n_dev)]
    shard_info = {s.device: (s.data, s.index[0])
                  for s in sino_sharded.addressable_shards}

    # Phase 1 — per-device partial back projection (same as production).
    def _partial(i, device):
        local_views, vslice = shard_info[device]
        start, stop, step = vslice.indices(num_views)
        global_view_idx = jnp.arange(start, stop, step)
        local_pixels = jax.device_put(idx, device)
        return model.projector_functions.sparse_back_project(
            local_views, local_pixels, view_indices=global_view_idx, coeff_power=1)

    def phase1():
        return mjs.run_per_device(devices, _partial)

    # Materialize partials once so Phase 2 is timed on ready inputs (the cross-
    # device reduce-scatter alone, not the compute that feeds it).
    partials = jax.block_until_ready(mjs.run_per_device(devices, _partial))

    def _reduce_band(t, owner):
        band = bands[t]
        contribs = [mjs.move_shard(partials[d][:, band], owner,
                                   dev2dev_safe=model.dev2dev_safe)
                    for d in range(n_dev)]
        owned = contribs[0]
        for c in contribs[1:]:
            owned = owned + c
        return owned

    def phase2():
        return mjs.run_per_device(devices, _reduce_band)

    def total():
        return model.sparse_back_project(sino_sharded, idx)

    return phase1, phase2, total, n_dev


def main():
    print("=" * 78)
    print(f"  back-projection phase ablation — mbirjax at {mbirjax.__file__}")
    print(f"  available CPU devices: {len(jax.devices('cpu'))}")
    print("=" * 78)

    for size in SIZES:
        label = "x".join(str(s) for s in size)
        print(f"\n### size {label}  (views x rows x channels) ###")
        print(f"{'n_dev':>5} | {'phase1_ms':>10} {'p1_speedup':>10} | "
              f"{'phase2_ms':>10} | {'total_ms':>9} {'p1+p2_ms':>9}")
        print("-" * 66)
        p1_base = None
        for n in DEVICE_COUNTS:
            devs = pick_devices(n)
            if devs is None:
                print(f"{n:>5} | (not enough devices)")
                continue
            num_views, num_rows, _ = size
            # Skip if the sharded axes don't divide n (configure_sharding would raise).
            model_probe = mbirjax.ParallelBeamModel(
                size, np.linspace(0, np.pi, num_views, endpoint=False))
            num_slices = model_probe.get_params('recon_shape')[2]
            if num_views % n != 0 or num_slices % n != 0:
                print(f"{n:>5} | (views/slices not divisible by {n}; skipped)")
                continue

            model, sino_sharded, idx = build(size, devs)
            phase1, phase2, total, n_dev = make_phase_fns(model, sino_sharded, idx)
            p1 = timeit(phase1)
            p2 = timeit(phase2)
            tot = timeit(total)
            if p1_base is None:
                p1_base = p1
            print(f"{n:>5} | {p1:>10.2f} {p1_base / p1:>9.2f}x | "
                  f"{p2:>10.2f} | {tot:>9.2f} {p1 + p2:>9.2f}")

    print("\nReading: if phase1 keeps speeding up while total flattens, the cap is "
          "the\nreduce-scatter (phase2 — comms-bound).  If phase1 itself stops "
          "scaling, the\ncap is the shared CPU memory bus (bandwidth-bound).")


if __name__ == "__main__":
    main()
