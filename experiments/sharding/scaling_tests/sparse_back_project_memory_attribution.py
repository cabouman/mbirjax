"""
experiments/sharding/scaling_tests/sparse_back_project_memory_attribution.py
─────────────────────────────────────────────────────────────────────────────
Attribute the per-device peak memory of sharded back projection, and sweep the
slice-band length to find the memory/throughput knee.

Context: the GPU scaling run showed peak-per-device ÷ sinogram-shard ≈ 11× at
1024³/4-device.  The necessary floor is only ~1.8× (sino view-shard + recon
output shard), so ~9× is working buffers.  Two of those are full-slice-span and
constant in device count (which is why GPU memory scaled sub-1/n):
  - the full-cylinder partial   (num_pixels × num_slices)
  - the vmap-over-views buffer   (view_batch × pixel_batch × num_slices)

This script does two things at a fixed (size, device count):

  PART A — attribution.  Build the sharded model exactly as the scaling harness
    does, report the batch sizes it actually ends up using (configure_sharding
    does NOT run set_devices_and_batch_sizes, so they are whatever the full-
    problem-on-one-GPU path left), and compute the exact byte size of each
    buffer.  This decomposes the peak.

  (A view/pixel batch-size sweep ("Part B") was here originally; it showed batch
  tuning leaves the peak essentially flat, so it was removed.  See
  sharding_status.md.  The vmap batch knobs are still measured/reported by Part A.)

  PART C — slice-band sweep (the structural lever).  Vary
    back_project_slice_band (how the slice axis is streamed in sharded back
    projection) with default batch sizes, measuring peak and time per band.  The
    largest band (= slices_per_dev) approximates no sub-tiling; smaller bands make
    each owner's reduce gather fewer slices -> lower peak, at the cost of more
    (smaller) projector calls.  Pick the band at the memory/throughput knee (the
    fbp ROW_FILTER_BATCH analogue).

Isolated-subprocess harness like the other scaling scripts (orchestrator touches
no JAX; one worker subprocess per measurement).  Run from the beta worktree root:

    python experiments/sharding/scaling_tests/sparse_back_project_memory_attribution.py
"""
import os
import sys
import argparse

import scaling_common as sc

import numpy as np

# NOTE: mbirjax (and therefore jax) is imported INSIDE the worker functions, not
# at the top level — same reason as the other scaling drivers: the default
# (no --worker) role is a JAX-free orchestrator that spawns worker subprocesses,
# so a top-level import would pull JAX into the orchestrator and pollute the
# per-device peak_bytes_in_use measurement the workers take.


OP_NAME = "sparse_back_project_mem_attr"

# ── Run configuration (edit here) ─────────────────────────────────────────────
SIZE = (1024, 1024, 1024)        # (views, rows, channels); the regime where 11× was seen
N_DEVICES = 4
WARMUP = 1
TRIALS = 3
SEED = 0

_OOM_MARKERS = ("RESOURCE_EXHAUSTED", "OUT OF MEMORY", "OOM", "BAD_ALLOC",
                "FAILED TO ALLOCATE", "WORK AREA", "SCRATCH ALLOCATOR")

_F4 = 4   # bytes per float32


# ── Shared builder ────────────────────────────────────────────────────────────
def _build(devs, view_batch=-1, pixel_batch=-1):
    """Sharded ParallelBeamModel at SIZE, optionally overriding the vmap batch
    sizes and recompiling the projector so the override actually takes effect.

    Order: configure_sharding first (sets the mesh; it does not touch batch sizes
    or the projector), then override + create_projectors() last, so the final
    compiled projector carries the requested batch sizes.
    """
    import mbirjax
    n_views, _, _ = SIZE
    angles = np.linspace(0, np.pi, n_views, endpoint=False)
    model = mbirjax.ParallelBeamModel(SIZE, angles)
    model.configure_sharding(devs)
    if view_batch and view_batch > 0:
        model.view_batch_size_for_vmap = int(view_batch)
    if pixel_batch and pixel_batch > 0:
        model.pixel_batch_size_for_vmap = int(pixel_batch)
    if (view_batch and view_batch > 0) or (pixel_batch and pixel_batch > 0):
        model.create_projectors()   # recompile with the overridden batch sizes
    return model


def _buffer_sizes(num_pixels, num_slices, view_batch_eff, pixel_batch):
    """Exact byte sizes of the per-device buffers (float32)."""
    n_views, n_rows, n_chan = SIZE
    n_dev = N_DEVICES
    return {
        "sino_view_shard":       (n_views // n_dev) * n_rows * n_chan * _F4,
        "recon_output_shard":    num_pixels * (num_slices // n_dev) * _F4,
        "full_cylinder_partial": num_pixels * num_slices * _F4,
        "vmap_over_views":       view_batch_eff * pixel_batch * num_slices * _F4,
    }


# ── Worker: PART A (attribution) ──────────────────────────────────────────────
def worker_attribute(out_file):
    import mbirjax
    devs = sc.pick_devices(N_DEVICES)
    if devs is None:
        sc.write_worker_result(out_file, {"error": f"need {N_DEVICES} devices"})
        return
    model = _build(devs)                     # baseline batch sizes (no override)
    recon_shape = model.get_params('recon_shape')
    num_slices = int(recon_shape[2])
    idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=model.get_params('use_ror_mask'))
    num_pixels = int(len(idx))
    vb = int(model.view_batch_size_for_vmap)
    pb = int(model.pixel_batch_size_for_vmap)
    views_per_dev = SIZE[0] // N_DEVICES
    vb_eff = min(vb, views_per_dev)          # batch is capped by local view count
    bufs = _buffer_sizes(num_pixels, num_slices, vb_eff, pb)
    sino_shard = bufs["sino_view_shard"]
    result = {
        "size": list(SIZE), "n_devices": N_DEVICES,
        "num_pixels": num_pixels, "num_slices": num_slices,
        "view_batch_size_for_vmap": vb, "pixel_batch_size_for_vmap": pb,
        "view_batch_eff": vb_eff,
        "sino_shard_bytes": sino_shard,
        "buffers_bytes": bufs,
        "buffers_x_shard": {k: v / sino_shard for k, v in bufs.items()},
        "resident_sum_bytes": sum(bufs.values()),
        "resident_sum_x_shard": sum(bufs.values()) / sino_shard,
    }
    sc.write_worker_result(out_file, result)
    gb = 1024 ** 3
    print(f"[attribute] vb={vb} (eff {vb_eff}), pb={pb}, num_pixels={num_pixels}, "
          f"num_slices={num_slices}")
    for k, v in bufs.items():
        print(f"  {k:24s} {v/gb:7.3f} GB   {v/sino_shard:6.2f}× shard")
    print(f"  {'resident (sum)':24s} {sum(bufs.values())/gb:7.3f} GB   "
          f"{sum(bufs.values())/sino_shard:6.2f}× shard")


# ── Worker: PART C (slice-band config) ────────────────────────────────────────
def worker_measure(view_batch, pixel_batch, out_file, slice_band=-1):
    """Measure peak per-device memory + time for one configuration.

    PART C varies the back-projection slice band (batch sizes left at default) by
    setting model.back_project_slice_band, which controls how the slice axis is
    streamed in sharded back projection, in its own fresh subprocess so the
    cumulative peak reads cleanly.  (view_batch/pixel_batch are retained so a
    one-off batch override is still possible, but the batch sweep was removed.)
    """
    import mbirjax
    devs = sc.pick_devices(N_DEVICES)
    if devs is None:
        sc.write_worker_result(out_file, {"error": f"need {N_DEVICES} devices"})
        return
    try:
        model = _build(devs, view_batch, pixel_batch)
        if slice_band and slice_band > 0:
            model.back_project_slice_band = int(slice_band)
        # Report the tiling the code will actually use for this configuration.
        num_slices = int(model.get_params('recon_shape')[2])
        slices_per_dev = num_slices // N_DEVICES
        eff_band = model._slice_band_length(slices_per_dev, N_DEVICES)
        bands_per_owner = len(model._balanced_slice_bounds(slices_per_dev, eff_band))
        sino_np = np.random.default_rng(SEED).random(SIZE, dtype=np.float32)
        sino_sharded = model._shard_sinogram(sino_np)
        idx = mbirjax.gen_full_indices(model.get_params('recon_shape'),
                                       use_ror_mask=model.get_params('use_ror_mask'))
        stats, _ = sc.time_op(lambda: model.sparse_back_project(sino_sharded, idx),
                              WARMUP, TRIALS)
        mem_mb, mem_kind = sc.peak_memory_mb(devs)
        result = {
            "view_batch_req": view_batch, "pixel_batch_req": pixel_batch,
            "slice_band_req": slice_band,
            "view_batch_actual": int(model.view_batch_size_for_vmap),
            "pixel_batch_actual": int(model.pixel_batch_size_for_vmap),
            "slice_band_eff": int(eff_band), "bands_per_owner": int(bands_per_owner),
            "min_ms": stats["min_ms"], "mem_mb": mem_mb, "mem_kind": mem_kind,
        }
    except Exception as e:   # noqa: BLE001 — measurement harness: report, don't crash
        msg = str(e).replace("\n", " ")
        is_oom = any(k in msg.upper() for k in _OOM_MARKERS)
        result = {"view_batch_req": view_batch, "pixel_batch_req": pixel_batch,
                  "slice_band_req": slice_band, "oom": is_oom, "error": msg[:300]}
        print(f"  vb={view_batch} pb={pixel_batch} band={slice_band}  "
              f"{'OOM' if is_oom else 'ERROR'}: {msg[:120]}")
    sc.write_worker_result(out_file, result)


def run_worker(argv):
    p = argparse.ArgumentParser(description="mem attribution worker (internal)")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--mode", choices=["attribute", "measure"], required=True)
    p.add_argument("--view-batch", type=int, default=-1)
    p.add_argument("--pixel-batch", type=int, default=-1)
    p.add_argument("--slice-band", type=int, default=-1)
    p.add_argument("--out-file", required=True)
    a = p.parse_args(argv)
    import mbirjax  # device-setup-first: before any jax backend init
    if a.mode == "attribute":
        worker_attribute(a.out_file)
    else:
        worker_measure(a.view_batch, a.pixel_batch, a.out_file,
                       slice_band=a.slice_band)


# ── Orchestrator (touches no JAX) ─────────────────────────────────────────────
def _beta_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        os.pardir, os.pardir, os.pardir))


def main():
    script = os.path.abspath(__file__)
    beta_root = _beta_root()
    existing_pp = os.environ.get("PYTHONPATH", "")
    worker_env = {
        "PYTHONPATH": beta_root + (os.pathsep + existing_pp if existing_pp else ""),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9",
    }
    size_label = "x".join(str(s) for s in SIZE)
    print("=" * 78)
    print(f"  back-projection memory attribution — size {size_label}, "
          f"{N_DEVICES} devices")
    print(f"  beta root: {beta_root}")
    print("=" * 78)

    # PART A — attribution.
    print("\n--- PART A: buffer attribution (exact float32 sizes) ---")
    attr, rc = sc.run_worker(script, ["--worker", "--mode", "attribute"],
                             extra_env=worker_env)
    if attr is None or attr.get("error"):
        print(f"  attribution failed (rc={rc}): {attr}")
        return
    sino_shard = attr["sino_shard_bytes"]

    # (Part B, the view/pixel batch sweep, was removed: it confirmed batch tuning
    # leaves the peak flat -- see sharding_status.md.  Part C is the real lever.)

    # PART C — slice-band sweep (the structural lever): vary
    # back_project_slice_band with default batch sizes.  Larger band -> fewer,
    # bigger bands (more memory, fewer projector calls); smaller band -> the owner
    # gathers fewer slices per reduce (less memory, more calls).  The largest band
    # (= slices_per_dev, one band per owner) approximates no sub-tiling; the knee
    # is where peak stops falling or time starts rising.  Default band is
    # max(1, slices_per_dev // n_dev).
    spd = attr["num_slices"] // N_DEVICES
    band_vals, b = [], spd
    while b >= 8:
        band_vals.append(b)
        b //= 2
    default_band = max(1, spd // N_DEVICES)
    print("\n--- PART C: slice-band sweep (default batch sizes) ---")
    print(f"  slices_per_dev={spd}, default band={default_band}")
    print(f"{'band':>6} {'bands/own':>9} | {'min_ms':>9} | {'peak_MB':>9} {'peak/shard':>10}")
    print("-" * 56)
    band_rows = []
    for band in band_vals:
        res, rc = sc.run_worker(
            script, ["--worker", "--mode", "measure", "--slice-band", str(band)],
            extra_env=worker_env)
        if res is None:
            print(f"{band:>6} | worker produced no result (rc={rc})")
            continue
        if res.get("oom"):
            print(f"{band:>6} | OOM")
            band_rows.append(res)
            continue
        if res.get("error"):
            print(f"{band:>6} | ERROR: {str(res['error'])[:60]}")
            continue
        ratio = res["mem_mb"] / (sino_shard / (1024 ** 2))
        tag = "  (default)" if res["slice_band_eff"] == default_band else ""
        print(f"{res['slice_band_eff']:>6} {res['bands_per_owner']:>9} | "
              f"{res['min_ms']:>9.1f} | {res['mem_mb']:>9.1f} {ratio:>9.2f}×{tag}")
        band_rows.append({**res, "peak_over_shard": ratio})

    results = {"op": OP_NAME, "size": list(SIZE), "n_devices": N_DEVICES,
               "attribution": attr, "band_sweep": band_rows}
    sc.save_yaml(os.path.join(sc.RESULTS_DIR, f"{OP_NAME}_gpu.yaml"), results)
    print("\nReading: PART A names the buffers (the full-cylinder partial + vmap "
          "buffer\ndominate); PART C is the lever — how far the slice-band streaming "
          "drives\npeak/shard down, and the time cost of more, smaller bands.  Pick "
          "the band\nat the memory/throughput knee (the fbp ROW_FILTER_BATCH "
          "analogue).\n(Batch tuning was ruled out earlier — it left the peak flat — "
          "so Part B is gone.)")
    print("\nDone.")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        run_worker(sys.argv[1:])
    else:
        main()
