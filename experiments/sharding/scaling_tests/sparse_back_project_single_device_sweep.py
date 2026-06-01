"""
experiments/sharding/scaling_tests/sparse_back_project_single_device_sweep.py
─────────────────────────────────────────────────────────────────────────────
Single-GPU (n_dev=1) slice-band sweep for sharded back projection: how far does
streaming the slice axis cut peak memory on ONE device, and can it fit a recon
that OOMs unstreamed?  This is the "stretch the recon on one GPU" measurement.

Why a separate sweep: at n_dev=1 the *default* band is full (the budget default
bounds the cross-device reduce gather, which is vacuous with one device), so a
single device does NOT stream by default.  But the streaming code path handles
one device fine — set back_project_slice_band explicitly and it tiles the slice
axis on that one device.  This script sweeps that knob.

What streaming can and cannot shrink on one device:
  - shrinks: the full-cylinder partial + the vmap-over-views buffer (both
    proportional to num_slices) -> down to one band.
  - floors (NOT shrunk here): the full sinogram (input) and the recon output.
    Going below these needs host<->device streaming of sino/recon (a bigger
    change), so the table reports the floor for context.

Isolated-subprocess harness (orchestrator touches no JAX; one fresh worker per
band so the peak_bytes_in_use high-water mark reads cleanly).  Band lengths are
swept DESCENDING (full -> small); a band that OOMs is recorded and the sweep
CONTINUES (smaller bands use less memory), so the table shows exactly where
streaming starts to fit.  Run from the beta worktree root (on the GPU box):

    python experiments/sharding/scaling_tests/sparse_back_project_single_device_sweep.py
"""
import os
import sys
import argparse

import scaling_common as sc

import numpy as np

# mbirjax is imported INSIDE the workers (JAX-free orchestrator); see the other
# scaling drivers for the rationale.

OP_NAME = "sparse_back_project_1dev"

# ── Run configuration (edit here) ─────────────────────────────────────────────
N_DEVICES = 1
# Primary: 1024^3 (known ~29.7 GB unstreamed on an H100).  Stretch: a size that
# likely OOMs unstreamed but should fit once streamed — this is the headline
# "fit a bigger recon" demonstration.  The stretch size is slow (tens of seconds
# per call); trim or drop it if you just want the 1024^3 curve.
SIZES = [(1024, 1024, 1024), (1448, 1448, 1448)]
# Band lengths to sweep, as integer divisors of num_slices: full (=no streaming,
# the unstreamed baseline), then 1/2, 1/4, ... .  full may OOM at the stretch
# size — that is the point.
BAND_DIVISORS = [1, 2, 4, 8, 16, 32]
WARMUP = 1
TRIALS = 2           # memory is deterministic; a couple of trials steadies time
SEED = 0

_OOM_MARKERS = ("RESOURCE_EXHAUSTED", "OUT OF MEMORY", "OOM", "BAD_ALLOC",
                "FAILED TO ALLOCATE", "WORK AREA", "SCRATCH ALLOCATOR")
_F4 = 4   # bytes per float32


def parse_size_label(label):
    return tuple(int(x) for x in label.split("x"))


# ── Worker (one band on one device) ───────────────────────────────────────────
def worker_measure(size_label, slice_band, out_file):
    import mbirjax
    devs = sc.pick_devices(N_DEVICES)
    if devs is None:
        sc.write_worker_result(out_file, {"error": f"need {N_DEVICES} device(s)"})
        return
    size = parse_size_label(size_label)
    n_views, n_rows, n_chan = size
    try:
        angles = np.linspace(0, np.pi, n_views, endpoint=False)
        model = mbirjax.ParallelBeamModel(size, angles)
        model.configure_sharding(devs)                  # 1-device mesh
        if slice_band and slice_band > 0:
            model.back_project_slice_band = int(slice_band)
        recon_shape = model.get_params('recon_shape')
        num_slices = int(recon_shape[2])
        idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=model.use_ror_mask)
        num_pixels = int(len(idx))
        eff_band = model._slice_band_length(num_slices, N_DEVICES, num_pixels)
        num_bands = len(model._balanced_slice_bounds(num_slices, eff_band))
        sino_np = np.random.default_rng(SEED).random(size, dtype=np.float32)
        sino_sharded = model._shard_sinogram(sino_np)
        stats, _ = sc.time_op(lambda: model.sparse_back_project(sino_sharded, idx),
                              WARMUP, TRIALS)
        mem_mb, mem_kind = sc.peak_memory_mb(devs)
        result = {
            "size": size_label, "slice_band_req": slice_band,
            "slice_band_eff": int(eff_band), "num_bands": int(num_bands),
            "num_pixels": num_pixels, "num_slices": num_slices,
            "sino_bytes": n_views * n_rows * n_chan * _F4,
            "recon_out_bytes": num_pixels * num_slices * _F4,
            "min_ms": stats["min_ms"], "mem_mb": mem_mb, "mem_kind": mem_kind,
        }
    except Exception as e:   # noqa: BLE001 — harness: report, don't crash
        msg = str(e).replace("\n", " ")
        is_oom = any(k in msg.upper() for k in _OOM_MARKERS)
        result = {"size": size_label, "slice_band_req": slice_band,
                  "oom": is_oom, "error": msg[:300]}
        print(f"  {size_label} band={slice_band}  "
              f"{'OOM' if is_oom else 'ERROR'}: {msg[:110]}")
    sc.write_worker_result(out_file, result)


def run_worker(argv):
    p = argparse.ArgumentParser(description="single-device band sweep worker")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--size", required=True)
    p.add_argument("--slice-band", type=int, default=-1)
    p.add_argument("--out-file", required=True)
    a = p.parse_args(argv)
    import mbirjax  # device-setup-first
    worker_measure(a.size, a.slice_band, a.out_file)


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
    print("=" * 78)
    print(f"  back projection — SINGLE-DEVICE slice-band sweep ({N_DEVICES} device)")
    print(f"  beta root: {beta_root}")
    print("=" * 78)

    gb = 1024 ** 2 * 1024
    all_rows = {}
    for size in SIZES:
        label = "x".join(str(s) for s in size)
        num_slices = size[1]                              # parallel beam: slices = rows
        bands = sorted({max(1, num_slices // d) for d in BAND_DIVISORS}, reverse=True)
        print(f"\n### size {label}  (bands, full -> small) ###")
        print(f"{'band':>6} {'#bands':>6} | {'min_ms':>10} | {'peak_MB':>9} "
              f"{'vs_full':>7} {'peak/floor':>10}")
        print("-" * 60)
        rows, full_peak = [], None
        floor_mb = None
        for band in bands:
            res, rc = sc.run_worker(
                script, ["--worker", "--size", label, "--slice-band", str(band)],
                extra_env=worker_env)
            if res is None:
                print(f"{band:>6} | worker produced no result (rc={rc})")
                continue
            if res.get("oom"):
                print(f"{band:>6} {'':>6} | {'OOM':>10}")
                rows.append(res)
                continue
            if res.get("error"):
                print(f"{band:>6} | ERROR: {str(res['error'])[:50]}")
                continue
            if floor_mb is None:
                floor_mb = (res["sino_bytes"] + res["recon_out_bytes"]) / (1024 ** 2)
            if band == bands[0] and not res.get("oom"):
                full_peak = res["mem_mb"]
            vs_full = (res["mem_mb"] / full_peak) if full_peak else float("nan")
            peak_over_floor = res["mem_mb"] / floor_mb if floor_mb else float("nan")
            print(f"{res['slice_band_eff']:>6} {res['num_bands']:>6} | "
                  f"{res['min_ms']:>10.1f} | {res['mem_mb']:>9.1f} "
                  f"{vs_full:>6.2f}× {peak_over_floor:>9.2f}×")
            rows.append({**res, "vs_full": vs_full, "peak_over_floor": peak_over_floor})
        if floor_mb is not None:
            print(f"  floor (sino + recon output) = {floor_mb:.1f} MB "
                  f"({floor_mb/1024:.2f} GB) — streaming cannot go below this "
                  f"without host streaming")
        all_rows[label] = rows

    results = {"op": OP_NAME, "n_devices": N_DEVICES, "sizes":
               ["x".join(str(s) for s in z) for z in SIZES], "sweep": all_rows}
    sc.save_yaml(os.path.join(sc.RESULTS_DIR, f"{OP_NAME}_gpu.yaml"), results)
    print("\nReading: 'vs_full' is peak relative to the full-band (unstreamed) run; "
          "\n'peak/floor' is how close streaming gets to the sino+recon floor.  An "
          "OOM\nat full band that becomes a finite peak at a smaller band is the "
          "headline:\nstreaming fit a recon the unstreamed path could not.")
    print("\nDone.")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        run_worker(sys.argv[1:])
    else:
        main()
