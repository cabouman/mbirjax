"""
experiments/sharding/scaling_tests/sparse_back_project_scaling.py
─────────────────────────────────────────────────────────────────
Scaling + correctness driver for sharded back projection
(TomographyModel.sparse_back_project) under the view/slice sharding scheme.

Back projection is a reduce-scatter: the sinogram is view-sharded, so each device
back-projects ITS views onto the full voxel cylinders (Phase 1), then for each
slice-owner the per-device partials are summed over that owner's slice band
(Phase 2).  The timed unit is the INTERNAL ``sparse_back_project`` on a
PRE-SHARDED sinogram, returning the slice-sharded recon-at-indices (no gather) —
the analogue of how fbp_filter is timed on pre-sharded input.  This isolates the
reduce-scatter + compute and, importantly, the **transient per-device partial
buffer** ``(num_pixels, num_slices)`` whose memory is the open question for the
"do we need slice sub-tiling" decision (see the sharding implementation plan).

This mirrors fbp_filter_scaling.py: same isolated-subprocess harness (an
orchestrator that touches no JAX spawns fresh worker subprocesses so per-device
peak memory reads correctly), same two plotted views of one (size × device-count)
grid, same top-of-file run configuration (no CLI args for the human).

It measures EACH sharded back-projection implementation in BACK_PROJECT_PATHS
('band' = slice-banded reduce-scatter, 'pixel' = pixel-batched) in its own fresh
worker process per (size, path), writes a YAML + two plots per path (suffixed
with the path), and prints a band-vs-pixel time/memory comparison table at the
end -- the head-to-head readout the band-vs-pixel decision rests on.

  - orchestrator (default, no args)  : spawns workers per (path, size), collects
                                       YAMLs/plots, prints the comparison.
  - worker --mode setup              : reports platform/devices + single-device
                                       correctness vs the prerelease baseline.
  - worker --mode measure --size --path ... : times + measures memory for one
                                       size and path, device counts DESCENDING
                                       (8→4→2→1) so per-device allocation is
                                       ascending in the fresh process and peak
                                       reads correctly.

Run from the BETA worktree root:

    python experiments/sharding/scaling_tests/sparse_back_project_scaling.py
"""
import os
import gc
import sys
import argparse

import scaling_common as sc

import numpy as np

# NOTE: mbirjax (and therefore jax) is imported INSIDE the worker functions, not
# at the top level, on purpose.  This one file runs in two roles: the default
# (no --worker) is the ORCHESTRATOR, which only spawns worker subprocesses and
# collects their results and must stay completely JAX-free; the --worker
# invocations are WORKER subprocesses that actually touch JAX.  A top-level
# `import mbirjax` would run in both roles, pulling jax into the orchestrator —
# so the orchestrator could hold a (GPU) backend while a worker is measuring
# peak_bytes_in_use, contaminating exactly the per-device memory number the
# isolated-subprocess design exists to measure cleanly.  Importing mbirjax only
# in the worker keeps the orchestrator a pure, JAX-free supervisor.  (The import
# also has a load-bearing side effect — it runs mbirjax._device_setup, which sets
# the virtual-CPU-device XLA flag, and must precede jax backend init; see the
# import site in worker_measure.)


OP_NAME = "sparse_back_project"

# ── Run configuration (edit here; no CLI args for the human) ──────────────────
# Device-count ladder.  None → the automatic powers-of-two ladder
# (default_device_counts).  Set explicitly to override — e.g. [1, 2, 3] to use
# only the first three GPUs and skip a known-bad 4th card (pick_devices takes the
# first n, so [0,1,2]).  Sizes must be divisible by each count used.
DEVICE_COUNTS = [1, 2, 3, 4]  # [1, 2, 3]
DEVICE_COUNTS = np.array(DEVICE_COUNTS)

# Problem sizes (n_views, n_rows, n_channels).  Back projection is MUCH heavier
# per element than fbp_filter (it is the projector: cost ~ views × pixels × psf ×
# slices), so the sizes here are smaller than the fbp ones — tune freely.
# Divisibility: the view axis (n_views) and the slice axis (≈ n_rows for parallel
# beam) must both be divisible by every device count in the ladder, else
# configure_sharding raises and the point is skipped.  The GPU sizes are
# divisible by 1/2/3/4 (multiples of 12, ~the old 256/512/1024) so the ladder can
# include 3 devices — useful for skipping a throttling 4th card on a node by
# running on the cooler first three GPUs (see DEVICE_COUNTS).
SIZES_BY_12 = {
    "cpu": [(64, 64, 64), (128, 128, 128), (256, 256, 256), (400, 400, 400)],
    "gpu": [(252, 252, 252), (504, 504, 504), (1008, 1008, 1008)],
}
SIZES_BY_16 = {
    "cpu": [(64, 64, 64), (128, 128, 128), (256, 256, 256), (400, 400, 400)],
    "gpu": [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)],
}

SIZE = SIZES_BY_12 if 3 in DEVICE_COUNTS else SIZES_BY_16

WARMUP = 1
TRIALS = 3
CORRECTNESS_THRESHOLD = 1e-4

# Sharded back-projection path(s) to measure.  The 'pixel' path was RETIRED in the
# P2 decision (2026-06-03): band is the sole sharded back projection, so this is
# just ('band',).  The path / B_p-sweep plumbing below (and the model's
# _back_project_path switch it set) is now vestigial — left in place for a future
# band-vs-alternative comparison; strip in a cleanup pass if not reused.
BACK_PROJECT_PATHS = ("band",)

# Was the pixel-path B_p sweep (pixel retired; keep None).
PIXEL_BATCH_SWEEP = None

CORRECTNESS_SIZE = (80, 48, 64)   # small, fixed, NON-symmetric (distinct views/
                                  # rows/channels) so shape/axis bugs surface;
                                  # comparison is size-independent
CORRECTNESS_SEED = 1234

# Substrings (upper-cased) that mark a caught failure as memory exhaustion.
# Back projection allocates via the projector / scatter-add path (not cuFFT), so
# the cuFFT-specific markers are not needed here — but they are harmless to keep
# and make this list reusable across ops.
_OOM_MARKERS = ("RESOURCE_EXHAUSTED", "OUT OF MEMORY", "OOM", "BAD_ALLOC",
                "FAILED TO ALLOCATE", "WORK AREA", "SCRATCH ALLOCATOR",
                "FAILED TO CREATE CUFFT")


# ── Op-specific builders (used by the worker) ─────────────────────────────────
def make_model(size, devices=None, path="band", pixel_batch=None):
    """Build a ParallelBeamModel for the given (views, rows, channels).

    ``path`` selects the sharded back-projection implementation ('band' = the
    slice-banded reduce-scatter, 'pixel' = the pixel-batched one) so the two can
    be measured side by side; it only affects the sharded (mesh) path.
    ``pixel_batch`` (when not None) pins the pixel path's B_p
    (back_project_pixel_batch) for the B_p sweep; None uses the auto default.

    Returns None if the requested device count does not evenly divide the
    sharded axes (configure_sharding raises) — the caller skips that point.
    """
    import mbirjax
    n_views, n_rows, n_channels = size
    angles = np.linspace(0, np.pi, n_views, endpoint=False)
    model = mbirjax.ParallelBeamModel((n_views, n_rows, n_channels), angles)
    if devices is not None:
        try:
            model.configure_sharding(devices)
        except ValueError as e:
            print(f"    (skip: {len(devices)} devices incompatible with size "
                  f"{sc.size_label(size)}: {e})")
            return None
    model._back_project_path = path
    if pixel_batch is not None:
        model.back_project_pixel_batch = pixel_batch
    return model


def make_input(size, seed=0):
    """Deterministic random sinogram of the given shape (numpy float32).

    Back projection is linear, so a random sinogram is a valid input for both
    timing and cross-mode/cross-platform correctness comparison.
    """
    rng = np.random.default_rng(seed)
    return rng.random(size, dtype=np.float32)


def make_indices(model):
    """Full field-of-view pixel indices for the model (deterministic per size)."""
    import mbirjax
    recon_shape = model.get_params('recon_shape')
    return mbirjax.gen_full_indices(recon_shape, use_ror_mask=model.get_params('use_ror_mask'))


def run_back_project(model, sino, pixel_indices):
    """The timed op: reduce-scatter back projection of a (pre-sharded) sinogram.

    Returns the slice-sharded recon-at-indices (num_pixels, num_slices); no
    gather, so this measures the reduce-scatter + compute, not the host transfer.
    """
    return model.sparse_back_project(sino, pixel_indices)


def parse_size_label(label):
    """'256x256x256' -> (256, 256, 256)."""
    return tuple(int(x) for x in label.split("x"))


# ── Worker side (runs in an isolated subprocess) ──────────────────────────────
def worker_setup(out_file):
    """Report platform, device count/label, and the single-device correctness
    check against the prerelease baseline.  One launch, before sizes are known."""
    import mbirjax  # device-setup-first: before any jax import (via sc below)
    plat, max_dev = sc.detect_platform()
    dev_label = sc.device_label()

    ref, meta = sc.load_baseline(OP_NAME)
    # Single-device (no mesh) == the unchanged prerelease back-projection body.
    model = make_model(CORRECTNESS_SIZE, devices=None)
    sino = make_input(CORRECTNESS_SIZE, seed=CORRECTNESS_SEED)
    idx = make_indices(model)
    beta_out = np.asarray(run_back_project(model, sino, idx))

    if ref is None:
        corr = {"baseline_present": False}
        print("[setup] no prerelease baseline; correctness skipped "
              "(run sparse_back_project_capture_baseline.py from a prerelease checkout)")
    else:
        captured_on = meta.get("captured_on_platform", "unknown")
        m = sc.correctness_metrics(ref, beta_out, threshold=CORRECTNESS_THRESHOLD)
        corr = {"baseline_present": True, "baseline_platform": captured_on,
                "current_platform": plat, "cross_platform": captured_on != plat, **m}
        print(f"[setup] correctness vs {captured_on} baseline: "
              f"max_abs_diff={m['max_abs_diff']:.3e}  "
              f"pct_above={m['pct_above_threshold']:.6f}%"
              + ("   <-- CROSS-PLATFORM" if captured_on != plat else ""))

    # Record which physical GPUs / interconnect / NUMA this allocation got, and
    # whether the host-bounce transfer path is active -- both can swing
    # multi-device performance, so they are logged with every run.
    topology = sc.gpu_topology() if plat == "gpu" else {}
    dev2dev_safe = None
    if plat == "gpu":
        try:
            import jax
            import mbirjax._sharding as mjs
            gpus = jax.devices("gpu")
            if len(gpus) > 1:
                dev2dev_safe = bool(mjs.is_dev2dev_safe(gpus))
        except Exception:   # noqa: BLE001 — best effort, never abort setup
            pass

    pkg_path = os.path.dirname(mbirjax.__file__)
    beta_state, branch = sc.beta_status(pkg_path)   # by git branch, not dir name
    result = {"platform": plat, "max_devices": max_dev, "device_label": dev_label,
              "mbirjax_path": pkg_path, "beta_state": beta_state, "branch": branch,
              "correctness": corr, "topology": topology, "dev2dev_safe": dev2dev_safe}
    sc.write_worker_result(out_file, result)
    print(f"[setup] platform={plat}  max_devices={max_dev}  ({dev_label})")
    if dev2dev_safe is not None:
        print(f"[setup] dev2dev_safe={dev2dev_safe}"
              + ("" if dev2dev_safe else "  <-- HOST-BOUNCE active (slow d2d!)"))
    if topology.get("devices"):
        print("[setup] GPUs:\n    " + topology["devices"].replace("\n", "\n    "))


def worker_measure(size_label, device_counts, warmup, trials, out_file, path="band",
                   pixel_batch=None):
    """Time + measure memory for one size and back-projection path, device counts
    DESCENDING.

    ``pixel_batch`` pins the pixel path's B_p (None = auto).  After each timed
    config a per-GPU clock/temp sample is recorded and a loud warning printed if
    any GPU was thermally throttling (which silently caps multi-device scaling).

    Descending order (8→4→2→1) makes per-device allocation ascending within this
    fresh process, so the cumulative peak_bytes_in_use equals each config's own
    allocation when read right after it.  Once a config OOMs, every later
    (fewer-device) config needs MORE per-device memory and would OOM too — so we
    catch the OOM, record it, and stop the descent.  Results are written
    incrementally so even a hard crash returns the completed configs.
    """
    # Imported for its IMPORT SIDE EFFECT, not its API (mbirjax is not used by
    # name in this function; make_model imports it again where it's actually
    # called).  mbirjax's first import runs mbirjax._device_setup, which sets
    # XLA_FLAGS=--xla_force_host_platform_device_count from MBIRJAX_NUM_CPU_DEVICES
    # (the orchestrator sets that env for CPU runs).  That flag only takes effect
    # if it is in place BEFORE JAX initializes its backend, and scaling_common
    # imports jax LAZILY — so the first thing that would trigger backend init is
    # sc.pick_devices(...) just below.  Importing mbirjax here, ahead of that,
    # guarantees the virtual CPU device count is established; without it the
    # worker would come up with a single CPU device and every n>1 config would be
    # skipped (so the whole point of the sweep — multi-device scaling — is lost).
    import mbirjax  # noqa: F401  (device-setup side effect; must precede jax init)
    size = parse_size_label(size_label)
    desc = sorted(set(device_counts), reverse=True)
    bp_label = "auto" if pixel_batch is None else str(pixel_batch)
    print(f"\n[measure {size_label} | path={path} | B_p={bp_label}]  "
          f"device counts (descending): {desc}")
    sino_np = make_input(size, seed=0)
    rows = []
    failures = []
    mem_kind = "n/a"

    def _publish():
        sc.write_worker_result(out_file, {"size": size_label, "mem_kind": mem_kind,
                                          "rows": rows, "failures": failures})

    for n in desc:
        devs = sc.pick_devices(n)
        if devs is None:
            print(f"  n_devices={n}: not enough devices, skipping")
            continue
        model = make_model(size, devices=devs, path=path, pixel_batch=pixel_batch)
        if model is None:
            continue
        try:
            # Pre-shard the sinogram and precompute indices OUTSIDE the timing
            # loop: measure the reduce-scatter + compute, not the entry scatter.
            sino = model._shard_sinogram(sino_np)
            idx = make_indices(model)
            stats, _ = sc.time_op(lambda: run_back_project(model, sino, idx),
                                  warmup, trials)
            mem_mb, mem_kind = sc.peak_memory_mb(devs)
            # Sample GPU clocks/temps right after the run (still warm): a card
            # throttling here means this multi-device timing is unreliable.
            gpu_state = sc.sample_gpu_state()
        except Exception as e:   # noqa: BLE001 — measurement harness: never abort the sweep
            msg = str(e).replace("\n", " ")
            is_oom = any(k in msg.upper() for k in _OOM_MARKERS)
            failures.append({"n_devices": n, "oom": is_oom, "error": msg[:300]})
            print(f"  n_devices={n:2d}  {'OOM' if is_oom else 'ERROR'}: {msg[:120]}")
            _publish()
            if is_oom:
                print(f"  stopping descent at {size_label}: fewer-device configs "
                      f"need more per-device memory and would also OOM")
                break
            continue
        hot = sc.throttled_gpus(gpu_state)
        rows.append({"n_devices": n, **stats, "mem_mb": mem_mb,
                     "gpu_state": gpu_state, "throttled": bool(hot)})
        print(f"  n_devices={n:2d}  min={stats['min_ms']:8.2f} ms  "
              f"mean={stats['mean_ms']:8.2f} ms  mem={mem_mb:8.1f} MB ({mem_kind})")
        if hot:
            print("  !! THROTTLING — this timing is UNRELIABLE: "
                  + ", ".join(f"GPU{g['index']}={g['sm_mhz']}MHz@{g['temp_c']}C"
                              for g in hot))
        # Publish partial progress and free this config before the next (larger)
        # one so peak_bytes_in_use reflects each config alone.
        _publish()
        del model, sino
        gc.collect()   # release device buffers before the next config allocates
    _publish()


def run_worker(argv):
    """Dispatch a --worker invocation (internal; the orchestrator builds argv)."""
    p = argparse.ArgumentParser(description="sparse_back_project scaling worker (internal)")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--mode", choices=["setup", "measure"], required=True)
    p.add_argument("--size", default=None, help="LxRxC, for --mode measure")
    p.add_argument("--device-counts", type=int, nargs="+", default=None)
    p.add_argument("--warmup", type=int, default=WARMUP)
    p.add_argument("--trials", type=int, default=TRIALS)
    p.add_argument("--path", default="band", help="sharded back-projection path")
    p.add_argument("--pixel-batch", type=int, default=None,
                   help="pin the pixel path's B_p (omit for auto)")
    p.add_argument("--out-file", required=True)
    a = p.parse_args(argv)
    if a.mode == "setup":
        worker_setup(a.out_file)
    else:
        worker_measure(a.size, a.device_counts, a.warmup, a.trials, a.out_file,
                       path=a.path, pixel_batch=a.pixel_batch)


# ── Orchestrator (default; touches no JAX) ────────────────────────────────────
def _beta_root():
    """Beta worktree root, derived from this file's location.

    This file is at <beta>/experiments/sharding/scaling_tests/, so the worktree
    root is three directories up from the file's directory.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        os.pardir, os.pardir, os.pardir))


def _print_path_comparison(grids_by_variant, size_labels, device_counts):
    """Print a side-by-side time/memory comparison of the measured variants.

    For each size and device count, shows each variant's min time and peak memory
    (with a [THROTTLED] mark when a GPU was throttling, i.e. that timing is
    unreliable), and -- when a 'band' variant is present -- each other variant's
    time/memory ratio vs band (a ratio below 1 means faster / smaller).  This is
    the head-to-head readout the band-vs-pixel and B_p decisions rest on.
    """
    variants = list(grids_by_variant.keys())
    print("\n" + "=" * 72)
    print("  variant comparison — min time (ms) / peak mem (MB) per (size, n_dev)")
    print("  (ratios are vs 'band'; [THROTTLED] = a GPU throttled, timing unreliable)")
    print("=" * 72)
    for label in size_labels:
        print(f"\n  {label}")
        byn = {v: {r["n_devices"]: r for r in grids_by_variant[v].get(label, [])}
               for v in variants}
        for n in device_counts:
            print(f"    n={n}")
            band = byn.get("band", {}).get(n)
            for v in variants:
                r = byn[v].get(n)
                if not r:
                    print(f"        {v:<16s} --")
                    continue
                line = f"        {v:<16s} {r['min_ms']:9.2f} ms  {r['mem_mb']:9.1f} MB"
                if band and v != "band":
                    line += (f"   vs band: t={r['min_ms'] / band['min_ms']:.2f} "
                             f"m={r['mem_mb'] / band['mem_mb']:.2f}")
                if r.get("throttled"):
                    line += "  [THROTTLED]"
                print(line)


def main():
    script = os.path.abspath(__file__)

    # Force the beta worktree onto each worker's PYTHONPATH so `import mbirjax`
    # resolves to beta regardless of how the orchestrator was launched.
    # Preallocate the pool up front (no per-call growth → clean timing), with
    # MEM_FRACTION raised so the largest configs don't OOM on the cap;
    # peak_bytes_in_use still tracks in-use tensors, so memory stays accurate.
    beta_root = _beta_root()
    if not os.path.isdir(os.path.join(beta_root, "mbirjax")):
        print(f"  WARNING: no mbirjax/ under derived beta root {beta_root}")
    existing_pp = os.environ.get("PYTHONPATH", "")
    worker_env = {
        "PYTHONPATH": beta_root + (os.pathsep + existing_pp if existing_pp else ""),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9",
    }

    print("=" * 72)
    print("  sparse_back_project scaling — isolated-subprocess harness (orchestrator)")
    print(f"  beta root: {beta_root}")
    print("=" * 72)

    # 1. Setup worker: platform, device count/label, correctness.
    setup, rc = sc.run_worker(
        script, ["--worker", "--mode", "setup"], extra_env=worker_env)
    if setup is None:
        print(f"  ERROR: setup worker produced no result (rc={rc}); aborting.")
        return
    plat = setup["platform"]
    max_dev = setup["max_devices"]
    dev_label = setup["device_label"]
    corr = setup["correctness"]
    mpath = setup.get("mbirjax_path", "?")
    beta_state = setup.get("beta_state", "unknown")
    branch = setup.get("branch")
    topology = setup.get("topology") or {}
    dev2dev_safe = setup.get("dev2dev_safe")
    if beta_state == "beta":
        label = f"*** beta ***  (branch {branch})"
    elif beta_state == "not-beta":
        label = f"### NOT beta — branch {branch} — check PYTHONPATH ###"
    else:
        label = "(branch undetermined — verify path manually)"
    print(f"  mbirjax: {label}   {mpath}")
    print(f"  platform: {plat}   max devices: {max_dev}   ({dev_label})")
    if corr.get("baseline_present"):
        print(f"  correctness: max_abs_diff={corr['max_abs_diff']:.3e}  "
              f"pct_above={corr['pct_above_threshold']:.6f}%"
              + ("   <-- CROSS-PLATFORM" if corr.get("cross_platform") else ""))
    else:
        print("  correctness: no baseline present")
    if dev2dev_safe is not None:
        print(f"  dev2dev_safe: {dev2dev_safe}"
              + ("" if dev2dev_safe else "  <-- HOST-BOUNCE active (slow d2d!)"))
    if topology.get("topo"):
        # The GPU↔GPU interconnect + NUMA placement (the allocation-quality
        # variable behind run-to-run multi-device scaling surprises).
        print("  GPU topology (nvidia-smi topo -m):")
        print("    " + topology["topo"].replace("\n", "\n    "))

    sizes = SIZES[plat]
    size_labels = [sc.size_label(s) for s in sizes]
    # DEVICE_COUNTS override (e.g. [1,2,3] to skip a bad 4th card); else auto.
    device_counts = [n for n in (DEVICE_COUNTS or sc.default_device_counts(max_dev))
                     if n <= max_dev]
    # On CPU, pin the virtual device count so each worker matches this count.
    if plat == "cpu":
        worker_env["MBIRJAX_NUM_CPU_DEVICES"] = str(max_dev)
    print(f"  sizes: {size_labels}")
    print(f"  device counts: {device_counts}")

    # 2. Build the variants to measure: each back-projection path, and for the
    #    pixel path optionally one variant per swept B_p value.  Each variant runs
    #    a fresh worker per size (clean peak memory) and gets its own YAML + plots.
    variants = []   # (path, pixel_batch, label)
    for path in BACK_PROJECT_PATHS:
        if path == "pixel" and PIXEL_BATCH_SWEEP:
            for bp in PIXEL_BATCH_SWEEP:
                variants.append(("pixel", bp,
                                 f"pixel_bp{'auto' if bp is None else bp}"))
        else:
            variants.append((path, None, path))

    grids_by_variant = {}
    for path, pixel_batch, vlabel in variants:
        print(f"\n=== variant: {vlabel} ===")
        grid = {}
        failures_by_size = {}
        mem_kind = "n/a"
        for label in size_labels:
            args = ["--worker", "--mode", "measure", "--size", label,
                    "--device-counts", *[str(n) for n in device_counts],
                    "--warmup", str(WARMUP), "--trials", str(TRIALS),
                    "--path", path]
            if pixel_batch is not None:
                args += ["--pixel-batch", str(pixel_batch)]
            res, rc = sc.run_worker(script, args, extra_env=worker_env)
            rows = (res or {}).get("rows") or []
            fails = (res or {}).get("failures") or []
            if fails:
                failures_by_size[label] = fails
                for fl in fails:
                    tag = "OOM" if fl.get("oom") else "ERROR"
                    print(f"  size {label}: n_devices={fl['n_devices']} {tag}")
            if not rows:
                print(f"  size {label}: worker returned no rows (rc={rc}); skipping")
                grid[label] = []
                continue
            mem_kind = res.get("mem_kind", mem_kind)
            sc.annotate_speedups(rows)        # speedup vs 1 device
            sc.annotate_mem_fraction(rows)    # memory relative to 1 device
            rows.sort(key=lambda r: r["n_devices"])   # ascending for readable YAML
            grid[label] = rows
            by_n = {r["n_devices"]: r for r in rows}
            summary = "  ".join(f"{n}d={by_n[n]['speedup']:.2f}x" for n in sorted(by_n))
            print(f"  size {label} speedup: {summary}")

        # Persist this variant's results (YAML) and two plots, suffixed with vlabel.
        results = {
            "op": OP_NAME, "back_project_path": path, "pixel_batch": pixel_batch,
            "platform": plat, "device_label": dev_label, "mbirjax_path": mpath,
            "warmup": WARMUP, "trials": TRIALS,
            "device_counts": device_counts, "sizes": size_labels,
            "mem_kind": mem_kind, "correctness": corr,
            "dev2dev_safe": dev2dev_safe, "topology": topology,
            "grid": grid, "failures": failures_by_size,
        }
        sc.save_yaml(os.path.join(sc.RESULTS_DIR, f"{OP_NAME}_{vlabel}_{plat}.yaml"),
                     results)
        if any(grid.values()):
            title = f"{OP_NAME} ({vlabel})"
            sc.plot_device_sweep(
                title, grid, device_counts, size_labels, dev_label, mem_kind,
                os.path.join(sc.RESULTS_DIR, f"{OP_NAME}_{vlabel}_{plat}_device_sweep.png"))
            sc.plot_size_sweep(
                title, grid, device_counts, size_labels, dev_label, mem_kind,
                os.path.join(sc.RESULTS_DIR, f"{OP_NAME}_{vlabel}_{plat}_size_sweep.png"))
        grids_by_variant[vlabel] = grid

    # 3. Side-by-side comparison of the variants (when more than one ran).
    if len(variants) > 1:
        _print_path_comparison(grids_by_variant, size_labels, device_counts)

    print("\nDone.")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        run_worker(sys.argv[1:])
    else:
        main()
