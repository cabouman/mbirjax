"""
experiments/sharding/scaling_tests/cone_baseline_scaling.py
───────────────────────────────────────────────────────────
P6 Step 0 — single-device CONE baseline for time + peak memory, on the CURRENT
(pre-port) code.  This is the "no regression (judged, not literal)" reference the
P6 cone port is measured against (plan: p6_projector_rework_proposal.md §8a).

Why a small dedicated driver, not an edit to the three device-sweep drivers:
cone does not support sharding yet, so the device sweep is just n_dev=1 and the
sweep drivers' sharded machinery (configure_sharding, pre-shard, reduce-scatter)
is irrelevant.  This driver REUSES the proven measurement primitives in
``scaling_common`` (isolated-subprocess harness, ``time_op`` warmup/trials, the
free-previous-result discipline, ``peak_memory_mb``, throttle sampling, YAML) so
the cone numbers are produced with the SAME timing discipline as the parallel
ones — and, crucially, the pre-port-vs-post-port cone comparison is apples-to-
apples because both are produced by THIS tool.

It measures four ops, each in its own fresh worker subprocess per size (so peak
memory reads cleanly):

  * forward   — sparse_forward_project (single device)
  * back      — sparse_back_project    (single device)
  * vcd_const — full vcd_recon, constant (all-ones) weights
  * vcd_nonc  — full vcd_recon, NON-constant weights (defeats the const-weights
                fast path, exercising the weighted error-sinogram path)

GEOMETRY defaults to 'cone'; set it to 'parallel' to produce the parallel
single-device reference with identical discipline (the scaling-SHAPE sanity
reference; the multi-device parallel sweep already lives in the other drivers).

  - orchestrator (default, no args)  : spawns a worker per (op, size), collects
                                       YAMLs, prints a summary table.
  - worker --mode setup              : reports platform / devices.
  - worker --mode measure --op --size: times + measures memory for one op+size.

Run from the BETA worktree root (no CLI args for the human; edit the config block):

    python experiments/sharding/scaling_tests/cone_baseline_scaling.py
"""
import os
import sys
import gc
import argparse

import scaling_common as sc

import numpy as np

# NOTE: mbirjax (and therefore jax) is imported INSIDE the worker functions only,
# never at module top level.  The default (no --worker) role is the ORCHESTRATOR,
# which must stay JAX-free so it holds no device backend while a worker measures
# peak_bytes_in_use (the isolated-subprocess design).  The worker's first
# `import mbirjax` also runs mbirjax._device_setup (XLA device-count flag) and so
# must precede any jax backend init — see the import sites below.


# ── Run configuration (edit here; no CLI args for the human) ──────────────────
# Geometry under test.  'cone' is the P6 target; 'parallel' reproduces the
# single-device parallel reference with identical timing discipline.
GEOMETRY = "cone"

# Cone is single-device-only on the current code, so the baseline is n_dev=1.
# (Kept as a list so the same tool extends to a device sweep post-port by simply
# adding counts — at which point configure_sharding will be wired in make_model.)
DEVICE_COUNTS = [1]

# Ops to measure (each its own fresh worker per size).
OPS = ("forward", "back", "vcd_const", "vcd_nonc")

# Problem sizes as SINOGRAM shape (n_views, n_det_rows, n_det_channels).  For cone
# the RECON shape is auto-derived (magnification + detector extent), so it differs
# from the sinogram shape and is recorded per size in the YAML.  CPU sizes are
# kept modest (Greg runs 64-256 locally); GPU sizes are for the cluster.
SIZES = {
    "cpu": [(64, 64, 64), (128, 128, 128), (256, 256, 256)],
    "gpu": [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)],
}

# Representative cone geometry: magnification 2 (source_detector_dist =
# 4*channels, source_iso_dist = half that) — the test-suite convention
# (tests/geometries/test_fbp_fdk.py, test_vcd.py).
CONE_SDD_OVER_CHANNELS = 4.0

WARMUP = 1
# Timed trials per op.  The PROJECTORS are the primary scaling ruler (the
# cone-specific change), are cheap, and benefit from a min-of-a-few; VCD is a
# long correctness/INTEGRATION anchor (NOT a scaling ruler — few iters
# under-amortize fixed per-recon overhead), so one timed pass suffices.
TRIALS_BY_OP = {"forward": 3, "back": 3, "vcd_const": 1, "vcd_nonc": 1}
# VCD iterations per timed recon.  Kept small: VCD here checks that the integrated
# recon is correct/bounded, not how it scales (see the §8a ruler note).
MAX_ITERATIONS = 3

# Filename tag (distinguishes this clean re-run from the earlier swap-contaminated
# capture so both can be compared).  Empty string -> no tag.
RUN_TAG = "clean"

# Deterministic seeds (timing reproducibility; values don't affect time/memory).
INPUT_SEED = 0
MEASURE_SEED = 7         # subset-shuffle seed; VCD draws partitions from global RNG


# ── Op-specific builders (used by the worker) ─────────────────────────────────
def make_model(size):
    """Build a single-device model of the configured GEOMETRY for SINOGRAM ``size``.

    ``size`` = (n_views, n_det_rows, n_det_channels).  No sharding is configured
    (cone is single-device-only on the current code; this is the baseline).  The
    recon shape is auto-derived by the model.
    """
    import mbirjax
    n_views, n_rows, n_channels = size
    angles = np.linspace(0, np.pi, n_views, endpoint=False)
    if GEOMETRY == "parallel":
        model = mbirjax.ParallelBeamModel((n_views, n_rows, n_channels), angles)
    elif GEOMETRY == "cone":
        sdd = CONE_SDD_OVER_CHANNELS * n_channels
        sid = sdd / 2.0
        model = mbirjax.ConeBeamModel((n_views, n_rows, n_channels), angles,
                                      source_detector_dist=sdd, source_iso_dist=sid)
    else:
        raise ValueError(f"unknown GEOMETRY {GEOMETRY!r} (expected 'cone' or 'parallel')")
    model.set_params(verbose=0)
    return model


def make_indices(model):
    """Full field-of-view pixel indices for the model (deterministic per size)."""
    import mbirjax
    recon_shape = model.get_params('recon_shape')
    return mbirjax.gen_full_indices(recon_shape, use_ror_mask=model.get_params('use_ror_mask'))


def make_cylinders(num_pixels, num_slices, seed=INPUT_SEED):
    """Deterministic random recon cylinders (num_pixels, num_slices) float32."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((num_pixels, num_slices), dtype=np.float32)


def make_sinogram(size, seed=INPUT_SEED):
    """Deterministic random sinogram of SINOGRAM ``size`` (numpy float32).

    Projection is linear, so a random sinogram is a valid timing/memory input.
    """
    rng = np.random.default_rng(seed)
    return rng.random(size, dtype=np.float32)


def make_nonconst_weights(sino_shape, seed=INPUT_SEED + 1):
    """Deterministic NON-constant positive weights matching the sinogram shape.

    Non-constant weights defeat the constant-weights fast path so the weighted
    error-sinogram path is exercised (the path cone's sharded VCD will also use).
    Values in [0.5, 1.5) — positive and varying; the magnitude is immaterial to
    time/memory.
    """
    rng = np.random.default_rng(seed)
    return (0.5 + rng.random(sino_shape, dtype=np.float32)).astype(np.float32)


def build_partitions(model, sino_np, weights, max_iterations):
    """Build the VCD partitions + sequence once (device-independent, outside timing).

    Uses initialize_recon, which constructs the pixel partitions (consuming the
    global RNG) and the partition sequence.
    """
    (_sino, _weights, _init, partitions, partition_sequence,
     _granularity, _reg) = model.initialize_recon(
        sino_np, weights=weights, max_iterations=max_iterations, print_logs=False)
    return partitions, partition_sequence


def run_forward(model, cylinders, pixel_indices):
    """Timed op: single-device forward projection."""
    return model.sparse_forward_project(cylinders, pixel_indices)


def run_back(model, sino, pixel_indices):
    """Timed op: single-device back projection."""
    return model.sparse_back_project(sino, pixel_indices)


def run_vcd(model, sino_np, weights, partitions, partition_sequence, max_iterations):
    """Timed op: one full VCD reconstruction (single device).

    Seeds the global RNG so the subset order is identical on every call (stable
    timing).  ``init_recon=None`` lets vcd_recon compute its own direct_recon init
    (part of the realistic per-recon cost).
    """
    np.random.seed(MEASURE_SEED)
    recon, _stats = model.vcd_recon(
        sino_np, partitions, partition_sequence,
        stop_threshold_change_pct=0.0, weights=weights, init_recon=None)
    return recon


def parse_size_label(label):
    """'256x256x256' -> (256, 256, 256)."""
    return tuple(int(x) for x in label.split("x"))


# ── Worker side (runs in an isolated subprocess) ──────────────────────────────
def worker_setup(out_file):
    """Report platform + device count/label.  No cross-version baseline exists for
    cone (none is checked in; binaries don't go on github), so correctness here is
    a light shape/finite sanity at a small size rather than a baseline diff."""
    import mbirjax  # device-setup-first: before any jax import (via sc below)
    plat, max_dev = sc.detect_platform()
    dev_label = sc.device_label()

    # Light sanity: a small single-device recon is finite and has the right shape.
    small = (40, 40, 64)
    model = make_model(small)
    idx = make_indices(model)
    sino = make_sinogram(small, seed=INPUT_SEED)
    weights = make_nonconst_weights(small)
    parts, seq = build_partitions(model, sino, np.ones(small, np.float32), MAX_ITERATIONS)
    model.setup_logger(print_logs=False)
    recon = np.asarray(run_vcd(model, sino, np.ones(small, np.float32), parts, seq,
                               max_iterations=3))
    recon_shape = tuple(int(x) for x in model.get_params('recon_shape'))
    finite = bool(np.all(np.isfinite(recon)))
    corr = {"check": "single_device shape+finite sanity (no checked-in baseline)",
            "geometry": GEOMETRY, "sino_size": list(small),
            "recon_shape": list(recon_shape), "recon_finite": finite,
            "baseline_present": False}
    print(f"[setup] {GEOMETRY} sanity: recon_shape={recon_shape} finite={finite}")

    result = sc.build_setup_result(plat, max_dev, dev_label, corr)
    sc.write_worker_result(out_file, result)


def worker_measure(op, size_label, device_counts, warmup, trials, out_file):
    """Time + measure memory for one op and SINOGRAM size, device counts descending.

    For cone the only count is 1 (single device).  build_and_time prepares the
    op's input OUTSIDE the timing loop and times only the op.
    """
    import mbirjax  # noqa: F401  (device-setup side effect; must precede jax init)
    size = parse_size_label(size_label)
    sino_np = make_sinogram(size, seed=INPUT_SEED)

    # Build the model + op input once for this size (device-independent here, since
    # the only device count is 1; prepared outside the timing loop).
    base_model = make_model(size)
    recon_shape = tuple(int(x) for x in base_model.get_params('recon_shape'))
    idx = make_indices(base_model)
    num_pixels = len(idx)
    num_slices = recon_shape[2]

    if op in ("vcd_const", "vcd_nonc"):
        weights = (np.ones(size, np.float32) if op == "vcd_const"
                   else make_nonconst_weights(size))
        partitions, partition_sequence = build_partitions(
            base_model, sino_np, weights, MAX_ITERATIONS)
    elif op == "forward":
        cylinders = make_cylinders(num_pixels, num_slices, seed=INPUT_SEED)
    del base_model
    gc.collect()

    def build_and_time(n, devs):
        model = make_model(size)               # n is always 1 (single device)
        if op == "forward":
            run_fn = lambda: run_forward(model, cylinders, idx)
        elif op == "back":
            run_fn = lambda: run_back(model, sino_np, idx)
        else:
            model.setup_logger(print_logs=False)
            run_fn = lambda: run_vcd(model, sino_np, weights, partitions,
                                     partition_sequence, MAX_ITERATIONS)
        stats, _ = sc.time_op(run_fn, warmup, trials)
        mem_mb, mem_kind = sc.peak_memory_mb(devs)
        return stats, mem_mb, mem_kind

    rows, _failures = sc.run_measure_loop(
        size_label, device_counts, out_file, build_and_time,
        header_extra=f" | {GEOMETRY} | op={op} | recon={recon_shape}")
    # Record the auto-derived recon shape alongside the rows (cone's recon shape
    # differs from the sinogram size and is needed to interpret the numbers).
    for r in rows:
        r["recon_shape"] = list(recon_shape)
    sc.write_worker_result(out_file, {"size": size_label, "op": op,
                                      "recon_shape": list(recon_shape),
                                      "rows": rows})


def run_worker(argv):
    """Dispatch a --worker invocation (internal; the orchestrator builds argv)."""
    p = argparse.ArgumentParser(description="cone baseline worker (internal)")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--mode", choices=["setup", "measure"], required=True)
    p.add_argument("--op", default=None, choices=list(OPS))
    p.add_argument("--size", default=None, help="LxRxC, for --mode measure")
    p.add_argument("--device-counts", type=int, nargs="+", default=None)
    p.add_argument("--warmup", type=int, default=WARMUP)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--out-file", required=True)
    a = p.parse_args(argv)
    if a.mode == "setup":
        worker_setup(a.out_file)
    else:
        worker_measure(a.op, a.size, a.device_counts, a.warmup, a.trials, a.out_file)


# ── Orchestrator (default; touches no JAX) ────────────────────────────────────
def _print_summary(grids_by_op, size_labels):
    """Print min time (ms) / peak mem (MB) per (op, size) — the baseline table."""
    print("\n" + "=" * 72)
    print(f"  {GEOMETRY} single-device baseline — min time (ms) / peak mem (MB)")
    print("  ([THROTTLED] = a GPU throttled, timing unreliable)")
    print("=" * 72)
    header = "  {:<10s}".format("size") + "".join(f"{op:>22s}" for op in OPS)
    print(header)
    for label in size_labels:
        cells = []
        for op in OPS:
            rows = grids_by_op.get(op, {}).get(label, [])
            r = next((x for x in rows if x["n_devices"] == 1), None)
            if not r:
                cells.append(f"{'--':>22s}")
                continue
            mark = " !" if r.get("throttled") else ""
            cells.append(f"{r['min_ms']:9.1f}/{r['mem_mb']:8.1f}{mark:>3s}".rjust(22))
        print(f"  {label:<10s}" + "".join(cells))
    print("\n  (recon shapes differ from sinogram size for cone — see YAMLs.)")


def main():
    script = os.path.abspath(__file__)
    worker_env = sc.build_worker_env()

    print("=" * 72)
    print(f"  cone baseline ({GEOMETRY}) — isolated-subprocess harness (orchestrator)")
    print(f"  beta root: {sc.beta_root()}")
    print("=" * 72)

    # 1. Setup worker: platform, device count/label, sanity.
    setup, rc = sc.run_worker(
        script, ["--worker", "--mode", "setup"], extra_env=worker_env)
    if setup is None:
        print(f"  ERROR: setup worker produced no result (rc={rc}); aborting.")
        return
    plat, max_dev, dev_label, corr, mpath = sc.print_setup_banner(setup)
    topology = setup.get("topology") or {}
    dev2dev_safe = setup.get("dev2dev_safe")

    sizes = SIZES[plat]
    size_labels = [sc.size_label(s) for s in sizes]
    device_counts = [n for n in DEVICE_COUNTS if n <= max_dev]
    if plat == "cpu":
        worker_env["MBIRJAX_NUM_CPU_DEVICES"] = str(max(device_counts))
    print(f"  geometry: {GEOMETRY}   sizes: {size_labels}   device counts: {device_counts}")

    # 2. One fresh worker per (op, size) for a clean peak-memory read.
    grids_by_op = {}
    for op in OPS:
        print(f"\n=== op: {op} ===")
        grid = {}
        for label in size_labels:
            args = ["--worker", "--mode", "measure", "--op", op, "--size", label,
                    "--device-counts", *[str(n) for n in device_counts],
                    "--warmup", str(WARMUP), "--trials", str(TRIALS_BY_OP[op])]
            res, rc = sc.run_worker(script, args, extra_env=worker_env)
            rows = (res or {}).get("rows") or []
            if not rows:
                print(f"  size {label}: worker returned no rows (rc={rc}); skipping")
                grid[label] = []
                continue
            grid[label] = rows
            r1 = next((r for r in rows if r["n_devices"] == 1), rows[0])
            print(f"  size {label}: min={r1['min_ms']:.1f} ms  mem={r1['mem_mb']:.1f} MB"
                  + ("  [THROTTLED]" if r1.get("throttled") else ""))
        grids_by_op[op] = grid

    # 3. Persist one YAML for the whole baseline run + print the summary table.
    results = {
        "kind": "cone_baseline", "geometry": GEOMETRY, "platform": plat,
        "device_label": dev_label, "mbirjax_path": mpath,
        "warmup": WARMUP, "trials_by_op": TRIALS_BY_OP, "max_iterations": MAX_ITERATIONS,
        "device_counts": device_counts, "sizes": size_labels, "ops": list(OPS),
        "cone_sdd_over_channels": CONE_SDD_OVER_CHANNELS, "run_tag": RUN_TAG,
        "correctness": corr, "dev2dev_safe": dev2dev_safe, "topology": topology,
        "grids_by_op": grids_by_op,
    }
    tag = f"_{RUN_TAG}" if RUN_TAG else ""
    sc.save_yaml(os.path.join(sc.RESULTS_DIR, f"cone_baseline_{GEOMETRY}{tag}_{plat}.yaml"),
                 results)
    _print_summary(grids_by_op, size_labels)
    print("\nDone.")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        run_worker(sys.argv[1:])
    else:
        main()
