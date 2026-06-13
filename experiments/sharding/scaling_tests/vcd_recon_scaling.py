"""
experiments/sharding/scaling_tests/vcd_recon_scaling.py
───────────────────────────────────────────────────────
Scaling + correctness driver for end-to-end sharded VCD reconstruction
(TomographyModel.vcd_recon) under the view/slice sharding scheme.

VCD is the iterative loop that composes the validated sharded forward/back
projectors (all-gather / reduce-scatter) with the slice-sharded qGGMRF prior.  The
timed unit is the INTERNAL ``vcd_recon`` on a PRE-SHARDED sinogram with FIXED
partitions, returning the slice-sharded recon (no exit gather) — the analogue of
how the projector drivers time their internal sharded-contract method on
pre-sharded data.  This isolates the loop's compute + cross-device movement + the
peak transient memory of one full reconstruction, excluding the user-facing
``recon``'s entry shard / exit gather and partition build.

Determinism: VCD draws on the GLOBAL numpy RNG to shuffle subset order each
iteration, so the timed callable seeds the global RNG before each call — every
warmup/trial does identical work, and the result is reproducible across device
counts (only the distributed reduce order differs from single device).  Partitions
are built ONCE per size (outside the timed region) so partition construction is not
re-timed and is identical across device counts.

Correctness (setup worker) runs the FULL single-device ``recon`` at a small fixed
size and compares the recon volume against the prerelease baseline
(``vcd_recon_capture_baseline.py``) — a FLOAT-NOISE gate (iterative accumulation
amplifies reduce-order differences; the prior's no-halo path is bit-exact with
prerelease).  This is shared verbatim with the capture script via
``run_reference_recon`` so capture and check do the identical RNG draw sequence.

  - orchestrator (default, no args)  : spawns workers per size, collects YAML/plots.
  - worker --mode setup              : platform/devices + recon-vs-baseline check.
  - worker --mode measure --size ... : times + measures memory for one size, device
                                       counts DESCENDING so per-device peak reads
                                       correctly in the fresh process.

Mirrors sparse_forward_project_scaling.py (same isolated-subprocess harness, same
two plotted views, same top-of-file run configuration; no CLI args for the human).

Run from the BETA worktree root:

    python experiments/sharding/scaling_tests/vcd_recon_scaling.py
"""
import os
import gc
import sys
import argparse

import scaling_common as sc

import numpy as np

# NOTE: mbirjax (and therefore jax) is imported INSIDE the worker functions, not at
# the top level — see the long note in sparse_forward_project_scaling.py.  The
# orchestrator must stay JAX-free so it holds no device backend while a worker
# measures peak memory.


OP_NAME = "vcd_recon"

# ── Run configuration (edit here; no CLI args for the human) ──────────────────
DEVICE_COUNTS = [1, 2, 4]     # cluster has 4 GPUs (2 per NUMA node); 2->4 spans the NUMA split

# Number of VCD iterations per timed reconstruction.  VCD is iterative and much
# heavier per element than a single projector, so the sizes below are smaller than
# the projector drivers' and the iteration count is modest.
MAX_ITERATIONS = 15

# GPU memory knobs.
#
# IMPORTANT (measured 2026-06-05): peak_bytes_in_use is the peak *live* working set and is
# essentially the SAME whether preallocation is on or off (verified: 1d 512³ reads ~41.6 GB
# either way).  Under a generous budget XLA does no rematerialization, so this peak is the
# natural full-speed working set -- a REAL number, not a preallocation artifact (do not extrapolate
# across sizes though).  So PREALLOCATE=False does NOT reveal the capacity floor.
#
# To measure the true capacity floor / OOM threshold, ARTIFICIALLY RESTRICT the budget: keep
# PREALLOCATE=True and LOWER MEM_FRACTION (a hard pool cap).  XLA then rematerializes to fit, so
# the peak drops to the minimum-feasible (slower) -- and a size that exceeds it OOMs, which is the
# honest "max recon per GPU" signal.  (MEM_FRACTION is ignored when PREALLOCATE=False.)
MEM_PREALLOCATE = True     # leave True; flipping it does NOT change peak_bytes_in_use
MEM_FRACTION = 0.9         # pool fraction (hard cap when preallocating); LOWER it to probe the
                           # capacity floor / OOM threshold, e.g. 0.25 for a ~20 GB cap on an 80 GB GPU

# Problem sizes (n_views, n_rows, n_channels).  n_views and the slice axis (≈ n_rows
# for parallel beam) must both divide every device count used, else configure_sharding
# raises and the point is skipped.  All sizes below divide 1/2/3/4.  GPU uses the
# non-power-of-2 validation sizes used throughout the sharded-VCD work: 504³ is where
# the per-subset memory leak was found and fixed; 1008³ is where the 2→4-device 2.20×
# scaling was measured and where 1-device used to OOM (with the in-place error-sinogram
# fix it should now complete near the no-mesh ~50 GB).
SIZES = {
    "cpu": [(48, 48, 48), (96, 96, 96), (192, 192, 192)],
    "gpu": [(1024, 1024, 1024), (1800, 1800, 1800), (2048, 2048, 2048)],
}

WARMUP = 0
TRIALS = 1          # VCD is slow; a couple of timed trials is enough

# Correctness (full single-device recon vs prerelease baseline).  Imported from the
# capture script so capture and check share one definition (and one RNG sequence).
from vcd_recon_capture_baseline import (   # noqa: E402  (intentional: pure constants/fn)
    CORRECTNESS_SIZE, CORRECTNESS_SEED, CORRECTNESS_MAX_ITERATIONS)
CORRECTNESS_THRESHOLD = 1e-4

# Per-call subset-shuffle seed (timing determinism); distinct from the correctness
# seed so the two concerns are independent.
MEASURE_SEED = 7


# ── Op-specific builders (used by the worker) ─────────────────────────────────
def make_model(size, devices=None):
    """Build a ParallelBeamModel for (views, rows, channels), verbose off.

    Returns None if the requested device count does not evenly divide the sharded
    axes (configure_sharding raises) — the caller skips that point.
    """
    import mbirjax
    n_views, n_rows, n_channels = size
    angles = np.linspace(0, np.pi, n_views, endpoint=False)
    model = mbirjax.ParallelBeamModel((n_views, n_rows, n_channels), angles)
    model.set_params(verbose=0)
    if devices is not None:
        try:
            model.configure_sharding(devices)
        except ValueError as e:
            print(f"    (skip: {len(devices)} devices incompatible with size "
                  f"{sc.size_label(size)}: {e})")
            return None
    return model


def parse_size_label(label):
    """'256x256x256' -> (256, 256, 256)."""
    return tuple(int(x) for x in label.split("x"))


def build_partitions(ref_model, sino_np, max_iterations):
    """Build the VCD partitions + partition_sequence once (device-independent).

    Uses initialize_recon, which constructs the pixel partitions (consuming the
    global RNG) and the partition sequence.  Built outside the timed region and
    reused for every device count of this size so partition construction is not
    re-timed and is identical across device counts.
    """
    weights = get_weights(sino_np.shape)
    (_sino, _weights, _init, partitions, partition_sequence,
     _granularity, _reg) = ref_model.initialize_recon(
        sino_np, weights=weights, max_iterations=max_iterations, print_logs=False)
    return partitions, partition_sequence


def get_weights(sino_shape):
    """Return a numpy array of all 1s in the shape of sino."""
    return np.ones(sino_shape)


def run_vcd(model, sino_sharded, partitions, partition_sequence, max_iterations):
    """The timed op: one full VCD reconstruction on the pre-sharded sinogram.

    Seeds the global RNG so subset order is identical on every call (stable timing,
    reproducible across device counts).  ``init_recon=None`` lets vcd_recon compute
    its own direct_recon init (part of the realistic per-recon cost).  Returns the
    slice-sharded recon (no exit gather).
    """
    weights = get_weights(sino_sharded.shape)
    np.random.seed(MEASURE_SEED)
    recon, _stats = model.vcd_recon(
        sino_sharded, partitions, partition_sequence,
        stop_threshold_change_pct=0.0, weights=weights, init_recon=None)
    return recon


# ── Worker side (runs in an isolated subprocess) ──────────────────────────────
def worker_setup(out_file):
    """Report platform, device count/label, and the recon-vs-baseline correctness
    check (full single-device recon vs the prerelease reference volume)."""
    import mbirjax  # device-setup-first: before any jax import (via sc below)
    from vcd_recon_capture_baseline import run_reference_recon
    plat, max_dev = sc.detect_platform()
    dev_label = sc.device_label()

    recon_vol, _stats = run_reference_recon(
        CORRECTNESS_SIZE, CORRECTNESS_SEED, CORRECTNESS_MAX_ITERATIONS)

    corr = {"check": "single_device_recon vs prerelease_baseline",
            "size": list(CORRECTNESS_SIZE), "seed": CORRECTNESS_SEED,
            "max_iterations": CORRECTNESS_MAX_ITERATIONS,
            "threshold": CORRECTNESS_THRESHOLD}
    ref, bmeta = sc.load_baseline(OP_NAME)
    if ref is None:
        corr["baseline_present"] = False
        print("[setup] no prerelease baseline — run "
              "vcd_recon_capture_baseline.py from a prerelease checkout")
    else:
        captured_on = bmeta.get("captured_on_platform", "unknown")
        bm = sc.correctness_metrics(ref, recon_vol, threshold=CORRECTNESS_THRESHOLD)
        corr.update({"baseline_present": True, "baseline_platform": captured_on,
                     "current_platform": plat, "cross_platform": captured_on != plat,
                     "max_abs_diff": bm.get("max_abs_diff"),
                     "pct_above_threshold": bm.get("pct_above_threshold")})
        if "error" in bm:
            corr["shape_error"] = bm["error"]
            print(f"[setup] baseline shape mismatch: {bm['error']}")
        else:
            print(f"[setup] vs prerelease baseline ({captured_on}): "
                  f"max_abs_diff={bm['max_abs_diff']:.3e}  "
                  f"pct_above={bm['pct_above_threshold']:.6f}%"
                  + ("   <-- CROSS-PLATFORM" if captured_on != plat else ""))

    result = sc.build_setup_result(plat, max_dev, dev_label, corr)
    sc.write_worker_result(out_file, result)


def worker_measure(size_label, device_counts, warmup, trials, out_file):
    """Time + measure memory for one size, device counts DESCENDING.

    Descending order (8→4→2→1) makes per-device allocation ascending within this
    fresh process, so the cumulative peak_bytes_in_use equals each config's own
    allocation when read right after it.  An OOM stops the descent.  Results are
    written incrementally + a per-GPU throttle sample is recorded each config.
    """
    import mbirjax  # noqa: F401  (device-setup side effect; must precede jax init)
    size = parse_size_label(size_label)

    # Build the sinogram + the partitions ONCE for this size (shapes/partitions are
    # device-independent; outside the timing loop).
    ref_model = make_model(size, devices=None)
    np.random.seed(MEASURE_SEED)
    sino_np = np.random.rand(*size).astype(np.float32)
    partitions, partition_sequence = build_partitions(ref_model, sino_np, MAX_ITERATIONS)
    del ref_model
    gc.collect()

    def build_and_time(n, devs):
        model = make_model(size, devices=devs)
        if model is None:
            return None
        # vcd_recon logs via self.logger (normally set up by recon/initialize_recon,
        # which we bypass here); give each per-device model a quiet logger.
        model.setup_logger(print_logs=False)
        # Pre-shard the sinogram OUTSIDE the timing loop: measure the loop's
        # all-gather/reduce-scatter + compute, not the entry scatter.
        sino_sharded = model._shard_sinogram(sino_np)
        stats, _ = sc.time_op(
            lambda: run_vcd(model, sino_sharded, partitions,
                            partition_sequence, MAX_ITERATIONS),
            warmup, trials)
        mem_mb, mem_kind = sc.peak_memory_mb(devs)
        return stats, mem_mb, mem_kind

    sc.run_measure_loop(size_label, device_counts, out_file, build_and_time,
                        header_extra=f"  ({MAX_ITERATIONS} iters)")


def run_worker(argv):
    """Dispatch a --worker invocation (internal; the orchestrator builds argv)."""
    p = argparse.ArgumentParser(description="vcd_recon scaling worker (internal)")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--mode", choices=["setup", "measure"], required=True)
    p.add_argument("--size", default=None, help="LxRxC, for --mode measure")
    p.add_argument("--device-counts", type=int, nargs="+", default=None)
    p.add_argument("--warmup", type=int, default=WARMUP)
    p.add_argument("--trials", type=int, default=TRIALS)
    p.add_argument("--out-file", required=True)
    a = p.parse_args(argv)
    if a.mode == "setup":
        worker_setup(a.out_file)
    else:
        worker_measure(a.size, a.device_counts, a.warmup, a.trials, a.out_file)


# ── Orchestrator (default; touches no JAX) ────────────────────────────────────
def main():
    script = os.path.abspath(__file__)

    # Each worker inherits the beta worktree on PYTHONPATH + the JAX allocator
    # knobs (see sc.build_worker_env).  vcd exposes MEM_* so a low MEM_FRACTION can
    # probe the per-GPU capacity (OOM) threshold.
    worker_env = sc.build_worker_env(mem_fraction=MEM_FRACTION,
                                     preallocate=MEM_PREALLOCATE)
    if MEM_FRACTION < 0.9:
        print(f"  CAPACITY PROBE: MEM_FRACTION={MEM_FRACTION} (hard pool cap) — a size that "
              f"exceeds the floor will OOM; that OOM threshold is the honest max-recon-per-GPU.")

    print("=" * 72)
    print("  vcd_recon scaling — isolated-subprocess harness (orchestrator)")
    print(f"  beta root: {sc.beta_root()}")
    print(f"  iterations per recon: {MAX_ITERATIONS}")
    print("=" * 72)

    # 1. Setup worker: platform, device count/label, recon-vs-baseline correctness.
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
    device_counts = [n for n in (DEVICE_COUNTS or sc.default_device_counts(max_dev))
                     if n <= max_dev]
    if plat == "cpu":
        worker_env["MBIRJAX_NUM_CPU_DEVICES"] = str(max_dev)
    print(f"  sizes: {size_labels}")
    print(f"  device counts: {device_counts}")

    # 2. One fresh worker per size (clean peak memory); collect the grid.
    grid = {}
    failures_by_size = {}
    mem_kind = "n/a"
    for label in size_labels:
        args = ["--worker", "--mode", "measure", "--size", label,
                "--device-counts", *[str(n) for n in device_counts],
                "--warmup", str(WARMUP), "--trials", str(TRIALS)]
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
        sc.annotate_speedups(rows)
        sc.annotate_mem_fraction(rows)
        rows.sort(key=lambda r: r["n_devices"])
        grid[label] = rows
        by_n = {r["n_devices"]: r for r in rows}
        summary = "  ".join(f"{n}d={by_n[n]['speedup']:.2f}x" for n in sorted(by_n))
        print(f"  size {label} speedup: {summary}")

    # 3. Persist results (one YAML) and two plots.
    results = {
        "op": OP_NAME, "platform": plat, "device_label": dev_label,
        "mbirjax_path": mpath, "warmup": WARMUP, "trials": TRIALS,
        "max_iterations": MAX_ITERATIONS,
        "device_counts": device_counts, "sizes": size_labels,
        "mem_kind": mem_kind, "time_ideal": "voxels_views", "correctness": corr,
        "dev2dev_safe": dev2dev_safe, "topology": topology,
        "grid": grid, "failures": failures_by_size,
    }
    sc.save_yaml(os.path.join(sc.RESULTS_DIR, f"{OP_NAME}_{plat}.yaml"), results)
    if any(grid.values()):
        sc.plot_device_sweep(
            OP_NAME, grid, device_counts, size_labels, dev_label, mem_kind,
            os.path.join(sc.RESULTS_DIR, f"{OP_NAME}_{plat}_device_sweep.png"))
        sc.plot_size_sweep(
            OP_NAME, grid, device_counts, size_labels, dev_label, mem_kind,
            os.path.join(sc.RESULTS_DIR, f"{OP_NAME}_{plat}_size_sweep.png"))

    print("\nDone.")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        run_worker(sys.argv[1:])
    else:
        main()
