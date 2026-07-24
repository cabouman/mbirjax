"""
plans/experiments/sharding/scaling_tests/cone_baseline_scaling.py
───────────────────────────────────────────────────────────
P6 — CONE baseline for time + peak memory (single- AND multi-device), on the
CURRENT code.  This is the "no regression (judged, not literal)" reference the
P6 cone port is measured against (plan: p6_projector_rework_proposal.md §8a) and,
now that cone shards (B4), the multi-device scaling sweep (time/peak per count).

Why a small dedicated driver, not an edit to the three device-sweep drivers:
this driver fixes the cone geometry + magnification and auto-derives the recon
shape (which the generic drivers don't), and sweeps device counts by PINNING each
count with ``model.configure_devices(devs)`` so a count actually uses that many
devices — without the pin, every model auto-shards across ALL devices at
construction (set_devices()), so each count would measure the same max-device
run.  This driver REUSES the proven measurement primitives in
``scaling_common`` (isolated-subprocess harness, ``time_op`` warmup/trials, the
free-previous-result discipline, ``peak_memory_mb``, throttle sampling, YAML) so
the cone numbers are produced with the SAME timing discipline as the parallel
ones — and, crucially, the pre-port-vs-post-port cone comparison is apples-to-
apples because both are produced by THIS tool.

It measures three ops, each in its own fresh worker subprocess per size (so peak
memory reads cleanly):

  * forward   — sparse_forward_project
  * back      — sparse_back_project
  * vcd_const — full vcd_recon, constant (all-ones) weights

GEOMETRY defaults to 'cone'; set it to 'parallel' to produce the parallel
single-device reference with identical discipline (the scaling-SHAPE sanity
reference; the multi-device parallel sweep already lives in the other drivers).

  - orchestrator (default, no args)  : spawns a worker per (op, size), collects
                                       YAMLs, prints a summary table.
  - worker --mode setup              : reports platform / devices.
  - worker --mode measure --op --size: times + measures memory for one op+size.

Run from the BETA worktree root (no CLI args for the human; edit the config block):

    python plans/experiments/sharding/scaling_tests/cone_baseline_scaling.py
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
GEOMETRY = "parallel"

# Device counts to sweep.  Each (op, size) worker measures every count the hardware
# has, PINNING each with model.configure_devices() (see build_and_time).  A count that
# doesn't divide the recon slice axis is skipped for cone (slice padding is P6 B5, not
# done); the cubic sweep sizes below divide 1/2/4 cleanly, so no count is skipped there.
DEVICE_COUNTS = [1, 2, 4]

# Ops to measure (each its own fresh worker per size).
OPS = ("direct_filter", )  #, "forward", "back", "vcd_const")

# Ops that run a full VCD recon (subject to the VCD_MIN_DEVICES skip below).
VCD_OPS = ("vcd_const",)

# Skip a VCD measurement that is known not to fit, to avoid a long near-OOM
# rematerialization thrash before the inevitable OOM.  Maps a sinogram-size label to
# the SMALLEST device count to ATTEMPT for the VCD ops at that size; smaller counts are
# skipped with a note (the projectors are cheap and always run every count).  The n=1
# OOM is itself the capacity result, so let it OOM ONCE, then raise the floor for reruns.
# Empty dict -> attempt every count.  Example below: a 1024^3 cone VCD OOMs on one 80 GB GPU.
VCD_MIN_DEVICES = {
    "1024x1024x1024": 2,
}

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
TRIALS_BY_OP = {"forward": 3, "back": 3, "vcd_const": 1, "direct_filter": 3}
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


def build_partitions(model, sino_np, weights, max_iterations):
    """Build the VCD partitions + sequence once (device-independent, outside timing).

    Uses initialize_recon, which constructs the pixel partitions (consuming the
    global RNG) and the partition sequence.
    """
    (_sino, _weights, _init, partitions, partition_sequence,
     _granularity, _reg) = model.initialize_recon(
        sino_np, weights=weights, max_iterations=max_iterations, print_logs=False)
    return partitions, partition_sequence

def run_filter(model, sino):
    """Timed op: the FBP/FDK filter, kept in the device (sharded) form.

    ``output_sharded=True`` so we measure the FILTER, not a full-sinogram gather at exit:
    the user-facing default gathers (a fixed full-sinogram cost that does not shard --
    it dominates the fast filter and flattens the scaling), whereas the dedicated
    fbp_filter_scaling baseline measured prerelease's INTERNAL fbp_filter, which returned
    the sharded form with no gather.  Falls back to the plain call on code that predates
    the output_sharded kwarg (e.g. prerelease's fbp_filter signature)."""
    try:
        return model.direct_filter(sino, output_sharded=True)
    except TypeError:
        return model.direct_filter(sino)


def run_forward(model, cylinders, pixel_indices):
    """Timed op: single-device forward projection."""
    return model.sparse_forward_project(cylinders, pixel_indices)


def run_back(model, sino, pixel_indices):
    """Timed op: single-device back projection."""
    return model.sparse_back_project(sino, pixel_indices)


def to_device(model, arr, kind):
    """Pre-place a HOST input on the model's device form, OUTSIDE the timing loop.

    The timed op must measure compute, not the host->device transfer + scatter that a
    numpy input incurs on every call (fbp_filter_scaling does the same -- "measure the
    op, not the scatter").  This matters most for the fast filter op, where the transfer
    otherwise dominates; it also de-inflates the projector times.  ``kind`` is 'sino'
    (view-sharded) or 'recon' (slice-sharded).  Falls back to a single-device device_put
    on pre-sharding code (f23d3964), where the _shard_* helpers do not exist.  Blocks so
    the transfer is fully complete before timing begins.
    """
    import jax
    if kind == "sino" and hasattr(model, "_shard_sinogram"):
        placed = model._shard_sinogram(arr)
    elif kind == "recon" and hasattr(model, "_shard_recon"):
        placed = model._shard_recon(arr)
    else:
        placed = jax.device_put(arr, model.main_device)
    jax.block_until_ready(placed)
    return placed


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


def path_info(model, op, devs, num_pixels, num_slices):
    """Record WHICH code path this measurement used, so the YAML is self-documenting.

    The key field is ``is_sharded``: True = the placement/banded path (a 1-device
    sharded run is a trivial mesh, NOT the legacy single-device path); False = the
    legacy monolithic path (e.g. pre-sharding code).  These two paths have different
    time/memory at the same device count, so recording the flag per row removes the
    legacy-vs-sharded ambiguity that bit the n=1 comparison.  For the back op we also
    record the sharded band length / band count (the streaming that drives back memory
    and the horizontal-recompute cost), best-effort so a future internal-API change
    records None instead of breaking the measurement.
    """
    info = {"is_sharded": True,   # every model runs the placement/banded path now (is_sharded retired)
            "n_shard_devices": len(getattr(model, "shard_devices", None) or devs),
            "platform": devs[0].platform}
    if op == "back":
        try:
            slices_per_dev = num_slices // len(devs)
            fixed = getattr(model, "back_project_slice_band", None)
            band_len = model._slice_band_length(slices_per_dev, len(devs), num_pixels,
                                                fixed_band=fixed)
            bounds = model._balanced_slice_bounds(slices_per_dev, band_len)
            info["back_band_len"] = int(band_len)
            info["back_n_bands_per_shard"] = len(bounds)
        except Exception:   # internal API differs (e.g. legacy code) -> record None
            info["back_band_len"] = None
            info["back_n_bands_per_shard"] = None
    return info


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

    build_and_time prepares the op's input OUTSIDE the timing loop, PINS the model
    to exactly the device count under test (model.configure_devices(devs)), and times
    only the op.  Host inputs (sinogram, cylinders, indices, VCD partitions) are built
    once on a single-device base model and reused across counts; the sharded
    recon/projector entry re-places them per configuration.
    """
    import mbirjax  # noqa: F401  (device-setup side effect; must precede jax init)
    size = parse_size_label(size_label)
    sino_np = make_sinogram(size, seed=INPUT_SEED)

    # Build the model + op input once for this size (device-independent: host arrays the
    # per-count models re-place at entry).  Pin the base model to ONE device so the
    # derived inputs (especially the VCD partitions from initialize_recon) carry no
    # multi-device placement; build_and_time configures the real count per measurement.
    base_model = make_model(size)
    if hasattr(base_model, "configure_devices"):   # absent on pre-sharding code (f23d3964)
        base_model.configure_devices(1)
    recon_shape = tuple(int(x) for x in base_model.get_params('recon_shape'))
    idx = make_indices(base_model)
    num_pixels = len(idx)
    num_slices = recon_shape[2]

    if op in VCD_OPS:
        weights = np.ones(size, np.float32)
        partitions, partition_sequence = build_partitions(
            base_model, sino_np, weights, MAX_ITERATIONS)
    elif op == "forward":
        cylinders = make_cylinders(num_pixels, num_slices, seed=INPUT_SEED)
    del base_model
    gc.collect()

    # Path info per measured device count (merged into the rows after the loop, so the
    # YAML records which code path each row used -- see path_info()).
    path_by_n = {}

    def build_and_time(n, devs):
        # Skip a VCD config known not to fit (avoids a long near-OOM rematerialization
        # thrash before the inevitable OOM); the projectors are cheap and always run.
        if op in VCD_OPS and n < VCD_MIN_DEVICES.get(size_label, 1):
            print(f"  n_devices={n}: skipping {op} at {size_label} "
                  f"(below VCD_MIN_DEVICES={VCD_MIN_DEVICES.get(size_label)}; known OOM)")
            return None
        model = make_model(size)
        # Pin EXACTLY these n devices (devs == pick_devices(n)), so the model runs on the
        # same devices peak_memory_mb(devs) reads.  Without this the model auto-shards
        # across ALL available devices at construction (set_devices()), making every
        # device-count iteration measure the same max-device run (the flat-curve bug).
        # hasattr guard: configure_devices is absent on pre-sharding code (e.g. the
        # f23d3964 single-device cone baseline) -- run that with DEVICE_COUNTS = [1] so
        # the SAME script measures the apples-to-apples single-device reference.
        if hasattr(model, "configure_devices"):
            model.configure_devices(devs)
        # Cone slice-padding (B5) is not implemented: a count that doesn't divide the
        # recon slice axis pads, and the cone forward gather would assemble the padded
        # cylinder (wrong numbers).  Skip such counts rather than report garbage; the
        # configured cubic sweep sizes divide cleanly, so this is purely defensive.
        if GEOMETRY == "cone" and getattr(model, "recon_placement", None) is not None \
                and model.recon_placement.is_padded:
            print(f"  n_devices={n}: slice axis not divisible by {n} "
                  f"(cone padding is P6 B5, not done) — skipping")
            return None
        # Record which code path this row used (legacy vs sharded; back band count).
        path_by_n[n] = path_info(model, op, devs, num_pixels, num_slices)
        # Pre-place the big host inputs on this config's device form OUTSIDE the timing
        # loop, so time_op measures the op and not the per-call host->device transfer +
        # scatter (see to_device).  idx is tiny (left on host); VCD keeps its numpy inputs
        # since vcd_recon owns its entry placement and the one-time transfer is negligible
        # against a minutes-long recon.
        if op == "direct_filter":
            sino_dev = to_device(model, sino_np, "sino")
            run_fn = lambda: run_filter(model, sino_dev)
        elif op == "forward":
            cyl_dev = to_device(model, cylinders, "recon")
            run_fn = lambda: run_forward(model, cyl_dev, idx)
        elif op == "back":
            sino_dev = to_device(model, sino_np, "sino")
            run_fn = lambda: run_back(model, sino_dev, idx)
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
    # Record the auto-derived recon shape + the code-path info alongside the rows (cone's
    # recon shape differs from the sinogram size; the path info says legacy vs sharded).
    for r in rows:
        r["recon_shape"] = list(recon_shape)
        r.update(path_by_n.get(r["n_devices"], {}))
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
    """Per op: min time (ms) / peak mem (MB) / speedup-vs-1-device, for each
    (size, device count) — the scaling table.  ``[!]`` flags a throttled GPU
    (timing unreliable).  Speedup is min_ms(1 dev) / min_ms(n dev); on CPU the
    memory column is whole-process RSS (not per-device), so it does not shard."""
    for op in OPS:
        print("\n" + "=" * 78)
        print(f"  {GEOMETRY} {op} — min time (ms) / peak mem (MB) / speedup vs 1 device")
        print("=" * 78)
        print("  {:<11s}{:>7s}{:>12s}{:>12s}{:>10s}".format(
            "size", "n_dev", "min_ms", "mem_mb", "speedup"))
        for label in size_labels:
            rows = sorted(grids_by_op.get(op, {}).get(label, []),
                          key=lambda r: r["n_devices"])
            if not rows:
                print(f"  {label:<11s}{'--':>7s}")
                continue
            sc.annotate_speedups(rows)   # adds 'speedup' relative to the 1-device run
            for r in rows:
                mark = " !" if r.get("throttled") else ""
                print("  {:<11s}{:>7d}{:>12.1f}{:>12.1f}{:>9.2f}x{:<2s}".format(
                    label, r["n_devices"], r["min_ms"], r["mem_mb"],
                    r.get("speedup", float("nan")), mark))
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
            for r in sorted(rows, key=lambda r: r["n_devices"]):
                path = "sharded" if r.get("is_sharded") else "legacy"
                if r.get("back_n_bands_per_shard"):
                    path += f", {r['back_n_bands_per_shard']}b x {r.get('back_band_len')}"
                print(f"  size {label}  n_dev={r['n_devices']}: "
                      f"min={r['min_ms']:.1f} ms  mem={r['mem_mb']:.1f} MB  [{path}]"
                      + ("  [THROTTLED]" if r.get("throttled") else ""))
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
