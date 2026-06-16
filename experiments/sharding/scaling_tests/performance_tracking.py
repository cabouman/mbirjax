"""
experiments/sharding/scaling_tests/performance_tracking.py
──────────────────────────────────────────────────────────
Nightly / manual REGRESSION engine.  Sweeps GEOMETRY × OP × size × device-count over the
existing ``scaling_common`` harness, measuring time + peak memory per cell (a tolerant
correctness fingerprint and the diff/gate come in later phases), and writes ONE dated YAML.

Full design + rationale: ``experiments/sharding/plans/performance_tracking_plan.md``.

This is a THIN driver over ``scaling_common.py`` — do NOT rebuild the measurement machinery.
It reuses the isolated-subprocess discipline, ``time_op`` warmup/trials, ``peak_memory_mb``,
the device-count OOM-descent (``run_measure_loop``), throttle sampling, the path/band YAML
fields, and YAML I/O.  The genuinely new pieces over ``cone_baseline_scaling.py`` are:

  * **GEOMETRY is a sweep dimension** (not a module constant), so a ``Config`` object drives the
    run.  A fresh worker subprocess cannot see the orchestrator's ``Config``, so the orchestrator
    serializes it to a temp YAML and passes ``--config`` to each worker (sweep coordinates go on
    argv for readability).  This is strictly more robust than cone_baseline's module-global read
    and is what lets one engine serve the nightly and the manual launcher.
  * **inline mode** (``Config.inline`` / ``--inline``) runs the worker body IN-PROCESS — no
    subprocess hop, fully step-through-able in PyCharm.  The cost: peak memory is then cumulative
    across the sweep, not per-config (``peak_bytes_in_use`` is a high-water mark), so inline is
    for debugging logic/correctness; trust the isolated-subprocess numbers for the memory ruler.

Roles (mirrors cone_baseline_scaling.py):
  - orchestrator (default, no args)            : ``run(Config)`` — per (geom, op, size) spawn a
                                                  worker (or call inline), collect rows, write YAML.
  - worker --mode setup                        : report platform / devices.
  - worker --mode measure --config --geometry --op --size : measure one cell group (all device
                                                  counts) and write its rows.

mbirjax/jax are imported INSIDE the worker functions only (device-setup-first; the default
orchestrator role stays JAX-free so a subprocess worker can read peak memory cleanly).  In
``--inline`` mode the orchestrator DOES import them (the documented tradeoff above).
"""
import os
import sys
import gc
import argparse
import tempfile
import dataclasses
from dataclasses import dataclass, field
from collections import OrderedDict

import scaling_common as sc

import numpy as np


# ── Run configuration ─────────────────────────────────────────────────────────
# The Config defaults encode the NIGHTLY sweep; the manual launcher and main()
# override a subset.  See the plan for the field-by-field rationale.  A worker
# reconstructs this from the temp YAML the orchestrator writes (from_dict tolerates
# extra/missing keys so the schema can evolve without breaking serialized configs).
@dataclass
class Config:
    # sweep dimensions
    geometries: list = field(default_factory=lambda: ["parallel", "cone"])
    ops: list = field(default_factory=lambda: ["direct_filter", "forward", "back", "vcd_nonconst"])
    device_counts: list = field(default_factory=lambda: [1, 2, 4])
    # SINOGRAM sizes (n_views, n_rows, n_channels) — ASYMMETRIC (all three differ) to surface
    # axis swaps; one DIVIDING + one NON-DIVIDING (all-odd) per platform to exercise padding;
    # plus a GPU 1024-class capacity size.  The recon shape is auto-derived per geometry.
    sizes: dict = field(default_factory=lambda: {
        "cpu": [(128, 112, 96), (129, 113, 97), (200, 208, 160)],
        "gpu": [(512, 448, 384), (513, 449, 385), (1024, 1008, 992)],
    })
    # Sizes where every op runs trials=1 (capacity/memory check, not a timing ruler).
    single_trial_sizes: list = field(default_factory=lambda: ["1024x1008x992"])

    # vcd (not yet wired up)
    vcd_iterations: int = 3
    weight_mode: str = "nonconstant"
    weight_seed: int = 13

    # measurement
    warmup: int = 1
    trials_by_op: dict = field(default_factory=lambda: {
        "direct_filter": 3, "forward": 3, "back": 3, "vcd_nonconst": 1})
    inline: bool = False   # True = single-process, debuggable (memory not per-config)

    # geometry / seeds
    cone_sdd_over_channels: float = 4.0
    input_seed: int = 0
    measure_seed: int = 7

    # io / provenance
    out_dir: str = ""      # stable nightly dir, or results/manual/<tag> (required at run time)
    date: str = ""         # stamped by the orchestrator (never datetime.now() in a worker)
    run_tag: str = ""
    lib_root: str = ""     # library checkout to MEASURE (PYTHONPATH + provenance); "" -> beta_root()
                           # (this harness's own checkout).  The nightly sets it to a per-branch worktree.

    # diff / gate
    gate: bool = True               # set the process exit code on a HARD regression
    compare_to_prior: bool = True   # compare against the most-recent prior dated file in out_dir
    golden_path: str = ""           # explicit golden file (empty -> resolve from golden_dir, if set)
    golden_dir: str = ""            # base dir for golden_<plat>.yaml + main_baseline_<plat>.yaml
                                    # (e.g. the metrics repo's golden/).  "" -> script-relative
                                    # RESULTS_DIR/golden and NO auto golden gate (historical behavior)
    main_baseline_path: str = ""    # "" -> auto-discover <golden_dir>/main_baseline_<plat>.yaml
                                    # for SOFT "vs main (1 device)" relative-perf notes (not gated)
    mem_hard_pct: float = 8.0       # memory growth threshold (%); HARD on GPU, soft on CPU
    speedup_warn_pct: float = 15.0  # speedup-ratio drop WARN threshold (%); soft on all platforms
    time_soft_pct: float = 25.0     # absolute-time WARN threshold (%)
    fp_rtol_single: float = 1e-5    # fingerprint robust-aggregate rel tol (single-shot ops)
    fp_rtol_iter: float = 1e-4      # ... for the iterated vcd
    k_sample_tol: int = 1           # allowed deviating fingerprint samples before a soft flag

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


# ── Op-specific builders (used by the worker body) ────────────────────────────
def make_model(config, geometry, size):
    """Build a single-device model of ``geometry`` for SINOGRAM ``size``.

    ``size`` = (n_views, n_rows, n_channels).  The recon shape is auto-derived by the model
    (for cone it differs from the sinogram shape).  Representative cone geometry: magnification
    2 (source_detector_dist = cone_sdd_over_channels * channels, source_iso_dist = half that),
    matching the test-suite convention.
    """
    import mbirjax
    n_views, n_rows, n_channels = size
    angles = np.linspace(0, np.pi, n_views, endpoint=False)
    if geometry == "parallel":
        model = mbirjax.ParallelBeamModel((n_views, n_rows, n_channels), angles)
    elif geometry == "cone":
        sdd = config.cone_sdd_over_channels * n_channels
        sid = sdd / 2.0
        model = mbirjax.ConeBeamModel((n_views, n_rows, n_channels), angles,
                                      source_detector_dist=sdd, source_iso_dist=sid)
    else:
        raise ValueError(f"unknown geometry {geometry!r} (expected 'parallel' or 'cone')")
    model.set_params(verbose=0)
    return model


def make_indices(model):
    """Full field-of-view pixel indices for the model (deterministic per size)."""
    import mbirjax
    recon_shape = model.get_params('recon_shape')
    return mbirjax.gen_full_indices(recon_shape, use_ror_mask=model.get_params('use_ror_mask'))


def make_cylinders(num_pixels, num_slices, seed):
    """Deterministic random recon cylinders (num_pixels, num_slices) float32."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((num_pixels, num_slices), dtype=np.float32)


def make_sinogram(config, size):
    """Deterministic random sinogram of SINOGRAM ``size`` (numpy float32).

    Projection is linear, so a random sinogram is a valid timing/memory input.
    """
    rng = np.random.default_rng(config.input_seed)
    return rng.random(size, dtype=np.float32)


def to_device(model, arr, kind):
    """Pre-place a HOST input on the model's device form, OUTSIDE the timing loop.

    The timed op must measure compute, not the host->device transfer + scatter a numpy input
    incurs on every call ("measure the op, not the scatter").  ``kind`` is 'sino' (view-sharded)
    or 'recon' (slice-sharded).  Falls back to a single-device device_put on pre-sharding code.
    Blocks so the transfer is complete before timing begins.
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


def run_filter(model, sino):
    """Timed op: the FBP/FDK filter, kept in the device (sharded) form.

    ``output_sharded=True`` so we measure the FILTER, not a full-sinogram gather at exit (the
    user-facing default gathers — a fixed cost that does not shard and flattens the scaling).
    Falls back to the plain call on code predating the kwarg.
    """
    try:
        return model.direct_filter(sino, output_sharded=True)
    except TypeError:
        return model.direct_filter(sino)


def run_forward(model, cylinders, pixel_indices):
    """Timed op: forward projection."""
    return model.sparse_forward_project(cylinders, pixel_indices)


def run_back(model, sino, pixel_indices):
    """Timed op: back projection."""
    return model.sparse_back_project(sino, pixel_indices)


def make_weights(config, size):
    """Deterministic NONCONSTANT weights (positive) for the weighted VCD path.

    All-ones weights skip the weighted gradient/Hessian path; a seeded uniform draw in
    [0.5, 1.5] exercises it while staying positive and reproducible.
    """
    rng = np.random.default_rng(config.weight_seed)
    return rng.uniform(0.5, 1.5, size=size).astype(np.float32)


def build_partitions(model, sino_np, weights, max_iterations, seed=None):
    """Build the VCD pixel partitions + sequence once (device-independent, outside timing).

    initialize_recon constructs the partitions (consuming the global RNG) and the partition
    sequence; we keep only those two.  ``seed`` pins the partition grouping: gen_pixel_partition
    draws from the UN-seeded global RNG, so without this the partitions — and therefore the VCD
    recon — vary run to run (verified ~4e-2), which would make the day-over-day VCD fingerprint
    false-positive.  Seeding here makes VCD reproducible across runs.
    """
    if seed is not None:
        np.random.seed(seed)
    (_sino, _weights, _init, partitions, partition_sequence,
     _granularity, _reg) = model.initialize_recon(
        sino_np, weights=weights, max_iterations=max_iterations, print_logs=False)
    return partitions, partition_sequence


def run_vcd(model, sino_np, weights, partitions, partition_sequence, measure_seed):
    """Timed op: one full VCD reconstruction with NONCONSTANT weights.

    Seeds the global RNG so the subset order is identical on every call (stable timing).
    ``init_recon=None`` lets vcd_recon compute its own direct_recon init (part of the real
    per-recon cost).  Returns only the recon (the fingerprint/correctness target).
    """
    np.random.seed(measure_seed)
    recon, _stats = model.vcd_recon(
        sino_np, partitions, partition_sequence,
        stop_threshold_change_pct=0.0, weights=weights, init_recon=None)
    return recon


# ── Correctness fingerprint ───────────────────────────────────────────────────
def _crop_to_true_shape(arr, true_shape):
    """Crop a possibly-padded device-form output to the TRUE shape and check the padding.

    At a non-dividing count an op may return the padded device form (e.g. 49->50 views,
    41->42 slices).  The fingerprint must be on the TRUE shape so it is comparable across
    device counts / vs golden.  Returns ``(cropped, padding_zero)`` where padding_zero is:
      - None  if arr is not padded (shape already == true_shape),
      - True/False whether the padded OVERHANG is exactly 0 (a constructed-zero invariant; a
        non-zero overhang is a real padding-leak bug, surfaced rather than hidden).
    """
    arr = np.asarray(arr)
    true_shape = tuple(int(s) for s in true_shape)
    if arr.shape == true_shape:
        return arr, None
    padding_zero = True
    for ax, (a, t) in enumerate(zip(arr.shape, true_shape)):
        if a > t:   # overhang along this axis must be exactly zero
            overhang = arr.take(range(t, a), axis=ax)
            if not bool(np.all(overhang == 0.0)):
                padding_zero = False
    cropped = arr[tuple(slice(0, t) for t in true_shape)]
    return cropped, padding_zero


def fingerprint(result, true_shape, k_samples=12):
    """Tolerant correctness fingerprint of an op output, computed on the TRUE shape.

    Reductions {sum, mean, l2norm} are accumulated in float64 so the fingerprint reflects the
    array's value, not float32 accumulation order (which varies with device count).  ``samples``
    are the exact values at K evenly-spaced, deterministic flat indices.  ``shape``/``dtype`` are
    the exact (structural) part of the gate.  See _crop_to_true_shape for the padding handling.
    """
    cropped, padding_zero = _crop_to_true_shape(result, true_shape)
    flat = np.asarray(cropped).ravel()
    n = int(flat.size)
    flat64 = flat.astype(np.float64)
    idx = (np.linspace(0, n - 1, min(k_samples, n)).astype(int) if n else np.array([], int))
    return {
        "sum": float(flat64.sum()),
        "mean": float(flat64.mean()) if n else 0.0,
        "l2norm": float(np.sqrt(np.sum(flat64 * flat64))),
        "min": float(flat.min()) if n else 0.0,
        "max": float(flat.max()) if n else 0.0,
        "samples": [float(flat[i]) for i in idx],
        "shape": list(np.asarray(cropped).shape),
        "dtype": str(np.asarray(result).dtype),
        "padding_zero": padding_zero,
    }


def path_info(model, op, devs, num_pixels, num_slices):
    """Record WHICH code path this measurement used, so the YAML is self-documenting.

    ``is_sharded``: True = placement/banded path (a 1-device sharded run is a trivial mesh,
    NOT the legacy single-device path).  For the back op also record the sharded band length /
    band count (best-effort), which drive back memory and the horizontal-recompute cost.
    """
    info = {"is_sharded": bool(getattr(model, "is_sharded", False)),
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
    """'128x112x96' -> (128, 112, 96)."""
    return tuple(int(x) for x in label.split("x"))


# ── Worker body (callable inline OR in an isolated subprocess) ────────────────
def measure_cell_group(config, geometry, op, size_label, device_counts, out_file):
    """Measure one (geometry, op, size) across ``device_counts`` (descending, OOM-aware).

    Builds the model + op input once for this size on a single-device base model (host arrays
    the per-count models re-place at entry), then for each device count PINS the model to
    exactly that count, pre-places inputs on the device form, and times only the op.  Cone slice
    padding is not yet implemented, so cone at a non-dividing count is NOT skipped — the op runs
    and any failure is captured as a failure cell (see build_and_time).

    ``out_file`` is used by ``run_measure_loop`` for incremental partial publishing; the caller
    (worker entry, or the inline orchestrator) is responsible for it.  Returns the result dict.
    """
    import mbirjax  # noqa: F401  (device-setup side effect; must precede jax init)
    size = parse_size_label(size_label)
    sino_np = make_sinogram(config, size)

    # Build the model + op input once (device-independent host arrays).  Pin the base model to
    # ONE device so derived inputs carry no multi-device placement; build_and_time configures
    # the real count per measurement.
    base_model = make_model(config, geometry, size)
    if hasattr(base_model, "configure_devices"):   # absent on pre-sharding code
        base_model.configure_devices(1)
    recon_shape = tuple(int(x) for x in base_model.get_params('recon_shape'))
    idx = make_indices(base_model)
    num_pixels = len(idx)
    num_slices = recon_shape[2]
    cylinders = (make_cylinders(num_pixels, num_slices, config.input_seed)
                 if op == "forward" else None)
    # VCD inputs (built once, device-independent): nonconstant weights + the pixel partitions.
    weights = partitions = partition_sequence = None
    if op == "vcd_nonconst":
        weights = make_weights(config, size)
        partitions, partition_sequence = build_partitions(
            base_model, sino_np, weights, config.vcd_iterations, seed=config.measure_seed)
    del base_model
    gc.collect()

    # TRUE (unpadded) output shape per op, for the fingerprint crop: filter/forward emit the
    # sinogram shape; back emits (num_pixels, num_slices); vcd emits the recon shape.
    op_true_shape = {
        "direct_filter": tuple(size),
        "forward": tuple(size),
        "back": (num_pixels, num_slices),
        "vcd_nonconst": tuple(recon_shape),
    }.get(op, tuple(size))

    trials = 1 if size_label in config.single_trial_sizes else config.trials_by_op.get(op, 3)
    path_by_n = {}
    fp_by_n = {}
    skips = []

    def build_and_time(n, devs):
        # We do NOT pre-skip any (size, count) we expect to OOM (Greg's call): the run_measure_loop
        # descent already handles it — an OOM at device count n stops the descent, skipping the
        # smaller counts (which need MORE per-device memory and would also OOM).  So an OOM is just
        # recorded as a failure cell; nightly runs let it go until done.
        model = make_model(config, geometry, size)
        # Pin EXACTLY these n devices, so the model runs on the same devices peak_memory_mb(devs)
        # reads.  Without this the model auto-shards across ALL devices at construction.
        if hasattr(model, "configure_devices"):
            model.configure_devices(devs)
        # Cone slice padding is not yet implemented, so a non-dividing count for CONE is NOT
        # skipped — we let the op run so the harness records ground truth (Greg's call: validate
        # the failure-capture path + track the eventual fix as a datable failure->success
        # transition).  Empirically (this session, padded cone): `forward` RAISES (run_measure_loop
        # captures it as a failure and continues the descent, since it is not an OOM), while `back`
        # and `direct_filter` already tolerate padding and return the padded DEVICE form (e.g.
        # 49->50 views, 41->42 slices).  No allowlist: the gate fires only on a CHANGE in cell
        # status vs the prior day / golden, so a persistent known failure stays a visible wart
        # without alarming, and the fix surfaces as a fail->ok improvement (prompting a deliberate
        # golden re-capture).
        path_by_n[n] = path_info(model, op, devs, num_pixels, num_slices)
        # Pre-place big host inputs on this config's device form OUTSIDE the timing loop.
        if op == "direct_filter":
            sino_dev = to_device(model, sino_np, "sino")
            run_fn = lambda: run_filter(model, sino_dev)
        elif op == "forward":
            cyl_dev = to_device(model, cylinders, "recon")
            run_fn = lambda: run_forward(model, cyl_dev, idx)
        elif op == "back":
            sino_dev = to_device(model, sino_np, "sino")
            run_fn = lambda: run_back(model, sino_dev, idx)
        elif op == "vcd_nonconst":
            model.setup_logger(print_logs=False)
            run_fn = lambda: run_vcd(model, sino_np, weights, partitions,
                                     partition_sequence, config.measure_seed)
        else:
            raise ValueError(f"op {op!r} not implemented")
        stats, result = sc.time_op(run_fn, config.warmup, trials)
        mem_mb, mem_kind = sc.peak_memory_mb(devs)
        # Correctness fingerprint AFTER the memory read (the host gather must not inflate the
        # device peak), on the TRUE shape (crop + padding-zero check; see fingerprint()).
        fp_by_n[n] = fingerprint(result, op_true_shape)
        return stats, mem_mb, mem_kind

    rows, failures = sc.run_measure_loop(
        size_label, device_counts, out_file, build_and_time,
        header_extra=f" | {geometry} | op={op} | recon={recon_shape}",
        print_traceback=False)   # expected failures (e.g. cone padding) -> clean one-liner
    # Stamp each row with its sweep coordinates + the auto-derived recon shape + code-path info.
    for r in rows:
        r["geometry"] = geometry
        r["op"] = op
        r["size"] = size_label
        r["recon_shape"] = list(recon_shape)
        r["trials"] = trials
        r.update(path_by_n.get(r["n_devices"], {}))
        fp = fp_by_n.get(r["n_devices"])
        if fp is not None:
            r["fingerprint"] = fp
    return {"geometry": geometry, "op": op, "size": size_label,
            "recon_shape": list(recon_shape), "rows": rows,
            "failures": failures, "skips": skips}


# ── Worker entry (internal; the orchestrator builds argv) ─────────────────────
def worker_setup(out_file):
    """Report platform + device count/label (no cross-version baseline yet)."""
    import mbirjax  # noqa: F401  device-setup-first
    plat, max_dev = sc.detect_platform()
    dev_label = sc.device_label()
    corr = {"check": "no correctness fingerprint yet", "baseline_present": False}
    result = sc.build_setup_result(plat, max_dev, dev_label, corr)
    sc.write_worker_result(out_file, result)


def run_worker(argv):
    """Dispatch a --worker invocation (internal)."""
    p = argparse.ArgumentParser(description="performance_tracking worker (internal)")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--mode", choices=["setup", "measure"], required=True)
    p.add_argument("--config", default=None, help="path to the serialized Config YAML")
    p.add_argument("--geometry", default=None)
    p.add_argument("--op", default=None)
    p.add_argument("--size", default=None, help="LxRxC")
    p.add_argument("--device-counts", type=int, nargs="+", default=None)
    p.add_argument("--out-file", required=True)
    a = p.parse_args(argv)
    if a.mode == "setup":
        worker_setup(a.out_file)
        return
    config = Config.from_dict(sc.load_yaml(a.config))
    res = measure_cell_group(config, a.geometry, a.op, a.size, a.device_counts, a.out_file)
    sc.write_worker_result(a.out_file, res)   # final authoritative result (over run_measure_loop's)


# ── Orchestrator (default; touches no JAX unless inline) ──────────────────────
def _inline_setup(config):
    """Inline-mode platform probe.  The CPU device-count XLA flag is derived from
    MBIRJAX_NUM_CPU_DEVICES on the FIRST ``import mbirjax``, so set it BEFORE that import
    (setdefault respects a value the shell/cluster already set).  Ignored on GPU.
    """
    os.environ.setdefault("MBIRJAX_NUM_CPU_DEVICES", str(max(config.device_counts)))
    import mbirjax  # noqa: F401  device-setup-first
    plat, max_dev = sc.detect_platform()
    print(f"[inline setup] platform={plat}  max_devices={max_dev}  ({sc.device_label()})")
    return plat, max_dev


def _git_provenance(root):
    """{git_commit, git_branch, git_dirty} for the checkout at ``root`` (best-effort)."""
    import subprocess
    def _g(args):
        try:
            r = subprocess.run(["git", "-C", root, *args],
                               capture_output=True, text=True, timeout=5)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None
    return {"git_commit": _g(["rev-parse", "HEAD"]),
            "git_branch": _g(["rev-parse", "--abbrev-ref", "HEAD"]),
            "mbirjax_version": sc.pyproject_version(root),
            "git_dirty": bool(_g(["status", "--porcelain"]))}


# ── Record book (best-ever per cell/metric + the commit that set it) ───────────
# Categories tracked, and whether best is the MIN (time/memory) or MAX (speedup).
RECORD_METRICS = {"min_ms": "min", "mem_mb": "min", "speedup": "max"}


def update_records(records, cells, commit, date):
    """Update the cumulative best-per-(cell, metric) record book IN PLACE and annotate cells.

    ``records`` (loaded from records_<plat>.yaml, or {}) maps "geom|op|size|n_dev" -> per-metric
    {value, commit, date, prev}.  For each MEASURED cell, every RECORD_METRICS metric is compared
    against the stored best (min for time/memory, max for speedup): a first-ever value establishes
    a baseline (silent); a value that BEATS the prior best overwrites it (keeping prev) and is a
    "win" — the cell gains a ``new_records`` list naming the won metrics.  The trivial n=1 speedup
    (always 1.0) is excluded.  Returns ``(new_lines, n_baselines)`` for the run summary.
    """
    new_lines, n_baselines = [], 0
    for c in cells:
        if c.get("failed") or c.get("skipped"):
            continue
        key = f"{c['geometry']}|{c['op']}|{c['size']}|{c['n_devices']}"
        rec = records.setdefault(key, {})
        won = []
        for metric, direction in RECORD_METRICS.items():
            if metric not in c:
                continue
            if metric == "speedup" and c["n_devices"] == 1:
                continue   # trivially 1.0 at one device
            val = float(c[metric])
            cur = rec.get(metric)
            if cur is None:
                rec[metric] = {"value": val, "commit": commit, "date": date, "prev": None}
                n_baselines += 1
            elif (val < cur["value"] if direction == "min" else val > cur["value"]):
                new_lines.append(f"  NEW RECORD  {key}  {metric}={val:.4g} "
                                 f"(prev {cur['value']:.4g} @ {(cur.get('commit') or '?')[:8]})")
                rec[metric] = {"value": val, "commit": commit, "date": date,
                               "prev": cur["value"]}
                won.append(metric)
        if won:
            c["new_records"] = won
    return new_lines, n_baselines


# ── Diff + gate (compare a run vs prior / golden; classify; set exit code) ─────
def _cell_key(c):
    return f"{c['geometry']}|{c['op']}|{c['size']}|{c['n_devices']}"


def _cell_status(c):
    """ok (measured) / failed / skipped / absent (None)."""
    if c is None:
        return "absent"
    if c.get("failed"):
        return "failed"
    if c.get("skipped"):
        return "skipped"
    return "ok"


def _expected_cells(result):
    """The (geom|op|size|n_dev) keys this run's config was supposed to attempt."""
    cfg = result.get("config", {})
    plat = result["platform"]
    sizes = [sc.size_label(s) for s in cfg.get("sizes", {}).get(plat, [])]
    keys = set()
    for g in cfg.get("geometries", []):
        for op in cfg.get("ops", []):
            for s in sizes:
                for n in result.get("device_counts", []):
                    keys.add(f"{g}|{op}|{s}|{n}")
    return keys


def _fmt_delta(today, ref, unit=""):
    """'<today> vs <expected> (<+abs>, <+pct>)' — shows BOTH the absolute and the % difference
    so a reader can judge importance (a big % on a tiny absolute is often noise, and vice versa)."""
    d = today - ref
    pct = (d / ref * 100.0) if ref else float("nan")
    return f"{today:g}{unit} vs {ref:g}{unit} expected ({d:+g}{unit}, {pct:+.1f}%)"


def _gate_fingerprint(key, tf, rf, op, lab, config, hard, soft):
    """Correctness gate on the tolerant fingerprint (see §7): exact shape/dtype, robust
    aggregates within rtol (HARD), a few sample deviations allowed (SOFT), new padding leak (HARD).
    Each aggregate finding shows the relative diff vs the tolerance plus the absolute change."""
    if not tf or not rf:
        return
    if tf.get("shape") != rf.get("shape"):
        hard.append(f"[{lab}] {key} fingerprint shape {rf.get('shape')} -> {tf.get('shape')}")
        return
    if tf.get("dtype") != rf.get("dtype"):
        hard.append(f"[{lab}] {key} fingerprint dtype {rf.get('dtype')} -> {tf.get('dtype')}")
    rtol = config.fp_rtol_iter if op == "vcd_nonconst" else config.fp_rtol_single
    for m in ("sum", "mean", "l2norm"):
        rv, tv = rf.get(m), tf.get(m)
        if rv is None or tv is None:
            continue
        reldiff = abs(tv - rv) / (abs(rv) or 1.0)
        if reldiff > rtol:
            hard.append(f"[{lab}] {key} fingerprint {m}: reldiff {reldiff:.2e} > rtol {rtol:g} "
                        f"(Δ {tv - rv:+.3g}; {tv:g} vs {rv:g} expected)")
    rs_, ts_ = rf.get("samples") or [], tf.get("samples") or []
    if rs_ and len(rs_) == len(ts_):
        dev = sum(1 for a, b in zip(rs_, ts_) if abs(b - a) / (abs(a) or 1.0) > rtol)
        if dev > config.k_sample_tol:
            soft.append(f"[{lab}] {key} {dev}/{len(rs_)} fingerprint samples deviate (rtol {rtol:g})")
    if tf.get("padding_zero") is False and rf.get("padding_zero") is not False:
        hard.append(f"[{lab}] {key} padding leak: padding_zero {rf.get('padding_zero')} -> False")


def _gate_metrics(key, t, r, lab, plat, config, hard, soft):
    """Metric gates for an ok->ok cell.  Structural changes and the correctness fingerprint are
    HARD on every platform.  Of the PERFORMANCE signals, only MEMORY is HARD, and only on GPU,
    where peak_bytes_in_use is ~deterministic (it is also what catches the gather-bug class —
    memory that fails to shard); on CPU memory is whole-process RSS (coarse) so it is SOFT.
    Speedup and absolute time are SOFT on every platform — both derive from timings, which are
    noisy even on GPU (especially small runs).  Every delta shows the value vs expected with BOTH
    the absolute and the percentage difference."""
    pre = f"[{lab}] {key} "
    # memory — HARD on GPU, SOFT (coarse RSS) on CPU.
    rm, tm = r.get("mem_mb"), t.get("mem_mb")
    if rm and tm is not None and (tm - rm) / rm * 100.0 > config.mem_hard_pct:
        bucket = hard if plat == "gpu" else soft
        cpu_note = "" if plat == "gpu" else " [CPU RSS, coarse]"
        bucket.append(pre + "memory " + _fmt_delta(tm, rm, " MB") + cpu_note)
    # speedup-ratio drop — SOFT everywhere (ratio of noisy timings).
    rsp, tsp = r.get("speedup"), t.get("speedup")
    if t["n_devices"] > 1 and rsp and tsp is not None and (rsp - tsp) / rsp * 100.0 > config.speedup_warn_pct:
        soft.append(pre + "speedup " + _fmt_delta(tsp, rsp))
    # absolute time — SOFT everywhere.
    rt, tt = r.get("min_ms"), t.get("min_ms")
    if rt and tt is not None and (tt - rt) / rt * 100.0 > config.time_soft_pct:
        soft.append(pre + "time " + _fmt_delta(tt, rt, " ms"))
    # structural — HARD everywhere.
    if bool(t.get("is_sharded")) != bool(r.get("is_sharded")):
        hard.append(pre + f"is_sharded {r.get('is_sharded')} -> {t.get('is_sharded')}")
    tb, rb = t.get("back_n_bands_per_shard"), r.get("back_n_bands_per_shard")
    if (tb is not None or rb is not None) and tb != rb:
        hard.append(pre + f"back band count {rb} -> {tb}")
    _gate_fingerprint(key, t.get("fingerprint"), r.get("fingerprint"), t.get("op", ""),
                      lab, config, hard, soft)


def _compare_cell(key, t, r, lab, plat, expected, oom_gos, config, hard, soft):
    """Classify one cell vs one reference (see plan §10a status transitions)."""
    ts, rs = _cell_status(t), _cell_status(r)
    if rs == "absent":
        soft.append(f"[{lab}] new cell, no baseline (not gated): {key}")
        return
    if ts == "absent":
        gos = tuple(key.split("|")[:3])
        if key not in expected:
            soft.append(f"[{lab}] dropped from sweep: {key}")
        elif gos in oom_gos:
            soft.append(f"[{lab}] {key} not measured (OOM-descent stopped at higher n_dev)")
        else:
            hard.append(f"[{lab}] expected cell vanished (no row/skip/fail): {key}")
        return
    if ts == "failed" and rs == "ok":
        hard.append(f"[{lab}] {key} REGRESSED: was ok, now fails ({str(t.get('error',''))[:50]})")
        return
    if ts == "ok" and rs == "failed":
        soft.append(f"[{lab}] {key} improved: was failing, now ok (consider re-baselining golden)")
        return
    if ts != "ok" or rs != "ok":   # skip<->fail combos, or unchanged fail/skip (quiet)
        if ts != rs:
            soft.append(f"[{lab}] {key} status {rs} -> {ts}")
        return
    _gate_metrics(key, t, r, lab, plat, config, hard, soft)   # ok -> ok


def gate_run(result, references, config):
    """Compare ``result`` against each (label, ref_result) and return the gate dict.

    Fires on a CHANGE vs the reference (plan §10/§10a): ok->fail / memory / speedup / structural /
    correctness are HARD; absolute time / added-dropped / improvements are SOFT; persistent
    failures are quiet.  Cold start (no usable reference) is all-SOFT, never a fail.
    """
    hard, soft = [], []
    refs = [(lab, r) for lab, r in references if r]
    if not refs:
        return {"result": "warn", "hard": [], "compared_to": [],
                "soft": ["no prior or golden to compare against (cold start) — nothing gated"]}
    plat = result.get("platform", "")
    expected = _expected_cells(result)
    oom_gos = {(c["geometry"], c["op"], c["size"])
               for c in result["cells"] if c.get("failed") and c.get("oom")}
    today = {_cell_key(c): c for c in result["cells"]}
    for lab, ref in refs:
        refcells = {_cell_key(c): c for c in ref.get("cells", [])}
        for key in sorted(set(today) | set(refcells)):
            _compare_cell(key, today.get(key), refcells.get(key), lab, plat,
                          expected, oom_gos, config, hard, soft)
    return {"result": "fail" if hard else ("warn" if soft else "pass"),
            "hard": hard, "soft": soft, "compared_to": [lab for lab, _ in refs]}


def _find_prior(out_dir, plat, today_date):
    """Most-recent prior dated file in out_dir (excluding today's), or None.

    Filenames embed the date/timestamp, so a lexicographic sort is chronological.
    """
    import glob
    today_file = os.path.join(out_dir, f"regression_{plat}_{today_date}.yaml")
    files = sorted(f for f in glob.glob(os.path.join(out_dir, f"regression_{plat}_*.yaml"))
                   if f != today_file)
    return files[-1] if files else None


def _print_gate(g):
    print("\n" + "=" * 78)
    print(f"  GATE: {g['result'].upper()}   (vs {', '.join(g['compared_to']) or 'nothing'})")
    print("=" * 78)
    for h in g.get("hard", []):
        print("  HARD  " + h)
    for s in g.get("soft", []):
        print("  warn  " + s)
    if not g.get("hard") and not g.get("soft"):
        print("  no changes vs reference")


def main_perf_notes(result, main_baseline):
    """SOFT, never-gated 'vs main (1 device)' relative-performance notes.

    Compares this run's n=1 cells against the main-branch 1-device baseline (captured by
    capture_main_baseline.py) — the pre-sharding reference — so we can see whether the sharding
    machinery added overhead at a single device.  Each delta shows the value vs main with both
    absolute and percentage difference.  Returns a list of note strings (informational only).
    """
    mb = {_cell_key(c): c for c in main_baseline.get("cells", [])
          if c.get("n_devices") == 1 and not c.get("failed") and not c.get("skipped")}
    notes = []
    for c in result.get("cells", []):
        if c.get("n_devices") != 1 or c.get("failed") or c.get("skipped"):
            continue
        m = mb.get(_cell_key(c))
        if not m:
            continue
        parts = []
        if c.get("min_ms") and m.get("min_ms"):
            parts.append("time " + _fmt_delta(c["min_ms"], m["min_ms"], " ms"))
        if c.get("mem_mb") and m.get("mem_mb"):
            parts.append("mem " + _fmt_delta(c["mem_mb"], m["mem_mb"], " MB"))
        if parts:
            notes.append(f"{c['geometry']}|{c['op']}|{c['size']}  " + "  ".join(parts))
    return notes


def _print_summary(cells):
    """Per (geometry, op): min time (ms) / peak mem (MB) / speedup, for each (size, n_dev)."""
    print("\n" + "=" * 78)
    print("  REGRESSION SUMMARY — min time (ms) / peak mem (MB) / speedup vs fewest devices")
    print("=" * 78)
    groups = OrderedDict()
    for c in cells:
        groups.setdefault((c["geometry"], c["op"]), []).append(c)
    for (g, op), rows in groups.items():
        print(f"\n  {g} | {op}")
        print("  {:<12s}{:>6s}{:>11s}{:>11s}{:>9s}".format(
            "size", "n_dev", "min_ms", "mem_mb", "speedup"))
        for r in sorted(rows, key=lambda r: (r["size"], r["n_devices"])):
            if r.get("skipped"):
                print(f"  {r['size']:<12s}{r['n_devices']:>6d}   [skip] {r['reason']}")
                continue
            if r.get("failed"):
                tag = "OOM" if r.get("oom") else "FAIL"
                print(f"  {r['size']:<12s}{r['n_devices']:>6d}   [{tag}] {str(r.get('error', ''))[:58]}")
                continue
            mark = " !" if r.get("throttled") else ""
            print("  {:<12s}{:>6d}{:>11.1f}{:>11.1f}{:>8.2f}x{}".format(
                r["size"], r["n_devices"], r["min_ms"], r["mem_mb"],
                r.get("speedup", float("nan")), mark))


def run(config):
    """Run the full GEOMETRY × OP × size × device-count sweep and write the dated YAML."""
    if not config.out_dir:
        raise ValueError("Config.out_dir is required")
    if not config.date:
        raise ValueError("Config.date is required (stamp it in the orchestrator)")
    os.makedirs(config.out_dir, exist_ok=True)
    script = os.path.abspath(__file__)

    print("=" * 72)
    print(f"  performance_tracking — {'INLINE (single process)' if config.inline else 'isolated-subprocess'} harness")
    print(f"  beta root: {sc.beta_root()}")
    print("=" * 72)

    worker_env = None
    cfg_path = None
    if config.inline:
        plat, max_dev = _inline_setup(config)
        dev_label = sc.device_label()
    else:
        worker_env = sc.build_worker_env(lib_root=config.lib_root or None)
        # Bound the CPU virtual-device count by THIS sweep (config.device_counts), not by
        # mbirjax's DEFAULT_MAX_CPU_DEVICES.  This MUST be set before the setup probe: the probe
        # imports mbirjax, and with no override mbirjax resolves only DEFAULT_MAX_CPU_DEVICES
        # devices -> detect_platform reports that as max_dev -> the sweep is silently capped (e.g.
        # 4 dropped when the library default is 2).  Harmless on GPU (CPU-backend flag only).
        worker_env["MBIRJAX_NUM_CPU_DEVICES"] = str(max(config.device_counts))
        setup, rc = sc.run_worker(script, ["--worker", "--mode", "setup"], extra_env=worker_env)
        if setup is None:
            print(f"  ERROR: setup worker produced no result (rc={rc}); aborting.")
            return None
        plat, max_dev, dev_label, _corr, _mpath = sc.print_setup_banner(setup)

    sizes = config.sizes[plat]
    size_labels = [sc.size_label(s) for s in sizes]
    device_counts = [n for n in config.device_counts if n <= max_dev]
    print(f"  geometries: {config.geometries}   ops: {config.ops}")
    print(f"  sizes: {size_labels}   device counts: {device_counts}")

    if not config.inline:
        fd, cfg_path = tempfile.mkstemp(suffix=".yaml", prefix="perf_cfg_")
        os.close(fd)
        sc.save_yaml(cfg_path, config.to_dict())

    cells = []
    for geometry in config.geometries:
        for op in config.ops:
            for label in size_labels:
                print(f"\n=== {geometry} | {op} | {label} ===")
                if config.inline:
                    fd, tmp = tempfile.mkstemp(suffix=".yaml", prefix="perf_inline_")
                    os.close(fd)
                    try:
                        res = measure_cell_group(config, geometry, op, label,
                                                 device_counts, tmp)
                    finally:
                        if os.path.exists(tmp):
                            os.remove(tmp)
                else:
                    args = ["--worker", "--mode", "measure", "--config", cfg_path,
                            "--geometry", geometry, "--op", op, "--size", label,
                            "--device-counts", *[str(n) for n in device_counts]]
                    res, _rc = sc.run_worker(script, args, extra_env=worker_env)
                if not res:
                    print(f"  (no result for {geometry}/{op}/{label})")
                    continue
                rows = res.get("rows") or []
                sc.annotate_speedups(rows)   # 'speedup' vs the fewest-device run in this group
                cells.extend(rows)
                for f in (res.get("failures") or []):
                    cells.append({"geometry": geometry, "op": op, "size": label,
                                  "n_devices": f["n_devices"], "failed": True,
                                  "oom": bool(f.get("oom")), "error": f.get("error")})
                for s in (res.get("skips") or []):
                    cells.append({"geometry": geometry, "op": op, "size": label,
                                  "n_devices": s["n_devices"], "skipped": True,
                                  "reason": s["reason"]})

    if cfg_path and os.path.exists(cfg_path):
        os.remove(cfg_path)

    prov = _git_provenance(config.lib_root or sc.beta_root())   # provenance of the LIBRARY under test

    # Update the cumulative record book (best per cell/metric + the commit that set it).  It lives
    # in out_dir, so nightly (results/regression/) and manual (results/manual/<tag>/) runs keep
    # SEPARATE books.  Done before writing the dated YAML so its cells carry the per-cell
    # `new_records` annotation; the records file itself, once versioned in mbirjax_metrics, has a
    # git history that IS the record-progression log.
    records_path = os.path.join(config.out_dir, f"records_{plat}.yaml")
    records = (sc.load_yaml(records_path) or {}) if os.path.exists(records_path) else {}
    new_lines, n_baselines = update_records(records, cells, prov.get("git_commit") or "?",
                                            config.date)
    sc.save_yaml(records_path, records)

    result = {
        "kind": "regression", "date": config.date, "platform": plat,
        "device_label": dev_label, **prov,
        "config": config.to_dict(), "device_counts": device_counts, "cells": cells,
    }

    # Diff + gate: compare against the most-recent prior dated file (and golden, if configured),
    # classify per the §10/§10a rules, and stash the gate dict in the YAML.  Done before the write
    # so the dated file records its own verdict; the exit code is set by main() from result.gate.
    # golden_dir (e.g. the metrics repo's golden/) overrides the script-relative default for BOTH the
    # golden gate and the vs-main baseline; plat is resolved here so the files carry the platform
    # suffix.  Empty -> historical behavior (RESULTS_DIR/golden; golden gate only if golden_path set).
    golden_base = config.golden_dir or os.path.join(sc.RESULTS_DIR, "golden")
    golden_path = config.golden_path or (
        os.path.join(golden_base, f"golden_{plat}.yaml") if config.golden_dir else "")

    gate_dict = None
    if config.compare_to_prior or golden_path:
        refs = []
        if config.compare_to_prior:
            pp = _find_prior(config.out_dir, plat, config.date)
            if pp:
                refs.append((f"prior:{os.path.basename(pp)}", sc.load_yaml(pp)))
        if golden_path and os.path.exists(golden_path):
            refs.append(("golden", sc.load_yaml(golden_path)))
        gate_dict = gate_run(result, refs, config)
        result["gate"] = gate_dict

    # Relative-performance note vs the main-branch 1-device baseline (auto-discovered under
    # golden_base).  SOFT and never gated — shows whether the sharding machinery added 1-device
    # overhead vs released main.  Only n=1 cells are compared (main has no sharding).
    mb_path = (config.main_baseline_path
               or os.path.join(golden_base, f"main_baseline_{plat}.yaml"))
    vs_main = []
    if os.path.exists(mb_path):
        vs_main = main_perf_notes(result, sc.load_yaml(mb_path) or {})
        result["vs_main"] = {"baseline": os.path.basename(mb_path), "notes": vs_main}

    out_path = os.path.join(config.out_dir, f"regression_{plat}_{config.date}.yaml")
    sc.save_yaml(out_path, result)
    _print_summary(cells)
    if new_lines:
        print(f"\n  {len(new_lines)} NEW RECORD(S) this run:")
        for line in new_lines:
            print(line)
    elif n_baselines:
        print(f"\n  established {n_baselines} baseline record(s) (first run for these cells)")
    if gate_dict:
        _print_gate(gate_dict)
    if vs_main:
        print("\n  Relative to main (1 device) — informational:")
        for n in vs_main:
            print("    " + n)
    print(f"\nOutput written to: {out_path}")
    print(f"Record book:       {records_path}")
    print("Done.")
    return result


# ── Golden capture (the reference the gate compares against) ───────────────────
def _merge_golden(existing, fresh, only):
    """Selective refresh: replace the captured (geometry, op) cells, keep the rest of the golden.

    Existing metadata is preserved; a ``refresh_log`` entry records what was recaptured, when, and
    at which commit — the deliberate-baseline-change audit trail."""
    fresh_gos = {(c["geometry"], c["op"]) for c in fresh.get("cells", [])}
    kept = [c for c in (existing.get("cells") or []) if (c["geometry"], c["op"]) not in fresh_gos]
    merged = dict(existing)
    merged["kind"] = "golden"
    merged["cells"] = kept + fresh.get("cells", [])
    merged.setdefault("refresh_log", []).append(
        {"date": fresh.get("date"), "commit": fresh.get("git_commit"), "only": list(only)})
    return merged


def capture_golden(config, golden_dir, only=None):
    """Capture (or selectively refresh) the golden reference, written to golden_<plat>.yaml.

    Runs the sweep with the gate + prior-comparison OFF (the golden IS the reference, not a tracked
    run).  ``only`` (a list of geometry and/or op names) limits the capture to a subset and MERGES
    it into the existing golden — additive refresh for a newly-ported geometry or an intentional
    baseline change, leaving the other cells untouched; None -> full capture (overwrite).  The
    capture run's dated/records files go to a throwaway temp dir.  (The representative .npy
    deep-diff array + the push to mbirjax_metrics are the operational layer, deferred.)
    """
    import tempfile
    import shutil
    cap = Config.from_dict(config.to_dict())
    cap.gate = False
    cap.compare_to_prior = False
    cap.golden_path = ""
    cap.date = "golden"
    cap.out_dir = tempfile.mkdtemp(prefix="golden_capture_")
    if only:
        geoms = [g for g in cap.geometries if g in only]
        ops = [o for o in cap.ops if o in only]
        if geoms:
            cap.geometries = geoms
        if ops:
            cap.ops = ops
    try:
        result = run(cap)
    finally:
        shutil.rmtree(cap.out_dir, ignore_errors=True)
    if result is None:
        return None
    result["kind"] = "golden"
    os.makedirs(golden_dir, exist_ok=True)
    golden_path = os.path.join(golden_dir, f"golden_{result['platform']}.yaml")
    if only and os.path.exists(golden_path):
        sc.save_yaml(golden_path, _merge_golden(sc.load_yaml(golden_path) or {}, result, only))
        print(f"\nGolden REFRESHED ({','.join(only)}): {golden_path}")
    else:
        sc.save_yaml(golden_path, result)
        print(f"\nGolden captured: {golden_path}")
    return golden_path


def main():
    """Default entry: the nightly config, dated today, into results/regression/.

    Exit code: non-zero on a HARD-gate regression when gating is on, so a cron/slurm wrapper
    surfaces it as a real alert.
    """
    from datetime import datetime
    config = Config()
    config.out_dir = os.path.join(sc.RESULTS_DIR, "regression")
    config.date = datetime.now().strftime("%Y%m%d")
    result = run(config)
    if config.gate and result and (result.get("gate") or {}).get("result") == "fail":
        sys.exit(1)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        run_worker(sys.argv[1:])
    else:
        main()
