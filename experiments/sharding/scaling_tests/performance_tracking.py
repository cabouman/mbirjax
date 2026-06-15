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
    # The three non-VCD ops are implemented; vcd_nonconst is not yet wired up.
    ops: list = field(default_factory=lambda: ["direct_filter", "forward", "back"])
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
    # Smallest device count to ATTEMPT for VCD at a size known to OOM below it (let it OOM once).
    vcd_min_devices: dict = field(default_factory=lambda: {"1024x1008x992": 2})

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
    del base_model
    gc.collect()

    trials = 1 if size_label in config.single_trial_sizes else config.trials_by_op.get(op, 3)
    path_by_n = {}

    def build_and_time(n, devs):
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
        # golden re-capture).  The vcd_min_devices OOM-floor remains a deliberate skip (avoid an
        # hour of OOM thrash) — a different case.
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
        else:
            raise ValueError(f"op {op!r} not implemented here (vcd_nonconst is not yet wired up)")
        stats, _ = sc.time_op(run_fn, config.warmup, trials)
        mem_mb, mem_kind = sc.peak_memory_mb(devs)
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
    return {"geometry": geometry, "op": op, "size": size_label,
            "recon_shape": list(recon_shape), "rows": rows, "failures": failures}


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
            "git_dirty": bool(_g(["status", "--porcelain"]))}


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
        worker_env = sc.build_worker_env()
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

    if cfg_path and os.path.exists(cfg_path):
        os.remove(cfg_path)

    result = {
        "kind": "regression", "date": config.date, "platform": plat,
        "device_label": dev_label, **_git_provenance(sc.beta_root()),
        "config": config.to_dict(), "cells": cells,
    }
    out_path = os.path.join(config.out_dir, f"regression_{plat}_{config.date}.yaml")
    sc.save_yaml(out_path, result)
    _print_summary(cells)
    print(f"\nOutput written to: {out_path}")
    print("Done.")
    return result


def main():
    """Default entry: the nightly config, dated today, into results/regression/."""
    from datetime import datetime
    config = Config()
    config.out_dir = os.path.join(sc.RESULTS_DIR, "regression")
    config.date = datetime.now().strftime("%Y%m%d")
    run(config)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        run_worker(sys.argv[1:])
    else:
        main()
