"""
experiments/sharding/scaling_tests/sparse_back_project_band_sweep.py
─────────────────────────────────────────────────────────────────────────────
MULTI-device slice-band sweep for sharded back projection: at a FIXED device
count, does a LARGER band (up to the full per-device slice-shard, i.e. one band /
no streaming) change wall-time and peak memory versus the default band?

Why this exists
───────────────
The default band length is ``reduce_band = slices_per_dev // n_dev ≈
num_slices/n_dev^2`` (TomographyModel._slice_band_length).  The second 1/n_dev
pins the cross-device reduce gather transient (``n_dev × num_pixels × band``) to
~one output slice-shard, so per-device peak tracks 1/n_dev.  But when memory is
NOT the binding constraint (the recon already fits comfortably across the
devices), that default OVER-shrinks the band: a problem that fits on 2 GPUs does
not need band/2 again at 4 GPUs.  A bigger band means fewer, larger projector
calls -- and fewer per-band reduce-scatter round trips, which on a multi-socket
box cross the NUMA boundary and may NOT be hidden by launch throughput (the
suspected mechanism behind the multi-device time wall).

The model docstring asserts "time is essentially flat across B on GPUs" from an
earlier sweep; this driver tests that claim in the regime that matters -- small
bands at higher n_dev, where a per-band cross-NUMA sync is most likely to bite.

What it measures
────────────────
For each size and each band MODE it times the internal ``sparse_back_project`` on
a PRE-SHARDED sinogram (the reduce-scatter + compute, no entry scatter, no exit
gather -- exactly like sparse_back_project_scaling.py) and records peak
``peak_bytes_in_use``, the effective band, and the number of bands.  Band modes:

  * ``auto``  : the model default (back_project_slice_band unset).
  * ``div{d}``: back_project_slice_band = slices_per_dev // d, for d in
                SHARD_DIVISORS.  d=1 is the FULL per-device shard (one band, no
                streaming -- the "don't shrink it" extreme); larger d streams more.

At n_dev=N the default ``auto`` is ≈ ``div{N}``, so the ``div`` ladder brackets
the current default and the full band -- the head-to-head the band-sizing
decision rests on.  The result is band-INVARIANT (banding only tiles the slice
axis with no overlap; covered by the band unit tests), so this sweep is about
time/memory, not correctness; a single-device sanity check vs the prerelease
baseline still runs in setup.

Isolated-subprocess harness (orchestrator touches no JAX; one fresh worker per
(mode, size) so peak memory reads cleanly), device counts DESCENDING.  Run from
the BETA worktree root on the GPU box:

    python experiments/sharding/scaling_tests/sparse_back_project_band_sweep.py
"""
import os
import sys
import argparse

import scaling_common as sc

import numpy as np

# mbirjax (and therefore jax) is imported INSIDE the worker functions, never at
# the top level: this file runs as a JAX-free orchestrator (spawns workers) and
# as worker subprocesses (touch JAX).  See sparse_back_project_scaling.py for the
# full rationale (orchestrator must hold no GPU backend while a worker measures
# peak memory; the worker's `import mbirjax` also runs device-setup before jax).

OP_NAME = "sparse_back_project_band_sweep"
BASELINE_OP = "sparse_back_project"   # reuse the existing baseline for the setup check

# ── Run configuration (edit here; no CLI args for the human) ──────────────────
# Device-count ladder.  None -> automatic powers-of-two ladder.  Set explicitly
# to skip a known-bad/throttling card (pick_devices takes the first n).  Each
# size must be divisible by every count used (else configure_sharding raises and
# the point is skipped).  The band question is sharpest at n_dev >= 2 (n_dev=1 is
# covered by sparse_back_project_single_device_sweep.py); 1 is kept as the anchor.
DEVICE_COUNTS = [1, 2, 4]   # e.g. [1, 2, 4, 8] on an 8-GPU node

# Problem sizes (n_views, n_rows, n_channels).  Pick sizes that FIT COMFORTABLY
# across the device ladder (the regime where the band question lives -- if the
# recon barely fits, the default small band is forced and there is nothing to
# sweep).  Divisible by 1/2/4 (and 8) so the ladder applies.
SIZES_BY_16 = {
    "cpu": [(128, 128, 128), (256, 256, 256)],
    "gpu": [(512, 512, 512), (1024, 1024, 1024)],
}
SIZES = SIZES_BY_16

# Band = slices_per_dev // d for each d (1 = full shard / no streaming).  The
# 'auto' default (≈ slices_per_dev // n_dev) is always measured alongside.
SHARD_DIVISORS = [1, 2, 4]

WARMUP = 1
TRIALS = 3
CORRECTNESS_THRESHOLD = 1e-4
CORRECTNESS_SIZE = (80, 48, 64)   # small, NON-symmetric so axis bugs surface
CORRECTNESS_SEED = 1234
_F4 = 4   # bytes per float32


# ── Op-specific builders (used by the worker) ─────────────────────────────────
def make_model(size, devices=None):
    """Build a ParallelBeamModel; configure sharding over ``devices`` if given.

    Returns None if the device count does not evenly divide the sharded axes
    (configure_sharding raises) -- the caller skips that point.
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
    return model


def make_input(size, seed=0):
    """Deterministic random sinogram (numpy float32); back projection is linear."""
    return np.random.default_rng(seed).random(size, dtype=np.float32)


def make_indices(model):
    import mbirjax
    recon_shape = model.get_params('recon_shape')
    return mbirjax.gen_full_indices(recon_shape, use_ror_mask=model.get_params('use_ror_mask'))


def _band_for_mode(model, n_dev, num_slices, num_pixels, divisor):
    """Set back_project_slice_band per the mode and return (eff_band, num_bands).

    ``divisor`` is None for 'auto' (leave the band unset -> model default), else
    band = slices_per_dev // divisor (capped at slices_per_dev by the model).
    """
    slices_per_dev = max(1, num_slices // n_dev)
    if divisor is None:
        if hasattr(model, 'back_project_slice_band'):
            del model.back_project_slice_band
    else:
        model.back_project_slice_band = max(1, slices_per_dev // divisor)
    fixed = getattr(model, 'back_project_slice_band', None)
    eff_band = int(model._slice_band_length(slices_per_dev, n_dev, num_pixels, fixed_band=fixed))
    num_bands = len(model._balanced_slice_bounds(slices_per_dev, eff_band))
    return eff_band, num_bands, slices_per_dev


# ── Worker side (isolated subprocess) ─────────────────────────────────────────
def worker_setup(out_file):
    """Platform / device label + single-device correctness vs the prerelease
    sparse_back_project baseline (the result is band-invariant, so this only
    confirms the build is sane)."""
    import mbirjax  # device-setup-first
    plat, max_dev = sc.detect_platform()
    dev_label = sc.device_label()

    ref, meta = sc.load_baseline(BASELINE_OP)
    model = make_model(CORRECTNESS_SIZE, devices=None)
    sino = make_input(CORRECTNESS_SIZE, seed=CORRECTNESS_SEED)
    idx = make_indices(model)
    beta_out = np.asarray(model.sparse_back_project(sino, idx))
    if ref is None:
        corr = {"baseline_present": False}
        print(f"[setup] no {BASELINE_OP} baseline; correctness skipped")
    else:
        captured_on = meta.get("captured_on_platform", "unknown")
        m = sc.correctness_metrics(ref, beta_out, threshold=CORRECTNESS_THRESHOLD)
        corr = {"baseline_present": True, "baseline_platform": captured_on,
                "current_platform": plat, "cross_platform": captured_on != plat, **m}
        print(f"[setup] correctness vs {captured_on} baseline: "
              f"max_abs_diff={m['max_abs_diff']:.3e}  pct_above={m['pct_above_threshold']:.6f}%"
              + ("   <-- CROSS-PLATFORM" if captured_on != plat else ""))

    sc.write_worker_result(out_file, sc.build_setup_result(plat, max_dev, dev_label, corr))


def worker_measure(size_label, device_counts, warmup, trials, out_file, divisor=None):
    """Time + measure memory for one size and band mode, device counts DESCENDING.

    ``divisor`` selects the band (None = 'auto' default; else slices_per_dev //
    divisor).  Band metadata (effective band, #bands, slices_per_dev) is folded
    into the per-row stats so it lands in the YAML next to time/memory.
    """
    import mbirjax  # noqa: F401  (device-setup side effect; must precede jax init)
    size = tuple(int(x) for x in size_label.split("x"))
    sino_np = make_input(size, seed=0)

    def build_and_time(n, devs):
        model = make_model(size, devices=devs)
        if model is None:
            return None
        recon_shape = model.get_params('recon_shape')
        num_slices = int(recon_shape[2])
        idx = make_indices(model)
        num_pixels = int(len(idx))
        eff_band, num_bands, slices_per_dev = _band_for_mode(
            model, n, num_slices, num_pixels, divisor)
        # Pre-shard the sinogram OUTSIDE the timing loop (measure reduce-scatter +
        # compute, not the entry scatter), like sparse_back_project_scaling.py.
        sino = model._shard_sinogram(sino_np)
        stats, _ = sc.time_op(lambda: model.sparse_back_project(sino, idx), warmup, trials)
        stats = {**stats, "slice_band_eff": eff_band, "num_bands": num_bands,
                 "slices_per_dev": slices_per_dev, "num_pixels": num_pixels,
                 "num_slices": num_slices}
        mem_mb, mem_kind = sc.peak_memory_mb(devs)
        return stats, mem_mb, mem_kind

    mode = "auto" if divisor is None else f"div{divisor}"
    sc.run_measure_loop(size_label, device_counts, out_file, build_and_time,
                        header_extra=f" | band={mode}")


def run_worker(argv):
    p = argparse.ArgumentParser(description="back-projection band sweep worker (internal)")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--mode", choices=["setup", "measure"], required=True)
    p.add_argument("--size", default=None)
    p.add_argument("--device-counts", type=int, nargs="+", default=None)
    p.add_argument("--warmup", type=int, default=WARMUP)
    p.add_argument("--trials", type=int, default=TRIALS)
    p.add_argument("--divisor", type=int, default=None,
                   help="band = slices_per_dev // divisor; omit for the 'auto' default")
    p.add_argument("--out-file", required=True)
    a = p.parse_args(argv)
    if a.mode == "setup":
        worker_setup(a.out_file)
    else:
        worker_measure(a.size, a.device_counts, a.warmup, a.trials, a.out_file,
                       divisor=a.divisor)


# ── Orchestrator (default; touches no JAX) ────────────────────────────────────
def _print_band_comparison(grids_by_mode, size_labels, device_counts):
    """Side-by-side time/memory per (size, n_dev) for each band mode, ratios vs 'auto'.

    This is the headline readout: at each (size, n_dev) it shows every band mode's
    min time / peak memory / #bands, and -- vs the 'auto' default -- the time and
    memory ratios (t<1 = faster than the default; m>1 = more memory).  A mode that
    is t<1 with acceptable m is evidence that the default over-shrinks the band.
    """
    modes = list(grids_by_mode.keys())
    print("\n" + "=" * 78)
    print("  band comparison — min time (ms) / peak mem (MB) / #bands per (size, n_dev)")
    print("  (ratios are vs 'auto'; [THROTTLED] = a GPU throttled, timing unreliable)")
    print("=" * 78)
    for label in size_labels:
        print(f"\n  {label}")
        byn = {m: {r["n_devices"]: r for r in grids_by_mode[m].get(label, [])}
               for m in modes}
        for n in device_counts:
            print(f"    n={n}")
            base = byn.get("auto", {}).get(n)
            for m in modes:
                r = byn[m].get(n)
                if not r:
                    print(f"        {m:<8s} --")
                    continue
                line = (f"        {m:<8s} {r['min_ms']:9.2f} ms  {r['mem_mb']:9.1f} MB  "
                        f"B={r.get('slice_band_eff','?'):>5} #{r.get('num_bands','?'):<3}")
                if base and m != "auto":
                    line += (f"   vs auto: t={r['min_ms']/base['min_ms']:.2f} "
                             f"m={r['mem_mb']/base['mem_mb']:.2f}")
                if r.get("throttled"):
                    line += "  [THROTTLED]"
                print(line)


def main():
    script = os.path.abspath(__file__)
    worker_env = sc.build_worker_env()

    print("=" * 78)
    print("  sparse_back_project BAND sweep — isolated-subprocess harness (orchestrator)")
    print(f"  beta root: {sc.beta_root()}")
    print("=" * 78)

    setup, rc = sc.run_worker(script, ["--worker", "--mode", "setup"], extra_env=worker_env)
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
    print(f"  band modes: auto + {['div%d' % d for d in SHARD_DIVISORS]}")

    # One variant per band mode: 'auto' (default) then each div{d}.  Each runs a
    # fresh worker per size (clean peak memory) and gets its own YAML + plots.
    modes = [("auto", None)] + [(f"div{d}", d) for d in SHARD_DIVISORS]

    grids_by_mode = {}
    for mlabel, divisor in modes:
        print(f"\n=== band mode: {mlabel} ===")
        grid = {}
        failures_by_size = {}
        mem_kind = "n/a"
        for label in size_labels:
            args = ["--worker", "--mode", "measure", "--size", label,
                    "--device-counts", *[str(n) for n in device_counts],
                    "--warmup", str(WARMUP), "--trials", str(TRIALS)]
            if divisor is not None:
                args += ["--divisor", str(divisor)]
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
            summary = "  ".join(
                f"{n}d={by_n[n]['min_ms']:.0f}ms/{by_n[n]['num_bands']}b" for n in sorted(by_n))
            print(f"  size {label}: {summary}")

        results = {
            "op": OP_NAME, "band_mode": mlabel, "shard_divisor": divisor,
            "platform": plat, "device_label": dev_label, "mbirjax_path": mpath,
            "warmup": WARMUP, "trials": TRIALS,
            "device_counts": device_counts, "sizes": size_labels,
            "mem_kind": mem_kind, "time_ideal": "voxels_views", "correctness": corr,
            "dev2dev_safe": dev2dev_safe, "topology": topology,
            "grid": grid, "failures": failures_by_size,
        }
        sc.save_yaml(os.path.join(sc.RESULTS_DIR, f"{OP_NAME}_{mlabel}_{plat}.yaml"), results)
        if any(grid.values()):
            title = f"{OP_NAME} (band={mlabel})"
            sc.plot_device_sweep(title, grid, device_counts, size_labels, dev_label, mem_kind,
                                 os.path.join(sc.RESULTS_DIR, f"{OP_NAME}_{mlabel}_{plat}_device_sweep.png"))
            sc.plot_size_sweep(title, grid, device_counts, size_labels, dev_label, mem_kind,
                               os.path.join(sc.RESULTS_DIR, f"{OP_NAME}_{mlabel}_{plat}_size_sweep.png"))
        grids_by_mode[mlabel] = grid

    _print_band_comparison(grids_by_mode, size_labels, device_counts)
    print("\nReading: at each (size, n_dev), compare each div{d} mode vs 'auto'.  If a larger "
          "band\n(smaller d, e.g. div1 = full shard) gives t<1 (faster) without OOM, the default "
          "band\nover-shrinks and band sizing should be memory-budget-driven, not n_dev^2.")
    print("\nDone.")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        run_worker(sys.argv[1:])
    else:
        main()
