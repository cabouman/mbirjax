"""A/B + MEMORY + kernel-isolation for cone single-device back projection.

Question: on ONE device, what is the best back-projection path -- in BOTH time and
peak memory?  Evidence for the proposed n=1 back short-circuit, and for whether a
"lightweight banded" path can give us speed AND low memory (no tradeoff).

Five configs, all at n=1, all fed the SAME already-on-device sinogram (so we measure
the OP, not a host->device transfer):

  A         model.sparse_back_project              the SHARDED path (today's default):
                                                   band loop + thread pool + reduce-scatter
                                                   + assemble_sharded (driver machinery)
  B         model._sparse_back_project_single_device   the single-device path (= main's
                                                   behavior; the simple short-circuit target)
  Kpix      projector_functions.sparse_back_project    driver-less pixel-batch kernel
                                                   (B's compute, no single-device driver)
  Kband1    projector_functions.sparse_back_project_band(0, nsl)   driver-less BAND kernel,
                                                   ONE band of all slices (A's kernel, UNBANDED)
  Kbandloop manual band loop over sparse_back_project_band, concatenate, NO thread pool /
                                                   sum / assemble   <-- the "lightweight banded
                                                   n=1" candidate: banding (low memory) without
                                                   the sharded driver

Readout (the decisive comparisons):
  * Kpix vs B            -> does the single-device driver add overhead?  (expect ~equal)
  * Kpix vs Kband1       -> pixel(rolled) vs band(single) KERNEL cost on this platform
  * Kbandloop vs B (time) and vs A (memory)  -> if time ~ B and memory ~ A, the lightweight
                            banded path is the best-of-both (fast AND capacity-safe).
  * A vs Kbandloop (time)-> the sharded-driver overhead (thread pool / sum / assemble).

Per-config peak memory only means something if each config runs in its OWN fresh process
(``peak_bytes_in_use`` is a process-cumulative high-water mark), so the orchestrator (this
file, run with NO arguments) stays JAX-free and spawns one isolated worker subprocess per
(size, config), exactly like performance_tracking.py.

Run it with no arguments:

    python cone_back_ab_memory.py

(The ``--worker`` invocation below is internal self-spawning; you never type it.)
"""

import os
import sys

# ─── Editable run parameters (versioned; no command-line args) ────────────────
# Sinogram sizes = (n_views, n_rows, n_channels); the cone GPU sweep cells where the
# +127% time penalty (512 cell) and the memory inversion (1024 cell) were seen.  Run
# just [(512, 448, 384)] first for a quick read; add 1024 for the capacity question.
SIZES = [(512, 448, 384), (1024, 1008, 992)]
# Which paths to measure, in increasing cost order (cheap ones first).  Trim freely.
CONFIGS = ["B", "Kpix", "Kband1", "Kbandloop", "A"]
GEOMETRY = "cone"
WARMUP = 1      # untimed calls (compile + caches), per config
TRIALS = 3      # timed calls; min is reported (robust to GPU run-to-run scatter)
# ──────────────────────────────────────────────────────────────────────────────

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scaling_common as sc   # noqa: E402  (jax-free at import; functions import jax lazily)

# Human-readable labels for the summary.
LABELS = {
    "A": "sharded (driver)",
    "B": "single-device",
    "Kpix": "kernel:pixel (driverless)",
    "Kband1": "kernel:band x1 (driverless)",
    "Kbandloop": "kernel:band loop (driverless)",
}


def _build_run(model, sino_dev, indices, which):
    """Return (run_fn, info) for one config.  ``info`` carries band metadata (Kbandloop)."""
    import jax.numpy as jnp
    pf = model.projector_functions
    nsl = int(model.get_params("recon_shape")[2])
    npix = int(len(indices))
    info = {}
    if which == "A":
        return (lambda: model.sparse_back_project(sino_dev, indices)), info
    if which == "B":
        return (lambda: model._sparse_back_project_single_device(sino_dev, indices)), info
    if which == "Kpix":
        return (lambda: pf.sparse_back_project(sino_dev, indices)), info
    if which == "Kband1":
        return (lambda: pf.sparse_back_project_band(sino_dev, indices, 0, nsl)), info
    if which == "Kbandloop":
        # Same band sizing the sharded path uses at n=1, but a plain single-device loop:
        # no thread pool, no cross-device sum, no assemble_sharded.
        band_len = type(model)._slice_band_length(nsl, 1, npix)
        bounds = type(model)._balanced_slice_bounds(nsl, band_len)
        info = {"band_len": int(band_len), "num_bands": len(bounds)}

        def run():
            bands = [pf.sparse_back_project_band(sino_dev, indices, g0, g1 - g0)
                     for (g0, g1) in bounds]
            return bands[0] if len(bands) == 1 else jnp.concatenate(bands, axis=1)
        return run, info
    raise ValueError(f"unknown config {which!r}")


# ─── Worker: measure ONE (size, config) in this fresh process, write YAML ─────
def _worker(size_label, which, out_file):
    import jax  # noqa: F401  (ensure backend is up before peak_memory_mb)
    import performance_tracking as pt

    cfg = pt.Config()
    size = tuple(int(x) for x in size_label.split("x"))
    devs = sc.pick_devices(1)

    model = pt.make_model(cfg, GEOMETRY, size)
    model.configure_devices(1)
    indices = pt.make_indices(model)
    # Pre-place the sinogram in the device (sharded) form, OUTSIDE the timing loop --
    # the same on-device array the real short-circuit would hand any of these paths.
    sino_dev = pt.to_device(model, pt.make_sinogram(cfg, size), "sino")

    run, info = _build_run(model, sino_dev, indices, which)
    stats, _ = sc.time_op(run, WARMUP, TRIALS)
    mem_mb, mem_kind = sc.peak_memory_mb(devs)
    plat, _ = sc.detect_platform()
    sc.write_worker_result(out_file, {
        "size": size_label, "which": which, "platform": plat,
        "recon_shape": [int(x) for x in model.get_params("recon_shape")],
        "min_ms": float(stats["min_ms"]), "mean_ms": float(stats["mean_ms"]),
        "mem_mb": float(mem_mb), "mem_kind": mem_kind, "info": info,
    })


# ─── Orchestrator: spawn workers, collect, print table + vs-main + key ratios ──
def _load_main_back(plat, size_label):
    """Best-effort (min_ms, mem_mb) for cone `back` 1-device from the main baseline."""
    from ruamel.yaml import YAML
    yaml = YAML()
    for sub in ("golden", f"golden_{plat}"):
        path = os.path.join(HERE, "results", sub, f"main_baseline_{plat}.yaml")
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                data = yaml.load(f)
        except Exception:
            continue
        for c in (data or {}).get("cells", []):
            if (c.get("geometry") == GEOMETRY and c.get("op") == "back"
                    and c.get("size") == size_label and c.get("n_devices") == 1):
                return c.get("min_ms"), c.get("mem_mb")
    return None, None


def _pct(x, ref):
    return f"{100.0 * (x - ref) / ref:+.0f}%" if (x and ref) else " n/a"


def _ratio(x, y):
    return f"{x / y:.2f}" if (x and y) else "n/a"


def _orchestrate():
    print("=" * 80)
    print("  cone back, 1 device — A/B + kernel isolation (per-config peak via subprocess)")
    print("=" * 80)
    extra_env = sc.build_worker_env()
    results = {}   # (size_label, which) -> result dict
    for size in SIZES:
        size_label = "x".join(str(x) for x in size)
        for which in CONFIGS:
            res, rc = sc.run_worker(
                os.path.abspath(__file__),
                ["--worker", "--size", size_label, "--which", which],
                extra_env=extra_env)
            if res is None:
                print(f"  {size_label}  {which:10s}: FAILED (returncode {rc})")
                continue
            results[(size_label, which)] = res
            extra = ""
            if res.get("info"):
                extra = f"   [{res['info'].get('num_bands')} bands x {res['info'].get('band_len')}]"
            print(f"  {size_label}  {which:10s}: {res['min_ms']:9.1f} ms   "
                  f"{res['mem_mb']:9.1f} MB ({res['mem_kind']}){extra}")

    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    for size in SIZES:
        size_label = "x".join(str(x) for x in size)
        got = {w: results.get((size_label, w)) for w in CONFIGS}
        if not any(got.values()):
            continue
        plat = next((r["platform"] for r in got.values() if r), "gpu")
        main_ms, main_mem = _load_main_back(plat, size_label)
        recon = next((r["recon_shape"] for r in got.values() if r), None)
        print(f"\n{size_label}  (recon {recon}):")
        for w in CONFIGS:
            r = got.get(w)
            if not r:
                continue
            vs = f"   vs main: t {_pct(r['min_ms'], main_ms)}  m {_pct(r['mem_mb'], main_mem)}"
            print(f"  {w:10s} {LABELS[w]:30s} {r['min_ms']:9.1f} ms  {r['mem_mb']:9.1f} MB{vs}")
        if main_ms or main_mem:
            print(f"  {'main':10s} {'baseline':30s} "
                  f"{(main_ms or 0):9.1f} ms  {(main_mem or 0):9.1f} MB")
        a, b = got.get("A"), got.get("B")
        kbl = got.get("Kbandloop")
        print("  -- key ratios --")
        if a and b:
            print(f"    A/B            : time {_ratio(a['min_ms'], b['min_ms'])}  "
                  f"mem {_ratio(a['mem_mb'], b['mem_mb'])}")
        if kbl and b:
            print(f"    Kbandloop / B  : time {_ratio(kbl['min_ms'], b['min_ms'])}   "
                  f"(want ~1.0 -> banding costs no time vs single-device)")
        if kbl and a:
            print(f"    Kbandloop / A  : time {_ratio(kbl['min_ms'], a['min_ms'])}  "
                  f"mem {_ratio(kbl['mem_mb'], a['mem_mb'])}   "
                  f"(want time<1, mem~1 -> drops driver overhead, keeps A's low memory)")
    print("\n  Best-of-both if, at 1024:  Kbandloop time ~ B  AND  Kbandloop mem ~ A (<< B).")
    print("  Then a lightweight banded n=1 path beats both choosing B (memory) and keeping A (time).")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        av = sys.argv
        _worker(av[av.index("--size") + 1], av[av.index("--which") + 1],
                av[av.index("--out-file") + 1])
    else:
        _orchestrate()
