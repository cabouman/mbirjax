"""A/B + MEMORY for cone single-device back projection.

Question this answers: on ONE device, is the current default (the SHARDED path on a
trivial 1-device mesh) better or worse than the plain single-device path -- in BOTH
time and peak memory?  This is the evidence for the proposed n=1 back short-circuit
(route a trivial 1-device mesh to ``_sparse_back_project_single_device``, GPU only).

  A = model.sparse_back_project(...)            -> the sharded path (today's 1-device default)
  B = model._sparse_back_project_single_device  -> the single-device path (the short-circuit target)

Both are fed the SAME already-on-device sinogram (what the real short-circuit would
hand B), so we measure the OP, not a host->device transfer.

Per-config peak memory only means something if each config runs in its OWN fresh
process: ``peak_bytes_in_use`` is a process-cumulative high-water mark, so measuring
A then B in one process would report B's peak as max(A, B).  So the orchestrator (this
file, run with NO arguments) stays JAX-free and spawns one isolated worker subprocess
per (size, config), exactly like performance_tracking.py.  It then prints A vs B (time
+ memory) and, if a main-branch baseline is found, the vs-main comparison for each.

Run it with no arguments:

    python cone_back_ab_memory.py

(The ``--worker`` invocation below is internal self-spawning; you never type it.)
"""

import os
import sys

# ─── Editable run parameters (versioned; no command-line args) ────────────────
# Sinogram sizes = (n_views, n_rows, n_channels); these are the cone GPU sweep
# cells where the +45% time penalty (512 cell) and the +25% memory note (1024
# cell) were seen.  Trim to [(512, 448, 384)] for a quick single-size check.
SIZES = [(512, 448, 384), (1024, 1008, 992)]
GEOMETRY = "cone"
WARMUP = 1      # untimed calls (compile + caches), per config
TRIALS = 3      # timed calls; min is reported (robust to GPU run-to-run scatter)
# ──────────────────────────────────────────────────────────────────────────────

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scaling_common as sc   # noqa: E402  (jax-free at import; functions import jax lazily)


# ─── Worker: measure ONE (size, config) in this fresh process, write YAML ─────
def _worker(size_label, which, out_file):
    """Measure one config at one size and publish min_ms + peak mem to out_file.

    ``which`` is 'A' (sharded path) or 'B' (single-device path).  Runs on a single
    device; ``peak_bytes_in_use`` read right after the timing loop is therefore this
    config's own high-water mark (the loop frees the prior output before each call).
    """
    import jax  # noqa: F401  (ensures the backend is up before peak_memory_mb)
    import performance_tracking as pt

    cfg = pt.Config()
    size = tuple(int(x) for x in size_label.split("x"))
    devs = sc.pick_devices(1)

    model = pt.make_model(cfg, GEOMETRY, size)
    model.configure_devices(1)
    indices = pt.make_indices(model)
    # Pre-place the sinogram in the device (sharded) form, OUTSIDE the timing loop --
    # the same on-device array the real short-circuit would hand to B.
    sino_dev = pt.to_device(model, pt.make_sinogram(cfg, size), "sino")

    if which == "A":
        run = lambda: model.sparse_back_project(sino_dev, indices)               # sharded path
    else:
        run = lambda: model._sparse_back_project_single_device(sino_dev, indices)  # single-device path

    stats, _ = sc.time_op(run, WARMUP, TRIALS)
    mem_mb, mem_kind = sc.peak_memory_mb(devs)
    plat, _ = sc.detect_platform()
    sc.write_worker_result(out_file, {
        "size": size_label, "which": which, "platform": plat,
        "recon_shape": [int(x) for x in model.get_params("recon_shape")],
        "min_ms": float(stats["min_ms"]), "mean_ms": float(stats["mean_ms"]),
        "mem_mb": float(mem_mb), "mem_kind": mem_kind,
    })


# ─── Orchestrator: spawn workers, collect, print A/B + vs-main ────────────────
def _load_main_back(plat, size_label):
    """Best-effort (min_ms, mem_mb) for cone `back` 1-device from the main baseline.

    Looks in results/golden/ and results/golden_gpu/ for main_baseline_<plat>.yaml
    (the file captured by capture_main_baseline.py).  Returns (None, None) if absent.
    """
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
    return f"{100.0 * (x - ref) / ref:+.0f}%" if (x and ref) else "  n/a"


def _orchestrate():
    print("=" * 78)
    print("  cone back, 1 device:  A = sharded path   B = single-device path")
    print("  (each config measured in its own subprocess -> per-config peak memory)")
    print("=" * 78)
    extra_env = sc.build_worker_env()
    results = {}   # (size_label, which) -> result dict
    for size in SIZES:
        size_label = "x".join(str(x) for x in size)
        for which in ("A", "B"):
            res, rc = sc.run_worker(
                os.path.abspath(__file__),
                ["--worker", "--size", size_label, "--which", which],
                extra_env=extra_env)
            if res is None:
                print(f"  {size_label}  {which}: FAILED (returncode {rc})")
                continue
            results[(size_label, which)] = res
            print(f"  {size_label}  {which}: {res['min_ms']:9.1f} ms   "
                  f"{res['mem_mb']:9.1f} MB ({res['mem_kind']})")

    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    for size in SIZES:
        size_label = "x".join(str(x) for x in size)
        a = results.get((size_label, "A"))
        b = results.get((size_label, "B"))
        if not a or not b:
            print(f"\n{size_label}: incomplete (A or B missing)")
            continue
        plat = a.get("platform", "gpu")
        main_ms, main_mem = _load_main_back(plat, size_label)
        print(f"\n{size_label}  (recon {a.get('recon_shape')}):")
        print(f"  A sharded        : {a['min_ms']:9.1f} ms   {a['mem_mb']:9.1f} MB")
        print(f"  B single-device  : {b['min_ms']:9.1f} ms   {b['mem_mb']:9.1f} MB")
        if main_ms or main_mem:
            ms = f"{main_ms:9.1f} ms" if main_ms else "      n/a"
            mm = f"{main_mem:9.1f} MB" if main_mem else "      n/a"
            print(f"  main (baseline)  : {ms}   {mm}")
        print(f"  A/B   : time {a['min_ms'] / b['min_ms']:.2f}   "
              f"mem {a['mem_mb'] / b['mem_mb']:.2f}")
        if main_ms or main_mem:
            print(f"  vs main: A  time {_pct(a['min_ms'], main_ms)}  mem {_pct(a['mem_mb'], main_mem)}"
                  f"   |  B  time {_pct(b['min_ms'], main_ms)}  mem {_pct(b['mem_mb'], main_mem)}")
    print("\n(Expectation: GPU B <= A in both time and memory; the short-circuit recovers")
    print(" the driver overhead and, at 1024, the band-reassembly memory.  Confirm here.)")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        a = sys.argv
        _worker(a[a.index("--size") + 1], a[a.index("--which") + 1],
                a[a.index("--out-file") + 1])
    else:
        _orchestrate()
