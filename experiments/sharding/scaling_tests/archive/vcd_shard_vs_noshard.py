"""
experiments/sharding/scaling_tests/vcd_shard_vs_noshard.py
──────────────────────────────────────────────────────────
A demo-style head-to-head of the user-facing ``recon`` (full multi-granular VCD)
run two ways on the SAME Shepp-Logan data:

  - **no-shard**: the single-device path (no mesh configured) — what users get today.
  - **shard**   : the placement path (``configure_sharding``) — 4 virtual CPU
                  devices on a CPU box, or all usable GPUs on a GPU box.

It reports wall-clock time, peak memory, and reconstruction accuracy (NRMSE vs the
phantom) for each, plus the shard/no-shard ratios.  This is the "does the sharding
actually pay off, and does it reconstruct the same thing" sanity demo to go with
the rigorous ``vcd_recon_scaling.py`` (which times the internal loop on pre-sharded
data); here we run the WHOLE user-facing ``recon`` (entry shard + loop + exit
gather), so it reflects the real end-user call.

Process isolation is the point: a single process would (a) carry the no-shard run's
compiled cache and device buffers into the shard run, and (b) report a
process-cumulative peak that conflates the two.  So — like the sweep harnesses —
the JAX-free orchestrator spawns ONE fresh worker subprocess per mode; the CPU
virtual-device count for the shard run is set via the worker's environment (before
its JAX import).

Memory caveat: on CPU the metric is whole-process RSS (shared host RAM), so
sharding across virtual CPU devices does NOT reduce it — expect roughly equal (or
slightly higher, from mesh overhead) RSS.  The real per-device memory win shows on
GPU, where the metric is per-device ``peak_bytes_in_use``; even there the
user-facing ``recon`` gathers the final volume to one device at exit, so for the
pure sharded-loop peak use ``vcd_recon_scaling.py``.

Run from the BETA worktree root (no CLI args):

    python experiments/sharding/scaling_tests/vcd_shard_vs_noshard.py
"""
import os
import sys
import argparse

import scaling_common as sc

import numpy as np

# mbirjax/jax are imported INSIDE the workers only (JAX-free orchestrator), so the
# orchestrator holds no backend while a worker measures peak memory.


OP_NAME = "vcd_shard_vs_noshard"

# ── Run configuration (edit here; no CLI args for the human) ──────────────────
# Problem size.  num_det_rows -> recon slices; keep num_views and num_det_rows
# divisible by the shard device counts you care about (4 for CPU; the GPU run uses
# the largest usable count that divides both, so 256 covers 1/2/4/8 but not 3).
#
# SIZE MATTERS — read before interpreting results.  VCD projects/priors over a
# SUBSET of pixels each call, so its per-op work is small; like the bare projectors,
# CPU sharding only pays off once the recon is big enough to amortize the
# per-device + per-subset overhead.  Measured on this box (4 virtual CPU devices, 8
# devices available):
#     64^3 : ~0.35x (BACKWARDS — overhead-bound; bare back-proj is 0.44x here too)
#     128^3: ~0.86x (still overhead-bound)
#     256^3: 2 dev 1.63x, 4 dev 1.86x, 8 dev 1.73x  (4 dev is the peak; 8 dev
#            regresses — bandwidth-bound on the shared CPU bus, like back projection)
# So the default below is 256^3 with CPU_SHARD_DEVICES=4 (the CPU sweet spot).  Drop
# the size to 128 for a quick (but <1x) smoke run; the GPU crossover is at a larger
# size still.  For rigorous per-size/per-device curves use vcd_recon_scaling.py.
NUM_VIEWS = 256
NUM_DET_ROWS = 256
NUM_DET_CHANNELS = 256
SHARPNESS = 1.0
MAX_ITERATIONS = 8                # comparison only needs a few iters; keeps runtime sane
STOP_THRESHOLD_CHANGE_PCT = 0.0   # force exactly MAX_ITERATIONS for a fair compare

CPU_SHARD_DEVICES = 4             # virtual CPU devices for the shard run on a CPU box
SEED = 0                          # subset-order RNG seed (set per timed call)


# ── Data + model builders (worker side) ───────────────────────────────────────
def build_data():
    """Deterministic Shepp-Logan phantom + parallel-beam sinogram (host numpy).

    Returns (phantom, sinogram, angles).  Passing a host (numpy) sinogram to
    ``recon`` means the shard run does not pay a single-device entry commit (a
    numpy input is sharded directly from the host), so the memory comparison is
    fair.
    """
    import mbirjax as mj
    phantom, sinogram, params = mj.generate_demo_data(
        object_type='shepp-logan', model_type='parallel',
        num_views=NUM_VIEWS, num_det_rows=NUM_DET_ROWS,
        num_det_channels=NUM_DET_CHANNELS)
    return np.asarray(phantom), np.asarray(sinogram), params['angles']


def largest_usable_count(n_request, num_views, num_slices):
    """Largest device count <= n_request that evenly divides BOTH sharded axes.

    configure_sharding requires the view axis (num_views) and the slice axis
    (num_slices) to be divisible by the device count, so this picks the most
    devices we can actually use for the given problem.
    """
    for n in range(int(n_request), 0, -1):
        if num_views % n == 0 and num_slices % n == 0:
            return n
    return 1


def run_one(mode):
    """Build data + model, run the full recon once (timed), measure memory + NRMSE.

    Args:
        mode (str): 'noshard' (single device, no mesh) or 'shard' (configure_sharding
            over the usable devices).

    Returns:
        dict with mode, n_devices, device_label, min_ms/mean_ms, mem_mb, mem_kind,
        nrmse.
    """
    import mbirjax as mj  # noqa: F401  (device-setup-first; precedes any jax import via sc)
    phantom, sinogram, angles = build_data()
    plat, max_dev = sc.detect_platform()

    model = mj.ParallelBeamModel(sinogram.shape, angles)
    model.set_params(sharpness=SHARPNESS, verbose=0)

    if mode == "shard":
        target = CPU_SHARD_DEVICES if plat == "cpu" else max_dev
        recon_shape = model.get_params('recon_shape')
        n = largest_usable_count(min(target, max_dev), sinogram.shape[0], recon_shape[2])
        devs = sc.pick_devices(n)
        model.configure_sharding(devs)
        used_devices = devs
        n_devices = n
    else:
        # No mesh: the single-device path.  Memory is measured on device 0.
        used_devices = sc.pick_devices(1)
        n_devices = 1

    def _recon():
        np.random.seed(SEED)   # deterministic partitions + subset order each call
        recon, _ = model.recon(sinogram, weights=None, max_iterations=MAX_ITERATIONS,
                               stop_threshold_change_pct=STOP_THRESHOLD_CHANGE_PCT,
                               print_logs=False)
        return recon

    # warmup excludes compile; one timed trial is enough for an illustrative demo
    # (use vcd_recon_scaling.py for rigorous min-over-trials numbers).
    stats, recon = sc.time_op(_recon, warmup=1, trials=1)
    mem_mb, mem_kind = sc.peak_memory_mb(used_devices)
    recon_np = np.asarray(recon)
    nrmse = float(np.linalg.norm(recon_np - phantom) / np.linalg.norm(phantom))

    return {"mode": mode, "n_devices": n_devices, "platform": plat,
            "device_label": sc.device_label(),
            "min_ms": stats["min_ms"], "mean_ms": stats["mean_ms"],
            "mem_mb": mem_mb, "mem_kind": mem_kind, "nrmse": nrmse}


# ── Worker dispatch ───────────────────────────────────────────────────────────
def run_worker(argv):
    p = argparse.ArgumentParser(description="vcd shard-vs-noshard worker (internal)")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--mode", choices=["noshard", "shard"], required=True)
    p.add_argument("--out-file", required=True)
    a = p.parse_args(argv)
    import mbirjax  # noqa: F401  device-setup-first
    result = run_one(a.mode)
    sc.write_worker_result(a.out_file, result)
    print(f"[{a.mode}] n_devices={result['n_devices']}  "
          f"min={result['min_ms']:.1f} ms  mem={result['mem_mb']:.1f} MB "
          f"({result['mem_kind']})  nrmse={result['nrmse']:.4f}")


# ── Orchestrator (touches no JAX) ─────────────────────────────────────────────
def _beta_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        os.pardir, os.pardir, os.pardir))


def main():
    script = os.path.abspath(__file__)
    beta_root = _beta_root()
    existing_pp = os.environ.get("PYTHONPATH", "")
    base_env = {
        "PYTHONPATH": beta_root + (os.pathsep + existing_pp if existing_pp else ""),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9",
    }

    print("=" * 72)
    print("  VCD shard vs no-shard — full recon() head-to-head (isolated processes)")
    print(f"  beta root: {beta_root}")
    print(f"  size: views={NUM_VIEWS} rows={NUM_DET_ROWS} channels={NUM_DET_CHANNELS}"
          f"   iterations: {MAX_ITERATIONS}")
    print("=" * 72)

    # No-shard worker: single device (do NOT force virtual CPU devices).
    noshard, rc1 = sc.run_worker(script, ["--worker", "--mode", "noshard"],
                                 extra_env=base_env)
    # Shard worker: on CPU, request CPU_SHARD_DEVICES virtual devices (set before
    # the worker's JAX import); on GPU the flag is ignored and all GPUs are seen.
    shard_env = dict(base_env)
    shard_env["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "")
        + f" --xla_force_host_platform_device_count={CPU_SHARD_DEVICES}").strip()
    shard_env["MBIRJAX_NUM_CPU_DEVICES"] = str(CPU_SHARD_DEVICES)
    shard, rc2 = sc.run_worker(script, ["--worker", "--mode", "shard"],
                               extra_env=shard_env)

    if not noshard or not shard:
        print(f"  ERROR: a worker produced no result (noshard rc={rc1}, shard rc={rc2}).")
        return

    plat = noshard["platform"]
    dev_label = noshard["device_label"]
    print()
    print(f"  platform: {plat}   ({dev_label})")
    print(f"  {'mode':<9} {'devices':>7} {'time_ms':>12} {'mem_MB':>10} {'nrmse':>9}")
    print("  " + "-" * 50)
    for r in (noshard, shard):
        print(f"  {r['mode']:<9} {r['n_devices']:>7} {r['min_ms']:>12.1f} "
              f"{r['mem_mb']:>10.1f} {r['nrmse']:>9.4f}")

    speedup = noshard["min_ms"] / shard["min_ms"] if shard["min_ms"] else float("nan")
    mem_ratio = shard["mem_mb"] / noshard["mem_mb"] if noshard["mem_mb"] else float("nan")
    print("  " + "-" * 50)
    print(f"  shard speedup vs no-shard : {speedup:.2f}x  "
          f"({shard['n_devices']} devices)")
    print(f"  shard memory / no-shard   : {mem_ratio:.2f}x  ({shard['mem_kind']})")
    if plat == "cpu":
        print("  note: CPU memory is whole-process RSS (shared host RAM) — sharding")
        print("        across virtual CPU devices does not reduce it.  The per-device")
        print("        memory win shows on GPU (peak_bytes_in_use).")
    print("  note: this is the full user-facing recon() (entry shard + loop + exit")
    print("        gather).  For the pure sharded-loop peak, use vcd_recon_scaling.py.")

    results = {"op": OP_NAME, "platform": plat, "device_label": dev_label,
               "size": [NUM_VIEWS, NUM_DET_ROWS, NUM_DET_CHANNELS],
               "max_iterations": MAX_ITERATIONS,
               "noshard": noshard, "shard": shard,
               "speedup": speedup, "mem_ratio": mem_ratio}
    sc.save_yaml(os.path.join(sc.RESULTS_DIR, f"{OP_NAME}_{plat}.yaml"), results)
    print("\nDone.")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        run_worker(sys.argv[1:])
    else:
        main()
