"""
experiments/sharding/scaling_tests/fbp_filter_row_batch_sweep.py
─────────────────────────────────────────────────────────────────
Sweep the row_batch fbp_filter kernel's batch B (_FBP_ROW_BATCH) at a fixed
large problem size, across device counts, to find the throughput/memory sweet
spot and inform the B(c) memory budget.  The cuFFT work area is ~ B * fft_len(c),
so B trades per-device memory against parallel width.

ISOLATED-SUBPROCESS HARNESS (same pattern as fbp_filter_scaling.py): the
orchestrator touches no JAX; one FRESH worker per (device_count, B) config, so
each B gets a clean cuFFT plan and a clean peak-memory reading — and a fresh
process traces the row_batch kernel once with the chosen B (no jit-cache-vs-
module-global staleness).  B ascends per device count and stops on OOM, since a
larger B needs more memory ("largest B that fits").

Run config is in the top-of-file constants (no human CLI).  Run from the BETA
worktree root; the orchestrator forces beta onto each worker's PYTHONPATH:

    python experiments/sharding/scaling_tests/fbp_filter_row_batch_sweep.py
"""
import os
import sys
import argparse

import scaling_common as sc
import fbp_filter_scaling as ffs   # reuse builders, _beta_root, _OOM_MARKERS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator


OP_NAME = "fbp_filter_row_batch"

# ── Run configuration (edit here; no CLI args for the human) ──────────────────
# Stress sizes per platform, VARYING c (channels) at fixed v=r, to capture how
# the cuFFT work area — and thus the best B — scales with c (work area ~
# B*fft_len(c)).  n_views (=v) must be divisible by every device count used.
SIZES = {
    "cpu": [(512, 512, 128), (512, 512, 256), (512, 512, 512)],
    "gpu": [(1624, 1624, 512), (1624, 1624, 1024), (1624, 1624, 1624)],
}
# Device counts to sweep B at (clamped to the available max); a middle value
# included so the device scaling of the memory/throughput tradeoff is visible.
DEVICE_COUNTS = {"cpu": [1, 4, 8], "gpu": [1, 2, 4]}
# Row-batch B values (powers of two; the kernel caps B at total_rows).
ROW_BATCHES = [64, 256, 1024, 4096]
WARMUP = 1
TRIALS = 3


# ── Worker side (isolated subprocess) ─────────────────────────────────────────
def worker_probe(out_file):
    """Report platform, device count/label, and beta status."""
    import mbirjax  # device-setup-first
    plat, max_dev = sc.detect_platform()
    pkg = os.path.dirname(mbirjax.__file__)
    state, branch = sc.beta_status(pkg)
    sc.write_worker_result(out_file, {
        "platform": plat, "max_devices": max_dev, "device_label": sc.device_label(),
        "beta_state": state, "branch": branch, "mbirjax_path": pkg})


def worker_measure_one(size_label, n_devices, row_batch, warmup, trials, out_file):
    """Measure time + peak memory for one (size, n_devices, B) config."""
    import mbirjax  # device-setup-first
    import mbirjax.parallel_beam as pb
    pb._FBP_FILTER_KERNEL = "row_batch"
    pb._FBP_ROW_BATCH = row_batch          # picked up at first (fresh) trace
    size = ffs.parse_size_label(size_label)
    devs = sc.pick_devices(n_devices)
    res = {"n_devices": n_devices, "row_batch": row_batch, "size": size_label}
    if devs is None:
        res["error"] = "not enough devices"
        sc.write_worker_result(out_file, res)
        return
    try:
        model = ffs.make_model(size, devices=devs)
        if model is None:
            res["error"] = "model build failed (divisibility?)"
            sc.write_worker_result(out_file, res)
            return
        sino = model._shard_sinogram(ffs.make_input(size, seed=0))
        stats, _ = sc.time_op(lambda: ffs.run_fbp_filter(model, sino), warmup, trials)
        mem_mb, mem_kind = sc.peak_memory_mb(devs)
        res.update(stats)
        res["mem_mb"] = mem_mb
        res["mem_kind"] = mem_kind
        print(f"  {size_label} ndev={n_devices} B={row_batch:5d}  "
              f"min={stats['min_ms']:9.2f} ms  mem={mem_mb:9.1f} MB ({mem_kind})")
    except Exception as e:   # noqa: BLE001 — measurement harness
        msg = str(e).replace("\n", " ")
        res["oom"] = any(k in msg.upper() for k in ffs._OOM_MARKERS)
        res["error"] = msg[:300]
        print(f"  {size_label} ndev={n_devices} B={row_batch:5d}  "
              f"{'OOM' if res['oom'] else 'ERROR'}: {msg[:100]}")
    sc.write_worker_result(out_file, res)


def run_worker(argv):
    p = argparse.ArgumentParser(description="row_batch B-sweep worker (internal)")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--mode", choices=["probe", "measure"], required=True)
    p.add_argument("--size", default=None)
    p.add_argument("--n-devices", type=int, default=None)
    p.add_argument("--row-batch", type=int, default=None)
    p.add_argument("--warmup", type=int, default=WARMUP)
    p.add_argument("--trials", type=int, default=TRIALS)
    p.add_argument("--out-file", required=True)
    a = p.parse_args(argv)
    if a.mode == "probe":
        worker_probe(a.out_file)
    else:
        worker_measure_one(a.size, a.n_devices, a.row_batch, a.warmup, a.trials,
                           a.out_file)


# ── Plot ──────────────────────────────────────────────────────────────────────
def _shard_mb(size_label, n_devices):
    nv, nr, nc = (int(x) for x in size_label.split("x"))
    return nv * nr * nc * 4 / n_devices / (1024 ** 2)   # float32 shard, per device


def _b_sweep_limits(grid, device_counts, sizes):
    """Shared y-limits across the per-device figures so they are comparable.

    Returns (time_ylim, mem_ylim): time on a log axis (multiplicative pad), the
    memory ratio on a linear axis (additive pad, always including the ideal 2×).
    """
    times, ratios = [], []
    for n in device_counts:
        per = grid.get(n, {})
        for size_label in sizes:
            shard = _shard_mb(size_label, n)
            for e in per.get(size_label, []):
                if "min_ms" in e:
                    times.append(e["min_ms"])
                    ratios.append(e["mem_mb"] / shard)
    if not times:
        return None, None
    tlo, thi = min(times), max(times)
    tpad = (thi / tlo) ** 0.06 if tlo > 0 else 1.0
    rlo, rhi = min(ratios + [2.0]), max(ratios + [2.0])   # keep the ideal line in view
    rpad = (rhi - rlo) * 0.06 or 1.0
    return (tlo / tpad, thi * tpad), (rlo - rpad, rhi + rpad)


def plot_b_sweep_for_device(per_size, sizes, row_batches, n_devices, dev_label,
                            out_path, time_ylim=None, mem_ylim=None):
    """For one device count: time vs B and peak-memory-÷-shard vs B, one curve
    per size (varying c), so the c-dependence of the best B is visible.  Shared
    y-limits (time_ylim/mem_ylim) make the per-device figures comparable."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for size_label in sizes:
        entries = [e for e in per_size.get(size_label, []) if "min_ms" in e]
        if not entries:
            continue
        bs = [e["row_batch"] for e in entries]
        shard = _shard_mb(size_label, n_devices)
        ax1.plot(bs, [e["min_ms"] for e in entries], "o-", label=size_label)
        ax2.plot(bs, [e["mem_mb"] / shard for e in entries], "s-", label=size_label)
    ax2.axhline(2.0, ls="--", color="gray", alpha=0.7, label="ideal (2×)")
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(row_batches)                          # ticks only at the B values
        ax.set_xticklabels([str(b) for b in row_batches])   # decimal labels (64, 256, …)
        ax.xaxis.set_minor_locator(NullLocator())           # no extra log minor ticks
        ax.set_xlabel("row batch B (_FBP_ROW_BATCH)")
        ax.grid(True, alpha=0.3)
        ax.legend(title="size (v×r×c)")
    ax1.set_ylabel("min time (ms)")
    ax1.set_yscale("log")
    ax1.set_title("time vs B")
    if time_ylim:
        ax1.set_ylim(*time_ylim)
    ax2.set_ylabel("peak mem/device ÷ shard")
    ax2.set_title("per-device memory ÷ shard size vs B")
    if mem_ylim:
        ax2.set_ylim(*mem_ylim)
    fig.suptitle(f"fbp_filter row_batch B-sweep — {n_devices} dev — {dev_label}",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_all(grid, sizes, row_batches, device_counts, dev_label, plat):
    """Plot every device-count figure with shared (comparable) y-limits."""
    time_ylim, mem_ylim = _b_sweep_limits(grid, device_counts, sizes)
    for n in device_counts:
        if any(grid.get(n, {}).get(sl) for sl in sizes):
            plot_b_sweep_for_device(
                grid[n], sizes, row_batches, n, dev_label,
                os.path.join(sc.RESULTS_DIR, f"{OP_NAME}_{plat}_{n}dev.png"),
                time_ylim, mem_ylim)


# ── Orchestrator (touches no JAX) ─────────────────────────────────────────────
def main():
    script = os.path.abspath(__file__)
    beta_root = ffs._beta_root()
    existing_pp = os.environ.get("PYTHONPATH", "")
    # Preallocate the pool up front so there's no per-call cudaMalloc heap growth
    # → clean timing; MEM_FRACTION raised from the 0.75 default so the largest
    # configs don't OOM on the cap rather than on real usage.  peak_bytes_in_use
    # tracks in-use tensors (not the preallocated pool), so memory should stay
    # accurate — the sanity check below verifies true == false on one config.
    worker_env = {
        "PYTHONPATH": beta_root + (os.pathsep + existing_pp if existing_pp else ""),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9",
    }
    print("=" * 72)
    print("  fbp_filter row_batch B-sweep (orchestrator)")
    print(f"  beta root: {beta_root}")
    print("=" * 72)

    probe, rc = sc.run_worker(script, ["--worker", "--mode", "probe"],
                              extra_env=worker_env)
    if probe is None:
        print(f"  ERROR: probe worker produced no result (rc={rc}); aborting.")
        return
    plat = probe["platform"]
    max_dev = probe["max_devices"]
    dev_label = probe["device_label"]
    state = probe.get("beta_state", "unknown")
    branch = probe.get("branch")
    label = {"beta": f"*** beta ***  (branch {branch})",
             "not-beta": f"### NOT beta — branch {branch} ###"}.get(
                 state, "(branch undetermined — verify path)")
    print(f"  mbirjax: {label}   {probe.get('mbirjax_path')}")
    print(f"  platform: {plat}   max devices: {max_dev}   ({dev_label})")
    if plat == "cpu":
        worker_env["MBIRJAX_NUM_CPU_DEVICES"] = str(max_dev)

    sizes = SIZES[plat]
    size_labels = [sc.size_label(s) for s in sizes]
    device_counts = [n for n in DEVICE_COUNTS[plat] if n <= max_dev]
    row_batches = sorted(ROW_BATCHES)
    print(f"  sizes: {size_labels}   device counts: {device_counts}   B: {row_batches}")

    # ── Preallocate sanity: does peak_bytes_in_use under preallocate=true match
    # the =false value?  If yes, this one (true) run gives clean timing AND
    # accurate memory; if =true reports ~the pool instead, memory needs =false.
    s_size, s_nd = size_labels[0], device_counts[-1]
    s_B = row_batches[len(row_batches) // 2]
    print(f"\n[preallocate sanity] {s_size} ndev={s_nd} B={s_B}: peak true vs false")
    for pre in ("true", "false"):
        env = dict(worker_env)
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = pre
        res, _ = sc.run_worker(
            script, ["--worker", "--mode", "measure", "--size", s_size,
                     "--n-devices", str(s_nd), "--row-batch", str(s_B),
                     "--warmup", "1", "--trials", "1"], extra_env=env)
        if res and "mem_mb" in res:
            print(f"  preallocate={pre:5s}: peak={res['mem_mb']:9.1f} MB  "
                  f"min={res['min_ms']:7.1f} ms")
        else:
            print(f"  preallocate={pre:5s}: no result ({res})")

    grid = {}   # grid[n][size_label] = [rows over B]
    for n in device_counts:
        grid[n] = {}
        for size_label in size_labels:
            grid[n][size_label] = []
            for B in row_batches:   # ascending → memory ascends; stop on OOM
                args = ["--worker", "--mode", "measure", "--size", size_label,
                        "--n-devices", str(n), "--row-batch", str(B),
                        "--warmup", str(WARMUP), "--trials", str(TRIALS)]
                res, rc = sc.run_worker(script, args, extra_env=worker_env)
                if res is None:
                    print(f"  {size_label} ndev={n} B={B}: no result (rc={rc})")
                    continue
                grid[n][size_label].append(res)
                if res.get("oom"):
                    print(f"  {size_label} ndev={n}: OOM at B={B}; stopping "
                          f"B-ascent (larger B needs more per-device memory)")
                    break

    results = {
        "op": OP_NAME, "platform": plat, "device_label": dev_label,
        "fbp_kernel": "row_batch", "sizes": size_labels,
        "device_counts": device_counts, "row_batches": row_batches,
        "mbirjax_path": probe.get("mbirjax_path"),
        "warmup": WARMUP, "trials": TRIALS,
        "grid": {str(n): per for n, per in grid.items()},
    }
    sc.save_yaml(os.path.join(sc.RESULTS_DIR, f"{OP_NAME}_{plat}.yaml"), results)
    plot_all(grid, size_labels, row_batches, device_counts, dev_label, plat)
    print("\nDone.")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        run_worker(sys.argv[1:])
    else:
        main()
