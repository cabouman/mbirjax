"""
experiments/sharding/scaling_tests/vcd_mesh_sweep.py
────────────────────────────────────────────────────
Orchestrate the vcd_mesh_mem_attribution.py memory sweeps, ONE CONFIG PER SUBPROCESS.

WHY subprocesses: peak_bytes_in_use is a process-cumulative high-water mark, so a clean
peak per config requires a fresh process.  This orchestrator launches the worker
(vcd_mesh_mem_attribution.py) once per config, passing the config via VMA_* env vars
(an orchestration mechanism — the worker's by-hand constants still work when unset).
The orchestrator itself imports neither jax nor mbirjax, so it holds no device memory
while a worker measures.

USAGE (no CLI args — edit CONFIGS / SIZE / WORST_NS below, then run):
    source .../conda.sh && conda activate mbirjax
    cd experiments/sharding/scaling_tests
    nvidia-smi dmon -s pct -c 5            # pick a cool card
    export CUDA_VISIBLE_DEVICES=0          # pin it (inherited by every worker)
    python vcd_mesh_sweep.py

Each worker streams its own logs + mem_report + RESULT line, and writes its per-config
YAML (results/vcd_mesh_attrib_<plat>_<tag>.yaml).  After all runs this prints a summary
table and writes results/vcd_mesh_sweep_summary.csv.  A config that OOMs/crashes is
recorded as FAILED and the sweep continues.

WORKFLOW: run Phase 1 first (subset sweep), read the table, set WORST_NS to the worst
subset-count, then rerun (Phase 2/3 rows use WORST_NS).  Comment rows out to skip them.
"""
import os
import sys
import csv
import glob
import subprocess

import yaml   # for reading the per-config result YAMLs (no jax)

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "vcd_mesh_mem_attribution.py")
RESULTS_DIR = os.path.join(HERE, "results")

# ── Sweep configuration (edit here) ───────────────────────────────────────────
SIZE = "504,504,504"   # applied to every config below (where 3.31x was measured)
WORST_NS = 128         # Phase 1's worst subset-count; Phase 2/3 rows use this


def cfg(label, **kw):
    """One sweep config.  Defaults match the worker's defaults; override per row."""
    d = {"label": label, "size": SIZE, "mesh": True, "block": False,
         "block_subsets": False, "iters": 1, "num_subsets": None,
         "back_band": None, "forward_band": None, "instrument": False}
    d.update(kw)
    return d


# ── FIX VALIDATION — in-place donated error-sinogram update ───────────────────
# Root cause found: jax holds sharded arrays in internal reference cycles, so the
# out-of-place per-subset error-sinogram update leaked one full view-sharded sino per
# subset (freed only by gc).  Fix: update_error_sinogram updates it in place via buffer
# donation (+ explicit .delete() of the transient delta_sinogram).  These runs confirm
# the sharded (1-device-mesh) peak now matches the no-mesh floor instead of ballooning.
# Each config runs in its own subprocess, so peaks are clean.  The 504³ pair runs first
# (fast ~50 s each) for a quick confirm; the 1008³ pair is the capability check (the
# mesh case used to OOM) and is slower (~7-8 min each) -- Ctrl-C after 504³ if you just
# want the quick result.  Expected: mesh ≈ no-mesh at both sizes (504³ mesh was 25.8 GB,
# should now be ~6.9; 1008³ mesh used to OOM, should now complete near the no-mesh ~50 GB).
CONFIGS = [
    cfg("fix_nomesh_504",  mesh=False, iters=5, size="504,504,504"),
    cfg("fix_mesh_504",    mesh=True,  iters=5, size="504,504,504"),
    cfg("fix_nomesh_1008", mesh=False, iters=5, size="1008,1008,1008"),
    cfg("fix_mesh_1008",   mesh=True,  iters=5, size="1008,1008,1008"),
]

# config key -> worker env var
ENV_MAP = {"size": "VMA_SIZE", "mesh": "VMA_MESH", "block": "VMA_BLOCK",
           "block_subsets": "VMA_BLOCK_SUBSETS", "iters": "VMA_ITERS",
           "num_subsets": "VMA_NUM_SUBSETS", "back_band": "VMA_BACK_BAND",
           "forward_band": "VMA_FORWARD_BAND", "instrument": "VMA_INSTRUMENT"}


def build_tag(c):
    """Reconstruct the worker's YAML tag so we can find the file it wrote."""
    return (f"it{c['iters']}_ns{c['num_subsets']}_mesh{int(c['mesh'])}"
            f"_block{int(c['block'])}_bs{int(c['block_subsets'])}"
            f"_bb{c['back_band']}_fb{c['forward_band']}")


def find_result(c):
    """Load the worker's YAML for config c (platform-agnostic glob); None if absent."""
    matches = glob.glob(os.path.join(RESULTS_DIR, f"vcd_mesh_attrib_*_{build_tag(c)}.yaml"))
    if not matches:
        return None
    newest = max(matches, key=os.path.getmtime)
    with open(newest) as f:
        return yaml.safe_load(f)


def run():
    rows = []
    for i, c in enumerate(CONFIGS, 1):
        env = dict(os.environ)
        for k, ev in ENV_MAP.items():
            env[ev] = str(c[k])
        print(f"\n{'='*78}\n[{i}/{len(CONFIGS)}] {c['label']}   {c}\n{'='*78}", flush=True)
        proc = subprocess.run([sys.executable, WORKER], env=env)

        res = find_result(c)
        if proc.returncode != 0 or res is None:
            status = f"FAILED(rc={proc.returncode})"      # most often an OOM crash
            rows.append({**c, "status": status, "time_ms": None,
                         "peak_mb": None, "bytes_in_use_end_mb": None})
            print(f"  -> {status}; continuing", flush=True)
        else:
            rows.append({**c, "status": "ok", "time_ms": res.get("time_ms"),
                         "peak_mb": res.get("peak_mb"),
                         "bytes_in_use_end_mb": res.get("bytes_in_use_end_mb")})

    _print_summary(rows)
    _write_csv(rows)


def _fmt(x, nd=0):
    return "—" if x is None else (f"{x:.{nd}f}")


def _print_summary(rows):
    print(f"\n{'='*78}\nSUMMARY  (peak vs live_end: small-live/large-peak = freed "
          f"transient or reserved\nworkspace; both-large = live accumulation)\n{'='*78}")
    hdr = (f"{'label':<20} {'mesh':<5} {'ns':>5} {'it':>3} {'blk':<4} {'bsub':<5} "
           f"{'bb/fb':>7} {'time_ms':>9} {'peak_MB':>9} {'live_MB':>9}  status")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        band = f"{r['back_band']}/{r['forward_band']}"
        print(f"{r['label']:<20} {str(r['mesh']):<5} {str(r['num_subsets']):>5} "
              f"{r['iters']:>3} {str(r['block']):<4} {str(r['block_subsets']):<5} "
              f"{band:>7} {_fmt(r['time_ms']):>9} {_fmt(r['peak_mb'],1):>9} "
              f"{_fmt(r['bytes_in_use_end_mb'],1):>9}  {r['status']}")


def _write_csv(rows):
    path = os.path.join(RESULTS_DIR, "vcd_mesh_sweep_summary.csv")
    fields = ["label", "size", "mesh", "num_subsets", "iters", "block",
              "block_subsets", "back_band", "forward_band", "instrument",
              "time_ms", "peak_mb", "bytes_in_use_end_mb", "status"]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"\nWrote {path}")


if __name__ == "__main__":
    run()
