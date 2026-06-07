"""
experiments/sharding/scaling_tests/vcd_mesh_mem_attribution.py
──────────────────────────────────────────────────────────────
Attribute the 1-device-MESH VCD memory overhead, and test WHY it happens.

CONTEXT: on one H100 the sharded (1-device *mesh*) VCD path peaks at ~3.31x the
plain single-device peak at 504^3 (24.6 vs 7.4 GB) and OOMs at 1008^3 where the
single-device path fits (50.5 GB).  Reading the code:
  * Each jitted projector call is internally memory-bounded (it batches pixels by
    2048 and views by 512), so a single call does NOT scale its transient with the
    number of pixels.  => the overhead is NOT one big unbounded kernel.
  * BUT the single-device projector bodies call block_until_ready() per batch
    (tomography_model.py:1071, :1311) -- backpressure that serializes execution and
    bounds the in-flight working set.  The SHARDED path has NONE: run_per_device is
    async by design (thread_execution.py:58-62), the band loops don't block between
    bands, the per-subset block_until_ready are commented out, and the alpha fix
    removed the last per-subset host sync.  So the sharded path runs fully async:
    bands, subsets, and the setup ops overlap, and their transients coexist.

HYPOTHESIS (the leading one): the mesh overhead is JAX "running ahead of itself" --
async dispatch with no backpressure -- not a structural per-call blowup.  This file
sets up two single-variable tests to confirm/refute it and to point at the fix.

FINDING (first GPU pass, 504^3): at MAX_ITERATIONS=1 there is NO blowup -- mesh peak
6.9 GB ~= no-mesh 7.8 GB, and BLOCK/band change nothing.  The 3.31x only appears at 5
iters (mesh 24.6 GB) while the no-mesh peak stays flat in iters.  => the overhead
ACCUMULATES PER ITERATION in the mesh path (1 iter does NOT reproduce it -- my earlier
default was a ruler error).  So run the matrix at MAX_ITERATIONS>=5.  An iteration sweep
(1..5) gives the growth curve, and bytes_in_use(end) vs peak distinguishes live-array
accumulation (both large) from reserved compilation workspace (peak large, live small).

TWO AXES (do not conflate): a VCD "iteration" is one PASS over one PARTITION, and a
partition has some number of SUBSETS.  More passes != more subsets -- a single pass with
128 subsets runs 128 async subset updates, while a pass with 1 subset runs one full-FOV
update.  NUM_SUBSETS forces a single partition of a chosen subset-count, reused every
pass, so you can sweep subset-count at fixed passes (isolates PER-SUBSET accumulation)
and passes at fixed subset-count (isolates PER-PASS accumulation) independently.  This is
the likely real axis if the cause is the async, unblocked per-subset loop.

────────────────────────────────────────────────────────────────────────────────
TEST 1 — BACKPRESSURE (the decisive one).  Set BLOCK_PROJECTIONS=True to wrap
  model.sparse_back_project / sparse_forward_project so each returns only after
  block_until_ready().  Those two methods carry the per-subset loop AND the full-FOV
  setup ops (direct_recon / forward_project / compute_hessian_diagonal all route
  through them), so this serializes the whole pipeline -- with NO library edit.
    Async run-ahead is CONFIRMED iff, vs the BLOCK_PROJECTIONS=False baseline,
    peak DROPS and wall-time RISES.  (A static working-set problem would move time
    but not peak.)

TEST 2 — BAND SIZE (the discriminator).  Set BACK_BAND / FORWARD_BAND to force the
  slice-band length (overriding the default cap, which is ~full at 504^3).  Sweep
  {None, 64, 16} on the 1-device mesh and watch peak:
    * smaller band -> LOWER peak  => the slice-axis working set is the lever
      (fix = make the band a real memory budget at small sizes / n_dev=1).
    * smaller band -> HIGHER or FLAT peak  => async overlap (more in-flight band
      ops), corroborating TEST 1.

────────────────────────────────────────────────────────────────────────────────
HOW TO READ peak: mem_report prints `peak_bytes_in_use` (a high-water mark since
process start), so the "after recon" value is the run's peak regardless of
INSTRUMENT.  `bytes_in_use` + the live-array list explain the resident set.

RECOMMENDED MATRIX (edit the config, rerun; each writes its own YAML):
  A. reference     : MESH=False                              -> no-mesh anchor (~7.4 GB @504^3)
  B. mesh baseline : MESH=True                               -> reproduce the ~3.3x
  C. backpressure  : MESH=True, BLOCK_PROJECTIONS=True       -> TEST 1
  D. band sweep    : MESH=True, BACK_BAND=FORWARD_BAND=64    -> TEST 2
                     MESH=True, BACK_BAND=FORWARD_BAND=16
Keep INSTRUMENT=False for A–D (the phase-trace adds its own blocking, which would
confound the backpressure baseline).  Turn INSTRUMENT=True only when you want the
per-phase peak curve to localize WHICH op dominates (use a debugger inside that op
to name the arrays).

No CLI args by design — edit the config block below.
"""
import os
import gc
import time

# Import mbirjax before jax (it sets the XLA device flags on import); then jax is
# safe to import at module scope so mem_report can use it.
import mbirjax
import jax
import numpy as np

import scaling_common as sc


# ── Run configuration (edit here) ─────────────────────────────────────────────
SIZE = (504, 504, 504)   # where 3.31x was measured; drop to (252,252,252) for fast iteration
MAX_ITERATIONS = 5       # IMPORTANT: mesh peak GROWS with iterations (504^3: 1 iter=6.9GB
                         # vs 5 iter=24.6GB), so 1 iter does NOT reproduce the overhead.
                         # Use >=5 to see it.  (No-mesh peak is ~flat in iterations.)
NUM_SUBSETS = None       # None = model's default multi-granular schedule.  Set to an int
                         # to force ONE partition with that many subsets, used every pass
                         # (granularity=[N], partition_sequence=[0]).  Sweep this at fixed
                         # MAX_ITERATIONS to isolate per-subset vs per-pass accumulation.
SEED = 0
SHARPNESS = 1.0

MESH = True              # True = 1-device mesh (sharded path); False = plain single-device
BLOCK_PROJECTIONS = False  # serialize the projectors with block_until_ready()
BLOCK_SUBSETS = False    # serialize the SUBSET STATE: block (flat_recon, error_sinogram,
                         # ...) after each subset update (backpressure at the right level
                         # if the accumulation is the async per-subset state chain)
FORWARD_BAND = None      # TEST 2: force forward slice-band length (None = default cap)
BACK_BAND = None         # TEST 2: force back slice-band length (None = default cap)

NONCONST_WEIGHTS = False # pass random positive weights -> exercises the per-subset
                         # weights*error_sinogram transient (and its cleanup .delete)
POSITIVITY = False       # set positivity_flag -> exercises the positivity-branch
                         # delta_sinogram recompute
MEMPROBE = False         # leak localizer: print live big-view-sharded-sino count at each
                         # sub-step of vcd_subset_updater ([mp <tag>] N).  Compare a
                         # NONCONST_WEIGHTS=true vs =false run: the tag whose count climbs
                         # across subsets only for non-const is the leaking op.

INSTRUMENT = False       # auto-wrap big phases with mem_report (adds blocking — keep False for A–D)
TOP_N = 15               # how many of the largest live arrays to list


# --- env overrides (set by the sweep orchestrator vcd_mesh_sweep.py; leave them unset to
# use the hand-edited values above).  This is an orchestration mechanism — not for typing
# by hand — so the by-hand "edit a constant and rerun" workflow above is unaffected. ---
def _as_bool(s):
    return str(s).strip().lower() in ("1", "true", "yes", "on")


def _as_opt_int(s):
    s = str(s).strip()
    return None if s.lower() in ("none", "") else int(s)


if os.environ.get("VMA_SIZE"):
    SIZE = tuple(int(x) for x in os.environ["VMA_SIZE"].split(","))
if os.environ.get("VMA_ITERS"):
    MAX_ITERATIONS = int(os.environ["VMA_ITERS"])
if "VMA_NUM_SUBSETS" in os.environ:
    NUM_SUBSETS = _as_opt_int(os.environ["VMA_NUM_SUBSETS"])
if "VMA_MESH" in os.environ:
    MESH = _as_bool(os.environ["VMA_MESH"])
if "VMA_BLOCK" in os.environ:
    BLOCK_PROJECTIONS = _as_bool(os.environ["VMA_BLOCK"])
if "VMA_BLOCK_SUBSETS" in os.environ:
    BLOCK_SUBSETS = _as_bool(os.environ["VMA_BLOCK_SUBSETS"])
if "VMA_FORWARD_BAND" in os.environ:
    FORWARD_BAND = _as_opt_int(os.environ["VMA_FORWARD_BAND"])
if "VMA_BACK_BAND" in os.environ:
    BACK_BAND = _as_opt_int(os.environ["VMA_BACK_BAND"])
if "VMA_INSTRUMENT" in os.environ:
    INSTRUMENT = _as_bool(os.environ["VMA_INSTRUMENT"])
if "VMA_NONCONST_WEIGHTS" in os.environ:
    NONCONST_WEIGHTS = _as_bool(os.environ["VMA_NONCONST_WEIGHTS"])
if "VMA_POSITIVITY" in os.environ:
    POSITIVITY = _as_bool(os.environ["VMA_POSITIVITY"])
if "VMA_MEMPROBE" in os.environ:
    MEMPROBE = _as_bool(os.environ["VMA_MEMPROBE"])


def _nbytes(a):
    try:
        return int(a.nbytes)
    except Exception:       # noqa: BLE001 — some live objects may not expose nbytes
        return 0


def mem_report(label="", device=None):
    """Print current + peak device bytes and the largest live jax arrays.

    Call at a phase boundary or from the debugger.  Returns peak MB.  On CPU
    `memory_stats()` may be empty (cur/peak show 0) but the live-array inventory
    still works.
    """
    if device is None:
        device = jax.devices()[0]
    try:
        stats = device.memory_stats() or {}
    except Exception:       # noqa: BLE001 — CPU backend may not implement memory_stats
        stats = {}
    cur = stats.get("bytes_in_use", 0) / 1e6
    peak = stats.get("peak_bytes_in_use", 0) / 1e6

    arrays = sorted(jax.live_arrays(), key=_nbytes, reverse=True)
    total = sum(_nbytes(a) for a in arrays) / 1e6

    print(f"\n=== mem_report [{label}] ===")
    print(f"  device bytes_in_use={cur:10.1f} MB   peak_bytes_in_use={peak:10.1f} MB")
    # Full allocator stats: pool_bytes / bytes_reserved / peak_pool_bytes /
    # largest_free_block_bytes / num_allocs distinguish a genuinely-large LIVE working
    # set from a BFC POOL that grew via many alloc/free cycles and stuck (pool ≫ in_use).
    if stats:
        print("  full memory_stats (bytes->MB):")
        for k in sorted(stats):
            v = stats[k]
            if isinstance(v, (int, float)) and "bytes" in k:
                print(f"    {k:<30} = {v / 1e6:12.1f} MB")
            else:
                print(f"    {k:<30} = {v}")
    print(f"  live jax arrays: {len(arrays)}   sum={total:10.1f} MB   (top {TOP_N}):")
    for a in arrays[:TOP_N]:
        mb = _nbytes(a) / 1e6
        shp = tuple(getattr(a, "shape", ()))
        dt = getattr(a, "dtype", "?")
        # NamedSharding (a sharded intermediate) vs SingleDeviceSharding (gathered/plain).
        shd = type(getattr(a, "sharding", None)).__name__
        print(f"    {mb:10.1f} MB  shape={shp}  dtype={dt}  sharding={shd}")
    print("=== end ===\n")
    return peak, cur


def _instrument(model):
    """Wrap the big vcd_recon phases so mem_report fires before/after each.

    Each wrapped method blocks on its result so the peak is realized before we read
    it.  NOTE: this blocking serializes the setup phases, so leave INSTRUMENT=False
    for the backpressure/band tests (it would mask the async baseline).
    get_forward_model_loss is a staticmethod, but a plain function set as an instance
    attribute shadows it and is called without `self`, which is what vcd_recon does.
    """
    phases = ("direct_recon", "compute_hessian_diagonal", "forward_project",
              "vcd_partition_iterator", "get_forward_model_loss")

    def make_wrapper(name, orig):
        def wrapped(*args, **kwargs):
            mem_report(f"before {name}")
            result = orig(*args, **kwargs)
            try:
                jax.block_until_ready(result)   # accepts pytrees (tuples) too
            except Exception:                   # noqa: BLE001 — scalars / non-arrays
                pass
            mem_report(f"after {name}")
            return result
        return wrapped

    for name in phases:
        setattr(model, name, make_wrapper(name, getattr(model, name)))


def _block_projections(model):
    """TEST 1: make sparse_back_project / sparse_forward_project block on their result.

    This reintroduces the backpressure the single-device bodies have and the sharded
    path lacks.  These two methods are the ones the per-subset updater and the
    full-FOV ops call, so wrapping them serializes essentially the whole pipeline
    without touching library code.  create_vcd_subset_updater captures
    `self.sparse_*_project` when recon() runs, so patching here (before recon) takes
    effect inside the subset loop too.
    """
    def make_wrapper(orig):
        def wrapped(*args, **kwargs):
            result = orig(*args, **kwargs)
            jax.block_until_ready(result)
            return result
        return wrapped

    for name in ("sparse_back_project", "sparse_forward_project"):
        setattr(model, name, make_wrapper(getattr(model, name)))


def _block_subsets(model):
    """Backpressure at the subset-state level: block each subset update's outputs.

    Wraps create_vcd_subset_updater so the per-subset updater it returns blocks on its
    full return value -- (flat_recon, error_sinogram, ell1, alpha) -- before the loop
    proceeds.  This serializes the per-subset STATE chain that p3_block (projector
    outputs only) left running async, so it tests whether the accumulating transient is
    that chain.  create_vcd_subset_updater is called inside recon(), so patching here
    (before recon) is picked up.
    """
    orig_create = model.create_vcd_subset_updater

    def wrapped_create(*a, **k):
        updater = orig_create(*a, **k)

        def blocking_updater(*ua, **uk):
            out = updater(*ua, **uk)
            jax.block_until_ready(out)
            return out
        return blocking_updater

    model.create_vcd_subset_updater = wrapped_create


def run_one():
    n_views, _, _ = SIZE
    angles = np.linspace(0, np.pi, n_views, endpoint=False)
    model = mbirjax.ParallelBeamModel(SIZE, angles)
    model.set_params(sharpness=SHARPNESS, verbose=1)   # verbose>=1 prints phase logs
    if NUM_SUBSETS is not None:
        # Force a single partition with NUM_SUBSETS subsets, reused every pass, so
        # subset-count and pass-count can be swept independently.
        model.set_params(granularity=[int(NUM_SUBSETS)], partition_sequence=[0])
    if POSITIVITY:
        model.set_params(positivity_flag=True)
    if MEMPROBE:
        model._vcd_memprobe = True   # enables the per-sub-step leak localizer in vcd_subset_updater

    if MESH:
        # Trivial 1-device mesh -> recon takes the sharded VCD path on one device.
        model.configure_sharding(jax.devices()[:1])
        # Force the slice-band lengths if requested (read by _slice_band_length).
        if BACK_BAND is not None:
            model.back_project_slice_band = BACK_BAND
        if FORWARD_BAND is not None:
            model.forward_project_slice_band = FORWARD_BAND

    if BLOCK_PROJECTIONS:
        _block_projections(model)
    if BLOCK_SUBSETS:
        _block_subsets(model)
    if INSTRUMENT:
        _instrument(model)

    np.random.seed(SEED)
    sino = np.random.rand(*SIZE).astype(np.float32)     # host array; recon shards at entry
    weights = None
    if NONCONST_WEIGHTS:
        weights = np.random.uniform(0.5, 1.5, SIZE).astype(np.float32)  # positive, non-constant

    mem_report("before recon")
    t0 = time.perf_counter()
    recon, _ = model.recon(sino, weights=weights, max_iterations=MAX_ITERATIONS,
                           stop_threshold_change_pct=0.0, print_logs=False)
    jax.block_until_ready(recon)
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    # Collect unreferenced internal arrays first so live_end reflects what recon
    # actually RETAINS (not arrays merely awaiting gc).  peak is a high-water mark, so
    # it is unaffected by gc timing.  (recon/sino are still referenced -> the output
    # volume legitimately stays; sino is a host array, not a live jax array.)
    gc.collect()
    peak_mb, cur_mb = mem_report("after recon (post-gc)")

    del recon, sino
    gc.collect()
    return elapsed_ms, peak_mb, cur_mb


def main():
    plat, _ = sc.detect_platform()
    cfg = (f"MESH={MESH}  BLOCK_PROJECTIONS={BLOCK_PROJECTIONS}  "
           f"BLOCK_SUBSETS={BLOCK_SUBSETS}  "
           f"BACK_BAND={BACK_BAND}  FORWARD_BAND={FORWARD_BAND}  "
           f"NONCONST_WEIGHTS={NONCONST_WEIGHTS}  POSITIVITY={POSITIVITY}  "
           f"INSTRUMENT={INSTRUMENT}  SIZE={SIZE}  iters={MAX_ITERATIONS}  "
           f"num_subsets={NUM_SUBSETS}")
    print(f"platform={plat}  devices={jax.devices()}")
    print(cfg)

    elapsed_ms, peak_mb, cur_mb = run_one()

    print(f"\nRESULT [{plat}]: time={elapsed_ms:.0f} ms   peak={peak_mb:.1f} MB   "
          f"live_end={cur_mb:.1f} MB   ({cfg})")

    # Record per-config so a matrix of runs does not overwrite itself.  max_iterations
    # is in the tag because the mesh peak depends on it (it accumulates per iteration).
    tag = (f"it{MAX_ITERATIONS}_ns{NUM_SUBSETS}_mesh{int(MESH)}"
           f"_block{int(BLOCK_PROJECTIONS)}_bs{int(BLOCK_SUBSETS)}"
           f"_bb{BACK_BAND}_fb{FORWARD_BAND}"
           f"_nw{int(NONCONST_WEIGHTS)}_pos{int(POSITIVITY)}")
    out = {"op": "vcd_mesh_mem_attribution", "platform": plat,
           "size": sc.size_label(SIZE), "max_iterations": MAX_ITERATIONS,
           "num_subsets": NUM_SUBSETS,
           "mesh": MESH, "block_projections": BLOCK_PROJECTIONS,
           "block_subsets": BLOCK_SUBSETS,
           "back_band": BACK_BAND, "forward_band": FORWARD_BAND,
           "nonconst_weights": NONCONST_WEIGHTS, "positivity": POSITIVITY,
           "instrument": INSTRUMENT, "time_ms": elapsed_ms,
           "peak_mb": peak_mb, "bytes_in_use_end_mb": cur_mb}
    sc.save_yaml(os.path.join(sc.RESULTS_DIR, f"vcd_mesh_attrib_{plat}_{tag}.yaml"), out)


if __name__ == "__main__":
    main()
