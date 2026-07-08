# Current forward plan

(Originally the post-sharding forward plan; renamed 2026-07-08 -- this is the EVOLVING
running list of open work, at plans/current_plans.md.)

**Created 2026-07-03; updated 2026-07-08** (after the projector-kernel campaign on
`greg/kernel_investigation`).  Open items carried forward now that the core sharding
(ParallelBeam + all geometries), the MAR/preprocessing sharding, and the large-problem memory
work have shipped on `greg/shard_profiling`.  This is the running "what's left" list; detail
lives in the source docs cited per item (`mar_refactor_plan.md`,
`sharding_implementation_plan_v3.md` §4/§5/§6, `sinogram_sharding.md`).
Roughly ordered by likely value; none is blocking.

---

## 0.5 Done on `greg/kernel_investigation` (2026-07-07/08) — the projector-kernel campaign

Item 3's "forward-kernel internals (future project)" became a full campaign; record:
`plans/projector_kernels/fwd_back_findings.md`.  In brief:

- **TilePolicy** (`model.tiles`): every projector batching/banding knob + the kernel-algorithm
  flags consolidated into one per-layout selection method, read late-bound.
- **Sorted channel reduction** for the GPU forward horizontal fans (the scatter's colliding
  atomic adds were ~the whole forward kernel cost).  Parallel/cone/multiaxis enable it;
  TRANSLATION measured a 4.5–6.5× collision-cliff LOSS at its real detector shapes and keeps
  the scatter — three guard constants in `projectors.py` encode the measured crossovers.
- **Stacked back gather** (parallel only; measured a composition no-op behind the vertical
  fans of the other three geometries — three separate confirmations).
- **DRY fan kernels**: the trapezoid tap machinery lives once in `projectors.py`
  (`horizontal_fan_project` / `horizontal_fan_back` / `vertical_fan_band_gather`).
- **Concrete scatter centers** — the horizontal-fan rounding-bug fix (parallel's compiled
  programs are round-free; the vertical fans' per-slice rounds are documented accepted risk:
  `plans/bugs_and_artifacts/jax rounding bug/phase_d_design.md`).  En route: the
  eager-gather lesson (one eager array op per wrapper call cost VCD +35%; `lessons.md` §3).
- **Scoreboard (H100, 1024³ n=1, campaign start → end):** parallel forward 35.0 → 7.9 s,
  parallel back 18.2 → 10.5 s, cone forward 41.5 → 18.8 s; multiaxis forward 1.2–1.4×;
  VCD neutral; memory flat at capacity (cone 1024³ −3.2 GB).
- **Docs consolidated** into top-level `plans/` (this file's new home; `plans/README.md`).

**New opens from the campaign** (carried in §3 below): nightly memory-gate acks; the VCD
host-dispatch-bound observation; optional refinements (sort-permutation caching, extending
the collision guard to parallel/cone).

## 0. Done during PR prep (2026-07-03)

- **MAR test coverage** (`tests/sharding/test_mar.py` — was zero): `_argmin_3d` vs flat argmin incl.
  tie cases; corrected-sino 1-vs-3-device consistency with view AND slice padding (tol 1e-3 — the BH
  constraint SELECTION is discretely sensitive to reduce-order noise); the forced-constraint path
  (`tolerance=1e10` — where the flat-index bugs lived); the empty-plastic `ValueError` guard; the
  `recon_plastic_metal` `output_sharded` contract (seeded partitions).
- **`vcls.py` eager norms** (the last `jnp.linalg` survey stragglers): per-view normalize +
  projection folded into jitted helpers (`_normalize_by_norm` / `_normalize_and_project`); also
  collapsed a historical double-eps on the `recon_i` path (~1e-12).

## 1. Size-adaptive memory knobs (biggest near-term win)

The recon's peak on large multi-GPU problems is now set by a couple of hardwired batch/granularity
constants that were tuned for small volumes.  Both should become **size-adaptive but hardware-independent**
(a function of the *problem*, identical on any device count — else results would drift with hardware and
break the cross-device correctness gating).

- **Adaptive `view_batch_size_for_vmap`.**  **LARGELY DONE via the TilePolicy (2026-07-07):**
  the knob is now per-op (forward keeps a flat OOM-safe cap 128; back single-vmaps its
  per-device shard, cap 512 sharded / 128 single-device), selected per device layout and read
  late-bound (the stale-binding bug that motivated this item is fixed).  RESIDUAL (optional):
  true size-adaptive scaling and divisor-of-shard sizing to avoid ragged tail batches —
  revisit only if a measured case wants it.
- **Size-only adaptive `partition_sequence` / starting granularity** (`sharding_implementation_plan_v3.md`
  §5 P1-design).  The granularity-1 first VCD iteration updates the whole recon in one subset → the biggest
  subset-domain arrays.  Deriving the coarsest starting granularity from voxel count (skip granularity 1 for
  large problems) shrinks those.  **STUDIED 2026-07-05, findings pending team review** (full record:
  `plans/partition_sequence/partition_sequence_plan.md`, PROPOSED-defaults + metric-caveat
  sections): a fine flat tail (finer converges faster per iteration = VCD is coordinate descent; the
  old `[0,2,4,6,7]` coarse ramp added ~0 convergence but a gran-1 memory spike + extra compiles)
  SUBSUMES this memory item — it never uses a coarse granularity, so peak sits at the fixed-array
  floor at every size, no adaptive START needed for the default.  Two proposed sequence options
  (**`[7]`** flat-128, or **`[4, 7]`** as a cheap coarse-start hedge); plus raise **`max_iterations`
  from 15 into the ~25–50 range** (the 15-cap was strangling the 0.2% stop on hard objects),
  threshold stays 0.2.  An adaptive coarse start remains an OPTIONAL advanced knob.  Not yet
  decided/implemented — team review, then Greg's call.
- These two are the same class of policy (problem-size → memory knob); worth unifying under one place.

## 2. Sinogram weight edge tapering to speed convergence and reduce flash

ConeBeamModel → split_sino_recon() uses a per-detector-row sine filter on sinogram weights to reduce ringing 
artifacts associated with windowing with a rect in detector rows when splitting the full recon into 
two pieces.  Also, observation indicates that objects that extend outside the field of view
lead to slower convergence.  
- Hence we should investigate whether a geometry-adaptive (and possibly data-adaptive) 
filtering of sinogram edges (both in rows and channels) may speed convergence and/or reduce
the 'flash' associated with objects partially outside the FoV.

## 3. Projector / batching profiling and cleanup

- ~~Profile the existing projector performance~~ / ~~Forward-kernel internals (future
  project)~~ — **DONE 2026-07-07/08 as the projector-kernel campaign** (§0.5 above; full
  record `plans/projector_kernels/fwd_back_findings.md`).  Forward is no longer the dominant
  projector cost (parallel fwd is now ~1.3× FASTER than back at 1024³ n=1); the residual
  kernel headroom (forward sorted-reduce and back gather both sit ~10× above compute-only
  bounds) is custom-kernel territory, deferred.
- ~~Simplify the sparse-projector batching machinery~~ — **CLOSED 2026-07-04 with the code
  UNCHANGED**: the full investigation (census, balanced-batching v2, windowed-read patch, band
  pixel-width tuning — each measured and retired; record in
  `plans/projector_batching/batching_refactor_design.md`) established the scan/map/vmap nest
  is load-bearing piece-by-piece and the batch constants are effectively optimal.  Repeated hard
  lesson: driver-level wins (band −10% at width 2016 on H100; 1.4–1.5× on CPU) did NOT survive the
  full recon path on either platform — micro benchmarks of shape-dependent kernel effects don't
  compose; the full path is the arbiter.  Durable constants for the adaptive-knob work: flat GPU
  vmap-width knee; isolated-driver k ≈ 1.0 forward / 1.9 back.
- **Opens from the campaign (2026-07-08):**
  - **Nightly memory-gate acks** for the deliberate time-for-memory trades: parallel fwd
    small cells (+27–58% rel, ≤0.65 GB), cone fwd 1024³ (+9.6/29/46% at n=1/2/4), mid-size
    fwd cells (+~0.5 GB, materialized scatter centers + chunk concat), 513-class back +8%.
    Also confirm `greg/kernel_investigation` is nightly-tracked (one legal fingerprint-flip
    cycle at rounding ties when the centers change lands).
  - **VCD is host-dispatch-bound at interactive sizes** (measured: 200³ VCD is ~95% host
    time; device kernels are ~0.1 s of a ~2 s recon).  A future speed item with large
    headroom: reduce per-subset dispatches / host syncs (e.g. jit a whole subset update or
    iteration; the concrete-centers plumbing supports precomputed-per-partition centers if
    that jit ever lands).  cProfile is the instrument (`lessons.md` §3/§5).
  - **Optional refinements:** cached per-(view, pixel-batch) sort permutations for the
    sorted reduce (possible now that centers are concrete); extend the collision-ratio
    guard to parallel/cone if very wide detectors with modest pixel batches become real.
  - **Residual rounding-bug risk:** the six vertical-fan per-slice round sites keep the
    in-jit precondition (not materializable) — accepted + monitored
    (`plans/bugs_and_artifacts/jax rounding bug/phase_d_design.md` §7).
- **Minor opens:** `configure_devices`/`use_gpu` unification; forward pixel-batch default.

## 4. MAR Phase 3 — subsample / speed up the BH model fit

From `mar_refactor_plan.md` Phase 3.  **This is now a SPEED-only item**: the fit's memory was solved by
Phase 2 (the `HtH`/`Hty` inner products and constraint argmins run on the view-sharded sinograms), so
the fit cannot OOM — subsampling would only reduce its time.  The OSQP beam-hardening fit is
statistical, so it *could* run on a subsample — but a **uniform** view/stride subsample is wrong: the
model is only identifiable from pixels spanning diverse **metal path length**, which are sparse in a
mostly-plastic object.
- Needs **metal-thresholded targeted subsampling** (stratify by estimated metal magnitude; keep high-metal
  pixels + a plastic sample).
- Cheap independent win to fold in: **cache each `H` column once** instead of the O(num_cols²) recompute.
- **A/B the estimation in isolation first** (fitted `theta` + corrected recon, full vs subsample; sweep the
  subsample size/strategy for the knee) before wiring into the loop.  Not byte-identical by design; gate on
  the corrected recon within a documented tolerance.

## 5. Device-count / communication policy

- **Choose-N-vs-communication policy** (`sharding_implementation_plan_v3.md` §5): when does adding a device
  pay vs. its comms cost?  Includes the **CPU-cluster auto-sharding policy** (real-cluster perf + a
  virtual-vs-real-CPU topology rule).
- **Auto device-count basis — recon-slices vs sino-views** (§6, OPEN).  `_auto_device_count` trims on the
  recon-**slice** axis, but projection compute lives on the **view-owners**, so the slice axis is the wrong
  proxy for "does this device do real work."  Revisit the basis (likely views, or both) as part of choose-N.

## 6. Geometry fidelity

- **Multiaxis vertical-fan `1/cos(elevation)` path-length factor — OPEN** (§6).  Vertical fan uses
  `scaling = 1.0`; for elevation ≠ 0 / non-unit slice aspect the absolute magnitude is self-consistent
  (adjoint holds) but not anchored to a reference.  The right factor must be **derived from the multiaxis
  path length** (the detector is ⟂ to the tilted ray — no cone-style incidence obliquity), not copied from
  cone.  Take up as a separate change with a **physical-fidelity gate** (forward-project a known object vs an
  analytic line integral).  Acceptable as-is for an MBIR initializer.

## 7. Performance (deferred — only if it ever matters)

- **B4.5 — band-kernel GPU cost** (§4).  The band (reduce-scatter) back kernel is ~2.25× slower than the
  rolled-pixel kernel on GPU, and the two are platform-opposite (CPU likes band, GPU likes pixel).  Multi-
  device back doesn't pay in *time* until ≥3 GPUs; VCD stays monotonic because the forward masks it.
  Deferred (sharding is the capacity tool, not a back-time lever).  **Alternative axis:** shard the sinogram
  by **detector row** instead of by view, aligning the sino's sharded axis with the recon's slice sharding →
  back projection becomes mostly-local (a footprint halo) instead of a view-reduce.  Parked; full analysis in
  `plans/sharding/sinogram_sharding.md`.
- **Prox-map (PnP) prior under sharding** — revisit only if a plug-and-play-at-scale need appears (§5).

## 8. Robustness / cleanup

- **Multi-GPU OOM that *hangs* → catchable error** (§5, STILL OPEN).  A GPU stuck in the BFC retry loop never
  reaches the NCCL rendezvous → "Acquire clique" timeout, so no exception is raised and the OOM hint never
  prints (the exact 2048³/8 cone case).  Converting the hang into a clean error is a bigger allocator/
  collective-timeout change; left as a separate follow-on.
- **Deferred docs-cleanup pass** (§5): the remaining unresolved Sphinx py-xrefs that silently render as
  plain `<code>` (no warning).  Detect by building the HTML and grepping for `<code class="xref py …">` not
  wrapped in `<a>`; fix per case (qualify / correct target / document the target / downgrade to literal).
- **Suite tidiness** (§6): seed the remaining unseeded-`np.random` tests; a pre-merge
  `import mbirjax`-before-`jax` sweep; public `shard_*` / `gather_*` wrappers.
- **>2^31 audit sweep**: grep for remaining flat-index / count-unsafe ops on full-size arrays
  (`argsort`, `searchsorted`, large `cumsum` indices, `nonzero`), applying `lessons.md` §4 (the
  `argmin`/`np.prod`/histogram-count instances are fixed; this closes the class).
