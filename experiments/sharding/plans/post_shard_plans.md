# Post-sharding forward plan

**Created 2026-07-03.**  Open items carried forward now that the core sharding (ParallelBeam + all
geometries), the MAR/preprocessing sharding, and the large-problem memory work have shipped on
`greg/shard_profiling`.  This is the running "what's left" list; detail lives in the source docs cited
per item (`mar_refactor_plan.md`, `sharding_implementation_plan_v3.md` §4/§5/§6, `sinogram_sharding.md`).
Roughly ordered by likely value; none is blocking.

---

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

- **Adaptive `view_batch_size_for_vmap`.**  Currently hardwired to **128** (was 512; the forward-projection
  transient scales with it and OOM'd cone 1024³).  Make it scale down with recon size (large recons want it
  smaller; small recons are unaffected), and pick a **divisor of the per-view-shard view count** so there is
  no ragged tail batch.  _(TODO note is in `tomography_model.py`; supersedes the hardwired 128.)_
- **Size-only adaptive `partition_sequence` / starting granularity** (`sharding_implementation_plan_v3.md`
  §5 P1-design).  The granularity-1 first VCD iteration updates the whole recon in one subset → the biggest
  subset-domain arrays.  Deriving the coarsest starting granularity from voxel count (skip granularity 1 for
  large problems) shrinks those.  **STUDIED 2026-07-05, findings pending team review** (full record:
  `experiments/partition_sequence/partition_sequence_plan.md`, PROPOSED-defaults + metric-caveat
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

- **Profile the existing projector performance**: Use jax/perfetto/tensorboard/nvidia tools to identify
  memory and computational bottlenecks and make a plan to improve performance.
- ~~Simplify the sparse-projector batching machinery~~ — **CLOSED 2026-07-04 with the code
  UNCHANGED**: the full investigation (census, balanced-batching v2, windowed-read patch, band
  pixel-width tuning — each measured and retired; record in
  `experiments/projector_batching/batching_refactor_design.md`) established the scan/map/vmap nest
  is load-bearing piece-by-piece and the batch constants are effectively optimal.  Repeated hard
  lesson: driver-level wins (band −10% at width 2016 on H100; 1.4–1.5× on CPU) did NOT survive the
  full recon path on either platform — micro benchmarks of shape-dependent kernel effects don't
  compose; the full path is the arbiter.  Durable constants for the adaptive-knob work: flat GPU
  vmap-width knee; isolated-driver k ≈ 1.0 forward / 1.9 back.
- **Forward-kernel internals (future project, noted 2026-07-04):** forward projection is ~2× back's
  time on GPU and 3–4× on CPU — the dominant projector cost.  Improving the forward kernel itself
  would also re-open the batch-width trade-offs above.
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
  `.claude/sinogram_sharding.md`.
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
