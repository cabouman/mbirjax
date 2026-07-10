# Current forward plan — goals for the next release

(The EVOLVING running list of open work, at `plans/current_plans.md`.  Rewritten
2026-07-10 to read forward-looking; completed campaigns live in their own findings docs,
cited where their results guide the work below.  Roughly ordered by likely value.)

## Goals for the next release

* **New padding** (§1) — per-end axial extension in `auto_set_recon_geometry`, the
  split_sino_recon h_recon formula (+ alignment opt-in, taper retirement), lateral
  truncation detect-and-warn.
* **New default partition sequence** (§2) — image-quality experiments to justify
  skipping the 1-subset case by default, plus the `max_iterations` raise; runs AFTER
  the padding lands, since padding changes both convergence and the metrics.
* **Possible further performance improvements** (§3) — the two quantified headroom
  pools: VCD host dispatch, and GPU kernel memory-access patterns.
* **MAR: cache H** (§4).
* **Capture the projector-kernel design in the dev docs** — DONE 2026-07-10: new
  "Projector kernel design" section in `docs/source/dev_sharding_overview.rst`
  (review welcome); the measured record stays in
  `plans/projector_kernels/fwd_back_findings.md`.
* **Cleanup** (§6).

---

## 1. Flash remediation: padding and the split seam

**State:** investigation complete; remedies designed and ready for discussion →
implementation → testing.  The synthesis with rationale, equations, implementation
sketches, and pros/cons is `plans/flash_remediation/phase_2d_remedies.html` (published
at `/depot/bouman/www/mbirjax/flash_remediation/`); the plan of record is
`plans/flash_remediation/flash_remediation_plan.md`.

**Forward, in implementation order:**

1. **Cone-beam per-end axial extension in `auto_set_recon_geometry`** — E_top/E_bot from
   the detector row edges (det_row_offset-aware; helical ends attach at z_max/z_min),
   R from the recon grid; `scale_recon_shape` stays a pure scaler + warns on
   uncompensated lateral growth.  Open implementation check: R = RoR-mask radius vs grid
   half-diagonal (what does the projector actually update?).
2. **split_sino_recon**: h_recon = ceil(h_sino·(1+R/SID)·ρ) + 2 with ρ =
   δ_row/(mag·δ_slice) as the default; `align_split_grid` opt-in (sub-slice grid shift —
   alignment is NOT reachable by index choice at ρ=1); retire the taper in the same
   change (it fails at 8× downsampling — the shipped default visibly stripes there).
3. **Lateral truncation detect-and-warn** via the `_get_sino_indicator` support mask
   already computed in `auto_set_regularization_params` (support touching the edge
   channels ⇒ truncated; free, no new threshold).  Deliberately NO auto-padding.
4. **Re-baseline the regression dashboards** in the same change (default shapes grow and
   values shift, for the better) + a release note.
5. **Phase 3 validation on real scans** before the defaults ship: SiC (axial), z62
   (radial), BGA (severe lateral, stretch).
6. Later: the analogous per-end bounds for translation and multiaxis-parallel.

## 2. Partition sequence and iteration defaults

**State:** done at the batch level.  The prerelease keeps the old default sequence,
with an easier knob for users to skip the 1-subset (granularity-1) case on large recons
(the change is deliberately conservative).

**Forward (after §1 lands — the padding changes both convergence and the metrics,
so the deciding experiments should run with it in place):** make skip-1 the default
for the next release of main — gated on
**image-quality comparison experiments** (old default vs the proposed sequences on
representative objects; visual quality first).  The 2026-07-05 study
(`plans/partition_sequence/partition_sequence_plan.md`, PROPOSED-defaults + metric-caveat
sections) supplies the candidates and the durable guidance:

- Candidate sequences: **`[7]`** (flat-128) or **`[4, 7]`** (a cheap coarse-start hedge).
  A fine flat tail converges faster per iteration AND never uses a coarse granularity, so
  the old granularity-1 memory spike disappears without any size-adaptive starting
  policy — an adaptive coarse start stays an optional advanced knob only.
- Raise **`max_iterations` from 15 into the ~25–50 range** (the 15-cap strangles the
  0.2% stop on hard objects); the threshold stays 0.2.
- Metric caveat to respect in the experiments: FoV-truncation flash inflates NRMSE and
  the change-% stop (see §1) — compare on cropped/remediated metrics or visually.

Supporting experiment (drafted, not run): the real-data partition-sequence convergence
study (`plans/experiments/partition_sequence/`), which also tests the
monotone-non-decreasing granularity theory.

## 3. Performance

Two quantified headroom pools guide any future performance work (both from the
2026-07 profiling/kernel campaigns; details in
`plans/projector_kernels/fwd_back_findings.md` and `plans/` profiling docs):

- **VCD is host-dispatch-bound at interactive sizes**: a 200³ VCD recon is ~95% host
  time (~0.1 s of device kernels in a ~2 s recon).  The large-headroom item is reducing
  per-subset dispatches / host syncs — e.g. jit a whole subset update or iteration (the
  concrete-centers plumbing supports precomputed-per-partition centers if that jit
  lands).  cProfile is the instrument, not kernel benches (`lessons.md` §3/§5).
- **GPU kernel headroom is ~10×**: the forward sorted-reduce and back gather both sit
  roughly 10× above compute-only bounds — compute is ~10% of kernel time — and ncu
  attributes it to memory ACCESS PATTERNS, not HBM bandwidth.  Closing it is
  custom-kernel territory; deferred, but the ceiling is known and large.

Small optional refinements (do only if a measured case wants them): cached
per-(view, pixel-batch) sort permutations for the sorted reduce; extend the
collision-ratio guard to parallel/cone if very wide detectors with modest pixel batches
become real; true size-adaptive `view_batch` scaling / divisor-of-shard sizing to avoid
ragged tail batches.

Deferred unless a driving workload appears:

- **Sinogram-by-row sharding** (aligns the sino's sharded axis with recon slices → back
  projection becomes mostly-local): prototyped and parked; benefit is
  GPU-communication-only (`plans/sharding/sinogram_sharding.md`).
- **Prox-map (PnP) prior under sharding.**

## 4. MAR: cache H

Reframed 2026-07-10 (from `mar_refactor_plan.md` Phase 3, which is SPEED-only — the
fit's memory was solved in Phase 2):

- **The direction is caching**: compute each `H` column once instead of the
  O(num_cols²) recompute — a cheap, self-contained win with no statistical questions.
- **Subsampling is deprioritized**: uniform view/stride subsampling is wrong for this
  fit (the model is identifiable only from pixels spanning diverse metal path length,
  which are sparse in a mostly-plastic object), and metal-thresholded stratification is
  exactly the finicky-threshold pattern §1's remedies work deliberately avoided.  If it
  is ever revisited: A/B the estimation in isolation first (fitted `theta` + corrected
  recon, full vs subsample) and gate on the corrected recon within a documented
  tolerance — not byte-identical by design.

## 5. Device-count policy — simple and robust

Preference (2026-07-10): a simple, robust rule — even if suboptimal — over a tuned
choose-N-vs-communication model; this area is potentially finicky for a modest payoff.

- **Concrete first step:** fix the auto-device-count basis.  `_auto_device_count` trims
  on the recon-SLICE axis, but projection compute lives on the VIEW-owners, so the slice
  axis is the wrong proxy for "does this device do real work" — switch the basis to
  views (or both).  Small, clearly right, and independent of any cost model.
- The full choose-N policy (when does adding a device pay vs its comms cost, incl. the
  CPU-cluster topology rule) stays deferred unless a real workload demands it
  (`sharding_implementation_plan_v3.md` §5/§6).

## 6. Miscellaneous / cleanup

- **Suite tidiness**: seed the remaining unseeded-`np.random` tests; a pre-merge
  `import mbirjax`-before-`jax` sweep; public `shard_*` / `gather_*` wrappers; simplify 
  tests and reduce time on tests.
- **>2^31 audit sweep**: grep for remaining flat-index / count-unsafe ops on full-size
  arrays (`argsort`, `searchsorted`, large `cumsum` indices, `nonzero`) per
  `lessons.md` §4 — the known instances are fixed; this closes the class.
- **Minor API opens**: `configure_devices`/`use_gpu` unification; the forward
  pixel-batch default.
- **Residual rounding-bug risk (monitor only)**: the six vertical-fan per-slice round
  sites keep the in-jit precondition — accepted + monitored
  (`plans/bugs_and_artifacts/jax rounding bug/phase_d_design.md` §7).

## 7. Possible future direction: multi-resolution reconstruction (post-next-main)

Coarse-to-fine MBIR: reconstruct at binned resolution(s), upsample as the init for the
next-finer level.  Added 2026-07-10 (Greg); investigation-first, not for the next main.

**Rationale.**  VCD is coordinate descent, so low-frequency corrections propagate slowly
at fine resolution; a coarse level handles them at ~1/8 the voxels (and ~1/8 the sino if
rows/channels bin).  The partition-sequence study's finding that the coarse-GRANULARITY
ramp added ~nothing supports this framing: granularity coarsening changes the update
grouping but still pays full-resolution cost per iteration — GRID coarsening is the
principled low-frequency accelerator that ramp was groping at.

**Where it pays (cost model):** large problems and cap-bound hard objects.  At small
sizes VCD is ~95% host-dispatch-bound (§3), so coarse levels cost fixed per-iteration
host overhead + a compile per shape, not 1/16 flops — don't expect interactive-size wins.

**Null hypothesis to kill first:** coarse-MBIR init must beat FDK/FBP init (one cheap
full-resolution call) on wall-clock-to-matched-quality.  Expected to win only under
heavy noise / sparse views / truncation-corrupted FBP, or where the 0.2%-stop drags.

**The matching problems (the real work) and what softens them:**
- *Volumes:* offsets are in ALU, hence scale-invariant across levels (verified on Lilly:
  −1.98 mm = −3.9 rows at 4× = −1.95 rows at 8×); the upsample must map voxel centers
  PHYSICALLY (the `recon_ijk_to_xyz` chain) since `auto_set_recon_geometry`'s ceils
  break exact shape nesting.  Init-only use makes residual sub-voxel phase error cheap
  (a few iterations, not an artifact) — but do it right; see the 2c misalignment lesson.
- *Parameters:* sharpness/snr_db are scale-free; `auto_set_sigma_y` already carries a
  pixel-pitch^0.5 consistency factor — per-level auto-regularization may be most of the
  answer.  Open question: qGGMRF edge-threshold scale consistency (test: coarse solution
  ≈ downsampled fine solution?).
- *Data per level:* bin the LOG sinogram linearly — provably consistent (it is exactly
  the projection of an axially/laterally smoothed object; the flash-remediation round-3
  result).  No per-level re-preprocessing.

**Pilot (before any library code):** the Lilly 8× workhorse + one large synthetic; A/B
wall-clock-to-matched-quality across {zero init, FDK init, 2-level, 3-level}, compiles
counted honestly, metrics flash-cropped (§2 caveat).  The flash-remediation Lilly
infrastructure (ds4/ds8 pipelines, converged references, seam/region metrics) is
directly reusable.  If the pilot wins: implementation is a `split_sino_recon`-shaped
driver (~100 lines — `copy_ct_model` per level, physical-coordinate upsample, per-level
auto-regularization, loose stopping on coarse levels).

---

**Recently completed (records live elsewhere):** the projector-kernel campaign
(design → `docs/source/dev_sharding_overview.rst` "Projector kernel design"; measured
record → `plans/projector_kernels/fwd_back_findings.md`); the flash-remediation
investigation Phases 1–2d (→ `plans/flash_remediation/`); multiaxis vertical-fan
path-length factor (shipped); the sparse-projector batching investigation (closed with
code unchanged → `plans/projector_batching/batching_refactor_design.md`).
