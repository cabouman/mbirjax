# Current forward plan — goals for the next release

(The EVOLVING running list of open work, at `plans/current_plans.md`.  Rewritten
2026-07-10 to read forward-looking; condensed 2026-07-13 (the Pallas projector-kernel
campaign and the padding/partition-sequence work completed — summarized here with
pointers to their findings docs).  Completed campaigns live in their own findings docs,
cited where their results guide the work below.  Roughly ordered by likely value.
This file should be cleaned periodically to avoid build-up of historical detail.)

## Goals for the next release

* **New padding** (§1) — DONE + real-data validated (per-end axial extension, the
  `split_sino_recon` h_recon formula, lateral detect-and-warn); open tail = dashboard
  re-baseline + release note, and the translation/multiaxis per-end bounds.
* **New default partition sequence** (§2) — DONE: the default is now `[2, 4, 6, 7]`
  (skips the 1-subset case), committed `42c0e23` per the study; open = the
  `max_iterations` raise.
* **Possible further performance improvements** (§3) — the GPU-kernel pool is DONE (the
  Pallas projector-kernel campaign shipped); the VCD host-dispatch pool remains open.
* **MAR: cache H** (§4).
* **Projector kernels in the dev docs** — DONE: `docs/source/dev_projector_kernels.rst`
  (the XLA + Pallas kernel design on one page); measured record in
  `plans/projector_kernels/gpu_headroom_findings.md`.
* **Cleanup** (§6).

---

## 1. Flash remediation: padding and the split seam

**State:** the padding remedies are DONE + real-data validated (2026-07-11/12); open
tail = dashboard re-baseline and the translation/multiaxis extension.  Full record with
rationale, equations, and the per-step validation is
`plans/flash_remediation/flash_remediation_plan.md` (synthesis + pros/cons:
`plans/flash_remediation/phase_2d_remedies.html`, published at
`/depot/bouman/www/mbirjax/flash_remediation/`).

**Done + validated** (commits `a872695` / `fcc0e9e` / `41ecbc2` / `dbc9c3b`): cone-beam
per-end axial extension in `auto_set_recon_geometry` (+ `get_support_radius`, helical-FDK
zero-coverage fix); the `split_sino_recon` h_recon formula with the `align_split_grid`
opt-in and taper retirement; lateral-truncation detect-and-warn (deliberately no
auto-padding); and the NSI auto-geometry cleanup matching the zeiss convention.  Phase-3
validation on real scans is complete — SiC/BGA (axial), Lilly (split), and the lateral
warning (BGA fires, SiC silent, **z62 reattributed as genuinely contained**, Lilly a true
one-sided-truncation positive).

**Remaining:**
- Re-baseline the regression dashboards (default shapes grow and values shift, for the
  better) + a release note; record the regime change as an `annotations.yaml` marker and
  a policy-block padding flag.
- Later: the analogous per-end bounds for translation and multiaxis-parallel.

## 2. Partition sequence and iteration defaults

**State:** DONE — the default partition sequence is now **`[2, 4, 6, 7]`** (a
monotone-non-decreasing ramp that skips the granularity-1 subset, so the old
granularity-1 memory spike disappears with no size-adaptive starting policy), committed
`42c0e23` per the 2026-07 image-quality study.  Candidates, the metric caveats, and the
monotone-granularity theory are in
`plans/partition_sequence/partition_sequence_plan.md`; the drafted real-data convergence
study is `plans/experiments/partition_sequence/`.

**Remaining:** raise **`max_iterations` from 15 into the ~25–50 range** (the 15-cap
strangles the 0.2% stop on hard objects; threshold stays 0.2).  Metric caveat for any
follow-up experiments: FoV-truncation flash inflates NRMSE and the change-% stop (§1) —
compare on cropped/remediated metrics or visually.

## 3. Performance

Two quantified headroom pools guide any future performance work (both from the
2026-07 profiling/kernel campaigns; details in
`plans/projector_kernels/gpu_headroom_findings.md`, `fwd_back_findings.md`, and the
`plans/` profiling docs):

- **VCD is host-dispatch-bound at interactive sizes**: a 200³ VCD recon is ~95% host
  time (~0.1 s of device kernels in a ~2 s recon).  The large-headroom item is reducing
  per-subset dispatches / host syncs — e.g. jit a whole subset update or iteration (the
  concrete-centers plumbing supports precomputed-per-partition centers if that jit
  lands).  cProfile is the instrument, not kernel benches (`lessons.md` §3/§5).
- **GPU kernel headroom — DONE (the Pallas projector-kernel campaign)**: a custom-kernel
  (Pallas/Triton) path now serves the parallel and cone projectors on allowlisted GPUs at
  2–9× the XLA kernels, shipped across single-device and multi-device band paths for both
  back and forward, and soak-validated end-to-end.  It hit the recalibrated prize (2–3×
  custom-kernel) and ended cone's multi-GPU anti-scaling.  Design + measured record:
  `docs/source/dev_projector_kernels.rst` and
  `plans/projector_kernels/gpu_headroom_findings.md`.

Small optional refinements (do only if a measured case wants them): cached
per-(view, pixel-batch) sort permutations for the sorted reduce (now approach A1 in the
headroom plan, with its memory cost quantified there); extend the
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

- **Compile-free memory preflight in `recon()`** (parked 2026-07-10; design agreed with
  Greg after a student's 2-GPU full recon at 1600×1617×1422 spent 32 min in XLA's BFC
  retry loop before surfacing RESOURCE_EXHAUSTED — ~1,900 warning lines; the allocator's
  retry policy is not user-tunable, so fail BEFORE the first doomed allocation).
  Two parts:
  1. *The gate*: a closed-form per-device peak ledger,
     `_estimate_peak_device_bytes(sino_shape, recon_shape, partition_sequence,
     weights_present, placement)`, checked ONCE at the top of `recon()` before any big
     allocation (pool ≈ fully free via `device.memory_stats()`), so failure lands in
     seconds with one readable error naming the dominant phase + the existing remedy
     hint; `skip_memory_preflight` override, ~10–15% margin.  The ledger enumerates,
     per phase, persistent set + largest co-live transient lineup: persistent = sino +
     weights + error_sino (3× sino-shaped) + flat_recon + fm_hessian (2× recon-shaped);
     subset update per granularity = the granularity-INDEPENDENT sino-shaped pair
     (`weighted_error_sinogram` + `delta_sinogram` — what killed the student's run;
     skip-1 sequences don't touch these) + 4–5 subset-shaped arrays (freed at the
     mid-updater `del`) + the projector transient as a measured geometry-dependent
     multiplier (constants already in code comments / the dashboard 12× aggregate).
     Max over phases and over the granularities actually in the sequence.  A MODEL, not
     a compile query, because the updater is eager Python — no single
     `memory_analysis()` sees the cross-call lineup — and because it must run before
     the compiles it would otherwise wait for.  Covers split_sino_recon (per half),
     prox, and the denoiser for free via the `vcd_recon` entry path.
  2. *Calibration, not user-facing*: at compile sites, a CI/debug-mode assertion
     compares each program's actual `compiled.memory_analysis().temp_size_in_bytes`
     against the modeled term and warns on excess; with the nightly `peak_bytes_in_use`
     gates, model drift is caught by the dashboard, not by user crashes.
  Implementation details settled: ledger terms overridable per geometry (cone vs
  parallel differ mainly in the projector multiplier); print the ledger at verbose≥2 on
  successful runs (free memory-budget printout, keeps the model inspectable).  Also
  catches most of the removed multi-GPU-OOM-hang family (deterministic OOMs die before
  any device enters a collective).  Related UX notes from the same incident: stdout
  block-buffering makes sweep logs non-chronological (`python -u`); `TF_CPP_MIN_LOG_LEVEL=2`
  can silence a residual BFC warning wall (document in the OOM hint, don't default).
- **Suite tidiness**: seed the remaining unseeded-`np.random` tests; a pre-merge
  `import mbirjax`-before-`jax` sweep; public `shard_*` / `gather_*` wrappers.
- **Suite efficiency**: simplify tests and reduce time on tests.
- **>2^31 audit — CLASS CLOSED 2026-07-10** (both faces swept package-wide: flat-index
  ops on full-size arrays, and Python-int element counts crossing into jnp arithmetic;
  the live `mar.py` `num_real_pixels` overflow was fixed with the `float()` idiom + a
  value-pinned regression test, and no other instances were found).  The `lessons.md` §4
  grep recipe is the re-check on new code; residual known edges (np.histogram on
  Windows/numpy<2, the caller-enforced float contract) are documented + accepted there.
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

**Recently completed (records live elsewhere):** the **Pallas projector-kernel campaign**
— the full custom-kernel path for both projectors and geometries, shipped and
soak-validated (design → `docs/source/dev_projector_kernels.rst`; measured record →
`plans/projector_kernels/gpu_headroom_findings.md`); the **default partition-sequence
change to `[2, 4, 6, 7]`** (→ `plans/partition_sequence/`); the **flash-remediation
padding remedies** (→ `plans/flash_remediation/`); the earlier projector-kernel /
profiling campaign (→ `plans/projector_kernels/fwd_back_findings.md`); the multiaxis
vertical-fan path-length factor (shipped); the sparse-projector batching investigation
(closed with code unchanged → `plans/projector_batching/batching_refactor_design.md`).
