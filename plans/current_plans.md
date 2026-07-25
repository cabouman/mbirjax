# Current forward plan — goals for the next release

(The EVOLVING running list of open work, at `plans/current_plans.md`.  Rewritten
2026-07-25.  Completed campaigns live in their own findings docs,
cited where their results guide the work below.  Roughly ordered by likely value.
This file should be cleaned periodically to avoid build-up of historical detail.)

## Goals for the next release

* **Simple per-iteration schedule for large sharpness and/or snr_db** (§1).
* **Wrapper or utility to provide interfaces that mimic LEAP (https://github.com/LLNL/leap) 
and SVMBIR (https://github.com/cabouman/svmbir) to operate MBIRJAX** (§2).
* **MAR: cache H** (§3).
* **Cleanup** (§4).

---

## 1. Remediation of streaks associated with large sharpness and/or snr_db.

**State:** Not started.

**Overview:** In early VCD iterations with large sharpness and/or snr_db, the 
recon can develop streaks associated with voxel cylinders (1D sets of voxels
parallel to the rotation axis, updated as a group in MBIRJAX).  We hypothesize
that this arises from large early steps with a random subset.  The steps are large 
because they are underregularized, and these large steps introduce large differences 
with neighboring voxels in some directions but small differences in other directions.
As a result, the prior essentially treats this streak as some kind of edge to be 
preserved rather than an artifact to be removed.  This is only a hypothesis at this time.  
This effect is most prominent in cone beam.  

**Goals:** 
1. A consistent reproduction of this effect in real and synthetic data and
an investigation of the role of sharpness and snr_db in modulating it.  
2. Evaluation of methods to reduce/eliminate this effect, including the use of
a simple per-iteration schedule to increase from low values to target high values.  The
methods should be simple and apply to multiple geometries.  
3. Implementation and validation of the chosen method.  

**Notes:** Roughly, for small Delta (the difference between adjacent voxels), 
the influence function in the QGGMRF prior is  `(1/2) abs(Delta / sigma_x)`, where
sigma_x is proportional to `2**sharpness` and is defined in tomography_model.py -> 
auto_set_sigma_x.  So, increasing sharpness widens this curve and reduces the penalty
between voxels at a fixed difference.  I.e., this changes the prior function. 
In contrast, snr_db is used to determine sigma_y, which is proportional to 
`10**(-snr_db/20)` and is defined in auto_set_sigma_y.  sigma_y is then used in 
vcd_recon to define fm_constant proportional to `1 / sigma_y^2`, and fm_constant is 
used to multiply the forward gradient and Hessian in the calculation of
delta_recon_at_indices.  I.e., increasing snr_db puts more weight on the forward 
but leaves the form of the prior model unchanged.  

A possibly incomplete list of things that would have to be changed to support a
sharpness/snr_db schedule:  auto_set_sigma_x, auto_set_sigma_y, qggmrf_params,
fm_constant, vcd_recon, vcd_subset_updater (possibly changed to accept qggmrf_params,
fm_constant as inputs), prox_map.  

## 2. Wrapper or utility to provide interfaces to LEAP and SVMBIR

**State:** Not started.

**Overview:** Some of our collaborators are using LEAP (https://github.com/LLNL/leap)
or SVMBIR (https://github.com/cabouman/svmbir - parallel beam only).  We'd like to 
lower the barriers to entry for them to transition to using MBIRJAX.  So, 
we'd like to develop an easy way to replace LEAP/SVMBIR with MBIRJAX.  

**Goals:** 
1. Download and run some examples using each of these packages.
2. Determine the scope of a possible translation from these to MBIRJAX.
3. Design, implement, and validate interfaces.  

## 3. MAR: cache H

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

## 4. Miscellaneous / cleanup

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
- **Device-count policy:** a simple, robust rule — even if suboptimal — over a tuned
 choose-N-vs-communication model; this area is potentially finicky for a modest payoff.
  1. **Concrete first step:** fix the auto-device-count basis.  `_auto_device_count` trims
  on the recon-SLICE axis, but projection compute lives on the VIEW-owners, so the slice
  axis is the wrong proxy for "does this device do real work" — switch the basis to
  views (or both).  Small, clearly right, and independent of any cost model.
  2. The full choose-N policy (when does adding a device pay vs its comms cost, incl. the
  CPU-cluster topology rule) stays deferred unless a real workload demands it
  (`sharding_implementation_plan_v3.md` §5/§6).
- **Suite efficiency:** simplify tests and reduce time on tests.
- **Minor API opens:** `configure_devices`/`use_gpu` unification; the forward
  pixel-batch default.
- **Residual rounding-bug risk (monitor only)**: the six vertical-fan per-slice round
  sites keep the in-jit precondition — accepted + monitored
  (`plans/bugs_and_artifacts/jax rounding bug/phase_d_design.md` §7).
- **Archive plans:** Many plan docs and scripts could be moved out of the repo and into 
 archived storage - e.g., another repo or data depot.  

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
