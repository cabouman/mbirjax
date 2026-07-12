# Flash remediation — recon-support padding for FoV truncation

**Status (2026-07-11): investigation COMPLETE (Phases 1–2d); implementation IN PROGRESS
with real-data validation after each step.  Step A (cone per-end axial extension) is
in-tree at commit `a872695` and validated on the real SiC scan at two scales; the BGA
axial-only check is running; steps B (split) and C (lateral warn) are next.**

Source item: `plans/current_plans.md` §1.  The per-case remedy spec — equations, code
sketches, pros/cons including do-nothing — is **`phase_2d_remedies.html`** (published at
`/depot/bouman/www/mbirjax/flash_remediation/`); the illustrated evidence pages per phase
are listed in `README.md`; scripts live in `plans/experiments/flash_remediation/`.

*This file was reorganized 2026-07-11 (Greg's request) into the settled story.  The full
chronological record — the announce-and-retract chains, per-round build-up narratives, and
superseded proposals — lives in this file's git history.*

## Problem

Objects extending outside the field of view converge slowly and produce a bright "flash"
artifact at the reconstruction boundary:

- **Radial (channel) flash:** a thin bright ring just inside the RoR boundary plus a
  whole-interior positive bias.  On the z62 cylinder a 5% radial crop drops the 0.2%-stop
  NRMSE 5× — the ring holds ~80% of the reported error and keeps the change-% stop metric
  elevated, so hard objects run into the iteration cap.
- **Axial (row) flash + z-ringing:** bright end slices, and on scans that leave the
  detector in rows (SiC), a z-ringing that grows with iterations at the truncated end.
- **Split seam:** `split_sino_recon` stripes at its stitch seam (±40% every-other-slice
  zigzag on Lilly; the shipped sine taper suppresses it at 4× downsampling but fails at
  8×, so the shipped default was visibly broken in a regime users run).

The previous mitigation was post-hoc masking (`export_recon_hdf5(remove_flash=True)`),
which discards the affected voxels without fixing the recon or the convergence drag.

## Mechanism (what the investigation established)

- **The flash is a model-support problem.**  Attenuation from material the model cannot
  represent must be explained by in-support voxels; the least-constrained ones — the RoR
  rim, the end slices — absorb it.  **Support beats down-weighting**: every taper variant
  lost to giving the model the voxels to explain the data, in every axis tested.
- **Rows vs channels are NOT symmetric.**  *Axial:* edge-row data is genuinely
  unexplainable (its rays traverse slices outside the slab), and the reachable z is
  geometry-BOUNDED — no measured ray reaches |z| > |v|·(SID+R)/SDD (verified bit-exact:
  material beyond the bound projects to nothing) — so "full" axial padding is always
  cheap and exact.  *Lateral:* material at radius r contaminates **every** channel
  |c| ≤ r at some view, so a channel taper cannot fix the interior, and no geometric
  bound exists — "cover the object" is not derivable from geometry or from truncated
  data.
- **Extensions are prior boundary conditions.**  The qGGMRF prior couples slices, so
  support extensions must be data-accurate; down-weighting the data that feeds them
  starves them into prior extrapolations (P2a: taper-alone 2.4× worse than any padding;
  tapering a fully padded slab actively hurts).
- **The split seam has a two-part cause.**  Each half is axially truncated at its
  extension end (unexplained depth h_sino·(1+R/SID)·ρ − h_recon — §1's wedge evaluated at
  the iso ray, hence ~1 slice), and the load-bearing driver is the SUB-ROW misalignment ε
  between the sinogram cut row and the recon split slice (ε ≈ 0.4 on Lilly): aligned
  symmetric truncation is benign at the default depth; the misalignment breaks the
  halves' symmetry and converts it into the alternating stripes (dose-response verified,
  object-independent).  Default synthetic models cannot show this — they place both grids
  symmetrically — which is why the synthetic study initially exonerated the design.

## Findings by case (settled results; illustrated pages have the figures)

**Axial** (Phase 1, P2a, P2a-R rider; `phase_2a_axial_results.html`):
padding to the geometric bound removes the flash and z-ringing (end NRMSE 9× on the
synthetic case; the {none < taper < pad} ranking held under wide cone, sharp
regularization, and photon noise); padding past the bound is exactly free (bit-exact —
the extra slices intersect no rays), so there is no over-padding risk and no threshold.
Every fractional axial quantity is governed by the fan ratio R/SID alone.  Honest
ceiling: the end wedge is half-sampled with or without padding — padding removes the
artifact, not the sampling physics.

**Radial** (P2b, 33 variants incl. knee-completion points; `phase_2b_radial_results.html`):
the knee is at "cover the object" at every overshoot and regime; every knee curve is a V
with an ASYMMETRIC penalty — under-padding costs orders of magnitude, over-padding costs
percent — so round the padding UP under uncertainty.  The channel taper is falsified
(rim cosmetic, interior untouched).  Cover quality degrades with overshoot (padding
cannot restore unmeasured redundancy), and severe truncation leaves a near-uniform DC
offset no padding can remove (interior-tomography ambiguity; needs a known-air anchor).
The only hard geometric limit is the rotation clearance R_obj < min(SID, SDD−SID) —
useless as a pad size, so cover is genuinely not derivable.

**Split seam** (P2c synthetic + Lilly 4×/8×; `phase_2c_split_results.html`):
with the sino overlap at full weight the split is numerically free ONCE the recon
extension covers everything the kept rows see; the shipped h_recon = h_sino symmetry is
exactly the shortfall the stripes live in.  The taper is a regime-dependent suppressor
(11× at 4× downsampling, only 1.3× at 8×); the geometry-derived extension
h_recon = ceil(h_sino·(1+R/SID)·ρ) + 2 fixes both regimes (seam 9.0e-4 at the formula
value; extra depth changes nothing), and restores normal seam convergence.  Removing the
misalignment driver outright (a sub-slice grid shift) reaches 1.1e-5 at the default
depth, but changes the output sampling — opt-in only.

## Remedies (plan of record; full spec with code sketches in `phase_2d_remedies.html`)

1. **Axial — extend automatically, per end, in `auto_set_recon_geometry`.**
   E_end = max(0, |v_end|·(SID+R)/SDD − H_iso/2) with v_end the detector row-EDGE heights
   (det_row_offset-aware — the two ends differ; helical excesses attach at z_max/z_min of
   the travel), R = the recon-support radius implied by the RoR-mask setting.  Always-on:
   truncation is the norm in practice (a holder virtually always leaves the FoV at one
   end), over-padding is provably harmless, and no threshold exists anywhere.
2. **Lateral — detect and warn; deliberately NO auto-padding.**  Detection reuses the
   `_get_sino_indicator` support mask `auto_set_regularization_params` already computes
   (support touching the edge channels ⇒ truncated; free, no new threshold).  The
   quantity automation would need — cover — is the one the data cannot supply, and the
   under-pad branch is catastrophic: the finicky-threshold case where assisted-manual
   beats automation.
3. **Split — h_recon from the geometry formula as the default; `align_split_grid` as an
   opt-in; retire the taper in the same change.**

Cross-cutting: automate only where the failure mode is benign; respect asymmetric
penalties (round up); validate on real scans before changing defaults.

## Implementation record

Greg's directive (2026-07-11): validate on REAL data after each step, not at the end.

### Step A — cone per-end axial extension (DONE 2026-07-11, commit `a872695`)

- `mj.get_support_radius` (vcd_utils): R to the voxel OUTER edge.  The RoR question from
  the spec was resolved from code: the VCD partitions AND the full-index forward/back
  projections all restrict to `get_2d_ror_mask`, so with the default inscribed ellipse
  R = ½·max(N_r·δ_row, N_c·δ_col); `use_ror_mask=False`/custom updates corners → the grid
  half-diagonal (conservative — over-padding is the cheap direction).
- Extension in `ConeBeamModel.auto_set_recon_geometry` via the model's own
  `detector_mn_to_uv` at the row edges m = −0.5 and N−0.5 (sign convention cannot drift),
  with the inf-SDD-safe far-side factor (1/mag + R/SDD); per-end ceil + clamp;
  recon_slice_offset recentered by (n_top−n_bot)/2 slices.  14 unit tests
  (`tests/geometries/test_auto_geometry.py`) including the physics gates: material beyond
  the computed bound forward-projects to EXACTLY zero, and the outermost auto slices are
  genuinely reached.
- **Real bug found by the suite:** `helical_fdk_z_weight` computed num_views/coverage
  with a central-ray visibility criterion, guarding only PADDED slices against zero
  coverage — the extension makes REAL slices with zero central-ray coverage reachable, so
  every helical `recon()` NaN-poisoned through its FDK initializer.  Fixed:
  zero-coverage slices get weight 0 (FDK has no data there; VCD fills them from grazing
  rays + the prior).  *Pattern to watch in later steps: the extension makes "real slice
  with zero X-coverage" a reachable state where code assumed it impossible.*
- Test-calibration churn, all restored to pre-change numbers rather than loosened: the
  FDK-quality and NSI-preprocessing gates now embed their phantom in the central-ray base
  slab (the extension slices are the half-sampled wedge — graded nowhere; a stretched
  phantom had also moved sinogram gradients onto fixed-seed defective pixels); a stash
  A/B verified the restored FDK metrics are identical to three decimals.  The cone
  padded-slices test class re-tuned to prime slice counts (5 det rows → 7/11 circular/
  helical) with a new guard test.
- NSI composition note: `nsi.py` applies det_row_offset only AFTER model construction, so
  the extension computes offset-blind there and NSI's own recon_slice_offset recenters;
  residual mis-centering = |offset|·R/SDD (sub-slice on Lilly).  The clean fix — the NSI
  pipeline re-runs auto-geometry after setting detector params and drops its hand
  compensation — changes preprocessing behavior and is deferred to its own step.

### Step A validation — SiC real scan (DONE 2026-07-11; decisive)

Single-variable A/B on the cached SiC sinograms (scripts `p3a_*`; volumes, trajectory
logs, and figures under `/depot/bouman/data/mbirjax_metrics/padding/`): same library at
`a872695`; variant *old* = pre-extension shape/offset forced via the `set_params` escape
hatch; sequence [0,2,4,6,7], seed 0, transmission_root weights, snapshots at iterations
15 and 50 of one continuous checkpointed run.

- **v4x_d4x (512-class; new shape 512²×656 — R/SID ≈ 0.28 on this wide-fan Versa):**
  SiC's truncation is one-sided (bottom).  OLD reproduces the artifact: end-slice
  interior-disk mean spike 0.39 (~8× body) plus crossing z-ringing arcs penetrating
  ~150–200 slices into the volume at the 1e-3..1e-2 relative level (deeper than the
  synthetics suggested).  NEW: the arcs are gone and the extension reconstructs the
  actual continuation of the object below the old boundary (half-sampled, smooth).
  Axial locality on real data: interior (middle-half slab) RMS(new−old)/bodyRMS =
  0.0075; the untruncated top end differs 0.023 and its extension is empty (4e-4 vs
  body 0.092) — always-on costs nothing where nothing is truncated.
- **Convergence (the §2 metric-caveat effect, measured):** the new slab crosses the 0.2%
  default stop at ~iteration 20; the old needs ~49.  At +12–15% per-iteration cost the
  extension is **≈2.2× faster wall-clock to the default stop**, and under the shipped
  max_iterations=15 cap the old shape never approaches the stop while the new one nearly
  reaches it.
- **v3x_d2x (1024-class, 1024²×1310):** replicates everything (interior 0.0083,
  truncated end 2.53, top 0.026, stop ~20 vs >50).  Provenance cross-check: the
  old-variant iter-50 volume vs the pre-existing prerelease-era reference recon agrees to
  **rel max diff 7.9e-7** — numerically the same reconstruction (the branch changed
  nothing at the old shape; the escape-hatch forcing is exact).
- Analysis tooling: 1024-class volumes exceed login-node memory limits —
  `p3a_sic_diff_stream.py` streams the per-slice stats/diff/cross-check in row blocks
  (~1 GB peak); reuse it for big-volume comparisons.

### Step A validation — BGA Normal scan (DONE 2026-07-11; axial-only A/B, job 13438912)

BGA (`17U1-250TC-Normal_Tomo_No_HART.txrm`; `experiment_zeiss.py` settings: d2x, view/2,
sharpness 1.5, snr_db 35) leaves the FoV both LATERALLY (untreated in both variants) and
axially at BOTH ends; new shape 766²×684 vs old 766²×484 (R/SID ≈ 0.41, the widest fan
yet: +100 slices per end).  **Answer: axial padding alone does NOT materially improve
the center-slice noise or the slow convergence — both are dominated by the untreated
lateral truncation, plus a separate center-slice artifact:**

- The "noise near the center few slices" is a sharply LOCALIZED spike of the in-plane
  high-pass noise index at the exact center slice (0.22 vs ~0.03 surroundings at iter
  15; still 0.11 vs ~0.02 at iter 50) and it is IDENTICAL in old and new — it is not a
  truncation artifact at all; cross-reference the known center-slice artifact
  (`plans/bugs_and_artifacts/center slice noise/`, observed in this same sharpness 1.5 /
  snr 35 regime).  The broad interior speckle (lateral contamination + sharp
  regularization) changes only ~5% (center-40 noise old/new = 1.06× at iter 15).
- Convergence unchanged: both variants still ~1.5%/iter at iteration 50, nowhere near
  the 0.2% stop (new 1.62 / old 1.48 at 50 — equal within trajectory noise).
- What the extension DOES do here, exactly as on SiC but at both ends: the end-slice
  flash and ringing arcs are removed, the end noise spikes drop (top end 0.032 → 0.017
  at iter 50), and the extensions reconstruct the real material continuation.  Interior
  RMS(new−old)/bodyRMS = 0.047 — larger than SiC's 0.0075 because the volume is far from
  converged, so the two trajectories differ more.
- Consistent with the mechanism section: axial remedies are axially LOCAL.  BGA needs
  the lateral treatment — step C's detect-and-warn plus deliberate cover padding, with
  P2b's severe-overshoot caveat (a DC offset survives even correct cover without an air
  anchor).

Scripts `p3b_*` (incl. `p3b_noise_probe.py`, the per-slice noise-index instrument); the
BGA cache, volumes, trajectory logs, and figures live under `padding/`.

### Step B — split_sino_recon extension + taper retirement (DONE 2026-07-11, commit `fcc0e9e`)

- `half_overlap_recon = ceil(half_overlap_sino·(1+R/SID)·ρ) + 2` (R via the shared
  `get_support_radius`; the two-branch `half_overlap_sino` sizing kept so the knob still
  spans ~half_overlap slices when recon slices are coarser than rows); the sine taper
  RETIRED (with the geometry-derived extension the overlap data is fully explainable —
  the taper was a regime-dependent suppressor); weight halves are now host views and
  `weights=None` passes through (the per-half all-ones array is no longer built);
  `align_split_grid` opt-in (cut-row search, effective at ρ≠1, + residual sub-slice grid
  shift ≤ δ_slice/2 — output NOT registration-identical to `recon()`); feasibility
  fallback when a half is thinner than the recon overlap; new
  `recon_dict['split_params']` reports the overlaps, the residual cut/split mismatch,
  and any grid shift.  Tests: `tests/geometries/test_split_overlap.py`; the
  production-size split-vs-unsplit gate unchanged (0.0505 vs 0.0487 pre-change).

### Step B validation — Lilly 4×/8× (DONE 2026-07-11, job 13439000; decisive)

Shipped `split_sino_recon` vs a matching unsplit `recon()` at the exact P2c regimes
(seam metric: per-slice interior-disk RMS of split−ref; the aligned variant is compared
against an unsplit reference on its own shifted grid).  15 iterations, seed 0,
transmission_root:

| regime | default (formula, no taper) | aligned (opt-in) | P2c yardsticks |
|---|---|---|---|
| ds4 (4,4)/ss2 | **4.1e-4** (18.8× bg) | **7.0e-5** (5.1× bg) | old taper 6.5e-4; ext. 5.7e-4 |
| ds8 (8,8)/ss8 | **9.5e-4** (38.1× bg) | **8.2e-5** (5.0× bg) | old taper 6.1e-3; ext. 9.0e-4 |

- The default meets/beats every yardstick: at 8× it reproduces the P2c formula-value fix
  (9.5e-4 vs the measured 9.0e-4; the old shipped taper managed only 6.1e-3 there), and
  at 4× it beats both the old taper and the P2c deep-extension number.
- Notably, the ds8 default ran at the WORST-CASE sub-slice mismatch (split_params
  reported 0.4999) and still hit the formula-level seam — the +2 margin absorbs the
  worst case, as designed.
- `align_split_grid` buys another ~6–12× (to ~5× background), with the residual
  mismatch at ~1e-14 and grid shifts of 0.05–0.11 ALU — the alignment bookkeeping
  verifies end-to-end on real data.
- The formula chose h_recon = 9 in both regimes (the P2c-validated value at 8×).

### Step C — lateral detect-and-warn (DONE 2026-07-11, commit `41ecbc2`)

- `TomographyModel._check_lateral_truncation`, called from
  `auto_set_regularization_params` on the support indicator it already computes: free,
  threshold-free (the 0.02 edge-fraction floor is a stray-pixel guard), skips the
  all-ones indicator fallback, respects `verbose`, and the message names the manual
  remedy with its usage rule (`scale_recon_shape(s, s)`, round UP) plus the
  severe-truncation DC-offset caveat.  Geometry-gated by documented no-op overrides in
  `TranslationModel` (a plate spanning the FoV is the normal condition) and
  `QGGMRFDenoiser` (image content at the frame edge is normal); cone/parallel/multiaxis
  inherit the active check.  Tests: `tests/geometries/test_lateral_warning.py` (7); the
  full suite confirmed no warning pollution (Shepp-Logan phantoms sit at ~0.9 FoV, under
  the indicator threshold).

### Step C validation — real scans incl. BGA (DONE 2026-07-11, `p3d_lateral_warn_check.py`)

| scan | verdict | edge fraction | reading |
|---|---|---|---|
| BGA Normal | **FIRED** | 86% | severe lateral truncation — as known |
| Lilly ds4  | **FIRED** | 16% | TRUE POSITIVE, previously unlabeled (see below) |
| SiC        | silent  | 0.000 | contained laterally — as known |
| z62        | silent  | 0.000 | contained — an ATTRIBUTION CORRECTION (see below) |

Two findings beyond the pass/fail:

- **z62 is genuinely contained** (edge channels exactly zero; support spans channels
  [94, 417] of 512, ~94 channels of air per side).  The partition study's z62 ring — a
  5% radial crop drops the 0.2%-stop NRMSE 5×, "the ring holds ~80% of the reported
  error" — is therefore NOT lateral-truncation flash: it is a ring at the RoR boundary
  in what is air, with the object at only ~63% of the FoV radius.  Lateral cover-padding
  would do nothing for z62; its ring needs its own investigation (open question, outside
  this program's remedies).  The §2 metric caveat (crop before comparing) still stands as
  a measurement practice.
- **Lilly is mildly laterally truncated on one side** — a previously unlabeled true
  positive: support spans the full channel range with strong right-edge attenuation
  (p99 0.34 vs indicator threshold 0.043; 16% of view-rows), because
  `auto_crop_sino_conebeam` clamped at the RAW detector boundary there (it cropped only
  14 channels total against its 20-pixel buffer).  The Lilly recons therefore carry a
  mild one-sided lateral flash contribution — worth remembering when reading Lilly
  metrics.
- **D. NSI pipeline auto-geometry cleanup — NEAR-TERM PRIORITY; must land before the
  re-baseline (or the NSI-scan baselines churn twice).**  The NSI flow (and any
  set-params-after-construction pipeline) computes the axial extension at model
  construction with DEFAULT detector pitches and offsets, so the extension is wrong in
  two ways: the PITCH face inflates R/SID by 1/δ_ch — measured 2× over-extension on
  Lilly ds4 (667 slices vs the intended ~569; ~8× at full resolution; ds8's δ_ch = 1.016
  is coincidentally near-correct), harmless for correctness (the extra slices are
  provably inert) but real memory — and the OFFSET face mis-centers by
  |det_row_offset|·R/SDD (sub-slice on Lilly).  Fix: the pipeline re-runs
  `auto_set_recon_geometry()` after setting the real detector params and drops its
  hand-set `recon_slice_offset = -det_row_offset/mag` compensation (`nsi.py:390`; the
  per-end extension now handles offsets correctly and better).  Touches the NSI usage
  flow + docs/demos; validate on Lilly (shape shrinks to the intended extension,
  seam/end behavior unchanged).
- **E. Re-baseline** the regression dashboards + release note (after C and D; default
  shapes grow and values shift, for the better); record the regime change both as an
  `annotations.yaml` marker and a policy-block padding flag.
- Later: analogous per-end bounds for translation and multiaxis-parallel; a cone
  `scale_recon_shape` override warning on uncompensated lateral growth.

## Method lessons (durable)

- Synthetic validations inherit their geometry's blind spots (the narrow fan; the
  locked-symmetric grids) — stress conclusions at the real datasets' governing ratios AND
  with their broken symmetries before generalizing.
- When a build-up search stalls, REVERSE-ablate the real failing case — it converges by
  construction.
- Show provenance before claiming: job logs print the live commit; assert
  `mbirjax.__file__` in-process (editable installs resolve through a meta-path finder, so
  PYTHONPATH cannot select the code under test).
- Linear binning of LOG sinograms is provably inert (it is the projection of a smoothed
  object); count-space binning is not.
- Metric hygiene: measure artifact indices on recon − truth, not on the raw recon; the
  flash inflates the change-% stop, so convergence comparisons run at matched iteration
  counts.

## Decisions log (condensed)

- 2026-07-08 (Greg): rows-vs-channels asymmetry agreed; padding (`scale_recon_shape`)
  first-class; visual quality is the initial metric; terminology "variants" (not arms);
  P2a/P2b/P2c designs approved, riders as independent single-variable probes; sbatch
  sweeps authorized.
- 2026-07-09 (Greg): real-data challenge of the synthetic split verdict (Lilly stripes)
  → reproduction closed at the sub-row cut-vs-split misalignment; `phase_2d_remedies.html`
  requested and refined (per-end excesses; R from the RECON grid; `scale_recon_shape`
  stays a pure scaler; lateral = detect-and-warn only; split ρ written explicitly);
  8× regime probe showed the shipped taper failing → h_recon formula is a defect fix.
- 2026-07-11 (Greg): implementation approved with real-data validation after EACH step;
  R helper in vcd_utils; keep the two-branch h_sino sizing; lateral-warn gating list
  approved; remedies-page sign fix + republish; step A committed as `a872695`; validation
  results organized under `/depot/bouman/data/mbirjax_metrics/padding/`; this file
  reorganized to the settled story.

## Pointers

- **Spec:** `phase_2d_remedies.html` (also at `/depot/bouman/www/mbirjax/flash_remediation/`).
- **Evidence pages:** `index.html` (overview) → `phase_1_results.html`,
  `phase_2a_axial_results.html`, `phase_2b_radial_results.html`,
  `phase_2c_split_results.html`.  Regeneration + publishing recipes: `README.md` here.
- **Scripts:** `plans/experiments/flash_remediation/` — `truncation_common.py`, the
  per-phase sweep scripts, and the `p3a_*`/`p3b_*` validation runners/renderers.
- **Data:** cached sinograms `/depot/bouman/data/mbirjax_metrics/partition_sequence/cache/`
  (+ the BGA cache under `padding/`); validation volumes, logs, and figures
  `/depot/bouman/data/mbirjax_metrics/padding/`; reference recons
  `/depot/bouman/data/mbirjax_metrics/recons/`.
