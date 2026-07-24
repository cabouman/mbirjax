# Flash remediation — recon-support padding for FoV truncation

**Status (2026-07-12): investigation COMPLETE (Phases 1–2d); implementation COMPLETE and
VALIDATED on real scans, step by step — A: cone per-end axial extension (`a872695`;
SiC at two scales + BGA axial-only), B: split_sino_recon geometry overlap + taper
retirement (`fcc0e9e`; Lilly 4×/8×), C: lateral detect-and-warn (`41ecbc2`; BGA/z62/
SiC/Lilly scorecard + the axial+lateral BGA comparison), D: NSI auto-geometry cleanup
(`dbc9c3b`; Lilly shape/values/seam).  The illustrated validation report is
`phase_3_results.html` (published).  Step E (re-baseline records + Lilly cache rebuild)
is in progress — see the implementation record below.**

Source item: `plans/current_plans.md` §1.  The per-case remedy spec — equations, code
sketches, pros/cons including do-nothing — is **`phase_2d_remedies.html`**; the
illustrated evidence pages per phase are listed in `README.md` and published at
`/depot/bouman/www/mbirjax/flash_remediation/`; scripts live in
`plans/experiments/flash_remediation/`.

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

### Step C follow-up — lateral+axial padding on the warned scans (DONE 2026-07-12, `p3e_*`)

The missing arm of the comparison: the warning's own remedy applied on top of the new
default (no-pad and axial-only already measured under steps A/B).  Lateral scaling grows
R and hence the axial bound, so the experiment applies the compensating slice growth
explicitly (`extend_axially_to_bound`, measured against the CURRENT slab — the concrete
form of the planned cone `scale_recon_shape` warning's advice; Lilly's NSI-inflated slab
already covered 1.25×R so it got +0 slices, while BGA needed +50/+100 per end at
s = 1.5/2.0).

- **BGA at s = 1.5 (1149², iter 50): cover reached — and it fixes what axial could not.**
  The axial-only run has a 3.2× ring at the old FoV boundary (radial peak 0.049 vs
  interior 0.015); the padded run erases it (old-boundary region at interior level) and
  shows NO ring at its own boundary (peak 0.006 vs inner 0.010 — natural decay).  By the
  P2b asymmetry, s = 1.5 is at/past cover, so the OOM'd s = 2.0 run is unnecessary — the
  knee question is answered from the s = 1.5 volume itself (the single-H100 limit sits
  between 1149²×784 and 1532²×884).  The center slice now reconstructs the actual board
  continuing past the old FoV.  Interior speckle drops (center-40 noise 0.043 → 0.035,
  off-center median 0.017 → 0.013), though the localized center-slice spike persists
  (the separate artifact).  Convergence: change-% at iter 50 drops 1.62 → 0.92 — better,
  but still far from the 0.2% stop, consistent with P2b's severe-overshoot caveat (the
  interior-tomography DC ambiguity keeps converging slowly; an air anchor remains the
  only fix for that residual).
- **Lilly at s = 1.25 (467², iter 15): a null result, honestly.**  The mild one-sided
  truncation (16% of view-rows) produces NO measurable ring at the central slab in
  either variant (annulus means ~1e-4 at all shared radii) — the warning correctly
  describes the DATA, while the reconstruction impact at ds4 is below noise at the
  probed slices.  (A slice-resolved hunt at the specific rows with edge support is
  possible if ever needed.)

Volumes/logs/figures: `padding/p3e_*`; the radial-profile and three-way center-slice
figures are the keepers for the results page.

### Step D — NSI pipeline auto-geometry cleanup (DONE 2026-07-12, commit `dbc9c3b`)

The NSI flow computed the axial extension at model construction with DEFAULT detector
pitches and offsets: the pitch face inflated R/SID by 1/δ_ch (measured 2× over-extension
on Lilly ds4; would be ~8× at full resolution), and the offset face mis-centered by
|det_row_offset|·R/SDD.  The fix aligns NSI with the zeiss convention — **construct →
`set_params(**optional_params)` → `auto_set_recon_geometry()`** — and
`nsi.compute_sino_and_params` no longer hand-sets `recon_slice_offset` (the per-end
extension places the slab from `det_row_offset` directly).  Design call (Greg): NO
backward compatibility and no deprecation warning — `recon_slice_offset` is a legitimate
`set_params` parameter, so loaders don't police it; a stale entry in an old parameter
dict simply passes through (and is overwritten by the auto call under the new
convention).  Companion changes: all `mbirjax_applications/nsi` scripts gained the auto
call (the split demo also lost a stray `recon_slice_offset = 0.0` override), and the
metrics `build_cache.py` NSI sidecar flipped to `auto_set_recon_geometry: True` for
future cache builds.

### Step D validation — Lilly, new flow (DONE 2026-07-12, job 13454976, `p3f_*`)

- **Shape/offset exactly as derived**: ds4 recon (374, 374, **572**) vs the inflated 667
  (95 slices ≈ 14% recovered; base 471 + per-end 55/46 from det_row_offset = −3.9 rows),
  offset +4.50 slices (the per-end value; the old hand compensation was +3.90).
- **Values preserved**: old-flow vs new-flow 15-iter recons agree on physically z-aligned
  interior profiles to **max rel 0.62%** over the whole shared z-range (the level set by
  the sub-slice grid difference + trajectory noise at 15 iterations).
- **Split seam still fixed under the new flow**: ds8 seam max RMS **5.6e-4** (background
  2.4e-5), formula still choosing h_recon = 9 at a near-worst-case sub-slice mismatch
  (−0.45) — at or better than the step-B validated level (9.5e-4 under the old flow).

### Step E — re-baseline and release record (IN PROGRESS 2026-07-12)

Landed in `mbirjax_metrics` (staged): the engine's policy block now records
`axial_extension` (capability-probed via `get_support_radius`, so the dashboard
auto-marks the cone-cell shape/value/memory step in every chart when a tracked branch
crosses the padding commits), the dashboard diffs it (`_POLICY_FIELDS`), and
`annotations.yaml` carries the narrative marker.  Timing: the nightly last measured
`401ad311` (2026-07-09, pre-padding), so these records land before the first
post-padding measurement.  Remaining operational actions (each needs a go-ahead):

1. **First post-padding nightly review + ack**: when the nightly measures `dbc9c3b`+,
   review the correctness divergences (expected: cone-family shapes/values step in the
   padded direction; split cells change with the taper retirement; memory up ~R/SID in
   cone recon cells), then bump the reviewed-through watermark via
   `action_scripts/clear_correctness.sh`.
2. **Rebuild the five Lilly caches — DONE 2026-07-12** (job 13459660; the old caches are
   parked in `cache/superseded_2026-07-12_pre_padding/`, MANIFEST regenerated).  All
   five rebuilt with identical sinograms, sidecar `auto_set_recon_geometry: True`, and
   offset-free `optional_params`; loaded per the sidecar they produce the lean new-flow
   grids — d4x tags (470, 470, 427) with the per-end offset +0.82 ALU, v3x_d2x
   (940, 940, 828).  (These caches use the `Autoinjector_HighRes_Horizontal` source with
   auto-crop — landscape, 470 channels — not the portrait D01788 copy the p3c/p3f
   validations loaded, hence the different numbers; both are the same new-flow
   machinery.)  The existing partition-sequence reference recons/floors correspond to
   the OLD grids and stay as historical records.

**Release note (draft, for the next release's PR/notes):**

- Cone-beam automatic reconstruction shapes now extend the slab, per end, to the
  cone-beam visibility bound: material a ray crosses near the slab ends is representable
  instead of flashing into the end slices.  Default shapes grow by ~R/SID
  (geometry-dependent, typically 10–40% more slices); truncated scans lose the end-slice
  flash/ringing, reconstruct the real out-of-slab continuation, and reach the stopping
  threshold in fewer iterations (SiC: ~iteration 20 vs ~49).  Offset detectors and
  helical travel are handled per end.  Setting `recon_shape` explicitly overrides, as
  before.
- `split_sino_recon` derives its reconstruction overlap from the same geometry
  (`ceil(h_sino·(1+R/SID)·ρ) + 2`); the sine weight taper is retired (it failed at
  coarse downsampling; the matched extension fixes the seam in every regime tested).
  New: `align_split_grid` opt-in (sub-slice grid alignment; not registration-identical
  to `recon()`), and `recon_dict['split_params']` reports the overlaps used.
- Lateral FoV truncation is now detected from the existing sinogram support indicator
  and warned with the remedy (`scale_recon_shape(s, s)`, round UP); deliberately no
  auto-padding (the required cover is not derivable from truncated data).
- NSI preprocessing: call `auto_set_recon_geometry()` after
  `set_params(**optional_params)` (as the zeiss flow does); `optional_params` no longer
  carries a hand-set `recon_slice_offset`.
- Later: analogous per-end bounds for translation and multiaxis-parallel; a cone
  `scale_recon_shape` override warning on uncompensated lateral growth (the p3e
  `extend_axially_to_bound` helper is the prototype); the z62 RoR-boundary-ring open
  question (decoupled from truncation by the step-C validation).

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
