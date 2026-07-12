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

### Step A validation — BGA Normal scan (RUNNING 2026-07-11, job 13438912)

BGA (`17U1-250TC-Normal_Tomo_No_HART.txrm`) leaves the FoV both LATERALLY and axially;
the old recon shows prominent noise near the center slices and converges very slowly.
Axial-only A/B (`p3b_*`, settings mirroring `experiment_zeiss.py`: d2x, view/2, sharpness
1.5, snr_db 35): with the lateral contamination present in BOTH variants, does the axial
extension alone improve the center-slice noise and the convergence?  *[result to be
added]*

### Remaining steps

- **B. split_sino_recon**: h_recon = ceil(h_sino·(1+R/SID)·ρ) + 2 (keep the existing
  two-branch h_sino sizing; shared `get_support_radius`; feasibility fallback for
  volumes too thin for the stitch overlap), `align_split_grid` opt-in, taper retired in
  the same change → validate on Lilly 4×/8× (the measured stripe fixes: taper 6.5e-4 at
  4× but only 1.3× at 8×; formula value 9.0e-4 at both).
- **C. Lateral detect-and-warn**: geometry-gated (cone/parallel/multiaxis on; translation
  and the denoiser no-op — both inherit the base `auto_set_regularization_params`), skip
  when the indicator is the all-ones fallback → check firing on z62/BGA and silence on
  contained scans.
- **D. Re-baseline** the regression dashboards + release note; record the regime change
  both as an `annotations.yaml` marker and a policy-block padding flag.
- Later: analogous per-end bounds for translation and multiaxis-parallel; the NSI
  pipeline auto-geometry cleanup; a cone `scale_recon_shape` override warning on
  uncompensated lateral growth.

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
