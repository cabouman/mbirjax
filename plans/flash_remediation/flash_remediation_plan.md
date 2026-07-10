# Flash remediation — sinogram weight edge tapering and recon-support padding

**Created 2026-07-08.  Status: Phase 1 DONE (findings below + `phase_1_results.html`);
Phase 2 IN PROGRESS — P2a DONE (axial: pad by the geometry-derived scale, no taper; row
taper retired for the axial case) AND P2b DONE (radial: pad to COVER, round up under uncertainty, no taper) AND P2c DONE + CORRECTED on real data (split seam: sino overlap load-bearing; the taper fixes a REAL geometry-dependent stripe artifact the narrow-fan synthetic missed — revised plan: geometry-derived h_recon, taper then unnecessary; NO code yet).  Phase 2 COMPLETE; next = Phase 3 real scans.**
Source item: `plans/current_plans.md` §2.  Scripts: `plans/experiments/flash_remediation/`.

## Problem

Objects that extend outside the field of view converge more slowly and produce a bright
"flash" artifact at the boundary of the reconstruction:

- **Radial (channel) flash:** a thin bright ring just inside the RoR boundary.  Quantified on
  real data by the partition-sequence study (`plans/partition_sequence/partition_sequence_plan.md`,
  metric-caveat section): on the z62 cylinder a 5% radial crop drops the 0.2%-stop NRMSE 5×
  (0.100 → 0.020) — the ring holds ~80% of the reported error.  It also keeps the change-% stop
  metric elevated, so hard objects run long (the 15-iteration cap binds).
- **Axial (row) flash + z-ringing:** bright end slices, and on SiC
  (`/depot/bouman/data/ORNL/versa/SiC-SiC_CompositeFFOV_tomo-A.txrm`, in-FoV in channels but
  extending beyond the detector in rows) a *ringing* artifact in z on the side that leaves the
  detector.
- Current mitigation is post-hoc removal (`export_recon_hdf5(remove_flash=True)` applies a
  cylindrical mask with radial/top/bottom margins) — it discards the affected voxels rather
  than fixing the recon or the convergence drag.

A stretch target (may fall out of scope): the BGA scan
(`/depot/bouman/data/Zeiss/purdue_BGA/17U1-250TC-Normal_Tomo_No_HART.txrm`) goes far outside
the FoV in channels and, at sharpness 1.5 / snr_db 35, shows center-slice artifacts documented
in `plans/bugs_and_artifacts/center slice noise/` — check whether remediation helps there too.

## Precedent and mechanism

**The precedent** (`cone_beam.py` `split_sino_recon`, ~line 1283): when a recon is split into
two detector-row halves, each half's *weights* are multiplied by a quarter-sine ramp over the
overlap rows (0 at the extreme row).  Each half-model literally cannot explain the other
half's slices' contribution to those rows; a rect cutoff of that inconsistent data rings, a
smooth ramp fades it out.

**Where weights enter VCD** (`tomography_model.py` `create_vcd_subset_updater`): the gradient
back-projection (`W·e`), the forward Hessian `diag(AᵀWA)` (`compute_hessian_diagonal`), the
line-search scalars (`get_forward_lin_quad`), and upstream `auto_set_sigma_y`.  All see the
same `W`, so a taper is a clean reweighting of the objective — it changes the answer, not the
algorithm's validity.  Ordering note: a taper applied before `auto_set_regularization_params`
shifts `sigma_y` slightly.

**Why the flash forms:** the default recon support has zero lateral margin
(`auto_set_recon_geometry`: recon width = detector FoV at iso, exactly), and VCD updates only
the inscribed RoR ellipse.  Attenuation from material the model cannot represent (outside the
support) must be explained by in-support voxels; the least-constrained ones — the RoR
boundary ring, the end slices — absorb it.  The inconsistency also leaves a persistent error-
sinogram residual that the interior keeps responding to, dragging convergence.

**Rows vs channels are NOT symmetric** (the load-bearing distinction, agreed 2026-07-08):

- **Rows (cone/multiaxis z-truncation).**  Edge detector rows collect rays that traverse
  slices outside the recon slab.  That data is *genuinely unexplainable* — exactly the
  split_sino_recon situation at the physical detector edge — so `W→0` there is principled,
  and the taper region is computable from geometry (which rows' rays exit through the slab's
  end faces: cone angle, magnification, recon z-extent).  Parallel beam has no row
  inconsistency (rows map 1:1 to slices).
- **Channels (lateral truncation).**  Material at radius r > R_FoV contaminates *every*
  channel |c| ≤ r at some view subset — not just edge channels (a point at radius r lies on
  lines at all signed distances ≤ r).  Edge channels are contaminated at more views, but they
  are also the *only* measurements constraining the in-RoR boundary voxels.  A channel taper
  removes real information along with inconsistency and cannot touch the contamination at
  interior channels.  The flash concentrates at the RoR edge because boundary voxels are the
  least-constrained dump site, not because the bad data lives only at edge channels.

That mechanism points at a competing fix for the channel case: **enlarge the recon support**
(`TomographyModel.scale_recon_shape`, which exists for exactly this purpose) so the model can
represent the outside material, absorb the flash in margin voxels, and crop on output.
Greg is not wedded to a channel taper; the padding variant is in scope as a first-class candidate.

**The split_sino_recon seam, and the prior-context role of extensions (2026-07-08 discussion).**
Two facts frame the split design:

1. *The forward model is (almost) separable at the iso-row split.*  split_sino_recon requires
   zero helical shifts, so the source sits in the iso plane at every view; a ray to a detector
   row above iso has z increasing monotonically from 0 along its whole path and never crosses
   the split — and symmetrically, a voxel above the iso plane only ever projects onto rows
   above iso.  The only cross-split coupling in the data term is *local* blur: the trapezoid
   voxel footprint (~psf_radius rows), the fractional iso-row rounding, and voxel-vs-row pitch
   mismatch — a few rows/slices, independent of cone angle.
2. *The qGGMRF prior is NOT separable* (Greg): neighboring slices couple, and over iterations
   the influence propagates several slices deep — the original motivation for the overlap.
   The extended (past-split) slices act as **prior boundary conditions** for the kept slices:
   what matters is their ACCURACY, since the kept boundary slices are smoothed toward them and
   the influence decays with a screening length set by the prior-to-data curvature ratio —
   short in well-measured slices, LONG in a weakly-measured extension.

Consequences for the design space:

- The sino overlap's real job is not to inform the kept slices (separability says it barely
  does) but to make the extended slices *data-accurate* so they are correct prior context.
- The sine taper reads as a **graded data-to-prior transition** across the extension — not
  just an anti-ringing window; a hard end-of-data boundary is exactly the kind of constraint
  edge Phase 1 showed rings.
- A truncate-the-sino-at-the-split variant (keep only the recon extension) is predicted to tie
  on objects smooth across the split but LOSE when structure (interface, laminate layer)
  crosses it: the extension becomes a prior extrapolation of the boundary, and the error
  propagates into kept slices.
- The same logic transfers to the physical detector edge (the SiC row case): **taper-alone
  starves the end slices of data** the way truncation starves the split extension — softer
  flash but potentially blurred real end structure.  Hypothesis: the taper's proper role is
  the graded transition at the edge of a (possibly partial) padded support, not a standalone
  fix.  Phase 2 tests this directly.
- Caveat: the separability argument rests on the circular orbit (source in the iso plane); if
  the zero-helical-shift restriction is ever relaxed, this analysis must be redone.

Two separable "speeds convergence" effects to keep apart in measurements: (1) genuinely
faster interior convergence; (2) the flash no longer inflating the change-% stop metric, so
the run stops earlier.  Both are wins; the characterization measures interior-only
convergence to distinguish them.

## Phase 2 plan (drafted 2026-07-08; execution on Greg's go-ahead)

Goal: on the Phase 1 synthetic cases, pick the remediation (and its parameter policy) per
axis, and settle the split_sino_recon seam design — so Phase 3 only has to validate the
winners on real scans.  All experiments stay in `plans/experiments/flash_remediation/`
(library code untouched in Phase 2; variants that need modified split internals are local
reimplementations in the experiment, using the same `copy_ct_model` pattern
split_sino_recon itself uses).  Everything runs on local CPU in minutes per variant; the full
sweeps are at most hours.  Visual quality decides (montages + profiles); the Phase 1
region metrics are the supporting record.

**Shared machinery to add to `truncation_common.py`:**
- Weight-taper builders: row taper (quarter-sine over k rows, per-edge selectable for
  one-sided cases) and channel taper (shape ∈ {sine, Tukey, erf} × width), returned as
  weights arrays for `recon(weights=...)`.
- Geometry-derived row-taper width: from source-in-iso-plane geometry, the rows whose rays
  exit the recon slab within the FoV (h/m·(1+R/SID) vs slab half-height) plus the PSF
  radius — the principled default k, swept around to confirm.
- A variant-sweep driver (list of (label, model, weights) triples through the Phase 1
  tracked-recon loop) so every variant produces the same metrics/figures.

### P2a — axial case: taper vs padding vs combination — **DONE 2026-07-08, see Findings**

On the one-sided z-truncation repro (laminate phantom — structure near the truncated end
makes over-smoothing visible):
- **Variants: padding level {none, partial, full} × row taper {off, on}** (taper on the
  truncated side only; width = geometry-derived k).  "Partial" padding deliberately covers
  only part of the object overshoot — the case where full padding is impractical.
- Add one **far-overshoot case** (object extends ~3× the covered half-slab) where full
  padding is unreasonable, to test taper + partial-pad as the fallback.
- Hypotheses to test: (1) taper-alone softens the flash but blurs real end-slice structure
  (prior-extrapolation starvation, per the mechanism section); (2) with full padding the
  taper is unnecessary; (3) the combo wins only at partial padding.
- Deliverable: the axial recommendation (expected: pad in z — slices are the cheap axis —
  with taper as the graded transition when padding must be partial) + the width/scale
  policy.

### P2b — radial case: padding-scale knee (+ taper falsification control) — **DONE
2026-07-08; findings below + `phase_2b_radial_results.html`**

**Design finalized 2026-07-08 (with Greg); run as parallel sbatch jobs on gautschi
(`radial_pad_sweep.py`, sections selectable at top of file).**  Geometry framing: the
radial case has NO visibility bound (any point, however far out, is measured in views
where it lies near the source–axis line), so the knee is empirical and the policy must
track the object's overshoot.  The only hard stop is MECHANICAL: the rotating object must
clear source and detector, R_obj < min(SID, SDD−SID) — 8× the FoV radius in the base
geometry — so the worst-case padding scale is min(SID, SDD−SID)/R.  ~25 variants:

- **Core padding-scale sweep** (overshoot 1.25×, default regime): scales {1.0, 1.1, 1.2,
  1.35≈cover, 1.5 over-cover control} + channel-taper-only (falsification control, 16-ch
  quarter-sine each side) + pad1.2+taper combo.
- **Overshoot axis**: {none, partial, cover} at overshoots 1.1× and 1.5×, plus an
  **extreme 4.0× showcase** (half the rotation bound) at {1.0, 2.5, 4.1} — the stress test
  the axial case could not have.
- **Regime riders** at {none, knee, cover} each, as INDEPENDENT single-variable probes
  (the P2a-R lesson): wide fan (SDD 4C→2.5C), sharp (sharpness 2/snr 35, 160 iterations —
  the R2 convergence lesson), photon noise + transmission weights at default reg.
- Key summary figure: **knee curves** (ring + interior NRMSE vs padding scale, per
  overshoot and per regime).
- Stretch (design-only): a **data-adaptive truncation detector** (edge-channel level above
  the air floor) as input to an automatic scale policy — wire up in Phase 4 if supported.
- Deliverable: the radial recommendation (expected: padding with an overshoot-tracking
  scale policy; taper at most as a partial-pad transition).  Results page:
  `phase_2b_radial_results.html`.

### P2c — split_sino_recon seam A/B — **DONE 2026-07-08; findings below +
`phase_2c_split_results.html`**

New script (`split_seam_repro.py`), object fully inside FoV and slab (no physical
truncation — isolates the SPLIT effects).  Phantom: laminate layers crossing the split +
a sphere straddling it (the hard, structure-at-the-seam case) and a smooth control.
Variants, each stitched and compared to an **unsplit full recon reference** (same
iterations):
1. **(i) current design**: sino overlap + sine taper + recon overlap, current defaults;
2. **(ii) truncate**: sino cut at the split row + recon overlap (the prior-extrapolation
   variant — predicted to lose on the structured phantom, tie on the smooth one);
3. **(iii) no-taper**: sino overlap + recon overlap enlarged so the extended rows are fully
   explainable, taper off (separates "taper vs no taper" from "truncate vs extend");
4. **(iv) tuned current**: (i) with geometry-derived overlap/taper width (PSF + pitch
   ratio + a screening-length allowance) instead of the fixed half_overlap default.
- Judge on: seam-region visuals (x-z montage), z-profile across the split, NRMSE in a
  ±few-slice seam slab vs the unsplit reference, and iterations-to-stop.
- Deliverable: the split design recommendation; if (ii)/(iii)/(iv) beat (i), a concrete
  proposed change to `split_sino_recon` (with the zero-helical-shift caveat documented).

### Order and decision outputs

Suggested order: **P2b → P2a → P2c** (P2b is the simplest and confirms the biggest Phase 1
win; P2c is self-contained and directly informs a shipping change) — reorder freely if SiC
urgency says axial first.  Phase 2 ends with: per-axis remediation recommendations +
parameter policies (geometry-derived widths, padding scales), the split design decision,
an opt-in-vs-default proposal, and the quality-gate definitions Phase 3 will apply to the
real scans (SiC, z62, BGA).

## Evaluation

- **Phase 1 (synthetic):** visual quality first (Greg, 2026-07-08 — objective metrics are
  finicky and often misleading); curves as supporting evidence: interior vs flash-region
  NRMSE against the known phantom per iteration, signed excess in the flash region, change-%
  trajectory, iterations-to-stop.
- **Real data (Phase 3):** visual quality on SiC (row case) and later BGA (channel case);
  quantitative metrics to be discussed once the synthetic behavior is understood.
- "Correctness" gating note for any eventual shipped change: fingerprints don't apply
  (weights/objective change by design); the gate must be defined at the objective level
  (e.g., interior quality not degraded on non-truncated scans, artifact metrics improved on
  truncated ones).  Placement (user-visible preprocessing vs internal, opt-in vs default) is
  deliberately undecided until the data is in.

## Phases

- **P1 — synthetic characterization (ACTIVE).**  Small (≤~128³-class), local CPU, fast
  iteration.  Two reproductions in `plans/experiments/flash_remediation/`:
  - *Lateral*: cone-beam phantom extending radially past the FoV (inside the slab in z) —
    target: the radial ring.
  - *Axial*: cone-beam phantom extending past the covered slab in z on ONE side (like SiC) —
    target: end-slice flash + one-sided z-ringing.
  Method: build the phantom on an enlarged recon grid on a "truth" model that shares the
  small detector (forward-projecting the big grid through the small detector IS the physical
  truncated measurement — no sinogram cropping needed); reconstruct with the default-shape
  model; snapshot per iteration; compare against the center crop of the ground truth phantom.
  Each script includes an optional padded variant (`scale_recon_shape`) as a mechanism check:
  if padding removes the artifact, the model-support explanation is confirmed.
- **P2 — candidate sweep on the synthetic cases** — detailed plan in "Phase 2 plan" above
  (P2a axial taper-vs-pad variants, P2b radial padding knee, P2c split-seam A/B; single-variable
  ablations throughout).
- **P3 — real scans:** SiC (row case; flash + z-ringing), z62 (channel-flash-dominated),
  sic composite as the control whose error is real structure; BGA as the stretch case.
  Real-data recipes: `mbirjax_metrics/experiments/partition_sequence/README.md`; cached
  sinos `/depot/bouman/data/mbirjax_metrics/partition_sequence/cache/`, existing recons
  `/depot/bouman/data/mbirjax_metrics/recons/`.
- **P4 — placement + API decision** (preprocessing helper vs internal; opt-in vs default;
  interaction with transmission/MAR weights — multiplicative composition is the natural
  form) and docs.

## Decisions log

- 2026-07-08 (Greg): rows-vs-channels distinction agreed; not wedded to a channel taper —
  `scale_recon_shape` padding in scope as the channel-case alternative.  Phase 1 on small
  synthetic data approved; visual quality is the initial metric; later phases adjusted as we
  go.  Plan docs here, scripts in `plans/experiments/flash_remediation/`.
- 2026-07-08 (Greg): terminology — experiment "arms" renamed to **variants** everywhere
  (docs, scripts, figure titles; metrics reproduced exactly on regeneration).
- 2026-07-08 (Greg + discussion): the split_sino_recon seam analysis recorded in the
  mechanism section — forward-model separability at the iso split PLUS the qGGMRF
  prior-context role of extensions (extensions must be data-accurate, taper = graded
  data-to-prior transition; taper-alone predicted to under-perform at physical edges).
  Phase 2 plan drafted (P2a/P2b/P2c above); execution awaits Greg's go-ahead.
- 2026-07-08 (Greg): reports reorganized — one page per sub-campaign
  (`phase_2a_axial_results.html` = the completed axial story;
  `phase_2b_radial_results.html` = radial, in progress) with `index.html` as the project
  overview.  P2b design approved incl. the extreme-overshoot showcase and the physical
  rotation bound (R_obj < min(SID, SDD−SID)); cluster sbatch jobs authorized for the
  sweeps (partition ai requires --cpus-per-task=14 per GPU).

## Findings

### Phase 1 — synthetic characterization (2026-07-08, local CPU)

**Illustrated report (self-contained, figures embedded): `phase_1_results.html` (this
directory).**  Scripts: `plans/experiments/flash_remediation/{lateral,z}_truncation_repro.py`
(+ figures in that directory's `figures/`, gitignored).  Both cases: cone beam, magnification 2, 128 views, 40
iterations, default parameters; ground truth = the phantom on the enlarged grid, metrics on the
default-grid center crop, NRMSE normalized by the ground-truth RMS over the RoR cylinder.

**Lateral truncation** (cylinder radius 1.25× FoV, contained in z; small grid (128,128,32)):

- The default recon reproduces the artifact textbook-style: a bright ring just inside the
  RoR boundary (radial profile peaks ~4.5× the body value at r≈62 of R=64), PLUS a uniform
  positive interior bias (~+8% of body value) and residual interior texture — lateral
  truncation contaminates the WHOLE volume, not just the rim (consistent with the mechanism:
  outside material projects onto interior channels too).
- Convergence: the interior converges FAST to the biased answer (plateau by ~iter 4 at
  interior NRMSE 0.225); the ring keeps slowly intensifying for tens of iterations
  (ring NRMSE 2.04 at iter 1 → 2.21 at iter 40).  In this simple case the damage is a bias
  floor, not a slow transient — the "slower convergence" observation did not reproduce here
  (may need structured/real data, or the z case's ringing).
- **The padded variant (scale_recon_shape ×1.35 lateral) removes essentially everything:**
  interior NRMSE 0.225 → 0.081 (~2.8×), ring NRMSE 2.21 → 0.060 (~37×), ring excess
  +0.028 → +0.0004.  Mechanism confirmed: the flash is a model-support problem, and for the
  lateral case support padding is a near-complete fix in one knob.  Note a channel taper
  CANNOT reach the interior bias component (the inconsistency lives at interior channels
  too), so for lateral truncation padding attacks the cause, tapering at best the symptom.

**Axial truncation, one-sided** (cylinder radius 0.75× FoV with laminate z-layers,
extending 60% of a half-slab past the top of the covered slab; small grid (96,96,64)):

- Reproduces the SiC signature: flash + z-RINGING at the truncated end only (z profile:
  undershoot ~-40% two slices in, overshoot ~+70% at the end slice), decaying into the
  volume; the contained end is clean (end_bot NRMSE ≈ 0.043 = interior level).  The
  artifact GROWS with iterations (end_top NRMSE 0.51 at iter 4 → 0.62 at iter 39) — the
  flash region is what keeps converging (slowly), consistent with the stop-metric drag on
  real scans.
- The interior is essentially UNAFFECTED (default 0.0472 vs padded 0.0464): unlike the
  lateral case, axial inconsistency stays localized to the truncated end slices.
- **The padded variant (scale_recon_shape ×1.7 in z) removes the ringing and most of the
  flash:** end_top NRMSE 0.62 → 0.071 (~9×), and it also converges faster in change-%.

**Implications for Phase 2** (Phase 1 close-out): support padding is the standard against
which any taper must be judged in BOTH axes; Phase 2 A/Bs taper vs padding per axis and
sweeps the padding knee.  (Superseded in detail by the Phase 2 plan section and the P2a
findings below.)

### Phase 2 findings — P2a, axial taper-vs-padding variants (2026-07-08)

**Illustrated report (self-contained, updated per sub-campaign): `phase_2a_axial_results.html`
(this directory).**  Script: `plans/experiments/flash_remediation/z_taper_pad_grid.py`
(figures `p2a_*`).  Same
one-sided SiC-like case as Phase 1 (NOTE: the laminate phase shifted with the larger
ground-truth phantom, so absolute numbers are not comparable to Phase 1's; all seven
variants here share one
phantom).  Variants: padding {none, partial 1.094, full 1.188, overfull 1.7} × row taper
{off, on}, taper widths geometry-derived (6 rows on the default slab, 3 on partial, 0 on
full — the full+taper variant forces 6 as the H2 probe).

**Structural result — z-padding is geometry-bounded.**  With the source in the iso plane,
no measured ray reaches |z| > h_max·(SID+R)/SDD, i.e. (SID+R)/SID × the half-slab (1.125
here).  Confirmed BIT-EXACTLY: extending the phantom from z_hi = 1.6 to 4.0 half-slabs
changed the sinogram by max |Δ| = 0.0.  Consequences: the planned "far-overshoot recon
case where full padding is impractical" cannot exist in z (dropped — replaced by this
forward-projection identity); "full" z-padding is always cheap (scale ≤ 1 + R/SID + psf
margin), and padding past the bound buys nothing (overfull ties full, 0.1410 vs 0.1415).

**Outcome across the seven variants (truncated-end NRMSE at iter 40; interior and contained-end flat ~0.054 /
~0.074 across all variants — axial locality re-confirmed):**

| variant | end_top NRMSE |
|---|---|
| none | 0.887 |
| taper only | 0.344 |
| pad partial | 0.154 |
| pad partial + taper | 0.215 |
| pad full | **0.141** |
| pad full + taper (forced) | 0.220 |
| pad overfull | 0.141 |

- **H1 confirmed:** taper-alone kills the ringing but replaces it with a broad bright
  smear over the end slices (visible in `p2a_xz_taper.png` — the starved end becomes a
  prior extrapolation, exactly the predicted failure mode); 2.4× worse than any padding.
- **H2 confirmed, strongly:** tapering a fully-padded slab actively HURTS (0.141 → 0.220)
  — the taper only removes data the padded model could use.
- **H3 REFUTED:** the combo also hurts at partial padding (0.154 → 0.215).  Reason: the
  visible-material band is thin (≤ 12.5% of the half-slab here), so even "partial"
  padding covers most of it and the taper has nothing left to fix — it only starves.
- **Honest ceiling:** every variant, including full padding, smooths the last laminate
  band in the top ~3 slices.  Top-of-slab voxels project past the detector edge for
  roughly half the views (the classic cone-beam end wedge) — that is *incomplete
  sampling*, not inconsistency, and no weighting or padding can restore it.
- Taper variants do have the friendliest stop metric (change% 0.030 vs none 0.074) — they
  quiet the metric by suppressing the region that was still (correctly) converging.

**P2a recommendation:** for axial truncation, pad in z by the geometry-derived scale
((max_visible_z + psf margin)/half_slab) and do NOT taper.  The row taper is retired for
the axial case — its intended niche (padding impractical) does not exist in z.  For P2c
this raises the prior that the no-taper split variant (iii) may beat the current taper
design (i); the split case still differs (the extension's data comes through overlap rows,
not edge rows), so P2c runs as planned.

### Phase 2 findings — P2a-R, robustness rider (2026-07-08)

Script: `plans/experiments/flash_remediation/z_robustness_check.py` (figures `p2ar_*`;
noise helper `truncation_common.add_transmission_noise`).  Re-checks the P2a RANKING
{none, taper, pad_full} in three regimes (Greg's generality questions (a)–(d)):

- **Governing-ratio analysis first** (recorded here because it shaped the configs): every
  FRACTIONAL quantity in the axial story — visibility bound (1 + R/SID), exit-row band
  (R/(SID+R)), fully-sampled core ((SID−R)/SID, so the end wedge is ~R/SID too) — is set
  by the LATERAL fan ratio R/SID = C·δ_c/(2·SDD) alone.  Magnification per se never enters
  (changing SID moves R proportionally), and row count changes nothing fractionally (it
  scales h_max and the slab together).  So (a) and (b) collapse into one axis: widen   R/SID.
- **R1 wide-cone** (SDD 4C→2.5C, rows 64→128 — the fan, and R/SID with it, widens too: 0.125→0.200): formulas VALIDATED —
  measured bound ratio 1.200 = predicted; far-overshoot identity again bit-exact 0.
  Ranking holds and the taper's deficit WIDENS (none 1.095 / taper-13-rows 0.718 /
  pad_full 0.253): the wider exit-row band starves more slices.  Honest-ceiling note: the
  end wedge grows with R/SID — even pad_full shows laminate-frequency residual near BOTH
  ends at this fan angle (half-sampled slices, not inconsistency).
- **R2 sharpness 2 / snr_db 35, no noise** (the BGA-artifact regularization regime):
  ranking holds (0.780 / 0.299 / 0.150), margins similar to the default regime.
- **R3 photon noise (i0 1e4) + transmission weights at DEFAULT regularization** (taper
  multiplied into the weights, as a user would; R2/R3 kept as INDEPENDENT single-variable
  probes — stacking noise on changed regularization would confound them, Greg): ranking
  holds decisively (0.685 / 0.255 / 0.075 — taper 3.4× behind padding); pad_full's
  difference image is pure noise where none/taper still show the organized flash band /
  smear.  (R3's absolutes aren't comparable to the unit-weight P2a run — the transmission
  weights shift the auto-calibrated sigma_y.)

**R2 convergence probe** (`r2_convergence_probe.py`, Greg's follow-up 2026-07-08): the
sharp regime converges ~3× slower (pad_full 0.2%-stop at iter 39 vs 13 at default reg), so
the 40-iter R2 montage is BARELY converged; its vertical streaks/top mottle are VCD
transients (the update unit is an (x,y) pixel COLUMN — Greg's hypothesis), gone by iter
160.  Streak index on recon−truth halves and keeps falling (5.0e-4 → 3.1e-4 → 2.4e-4 at
40/80/160) — no underdetermined floor.  RULER lesson: on the raw recon the index is flat
~6.7e-3 = the phantom's own structure (ground truth alone scores the same) — subtract the
truth before measuring artifacts.  Converged sharp-prior pad_full BEATS default-reg
pad_full (end 0.069 vs 0.141; interior 0.040 vs 0.055 — it resolves the final laminate
band default reg smooths), crossing over right where the default 0.2% stop fires.

**Conclusion: the P2a recommendation (geometry-derived z-pad, no taper) survives all
three regimes**; interior NRMSE was variant-independent in every config.  P2b should
carry sharpness and noise rider configs from the start, as separate probes (the radial
case's global bias makes regime-dependence more plausible there).

### Phase 2 findings — P2b, radial padding knee (2026-07-08)

**Illustrated report: `phase_2b_radial_results.html`.**  25 variants as 4 parallel sbatch
jobs on gautschi (H100s; <1 hr wall).  All variants CONVERGED (change% ≤ 0.1, metrics flat
iters 20→40), so the effects below are converged-solution properties.

- **The knee is at "cover the object", at every overshoot and every regime.**  Every
  scale set brackets its own cover AND goes past it (Greg's rigor demand 2026-07-08; 8
  knee-completion variants added after the first pass).  Small overshoots show a sharp V
  EXACTLY at cover (1.1×: min at scale 1.12, ring 0.046, clear reversal beyond — 0.077 at
  1.2, 0.197 at 1.35; 1.25×: V at 1.35, 0.063/0.081, reversal at 1.5 = 0.099/0.100).  At
  larger overshoot the minimum shifts MODESTLY BEYOND cover: 1.5× bottoms at scale 1.85
  (ring 0.118 vs 0.191 at nominal cover 1.6) then reverses cleanly (0.130/0.178/0.231 at
  2.1/2.4/2.8 — a second extension after Greg noted the curve wasn't plateauing; the
  earlier "hockey stick" read was a truncated x-range).  4× is still descending gently at
  4.6 (minimum likely similarly past cover; untested).  EVERY curve is a V, and the
  penalty is ASYMMETRIC — under-padding costs orders of magnitude, over-padding costs
  percent — so round the padding UP under uncertainty.  Beyond-cover behavior is
  object-dependent (extra lateral voxels intersect real rays and participate), unlike
  axial's exact overfull==full tie (no rays at all).  Regime
  curves (widefan R/SID 0.2, sharp-160it, noise+weights) descend to the same cover knee;
  beyond it default/widefan/noise revert up while sharp's ring edges down as its interior
  reverses: the policy is REGIME-ROBUST.
- **Channel taper falsified as predicted**: taper_only ring 2.21→1.44 (cosmetic) but
  interior 0.2245 vs none 0.2252 — UNTOUCHED.  pad1.2+taper buys a small ring improvement
  (0.207→0.189, interior unchanged) unlike axial (where it always hurt), but is not
  competitive with cover padding.  Taper retired for the radial case too.
- **Cover quality degrades with overshoot** (at-cover ring 0.046/0.063/0.19/1.21 at
  1.1/1.25/1.5/4.0×): padding lets the model represent outside material but cannot restore
  unmeasured redundancy.
- **Extreme 4.0× showcase** (half the rotation bound; cover recon 525×525): unpadded is
  catastrophic (ring 12.95, structure gone); cover recovers the spheres crisply but rides a
  near-uniform DC offset (most of the residual 1.21/0.68) — the interior-tomography
  ambiguity; removable post-hoc only with a known-air anchor (Phase 3 note for BGA-class
  scans).
- **The rotation bound is real and the assertion caught it**: the first wide-fan config
  (shrink SDD only) had the 1.25× object exactly touching the detector → job rejected by
  the script's bound check; fixed by shortening SID and SDD together at fixed
  magnification (R/SID 0.125→0.2).
- **Radial recommendation: pad to cover, no more, no less; no taper.**  "Cover" is not
  derivable from scanner geometry → the shipping policy needs the data-adaptive overshoot
  estimate (edge-channel level above air floor) — design moves to Phase 4.

### Phase 2 findings — P2c, split_sino_recon seam A/B (2026-07-08)

**Illustrated report: `phase_2c_split_results.html`.**  Script `split_seam_repro.py` (a
local parameterized reimplementation of the split geometry; library untouched).  Object
fully contained (no physical truncation → all seam error split-induced); four variants ×
two phantoms (laminate crossing the split + smooth control), each judged against an
unsplit 40-iter reference; seam region = interior disk × split ±4 slices.

- **The sino overlap is LOAD-BEARING**: truncate (sino cut at the iso row, recon overlap
  kept) costs seam NRMSE ~0.18 vs the reference (6× the reference's own truth error;
  visible dark band through the straddling sphere, ~20% z-profile sag recovering over ~3
  slices each side).  **On the smooth control too (0.17)** — REFUTING the "ties on smooth"
  prediction: data-starved extensions settle low regardless of object structure, and reach
  the seam via prior context AND the stitch blend.
- **With the overlap at full weight the split is numerically FREE**: no_taper matches the
  unsplit reference at seam NRMSE 2–3e-5 (the GPU run-to-run noise floor) on both
  phantoms.  The iso-plane separability is exact in practice once the extensions are fed.
- **The sine taper is unnecessary and (harmlessly) counterproductive**: current lands at
  ~1e-3 vs reference — invisible, but ~30× the no-taper deviation.  The overlap data is
  NOT inconsistent (every extended row's contribution is representable by the extended
  slices), so down-weighting it only perturbs the objective — the same lesson as P2a's
  physical edge.
- **Default extension depth suffices**: no_taper_deep == no_taper at 1e-5 (h=5 already
  covers the PSF+rounding coupling width).
- ~~Proposed library change: remove the sine taper~~ — **WITHDRAWN 2026-07-09 after
  Greg's real-data challenge; see the correction below.**

### P2c CORRECTION — real data overrules the synthetic (2026-07-09, Lilly D01788)

Greg: the synthetic verdict "doesn't match previous experience on real data" — the Lilly
D01788 NSI scan at commit 568f6b7 (2026-06-26, which has NO taper — the taper was added
later, evidently as the fix) shows stripes near the seam.  Reproduced and root-caused
(scripts `lilly_split_repro.py`, `lilly_split_ablations.py`; 4× downsample, view ss 2,
transmission_root weights; mag 4.69, **R/SID 0.21**, det_row_offset −3.9 rows, split at
slice 231):

- **Stripes confirmed**: ±40% every-other-slice zigzag over seam ±7–10 slices; RMS vs the
  matching unsplit recon spikes **74×** over background.  All recons ran the full 15 iters
  (change% 1.1–1.5% ≫ 0.2 stop) → per-half independent stopping ruled out.
- **Structural, not transient**: at 60 iterations the background converges 5× further but
  the seam spike PERSISTS (8.0e-3) — a converged artifact.  (Transient hypothesis
  refuted.)
- **Mechanism = P2a's axial truncation in miniature**: each half is truncated at its
  extension end with unexplained depth h_sino·(1+R/SID)·pitch − h_recon.  Lilly: 5·1.21 −
  5 ≈ **1.1 slices** → 74× stripes.  The original synthetic had R/SID 0.125 → **0.6
  slices** → invisible (2e-5): the synthetic verdict was an artifact of a fortunately
  narrow fan angle, and the sensitivity to fractional-slice mismatch is much steeper than
  assumed.  *(The fan-angle attribution was later SUPERSEDED — see the
  reproduction-closed bullet below: the wide-fan replication stayed clean; the real
  blind spot was the locked detector-row/slice grids.)*
- **Synthetic reproduction NOT yet achieved (honest gap, 2026-07-09)** — two attempts in
  `split_seam_repro.py`: `structured_widefan` (R/SID 0.2, unit weights, 40 iters;
  unexplained depth ~1.0 slices ≈ Lilly's) came back CLEAN (no_taper seam-vs-ref ~1e-4),
  so mismatch depth alone is NOT sufficient; `widefan_noise15` (photon noise +
  transmission weights + 15 iters) raised all variants to ~2e-3 but WITHOUT Lilly's
  signature ordering (taper 0.0026 ≥ no_taper 0.0020 ≥ deep 0.0016 — on Lilly the taper
  wins 11×) → noise-realization sensitivity, not the stripes.  Offset-asymmetry suspect
  weakened by inspection: the synthetic already carries a 0.5-row/slice fractional
  misalignment (even-count centered grids) vs Lilly's ~0.1–0.4; NSI's detector-rotation
  correction on this scan is a ~0.04° resample — also unlikely.  **A library-version
  confound ran through the whole comparison** (found 2026-07-09): the Lilly stripes come
  from 568f6b7 (6/26), ALL synthetic probes from `greg/kernel_investigation` post-kernel-
  campaign.  **RULED OUT same day** by `lilly_head_check.py` (the Lilly no-taper split
  re-run against the current branch, single variable = library version): the current
  branch stripes at full strength (peak per-slice seam RMS 7.3e-3 vs the 6/26 run's
  ~8e-3, >200× background) — the stripes are a property of the data + split geometry,
  not of either projector version, and the revised h_recon proposal applies to the
  current library as-is.  Provenance verified from records after Greg flagged checkout
  ambiguity (the repo hosting the editable install was switched 568f6b7 →
  kernel_investigation during the day): the git reflog (switch at 16:43) + volume
  timestamps confirm the repro (12:29) and ablations (16:12–16:31) ran at 568f6b7 and
  the head check (17:29) at e53d5c5, with the job log printing the live commit.
  GOTCHA for future version A/Bs: the conda env's editable install uses a META-PATH
  finder, so PYTHONPATH-ing a worktree would silently import the wrong version — always
  assert `mbirjax.__file__` inside the process.  Confirming variant (`lilly_shipped_check.py`,
  same day): the SHIPPED `split_sino_recon` (taper and all) at the current branch is
  CLEAN on Lilly — peak 4.1e-4 at the split, smooth decay, matching the 6/26-era taper
  ablation (6.5e-4) and Greg's stripes-went-away-with-the-taper experience — so the
  taper's fix carries over to the current library too.
- **The 8× regime probe (2026-07-09, Greg's request; `lilly_ds8_check.py` /
  `lilly_ds8_deep.py`): stripes persist AND the taper stops working.**  At 8× detector
  downsampling + 8× view subsampling (225 views, 235×187 detector, split at slice 116;
  the governing mismatch is scale-free, still ~1.05 unexplained slices) the no-taper
  split stripes at peak seam RMS 7.9e-3 — and the sine taper only trims it to **6.1e-3**
  (vs the 11× fix at 4×; banding visible in the tapered recon itself).  The
  geometry-derived extension still fixes it: **9.0e-4 at h_recon = 9 (the formula
  value), identical at h_recon = 12** — depth beyond the formula buys nothing, as
  cause-removal predicts.  Interpretation: the taper is a SUPPRESSOR whose effectiveness
  is regime-dependent (view count / per-bin statistics shifted the balance); the matched
  extension removes the cause and carries across regimes.  Practical upshot: **the
  shipped split stripes at coarse downsampling** — the h_recon proposal is a defect fix,
  not a cleanup.  8× = the fast-turnaround workhorse for further seam work.
- **Page rewrite (2026-07-09, Greg-approved storyline)**: `phase_2c_split_results.html`
  restructured real-data-first (problem → cause incl. the unexplained-band cartoon →
  interventions with windowed recon+difference montages at 4× and 8× → synthetic
  contributions + blind-spot lesson + honest gap → reproduction); the
  announce-then-retract record lives only here in the plan doc.  Figures:
  `lilly_variant_figures.py` (recon window 0–0.05 so the body spans the grayscale,
  difference ±0.025 ≈ p99 of |split−unsplit|; body 0.013–0.035, pins 0.17–0.2).
- **REPRODUCTION CLOSED + MECHANISM REFINED (2026-07-09 evening): the stripes' driver is
  the SUB-ROW MISALIGNMENT between the sino cut row and the recon split slice.**  The
  hunt (all in `split_seam_lilly8x.py`'s header + `lilly_consistency_check.py` /
  `lilly_cons2.py`):
  - Build-up rounds 1–4 at Lilly-matched R/SID 0.21 / 8×-scale ALL CLEAN (<2× bg): solid
    10× pins, transmission_root texture (exonerated by measurement too — Lilly line
    integrals are mild, p99 ~0.6, weights ≥~0.55), photon noise, segmented pins matching
    the real 1–3-slice chains (measured from the ref volume), row binning — LINEAR
    binning is PROVABLY inert (averaging line integrals = projecting an axially smoothed
    object: still consistent), transmission-space binning's Jensen gap too weak (~1e-3)
    at this contrast.
  - REVERSE ablation on the real problem (Greg's simplify-the-data direction): consistent
    sino (= forward projection of the ref recon) STRIPES 7.7e-3/298× → inconsistency
    ruled out; + unit weights STRIPES 8.5e-3 → weights ruled out; + axial offsets zeroed
    (odd slices → split exactly on iso) CLEAN 1.1e-5 → **740× drop, driver found**.
  - Forward confirmation, fully synthetic: det_row_offset ∈ {0.15, 0.30, 0.45} rows with
    the recon grid centered (relative mismatch = the offset): 7.3e-5 / 2.7e-4 / 3.3e-3 —
    steeply superlinear; smooth body at 0.45 = 3.3e-3, **object-INDEPENDENT** (the pins
    were only where the stripes were visible); every-other-slice zigzag ±15–19%
    one-sided above the split, decaying over ~10 slices — the Lilly signature, and the
    real Lilly point (mismatch |0.45−0.05| ≈ 0.4, 7.9e-3) sits on the synthetic curve.
    Figure `p2c8_offset_reproduction.png` (embedded in the page §4).
  - Mechanism refinement: at EXACT cut–split alignment the symmetric per-half end
    deficit is benign at h=5 even at R/SID 0.21 — the depth formula sets the scale but
    the misalignment breaks the halves' symmetry and converts it into the alternating
    seam disagreement.  Default synthetic models CANNOT show this artifact: they place
    both grids symmetrically, so cut and split come out mutually aligned (even the
    "0.5-offset" widefan case had both shifted together → relative mismatch 0).  Real
    scans position the grids independently (NSI: det_row_offset −1.95 rows,
    recon_slice_offset +1.95 slices → cut 0.05 off iso, split 0.45 off).
  - **NEW FIX CANDIDATE (plan only): align the split with the cut** — alignment alone
    took the consistent-Lilly seam 8.5e-3 → 1.1e-5 at default h_recon=5, better than the
    deep extension (9.0e-4), no extra slices.  IMPLEMENTABILITY CAVEAT (2026-07-09, late):
    at pitch ratio ρ=1 (the default) the row and slice grids are COMMENSURATE, so the
    mismatch is INVARIANT under every choice of cut row / split slice — alignment is only
    reachable via an OPT-IN sub-slice shift of the global recon grid (≤ δ_slice/2), which
    moves the output sampling vs recon() (not bit-comparable / registration-identical).
    When ρ≠1 an index search can reduce it without a shift.  Menu for Phase 4:
    geometry-derived h_recon as DEFAULT (margin absorbs the ≤0.5 mismatch) +
    `align_split_grid` opt-in, taper retired with the h_recon change (full analysis in
    `phase_2d_remedies.html`).
  - Method lessons: (a) synthetic blind spots now have two instances with one root —
    geometry symmetry; stress conclusions at real datasets' governing ratios AND with
    their broken symmetries; (b) when build-up search stalls, REVERSE-ablate the real
    failing case — it converges by construction (Greg's direction); (c) linear vs
    nonlinear binning: averaging line integrals is inert, count-space binning is not.
- **Phase 2d page ADDED (2026-07-09, Greg-requested): `phase_2d_remedies.html`** — the
  synthesis: per-case remedies with rationale, equations, implementation sketches, and
  pros/cons incl. deliberate do-nothing.  Verdicts: AXIAL = extend to the exact bound
  automatically in `auto_set_recon_geometry` (Greg's call: R computed from the RECON
  grid — composes with lateral padding; a sinogram-derived R would silently under-pad z
  after lateral cover-padding — and always-on because a holder virtually always leaves
  the FoV at one end, so truncation is the norm, waste ≤ the other end's ~5–10%, and no
  threshold exists; cost = default-shape change → re-baseline the regression
  dashboards).  REFINED PER GREG 2026-07-09 (late): compute the excess PER END —
  E_end = max(0, |v_end|·(SID+R)/SDD − H_iso/2) with v_end the detector row-edge heights
  INCLUDING det_row_offset (translate detector up → top wedge taller, bottom shorter; a
  symmetric scale under-pads one end by the offset term); ceil per end, clamp at 0,
  shift recon_slice_offset by (E_top−E_bot)/2; helical z-shifts attach per end (E_top at
  z_max, E_bot at z_min); implement via the model's own coordinate chain
  (recon_ijk_to_xyz → geometry_xyz_to_uv_mag → detector_uv_to_mn inverted at the row
  edges, worst case y=−R); side benefit = fixes today's det_row_offset-blind auto
  centering (NSI hand-compensates via recon_slice_offset).  `scale_recon_shape` STAYS a
  pure fixed scaler (Greg) — just WARN on uncompensated lateral growth.  Open
  implementation check: R = RoR-mask radius vs grid half-diagonal (√2 difference) —
  check what the projector actually updates.  Asymmetric side-view SVG added to the
  page (made strongly asymmetric per Greg; TL;DR reorganized to one linked bullet per
  case).  EDGE CASES analyzed on the page (Greg's question): det_channel_offset ≠ 0
  leaves the z formulas unchanged (oblique rays have smaller z-slope — dividing by SDD
  is conservative; the offset acts via the enlarged half-fan FoV → R, which flows
  through since R comes from the recon shape; also exposes the lateral auto-shape's own
  channel-offset blindness — sibling gap, out of scope); det_row_offset > H/2 is handled
  by the max(0,·) clamps unchanged (near end gets no extension; clamp activates already
  at offset ≈ (H/2)·(R/SID)/(1+R/SID)); add-only then wastes the unmeasured near end of
  the offset-blind base slab — the full fix would TRIM it to v_near·(SID−R)/SDD (the one
  appearance of the near-side factor), worth doing only for half-cone scans.
  Translation + multiaxis-parallel have analogous per-end bounds — future work,
  cone beam first (Greg). LATERAL = detect-and-warn only (cover not derivable, under-pad
  catastrophic, O(s²) memory — the deliberate do-nothing case; detection per Greg
  2026-07-09 = reuse the `_get_sino_indicator` support mask that
  `auto_set_regularization_params` already computes for `auto_set_sigma_y` — support
  touching the edge channels ⇒ truncated; free, no new threshold; its safety margin can
  miss very faint truncation, the benign direction); SPLIT = geometry h_recon
  default + `align_split_grid` opt-in + taper retired with it.  SPLIT REFINEMENTS per
  Greg 2026-07-09 (late): R from RECON parameters (as axial); ρ written explicitly as
  δ_row/(mag·δ_slice) — the physical content is the perpendicular-ray 1/mag conversion,
  = 1 at defaults, no "δ_row at iso" intermediate; page now states WHY h_recon is small:
  the seam sits at the perpendicular (iso) ray where cone divergence is minimal — §1's
  wedge evaluated at h≈0 (excess h_sino·ρ·R/SID ≈ 1 slice) vs the slab ends where it
  scales with the full detector half-height.  All plan-only.
- **Both fixes work on Lilly** (seam max vs matching unsplit): shipped taper 6.5e-4 (11×
  better); **geometry-derived deep extension (h_recon 12, NO taper) 5.7e-4 at 15 iters,
  decaying to 3.2e-4 at 60** — the deep extension restores normal convergence at the seam
  (the no-taper default-extension artifact does not decay).
- **REVISED proposal (PLAN ONLY — no code yet, per Greg 2026-07-09): set the recon
  overlap from geometry, h_recon = ceil(h_sino·(1+R/SID)·pitch_ratio) + psf margin** —
  after which the taper is unnecessary; whether to also retain the taper as
  belt-and-suspenders is Greg's call at implementation time.  Cost: a few extra slices
  per half.  (The campaign theme again: give the model support to explain the data rather
  than down-weighting data.)
- Methodological lesson (for lessons.md eventually): a synthetic validation inherits its
  geometry's blind spots — the fan angle silently gated this failure mode; stress
  synthetic conclusions at the real datasets' governing ratios before generalizing.
  *(The fan-angle attribution was later SUPERSEDED — the wide-fan replication stayed
  clean; the real blind spot was the locked detector-row/slice grids.  Final lesson form
  in the reproduction-closed bullet above.)*

