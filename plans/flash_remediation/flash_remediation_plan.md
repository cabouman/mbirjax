# Flash remediation — sinogram weight edge tapering and recon-support padding

**Created 2026-07-08.  Status: Phase 1 (synthetic characterization) in progress.**
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

Two separable "speeds convergence" effects to keep apart in measurements: (1) genuinely
faster interior convergence; (2) the flash no longer inflating the change-% stop metric, so
the run stops earlier.  Both are wins; the characterization measures interior-only
convergence to distinguish them.

## Candidate space (Phase 2)

- **Row taper** (cone/multiaxis): quarter-sine ramp per the precedent, width derived from
  geometry (rows whose rays leave the recon slab), applied to weights at both detector-row
  edges (or one edge, if only one side is truncated).
- **Channel taper:** shape ∈ {sine ramp, Tukey/cosine, erf} × width swept (let data pick the
  knee); data-adaptive gate — apply only when truncation is detected (edge-channel sinogram
  level above the air/noise floor), so non-truncated scans keep full data.
- **Recon-support padding** via `scale_recon_shape` (channel case: col/row scale; z case:
  slice scale) — the model-completion alternative; memory cost ~(scale)² laterally.
- Combinations (pad + taper).

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
  model; snapshot per iteration; compare against the center crop of the truth phantom.
  Each script includes an optional padded variant (`scale_recon_shape`) as a mechanism check:
  if padding removes the artifact, the model-support explanation is confirmed.
- **P2 — candidate sweep on the synthetic cases** (taper shapes/widths, padding scales,
  combinations; single-variable ablations).
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

## Findings

### Phase 1 — synthetic characterization (2026-07-08, local CPU)

**Illustrated report (self-contained, figures embedded): `phase_1_results.html` (this
directory).**  Scripts: `plans/experiments/flash_remediation/{lateral,z}_truncation_repro.py`
(+ figures in that directory's `figures/`, gitignored).  Both cases: cone beam, magnification 2, 128 views, 40
iterations, default parameters; truth = the phantom on the enlarged grid, metrics on the
default-grid center crop, NRMSE normalized by the truth RMS over the RoR cylinder.

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

**Implications for Phase 2:** support padding is the standard against which any taper must
be judged in BOTH axes.  The open value questions for a row/channel taper are cost (padding
in z is cheap — slices are the cheap axis; lateral padding costs ~scale² memory) and
whether a geometry-derived row taper can match the padded variant's end-slice quality at zero
memory cost.  Phase 2 should A/B: row taper vs z-padding (same case), channel taper vs
lateral padding, and padding scale sweeps to find the knee (how much support is enough).
