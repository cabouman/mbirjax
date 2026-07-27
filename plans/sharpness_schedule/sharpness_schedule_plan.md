# Sharpness / snr_db schedule — plan

(Plan of record for `plans/current_plans.md` §1: remediation of voxel-cylinder streaks
under large `sharpness` and/or `snr_db`.  Created 2026-07-25; revised same day after a
three-reviewer panel and Greg's direction (subset-footprint probe, one σ per
granularity, fine-start branch removed, standard of proof).  Supporting scripts:
`plans/experiments/sharpness_schedule/` (subfolders by topic).  Status: Phase A
starting.)

## Problem and hypothesis

At large `sharpness` and/or `snr_db`, early VCD iterations can develop streak artifacts
aligned with voxel cylinders (the z-parallel voxel groups VCD updates as a unit).
Working hypothesis (refined 2026-07-25):

> The coarse-subset iterations at the start of the default partition sequence should
> make mostly low-frequency corrections.  When the problem is under-regularized (the
> data-to-prior balance σx²/σy² is large) and the preconditioner is imperfect, a random
> subset's update instead injects spurious high-frequency content: cylinders inside the
> subset take large steps their out-of-subset neighbors don't.  Greg's refinement: many
> streaks likely arise from the FIRST updated subset (or first couple) of iteration 1 —
> the subsets that face the rawest residual.

Two candidate *persistence* mechanisms, deliberately kept separate (they scale
differently and imply different remedies):

- **(P-weak) prior weakness:** the restoring force is ∝1/σx² everywhere, so removal is
  simply slow at high sharpness; later smoothing would eventually work.
- **(P-sat) edge-preserving saturation:** injected differences that exceed ~T·σx sit in
  the saturated |Δ|^(q−1) regime (q=1.2) and are treated as edges; prevention is then
  essential.

The objective is convex (p=2, q=1.2 ≥ 1; quadratic data term), so there is no
multi-basin trapping: persistence can only be slow decay plus re-injection under
continued random subsets — and, for Phase C, no schedule prefix can change the limit
point, only the trajectory.

**The balance-factorization conjecture (to be carefully justified and documented, not
assumed).**  There is NO global invariance claim: the prior is edge-preserving by
design, so real object edges are EXPECTED to cross the threshold T·σx (Greg,
2026-07-25).  The defensible statement is a factorization: every update quantity is
built from data terms ∝ 1/σy² and per-difference prior weights (1/σx²)·φ(|Δ|/(T·σx))
(p=2; φ → 1 sub-threshold, φ ~ ds^(q−2) saturated).  Under balance-matched scaling
(σx, σy) → (cσx, cσy), every sub-threshold term keeps its data-to-prior ratio, and ALL
departure from trajectory invariance is carried by differences in the
transition/saturated region — real edges (always present) and streak-scale
differences.  There the saturated prior force scales as σx^(−q), beating the
sub-threshold σx^(−2) by c^(2−q): super-threshold differences feel a relatively
STRONGER restoring force at higher sharpness on the diagonal.  Honest coordinates:
(balance σx²/σy², threshold T·σx).  The empirical question is whether STREAK severity
collapses on the balance diagonal — judged on the TWO-SEED metric, which cancels
seed-independent edge rendering (the reference-based map conflates legitimate
edge-rendering changes with streak changes along the diagonal).  Scope caveats:
defaults p=2, T=1.  The full derivation goes in the findings doc; the conjecture
earns its place only if the Phase A collapse test supports it.

Predictions the experiments test:

1. Streaks form in the FIRST iteration (4 subsets in the default sequence
   `[2,4,6]+[7,8,9,10]…`) and decay slowly relative to the iteration budget (a
   long-tail run measures the decay *rate*, discriminating rate-limited from
   equilibrium-limited persistence).
2. **Footprint attribution:** the xy-footprint of streaks is enriched in the
   first-updated subset(s) of iteration 1, decaying with update rank.
3. **Balance collapse, with a signed departure:** on a balance-matched diagonal
   (Δsnr_db = −6.02·Δsharpness; σx ∝ 2^sharpness, σy ∝ 10^(−snr_db/20)), streak
   severity on the TWO-SEED metric is flat if sub-threshold dynamics dominate
   (P-weak), and DECREASES with sharpness if saturation of streak-scale differences
   participates (P-sat: injection is approximately balance-invariant while the
   threshold grows ∝ σx).  An increase along the diagonal would falsify the framework.
4. Severity depends on the subset-common-mode kick: the cone DC damping
   (`cone_beam.py` `_get_update_direction`, default `(a,b,p,c)=(0.25,100,0.7,0.5)`;
   `a=c=1` disables) damps the per-slice H⁻¹-weighted subset-mean of the total
   gradient, and disabling it is already known to make streaking worse.
5. Conservative σ on the coarse-granularity iterations prevents formation, without
   changing the final objective (the fine-granularity tail runs at target).

## Evidence so far

- Serious streaking on full-res `bga_no_hart` (Zeiss BGA scan, 2401→601 views at full
  res; registry settings snr_db=35, sharpness=1.5).  Possibly also at the downsampled
  registry setting.
- Disabling the cone DC damping (`a=c=1`) makes streaking more severe.
- Repeated fine-granularity (128-subset) sequences fail badly on some real data — the
  fine-start remedy branch is OFF the table, and the default `[2,4,6,7…]` sequence is
  fixed for now (compatibility/social constraints).  The remedy under study is the σ
  schedule alone.
- Observed on cone beam; parallel beam not yet checked.

## Standard of proof and deliverables

We are looking for a SIMPLE, ROBUST remedy that works at least as well as the current
code almost all of the time — backed by well-designed experiments, not exhaustive
ones.  The major findings are organized as a clear, convincing, relatively short
document WITH FIGURES, written to inform a technical lead (Greg or Charlie Bouman) for
a final go/no-go decision: a self-contained HTML findings page under
`plans/sharpness_schedule/` (flash-remediation pattern; publishable to the depot www
for review; LaTeX distillation later if needed).  Decision-relevant numbers live in
the committed findings doc, never only in gitignored `results/` or scratch.

## Plan outline

- **Phase A — reproduce and measure.**  Synthetic cone at 256³ with a ground truth
  phantom, plus the real BGA case; a validated streak metric with a reference-free
  primary discriminator; severity map oriented on the balance diagonal; mechanism
  probes (subset-footprint attribution, saturation, coarse-late); parallel-beam probe.
- **Phase B — evaluate the schedule.**  One-(σx, σy)-per-granularity variants via the
  segmented `vcd_recon` checkpoint driver (no library changes); pre-registered success
  criteria.
- **Phase C — implement and validate.**  Internal granularity-keyed schedule in
  `vcd_recon` (public API unchanged), correct under `first_iteration` restarts and
  gated off the prox path, default gated on the IQ evidence.

## Schedule structure (Greg, 2026-07-25)

ONE (σx, σy) pair per granularity: the schedule is a static map from the partition-
sequence entry's granularity to offsets (Δsharpness, Δsnr_db), with the finest
granularity pinned at (0, 0) — target values.  Within a granularity the σs never
change (no moving target; the objective changes only when the granularity does, and
the fine-granularity tail runs entirely at the target objective).  Consequences:

- Zero extra qGGMRF compiles: one σx per granularity, and each granularity's kernel
  compiles once anyway (static `qggmrf_params` + per-granularity subset shapes).
- `first_iteration` correctness is inherited from the existing partition-sequence
  slicing — the σs follow the granularity of the sliced entry.
- Search space: one-parameter slope families Δ(g) = −k·d(g) with granularity distance
  d(g) = levels below finest (d = 3,2,1,0 for the default 4/16/64/128-subset ramp).
- Documented semantics: granularity-keyed and stateless — a sequence revisiting
  coarse granularity late would re-apply that granularity's conservative σ.  Resolved
  2026-07-25 (Greg): fine-to-coarse sequences have never been found beneficial (theory
  and practice: resolve low frequencies first with coarse subsets, then refine), so
  the scenario is hypothetical and no clamp-to-target state is added.

## Phase A — reproduction + metric

**A1, synthetic cone 256³.**  Ground truth phantom: a case-matched "ball grid"
(uniform block + lattice of small dense spheres, mimicking the BGA solder balls);
Shepp-Logan is the named fallback only if generality is later questioned.

*Driver and RNG discipline (the instrument is validated before its readings are
trusted):*

- Segmented driver runs `vcd_recon` one iteration at a time via `return_checkpoint` /
  `init_error_sinogram` / `fm_hessian`.  Seed np.random ONCE per variant; generate
  partitions ONCE (same args `initialize_recon` uses, incl. `use_ror_mask` and output
  device); each segment then consumes exactly one subset-order permutation, so the
  stream matches a continuous run exactly.  Do NOT reseed per segment (that would
  repeat the same subset order every iteration).
- The driver saves the np.random state before each segment and replays it afterward to
  recover the segment's subset-order permutation exactly (one consumer per iteration,
  panel-verified) — required by the footprint probe.
- Baselines run through the SAME segmented driver (also avoids the
  `auto_regularize_flag` trap: a plain `recon()` re-auto-sets sigmas and erases
  overrides).
- One-time equivalence gate: segmented vs continuous `vcd_recon` at fixed parameters
  must agree to float noise before any driver-based curve is used.
- Fixed ruler: per-segment `fm_rmse` is divided by the *scheduled* σy; rescale by
  σy(i)/σy★ (exact — the loss is ∝1/σy) or compute the σ-free residual RMSE from the
  checkpoint's error sinogram via device-side reductions.  Log per-iteration
  `alpha_values` and flag clipping at `max_alpha=1.5` (free overshoot evidence).

*Streak metric:*

- **Primary (reference-free) two-seed discriminator:** d = recon(seedA) − recon(seedB)
  at identical settings.  Partition-draw-driven streaks decorrelate across seeds while
  true structure and systematic edge error do not, so the z-coherent in-plane
  high-pass energy of d isolates the hypothesized stochastic signature — and works
  identically on synthetic and real data.  (If reference-based severity is high but d
  is quiet, the artifact is data-anchored and the hypothesis is wrong — informative
  either way.)
- Secondary, per-run reference map: e = recon − ground truth phantom (real data: −
  converged reference, relative curves only); z-coherent part c(x,y) = interior-slice
  mean of e; S = in-plane high-pass energy of c over the ROR interior, with the
  z-incoherent high-pass energy as a control.  The per-run map is also the input to
  the footprint probe (membership is per-run, so the two-seed map cannot attribute).
- Validation: metrics must rank the known orderings (damping off > on; streaky
  settings > conservative) and match visual panels (mid-slice + (x,z)).

*Sweep and mechanism probes (each one driver run unless noted):*

- Severity map on a small grid oriented along/across the −6.02 dB-per-sharpness-unit
  balance diagonal (prediction 3), with ~3 partition seeds at key variants.
- Formation/decay: per-iteration metric curves; one long-tail run (50–100 iterations)
  at the worst variant for the decay rate (prediction 1).
- **Subset-footprint probe (prediction 2):** enrichment curve E(r) = mean per-run
  streak-map magnitude over the pixels of the rank-r-updated subset, normalized by the
  overall mean — computed for iterations 1–2, ~3 seeds.  Expect E(0) ≫ 1 with decay
  across ranks if the first-subset mechanism holds.
- Coarse-late probe: several fine iterations first, then one 4-subset iteration at
  target σ — injection there means coarse granularity (not early state) drives
  injection, so conservative σ must cover coarse iterations whenever they occur.
- Saturation probe (P-weak vs P-sat): track streak amplitude relative to T·σx per
  iteration, plus one control run with `q=2.0` (no saturation) — persistence at q=2
  refutes P-sat as the persistence mechanism.  Complements prediction 3's signed
  diagonal test.
- Noise on/off variants (noise-driven vs dynamics-driven injection): hold the WEIGHTS
  fixed across the two (noise changes weights, hence the preconditioner — a confound);
  transmission/Poisson noise at a level roughly matched to the BGA scan; noise seeded
  independently of partitions.

**A2, real data.**  Loading via the IQ pipeline's `load_sino_and_model`
(`experiments/IQ_evaluation/run_recons.py`), then the same segmented driver (the
pipeline's own `recon()` call is monolithic — it cannot drive schedules).  Downsampled
first for cycle time; full-res on gautschi.  Place sinogram/weights on device once per
variant; compute metrics in-stream (device-side reductions) — full-res per-iteration
volumes are tens of GB and are not persisted.  Two-seed discriminator is the primary
metric here too; footprint probe runs here as well (membership known per run).

**A3, parallel beam.**  A 2-variant probe (worst + one mid variant), not a full sweep.
Note the asymmetry: `ParallelBeamModel` has no DC damping (only cone overrides
`_get_update_direction`), which biases parallel toward MORE streaking — so a clean
parallel result is *stronger* evidence that cone weighting/geometry is implicated.

**Phase A deliverable:** the findings page with the severity map, formation/decay
curves, footprint-enrichment curves, the validated metric, the saturation/coarse-late
outcomes, and a go/no-go for the schedule remedy (prediction 5 then tested in
Phase B).

## Phase B — schedule evaluation (sketch; superseded by `phase_b_plan.md`, the
panel-reviewed plan of record for this phase, 2026-07-25)

Variants via the same driver, applying the granularity-keyed offsets through the
closed-form multipliers sigma_x(g) = sigma_x★·2^Δs(g),
sigma_y(g) = sigma_y★·10^(−Δdb(g)/20).  Primary 1D slope sweeps first: sharpness-only
(k_s), snr_db-only (k_db), and balance-diagonal-jointly; k chosen so the coarsest
granularity spans a conservative-to-mild range.  Controls after the primaries, guided
by the Phase A map: max_alpha-capped variant (cheap alternative remedy), schedule ×
DC damping interaction (note the damping also damps the prior's restoring common mode
at conservative settings).

Success criteria: pre-registered NUMERIC thresholds set at Phase B kickoff from the
Phase A map (streak metric within X% of the conservative baseline; final quality
within Y% of the fixed-target converged result; ≤ N extra iterations), across
synthetic and BGA — judged against the standard of proof above (at least as good as
current code almost all of the time; simple and robust over optimal).

## Phase C — implementation (sketch) and settled constraints

Mechanism findings from the code (2026-07-25 analysis, panel-verified):

- `fm_constant = 1/sigma_y²` is an eager Python float in the subset-updater closure
  (`tomography_model.py:3573`, applied :3656/:3659/:3784) — free to change per
  iteration; the Hessian diagonal is sigma_y-independent; the stats jit takes sigma_y
  traced.
- `sigma_x` is baked into the qGGMRF jits (static `qggmrf_params`, `qggmrf.py:71`
  etc.); with one σx per granularity this costs ZERO extra compiles (each
  granularity's kernel compiles once regardless).
- The loop state (recon, error_sinogram, fm_hessian) is regularization-independent, so
  per-granularity parameter changes need no state fix-up; convexity (above) guarantees
  the limit point is unchanged by the schedule (the fine tail runs at target).
- Implementation seam: recreate the (cheap, pure-Python) subset-updater closure at
  iterations where the granularity — hence the scheduled σs — changes, passing
  explicit overrides.

Settled design constraints (Greg, 2026-07-25):

- The schedule is managed INTERNALLY — not settable via `set_params`; the public API
  does not change.  (For experiments, an internal attribute in the style of
  `_dc_damping` is acceptable.)
- One (σx, σy) per granularity; finest = target (see Schedule structure).
- `first_iteration` correctness in BOTH `recon` and `prox_map`: granularity-keying
  inherits correctness from the partition-sequence slicing in `initialize_recon`, but
  two panel-verified specifics still need explicit handling:
  `prox_map(do_initialization=False)` reuses cached `prox_data` and never re-slices
  (`first_iteration` is display-only inside `vcd_recon`), and the `fm_constant` seam
  is SHARED with the prox path — so the schedule must be gated on `prox_input is None`
  explicitly, not assumed inert.  (Note the default prox sequence slice is all-coarse:
  an ungated schedule would run prox entirely at conservative σ.)
- Reporting keeps a fixed ruler: `fm_rmse` continues to use the target sigma_y;
  `recon_params` records the schedule actually applied.
- Suppress the `stop_threshold_change_pct` stop while running at non-target σ (for the
  default sequence: the coarse warmup iterations).
- A default-ON schedule changes default recon outputs: plan the regression-fingerprint
  re-baseline + `annotations.yaml` marker with the change (standing metrics-interplay
  note), and gate the default on the Phase A/B IQ evidence.

## Risks / open questions

- **256³ may not reproduce:** the BGA case is high-resolution with strong-contrast
  metal; if 256³ synthetic stays clean, fall back to the downsampled BGA as the
  workhorse and treat synthetic as a mechanism probe only.
- **Streaks may not be schedule-fixable:** if injection persists at conservative
  settings, or the footprint probe shows no first-subset enrichment, the mechanism
  story changes and the remedy moves toward preconditioning — the Phase A probes
  decide.
- **Metric residual risk:** the two-seed discriminator assumes streaks decorrelate
  across partition draws; if streaks turn out data-anchored (fixed locations), the
  reference-based S and visual panels carry the conclusion instead — the disagreement
  itself is diagnostic.
- **Cone-specific physics:** if A3 is clean, the cone weighting (vertical-fan path
  weights, DC damping's regime) is implicated and the schedule may need to compose
  with a cone-specific fix.

## Organization

- Scripts by topic under `plans/experiments/sharpness_schedule/`:
  `driver/` (segmented driver + equivalence gate — shared by all topics),
  `repro/` (A1 severity map, formation/decay, A3 parallel probe),
  `mechanism/` (footprint, saturation, coarse-late, collapse probes),
  `real_bga/` (A2 loaders + gautschi job scripts),
  `schedule/` (Phase B variants).
  Run parameters in YAML/top-of-file constants — no CLI args.
- Findings: `plans/sharpness_schedule/` — this plan + the self-contained HTML findings
  page (figures embedded; publishable to depot www).
- Runs: gautschi H100 via sbatch (`ai` partition, `--cpus-per-task=14`/GPU); local Mac
  for smoke tests only.  Large outputs on gautschi go to scratch
  (`/scratch/gautschi/buzzard/sharpness_schedule/`; purge-eligible — durable numbers
  and figures land in the committed findings page).
