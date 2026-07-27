# Phase B — evaluating the per-granularity regularization schedule

(Companion to `sharpness_schedule_plan.md`; created 2026-07-25 after Phase A completed;
revised same day after a three-reviewer panel (design/statistics, algorithm/math,
implementation) — findings folded in.  Phase A record: `findings.html` in this
directory.  Scripts for this phase: `plans/experiments/sharpness_schedule/schedule/`.
Status: awaiting Greg's approval.)

## Goal and standard of proof

Evaluate the remedy the study was built around: **one (σx, σy) pair per granularity**
— conservative regularization on the coarse early subsets, target values on the fine
ones, implemented for experiments through the segmented driver's per-entry offsets
(zero library changes).  The standard of proof (plan of record): a **simple, robust
schedule that is at least as good as current behavior almost always** — judged by
pre-registered criteria, not post-hoc reading.

Phase A facts that shape the design: injection happens in the coarse phase
(iterations 0–2) and its footprint spans all early subsets; under-regularization's
dominant effect is a slower heal; at full resolution the artifact ends 5× the
downsampled final; the primary metric is the validated two-seed S_low; convexity
guarantees the schedule leaves the converged answer unchanged.

## B0 — noise calibration BEFORE any schedule run (panel: the thresholds must be
anchored to measured noise)

From the EXISTING Phase A runs (no new reconstructions): compute all three pairwise
two-seed S_low values (seed pairs 1-2, 1-3, 2-3) at every saved snapshot for the
baseline at both scales, and the seed-to-seed spread and iteration-13→14 decrement of
es_rmse.  These anchor the thresholds below.  The thresholds may be revised ONCE from
B0 — before any schedule run — and are frozen thereafter.  (Caveat, stated up front:
three pairs sharing seeds are correlated, ≈2 effective observations; the C1 margin is
sized to dwarf, not to estimate, this spread.)

## Pre-registered success criteria

All comparisons: same seeds, same protocol, same case, schedule vs the fixed-target
baseline.  **All Phase B runs use 17 iterations** (so the fallback is evaluable
without reruns); judgment is at iteration 14; the fallback anchor is baseline at 14.

- **C1 — streak benefit (primary).**  Two-seed S_low at iteration 14 reduced to
  **≤ 1/2 of baseline**, on BOTH the downsampled and full-resolution BGA; AND the
  peak two-seed S_low over the snapshot grid reduced to **≤ 0.7× the baseline peak**
  (the schedule should reduce injection, not merely speed the heal).  With three
  seeds, C1 passes only if EVERY available seed pair clears the threshold.
- **C2 — no harm to convergence.**  Arbiter at the downsampled scale: the
  **target-objective gap** — the objective evaluated at the TARGET parameters
  (weighted data term at σy★ plus the qGGMRF prior loss at σx★), computed per
  iteration in-stream — within **+0.5% (relative, provisional until B0)** of baseline
  at iteration 14.  At full resolution (where the full-volume prior loss is
  impractical): the σ-free residual es_rmse and the data term at σy★, same +0.5%.
  Interior NRMSE vs the converged reference is REPORTED but not gated: the reference
  is smoother than the target, which biases NRMSE toward conservatively-scheduled
  variants (panel finding).  Fallback (N = 2): a variant failing C2 at iteration 14
  may pass by meeting the same thresholds at iteration 16 — against baseline at 14 —
  with C1 still satisfied at 14; otherwise the depth is too aggressive.
- **C3 — no harm where nothing is wrong.**  On the synthetic control: es_rmse and the
  target-objective gap within **+0.5%** of baseline at iteration 14.
- **Diagnostics (reported, not gated):** footprint E(0) (should move toward 1 if
  injection is suppressed); R_z; alpha clip counts at BOTH bounds (the eps floor
  detects over-damped no-op iterations; max_alpha detects overshoot); the
  prior-to-data preconditioner share per iteration on a fixed pixel sample
  (downsampled only); full S_low trajectories.

**Decision rule (ordered, complete):**
1. Among variants meeting C1 + C2 downsampled, prefer the simplest and shallowest
   (smallest b; single-knob over joint).  A D/S tie at equal b breaks toward **D**
   (it leaves the prior threshold T·σx untouched, the lower-risk change to edge
   rendering).
2. The selected variant must confirm C1 + C2 at full resolution.  If it fails there,
   promote the runner-up for ONE more full-res confirmation; if that fails too,
   report the trade-off frontier and stop for discussion.
3. The confirmed winner must pass C3; on failure the runner-up gets C3 once.
4. The **b = 8 extension** for the best family fires only if BOTH depths are
   C1-null (response flat — the range is below the knee); "prefer shallowest"
   otherwise ends the search at the first passing depth.
5. No other threshold adjustment after B0.

## Variant space: three balance-matched families

The schedule is Δ(g) = −k·d(g) with granularity distance d = 3, 2, 1, 0 for the
default sequence's 4-, 16-, 64-, 128-subset entries (d is POSITIONAL — a
pre-registered family choice, not proportional to log₂ num_subsets).  Families are
matched in **balance units** (σx²/σy² moves as 10^(0.602·Δs + Δdb/10);
Δs = +1 ≈ Δdb = +6.02 dB).  Let b = balance decrement per level in dB:

| family | per-level offsets | coarsest level (d = 3) at b = 4 |
|---|---|---|
| **D** (data-side)  | Δdb = −b·d, Δs = 0 | snr_db −12 |
| **S** (prior-side) | Δs = −(b/6.02)·d, Δdb = 0 | sharpness −2.0 |
| **J** (joint)      | Δs = −(b/2)/6.02·d, Δdb = −(b/2)·d | sharpness −1.0, snr_db −6 |

Primary sweep: **b ∈ {2, 4} × {D, S, J}** = 6 schedule variants + baseline.  At
equal b, the D and S sigma pairs differ by an EXACT common multiplier during the
coarse phase, and every update term (direction, damping, line search, clips) is
invariant under common (σx, σy) scaling except the qGGMRF threshold factors
φ(|Δ|/T·σx) — so **any systematic S–D gap at equal b is a threshold effect**, and
its SIGN is the discriminator (panel: register both directions): S ≫ D says the
lowered early threshold protects edges from mis-classification; D ≫ S says lowering
the threshold pushes mid-size injected deltas into saturation during formation.

All variants run with the DC damping at its default (the shipped configuration);
schedule × damping-off interaction is deferred.

## Protocol

Common snapshot grid at BOTH scales, pre-registered:
**{0, 1, 2, 3, 4, 5, 9, 14, 16}** — panel: the old grids missed iteration 2 at full
res and iteration 3 everywhere, and iteration 3 (the first target-σ fine iteration
under a schedule) is exactly where a relocated peak would land.  Peak = max over
this grid.  Full-resolution baselines are FRESH runs on this grid (the Phase A
full-res runs cannot support the peak criterion).

1. **B0** (compute node, no recons): noise calibration above; freeze thresholds.
2. **b1 — search (downsampled BGA).**  6 variants + baseline, seeds {1, 2}; baseline
   and every C1-passing variant get seed 3.  17 iterations; per-iteration in-stream
   metrics (S_low/R_z, v1 S/control, es_rmse, data term at σy★, prior loss at σx★,
   NRMSE-vs-reference, alpha clip counts, preconditioner share); two-seed (all
   pairs) at the snapshot grid; footprint records.  Plus the **long-tail pair**
   (baseline vs winner, 40 iterations, 2 seeds, downsampled): registered merge
   criterion — es_rmse and objective gaps within the B0-measured seed spread and the
   S_low pair ratio within the B0 pair spread by iteration 40; failure flags the
   depth as too aggressive.  Plus the **synthetic no-harm** runs (baseline + winner,
   seeds {1, 2}) → C3.
3. **b2 — confirmation (full-resolution BGA).**  Fresh baseline (3 seeds) + the
   selected variant (3 seeds) [+ runner-up (2 seeds) only if rule 2 fires], on the
   common grid, cached A3 reference reused (pinned path; existence asserted; 23-min
   regen documented if scratch purged).  2-GPU memory-lean path.

**Visual evaluation (Greg, 2026-07-25): metrics are not the whole judgment.**  Every
run keeps a PERMANENT set of images, written before any cleanup: the final 4-panel
(reconstruction + error, axial and (x,z)) and per-iteration (x,z) error images at
iterations {0, 2, 3, 5, 14, 16}; the analysis pass adds two-seed field images per
variant on a matched window.  The winner selection includes a visual review of these
panels alongside the C1/C2 numbers.

Storage note: the common grid stores ~9 snapshots × ~9 GB × ~6–8 full-res runs
(~0.5–0.7 TB, scratch); snapshot volumes are deleted after the digest/two-seed pass
— the permanent artifacts per run are records.npz, final_recon.npy, and the images.

## Mechanics

- Schedules enter through the driver's existing `offsets_by_entry` — e.g. family D
  at b = 4: {2: (0, −12), 4: (0, −8), 6: (0, −4)}; all 128-subset entries (7–10)
  default to (0, 0), which also covers the 40-iteration slice.  Closed-form
  multipliers on the once-computed targets; zero extra qGGMRF compiles.
- **Consolidation first** (panel: the run/save/two-seed machinery is triplicated and
  diverged — and b1 needs a2's loader with a3's v2 metrics, exactly the seam where
  an omission would land): lift `make_hook` / `save_run` / `two_seed_curves` into a
  shared `driver/run_io.py` (v2 always on; z_step, disk-vs-memory snapshots, and
  multi-pair two-seed as options), then `schedule/b1_sweep.py` and
  `schedule/b2_fullres.py` reduce to loader + variant tables.  Runs are idempotent
  (skip-if-run-dir-complete) so incremental third seeds don't rerun the world.
- **Identity gates** (cheap asserts, pre-registered): `targets` bitwise-equal across
  all runs of a case; the fine-tail per-iteration (σx, σy) exactly equal the
  targets; partitions identical across variants at the same seed; and a one-time
  extension of the equivalence gate: `offsets_by_entry` of all-(0, 0) reproduces the
  baseline bitwise on CPU.
- References pinned to the exact Phase A paths (a2_bga/reference_recon.npy,
  a3_fullres/reference_recon.npy), read-only.
- Outputs to scratch `sharpness_schedule/{b1, b2}/`; logs to the scratch logs dir;
  jobs on the `sharpness_main` venv; findings land as a new section of the page
  (trajectories vs baseline, the C1/C2 table, E(0) and clip diagnostics).

## Risks and honest-reporting notes

- **C2 is the real hurdle:** conservative coarse iterations may leave the fine phase
  behind at iteration 14; the 17-iteration protocol makes the bounded fallback
  evaluable without reruns, and anything beyond it is a failed variant.
- **Selection over 6 variants risks winner's curse:** the full-res confirmation on
  fresh baselines is the guard; near-threshold downsampled calls get the third seed
  before selection.
- **Redundancy with the DC damping:** benefit is measured with damping on (the
  shipped config); if the schedule adds little on top of the damping, that is the
  finding — reported, not massaged.
- **The seed-independent remainder** (~56% of the full-res low band) is out of reach
  by construction; C1 is defined on the two-seed component only, and the page keeps
  that distinction explicit.

## Out of scope for Phase B

The prox path (gated off; semantics unchanged); other datasets and geometries
(Phase C validation breadth); library implementation and default-on policy (Phase C,
including the regression-fingerprint re-baseline and `annotations.yaml` marker);
schedule × damping-off interaction; the balance-collapse mechanism test in the streak
regime (revisit only if Phase B results are confusing without it).
