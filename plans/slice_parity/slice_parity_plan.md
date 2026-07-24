# Slice-parity alternation: plan and experiment design

**Started 2026-07-12** (Greg's proposal, discussed same day; branch `greg/gpu_headroom` —
the exploratory campaign this belongs to; scripts in `plans/experiments/slice_parity/`).
Companion background: `plans/bugs_and_artifacts/center slice noise/
center_slice_preconditioner_notes.md` (the convergence diagnosis and the two z-only
preconditioner designs this idea competes with / complements).

## Summary findings (study complete through R2, 2026-07-12)

Evidence base: two toy rounds plus three real datasets (Lilly ds8/ds4, z62) × two
sharpness settings each, all against 150-iteration references, judged at the 15- and
30-iteration budgets.  Details and tables in the R1/R2 sections below; the "How to
read" section explains the metrics.

1. **Recommendation — drop the granularity-0 iteration: `[2,4,6,7]` is the
   memory-friendly default candidate.**  Removing the single-subset full-volume
   iteration (wanted for memory reasons) costs nothing: `[2,4,6,7]` is equivalent to
   or slightly better than the default `[0,2,4,6,7]` in every tested cell, at both 15
   and 30 iterations.  The one visible trace is the first ~3 iterations on data whose
   initial error is dominated by low spatial frequencies (z62), recovered by iteration
   5–10 — only a ≤3-iteration preview would notice.  No new machinery required.
   **Visual evidence:** side-by-side reconstructions (reference | default | skip 0 |
   difference, all datasets, 15/20 iterations) published at
   `/depot/bouman/www/mbirjax/skip_0_results/index.html` (page source =
   `r2_recon_compare.html` in this directory; figures live on depot only — no PNGs in
   the repo, per Greg).
2. **Parity (z-phase alternation) is a quality option, not a default ingredient.**
   In the realistic operating envelope (sharpness ≤ 2.0, max_iterations = 15) it buys
   only ~4–5% error reduction, while costing +50% cone projector work per iteration as
   implementable today.  Its margin becomes material only when three axes are pushed
   together: sharpness (15–19% at the s2.5 extreme, 15 iterations), iteration budget
   (23–35% at 30 iterations), and problem size (23% → 35% from ds8 → ds4; the
   production-scale margin is the one open question).  Even then it is a
   quality-at-fixed-iterations gain, not a speed gain: with a free slice-set-aware
   cone forward kernel it would save only 1–6 iterations out of 15–30.  Worth
   revisiting only if that kernel ships (see the kernel-campaign section) or a
   high-sharpness quality-critical use case appears.
   **Narrative write-up with charts:** `slice_parity_findings.html` (this directory),
   published at `/depot/bouman/www/mbirjax/slice_parity_findings/index.html` — the
   mechanism, the null control, the state-dependence, and the error-reduction trend
   (charts by `plans/experiments/slice_parity/parity_findings_figs.py`; figures on
   depot only, no PNGs in the repo).
3. **Flat-fine sequences without a coarse start are unsafe as defaults** at
   interactive iteration budgets: flat-128 is mildly worse on Lilly and catastrophic
   on z62 (essentially still at the FDK start after 15 iterations at high sharpness).
   Large low-frequency initial error cannot be corrected by fine scattered subsets —
   directly relevant to the §2 default-sequence decision.
4. **Block aspect ratio (g1×2 vs g2×1) gives no durable real-data benefit**, despite
   consistent toy-phantom wins.  Four separate toy schedule findings failed to
   transfer to real data in decision-relevant form — treat toy-case schedule results
   as hypotheses to check, never as decisions.

## Notation glossary

- **`S`** — number of in-plane (scattered-pixel) subsets in a partition.
- **`P`** — number of z-phase classes: P=2 = even/odd slices, P=3 = mod-3
  ("parity-2"/"parity-3").
- **`g<n>`** — granularity index on the library's doubling ladder: 2^n in-plane subsets
  (g7 = 128, g0 = 1).
- **`g<n>×<p>`** — block shape: 2^n in-plane subsets × p z-phases (e.g. g1×2, g5×4,
  g4×8).
- **`pseq [0,2,4,6,7]`** — `partition_sequence`: granularity index per iteration; the
  library repeats the last entry for all later iterations ([0,2,4,6,7] = current
  default).
- **flat-[7] / flat-128** — constant sequence at granularity 7 (128 subsets) — no
  coarse start.
- **su/it** — sub-updates per iteration = S × P (one sub-update = one subset-×-phase
  VCD step).
- **u (idealized cost units)** — full-projection equivalents: a P-phase cone iteration
  costs (P+1)/2 — forward ∝ P because the vertical fan scales with detector ROWS,
  back ≈ 1 (phases sum to one full backprojection).  Mask-form as-implemented cost is
  higher (≈ 2× projector cost for P=2); a slice-set-aware forward kernel would restore
  ≈ 1×.
- **s1.0 / s2.5** — mbirjax `sharpness` = 1.0 / 2.5, set BEFORE
  `auto_set_regularization_params` (auto-regularization then frozen).
- **ds8 / ds4** — NSI preprocessing `downsample_factor` (8,8) / (4,4) with
  `subsample_view_factor` 8 / 4.
- **C0–C3** — the R1 candidate schedules (table in §R1): C0 default, C1 parity-all,
  C2 g1×2-ramp composite, C3 flat-128.
- **cropped log10 NRMSE** — log10 of ‖x − x_ref‖/‖x_ref‖ restricted to the interior
  disk (0.85 × ROR radius) with 10% of slices excluded at each axial end.
- **x_∞ / reference** — deep run of the same MAP objective used as ground truth:
  300 iterations (toy rounds), 150 (R1).
- **1.2× / 2.0× marks** — quality thresholds = multiples of C0's final 30-iteration
  cropped NRMSE, per sharpness.
- **P1 / F1–F3 / FX / R1** — experiment rounds: P1 = first mask-based A/B,
  F = follow-ups, FX = post-F probes, R1 = real-data schedule protocol.
- **f1_base / f1_pall** — F1 variants: default sequence plain / with parity-2 at every
  iteration.
- **red–black GS** — red–black Gauss–Seidel: update one z-parity class while the other
  stays fixed.
- **VCD** — vectorized coordinate descent — the library's block-update solver.
- **MM / block-MM** — majorization–minimization: each sub-update minimizes a surrogate
  that upper-bounds the true cost on the updated block, so descent is monotone.
- **MRF** — Markov random field (the prior's nearest-neighbor edge structure).
- **FDK** — Feldkamp–Davis–Kress, the direct (filtered-backprojection-family) cone
  recon used as the warm start.
- **ROR** — region of reconstruction (the recon cylinder); its radius sets the
  cropped-metric disk.
- **z62 / Lilly** — datasets: ORNL Versa Z62 (radial character, partition-study cache)
  / NSI D01788 autoinjector (flash-remediation workhorse).

## The idea

Instead of (or before) adding an axial preconditioner to the per-cylinder surrogate,
**update slice-parity subsets in alternation**: layer even/odd (or mod-3) slice phases on
top of the existing scattered-pixel subsets, using the same VCD update loop — i.e. extend
the partition from (in-plane random blocks) to (in-plane random blocks) × (z parity
classes).  Red–black Gauss–Seidel in z, inside the existing block-MM framework.

## Why it targets the diagnosed mechanism (analysis, 2026-07-12 discussion)

The preconditioner notes establish (§6.2): the scattered-cylinder subsets handle the
prior EXACTLY; the only coupling the per-block diagonal surrogate omits is the **data
term's axial structure**, and the AᵀA PSF is only 3–5 slices wide axially (§7).  Hence:

- **Tridiagonal case** (half-bandwidth 1): within a parity class, slices z and z±2 do not
  interact through the data term — the block's axial data Hessian becomes DIAGONAL, so
  the existing per-voxel diagonal surrogate is ~exact in z.  Coupling to the other
  parity enters the gradient exactly at fixed values (GS conditioning).  Symptom B's
  mechanism (coherent axial overshoot from underestimated axial stiffness) is removed at
  the source rather than damped.
- **Pentadiagonal case** (half-bandwidth 2): z↔z±2 survives parity-2; a mod-3 phase
  partition decouples it.  The notes' measurement §10.1 (tri vs penta) doubles as the
  parity-2-vs-mod-3 decision.
- **The prior loses nothing**: its z-edges cross to FIXED voxels — classic red–black GS
  on a nearest-neighbor MRF; exact conditioning, no majorization slack in z.
- vs the notes' designs: no separability assumption (Design B's crux), no banded
  factorizations or majorizer-validity/double-counting bookkeeping (Design A §8), and
  monotone descent is preserved (each phase sub-update is still a valid MM step on the
  full cost).  Neither parity nor A/B touches CROSS-cylinder data coupling (expected
  small at sparse subset spacing, notes §10.2); the families stack if a hotspot survives.

**Honest caveats to test, not assume:**

1. **Low-z-frequency tail.**  Red–black GS is a smoother; the diagnosed slow subspace is
   LOW-z-frequency.  The rejoinder: that error is INJECTED by the overshoot (notes §6.3)
   and parity fixes the injection; the FBP warm start covers pre-existing low
   frequencies.  If a slow low-z-frequency data subspace exists independent of
   injection, neither parity nor a banded majorizer fixes it (momentum/NLCG or the
   multi-resolution direction, current_plans §7, would).  The experiment's long-tail
   behavior discriminates.
2. **Cone forward cost.**  The cone forward vertical fan costs ∝ detector ROWS, not
   slices (`cone_beam.py` row-batch loop), and a stride-2 slice set still hits ~every
   row — so parity sub-updates do NOT halve cone forward cost without kernel changes
   (mask form ≈ 2×/pass; compact form ≈ 1.5×/pass).  The convergence experiment is
   deliberately cost-blind (mask-based); production efficiency is a separate, later
   question tied to the kernel campaign (slice-index-set support).

## Greg's compensated variant + conjecture (2026-07-12)

Halve the xy subsets while doubling z-phases: (S/2 in-plane) × (2 parities) = the same
number of projections per iteration as today.  **Conjecture (to test): convergence
improves overall, because each subset projection gives a more isolated view of each
voxel in cone beam for all voxels away from the center slice** (halving the co-updated
slices reduces within-update data interference; the added in-plane density costs little
— scattered pixels at 2× density are still prior-sparse).

## Experiment P1 — mask-based convergence A/B (`parity_convergence_ab.py`)

Mask-based: in an experiment-local copy of the VCD subset updater, the update direction
is multiplied by a per-slice 0/1 phase mask BEFORE the line-search forward projection
(one masking site; alpha then optimizes along the masked direction and the
error-sinogram update stays consistent).  No library changes; 2× projector cost per pass
is irrelevant for a convergence study.

**Variants** (same seed, same partitions where applicable):

| variant | in-plane subsets | z phases | sub-updates/pass |
|---|---|---|---|
| (i) baseline | S (library default at the repro's granularity) | 1 (all slices) | S |
| (ii) parity-2 | S | even/odd | 2S |
| (iii) parity-3 | S | mod-3 | 3S |
| (iv) compensated | S/2 | even/odd | S |
| (v) parallel-beam null control | S | even/odd vs 1 | — |

(v) is the discriminator: parallel-beam slices are already data-decoupled, so parity
should change ~nothing there — any effect seen is not the diagnosed mechanism.

**Test case:** the center-slice-noise cone repro (the notes' 40-slice low-β/high-
sharpness case; exact settings from `plans/experiments/bugs_and_artifacts/center slice
noise/center_slice.py`), FBP-scaled start, converged reference x_∞ from a long run.

**Metrics** (the notes' §2 diagnostics, reused): per-slice error ‖x_j − x_∞‖ vs
iteration — reported per PASS and per PROJECTOR-CALL-EQUIVALENT (cost-normalized, so
(ii)/(iii) are not flattered by doing more work); per-slice increment norms and lag-1/2/3
cosines; the slice-19 trajectory (does the bad early step disappear?); whole-volume
NRMSE.

**Decision reads:**
- Parity-2 fixes Symptom B (axial increment coherence collapses) and slice-19's bad
  step shrinks → the mechanism is confirmed; compare (iv) vs (i) at EQUAL cost for
  Greg's conjecture.
- Parity-3 ≈ parity-2 → tridiagonal coupling (consistent with notes §10.1); parity-3
  clearly better → pentadiagonal reach matters.
- A slow tail SURVIVES all variants → the low-z-frequency caveat is real; points to
  momentum/multi-resolution, not finer blocking.
- Null control shows an effect → mechanism attribution is wrong somewhere; stop and
  rethink.

**Follow-ups queued behind P1:** the notes' §10 measurements 1–3 (PSF bandwidth,
cross-cylinder magnitude, separability); the cone parity cost measurement (fwd wall of
a masked half-slice cylinder vs full); scale-up beyond 40 slices if P1 is positive.

## P1 RESULTS (2026-07-12, local CPU run; raw data + figures in
## `plans/experiments/slice_parity/results/`, gitignored — numbers recorded here)

Setup as designed: cone 128×40×128 cube phantom, sharpness 3.0, noiseless, flat
sequences, 20 iterations, 100-iteration converged references; self-check passed at
exactly 0 (copied updater bitwise = library).  Final whole-volume NRMSE and per-iteration
log10-NRMSE trajectories:

| variant | sub-updates/iter | final NRMSE | log10 NRMSE @ iter 0 / 18 |
|---|---|---|---|
| cone baseline [7] | 128 | 0.1416 | −0.656 / −0.841 |
| cone parity-2 [7]×2 | 256 | 0.1543 | −0.606 / −0.804 |
| cone parity-3 [7]×3 | 384 | 0.1540 | −0.606 / −0.804 |
| **cone compensated [6]×2** | **128** | **0.1347** | **−0.675 / −0.864** |
| parallel baseline / parity-2 | 128 / 256 | 0.04217 / 0.04218 | identical |

**Verdicts:**

1. **The null control is exactly null** (parallel: 0.04217 vs 0.04218) — parity's effects
   flow entirely through the cone data term's z-coupling, as the mechanism analysis
   predicted.  Attribution confirmed.
2. **Greg's compensated variant WINS at equal cost, at every iteration** — the
   conjecture is supported: at fixed sub-updates/iteration, blocks that are
   wider-in-plane × thinner-in-z (S/2 × 2 phases) beat the standard shape.  It leads
   from iteration 1 (−0.675 vs −0.656) and holds ~0.02 log10 through iteration 18.
   Caveat: its tail slope is slightly SHALLOWER (−0.0078 vs −0.0085 log10/iter over
   iterations 10–18), so the lead may erode in very long runs — needs a longer-run variant.
3. **Pure z-refinement at fixed in-plane granularity LOSES**: parity-2/3 are uniformly
   ~0.04 log10 WORSE than baseline per iteration despite 2–3× the sub-updates.
   Interpretation: at flat-128 granularity the overshoot-injection mechanism barely
   binds (fine subsets don't inflate much), while parity phases ANCHOR updated slices to
   fixed z-neighbors through the prior — red-black GS is a smoother, and the dominant
   post-FDK error is low-z-frequency, exactly where full-cylinder joint updates (which
   let whole columns move freely) are structurally better.  Net: block-shape trade, not
   free z-decoupling.
4. **Slice 19 (hotspot) fine structure**: parity-2 improves the EARLY hotspot behavior
   (−1.01 vs −0.90 after iteration 1 — the predicted interference fix) but loses its
   lead by ~iteration 10; compensated starts WORSE at slice 19 (−0.68; bigger in-plane
   blocks = more cross-cylinder interference early) then converges fastest, overtaking
   everything by ~iteration 5, with its lag-1 increment cosine DECORRELATING (1.0 → ~0.2
   around iterations 6–8) where baseline/parity stay pinned at 0.99 — compensated stops
   creeping along the single slow direction; the others don't.
   NOTE: the notes' Symptom-A transient (slice-19 error INCREASING early) did NOT
   reproduce under flat-[7] sequences — it belongs to the coarse-start default sequence,
   which P1 deliberately avoided.  The coarse-granularity regime (where overshoot
   injection is strongest and parity should bind hardest) is untested — follow-up F1.

**Follow-ups (proposed):** F1 — default sequence [0,2,4,6,7] with parity masks applied
only at the coarse iterations (tests the injection regime the mechanism targets most).
F2 — block aspect-ratio sweep at equal cost: granularity {5,6,7} × phases {4,2,1}
(is compensated the knee, or does wider×thinner keep winning?).  F3 — longer runs
(50+ iterations: does compensated's lead persist against its shallower tail slope?).
F4 — noise + weights + a larger case before any policy conclusion.

## F-ROUND RESULTS (2026-07-12, local CPU; deep 300-iteration reference — the P1
## 100-iteration reference sat 2.8e-3 relative above it, far below all observed errors,
## so P1's conclusions stand)

All cone, sharpness 3.0.  Final log10 NRMSE at 20 iterations unless noted:

| group | variant | log10 NRMSE | read |
|---|---|---|---|
| F1 (default seq [0,2,4,6,7]) | base | −1.507 | slice-19 BAD STEP reproduced (−0.988→−0.976 rise at iter 2) |
| | parity-2 at coarse iters only | −1.540 | dip softened (~0.03 log better through the transient), not removed |
| | **parity-2 everywhere** | **−1.596** | best 20-iter result in the study; slice 19 −1.339 vs base −1.268 at iter 9 |
| F2 fine (equal cost, 128 su/it) | g5×4 / g6×2 / g7×1 | **−0.941** / −0.869 / −0.848 | MONOTONE: wider×thinner keeps winning; knee NOT reached |
| | g7×2 (2× cost) | −0.811 | flat-seq parity still loses, replicating P1 |
| F2c coarse (equal cost, 4 su/it) | g0×4 / g1×2 / g2×1 | −1.099 / **−1.139** / −1.129 | NON-monotone: the all-pixels×¼-slices extreme loses — in-plane isolation cannot be traded away entirely |
| F3 (60 iters, flat) | base / compensated | −1.054 / **−1.073** | lead PERSISTS; tail slopes equalize (−0.0042 vs −0.0043/iter) — P1's shallower-slope worry resolved |

**Synthesis:**

1. **Parity's value is STATE-DEPENDENT, and the P1/F1 contrast explains it**: from an FDK
   start under flat-fine sequences (P1), parity-2 loses — the dominant residual is
   low-z-frequency, where red-black GS's prior anchoring drags.  After coarse iterations
   clear the low-z-frequency error (F1's default sequence), parity-2 helps at EVERY
   granularity — f1_pall beats everything while also largely fixing the slice-19
   hotspot.  This is exactly the smoother character the plan's caveat predicted, now
   measured from both sides.
2. **The aspect-ratio trade has an interior optimum**: at the fine end, wider-in-plane ×
   thinner-in-z wins monotonically as far as tested (g5×4 best; try g4×8); at the coarse
   end it reverses at the extreme (g0×4 < g1×2) — in-plane subset isolation remains
   necessary.  Block SHAPE is a real, free design dimension, not a parity-on/off switch.
3. **The coarse-start default sequence dominates flat-fine on this case** (−1.51 vs
   −0.85 at 20 iterations; even 60 flat iterations only reach −1.05).  CAUTION: this
   small noiseless cube phantom rewards coarse low-frequency corrections; the
   partition-sequence study's real-data evidence for flat tails need not transfer —
   flag for the §2 default-sequence work, do not conclude from a toy.
4. **Obvious next variant (not yet run): the combined recipe at equal cost** — coarse-start
   sequence with parity everywhere AND a compensated fine tail (e.g. [0,2,4,6,6,...] with
   phases 2 throughout ≈ f1_pall's quality at f1_base's cost), plus the g4×8 probe of
   F2's unreached knee.  Then F4 (noise + weights + larger case, GPU) before any policy
   discussion.

## FX PROBES (2026-07-12, same setup/deep reference; run during the Greg discussion of
## the general state-dependence principle)

| variant | schedule | final log10 NRMSE | read |
|---|---|---|---|
| fx_g4x8 | flat (16 subsets × 8 phases), 128 su/it | −1.033 | flat fine-end trend STILL monotone past stride 8 — the PSF-bandwidth P-cap hypothesis is FALSIFIED as stated; more likely the toy's coarse-in-plane preference in disguise |
| fx_comb | ramp [0,2,4,6,6…]×P2 (compensated tail, 128 su/it) | −1.467 | compensated tail LOSES to f1_base's (128,1) tail in the ramp context — the flat-context compensated win does NOT carry past a coarse start |
| fx_comb_g1x2 | ramp [1,3,5,6,6…]×P2 (Greg's (2,2) start) | −1.478 | **best EARLY trajectory of all variants** (iters 1–3: −1.030/−1.078/−1.146, ahead of f1_pall) — g1×2 is a better ramp start than granularity-1; its (64,2) tail then decays slower |

**Third confirmation of state-dependence, now in both directions:** flat-context winners
(compensated shape) lose in ramp context; parity wins only post-coarse; the flat
fine-end "monotone aspect-ratio trend" is largely the toy's coarse-preference wearing a
z-phase costume.  Single-operating-point optimization on this phantom has hit
diminishing returns — the toy has yielded the STRUCTURE (state-dependence, interior
optimum, ramp-start candidates); the decision now needs full SCHEDULES on real data.
Composite candidate the fx data suggests: **g1×2-start parity ramp → (128,1) fine tail**
(best-of-both at equal cost), with f1_pall's (128,2) tail as the paid quality ceiling.

## R1 — the real-data schedule protocol (drafted 2026-07-12; Greg-approved framing:
## bounded search, decision-oriented)

**Question.**  Which SCHEDULE through the (in-plane subsets S, z-phases P) grid should be
the default-candidate, judged on real data at honest cost?  The toy rounds established
the structure (state-dependence both ways; interior optimum; g1×2 the best ramp start);
this protocol makes the decision.

**Search space (bounded).**  P ∈ {1, 2} (P≥4 showed no ramp-context value and pays real
cone forward cost); S on the doubling ladder; schedules start at ≥4 blocks (Greg).
Four candidates:

| id | schedule (pseq × phases) | rationale |
|---|---|---|
| C0 | [0,2,4,6,7] × 1 | current default (control) |
| C1 | [0,2,4,6,7] × 2 | parity everywhere — the toy quality ceiling; tail cost 1.5× idealized |
| C2 | [1,3,5,7] × [2,2,2,1] | the fx composite: g1×2-start parity ramp → plain (128,1) tail; ≈ C0 cost |
| C3 | [7] × 1 | flat-128 (the §2 release-candidate control) |

**Dataset and settings.**  Wave 1: Lilly D01788 at ds8 (the flash-remediation workhorse;
NSI `compute_sino_and_params`, downsample (8,8), view-subsample 8), `transmission_root`
weights, sharpness ∈ {1.0, 2.5} set BEFORE `auto_set_regularization_params` (then
auto_regularize off) — the production-ish and hard cases.  Wave 2 (after wave-1 reading):
z62 (radial character) and a ds4 confirmation of the winner (ds8 recons are
interactive-size, where VCD is host-dispatch-bound — WALL times at ds8 do not transfer;
see cost accounting).  8 variants + 2 references ≈ under an hour on one H100.

**References.**  Per sharpness: 150-iteration default-sequence recon (stop forced off),
cached in the staging dir.  All candidates target the same MAP objective.

**Metrics (decision order).**
1. Cropped NRMSE vs reference — interior disk (0.85 × ROR radius, the flash-analysis
   pattern) AND axial ends excluded (10%), per the §2 flash-metric caveat — as a function
   of (a) iteration and (b) IDEALIZED COST UNITS.
2. Iterations/cost-to-quality: first iteration reaching {2.0×, 1.2×} of C0's final
   (30-iteration) cropped NRMSE.
3. Per-slice error profiles (the axial structure is the point) + hotspot trajectories.
4. Wall time reported but CAVEATED at ds8 (host-dispatch-bound size; also mask-form
   parity pays 2× projector cost that a slice-set-aware implementation would not).

**Cost accounting (cone-specific, decided 2026-07-12).**  A P-phase iteration costs,
in full-projection equivalents: back ≈ 1 (slice gathers halve per phase, phases sum to
1); forward ≈ P (the vertical fan's cost scales with detector ROWS, which parity does
not reduce — the slice-set-aware kernel would restore ≈1, see the kernel section).
So a P=2 iteration is charged (P·fwd + back)/(fwd + back) ≈ **1.5 idealized units**
(2.0 units as-implemented in mask form).  C2 is cost-matched to C0 except its three
ramp iterations; C1's tail runs at 1.5×.  This charging is what makes the comparison
honest for cone TODAY; the 1.0× charging becomes real only if the kernel campaign ships
slice-set-aware forward fans.

**Decision rule.**  A candidate displaces C0 if it reaches the 1.2× quality mark at
≤0.8× C0's idealized cost at BOTH sharpness settings and is never worse at the 2.0×
mark; C3 is judged by the same rule (it is the §2 candidate).  Ties → prefer the
simpler schedule.  Runs seeded per call (identical partitions/order across variants).

Script: `plans/experiments/slice_parity/parity_realdata.py` (+ `.slurm`), staging
`~/parity_lilly` on gautschi.

## How to read the R1 results

- **The metric.**  Every number below is cropped log10 NRMSE against a 150-iteration
  reference recon of the SAME cost function (see glossary): log10 of the relative RMS
  difference over the interior region.  **More negative = less error = better.**
  Anchors: −1.0 means the recon is 10% RMS away from the converged answer, −1.3 ≈ 5%,
  −1.7 ≈ 2%.
- **Differences.**  "Variant A is +0.10 log10 better than B" means A's error is 10^0.10 ≈
  1.26× smaller — a 21% error reduction at the same point in the run.  In this
  document a **+ difference is always an improvement** (error reduction).
- **Cost units (u).**  1u = the projector work of one ordinary VCD iteration.  A
  parity (P=2) iteration is charged 1.5u, because the cone forward cost scales with
  detector rows and masked phases don't reduce it (glossary "u").  So 30 ordinary
  iterations cost 30u while 30 parity iterations cost 45u — whenever a parity variant
  "wins at the same iteration count", it paid 1.5× per iteration to get there.
- **The displacement rule** asks: to reach a fixed quality level, does the candidate
  need ≤0.8× the cost units C0 needs (and never more at the looser 2.0× level, at both
  sharpness settings)?  Deliberately strict: a candidate that ties C0 on cost but ends
  at better quality "does not displace" — it becomes a quality OPTION, not a new
  default.

## R1 WAVE-1 RESULTS (2026-07-12, gautschi 1×H100, Lilly ds8)

Provenance: job 13472514 FAILED (all 8 variants "Unable to get Blas support" — the
orchestrator generated references in-process and its resident XLA pool starved each variant
worker's cuBLAS init; the runner now subprocesses every GPU step, commit e8e4261).
References from the failed job were valid and cached; rerun job 13473504 COMPLETED,
all 8 variants ok.  Raw arrays in `~/parity_lilly` on gautschi + local gitignored
`results/r1/`; analysis by `r1_analysis.py`.

Final cropped log10 NRMSE at 30 iterations (lower = better) [@ total cost, wall]:

| variant | s1.0 | s2.5 |
|---|---|---|
| C0 default | −1.3050 [30.0u, 59.6s] | −1.0303 [30.0u, 58.5s] |
| C1 parity-all | **−1.3378** [45.0u, 85.9s] | **−1.1467** [45.0u, 83.0s] |
| C2 composite | −1.3123 [31.5u, 61.8s] | −1.0298 [31.5u, 60.9s] |
| C3 flat-128 | −1.1912 [30.0u, 49.6s] | −0.9147 [30.0u, 49.9s] |

Cost units needed to reach the quality marks (lower = cheaper; marks are 2.0× and 1.2×
of C0's final error):

| variant | s1.0 2.0× / 1.2× | s2.5 2.0× / 1.2× |
|---|---|---|
| C0 | 7.0 / 22.0 | 7.0 / 22.0 |
| C1 | 10.5 / 30.0 | 9.0 / 24.0 |
| C2 | 7.5 / 22.5 | 7.5 / 23.5 |
| C3 | 12.0 / never | 13.0 / never |

**Verdict (protocol rule): NO candidate displaces C0** — none reaches the 1.2× mark
cheaper than C0, let alone at ≤0.8× its cost.  C0 stays the default-candidate.

**In plain terms:**

1. **C1 (parity at every iteration) reaches the best final quality of any candidate,
   but pays more than that quality is worth in cost units.**  At the same 30-iteration
   count its error is 7% lower than C0's at sharpness 1.0 and 23% lower at sharpness
   2.5 — but it spent 45u to C0's 30u, and measured per unit of work it trails C0 at
   every quality mark.  The strong sharpness dependence is the proposed mechanism
   showing through: at high sharpness the prior is weak, so the axial data coupling
   (the thing parity fixes) is a larger share of the remaining error.  The gain is
   spread across the interior slices (median per-slice error 43% lower than C0 at
   s2.5, iteration 30); Lilly has no single-slice hotspot for parity to fix.
2. **C2 (the memory-friendly ramp) ties C0 everywhere** — final quality within 1–2%
   relative error, cost-to-quality within half a unit.  The toy-predicted better start
   is visible per-iteration but is exactly cancelled by the 1.5u charge on its three
   parity ramp iterations.  (This tie becomes a positive result in the memory
   discussion below: C2 never runs the granularity-0/1 full-volume iterations.)
3. **C3 (flat-128, no coarse start) is clearly worse on real data at this iteration
   budget**: it never reaches C0's 1.2× quality mark and needs ~1.8× the work to reach
   the 2.0× mark.  The toy's "coarse start dominates flat sequences" transferred to
   real data.  Directly relevant to the §2 default-sequence question at interactive
   iteration budgets (caveat: 30 iterations, ds8; the partition-sequence study's
   longer-horizon evidence is a separate regime).
4. **Kernel-campaign coupling, quantified**: if a slice-set-aware cone forward kernel
   made a parity iteration cost 1.0u, C1 would reach C0's 30-iteration quality in 27
   iterations (s1.0) / 21 (s2.5) — a real but modest saving that still misses the 0.8×
   displacement bar at s1.0 (0.91×; s2.5 passes at 0.73×).  The kernel's payoff is
   "parity becomes a free quality upgrade at a fixed iteration budget", not a faster
   default — flagged for the kernel session, nothing implemented here.
5. Wall clock at ds8 is dominated by host dispatch, not projector work (C1 measures
   1.44× C0's wall against its 2.0× mask-form projector cost) — recorded, but not a
   decision input at this size.

## R1 WAVE-2 RESULTS (2026-07-12, gautschi 1×H100: z62 full grid + Lilly ds4)

Provenance: job 13474445, both cases sequentially in one job (runner made multi-case,
commit 943c646).  z62: sino (201, 512, 512) from the partition-study cache, recon
512×512×640, all four candidates × both sharpness.  Lilly ds4: sino (450, 470, 374),
recon 374×374×667, C0/C1/C2 only (C3's read was already clear; ds4 variants are ~3× ds8
wall).  Staging `/scratch/gautschi/buzzard/parity_{z62,lilly_ds4}`; compact per-variant
arrays mirrored locally to `results/r1_z62/`, `results/r1_lilly_ds4/`.

Final cropped log10 NRMSE at 30 iterations (lower = better) [@ total cost]:

| variant | z62 s1.0 | z62 s2.5 | ds4 s1.0 | ds4 s2.5 |
|---|---|---|---|---|
| C0 default | −1.680 [30u] | −1.577 [30u] | −1.114 [30u] | −0.860 [30u] |
| C1 parity-all | **−1.724** [45u] | **−1.711** [45u] | **−1.154** [45u] | **−1.044** [45u] |
| C2 composite | −1.675 [31.5u] | −1.579 [31.5u] | −1.118 [31.5u] | −0.868 [31.5u] |
| C3 flat-128 | −0.934 [30u] | −0.327 [30u] | — | — |

State at 15 iterations (Greg: max_iterations=15 is the near-term production setting),
all three datasets — C1's error reduction vs C0 at the same iteration count:

| | s1.0 | s2.5 |
|---|---|---|
| ds8 @15 iters | +4% | +15% |
| z62 @15 iters | +5% | +15% |
| ds4 @15 iters | +5% | +19% |
| (for contrast, @30 iters) | +7% / +10% / +9% | +23% / +26% / +35% |

**Verdict: the wave-1 conclusions replicate on both datasets — nothing displaces C0.**
One nuance: ds4 s2.5 is the single cell where C1 wins a cost race even at the 1.5×
charge (16.5u vs C0's 19.0u to the 1.2× mark) — its tail advantage there is that
large — but it still loses at s1.0 (27u vs 20u), so the both-settings rule fails.

**In plain terms:**

1. **C1's quality margin is systematic and grows along three axes**: sharpness (weak
   prior), iteration count (the advantage accrues in the tail), and problem size
   (s2.5 @30: 23% at ds8 → 26% at z62 → 35% at ds4).  The size trend means the margin
   at production ds1–ds2 scale is an open question — the corner where parity matters
   may be larger than these runs show.
2. **C2 ties C0 on all three datasets** (finals within 2% relative error, 15-iteration
   values within noise) — consistent, useful evidence that schedules starting at ≥2
   subsets lose nothing on real data.  Note its g1×2 ramp start was slightly SLOWER
   than C0's early iterations on z62 — the toy's "g1×2 is the best ramp start" did not
   transfer (third instance of toy→real non-transfer; treat all toy schedule reads as
   hypotheses only).
3. **C3 collapses on z62**: −0.93 (s1.0) / −0.33 (s2.5) vs C0's −1.68 / −1.58 — at
   s2.5 it is barely past the FDK starting point after 30 iterations (monotone but
   crawling), and at 15 iterations it has essentially not moved (−0.21).  z62's initial
   error is dominated by low spatial frequencies (its radial/ring character), exactly
   what fine scattered subsets cannot correct.  Strongest real-data evidence yet that
   a flat-fine default is unsafe at interactive iteration budgets.
4. **z62 has a modest interior hotspot** (slice ≈400 at ~4.7× the median slice error,
   s1.0); C1 trims it to ~4.3× and cuts the interior median 46% at s2.5 — same
   distributed-improvement character as Lilly.
5. Reference-depth caveat: at z62 s1.0 the best variants end within ~2% RMS of the
   150-iteration reference, so orderings in the last ~0.05 log10 lean on reference
   quality; the displacement verdicts don't (they're decided at much coarser marks),
   but per-variant final rankings closer than that should not be over-read.

## R1 synthesis — the decision frame (discussion with Greg, 2026-07-12)

Greg's production envelope: sharpness realistically ≤1.5–2.0 (2.5 is extreme);
max_iterations=15 for the near term; and the granularity-0 (1-subset, full-volume)
iteration is likely to be dropped for memory reasons.

1. **Parity is not a default-schedule ingredient.**  In the production envelope its
   benefit is ≈4–5% error reduction at s1.0 and ≤15–19% even at the s2.5 extreme
   (15 iterations), against +50% projector cost per iteration as implemented — and
   even at kernel-restored 1.0× cost the saving is only 1–6 iterations out of 15–30.
   It remains a QUALITY OPTION for high-sharpness, generous-iteration, quality-critical
   recons, economical only if the slice-set-aware cone forward ships (kernel-session
   coupling #1), with an open upside at production resolution (the size trend above).
2. **Dropping granularity 0 looks ~free on real data** — C2 never used it and tied C0
   on three datasets.  Remaining question for the memory-driven default: does plain
   `[2,4,6,7]` do as well without any parity compensation?  → R2.
3. **R2 (proposed, pending Greg's go):** variants D0 = `[0,2,4,6,7]` (control),
   D1 = `[2,4,6,7]`, D2 = `[g1×2, 4,6,7]`, D3 = `[g1×2, g1×2, 4,6,7]`,
   D4 = `[2,2,4,6,7]` (plain cost-control for D3); sharpness {1.0, 2.0}; ds8 + z62;
   report at 15 AND 30 iterations.  Memory note: in today's mask form a g1×2 update
   still allocates full-height voxel buffers, so ONLY D1/D4 deliver the memory win
   now; D2/D3's memory benefit needs a compact slice-set updater (kernel-session
   coupling #2).  The convergence question (is the g1×2 shape a better coarse start
   than g2×1 on real data?) is what D2/D3 answer.

## R2 RESULTS — memory-driven schedules (2026-07-12, gautschi 1×H100, job 13479215)

The two questions from the synthesis: (a) does dropping the granularity-0 full-volume
iteration cost anything on real data?  (b) is a g1×2 coarse start better than the plain
g2×1 it would replace?  Variants D0 `[0,2,4,6,7]` (control), D1 `[2,4,6,7]`,
D2 `[g1×2, 4,6,7]`, D3 `[g1×2, g1×2, 4,6,7]`, D4 `[2,2,4,6,7]`; sharpness {1.0, 2.0}
(the realistic band); ds8 + z62; new 150-iteration s2.0 references.  Summaries
`r2_summary.json` per staging dir, mirrored to `results/r2_{ds8,z62}/`.

Final cropped log10 NRMSE at 30 iterations (lower = better), and the 15-iteration
error reduction vs D0 (+ = better):

| variant | ds8 s1.0 | ds8 s2.0 | z62 s1.0 | z62 s2.0 | @15 iters vs D0 (4 cells) |
|---|---|---|---|---|---|
| D0 `[0,2,4,6,7]` | −1.3050 | −1.1199 | −1.6796 | −1.6570 | — |
| D1 `[2,4,6,7]` | **−1.3128** | **−1.1289** | **−1.6905** | **−1.6645** | +2% / +3% / +3% / +1% |
| D2 `[g1×2,4,6,7]` | −1.3146 | −1.1303 | −1.6770 | −1.6632 | +3% / +3% / +1% / +2% |
| D3 `[g1×2,g1×2,4,6,7]` | −1.3062 | −1.1200 | −1.6672 | −1.6551 | +0% / +0% / −2% / −0% |
| D4 `[2,2,4,6,7]` | −1.3046 | −1.1197 | −1.6808 | −1.6554 | +0% / +0% / −0% / −2% |

**In plain terms:**

1. **Dropping the granularity-0 iteration is free — slightly beneficial, even.**
   D1 ties or beats D0 in every cell at both the 15- and 30-iteration budgets, and
   reaches most quality marks ~1 cost unit sooner.  The only place D0 leads is the
   first 1–3 iterations on z62 (the full-volume update clears that dataset's large
   low-frequency initial error fastest: iteration-1 error −0.70 vs D1's −0.64/−0.60),
   and D1 has caught up by iteration 5–10.  Only a ≤3-iteration preview use case would
   notice.
   **Visual head-to-head (capture job 13481068, Greg's metrics-can-mislead check):**
   `r2_recon_compare.html` (figures in `r2_recon_figs/`) shows reference | D0 | D1 |
   difference at 15 and 20 iterations for all three datasets × sharpness {1.0, 2.0};
   full volumes for slice_viewer in depot at
   `/depot/bouman/data/mbirjax_metrics/slice_parity/` — `recons/` (D0/D1 at 15/20/30
   iterations, all cases) and `refs/` (the 150-iteration references, case-prefixed);
   scratch staging `/scratch/gautschi/buzzard/parity_recons/` is the purge-eligible
   original.  Read: D0 and D1 are visually indistinguishable in
   every panel; the signed differences peak at ≲5% of the display window, concentrated
   as speckle on object boundaries and faint bands at the axial ends/flash regions —
   no coherent structural difference in the object interior, on any dataset, at either
   budget.  Provenance note: the capture run is CONTINUOUS (production-like segments)
   rather than the instrumented restart-per-iteration harness, and its trajectories
   sit ~0.01–0.02 log10 from the R2 numbers — the D1 ≥ D0 ordering held in all 10
   reference-checked cells (including the 20-iteration checkpoint, not measured
   in R2).
2. **The g1×2 coarse start is not better than plain g2×1 in any durable way.**  D2
   shows a real but transient edge on z62 around iterations 3–5 (e.g. −0.99 vs −0.93
   at iteration 3, s1.0) that fades to a tie by iteration 10–15, and it pays 1.5u for
   its parity iteration.  D3 vs D4 (the two-coarse-iteration pair) is a tie within
   noise everywhere.  Fourth instance of a toy schedule read (F2c's g1×2 > g2×1) not
   transferring to real data in decision-relevant form.
3. **Repeating the coarse rung is mildly wasteful**: D3/D4 trail D1 in every cell —
   spending the extra iteration climbing the ladder beats spending it re-running the
   same granularity.
4. **Recommendation: D1 `[2,4,6,7]` is the memory-friendly default candidate.**
   Under the strict 0.8× displacement rule it "does not displace" (it ties rather than
   wins big), but the driver here is the memory constraint, not speed — and D1 meets
   it with zero quality cost at every production-relevant budget, no new machinery,
   and no parity/kernel dependency.  Feeds the §2 default-sequence decision; the gaxb
   variants need no further real-data work unless the compact slice-set updater ships
   for other reasons.

## Interaction with the GPU-headroom kernel campaign

If parity survives P1, the concrete change to the kernel work is an INTERFACE decision,
not a redirection: **the Pallas kernels should take an arbitrary slice-index set as a
first-class argument** (contiguous bands and strided parity sets are both just index
vectors to a custom kernel; nearly free up front, an annoying retrofit later).  Parity
also doubles the number of per-subset calls at ~half payload, raising the value of the
small-call fixed-cost work (the 32%-concat finding in `gpu_headroom_findings.md`), and
an efficient cone parity FORWARD needs the slice-driven vertical-fan restructuring that
naturally belongs to the custom-kernel effort.
