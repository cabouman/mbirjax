# Slice-parity alternation: plan and experiment design

**Started 2026-07-12** (Greg's proposal, discussed same day; branch `greg/gpu_headroom` —
the exploratory campaign this belongs to; scripts in `plans/experiments/slice_parity/`).
Companion background: `plans/bugs_and_artifacts/center slice noise/
center_slice_preconditioner_notes.md` (the convergence diagnosis and the two z-only
preconditioner designs this idea competes with / complements).

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
   iterations 10–18), so the lead may erode in very long runs — needs a longer-run arm.
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

| group | arm | log10 NRMSE | read |
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
4. **Obvious next arm (not yet run): the combined recipe at equal cost** — coarse-start
   sequence with parity everywhere AND a compensated fine tail (e.g. [0,2,4,6,6,...] with
   phases 2 throughout ≈ f1_pall's quality at f1_base's cost), plus the g4×8 probe of
   F2's unreached knee.  Then F4 (noise + weights + larger case, GPU) before any policy
   discussion.

## FX PROBES (2026-07-12, same setup/deep reference; run during the Greg discussion of
## the general state-dependence principle)

| arm | schedule | final log10 NRMSE | read |
|---|---|---|---|
| fx_g4x8 | flat (16 subsets × 8 phases), 128 su/it | −1.033 | flat fine-end trend STILL monotone past stride 8 — the PSF-bandwidth P-cap hypothesis is FALSIFIED as stated; more likely the toy's coarse-in-plane preference in disguise |
| fx_comb | ramp [0,2,4,6,6…]×P2 (compensated tail, 128 su/it) | −1.467 | compensated tail LOSES to f1_base's (128,1) tail in the ramp context — the flat-context compensated win does NOT carry past a coarse start |
| fx_comb_g1x2 | ramp [1,3,5,6,6…]×P2 (Greg's (2,2) start) | −1.478 | **best EARLY trajectory of all arms** (iters 1–3: −1.030/−1.078/−1.146, ahead of f1_pall) — g1×2 is a better ramp start than granularity-1; its (64,2) tail then decays slower |

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
see cost accounting).  8 arms + 2 references ≈ under an hour on one H100.

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
simpler schedule.  Runs seeded per call (identical partitions/order across arms).

Script: `plans/experiments/slice_parity/parity_realdata.py` (+ `.slurm`), staging
`~/parity_lilly` on gautschi.

## R1 WAVE-1 RESULTS (2026-07-12, gautschi 1×H100, Lilly ds8)

Provenance: job 13472514 FAILED (all 8 arms "Unable to get Blas support" — the
orchestrator generated references in-process and its resident XLA pool starved each arm
worker's cuBLAS init; the runner now subprocesses every GPU step, commit e8e4261).
References from the failed job were valid and cached; rerun job 13473504 COMPLETED,
all 8 arms ok.  Raw arrays in `~/parity_lilly` on gautschi + local gitignored
`results/r1/`; analysis by `r1_analysis.py`.

Final cropped log10 NRMSE at 30 iterations [@ total idealized cost, wall]:

| arm | s1.0 | s2.5 |
|---|---|---|
| C0 default | −1.3050 [30.0u, 59.6s] | −1.0303 [30.0u, 58.5s] |
| C1 parity-all | **−1.3378** [45.0u, 85.9s] | **−1.1467** [45.0u, 83.0s] |
| C2 composite | −1.3123 [31.5u, 61.8s] | −1.0298 [31.5u, 60.9s] |
| C3 flat-128 | −1.1912 [30.0u, 49.6s] | −0.9147 [30.0u, 49.9s] |

Idealized cost to reach the quality marks (multiples of C0's 30-iteration final):

| arm | s1.0 2.0× / 1.2× | s2.5 2.0× / 1.2× |
|---|---|---|
| C0 | 7.0 / 22.0 | 7.0 / 22.0 |
| C1 | 10.5 / 30.0 | 9.0 / 24.0 |
| C2 | 7.5 / 22.5 | 7.5 / 23.5 |
| C3 | 12.0 / never | 13.0 / never |

**Verdict (protocol rule): NO candidate displaces C0** — none reaches the 1.2× mark at
≤0.8× C0's idealized cost; C1 and C3 are also worse at the 2.0× mark.  C0 stays the
default-candidate.

**Reads:**

1. **C1 (parity everywhere) is a real QUALITY ceiling, not a speed win**: better final
   at both settings, and strongly sharpness-dependent — +0.033 log10 at s1.0 but
   **+0.116 log10 at s2.5** (23% lower NRMSE), consistent with the mechanism (weaker
   prior → the data term's axial coupling dominates the error budget).  Its gain is
   DISTRIBUTED across the interior (median per-slice error −43% vs C0 at s2.5, iter 30),
   not a localized hotspot fix — no interior hotspot exists on this dataset (per-slice
   profiles are monotone toward the axial ends for every arm).
2. **C2 (g1×2-ramp composite) is a cost-neutral wash on real data**: it leads C0
   per-ITERATION early (the toy's fx read replicates directionally) but the 1.5×-charged
   ramp iterations eat exactly that lead in cost units (marks within ~0.5u of C0
   everywhere; finals within 0.007/0.0005 log10).  The toy composite's promise does NOT
   carry to real data at honest cone cost.
3. **C3 flat-128 clearly loses on real data at 30 iterations** (never reaches the 1.2×
   mark; ~1.7–1.9× C0's cost to the 2.0× mark) — the toy's "coarse-start dominates
   flat" DID transfer.  Directly relevant as a negative datum for the §2 flat-sequence
   candidacy at interactive iteration budgets (caveat: 30 iters, ds8, two sharpness
   settings; the partition-sequence study's longer-horizon evidence is a separate
   regime).
4. **Kernel-campaign coupling, quantified**: with a slice-set-aware cone forward
   (P=2 charged ~1.0×), C1's 30 iterations would cost 30u and it reaches C0's final at
   ITERATION 27 (s1.0) / 21 (s2.5) — i.e. even then C1 misses the 0.8× displacement bar
   at s1.0 (0.91×; s2.5 passes at 0.73×).  The kernel's payoff is therefore "parity
   becomes a free quality upgrade at a fixed iteration budget", not a rule-displacement
   — flag for the kernel session, no implementation here.
5. Wall times at ds8 are host-dispatch-bound (C1 1.44× C0 wall vs 2.0× mask-form
   projector cost) — per protocol, not decision inputs.

**Wave 2 (per protocol):** z62 (radial character; cached
`z62_v4x_d4x_nv201_nch512` case from the partition study, loaded via
`load_preprocessing` + sidecar) with the full C0–C3 × {1.0, 2.5} grid, and a Lilly ds4
confirmation restricted to C0/C1/C2 (C3's read is already clear; ds4 arms are ~8×
ds8 cost).  Decision discussion with Greg after both land.

## Interaction with the GPU-headroom kernel campaign

If parity survives P1, the concrete change to the kernel work is an INTERFACE decision,
not a redirection: **the Pallas kernels should take an arbitrary slice-index set as a
first-class argument** (contiguous bands and strided parity sets are both just index
vectors to a custom kernel; nearly free up front, an annoying retrofit later).  Parity
also doubles the number of per-subset calls at ~half payload, raising the value of the
small-call fixed-cost work (the 32%-concat finding in `gpu_headroom_findings.md`), and
an efficient cone parity FORWARD needs the slice-driven vertical-fan restructuring that
naturally belongs to the custom-kernel effort.
