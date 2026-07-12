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

## Interaction with the GPU-headroom kernel campaign

If parity survives P1, the concrete change to the kernel work is an INTERFACE decision,
not a redirection: **the Pallas kernels should take an arbitrary slice-index set as a
first-class argument** (contiguous bands and strided parity sets are both just index
vectors to a custom kernel; nearly free up front, an annoying retrofit later).  Parity
also doubles the number of per-subset calls at ~half payload, raising the value of the
small-call fixed-cost work (the 32%-concat finding in `gpu_headroom_findings.md`), and
an efficient cone parity FORWARD needs the slice-driven vertical-fan restructuring that
naturally belongs to the custom-kernel effort.
