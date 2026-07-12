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

## Interaction with the GPU-headroom kernel campaign

If parity survives P1, the concrete change to the kernel work is an INTERFACE decision,
not a redirection: **the Pallas kernels should take an arbitrary slice-index set as a
first-class argument** (contiguous bands and strided parity sets are both just index
vectors to a custom kernel; nearly free up front, an annoying retrofit later).  Parity
also doubles the number of per-subset calls at ~half payload, raising the value of the
small-call fixed-cost work (the 32%-concat finding in `gpu_headroom_findings.md`), and
an efficient cone parity FORWARD needs the slice-driven vertical-fan restructuring that
naturally belongs to the custom-kernel effort.
