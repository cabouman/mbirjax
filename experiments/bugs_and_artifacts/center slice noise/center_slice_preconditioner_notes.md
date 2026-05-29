# CT Reconstruction: Convergence Diagnosis & Axial Preconditioner Design

*Working notes — block-GS subset solver for weighted-L2 + qGGMRF MAP reconstruction,
implemented in JAX, targeting large multi-GPU recons. Written to be readable cold by a new
collaborator (or by us in six months).*

---

## Table of Contents

1. [Setup and problem statement](#1-setup-and-problem-statement)
2. [Notation](#2-notation)
3. [The question that started this](#3-the-question-that-started-this)
4. [Observed symptoms](#4-observed-symptoms)
5. [The investigation: hypotheses raised and discarded](#5-the-investigation-hypotheses-raised-and-discarded)
6. [Synthesis: what is actually going on](#6-synthesis-what-is-actually-going-on)
7. [Decision: repair block-GS with an axial preconditioner](#7-decision-repair-block-gs-with-an-axial-preconditioner)
8. [Design A — voxel-cylinder banded preconditioner](#8-design-a--voxel-cylinder-banded-preconditioner)
9. [Design B — slice-at-a-time tridiagonal preconditioner](#9-design-b--slice-at-a-time-tridiagonal-preconditioner)
10. [Measurements to take before building](#10-measurements-to-take-before-building)
11. [Deferred / parked items](#11-deferred--parked-items)
12. [One-paragraph status](#12-one-paragraph-status)

---

## 1. Setup and problem statement

We reconstruct a 3D volume `x` from cone-beam (CT) projection data `y` by minimizing a standard
Bayesian/MAP objective:

```
f(x) = (1/2) ||A x - y||_W^2  +  β R(x)
```

- **Data term** `(1/2)||Ax - y||_W^2`: weighted least squares. `A` is the cone-beam forward
  projector, `W` a diagonal statistical weighting on the sinogram. This term is **exactly
  quadratic** in `x`; its Hessian `A^T W A` is constant.
- **Prior term** `β R(x)`: a **qGGMRF** (q-generalized Gaussian MRF) with **axis-only
  nearest-neighbor coupling** — each voxel couples only to its 6 face-neighbors (±x, ±y, ±z),
  no diagonal/edge neighbors. The potential is quadratic near zero (**q0 = 2**, so the
  curvature at the origin ρ''(0) is finite) and transitions to a near-linear tail for large
  differences to preserve edges.
- **No nonnegativity constraint** is enforced (so there is no prox; "FISTA" here would just be
  accelerated gradient descent).

**Current solver.** Diagonally-preconditioned gradient descent organized as an *approximate
block Gauss–Seidel* (block-GS) scheme:

- Start from an **FBP/FDK** reconstruction, scaled to minimize the L2 norm of the error
  sinogram. This warm start solves the low spatial frequencies nearly exactly.
- Take one full gradient step, then proceed through **increasingly fine subsets**: partitions
  of 4, 16, 64, then 128 to the end. Each subset sub-update takes an (over)relaxed
  surrogate-optimal step that minimizes the *full* cost over the block's voxels.
- The step size uses a **separable quadratic surrogate (SQS)** majorizer — a per-voxel diagonal
  upper bound on the curvature — plus a quadratic surrogate for the prior.

**Subset structure (important — drives everything in §6).** The voxel subsets are *not* truly
random (that would be JAX-unfriendly). Instead: pick a random 2D subset of (row, col) positions,
then define a **voxel cylinder** = the full 1D vector over all slices (`z`) at each selected
(row, col). Each cylinder is projected as a unit. So the update granularity is a *scattered set
of complete z-columns*.

**Why we care / motivation.** The block-GS scheme works but carries significant overhead in
computation and code maintenance, especially as recons grow to ~2K³ sinograms/volumes on 2–4
GPUs (~32 GB per volume copy). JAX **recompiles for each distinct partition shape** (we have not
done the bookkeeping to keep shapes uniform), adding overhead. The motivating question was
whether a momentum-based method could reduce this — which turned into a diagnosis of *why*
convergence is slow in the first place.

---

## 2. Notation

| Symbol | Meaning |
|---|---|
| `x`, `x_j` | reconstruction volume; iterate at iteration `j` |
| `x_∞` | a deeply-converged reference reconstruction (many iterations) |
| `η_j = x_j − x_∞` | error of iterate `j` |
| `Δ_j = x_{j+1} − x_j` | increment (consecutive difference) at iteration `j` |
| `A`, `A^T` | cone-beam forward projector, back projector |
| `W` | diagonal sinogram weighting |
| `R`, `R''` | qGGMRF prior and its (local) curvature |
| `β` | regularization strength. "Low β" = weak prior = the harder, more interesting case |
| `D` | diagonal preconditioner (≈ inverse SQS majorizer diagonal) |
| `H = A^T W A + β R''` | full Hessian (R'' local for qGGMRF) |
| `μ` | per-iteration error contraction factor, `μ = ||η_{j+1}|| / ||η_j||` |
| `κ` | effective condition number of the preconditioned system, `≈ λ_max/λ_min` |
| `g(z)` | per-slice gain profile: center-peaked **sinc**-like function of axial position `z` |
| `N_z` | number of slices (axial dimension) |
| **cylinder** | the update unit: full z-column of voxels at one (row, col) |
| **node slice** | a well-constrained slice (trough of the increment-norm profile) |
| **resonant slice** | a poorly-constrained slice (peak of the increment-norm profile): axial ends + g(z) sidelobes |
| **slice 19** | the central slice (volume has 40 slices, 0–39); the peak of g(z); the worst-behaved slice |

**Diagnostic measured throughout:** per-slice, we compute consecutive increments
`Δ_j = x_{j+1} − x_j`, then plot (a) the per-slice **L2 norm** of `Δ_j` and (b) the per-slice
**cosine** of the angle between `Δ_{j+k}` and `Δ_j` for lags k = 1,2,3,… The cosine reveals
the *geometry* of successive steps (parallel / orthogonal / anti-parallel); the norm reveals
*convergence rate* by slice.

---

## 3. The question that started this

> "Our block-GS subset scheme works but has overhead in computation and code maintenance,
> especially for large multi-GPU recons. Should we switch to Nesterov / momentum?"

Answering this required understanding *what mechanism* the subsets provide and *why* plain
gradient descent is slow — i.e. a convergence diagnosis. The bulk of these notes is that
diagnosis. The original frequency-domain intuition (from prior project theory) was:
**full gradient steps reduce low spatial frequencies well; randomly-selected subsets better
target higher frequencies.** That intuition turns out to be a conditioning statement and
survives the investigation in modified form.

---

## 4. Observed symptoms

Two persistent, annoying behaviors at **low β**:

- **Symptom A — the central slices misbehave.** A few slices "nearly parallel to the shortest
  ray from source to detector" (the central slices, peaked at **slice 19**) plus some similarly
  "resonant" ones converge poorly. **This does not happen for parallel-beam geometry** — it is
  cone-beam-specific. Under the subset solver, slice 19 actually takes a **bad early step**: its
  error *increases* for the first ~2 iterations before recovering, and it never catches up to
  the well-constrained slices within the iteration budget.

- **Symptom B — axial correlation decays slowly.** Correlation *within cylinders* (along z)
  decays more slowly than correlation *within slices* (in-plane). Hypothesized cause: early
  coarse subsets overshoot to compensate for the many un-updated voxels.

Quantitatively (low-β case, per-slice diagnostics over ~18 iterations from the scaled-FBP start):

- The **first step does almost all the work** (increment ~1.0); subsequent increments are
  ~0.01–0.02 in the well-constrained bulk and ~0.1 at the axial ends.
- Increment-norm profile vs. slice has a clear **g(z) shape**: peaks at axial ends (slices 0,
  39) and at interior **sinc sidelobes** (≈ slices 6, 13, 19, 25, 31), troughs at well-
  constrained nodes (≈ 4, 10, 16, 22, 28, 35). The slices that actually oscillate coincide
  with the sidelobe peaks — a clear, confirmed correlation.

---

## 5. The investigation: hypotheses raised and discarded

This section is preserved deliberately: several plausible mechanisms were ruled out, and the
*way* each was ruled out is the useful part. (It also includes one measurement bug that
inverted a conclusion — flagged below.)

**H1 — OS limit cycle. DISCARDED.** Initial thought: ordered-subset methods orbit the minimizer
because subset-gradient approximations disagree. **Wrong for our case:** that pathology belongs
to *data-subset* OS (scaled subset gradient as an approximation to the full gradient). We do
*image-domain block coordinate descent*: each sub-update is a majorize–minimize step on the
*full* cost over a voxel block, so (with a true majorizer) every sub-update decreases the full
objective — monotone, no limit cycle. *Clue that killed it:* the cost provably decreases each
sub-update.

**H2 — Overshoot from overrelaxation. DISCARDED.** Next thought: the dominant preconditioned
eigenmode crosses αλ > 1 and sign-flips, aggravated by the fixed overrelaxation factor ω > 1.
*Clue that killed it:* setting **ω = 1 does not change the behavior.** With a true majorizer and
ω = 1, block MM is monotone and cannot overshoot — so the phenomenon is not overshoot in this
sense, and the overrelaxation hypothesis fails too.

**H3 — Complex-eigenvalue rotation. DISCARDED.** The lag-1 increment cosine sat near **−0.5**
(≈120°), which looks like a rotation — a complex-eigenvalue pair in a non-symmetric iteration
operator (the block-GS sweep is non-normal even though H is symmetric PD). A real prediction
followed: a 120° rotation gives lag-2 cosine = cos(240°) = −0.5. *Clue that killed it:* the
measured **lag-2 cosine was ≈ 0**, not −0.5. Coherent rotation ruled out.

**H4 — Stalled in a flat near-null space. DISCARDED (measurement bug).** A plot showed
increment norms **constant** across iterations 1–8 (no decay), which — with lag-1 cosine −0.5
and lag-≥2 cosine 0 — implied successive errors were mutually orthogonal and *non-decaying*:
i.e. the iteration *diffusing* in a flat, degenerate subspace that no optimizer can fix. This
would have made it a regularization problem, not an optimizer problem. **This was an artifact of
a bug in the diff computation.** After the fix: increment norms **decay** monotonically and
ordered by iteration, and the lag-1 cosine is **spatially structured and mostly positive**
(→ +1 at node slices). *Lesson: the cross-check diagnostic earned its keep — a tidy hypothesis
was overturned by a corrected measurement.*

**What survived (corrected data).** At **node slices**, consecutive increments are nearly
**parallel** (cosine → +1, sharpening with iteration; lag-3 ≈ lag-1, so the iterate creeps along
*one stable direction* for many iterations) — the textbook signature of **slow monotone descent
down a single low-curvature eigendirection**. At **resonant slices**, successive steps
**decorrelate** (cosine → 0). The early slight-negative cosine at central sidelobes is a small,
transient, localized effect — the residue of the original "oscillation" report, much smaller
than first framed.

**The decisive number.** Full diagonally-preconditioned GD (ω = 1, subsets off) error-decay
`log10||η_j||` on a node slice contracts at **μ ≈ 0.987–0.99**, i.e. λ_min ≈ 0.01–0.013 vs.
λ_max ≈ 1 ⇒ **κ ≳ 100**, and the rate **slows** over iterations (sub-geometric) ⇒ a
**spectrum of small eigenvalues with a tail**, not one isolated mode. GD reduces error only
~18% in 15 iterations here — effectively useless — which is exactly the headroom a momentum/
Krylov method (≈ 1 − 1/√κ per iteration) would exploit.

**The subset solver, by contrast** (same low-β case, scaled-FBP start, per-slice log10||η_j||):
node slices show a sharp knee at iterations ~2–4 (e.g. −1.13 → −1.40 in three steps) then a
smooth tail — roughly an **order of magnitude** of acceleration over full GD. **But slice 19
gets *worse* for two iterations** (−1.09 → −1.07 → −0.97) before recovering, and ends below
where the good slices started. So subsets are the *engine*, not overhead — and the central slice
is where they visibly hurt.

> The contrast between full-GD (slow everywhere, but slice-19 merely slow, never worse) and
> subsets (fast bulk, but slice-19 takes a bad step) is the clue that localizes the cause to the
> **block structure**, not the underlying operator.

---

## 6. Synthesis: what is actually going on

### 6.1 The convergence-limiting subspace is axial and low-z-frequency

κ ≳ 100 with a sub-geometric tail; amplitude tracks g(z); **slice 19's error transiently rises
while global energy falls** = energy redistributing between axially-coupled slices. A per-voxel
**diagonal** preconditioner is structurally blind to cross-slice coupling, so it cannot
precondition these modes. That is the root cause of slow convergence.

### 6.2 The subset geometry handles the prior exactly — so the omitted coupling is the data term

With axis-only nearest-neighbor qGGMRF and scattered-cylinder subsets:

- **In-plane (x,y):** selected cylinders are scattered ⇒ no two updated voxels are
  prior-neighbors ⇒ prior coupling between updated voxels is **zero in-plane**. The block
  surrogate is *exact* for the prior in-plane.
- **Axial (z):** each cylinder is the complete z-column; the prior's z-edges connect voxels
  **both in the block** ⇒ prior axial coupling is **fully captured**.

So the per-block surrogate handles the **entire prior exactly**. By elimination, the only
inter-block coupling it omits is the **data term** — cone-beam rays linking different cylinders
through the shared sinogram. Therefore:

> **Slice-19's bad step is purely data cross-cylinder coupling**, concentrated at high g(z)
> where curvature (and thus the aggressive surrogate step) is largest. A *prior*-based axial
> preconditioner would be a **red herring** — the prior axial coupling is already perfect.

### 6.3 Symptoms A and B are two faces of one operator

- **B is the source term.** Coarse subsets update few voxels; the surrogate-optimal step
  inflates to compensate (its denominator counts only in-block curvature). Because each cylinder
  updates as a unit, that inflation is **axially coherent** — the whole z-column is pushed too
  far together, injecting low-z-frequency error.
- **A's slow axial spectrum is the decay rate.** The coherent axial error B injects is exactly
  the low-z-frequency content that relaxes slowly (κ ≳ 100). **Slice 19 is the g(z) hotspot**
  where overshoot amplitude and coupling-omission are both worst.

> **B re-injects coherent axial error; the axial conditioning makes it slow to clear; slice 19
> is the g(z) hotspot of the same operator.** *(To confirm against physical intuition across β
> and geometry before treating as fully settled — see §10.4.)*

### 6.4 Why momentum/NLCG would work but is not the chosen path

A full-gradient method (e.g. preconditioned PR⁺ NLCG) **cannot produce slice-19's bad step** —
it never separates the blocks, so it respects all cross-slice coupling (like full GD, which is
slow but never makes slice 19 *worse*). It would accelerate the axial slow subspace by ≈ √κ
(≈ 10× rate improvement on κ ≈ 100). The line search is **free in projector terms** (the data
term is exactly quadratic; cache `A·d` and update the residual incrementally), memory cost is
only **+2 volumes** (vs. L-BFGS's 2m history volumes — infeasible at 2K³, ~32 GB/volume), and it
**compiles once**. *However:* on the bulk it likely only *ties* the subset knee (√κ ≈ subset
acceleration), winning mainly on the central slices and the structural ledger. Given the
preference to reuse existing code, we chose to **repair block-GS** instead (next section), with
NLCG parked as the "replace" alternative (§11).

---

## 7. Decision: repair block-GS with an axial preconditioner

Keep the subset/block-GS solver; **upgrade the per-cylinder surrogate** from a per-voxel diagonal
to an **axial preconditioner** that captures the z-coupling the diagonal misses. This is a
localized surrogate change — the sweep, partition schedule, and JAX shapes are unchanged.

Two designs follow: **A** (per-cylinder banded, accurate, more cost) and **B** (slice-at-a-time
tridiagonal, cheap, approximate). Recommendation: **try B first, fall back to A.**

A key enabling measurement is already in hand: the `A^T A` PSF for a typical voxel extends far
**in-plane** but only **3–5 slices axially**. So the axial operator is **narrow-banded
(tri/pentadiagonal)**, never dense — no `N_z × N_z` matrix is required.

---

## 8. Design A — voxel-cylinder banded preconditioner

**Form.** Per cylinder, replace the diagonal data majorizer with a **banded axial majorizer**:
the cylinder's own `A_cyl^T W A_cyl` restricted to its axial band (half-bandwidth b ≈ 1–2 from
the PSF ⇒ **tri- to pentadiagonal**).

- **Storage:** ~(2b+1)·N_z per cylinder (≈ 3·N_z or 5·N_z), vs. N_z² dense.
- **Apply:** banded Cholesky / Thomas back-substitution, O(N_z·b). Factor once per cylinder;
  reusable across iterations (q0 = 2 ⇒ finite ρ''(0) ⇒ a fixed majorizer exists).
- **Block step:** "solve the banded system" instead of "divide by the diagonal."

**Why it works.** The diagonal surrogate **underestimates axial stiffness** (omits z off-
diagonals) ⇒ oversteps coherently along z (Symptom B). The banded majorizer's curvature in that
direction is correctly larger ⇒ surrogate-optimal step correctly smaller ⇒ overshoot shrinks.
If the banded block is a true majorizer (data axial band + prior axial tridiagonal), each block
step **still monotonically decreases the full cost** — the descent guarantee is preserved.

**Captures / misses.**
- *Captures:* within-cylinder axial self-coupling = essentially all axial coupling (PSF ≤ 1–2
  slices). Directly addresses **Symptom B**.
- *Misses:* **cross-cylinder** data coupling (the residual part of Symptom A). Prediction:
  substantially fixes B, *partially* fixes A.

**Integration / open code question.** Is prior curvature currently assembled *separately* from
data curvature in the per-cylinder step, or already combined into one diagonal?
- Combined ⇒ replace that diagonal with one banded (data + prior axial) block.
- Separate ⇒ add a data-only axial band, watching for **double-counting** at the prior's axial
  edges (the prior tridiagonal is already handled).

---

## 9. Design B — slice-at-a-time tridiagonal preconditioner

**Idea.** Instead of one banded N_z operator *per cylinder* (K solves per subset), use a single
**tridiagonal-across-slices** operator, **~3·N_z entries total**, applied to all in-plane voxels.
It couples slice z ↔ z±1 with one scalar per slice-pair.

**Crux — the separability assumption.** This is correct only if the axial coupling is
approximately **separable**:

```
coupling(z, z', in-plane position) ≈ c(z, z') · (in-plane diagonal weight)
```

i.e. the axial coupling *shape* is the same everywhere in-plane and only its *magnitude* varies
with position (via g(z)).

**Feasibility.**
- *Cost:* O(N_z) storage, one Thomas sweep total per subset. For 2K³ on 2–4 GPUs this is a
  negligible add-on vs. K banded solves — very attractive.
- *Where it likely helps:* **Symptom B is itself coherent in z and roughly uniform in-plane** —
  exactly what a single slice-level tridiagonal represents well. So the cheap operator is
  **matched to the dominant symptom**. This is the encouraging case.
- *Build in from the start:* carry the g(z) profile as a per-slice scaling,
  `c(z,z')·√(g(z)g(z'))`, not a constant — a single scalar coupling cannot represent both strong
  central and weak end-slice coupling. Still 3·N_z.
- *Where it breaks down:* it **cannot localize in-plane**, so it addresses **B (uniform axial)
  well but A (the slice-19 spatial hotspot) essentially not at all**. And if the axial PSF
  *shape* (not just magnitude) varies in-plane (e.g. cone angle widens axial spread off-center),
  separability fails and Design A is needed.

**Recommendation — tiered, not either/or.**
1. **Start with B** (slice-level tridiagonal + g(z) scaling): cheap, targets the dominant
   symptom (B), small reversible surrogate change. If it removes most of the slow axial decay,
   the job is largely done at negligible cost.
2. **Fall back to A** only if (a) separability fails (axial PSF shape varies in-plane), or
   (b) the residual slice-19 hotspot needs in-plane-localized damping that B structurally
   cannot provide.

The cheap operator's weakness (no in-plane localization) is aligned with the *secondary* problem
(A), while its strength matches the *dominant* one (B) — which is why B is the right first
experiment.

---

## 10. Measurements to take before building

*(None require new solver code.)*

1. **Axial band width — tri vs. penta.** From the existing `A^T A` PSF: is meaningful axial
   coupling 3-slice or 5-slice? (PSF says 3–5 total; confirm which. Design B is tridiagonal by
   construction.)
2. **Cross-cylinder coupling magnitude.** Apply `A^T W A` to an impulse in one *whole cylinder*
   (the update unit) and measure how much lands on the nearest *other* cylinder at typical
   sparse-subset spacing, relative to self-coupling. Small (<~10%) ⇒ within-cylinder blocks
   suffice. (Expected small given sparse subsets — confirm anyway.)
3. **Separability check (decides B vs. A).** Does the *normalized* axial PSF shape change with
   in-plane position, or only its magnitude? Same-shape-everywhere ⇒ B justified; shape varies
   ⇒ need A.
4. **Confirm the §6.3 synthesis** against physical intuition across β and geometry: are B and A
   genuinely one operator's two faces, or is there reason to expect independent causes?

---

## 11. Deferred / parked items

- **Frozen-Hessian rerun** (replace qGGMRF R'' with a fixed surrogate at the warm start, rerun):
  off the critical path — the slice-19 mechanism (data block-coupling) was identified from the
  subset-vs-GD contrast without it. Revisit only if a method shows decorrelation the linear
  picture can't explain.
- **Full-gradient PR⁺ NLCG** (the "replace" alternative): diagonal or data-axial-band
  preconditioned; **free line search** via cached `A·d` residual update; **fp64 scalar
  accumulation**; **periodic residual recompute** `r = Ax − y` to control fp32 drift; **+2
  volumes** memory. Removes slice-19 by construction, ≈ √κ acceleration. Parked in favor of the
  repair path; revisit if repairing block-GS disappoints.
- **fp32 hygiene (any method, relevant at 2K³):** accumulate inner products / line-search
  scalars in fp64; periodically recompute the residual from scratch.
- **Preconditioner-strength null result (context):** globally scaling the data-Hessian diagonal
  0.5×–2.0× barely changes behavior. This is *expected* — under an exact/relaxed line search a
  global magnitude change is a no-op (numerator ∝ c, denominator ∝ c², step ∝ 1/c cancels). Only
  the **relative, slice-dependent** shaping of D matters, which is what the axial preconditioner
  provides. Do not re-run global-scaling experiments expecting signal.

---

## 12. One-paragraph status

Convergence at low β is limited by a slow, **axial, low-z-frequency subspace** (κ ≳ 100,
sub-geometric decay) that a per-voxel diagonal preconditioner cannot touch. Because the
scattered-cylinder subsets handle the **axis-only prior exactly** (zero in-plane prior coupling
between updated voxels; full axial coupling within each cylinder), the only coupling the
per-block surrogate omits is the **data term's axial structure** — which is why the central
slice (slice 19, the g(z) peak) takes a bad step under subsets but not under full GD. Two
symptoms — the slice-19 hotspot (A) and slow within-cylinder decay (B) — are two faces of the
same g(z)-weighted, axially-coupled data operator. Plan: keep block-GS and add an **axial (data)
preconditioner** to the per-cylinder surrogate. The `A^T A` PSF is only 3–5 slices axially, so
the operator is narrow-banded, never dense. **Try the cheap slice-level tridiagonal first (with
g(z) scaling); fall back to per-cylinder penta/tridiagonal if separability fails or if the
slice-19 hotspot needs in-plane localization.**
