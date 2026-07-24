# Partition-sequence investigation — plan

**Drafted 2026-07-04** (current_plans.md §1, second bullet).  Companion experiment code:
`mbirjax_metrics/experiments/partition_sequence/` (the sibling repo).  Goal: pick a better default
`partition_sequence` and gate the size-only adaptive starting-granularity policy (skip
granularity 1 on large recons — it sets the per-device memory peak) on convergence-quality
data from REAL scans.  Theory to test (Greg): monotone NON-DECREASING granularity sequences
converge best — repeats are fine, dips (e.g. the `slow_dip` in
`mbirjax_applications/nsi/Lilly_recon_partition_sequence.py`) should lose.

## Mechanics that shape the design (verified in code)

- Sequence entries are INDICES into granularity `[1,2,4,8,16,32,64,128,256]`; default
  `[0,2,4,6,7]`.  `gen_partition_sequence` extends by REPEATING THE LAST ENTRY, so the tail
  granularity governs every post-ramp iteration — it is a first-class variable, not an
  afterthought.
- `recon()` already records per-iteration change (`stop_threshold_change_pct` list, NMAE %),
  granularity, fm_rmse, alpha.  Per-iteration NRMSE-vs-reference needs intermediate recons:
  either chunked restarts (`first_iteration` + `init_recon`, re-seeding `np.random` per chunk
  so partitions match) or repeated from-scratch runs at a few `max_iterations` budgets (same
  seed → identical trajectory).  Measure the chunk-restart overhead once, then pick.
- Partitions come from global `np.random` → `np.random.seed(0)` before every run; one seed
  for the whole study (the partition-noise floor is known ~1e-4-class on recon fingerprints).
- **Subset ORDER is also drawn from global `np.random`, per iteration**
  (`vcd_partition_iterator` shuffles subsets — tomography_model ~L2964).  Two consequences,
  measured 2026-07-04: (a) chunked restarts differ from a monolithic run at subset-order
  noise level (rel ~0.3–0.5 at 2–3 far-from-converged toy iterations), NOT fp noise — both
  are valid VCD paths; (b) with chunk=1 every candidate draws its iteration-k permutation
  from the identical RNG state, so **chunked is the matched-randomness instrument** —
  monolithic runs of different sequences desync their RNG streams and carry permutation
  noise into every comparison.  Production-path (monolithic) fidelity is checked at P3.
- The per-iteration change metric depends on the iteration's granularity → thresholds are
  NOT comparable across sequences; NRMSE vs a converged reference is the primary metric,
  change-% secondary.

## Data (preprocess ONCE per dataset, cache to disk, reuse)

Subsample **4× detector (rows+channels), 8× views**; no-metal recons; sizes then fit a
regular `recon()` (no split_sino).  Cache per dataset: sinogram (f32, host-clipped ≥0,
auto-cropped where applicable) + geometry/optional params + sharpness/snr — weights are
regenerated (`gen_weights(..., 'transmission_root')`, cheap/deterministic).  Candidate sets
(loaders exist in `mbirjax_applications`):

1. **NSI Lilly** (`nsi.compute_sino_and_params` + `auto_crop_sino_conebeam`), num_metal=0.
2. **Zeiss ORNL Z62** (cone; the set whose memory pressure motivated `coarse_4_128`).
3. **Zeiss SiC Composite** (cone, view-aligned) — different object statistics.
4. Optionally one parallel-beam 'ultra' set for geometry diversity.

All runs: same seed, same init policy, per-dataset sharpness/snr as in the existing scripts.

## Protocol

- **Reference per dataset**: default sequence, `stop_threshold_change_pct=0.01` (cap ~100
  iterations); save the recon — the MAP minimizer is sequence-independent, so one reference
  serves all candidates.  Record its full trajectory (the baseline curve).
- **Candidates** (~12; one run each per dataset, plus repeats only where curves are close):
  - the four named ones: `default`, `coarse_4_128`, `slow_start`, `slow_dip` (the
    non-monotone control);
  - ramp starts: `[0,1,...,7]`, `[2,4,6,7]`, `[4,6,7]`, `[6,7]`, `[7]` (flat-finest);
  - dwell: `[2,2,3,3,4,4,5,5,6,6,7,7]`;
  - tail variants: ramps ending at index 6 / 7 / 8 (granularity 64 / 128 / 256).
- **Per run record**: NRMSE-vs-reference at each iteration (or at budget checkpoints),
  wall time per iteration, change-% trajectory, peak GPU memory (the starting-granularity
  memory story), final NRMSE at matched iteration AND matched wall-time budgets.
- **Verdicts**: per dataset, NRMSE-vs-time curves; a winner must dominate (or tie) across
  datasets; explicitly test monotone-vs-dip; the tail-granularity question; and whether
  skipping granularity 1 (`[2,...]` starts) costs any final quality — the adaptive-policy
  gate.

## Results (P1-P2, 2026-07-05 — subsampled 4×/8×, 1 GPU, full trajectories in
## /scratch/gautschi/buzzard/ps_study/ps_results*/; scratch is purged, decisions recorded here)

**Calibration.**  References: lilly 60 iters, z62 184, sic 461 (slow converger, ~0.994×/iter).
Chunk checks fp-class (1.5e-06..1.2e-04; lilly's top value is its positivity clipping), restart
overhead ×1.00 — the checkpointed harness is production-exact and free.  **Noise floors**
(5 partition seeds, 15 iters, pairwise masked NRMSE): lilly 0.0050, z62 0.0123, sic 0.0045 —
z62 readable only at NRMSE ≥ ~0.02.

**Sweep verdict (16 sequences over two rounds; consistent on all three datasets):**

1. **Convergence per ITERATION is nearly sequence-independent.**  z62: all candidates reach
   NRMSE 0.05 in 68–74 iters; sic at fixed budget: NRMSE spread ≤ the noise floor.  Only
   lilly shows a modest real edge for fine-heavy starts (≈2 iterations at NRMSE 0.01).
   Monotone never loses; `slow_dip` never wins; LONG COARSE PHASES (`ramp_full`, `dwell`)
   consistently cost the most time.
2. **Tail granularity is the big TIME lever via per-iteration COST**: 64-tails ≈ 17–23%
   faster than 128-tails at equal quality (z62: 90 vs 118 s to NRMSE 0.05; sic: ~288 vs
   ~348 s to iteration 140); 256-tails ~40–80% slower.
3. **Skipping granularity 1 cuts peak memory 21–35% at no quality cost** (z62 5.03 → 3.49
   GiB; lilly 3.88 → 3.15) — this GATES THE SIZE-ADAPTIVE STARTING-GRANULARITY POLICY
   POSITIVELY (current_plans.md §1).

**Best sequences: `[4, 6]` (granularity 16 → 64) and `[6]` (flat 64)** — vs the default
`[0,2,4,6,7]`: ~20–30% faster to every readable quality target AND ~20–30% lower peak, on
every dataset.  (`[7]` flat-128 wins lilly's peak-memory column but pays 128-tail time.)

**Scale check (P3, z62 at 2×/4× → 1024³, 2026-07-05):** the two levers SWAP importance at
scale.  Quality still schedule-independent (default / `[4,6]` / `[6]` all end NRMSE ~0.064).
The tail-granularity TIME edge SHRINKS to ~6% (default 1630 s vs `[4,6]` 1524 / `[6]` 1558) —
at scale, compute-per-projection dominates the per-subset dispatch overhead that made 64-tails
17–23% faster when subsampled (Greg's prediction: 128 never beats 64, confirmed, just by a
smaller margin).  But **skipping granularity 1 becomes a 30–38% PEAK-MEMORY win** (default
37.1 GiB → `[4,6]` 26.0 → `[6]` 23.0; 11–14 GiB at 1024³) — the dominant, growing benefit.

**CORRECTION — the 4×/8× tail conclusion was a SUBSAMPLING ARTIFACT (2026-07-05, Greg's
4×/4× suggestion).**  At 4×/8× only 101 views → so underdetermined the recon raced to its
(noisy) reference regardless of granularity, MASKING the tail dependence.  Re-run at realistic
sampling and the tail story is the OPPOSITE and SIZE-DEPENDENT:

- **Convergence per iteration IS granularity-dependent** (VCD = coordinate descent; finer =
  more Gauss-Seidel-like updates = faster convergence per iteration).  Flat coarse tails
  ([3]/[4]/[5] = gran 8/16/32) barely converge; finer converges much faster.
- **512³ (4×/4×, all 3 datasets z62/lilly/sic): finer tail [7]=128 is OPTIMAL on BOTH time and
  memory.**  z62: [7] 74 it / 188 s to 0.05 & 3.78 GiB, vs [6] 103 it / 219 s & 4.40 GiB.
  Lilly/sic agree.  `[8]=256` SATURATES — no convergence gain past 128, +overhead (z62: 257 s
  vs 188), same floor memory.
- **1024³ (z62): the optimum MOVES BACK to [6]=64.**  The convergence benefit of finer shrinks
  at scale (103→99→98 it for 64→128→256) while per-iteration cost rises, so [6] is fastest to
  target (1353 s vs [7] 1397 vs [8] 1488); and [6]/[7]/[8] all TIE on memory (~23 GiB, the
  fixed-array floor).  So at scale the tail moves NO memory — only skipping granularity 1 does.

**FINAL recommendation (supersedes the two above):**
1. **Skip granularity 1** — the firm, size-independent memory win (default 37 → 23–26 GiB at
   1024³), no quality cost.  THE lever.
2. **Tail 64–128 with a size-dependent optimum** (~128 at 512³, ~64 at 1024³) — a MODEST
   second-order effect (~3 % at 1024³, larger at small sizes); both far better than the
   extremes.  256 is never worth it.
3. Default `[0,2,4,6,7]`'s 128 tail is fine (optimal at 512³, ~3 % slow at 1024³); its only
   real flaw is the granularity-1 START.  So the minimal fix is a sequence like `[4,6,7]` /
   `[2,4,6,7]` (skip gran-1, keep a fine tail).  A coarse START buys a few % early convergence
   for a few GiB (it lifts peak above the floor) — a real but small tradeoff.

Durable form: the size-adaptive starting-granularity policy (current_plans.md §1), now
positively gated; the tail could be size-adaptive too but the payoff is small.  Cross-validated
on 3 datasets at 512³ + z62 at 1024³, one GPU, cone.

## PROPOSED default changes — for team review (2026-07-05)

The study points to a small, well-supported set of changes.  These are PROPOSALS for the team to
weigh in on, not settled decisions, before anything lands in `recon()`.

**Sequence — two primary options, `[7]` and `[4, 7]`** (both a flat-128 tail; both drop the deep
coarse ramp of today's `[0,2,4,6,7]`).  Shared basis: with a fine (128) tail, extra coarse
granularities add ~0 convergence (flat `[7]` already matches / beats the fully-ramped default to
every target — lilly 4 vs 6 iters, sic tie, z62 1024³ 99 vs 100), and each distinct granularity
costs one more subset-updater compile.
  - **`[7]`** (flat 128): simplest, ONE compile, and it SUBSUMES the granularity-1 memory win —
    never coarse, so peak sits at the fixed-array floor at every size (current_plans.md §1's
    memory concern met by the sequence itself; no adaptive start needed for the default).
  - **`[4, 7]`** (gran 16 → 128): one coarse warm-up iteration as a hedge for possibly-harder
    cases (poor init, object classes we haven't tested), at a modest cost — the gran-16 step
    lifts peak off the floor (its subset arrays are 8× a gran-128 step's) and adds a second
    compile.  Our data shows the warm-up buys ~nothing HERE, but it's cheap insurance the team
    may prefer.
  The choice is simplicity/lowest-memory (`[7]`) vs coarse-start insurance (`[4,7]`); both are
  clearly better than the current default.  An adaptive coarse start (start granularity from voxel
  count) remains an option if size-adaptivity is wanted later.

> **ADDENDUM (2026-07-18) — the flat-tail options above were REFUTED before shipping; decision
> record.**  The slice-parity study's broader evidence (`plans/slice_parity/slice_parity_plan.md`,
> Summary finding 3: three real datasets × two sharpness settings against 150-iteration
> references) found **flat-fine sequences without a coarse start unsafe as defaults** at
> interactive iteration budgets: `[7]` (flat 128) is mildly worse on Lilly and CATASTROPHIC on
> z62 — essentially still at the FDK start after 15 iterations at high sharpness.  Large
> low-frequency initial error cannot be corrected by fine scattered subsets; the coarse start is
> load-bearing on data whose initial error is low-frequency-dominated, even though this study's
> cells showed it "buys ~nothing."  The shipped default is the monotone ramp **`[2, 4, 6, 7]`**
> (commit `42c0e23`): coarse start kept, granularity-1 memory spike dropped.  Do not revisit a
> flat-fine default without re-running the z62 high-sharpness cell.

**`max_iterations`: raise from 15 into roughly the 25–50 range** (exact value TBD by the team);
`stop_threshold_change_pct` unchanged at 0.2.  The real issue is the 15-cap STRANGLING the
threshold: 0.2% binds at iter ~44–49 on hard objects, so the cap stops them first, far short
(change 1–2.4 %).  Raising the cap lets the threshold govern → consistent quality across objects
(easy ones still stop ~15).  0.2% is visually converged (convergence PNGs); ~25 is a lighter
default that clears most objects, ~50 lets even hard objects reach the threshold — the team can
pick where on that range to sit.  Tighten the threshold to 0.1% only for quantitative work.

## Metric caveat — the study NRMSE is FLASH-inflated, object-dependently

Radial-crop analysis on the flat-`[7]` snapshots (`experiments/.../figures/`, script
`mbirjax_metrics/.../radial_crop_nrmse.py`): the FoV-edge "flash" (current_plans.md §2 — objects
extending past the field of view) inflates the NRMSE, but HOW MUCH depends on the object:

- **Simple/solid (z62 cylinder): strongly flash-dominated.**  A 5 % radial crop drops the
  0.2 %-stop NRMSE **5× (0.100 → 0.020)** and further crops barely move it — the flash is a thin
  outer ring holding ~80 % of the reported error.  The INTERIOR is at ~2 % by iter 44; the
  full-RoR 0.10 badly overstates object error.
- **Structurally-complex (sic composite tube): mostly REAL.**  A 20 % crop takes 0.145 → 0.125
  only — the error is distributed object structure (wall texture, pores) that genuinely needs
  the iterations; cropping doesn't shortcut it.

So the **change-% stop metric is left UNMASKED on purpose** (Greg): a signal-threshold mask is
finicky (recon-sized, per-iteration re-estimation, histogram fragility, flash ≈ object intensity,
per-subset plumbing in `vcd_subset_updater`) and a fixed radial crop is object-shape-dependent
(would cut sic's boundary-hugging wall).  The flash-inclusive change-% is instead CONSERVATIVE
for the interior (the flash keeps it elevated until everything, interior included, has settled),
which is exactly the right bias for a default stop.  The flash is best fixed at the SOURCE
(the §2 sinogram edge-taper / sine filter), which would also make simple-object NRMSE track
quality — this study is good motivation to prioritize it.  Read absolute NRMSE-vs-iteration as an
UPPER bound on interior error (loose for simple objects, tight for complex); RELATIVE schedule
comparisons are unaffected (the flash is schedule-independent).

## Phases

- **P0** — cache-builder script: one cached bundle per dataset.  (As of 2026-07-06 the whole
  pipeline lives in `mbirjax_metrics/plans/experiments/partition_sequence/`, driven by one
  `config.yaml`; caches are on the shared depot dir
  `/depot/bouman/data/mbirjax_metrics/partition_sequence/cache/`.  See that folder's README.)
- **P1** — references + trajectory harness (chunk-vs-rerun decision measured here).
- **P2** — full candidate sweep on Lilly + Z62 (GPU; subsampled recons expected ~1–3 min
  each → the sweep is hours, run in pieces).
- **P3** — validate the shortlist on the remaining datasets + ONE full-scale (unsubsampled)
  spot check of the winner vs default (subsampling changes conditioning; the winner must
  survive at scale).
- **P4** — decision: new default sequence and/or the size-adaptive starting-granularity rule;
  update docs + the recon guidance.

## Decisions (2026-07-04, Greg)

1. **NRMSE region**: RoR mask + drop some top/bottom slices.
2. **Init**: `recon()` default init everywhere — which is the direct recon anyway, scaled to
   minimize the RMSE of the error sinogram.
3. **Reference tightness**: expect the PARTITION-CHOICE variability (different partition
   seeds at fixed 15 iterations, or at a fixed 0.2% threshold) to dominate the 0.01%
   reference wobble — VERIFY via a noise-floor calibration in P1: reference config × ~5
   seeds → pairwise masked NRMSE spread = the floor any candidate separation must exceed.
4. **Verdict axes**: report iterations AND time, but decide on **time and memory at fixed
   image quality**.
