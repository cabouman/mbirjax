# Partition-sequence investigation — plan

**Drafted 2026-07-04** (post_shard_plans §1, second bullet).  Goal: pick a better default
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

## Phases

- **P0** — cache-builder script (lives in `mbirjax_applications`, near the loaders; config
  at top, no CLI): one cached bundle per dataset on cluster scratch.
- **P1** — references + trajectory harness (chunk-vs-rerun decision measured here).
- **P2** — full candidate sweep on Lilly + Z62 (GPU; subsampled recons expected ~1–3 min
  each → the sweep is hours, run in pieces).
- **P3** — validate the shortlist on the remaining datasets + ONE full-scale (unsubsampled)
  spot check of the winner vs default (subsampling changes conditioning; the winner must
  survive at scale).
- **P4** — decision: new default sequence and/or the size-adaptive starting-granularity rule;
  update docs + the recon guidance.

## Open questions (for discussion before P0)

1. **NRMSE region**: full recon vs RoR-masked vs flash-removed?  Objects extending past the
   FoV converge slowly at the flash (post_shard_plans §2) — masking the exterior keeps the
   metric about the object.  Proposal: RoR mask + drop the top/bottom few slices.
2. **Init policy**: `recon()` default init for all runs (simplest, matches production), or
   direct-recon init (Zeiss script style)?  Proposal: default init; direct-init as a P3
   sensitivity check on one dataset.
3. **Reference tightness**: 0.01% change may still leave the reference measurably moving at
   granularity-128 tails; if the reference wobbles at the level that separates candidates,
   tighten (0.005 / more iterations) on the affected dataset.
4. **Time vs iterations**: iterations are seed-comparable but hide per-iteration cost
   differences (coarse iterations cost more per pixel-pass?  — actually each iteration
   processes all pixels once regardless of granularity, but subset count changes dispatch
   overhead and line-search sync counts).  Report both; decide on time.
