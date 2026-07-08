# plans/ — internal design docs, plans, and findings

This directory collects the project's internal (developer-facing) documentation: architecture
decisions, program plans and status, and the findings/design documents produced by
experiments.  User-facing documentation lives in `docs/` (readthedocs); experiment CODE and
data stay in `experiments/` — the rule for findings docs is:

> **the documentation for `experiments/X` lives at `plans/experiments/X`.**

Docs here are records: they capture the reasoning and measurements behind decisions at the
time they were made, and are not rewritten as the code evolves (code comments carry the
current state; these carry the why and the numbers).

## Architecture and decision records

- `back_projection_overview.md` — data layouts and the structure of forward/back projection
  across geometries.
- `sinogram_sharding.md` — why sinograms shard by VIEW and recons by SLICE, with the parked
  detector-row-sharding alternative and its halo analysis.

## Program plans and status

- `sharding/` — the multi-device sharding program (COMPLETE, shipped 2026-07).
  Start with `sharding/post_shard_plans.md` (what came next after the program closed) and
  `sharding/sharding_status.md` (the end-state summary).  `sharding_implementation_plan_v3.md`
  is the final plan of record (v1/v2 are its history); the remaining files are per-workstream
  designs (MAR refactor, preprocessing pipeline, correctness gating, performance tracking,
  increment designs).  `sharding/_file_index.md` has one line per file.
- `partition_sequence_plan.md` — the VCD partition-sequence convergence study (ACTIVE as of
  2026-07; its experiment code will live in `experiments/partition_sequence/`).

## Findings from experiments (companion code stays in `experiments/`)

- `experiments/projector_kernels/fwd_back_findings.md` — THE record of the 2026-07
  projector-kernel campaign: forward/back attribution, the sorted channel reduction and its
  guard constants, the TilePolicy, per-geometry rollouts (including translation's measured
  collision-cliff non-adoption), the DRY fan helpers, and the concrete-scatter-centers
  rounding-bug fix with its verification chain.  Benches in
  `experiments/projector_kernels/`.
- `experiments/projector_batching/` — the earlier projector-batching characterization and
  the retired v2 batching refactor (a worked example of driver-level wins failing to compose
  end-to-end).  Probes in `experiments/projector_batching/`.
- `experiments/sharding/parallel_performance/` — parallelization option studies (fbp filter
  strategies, forward-vs-back discussion).  Scripts in
  `experiments/sharding/parallel_performance/`.
- `experiments/bugs_and_artifacts/jax rounding bug/` — the XLA round-in-jit miscompilation:
  `jax_rounding_bug.md` (the bug record; the bug still exists in JAX) and `phase_d_design.md`
  (the concrete-input fix design + as-built notes; the horizontal fans are fixed, the
  vertical fans' per-slice rounds are documented accepted risk).  Repros in
  `experiments/bugs_and_artifacts/jax rounding bug/`.
- `experiments/bugs_and_artifacts/center slice noise/` — the center-slice noise
  investigation and preconditioner notes.

## Related working documents (not in this directory)

- `.claude/claude_prompt.md` — collaboration style and code-change workflow.
- `.claude/lessons.md` — the engineering playbook (operative rules distilled from these
  programs; the narrative history behind each rule lives in the documents above).
