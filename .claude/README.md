# Orientation — beta sharding worktree

This is the **beta sharding** worktree: branch `greg/parallel_sharding`, built
fresh from `prerelease`. Multi-device sharding is being reimplemented here under
a new view-sharded design.

**Read these first (in order):**
1. `../experiments/sharding/plans/sharding_status.md` — current phase, blockers,
   verified hardware facts.
2. `../experiments/sharding/plans/sharding_implementation_plan.md` — the detailed
   Phase 0–F checklist, migration table, principles, open questions.
3. `claude_prompt.md` (this dir) — collaboration style and code-change workflow.

**Sibling worktree (prior art):** `…/Research/mbirjax/` on `greg/parallel_tests`
(tag `research-snapshot-2026-05-29`) holds the original slice-axis
implementation we are porting from. Its `.claude/status.md` and the auto-memory
`sharding_progress.md` have the full research-branch history. That branch will
eventually be deleted — migrate anything worth keeping first.

**Environment:** conda env `mbirjax`
(`source /Users/gbuzzard/miniforge3/etc/profile.d/conda.sh && conda activate mbirjax`).
