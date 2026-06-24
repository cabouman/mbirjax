We're continuing the multi-GPU/CPU sharding port of `mbirjax` (working branch `greg/sharding_extensions`, rebased onto prerelease and PR-ready). **ParallelBeam, cone, and translation are done**; the cone PR is merged into prerelease. **Next substantive code = the MULTIAXIS port** — the last increment-D sub-effort: flip `MultiAxisParallelModel._supports_sharding` (mirroring the translation port), and land the FBP angular-weighting fix with it. Then increment E (the retirement cascade).

Read for orientation (these are the source of truth — verify any code claim against the actual files; docs/memory may lag):
1. `.claude/claude_prompt.md` — collaboration style + workflow.
2. `experiments/sharding/plans/sharding_status.md` — **TOP handoff = current state + next step (read first).**
3. `experiments/sharding/plans/sharding_implementation_plan_v3.md` — primary forward plan; §5 (execution order), §6 (the multiaxis FBP angular-weighting fix, decided in principle).
4. `experiments/sharding/plans/increment_d_translation_design.md` — the completed translation port; the template multiaxis mirrors (banded kernels, anchor rule, the `_supports_sharding` flip, inert padding, the lean test pattern).
5. `.claude/lessons.md` + `.claude/back_projection_overview.md` — jax/GPU playbook + projector internals (skim; consult when a problem rhymes with a past one).

Working reminders: stage only / draft commit messages — Greg commits from PyCharm; flag GPU items — Greg runs those on the cluster; do git surgery via CLI, not PyCharm; exact equality is never the gate for computed floats — use the scale-invariant `conftest.assert_sharded_allclose` for sharded-vs-single comparisons; and the sharded VCD loop is geometry-independent, so a new geometry needs no per-geometry sharded VCD-recon test.
