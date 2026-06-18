We're returning to the multi-GPU/CPU sharding work on `mbirjax` (branch `greg/conebeam_sharding`), with the performance dashboard essentially complete. The next step is the **cone-beam port, increment B5 — exactly-inert slice padding**, which unblocks the 4 deferred cone tests that fail at non-dividing slice counts.

First read, in order:
1. `.claude/claude_prompt.md` — collaboration style + workflow.
2. `experiments/sharding/plans/sharding_implementation_plan_v3.md` — the primary current-state + forward plan (read first); §4 (cone state) and §5 (B5 = next).
3. `experiments/sharding/plans/p6_increment_b_design.md` — the authoritative cone-port staged plan + progress; see the B5 bullet for scope.
4. `experiments/sharding/plans/sharding_status.md` — TOP HANDOFF for the latest session state.
5. `.claude/back_projection_overview.md` and `.claude/lessons.md` — projector internals + the jax/GPU playbook (skim; consult when a problem rhymes with a past one).

B5 scope (per v3 §4 / `p6_increment_b_design.md`): crop the cone forward gather to the real slice count (padded slices are zero → exact); reconcile the device-form-vs-real shape in the geometry tests and the internal `sparse_back_project` contract; consider a `_supports_slice_padding()` hook so "B4 = dividing counts only" is explicit (cone False until B5). The padding masks and the B1 global validity clip are already geometry-agnostic. Reproduce the failures on CPU with `MBIRJAX_NUM_CPU_DEVICES=4`.

In general, verify all code claims against the current files (docs/memory may lag). Reminders: stage only / draft commit messages — I commit from PyCharm; flag GPU items (I run those on the cluster); exact equality is never the gate for computed floats (tight allclose).
