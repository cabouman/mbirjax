We're returning to the multi-GPU/CPU sharding work on `mbirjax` (working branch `greg/sharding_extensions`; the cone-beam sharding is complete and in a PR to prerelease). The next step is the **library sharding port, increment D — the `TranslationModel` port**, starting with the **FDK filter (T1)**.

First read, in order:
1. `.claude/claude_prompt.md` — collaboration style + workflow.
2. `experiments/sharding/plans/sharding_implementation_plan_v3.md` — the primary current-state + forward plan (read first); §5 (D = next; translation + multiaxis).
3. `experiments/sharding/plans/increment_d_translation_design.md` — the authoritative translation-port staged plan (T1–T5); read the key finding (§0) and the staged table (§3).
4. `experiments/sharding/plans/p6_increment_b_design.md` — the cone port; reference for the techniques translation reuses (banded kernel, anchor rule, the FDK-filter cleanup, the GPU n=1 short-circuit).
5. `experiments/sharding/plans/sharding_status.md` — TOP HANDOFF for the latest session state.
6. `.claude/back_projection_overview.md` and `.claude/lessons.md` — projector internals + the jax/GPU playbook (skim; consult when a problem rhymes with a past one).

D-translation scope (per `increment_d_translation_design.md`): `TranslationModel` is `ConeBeamModel` *pre-port* (copied FDK filter, `entries_per_cylinder_batch`, same fan architecture), so the cone increments map ~1:1 and the sharding infra is already geometry-agnostic. Stages: T1 convert `fdk_filter` to the shared `_apply_direct_recon_filter` (no sharding flip — clean first move); T2 add the banded back kernel (anchor rule + global validity clip) and rewire the single-device kernel to a rolled `lax.map` over slice bands, deleting `entries_per_cylinder_batch` (used in BOTH back and forward); T3 the forward anchor fix (`k_global = g0 + arange(L)`); T4 flip `_supports_sharding()=True` (the base hooks then drive it); T5 inert slice padding (mostly free). Decisions (2026-06-18): **mirror cone, defer the kernel consolidation** (keeps row-sharding per-geometry open); **keep both back kernels + the GPU n=1 short-circuit** (need expected, but MEASURE the platform split). Multiaxis is the sibling sub-effort (its FBP fix is decided in v3 §6). Prereq for tracking the port: `translation` baselines in mbirjax_metrics (Greg + a separate session).

In general, verify all code claims against the current files (docs/memory may lag). Reminders: stage only / draft commit messages — I commit from PyCharm; flag GPU items (I run those on the cluster); exact equality is never the gate for computed floats (tight allclose).


Additional prompt after new code:
Note that new code is just a starting point for discussion and tuning.  
What do you think might be some good ways to investigate correctness, robustness, and performance for these changes?  
I'm not saying we need to add more tests, but we should at least think about what our existing tests do and do not cover, 
possible corner cases, and some ways to test performance.