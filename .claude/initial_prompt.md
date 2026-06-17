This is the mbirjax CT reconstruction project — multi-GPU/CPU sharding work in the
`mbirjax` worktree on branch `greg/parallel_sharding`.

Orient first by reading, in order:
1. `.claude/claude_prompt.md` — collaboration style + workflow (investigate and propose
   before editing; wait for approval on non-trivial changes; minimal, localized changes;
   suspect the ruler before the code; stay curious and challenge my assumptions).
2. `experiments/sharding/plans/sharding_status.md` — the TOP HANDOFF (current state + NEXT).
3. `experiments/sharding/plans/sharding_implementation_plan_v2.md` — forward plan.  Read §0
   and **§P6 (the "EXECUTION ORDER" block first, then the detail bullets)**; skim §Decisions
   and §Adjacent tasks for what is already resolved.  (`sharding_implementation_plan.md` =
   completed-work record + principles; the P3 "(g0, L) design note" in v2 is required reading
   before any projector work.)
4. Skim `.claude/lessons.md` — the jax/GPU/placement/measurement playbook ("Sharded VCD
   memory: reference cycles + buffer donation" and the no-bit-exact-for-computed-floats rule
   are the load-bearing entries).
5. `.claude/back_projection_overview.md` — required for P6 (projector internals).
Verify claims against current code; memory/docs may lag.

**Where we are (2026-06-12).**  **A PR to prerelease is OPEN for the team to test** — beta
functionality is complete for ParallelBeam.  P5 is done and GPU-validated end to end:
always-on placements, automatic multi-GPU (all devices, any view/slice counts via exactly
inert padding, validated at 1024×1023×1024 = NRMSE 2.6e-7 / 2.18× on 2 GPUs / memory halves),
`configure_devices` / `device_summary` / `prepare_sino_for_devices` / `output_sharded`.
Adjacent tasks done: settable view parameters (runtime-arg projectors, `set_view_parameters`,
vcls converted + its ParallelBeamModel crash fixed); `_auto_shard_cpu=True` default (CPU =
2-device parallel by default); per-user `~/.mbirjax/` home (jit cache, set-only-if-unset, +
log defaults).  Test suite overhauled: per-geometry tests under `tests/geometries/`,
measured/recorded gates, seeds before every recon, `pytest -n auto` ≈ 66 s
(`dev_scripts/run_tests.sh`).

**Next (details in the status NEXT + v2 §P6 EXECUTION ORDER):**
- **Team-reported issues from the PR take priority as they arrive.**
- **P6 step 1 — the projector-rework scoping proposal** (review with me BEFORE code): the
  banded (g0,L) interface per geometry, the anchor rule, per-geometry forward assembly
  (parallel = concat, z-based = ACCUMULATE), in-place accumulator + donation, delete
  `entries_per_cylinder_batch`, and the de-closuring/module-level-jit restructure (kills the
  per-model retrace cost — tracing, not XLA compile, dominates first-call latency; measured).
- Then cone port → translation/multiaxis → retirement cascade (main_device/sinogram_device,
  is_sharded guards, view_indices, RETIRE-AFTER sweep) → the multi-GPU user docs page
  ("sharding" sparingly — speak in terms of multiple GPUs adding capacity and speed).

Reminders:
- **Stage only / DRAFT commit messages; never run `git commit`** — I commit from PyCharm.
- Tests: `source /Users/gbuzzard/miniforge3/etc/profile.d/conda.sh && conda activate mbirjax`;
  full suite `pytest -n auto tests/` (~66 s; single-process ~165 s); sharding specifics via
  `MBIRJAX_NUM_CPU_DEVICES=4 pytest tests/sharding/`; per-geometry recon tests live in
  `tests/geometries/`.
- I run GPU/large recons on the cluster — **flag GPU items.**  Fresh `pip install -e .` on the
  cluster after edits (a stale build once impersonated a leak); pre-flight `nvidia-smi dmon`
  (thermal throttling); `results/`/`baselines/` are gitignored — record decisions in docs.
- **Correctness gates: exact equality is NEVER the gate for computed floats** — tight
  `allclose` (1e-5 single-shot, ~1e-4 iterated; measured GPU run-to-run noise ~8e-6 rel).
  Exact equality ONLY for data-movement identities and constructed-zero invariants
  (full statement in lessons.md).
- jax/GPU specifics (sharded-array reference cycles + buffer donation, `peak_bytes_in_use` as
  the memory ruler, the benign `CUDA_ERROR_NOT_PERMITTED` warning, time ∝ N⁴ / memory ∝ N³,
  tracing-vs-compile attribution): details in `.claude/lessons.md` — consult before
  re-deriving.
