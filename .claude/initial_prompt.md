We're mid-investigation on the mbirjax performance-tracking toolchain (branch `greg/conebeam_sharding`). 
The performance-tracking code is now in a separate repo, `mbirjax_metrics` which lives locally in a 
directory parallel to `mbirjax`.
First read, in order:

`experiments/sharding/plans/sharding_status.md` — the TOP HANDOFF (2026-06-16b) describes the now-built 
toolchain in `experiments/sharding/scaling_tests/` (engine `performance_tracking.py` + fingerprint + records + 
diff/gate + `capture_golden.py` + `capture_main_baseline.py`; usage in that dir's README.md; design in 
`plans/performance_tracking_plan.md`).  This code is duplicated in `mbirjax_metrics`, and the code in 
`mbirjax` is slated to be archived.  

Also, "HANDOFF (2026-06-16)" describes recent work to make 1-device sharded cone beam competitive with the
legacy unsharded code. The work there was motivated in part by the performance tracking tool, which is why
were focusing on that before continue further work on sharding. After the performance tracking is solid,
we'll return to the conebeam refactoring as described below.  

Immediate next step: Design and implement some form of visual interface to the data in `mbirjax_metrics`
with the goal of being able to compare branches, track performance over time, etc.  I'm open to suggestions
about how to approach that design.  

In general, verify all code claims against the current files (the memory/docs may lag). Finally, read the collaboration 
style + workflow: `.claude/claude_prompt.md` (propose before editing, minimal localized changes, 
suspect the ruler before the code, maintain curiosity and collaborative partnership).


-----
Previous version: 

This is the mbirjax CT reconstruction project — multi-GPU/CPU sharding work in the
`mbirjax` worktree on branch `greg/conebeam_sharding`.

Orient first by reading, in order:
1. `.claude/claude_prompt.md` — collaboration style + workflow (investigate and propose
   before editing; wait for approval on non-trivial changes; minimal, localized changes;
   suspect the ruler before the code; stay curious and challenge my assumptions).
2. `experiments/sharding/plans/sharding_status.md` — the TOP HANDOFF (current state + NEXT).
3. `experiments/sharding/plans/sharding_implementation_plan_v2.md` — forward plan.  Read §0
   and **§P6 (the "EXECUTION ORDER" block first, then the detail bullets)**; skim §Decisions
   and §Adjacent tasks for what is already resolved.  (`sharding_implementation_plan.md` =
   completed-work record + principles; the P3 "(g0, L) design note" in v2 is required reading
   before any projector work.)  **P6 cone-port specifics (active): `plans/p6_increment_b_design.md`
   (the staged plan + progress) and `plans/p6_projector_rework_proposal.md` (read its top STATUS
   block — §8a-design is the canonical, measurement-driven design; the 2026-06-12 body is partly
   superseded).  Sinogram row-sharding exploration: `.claude/sinogram_sharding.md`.**
4. Skim `.claude/lessons.md` — the jax/GPU/placement/measurement playbook ("Sharded VCD
   memory: reference cycles + buffer donation" and the no-bit-exact-for-computed-floats rule
   are the load-bearing entries).
5. `.claude/back_projection_overview.md` — required for P6 (projector internals).
Verify claims against current code; memory/docs may lag.

**Where we are (2026-06-14, P6 cone port — CONE SHARDS; filter unified).**  P5 done + GPU-validated.
P6: cone shards (increments A/B1–B4, committed — recon by slice ⇄ sinogram by view; banded
reduce-scatter BACK + gather+monolithic FORWARD = decision C; CPU-validated end to end at DIVIDING
counts via `tests/sharding/test_cone_sharded.py`).  **This session (full detail in the status TOP
handoff 2026-06-14b — not duplicated here):** unified the FBP/FDK filter into one shared
bounded/jitted/per-view-sharded codebase (cone's old unbounded FDK filter gone; `fdk_recon` now
on-device like `fbp_recon`); a cone **n=1 forward single-call fix** (recovers XLA rematerialization,
~32→16 GB at 1024³ n=1); a **`_gather_to_host` single-shard short-circuit** (single-GPU gather-at-exit
no longer pays a host round trip); `jax.devices` consolidated behind `_device_setup`; and
`cone_baseline_scaling.py` ruler-hardened.  **KEY lesson: most "regressions" we chased were RULER
bugs** (device count not pinned; host input timed; the 4b21a3c2 gather-at-exit timed) — not the cone
port.  All CPU-validated (full suite 165p @2dev, sharding 107p @4dev); staged for Greg's PyCharm
commit.  Background: `p6_increment_b_design.md`.

**KNOWN ISSUE — deferred to B5 (Greg's call; do NOT chase before B5).**  Flipping cone's
`_supports_sharding()` enabled the geometry-agnostic P5 PADDING, but cone padding IS B5 (not done).
So at non-dividing slice counts (≥4 devices) cone auto-pads and **4 tests FAIL on multi-device**:
`test_{adjoint,hessian}_anisotropic_cone` (test_projectors), `test_split_sino`,
`test_vcd_anisotropic_cone` (test_vcd).  Reproduce on CPU with `MBIRJAX_NUM_CPU_DEVICES=4` (the
suite default is 2 → no padding → why CPU passed).  Cause: anisotropic cone (voxel_slice_aspect=2.9
→ 14 slices) padded 14→16 at 4 dev; the forward gather assembles the PADDED cylinder and the
device-form padded shape leaks to tests that assume the real shape.  B5 fixes it.

**Next (details in the status TOP handoff NEXT):**
- **Daily regression-check tool — likely next (Greg's call):** a `cone_baseline_scaling` variant
  sweeping ALL geometries × ops, writing a dated YAML and diffing vs the previous day's to flag
  time/memory regressions.  Design + gotchas in the plan §Adjacent tasks ("Daily regression-check
  tool") — memory + speedup-ratio + structural flags are the robust gates; record the git hash for
  bisection.  Would have caught all three of this session's ruler/perf issues.
- **GPU re-validation (Greg):** re-run cone+parallel `cone_baseline_scaling` / `fbp_filter_scaling`
  with the ruler fixes — confirm the filter snaps back to prerelease numbers and the 1024³ cone VCD
  fits; interpret the B4.4 sweep (per-device peak ~1/n_dev; the CAPACITY win; the back
  horizontal-recompute penalty vs §8a).
- **B4.5:** hoist the cone back horizontal fan ONLY if the B4.4 penalty demands it (deferred behind
  measurement).
- **B5 (inert padding for cone) — FIXES the 4 deferred tests** (see KNOWN ISSUE above): forward
  gather crop to the real slice count; reconcile device-form-vs-real shape in the geometry tests /
  the internal `sparse_back_project` contract; a `_supports_slice_padding()` hook so "B4 = dividing
  counts" is explicit.
- Then C (parallel conversion + delete the monolithic cone kernel) → translation/multiaxis →
  retirement cascade → the multi-GPU user docs page.

Reminders:
- **Stage only / DRAFT commit messages; never run `git commit`** — I commit from PyCharm.
- Tests: `source /Users/gbuzzard/miniforge3/etc/profile.d/conda.sh && conda activate mbirjax`;
  full suite `python -m pytest -n auto tests/` (~66 s; single-process ~165 s); sharding specifics
  via `MBIRJAX_NUM_CPU_DEVICES=4 python -m pytest tests/sharding/`; per-geometry recon tests live
  in `tests/geometries/`.  **Use `python -m pytest`, not bare `pytest`** — on the cluster bare
  `pytest` resolved to a stale `~/.local/bin/pytest` whose shebang Python lacks pytest (fixed in
  `dev_scripts/run_tests.sh`).  **Multi-device cone exercises sharding on virtual CPUs at
  `MBIRJAX_NUM_CPU_DEVICES=4`** — that catches the padding bugs the 2-device default misses.
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
