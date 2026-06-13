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

**Where we are (2026-06-13).**  A PR to prerelease is open (ParallelBeam beta complete).
P5 is done and GPU-validated end to end (always-on placements, automatic multi-GPU via exactly
inert padding, `configure_devices` / `device_summary` / `prepare_sino_for_devices` /
`output_sharded`); adjacent tasks done (settable view parameters + vcls; `_auto_shard_cpu=True`;
per-user `~/.mbirjax/` home); test suite overhauled (per-geometry `tests/geometries/`,
`pytest -n auto` ≈ 66 s).  **Now in P6, the cone port, design measured-and-settled** (read
§8a-design + `p6_increment_b_design.md`): increment A (channel-major both cone horizontal fans)
is COMMITTED — a CPU win (~13×), GPU-NEUTRAL (so no easy single-GPU projector speedup from
kernel layout; the port's GPU value is CAPACITY).  Increment B1 (banded cone BACK kernel +
anchor/clip + `tests/geometries/test_cone_banded.py`) is committed, CPU+GPU green.  The cone
sharded **forward = C (per-pixel-batch all-gather + monolithic), GPU-confirmed; B (banded/streamed)
dropped** (slower, no memory win).  Sinogram row-sharding parked (`.claude/sinogram_sharding.md`).

**Next (details in the status NEXT + `p6_increment_b_design.md`):**
- **Remove the now-unneeded forward banded kernel** (`forward_project_band_to_one_view` /
  `forward_vertical_fan_band_*`, since forward = C) and adjust `test_cone_banded` (back
  correctness is already covered by the back band-decomposition + the monolithic adjoint).
- **B2 — the cone sharded driver**: banded BACK reduce-scatter + gather+monolithic FORWARD,
  delete `entries_per_cylinder_batch`; gate memory/timing vs the §8a baseline.  Then B3
  (de-closuring / module-level jit — kills the per-model retrace cost; tracing, not XLA
  compile, dominates first-call latency, measured), B4 (sharded cone + GPU validation,
  stage2-pattern), B5 (inert padding).
- Then C (parallel conversion + delete the monolithic cone kernel + transitional branch) →
  translation/multiaxis → retirement cascade (main_device/sinogram_device, is_sharded guards,
  view_indices, RETIRE-AFTER sweep) → the multi-GPU user docs page ("sharding" sparingly —
  speak in terms of multiple GPUs adding capacity and speed).

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
