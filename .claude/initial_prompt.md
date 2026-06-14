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

**Where we are (2026-06-13, P6 cone port — CONE NOW SHARDS).**  P5 done + GPU-validated (always-on
placements, automatic multi-GPU, exactly-inert padding, `configure_devices` / `device_summary` /
`prepare_sino_for_devices` / `output_sharded`).  P6 increments **A** (channel-major cone horizontal
fans — CPU win, GPU-neutral; the port's GPU value is CAPACITY), **B1** (banded cone BACK kernel),
**B2** (single-device cone on the banded back kernel; §8a-NEUTRAL both platforms), **B3**
(module-level projector drivers — jit cache SHARED across model instances; the blocker was a fresh
namedtuple CLASS per `get_geometry_parameters()` call, fixed via `make_geometry_params` — see
lessons.md), and **B4.1–B4.3 are COMMITTED.  CONE SHARDS** (recon by slice ⇄ sinogram by view;
banded reduce-scatter BACK + gather+monolithic FORWARD = decision C), CPU-validated end to end at
DIVIDING counts (`tests/sharding/test_cone_sharded.py`: back/forward/Hessian 1e-5 + 3-iter VCD 1e-4,
n=2/4, circular+helical — **this test file is UNTRACKED, commit it**).  Detail:
`p6_increment_b_design.md` PROGRESS block.

**KNOWN ISSUE — deferred to B5 (Greg's call; do NOT chase before B5).**  Flipping cone's
`_supports_sharding()` enabled the geometry-agnostic P5 PADDING, but cone padding IS B5 (not done).
So at non-dividing slice counts (≥4 devices) cone auto-pads and **4 tests FAIL on multi-device**:
`test_{adjoint,hessian}_anisotropic_cone` (test_projectors), `test_split_sino`,
`test_vcd_anisotropic_cone` (test_vcd).  Reproduce on CPU with `MBIRJAX_NUM_CPU_DEVICES=4` (the
suite default is 2 → no padding → why CPU passed).  Cause: anisotropic cone (voxel_slice_aspect=2.9
→ 14 slices) padded 14→16 at 4 dev; the forward gather assembles the PADDED cylinder and the
device-form padded shape leaks to tests that assume the real shape.  B5 fixes it.

**Next (details in the status NEXT + `p6_increment_b_design.md`):**
- **B4.4 (GPU — Greg running):** `cone_baseline_scaling.py` multi-device sweep (n_dev 1/2/4) at
  DIVIDING sizes (256/512/1024³ — no padding): per-device peak ~1/n_dev; the CAPACITY win (a 1024³
  VCD that OOM'd single-device now fits sharded); the back horizontal-recompute penalty vs §8a.
- **B4.5:** hoist the cone back horizontal fan ONLY if B4.4's penalty demands it (the pixel-batch-
  outer loop reorder; deferred behind measurement).
- **B5 (inert padding for cone) — FIXES the 4 deferred tests.**  For cone: forward gather crop to
  the real slice count (padded slices are zero → exact; prototyped+reverted in the B4.4 session);
  reconcile device-form-vs-real shape in the geometry tests / the internal `sparse_back_project`
  contract; consider a `_supports_slice_padding()` hook so the "B4 = dividing counts" contract is
  explicit; the masks are mostly geometry-agnostic (done).
- Then C (parallel conversion + delete the monolithic cone kernel + transitional branch) →
  translation/multiaxis → retirement cascade → the multi-GPU user docs page.

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
