

This is the mbirjax CT reconstruction project — multi-GPU/CPU sharding work in the
`mbirjax` worktree on branch `greg/parallel_sharding`.

Orient first by reading, in order:
1. `.claude/claude_prompt.md` — collaboration style + workflow (investigate and propose
   before editing; wait for approval on non-trivial changes; minimal, localized changes;
   suspect the ruler before the code; stay curious and challenge my assumptions).
2. `experiments/sharding/plans/sharding_status.md` — the TOP HANDOFF (current state + NEXT).
3. `experiments/sharding/plans/sharding_implementation_plan_v2.md` — forward plan.  Read §0,
   §"P5 Step 4 — divisibility padding", §P6 (+ the (g0,L) design-note amendments), §Decisions,
   §Adjacent tasks.  (`sharding_implementation_plan.md` = completed-work record + principles.)
4. Skim `.claude/lessons.md` — the jax/GPU/placement/measurement playbook ("Sharded VCD
   memory: reference cycles + buffer donation" is the load-bearing one).
Verify claims against current code; memory/docs may lag.

**Where we are (2026-06-11).**  Placement architecture live (ParallelBeam, always-on); hybrid
removed; P5 done except Step 4 Stage 2.  **Step 4 Stages 0–1 LANDED, CPU-green:** match-input
is retired for an explicit `output_sharded=False` kwarg on user-facing methods, and the VIEW
axis zero-pads to any device count with the padding **exactly inert** (entry zero-fill /
`prepare_sino_for_devices`, ONE forward-output mask, real-count normalizations; params always
hold the problem's REAL shapes, placements own the padded device shapes + per-shard ranges).
Auto = all GPUs dividing num_slices (views never constrain).  `self.use_gpu` retired (params =
request, placements = resolution; `model.device_summary` reports the outcome).  Sphinx docs:
stale use_gpu/hybrid entries fixed; the multi-GPU user page is planned (v2 §Adjacent tasks,
incl. the "sharding sparingly" vocabulary rule) for after P6.

**Next (details in the status NEXT):**
- **Step 4 Stage 2 — slice padding:** forced-zero padded slices + zero-weight padded rows +
  the qGGMRF mid-shard boundary mask (reflected BC at the last real slice).  **Write the
  kernel-level proposal against `mbirjax/qggmrf.py` for my review BEFORE implementing.**
- **GPU validation of Stages 0–1** (GPUs were down) — the status GPU items.
- Then **P6** (geometry port + deletion cascade; anchor rule + forward-banding-as-accumulation
  are recorded in the (g0,L) design note) and the **adjacent tasks** (settable view params,
  CPU-cluster auto-sharding, the docs page, seed test RNGs).

Reminders:
- **Stage only / DRAFT commit messages; never run `git commit`** — I commit from PyCharm.
- Tests: `source /Users/gbuzzard/miniforge3/etc/profile.d/conda.sh && conda activate mbirjax`;
  sharding suite via `MBIRJAX_NUM_CPU_DEVICES=4 pytest tests/sharding/` (view padding:
  `test_padding.py`; sharded VCD: `test_vcd_sharded.py`).
- I run GPU/large recons on the cluster — **flag GPU items.**  Fresh `pip install -e .` on the
  cluster after edits (a stale build once impersonated a leak); pre-flight `nvidia-smi dmon`
  (thermal throttling); `results/`/`baselines/` are gitignored — record decisions in docs.
- Correctness gate is **`allclose` ~1e-4**, not bit-exactness.
- jax/GPU specifics (sharded-array reference cycles + buffer donation, `peak_bytes_in_use` as
  the memory ruler, the benign `CUDA_ERROR_NOT_PERMITTED` warning, time ∝ N⁴ / memory ∝ N³):
  details in `.claude/lessons.md` — consult before re-deriving.
