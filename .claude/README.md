# Orientation — beta sharding worktree

This is the **beta sharding** worktree: branch `greg/parallel_sharding`, built
fresh from `prerelease`. Multi-device sharding is being reimplemented here under
a new view-sharded design.

**Read these first (in order):**
1. `claude_prompt.md` — guidelines for collaboration.
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

---

## CRITICAL: which `mbirjax` code runs (beta vs research)

Both worktrees share **one** conda env and **one** editable install (`pip install -e .`
was done from the research worktree).  So selecting the conda interpreter is NOT
enough to pick beta code — what decides it is `sys.path`:

- The **editable-install fallback always points at the research worktree**
  (`…/Research/mbirjax/mbirjax`).
- Beta code is used only when the **beta worktree root is earlier on `sys.path`**
  than the editable install.

How that plays out:
- **PyCharm**: puts the project root on PYTHONPATH automatically, so running from
  the **beta project** uses beta code (and the research project uses research).
  This "just works" — keep each project's run-config Working Directory at its own
  root.
- **`python -m pytest` from a worktree root**: cwd is on `sys.path[0]`, so the
  local `mbirjax/` shadows the install → uses **that worktree's** code. Good.
- **`python path/to/script.py`**: Python puts the **script's directory** (not the
  cwd) on `sys.path[0]`.  A script under `experiments/.../` has no local
  `mbirjax/`, so it falls through to the research editable install — **WRONG
  code**.  Fix: run with the beta root on PYTHONPATH, e.g.
  `PYTHONPATH=<beta-root> python experiments/.../script.py`.

**Every scaling script prints a banner: `*** beta ***` or `### RESEARCH ###`
(with a warning).  Always glance at it before trusting results.**

---

## Running the scaling scripts

Location: `experiments/sharding/scaling_tests/`.  From the **beta worktree root**:

```bash
source /Users/gbuzzard/miniforge3/etc/profile.d/conda.sh && conda activate mbirjax
cd <repo-root>                       # …/Research/mbirjax
PYTHONPATH="$PWD" python experiments/sharding/scaling_tests/fbp_filter_scaling.py
#   --size-set {quick,medium,large}   problem-size preset (default quick)
#   --device-counts 1 2 4 8           device sweep (default: power-of-2 ladder)
#   --warmup 1 --trials 3             timing iters
```
(In PyCharm just run the script from the beta project — PYTHONPATH is handled.)

- Results (YAML + PNG plots) go to `scaling_tests/results/` (gitignored).
- Correctness compares to a prerelease baseline in `scaling_tests/baselines/`
  (gitignored), captured once via `fbp_filter_capture_baseline.py` run from a
  prerelease checkout.  Without it, only a single-device self-reference is shown.
- Sinogram is sharded by **view** (axis 0), so the device count must divide
  `num_views`; incompatible (size, device-count) points are skipped.

## Running the sharding tests

```bash
cd <beta-root> && python -m pytest tests/test_sharding_*.py -q
```
(Run from the beta root so cwd→sys.path gives beta code.  These tests assert
beta-specific behavior, e.g. sinogram shards on axis 0, so they'd fail on
research code — a useful cross-check that you're on beta.)
