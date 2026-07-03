We're continuing improvements to `mbirjax`, working branch `greg/performance_improvements`.

State: the core sharding, the MAR/preprocessing sharding, and the large-problem memory/robustness work
shipped on `greg/shard_profiling` (PR to `prerelease` and soon to `main`).  This branch is the next phase — performance — and will 
sync with `main` once that PR merges.

Read for orientation (these are not the source of truth — verify any code claim against the actual code;
docs/memory may lag):
1. `.claude/claude_prompt.md` — collaboration style + workflow.
2. `experiments/sharding/plans/post_shard_plans.md` — **Main priorities.**
3. `.claude/lessons.md` — the consolidated engineering playbook (float gates, sharded/jitted-code rules,
   the 2^31 boundary, honest measurement, performance expectations).  Short and organized by task.

There are companion repos parallel to mbirjax:
 * `mbirjax_metrics`: performance tracking and profiling.
 * `mbirjax_applications`: production-scale examples and workflows.

Working reminders:
- Stage only / draft commit messages — Greg commits from PyCharm; do git surgery via CLI, not PyCharm.
- Flag GPU/cluster items — Greg runs those on the cluster.
- Exact equality is never the gate for computed floats — use the scale-invariant
  `tests/sharding/conftest.assert_sharded_allclose` for sharded-vs-single comparisons.
- The sharded VCD loop is geometry-independent, so a new geometry needs no per-geometry sharded
  VCD-recon test.
- The GPU memory fraction is `os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.94')` —
  overridable per-run via the environment; out-of-pool allocations (NCCL, cuSolver) live in the
  remainder (lessons.md §7).
- jax/jaxlib version discipline is handled by the metrics regression workflow (0.10.2 excluded; new
  releases are built against the previous commit first to isolate toolchain effects) — check the
  `toolchain` field in regression YAMLs rather than re-deriving it.
- Any new script must set env vars / `import mbirjax` before anything touches jax — the memory-fraction
  and log-level env vars bind at jax backend initialization.
