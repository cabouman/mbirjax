I value insight, careful reasoning, and collaborative investigation over rapid code generation.

This codebase is complex, and I know a great deal about its mathematical structure, algorithms, and historical design decisions. I can often answer questions about intent, architecture, and prior experiments. I rely on you to:

* catch errors in my reasoning,
* identify edge cases and hidden assumptions,
* suggest tools, libraries, or implementation approaches I may not know,
* call out asymptotic complexity, memory scaling, and likely GPU bottlenecks explicitly,
* be alert for shape polymorphism, unnecessary recompilation, memory blowups, and host-device transfer issues.
* and provide independent technical judgment rather than simply agreeing with me (please challenge my assumptions when appropriate).
* maintain a sense of curiosity — about what might be missing from an analysis, about what I might be thinking or overlooking, about where the project is leading, and about whether there is a better approach to whatever we are investigating. Ask follow-up questions when this sense is activated, rather than simply confirming and moving on.

Project orientation:

* This is the **beta sharding worktree** (`greg/parallel_sharding`, built from `prerelease`). At the start of a session or after context compaction, read:
  1. `experiments/sharding/plans/sharding_status.md` — current phase, what's blocked, verified hardware facts.
  2. `experiments/sharding/plans/sharding_implementation_plan.md` — the detailed Phase 0–F checklist, migration table, cross-cutting principles, and open design questions.
* Prior art lives on the research branch (`greg/parallel_tests`, tag `research-snapshot-2026-05-29`) in the sibling worktree `…/Research/mbirjax/`. That branch will eventually be deleted — migrate anything worth keeping first (the plan tracks this).
* Before context is compacted, update `sharding_status.md` (and the plan's checkboxes) if significant progress has been made, so the next session can orient quickly.

When investigating or diagnosing problems:

* Before launching a sequence of exploratory tool calls, briefly state the hypothesis or question being investigated. This allows for redirection before time is wasted on the wrong approach.
* Flag when a conclusion rests on limited evidence — for example, a test that covers only a subset of the relevant parameter space. Push for broader coverage before treating something as resolved.

Before creating new files or modifying existing ones:

1. First analyze the relevant files and summarize your understanding.
2. Propose a concrete plan and discuss tradeoffs, risks, and alternatives.
3. Ask questions if requirements or intent are unclear.
4. Wait for approval before making edits.

**Exception:** Small, self-contained experiment files in designated directories (e.g., `experiments/`) may be created when you have been explicitly asked to proceed. Mention the creation and its purpose.

When implementing:

* Prefer minimal, localized changes unless a broader refactor is justified.
* Preserve existing APIs and behavior unless explicitly discussed.
* Explain important design decisions and assumptions.
* Include comments that explain the intent and flow of the code at the level of a good python programmer who may not know jax or parallel programming well.
* Highlight anything that seems inconsistent, fragile, numerically unstable, inefficient, or technically risky.
* Avoid speculative cleanup or unrelated refactors.
* When possible, suggest ways to test or validate correctness and performance.

Scripts and reproducibility:

* I almost never run scripts with command-line arguments, and prefer not to add
  them.  Instead, put run parameters either in a checked-in config file (YAML) or
  as clearly-labeled values at the top of the script that are easy to find and
  edit.  Either way the configuration is versioned, so runs are reproducible.
* Favor evidence that is apples-to-apples: when comparing implementations, vary
  exactly one thing and hold the rest identical.  Take the extra few minutes to
  do it properly rather than moving quickly on improper evidence.

Performance and measurement (jax/GPU — learned on this project; full playbook
with the F1 case study in `.claude/lessons.md`):

* Per-worker compute must be jitted — eager, op-by-op dispatch silently kills
  multi-device scaling.
* For honest GPU memory: measure each config in an isolated subprocess, keep the
  orchestrator JAX-free (so it holds no device memory while a worker measures),
  and free the previous result before the next allocation (`peak_bytes_in_use`
  is a process-cumulative high-water mark).
* Fold scalars into a small operand (e.g. an f32 filter), never as an
  out-of-place full-array multiply — a float64 scalar like `np.pi` silently
  promotes the whole array to f64, doubling memory.
* The general measurement principles (suspect the ruler before the code;
  single-variable ablations; name+verify the assumption a fix rests on; sweep,
  don't guess) live in global memory and apply everywhere.

For mathematical or numerical code, prioritize correctness, conditioning, memory behavior, and computational efficiency over stylistic changes.

Remember, we're in the curiosity business, so a little bit of exploration and a lot of understanding are much better than a quick fix.
