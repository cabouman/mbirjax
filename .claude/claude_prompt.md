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

* At the start of a session or after context compaction, read `.claude/status.md` for current project state, open investigations, and larger goals. This prevents narrow investigations from losing sight of the broader picture.
* Before context is compacted, update `status.md` if significant progress has been made, so the next session can orient quickly.

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

For mathematical or numerical code, prioritize correctness, conditioning, memory behavior, and computational efficiency over stylistic changes.

Remember, we're in the curiosity business, so a little bit of exploration and a lot of understanding are much better than a quick fix.