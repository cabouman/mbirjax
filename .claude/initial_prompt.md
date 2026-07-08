We're starting the next work item in `mbirjax`.  The projector-kernel campaign is COMPLETE
and committed on `greg/kernel_investigation` (TilePolicy, sorted channel reduction with
measured guards, DRY fan kernels, the concrete-scatter-centers rounding-bug fix, and the
plans/ docs consolidation).

**IMPORTANT — workflow reminder:** do NOT start code OR changes to documents without a
discussion first.  Analyze, propose a concrete plan with tradeoffs, and wait for approval
(see `.claude/claude_prompt.md`).  This applies to experiment scripts too unless explicitly
told to proceed.

Read for orientation (verify any code claim against the actual code; docs may lag):
1. `.claude/claude_prompt.md` — collaboration style + workflow (stage only, no commits;
   GPU work runs on gautschi via sbatch; sweep, don't guess).
2. **`plans/current_plans.md`** — THE evolving forward plan.  §0.5 summarizes the finished
   kernel campaign; the numbered sections are the open items.
3. `plans/README.md` — the index of all internal plans/findings docs (docs at
   `plans/<area>/`, supporting scripts at `plans/experiments/<area>/`).
4. `.claude/lessons.md` — the engineering playbook.

## This session's focus: current_plans.md §2 — sinogram weight edge tapering

The item: `ConeBeamModel` → `split_sino_recon()` already uses a per-detector-row sine
filter on sinogram weights to reduce ringing from the rect window in detector rows when
splitting a recon in two.  Separately, objects extending outside the field of view are
observed to converge more slowly and produce a 'flash' artifact.

Investigate whether a geometry-adaptive (and possibly data-adaptive) tapering of sinogram
WEIGHTS at the edges — detector rows AND channels — can (a) speed convergence and/or
(b) reduce the flash from objects partially outside the FoV.

Suggested shape of the work (to be refined in discussion BEFORE anything is written):
- Start by understanding the existing precedent: the `split_sino_recon` sine filter (why it
  works, what it windows) and where sinogram weights enter the VCD updates.
- Characterize the failure mode first: a small reproducible case with an object extending
  outside the FoV (convergence curve + the flash), so any taper has a measurable target.
- Then propose taper candidates (shape, width, rows vs channels, geometry- vs data-adaptive)
  and an evaluation design (convergence metrics on synthetic + real scans; note that
  tapering intentionally changes the OBJECTIVE, so "correctness" needs a definition before
  any gating).

Open questions worth raising early: how to gate quality (fingerprint comparisons don't
apply when the weights intentionally change); whether the taper belongs in preprocessing
(user-visible weights) or inside recon (internal); interaction with the existing
`split_sino_recon` filter and with MAR's weight handling.

## Standing context

- Nightly watch items from the kernel campaign (memory-gate acks) are listed in
  `plans/current_plans.md` §3; they need the acknowledged-regression path, not code.
- Companion repos parallel to mbirjax: `mbirjax_metrics` (perf tracking; the
  measure_one_cell harness), `mbirjax_applications`.
- Cluster: gautschi via ssh (BatchMode key auth), partition `ai`, account `bouman`;
  standing bench infra in `~/viewbatch_fix_verify/`; snapshot dirs `~/kernel_ab_{old,new}`;
  `PYBIN=$HOME/.conda/envs/mbirjax/bin/python`.
- Any new script must set env vars / `import mbirjax` before anything touches jax.
