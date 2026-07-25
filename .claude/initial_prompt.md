We're continuing work on `mbirjax` (branch `sharpness_schedule`), with a focus on
designing and implementing a simple per-iteration schedule for sharpness and snr_db.

**IMPORTANT — workflow reminder:** discussion first for code AND doc changes; propose and
wait for approval (read `.claude/claude_prompt.md` closely for more).  Stage only, never
commit.  Terminology: "variants" (not arms/cells/grid for variant sets); "ground truth
phantom" (not truth grid).

Read for orientation (rely on code and results over memory and .md files):
1. `plans/current_plans.md` — THE evolving forward plan.  §1 lists the implementation
   order; the full design is in the remedies page (next item).
2. `.claude/lessons.md` — engineering playbook.
3. `.claude/cluster_use.md` — info about interacting with our GPU cluster and some other 
   miscellaneous info.

Also skim for context:
4. `plans/README.md` — the index of all internal plans/findings docs (docs at
   `plans/<area>/`, supporting scripts at `plans/experiments/<area>/`).

Current task:  Item #1 from `plans/current_plans.md`.

## Standing context

- Cluster: gautschi (ssh BatchMode); sbatch on partition `ai`, account `bouman`,
  **--cpus-per-task=14 required per GPU**; P2 job/staging dir `~/flash_p2b` (results in
  `results/`); Lilly data + recon volumes + analysis scripts at
  `/scratch/gautschi/buzzard/flash_lilly`.  Nested ssh via the login node; coordinate
  before heavy use.
- Metrics/dashboard interplay for the padding change: the nightly memory gate is
  vs-prior (it alerts ONCE, then the new numbers become the baseline), and the engine's
  new `policy` block records partition_sequence/max_iterations but NOT recon-shape
  defaults — so when the axial padding lands, either add a note to
  `mbirjax_metrics/results/annotations.yaml` (rendered as purple bottom-band chart
  markers with tooltips) or extend the policy block to record a padding flag, in the
  same change.
- Verification habits: pgrep -f self-matches its own ssh command (check log files, not
  process greps); VCD prints "Error sino RMSE" every iteration (don't grep bare "Error"
  for failure detection); the jax persistent compile cache makes warm cluster runs much
  faster than first runs.
- Preview: `.claude/launch.json` entry `flash-remediation-page` serves
  `plans/flash_remediation/` on port 8932 (`index.html` is the overview);
  `plans/flash_remediation/publish_pages.sh` rsyncs the self-contained pages to the
  public depot www (HTML only — the destination is publicly served).
