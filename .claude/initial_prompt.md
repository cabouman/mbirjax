We're continuing the flash-remediation program in `mbirjax` (branch
`greg/kernel_investigation`), picking up mid-investigation.

**IMPORTANT — workflow reminder:** discussion first for code AND doc changes; propose and
wait for approval (read `.claude/claude_prompt.md` closely for more).  Stage only, never commit.  Terminology:
"variants" (not arms/cells/grid for variant sets); "ground truth phantom" (not truth grid).

Read for orientation (verify claims against code; the first two carry the full state):
1. The flash-remediation memory (auto-loaded) — the running summary of Phases 1–2.
2. `plans/flash_remediation/flash_remediation_plan.md` — plan of record, all findings,
   INCLUDING the "P2c CORRECTION — real data overrules the synthetic" section that is the
   live thread.
3. `plans/flash_remediation/README.md` — pages + how to regenerate figures and refresh the
   self-contained HTML reports (`embed_report_figures.py`).
4. `.claude/lessons.md` — engineering playbook.

Also skim for context:
5. `plans/current_plans.md`** — THE evolving forward plan.  §0.5 summarizes the finished
   kernel campaign; the numbered sections are the open items.
6. `plans/README.md` — the index of all internal plans/findings docs (docs at
   `plans/<area>/`, supporting scripts at `plans/experiments/<area>/`).

## Where the last session stopped (the live thread)

Greg challenged the P2c synthetic verdict with real data; the Lilly D01788 investigation
(reproduction, ablations, revised plan-only proposal) is fully recorded in the plan doc's
P2c CORRECTION section.  What remains in flight:

1. **A synthetic reproduction of the Lilly seam stripes is still NOT achieved** (two
   attempts, results already retrieved to `plans/experiments/flash_remediation/results/`):
   `structured_widefan` (Lilly's R/SID 0.2, unit weights, 40 iters) was CLEAN (no_taper
   seam-vs-ref 1e-4), and `widefan_noise15` (photon noise + transmission weights + 15
   iters) raised all variants to ~2e-3 but WITHOUT Lilly's signature ordering (its taper
   0.0026 ≥ no_taper 0.0020 ≥ deep 0.0016 — on Lilly, taper beat no_taper 11×; the ~2e-3
   is likely noise-realization sensitivity, not the structural stripes).  Remaining
   suspects: detector-rotation correction residue, dynamic range / regularization balance,
   real object structure at the seam, det_row_offset asymmetry.  Either iterate further
   (cheap sbatch runs via `split_seam_repro.py` RUNS entries) or report the reproduction
   gap honestly — do not paper over it.
2. **`phase_2c_split_results.html` still needs the real-data correction section** (Greg
   approved): Lilly reproduction + figures + corrected verdict + the synthetic-repro
   outcome from (1).  Lilly figures are already at
   `plans/experiments/flash_remediation/figures/p2c_lilly_{seam_xz,seam_profiles,ablation_rms}.png`
   (gitignored) — register in `embed_report_figures.py` FIG_MAP, write captions, re-embed.
   Also revise the page's TL;DR/§3 verdict and the `index.html` Phase-2c card (both still
   state the withdrawn "drop the taper" conclusion).
3. The revised proposal (geometry-derived h_recon) is PLAN ONLY — Greg said no code yet.

## Standing context

- Cluster: gautschi (ssh BatchMode); sbatch on partition `ai`, account `bouman`,
  **--cpus-per-task=14 required per GPU**; P2 job/staging dir `~/flash_p2b` (results in
  `results/`); Lilly data + recon volumes + analysis scripts at
  `/scratch/gautschi/buzzard/flash_lilly` (NSI scan D01788; the redundant
  `Geometry.nsipro` was moved aside to avoid an interactive config prompt).  Greg's
  interactive node h003 has the editable mbirjax install pinned at commit 568f6b7
  (the 6/26 no-taper split) — nested ssh via the login node; coordinate before heavy use.
- Verification habits from this session: pgrep -f self-matches its own ssh command (check
  log files, not process greps); VCD prints "Error sino RMSE" every iteration (don't grep
  bare "Error" for failure detection); the jax persistent compile cache makes warm cluster
  runs much faster than first runs.
- After the P2c correction lands: Phase 3 (real scans: SiC axial, z62 radial, BGA severe
  truncation) and Phase 4 (policy/API: overshoot detector, z-pad geometry formula, split
  h_recon change) per the plan doc.
- Preview: `.claude/launch.json` entry `flash-remediation-page` serves
  `plans/flash_remediation/` on port 8932 (`index.html` is the overview).
