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

1. **The P2c page was REWRITTEN 2026-07-09** (Greg-approved real-data-first storyline:
   problem → cause → interventions → synthetic lessons + honest gap); figures are the
   windowed variant montages from `lilly_variant_figures.py`.  Key late finding folded
   in: **at 8× downsampling (the new fast-turnaround workhorse) the stripes persist and
   the shipped taper STOPS working** (6.1e-3 vs no-taper 7.9e-3), while the
   geometry-derived extension fixes it (9.0e-4; formula depth h_recon=9 == 12) — the
   h_recon proposal is a defect fix, not a cleanup.  All embedded, preview-verified,
   staged.
2. **The synthetic reproduction is CLOSED (2026-07-09 evening)** — the stripes' driver
   is the SUB-ROW MISALIGNMENT between the sino cut row and the recon split slice (~0.4
   rows on Lilly).  Found by REVERSE ablation on the real data (consistent sino stripes
   7.7e-3 → inconsistency out; unit weights stripe 8.5e-3 → weights out; zeroed axial
   offsets CLEAN 1.1e-5, a 740× drop) after eleven build-up conditions failed; confirmed
   by a fully synthetic dose-response (det_row_offset 0.15/0.30/0.45 rows →
   7e-5/2.7e-4/3.3e-3, object-INDEPENDENT, zigzag signature, the real Lilly point on the
   curve).  Mechanism refinement: aligned symmetric truncation is benign at h=5; default
   synthetic models lock the grids (mismatch always 0) so they can never show this
   artifact.  Full chain: plan doc P2c CORRECTION + `split_seam_lilly8x.py` header +
   `lilly_consistency_check.py`/`lilly_cons2.py`.  Also from earlier in the day: the
   library-version confound was checked and ruled out (both 568f6b7 and the current
   branch stripe; the shipped taper split at the current branch is clean at 4×,
   4.1e-4), with provenance proven from the git reflog after Greg flagged checkout
   ambiguity.  WORKFLOW LESSONS (Greg, 2026-07-09): show provenance BEFORE claiming
   version-dependent results (the editable install uses a META-PATH finder — assert
   mbirjax.__file__ in-process); when build-up search stalls, REVERSE-ablate the real
   failing case.
3. The proposals are PLAN ONLY — Greg said no code yet.  **`phase_2d_remedies.html`
   (added 2026-07-09) is the synthesis**: per-case verdicts with rationale, equations,
   code sketches, and pros/cons — axial = extend to the exact bound (1+R/SID)
   AUTOMATICALLY in auto_set_recon_geometry, R from the RECON grid so it composes with
   lateral padding (Greg: a holder always leaves the FoV at one end, so truncation is
   the norm; costs = default-shape change → re-baseline the regression dashboards);
   lateral = DETECT-AND-WARN only (deliberate do-nothing on
   auto-padding); split = geometry h_recon as default + `align_split_grid` opt-in (at
   ρ=1 the grids are commensurate, so alignment needs a sub-slice recon-grid shift — it
   is NOT reachable by index choice) + taper retired with the change.

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
