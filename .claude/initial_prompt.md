We're continuing work on `mbirjax` (branch `greg/kernel_investigation`), focusing on
section 1 of `plans/current_plans.md` — implementing the flash-remediation padding.

**IMPORTANT — workflow reminder:** discussion first for code AND doc changes; propose and
wait for approval (read `.claude/claude_prompt.md` closely for more).  Stage only, never
commit.  Terminology: "variants" (not arms/cells/grid for variant sets); "ground truth
phantom" (not truth grid).

Read for orientation (verify claims against code; the first three carry the full state):
1. `plans/current_plans.md` — THE evolving forward plan.  §1 lists the implementation
   order; the full design is in the remedies page (next item).
2. `plans/flash_remediation/phase_2d_remedies.html` — the implementation spec: per-case
   equations, code sketches keyed to the actual functions, pros/cons, and the settled
   design choices.  (Also published at `/depot/bouman/www/mbirjax/flash_remediation/`.)
3. The flash-remediation memory (auto-loaded) — the running summary of the investigation.
4. `.claude/lessons.md` — engineering playbook.

Also skim for context:
5. `plans/flash_remediation/flash_remediation_plan.md` — investigation findings.
6. `plans/flash_remediation/README.md` — pages + how to regenerate figures and refresh the
   self-contained HTML reports (`embed_report_figures.py`).  NOTE: the PNGs under
   `plans/experiments/flash_remediation/figures/` are gitignored — the base64 copies
   embedded in the HTML are the durable record; regenerate per the README if files are
   needed on disk.
7. `plans/README.md` — the index of all internal plans/findings docs (docs at
   `plans/<area>/`, supporting scripts at `plans/experiments/<area>/`).

## Summary of current status

The proposals are PLAN ONLY — no code yet.  **`phase_2d_remedies.html` is the
synthesis**: per-case verdicts with rationale, equations, code sketches, and pros/cons.

- **Axial** = per-end excesses E_top/E_bot added automatically in
  `auto_set_recon_geometry` (NOT a symmetric 1+R/SID scale — that under-pads one end
  whenever det_row_offset ≠ 0).  Each end: E = max(0, |v_end|·(SID+R)/SDD − H_iso/2)
  with v_end the detector row-edge heights INCLUDING det_row_offset; ceil per end;
  helical ends attach at z_max/z_min of the travel; shift recon_slice_offset by
  (E_top−E_bot)/2; R from the RECON grid.  `scale_recon_shape` stays a pure scaler and
  just WARNS on uncompensated lateral growth.  Rationale (Greg): a holder always leaves
  the FoV at one end, so truncation is the norm; over-padding past the bound is provably
  harmless.  Open implementation checks: R = RoR-mask radius vs grid half-diagonal
  (check what the projector actually updates), and implement via the model's own
  coordinate chain (`recon_ijk_to_xyz` → `geometry_xyz_to_uv_mag` →
  `detector_uv_to_mn` inverted at the row edges).
- **Lateral** = DETECT-AND-WARN only (deliberate do-nothing on auto-padding), reusing
  the `_get_sino_indicator` support mask already computed in
  `auto_set_regularization_params` (support touching the edge channels ⇒ truncated;
  free, no new threshold).
- **Split** = h_recon = ceil(h_sino·(1+R/SID)·ρ) + 2 with ρ = δ_row/(mag·δ_slice) as the
  default; `align_split_grid` opt-in (at ρ=1 the row and slice grids are commensurate,
  so alignment needs a sub-slice recon-grid shift — NOT reachable by index choice);
  taper retired in the same change (it fails at 8× downsampling — the shipped default
  visibly stripes there).

Implementation-session expectations: the default-shape change will touch shape-sensitive
tests — exact float equality is never the gate for computed floats (use the
`tests/sharding/conftest` helpers); re-baseline the regression dashboards in the same
change (§1 step 4); Phase-3 real-scan validation (SiC, z62, BGA) gates shipping the
defaults.

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
