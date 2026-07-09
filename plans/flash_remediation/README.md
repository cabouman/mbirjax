# flash_remediation — docs and reports

The FoV-truncation "flash" remediation program (source item: `plans/current_plans.md` §2).

- **`index.html`** — the project overview page; links every report below.
- **`flash_remediation_plan.md`** — the plan of record: mechanism analysis, phase plans,
  findings, and the decisions log.  Start here for the working record.
- **`phase_1_results.html`** — the illustrated Phase 1 characterization report.
- **`phase_2a_axial_results.html`** — Phase 2a: the axial taper-vs-padding story (complete).
- **`phase_2b_radial_results.html`** — Phase 2b: the radial padding-knee story (in progress;
  sweeps run as sbatch jobs on gautschi via `radial_pad_sweep.py`).

The reports are **self-contained**: every figure is embedded as a base64 data URI, because
the repo gitignores PNGs (`*.png` in the top-level `.gitignore`) — the reports survive a
fresh checkout even though the figure files do not.  Open them directly in a browser, or
serve the directory with the `flash-remediation-page` entry in `.claude/launch.json`
(port 8932).

## Regenerating the figures

All supporting scripts live in `plans/experiments/flash_remediation/` (per the plans/
layout rule).  Everything runs on local CPU in the `mbirjax` conda env, from that
directory, with no CLI arguments — run parameters are clearly-labeled values at the top of
each script:

```bash
source /Users/gbuzzard/miniforge3/etc/profile.d/conda.sh && conda activate mbirjax
cd plans/experiments/flash_remediation
```

| script | produces | runtime |
|---|---|---|
| `lateral_truncation_repro.py` | `figures/lateral_*` (Phase 1 radial case) | ~5 min |
| `z_truncation_repro.py` | `figures/z_*` (Phase 1 axial case) | ~10 min |
| `z_taper_pad_grid.py` | `figures/p2a_*` (P2a taper-vs-padding grid) | ~30 min |

Notes:

- **Figure-only tweaks:** `z_taper_pad_grid.py` has a `figures_only` flag at the top —
  set it `True` to rebuild that script's figures from the saved `results/p2a_*` files in
  about a minute (no recons).  The Phase 1 scripts are cheap enough to rerun fully.
- Figures land in `figures/`, metrics and final volumes in `results/` — both gitignored;
  numbers that drive decisions are recorded in the plan doc and the reports.
- Recon results are deterministic (each recon call re-seeds `np.random`, per
  `lessons.md` §2), so a rerun reproduces the reported metrics exactly on the same
  machine/toolchain.

## Refreshing the reports after regenerating figures

```bash
python embed_report_figures.py       # from plans/experiments/flash_remediation/
```

This swaps the new PNGs into both HTML reports **in place**, matching each `<img>` by its
unique `alt` text (so it works on the already-embedded files; it is idempotent).  When a
new figure is added to a report, add its `alt → png` entry to `FIG_MAP` at the top of that
script.  The report prose (values quoted in captions and tables) is hand-written — if a
rerun changes the numbers, update the text too.
