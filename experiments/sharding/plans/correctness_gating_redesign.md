# Correctness gating & visibility redesign — design note (DRAFT, 2026-06-21)

> **STATUS: COMPLETE — P1–P5 shipped in `mbirjax_metrics` (2026-06-21).  P4's CPU↔GPU check activates once the next nightly collects the shared cells; the P4 tolerance is SAFE-LOOSE pending one calibration pass.**  Written
> after reading the live harness code (`mbirjax_metrics`), not the docs.  Agreed shape: layered
> references (vs-main + cross-device + CPU↔GPU; prior-run demoted to a tripwire), adjoint left to the
> existing test suite, dashboard-native alert (sticky banner + tab/favicon badge — no email, no issue),
> and a single `cleared_through` watermark cleared via a guided script.  All §4 decisions settled;
> implementation status per phase in §5.  **Lives in the mbirjax repo plans dir by
> convention, but ALL code changes land in `mbirjax_metrics`** (`tooling/scaling_tests/performance_tracking.py`,
> `tooling/viewer/build_dashboard.py` + `dashboard.js` + `template.html`).  Line numbers are a
> 2026-06-21 snapshot — **trust the symbol name over the number.**  Sibling: `performance_tracking_plan.md`.

---

## 0. The two problems this fixes

1. **Correctness is second-class.**  A fingerprint divergence and a +20 % memory bump land in the
   *same* `gate.hard` list, both flip `result: fail`, and both render as the same red marker + one
   "hard-gate regressions" count.  Nothing distinguishes "the recon is wrong" from "it got slower."
   Correctness is a far more important signal and should look like one.

2. **The prior-run baseline *ratchets*, and a mid-week fail is invisible.**  The gate's sole reference
   is this branch's *previous* run (`_find_prior`), so it only catches **changes**.  If commit B
   introduces a wrong value and C–F keep it, **only B fires** — B's wrong value silently becomes the
   new baseline.  Ratcheting is fine for performance (track the current cost) but dangerous for
   correctness (wrong becomes "normal").  Combined with a pull-only dashboard, a divergence that
   surfaced on a day you didn't look is gone from view.

---

## 1. Verified facts (current correctness path — the ground truth this rests on)

- **The fingerprint** (`fingerprint()`, `performance_tracking.py`): per measured cell, a tolerant dict
  — float64 `{sum, mean, l2norm}` (so it reflects the *value*, not float32 accumulation order), exact
  `{min, max}`, 12 deterministic evenly-spaced `samples`, and structural `{shape, dtype, padding_zero}`,
  all on the TRUE (cropped/unpadded) output.  **No `.npy` arrays are stored.**
- **Computed for every op of every geometry** (`measure_cell_group` → `fp_by_n[n] = fingerprint(...)`),
  i.e. all four geometries: cone/parallel get `direct_filter, forward, back, vcd_nonconst`;
  translation/multiaxis get `direct_filter, forward, back`.
- **Gated only vs the branch's own prior run** (`_gate_fingerprint`, called from `_gate_metrics`):
  same `geom|op|size|ndev` cell, this commit vs `_find_prior`.  HARD on shape/dtype change, on
  `{sum,mean,l2norm}` reldiff > rtol, on a new padding leak; SOFT on > `k_sample_tol` (1) of 12 samples
  drifting.  Tolerance: **1e-5** single-shot (`fp_rtol_single`), **1e-4** for the iterated VCD
  (`fp_rtol_iter`).
- **Cross-device correctness is never asserted.**  Each `ndev` is compared only to its own prior, so
  "sharded n>1 matches single-device n=1" is *not* checked anywhere in the harness today.
- **Cross-platform correctness is never asserted.**  CPU and GPU runs are compared to their own priors,
  never to each other.
- **`build_dashboard` already loads every run's full YAML** (`_parse_run` reads `doc`), so it has every
  cell's fingerprint at build time even though `_slim_cell` drops it from the emitted JSON.  → corpus-level
  correctness analysis is cheap to add at build time without bloating the inlined `window.__METRICS__`.
- **The adjoint property `⟨Ax,y⟩ ≈ ⟨x,Aᵀy⟩` is already in the mbirjax test suite**, and the harness
  already captures + flags test failures (the red triangles + tests-failed tile/tooltip).  → we do
  **not** reimplement it here; projector-pair correctness flows in via the existing tests-failed signal.

---

## 2. Goals / non-goals

**Goals.**  (a) Correctness is its own severity category, visually and structurally distinct from perf.
(b) A persistent wrong value cannot hide — the reference fires *every run* while divergent, not just on
the run that changed.  (c) The signal reaches Greg without him reading logs or email — it lives on the
dashboard he already checks, and is passively visible even in a pinned tab.  (d) Acknowledged/known
divergences can be cleared (through a date) so the alert is signal, not standing noise.

**Non-goals.**  Reimplementing the adjoint test (already in the suite — §1).  Changing the *performance*
gate semantics.  A server-backed dashboard (it stays a static GitHub-Pages build; acknowledgement is a
committed file, not a live button).

---

## 3. Design decisions

### D1 — Correctness is its own severity tier

Split the gate result into two independent axes instead of one flat `result`:

- **perf**: `pass | warn | fail` (memory/time/speedup/structural — today's logic, minus correctness).
- **correctness**: `ok | divergent` (with the worst-firing reference + magnitude).

The hard strings already self-identify (`… fingerprint …` / `… padding leak …` vs `… memory …`), so
classification is a one-liner.  A run can now be *perf-fail but correct* (amber) vs *INCORRECT* (red) —
distinct states.  `result` is retained for back-compat as `fail if (perf==fail or correctness==divergent)`.

### D2 — Layered correctness references

Four references, each catching a different failure mode; they compose.  **Bold = new.**

| Reference | Catches | Robust to impl. diffs? | Baseline needed | Where computed | Tolerance |
|---|---|---|---|---|---|
| prior-run (kept) | day-over-day drift on a branch | yes (branch vs itself) | branch's prev run | engine (today) | tight (1e-5 / 1e-4) |
| **vs most-recent main** | persistent divergence from the canonical answer; the ratchet | tolerance-dependent (§D4) | latest main run | dashboard-build | value-preserving (≈1e-6), op-specific |
| **cross-device (n>1 vs n=1)** | sharding bugs (gather drops/dupes a voxel, band off-by-one) | **fully** — same commit/build, only the mesh differs | none (within-run) | dashboard-build | tight (accumulation-noise only) |
| **cross-platform (CPU vs GPU)** | backend-specific correctness (wrong on one platform only) | partial (different backends) | the other platform, same commit | dashboard-build | loose, calibrated |

Notes:
- **prior-run is demoted, not removed:** it stays a sensitive tripwire for *changes*, but is reported
  under the correctness tier at a lower weight ("something shifted") — it is the tight, branch-local
  signal that the looser vs-main can't provide.
- **cross-device** is the highest-value check for the sharding project and answers Greg's robustness
  worry directly: n=1 and n=2/4 are the *same build*, so any difference beyond accumulation order **is**
  a sharding bug — zero "different implementation" ambiguity.  Covers **all four** geometries — all are
  sharded now (translation/multiaxis ported 2026-06).  The harness sweeps n>1 automatically wherever
  `_supports_sharding()=True` (`geom_device_counts`), so the check has data for every geometry on a
  post-port checkout; on a pre-port reference (current `main`) those cells are simply n=1-only and the
  check is inactive there until the port lands on that line.
- **cross-platform** needs a small cell present on *both* platforms (§D7).

### D3 — Where each check lives

- **Engine (`performance_tracking.py`)**: keep the prior-run fingerprint gate; only *tag* its findings
  as correctness-category so the dashboard can separate them.  No new cross-corpus logic in the engine
  (a per-platform worktree doesn't have main's / the other platform's results to hand).
- **Dashboard build (`build_dashboard.py`)**: a new **correctness analyzer** that, over the full set of
  loaded run docs + the ack file (§D6), computes vs-main, cross-device, and cross-platform divergences
  from the stored fingerprints and emits only the *derived findings* into the JSON (not raw
  fingerprints).  This is also what the nightly rebuild scans to drive the alert (§D5) — one computation
  feeds banner, markers, tile badges, and the alert text.

Rationale: corpus-level comparisons need all runs at once, which is exactly what the dashboard builder
already has; the engine stays simple.

### D4 — Tolerances & calibration

The fingerprint is robust to *accumulation order* but not to *different float32 results*: two correct
implementations can differ ~1e-5–1e-4 relative (this is the `cone|direct_filter` **4.5e-5** seen on the
`987a2ad8` prerelease run — a legitimate sharded-vs-monolithic reorder, not a bug).  So tolerance is the
whole game for vs-main:

- **The sharding port is value-preserving by design** (commit messages claim ~7e-8 forward, bit-identical
  back).  Under that contract a **tight** vs-main tol (~1e-6) is justified and won't false-fire on
  legitimate value-preserving reimplementations — and the 4.5e-5 cone-filter case becomes something to
  *review*, which is the point.
- **Calibrate empirically, don't guess** (Greg's standing principle): the cross-device spread (same math,
  different reduction order) is the implementation-noise floor — set vs-main a few× above the observed
  floor, per op.  Real correctness bugs are typically gross (1e-2–O(1)), so there's a wide safe band.
- **Op-specific & reference-specific:** single-shot tight (1e-6), VCD looser (it's seed/iteration
  dependent), cross-platform loosest (CPU vs GPU float differ more).  All live in `Config` so they're
  swept, not hard-coded.

### D5 — Visibility: the dashboard IS the alert (answering "how would I see a push alert?")

Greg doesn't read stdout/logs and won't; email adds infra layers.  So the channel is the dashboard he
already opens — made impossible to miss, even passively:

1. **Sticky red banner at the top of the page**: "⚠ N unacknowledged correctness divergence(s) since
   <earliest date>", expandable to a list of `(branch · platform · cell · reference · magnitude ·
   first-seen date)`, each a link that jumps the dashboard to that run.  Persists across visits until
   acknowledged (§D6) or auto-resolved.  This is the inbox.
2. **Browser-tab signal (pure client-side, no infra)**: when the unacknowledged count > 0, set
   `document.title` to `⚠(N) mbirjax metrics` and swap the favicon for a red-badged variant.  A pinned
   or bookmarked tab then shows the alert **without opening or scrolling** — the closest thing to a push
   notification with zero new plumbing.
3. **Per-run surfacing**: a distinct correctness marker in History (a red ✕, larger than the tests-failed
   triangle) on any divergent run; a red "INCORRECT vs main" badge + tint on the run-shown tile, distinct
   from the amber perf tint; correctness called out separately in the run tooltip.
4. **Nightly summary line**: `run_one_night` already rebuilds the dashboard; the rebuild's correctness
   scan appends a `CORRECTNESS ALERT` block at the **end** of the nightly output (Greg sees it whenever he
   runs `run_one_night`).  **No GitHub issue** (clutter), **no email** (infra layers).

The banner + tab badge (1 & 2) are the load-bearing answer; 3 is the detail view; 4 is a bonus for the
hand-run nightly.

### D6 — Clearing the alert: a single "reviewed-through" watermark + a guided script

A vs-main / cross-platform divergence fires *every run* while it persists — good for not-missing-it, but
it must be clearable once reviewed, or it's standing noise.  Two clearing paths:

- **Auto-resolution** (no action): a divergence disappears when the branch's value returns to main's, or
  the branch merges / is deleted (its runs leave the corpus).

- **A single date watermark** — the simple design Greg leaned toward.  One field in a committed file:

  ```yaml
  # results/correctness_acks.yaml
  cleared_through: 2026-06-21   # every divergence on a commit dated <= this is "reviewed/accepted"
  ```

  Semantics: a divergence is acknowledged iff the run's **commit date ≤ cleared_through**.  Acknowledged
  divergences are shown **greyed** (audit trail intact) and excluded from the banner / tab count.  There
  is **no list and no composition** — a new clear simply **overrides** the date (monotonic forward by
  default).  This is the "I've reviewed everything through Friday, start fresh" button, and nothing else.

- **The guided script** (`action_scripts/clear_correctness.sh` — the dashboard banner points here; it
  wraps a small Python helper, run from a terminal): prints the current watermark and the divergences
  that *would* be cleared, then offers **"clear through today? [Y/n]"** —
  **defaulting to a blanket clear through the current date**.  (It will also accept an explicit earlier
  date if asked, but today is the default and the only thing it suggests.)  Writes `cleared_through`,
  ready to commit from PyCharm.  No finicky data entry — it's one confirmation.

**Known limitation (accepted for v1).**  A single watermark means clearing through today accepts
*everything* through today — you can't keep one known-and-accepted divergence suppressed while still being
alerted to a *different* one at the same date.  A persistent accepted divergence therefore re-alerts on
each new commit after the watermark, nudging you to either fix it or move the watermark forward.  If that
proves annoying, the deferred extension is per-`(branch, platform, cell-glob)` acks with their own
`through:` dates — explicitly **out of scope for v1** (Greg leans simple; §4.4 confirms no per-commit
granularity).  Composition is the reason to keep v1 a single value: with one watermark there is nothing to
compose, and "newest clear wins" is unambiguous.

### D7 — Data / coverage changes

- **Shared CPU/GPU cell per geometry** for the cross-platform check: today `geom_sizes` differ by
  platform (zero overlap — verified by dry-run).  **DONE (2026-06-21):** the §4.3 "smallest GPU size →
  CPU" pick would put cone 512³ on CPU (heavy nightly add), so we flipped to **largest CPU size → GPU**
  (Option B): the first GPU entry of each geometry now mirrors the largest CPU size — cone/parallel
  `200×208×160`, multiaxis `129×113×97`, translation `15×65×65`.  CPU unchanged; GPU adds a few tiny
  cells.  Same correctness coverage (fingerprint agreement is size-insensitive), near-zero cost.  The
  shared cells appear on the next nightly and the check activates automatically (all measured n, not just
  n=1 — the fingerprint is device-count-independent).
- **Fingerprints at build time**: already available (`_parse_run` loads the full doc).  The analyzer reads
  `doc["cells"][i]["fingerprint"]`; only derived findings go into the JSON.  (If we later want the raw
  numbers in the UI, emit a 2-field slim `{l2norm, sum}` — decide in implementation.)

---

## 4. Decisions (settled with Greg 2026-06-21)

1. **vs-main tolerance** — ship the value-preserving default (1e-6 single-shot / 1e-4 VCD); the nightly
   prints the observed cross-device spread so we tune from data later.  *(Greg: "go with your recommendation.")*
2. **Reference branch = `main`** (literally), not prerelease — untracked branches can be pulled into
   prerelease before they're fully vetted, so prerelease isn't a trustworthy correctness reference; `main`
   is.  *(Implementation note: still expose it as a single `Config`/builder constant so it's swappable.)*
3. **Cross-platform shared cell** = ~~smallest GPU size → CPU~~ → **flipped to largest CPU size → GPU**
   after the P4 dry-run showed the original puts cone 512³ on CPU (heavy nightly add); Option B has the
   same coverage at near-zero cost.  See D7.
4. **No per-commit acks** — the single `cleared_through` watermark (D6) is the whole ack model for v1.
5. **Banner + tab badge only** — no GitHub issue, no email; a `CORRECTNESS ALERT` line at the **end** of
   `run_one_night` (D5.4).

---

## 5. Phased implementation plan

- **P1 — Severity split + dashboard surfacing (no baseline change). ✅ DONE (2026-06-21).**  Classify
  correctness vs perf in `build_dashboard`; History ✕ marker, red tile badge/tint, sticky banner,
  tab-title/favicon badge.  Shipped.
- **P2 — Correctness analyzer: cross-device + vs-main. ✅ DONE (2026-06-21).**  Corpus dry-run first:
  cross-device noise floor ~1e-6 (clean, value-preserving), vs-main meaningful reorders at 4–8e-5 vs
  ~1e-6 drift → tolerances **1e-5 single / 1e-4 VCD / 1e-5 cross-device**.  Build-time analyzer with the
  degenerate-reference guard; unified `correctness` findings (prior + cross-device + vs-main) wired into
  the banner/markers/drill-down.  Reference branch = `main`.  *(Remaining P2 tail: emit the cross-device
  floor in the nightly — folded into P5.)*
- **P3 — Acknowledge-through-date. ✅ DONE (2026-06-21).**  Single `cleared_through` watermark
  (`results/correctness_acks.yaml`), folded in by `build_dashboard` (acked = greyed, dropped from
  banner/badge, audit retained); guided `action_scripts/clear_correctness.sh` (+ `clear_correctness.py`,
  reusing `collect_data`) defaulting to "clear through today? [Y/n]", with `--status` and explicit-date.
- **P4 — Cross-platform reference. ✅ DONE (2026-06-21, activates on next nightly).**  Analyzer matches
  CPU↔GPU runs by (branch, commit) and compares shared cells (symmetric — both sides flagged), guarded
  against degenerate baselines.  Shared cell = largest CPU size → GPU (D7).  Tolerance **1e-3 single /
  3e-3 VCD — SAFE-LOOSE**, to calibrate from the first shared-cell data (re-run the cross-platform
  dry-run and tighten).
- **P5 — CORRECTNESS ALERT block + cross-device-floor emission. ✅ DONE (2026-06-21).**  Both print at
  the end of every dashboard build (`build_dashboard._correctness_summary`) — i.e. "the rebuild's
  correctness scan" (D5.4).  *(Note: it lives in `build_dashboard`, not `run_one_night`: the nightly
  forwards to `run_regression.sh` which measures + pushes but does NOT build the dashboard; the build —
  where the corpus analysis runs — is the right home, and is what Greg runs to view results.)*  The ALERT
  lists the unacknowledged divergences (same set as the banner, latest-per-branch), or a clean
  "cleared through <date>" line; the floor reports the max cross-device reldiff for tuning.

Each phase is independently shippable and leaves the dashboard correct.  **State: P1–P5 shipped in
`mbirjax_metrics` (P4's cross-platform check activates once the shared cells land on the next nightly).**

---

## 6. Summary

Make correctness its own loud category (D1, D5); stop the ratchet by adding stable references that fire
every run while divergent — **vs-main** for the canonical target and **cross-device** for sharding,
which is fully implementation-robust (D2–D4); keep the alert on the dashboard Greg already checks, with a
passively-visible tab badge instead of email (D5); and let reviewed divergences be cleared through a date
so persistence stays signal (D6).  The adjoint property is intentionally left to the existing test suite.
