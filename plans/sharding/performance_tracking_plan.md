# Performance-tracking / nightly-regression tool — design plan

**STATUS: BUILT + CPU end-to-end verified; GPU cluster bring-up in progress (2026-06-16).**  See the
`sharding_status.md` **HANDOFF (2026-06-16b)** for the current, authoritative P5 state — the nightly
wrapper (§12, `dev_scripts/regression/`) is now built and deployed to the `mbirjax_metrics` repo
(`tooling/`), CPU verified end-to-end (a real run pushed + pulled), with cluster (GPU) bring-up
underway.  As-built deviations from the original §12/§13 sketch: harness LIVES IN the metrics repo
(deploy via `deploy_to_metrics.sh`); two-phase wrapper with a PERSISTENT `$WORK_DIR/metrics` clone
(`fetch`+`pull --rebase`, not throwaway — failed pushes self-heal); FIRE-ON-CHANGE (ls-remote vs
`state/`, not unconditional nightly); per-branch SHALLOW single-branch library clone (not full clone +
worktree); `lib_root`/`golden_dir`/`REG_GOLDEN_DIR`/`REG_SMOKE` engine knobs; node `PREAMBLE_FILE` +
`TOKEN_FILE` for the cluster; `TRACKED_BRANCHES=("greg/conebeam_sharding")` only (main/prerelease
deferred — unported → degenerate multi-device sweep).  The engine, fingerprint, `vcd_nonconst`, record
book, diff/gate, golden/main-baseline capture, and manual launcher are in
`plans/experiments/sharding/scaling_tests/` (usage: that dir's `README.md`).  This doc is the design
rationale; where as-built differs it is flagged inline (e.g. §10 gate-model, §6 no-auto-skip).
**Still NOT built:** GPU `nightly_regression.slurm` + `scrontab` (§14 P6); `enable_nightly.sh` schedule
not yet activated; deferred `compare_to_baseline.py` (.npy deep-diff, §14 P7) + visual surface (§14).

Author intent (Greg, this session): a standing day-over-day check that exercises every
geometry × op and flags time/memory/correctness drift, runnable as a nightly cron (CPU =
Greg's Mac) and a nightly slurm batch (GPU = cluster), **plus** an ad-hoc manual launcher
for the current working tree that never clobbers the nightly results.

**Built-this-session notes not yet woven into the body below:**
- **No OOM auto-skip** (`vcd_min_devices` removed): an OOM is recorded as a failure cell;
  `run_measure_loop`'s descent already stops at the first OOM (smaller counts need more per-device
  memory).  Supersedes §6's `vcd_min_devices` skip.
- **`main` 1-device baseline now captures time + memory + fingerprint** at the full sweep sizes
  (`main_baseline_<plat>.yaml`), not just the `.npy`; the engine **auto-discovers** it and prints a
  SOFT "vs main (1 device)" relative-perf note (never gated).  Extends §8.
- **VCD reproducibility fix:** `gen_pixel_partition` draws from the un-seeded global RNG (mbirjax,
  both branches), so `build_partitions` now seeds (`measure_seed`) — else the day-over-day VCD
  fingerprint would false-positive.  A docstring note on `recon`/`prox_map` documents this for
  library users.  (How this was found: the cross-version baseline flagged a 5e-2 "parallel VCD vs
  main" delta that turned out to be the un-seeded partitions, not a code difference.)

---

## 0. Why / what it must catch

Almost every "regression" chased in the 2026-06-13/14 measurement sessions was a **ruler
bug** or a quietly-introduced perf change found only by accident much later (the 4b21a3c2
gather-at-exit; the unbounded FDK filter; the n=1 forward per-dispatch memory).  A
day-over-day check that exercises geometry × op × device-count and flags drift would have
caught all three at the commit that introduced them.  The tool's job is to be the standing
ruler — and to NOT itself become a noisy ruler (hence the hard/soft gate taxonomy in §10).

Three independent regression surfaces, each with its own gate:
1. **Unit correctness** — the full pytest suite (`run_tests.sh`).
2. **Numeric correctness of the big ops** — a tolerant fingerprint of `direct_filter`,
   `sparse_forward_project`, `sparse_back_project`, and a short nonconstant-weight
   `vcd_recon`, compared day-over-day and vs a committed golden anchor.
3. **Performance** — `min_ms`, `peak` memory, speedup-ratio, and structural flags, same
   comparison targets.

---

## 1. Architecture — three layers, clean separation of concerns

```
 LAYER 1 — shell wrapper (env + orchestration)
   dev_scripts/regression/nightly_regression.sh   (cron on CPU / called by .slurm on GPU)
     1. resolve TODAY once (passed down via --date; never datetime.now() in a worker)
     2. for BRANCH in $BRANCHES:          # single list = the one place to change tracks;
          git worktree add <tmp> $BRANCH  # list two branches → two-track nightly
     3. install INTO the worktree:  weekly → clean_install_all.sh ; else pip install -e . (fast)
     4. run_tests.sh → capture summary + failure list → tests_<date>.yaml
     5. python <worktree>/plans/experiments/sharding/scaling_tests/performance_tracking.py \
            --out-dir <STABLE> --date TODAY --tests-result tests_<date>.yaml
     6. exit code → notify (cron MAILTO / slurm --mail-user)
     7. git worktree remove <tmp>
                                  │ invokes
                                  ▼
 LAYER 2 — the measurement ENGINE  (stays in scaling_tests/, beside scaling_common.py)
   plans/experiments/sharding/scaling_tests/performance_tracking.py
     • run(config): sweep GEOMETRY × OP × size × n_dev over the EXISTING scaling_common
       harness (isolated subprocess per (geom,op,size) by default; --inline for a
       debuggable single process)
     • per cell: min_ms, mem_mb, speedup, structural fields, correctness fingerprint
     • write <out-dir>/regression_<plat>_<YYYYMMDD>.yaml
     • diff vs most-recent-prior (+ committed golden) → hard/soft gate → exit code
                                  ▲ imports
                                  │
 LAYER 3 — manual launcher (current working tree)
   plans/experiments/sharding/scaling_tests/run_performance_local.py
     • params at the TOP (subset of geometries/ops/sizes/counts; INLINE; RUN_TAG)
     • forces out-dir = results/manual/<tag>/ and compare=off (or a chosen baseline)
       so it NEVER overwrites or gates against the nightly time series
```

**Why the split (load-bearing, not stylistic):** `clean_install_all.sh` runs
`conda remove --name mbirjax --all` and recreates the env (`dev_scripts/clean_install_all.sh:35`).
A python orchestrator cannot drive that — it would be deleting the conda env that owns its
own interpreter, then spawning workers via `sys.executable` into a just-recreated prefix.
So **install + tests are a shell concern that must finish before any python in the target
env starts**; python owns only measurement + diff/gate.

**File-location split (per the wrapper-home decision, §13):** the operational entrypoints
(`nightly_regression.sh`, `nightly_regression.slurm`) live in `dev_scripts/regression/`,
beside the `clean_install_all.sh` / `run_tests.sh` they call.  The python measurement code
(`performance_tracking.py`, `run_performance_local.py`, the golden-capture mode) stays in
`plans/experiments/sharding/scaling_tests/` because it is tightly coupled to `scaling_common.py`:
both `import scaling_common` resolution (the script's own dir on `sys.path[0]`) and the
`beta_root()` path derivation (`scaling_common.py:222` — three dirs up from scaling_tests/)
depend on living there.  Moving the python into dev_scripts/ would force extra `sys.path`
plumbing for no benefit.

---

## 2. Entry points and output isolation

| Entry | Runs from | Code under test | Install | Out dir | Compare / gate |
|---|---|---|---|---|---|
| **Nightly (CPU)** `dev_scripts/regression/nightly_regression.sh` via cron | temp worktree per branch in `$BRANCHES` | pinned `$BRANCHES[i]` HEAD | fast nightly / clean weekly | stable `~/mbirjax_regression/cpu/` | day-over-day + golden; exit code gates |
| **Nightly (GPU)** `dev_scripts/regression/nightly_regression.slurm` → same wrapper | temp worktree per branch | pinned `$BRANCHES[i]` HEAD | clean (cluster) | stable `<scratch>/mbirjax_regression/gpu/` | same |
| **Manual** `run_performance_local.py` | current repo (live tree) | **your in-progress edits** | none (uses live editable install) | `results/manual/<RUN_TAG>/` | off by default (or point at a chosen baseline) |

The manual launcher's whole point is to measure *uncommitted work in the current tree*, so
it deliberately does NOT use a worktree and does NOT touch the nightly out-dir or the gate
exit code.  It is the debugger-friendly path (set `INLINE = True`).

---

## 3. The worktree + install + out-dir wiring (the two couplings to get right)

Running the canonical nightly against a clean, pinned checkout gives reproducibility and
honest provenance (record `git rev-parse HEAD`; assert `git status --porcelain` is empty).
It composes with the existing path machinery for free: `build_worker_env` forces
`PYTHONPATH=beta_root()`, and `beta_root()` is derived from `scaling_common.py`'s own
location (`scaling_common.py:222`, `:233`).  So if the wrapper invokes the **copy of
`performance_tracking.py` inside the worktree**, every worker subprocess resolves
`import mbirjax` to that worktree automatically.

Two couplings the worktree creates — both easy to miss:

1. **Install from inside the worktree.**  `clean_install_all.sh` does `pip install -e ..`
   (editable, `dev_scripts/clean_install_all.sh:69`), which points the install at the
   directory it is run from.  The fast nightly `pip install -e .` must likewise run from the
   worktree root, or the editable install and `PYTHONPATH` will disagree about which checkout
   is under test.
2. **Results + golden live OUTSIDE the throwaway worktree.**  `RESULTS_DIR` is script-relative
   (`os.path.join(_HERE, "results")`, `scaling_common.py:60`).  If the dated YAMLs landed in
   the temp worktree they would vanish with it, and day-over-day could never see yesterday.
   So `performance_tracking.py` takes an explicit `--out-dir` pointing at a **stable** location;
   it reads the prior-day file from there too.  The golden fingerprint comes from the committed
   repo (re-checked-out fresh each night), so that one is fine.

---

## 4. Config model — one engine, many configs

`performance_tracking.py` is driven by a single `Config` (lightweight dataclass, serializes
to YAML).  Defaults encode the **nightly** sweep; launchers override a subset.

Fields (initial):

```
Config:
  # sweep
  geometries:    ["parallel", "cone"]          # add translation/multiaxis as they port
  ops:           ["direct_filter", "forward", "back", "vcd_nonconst"]
  device_counts: [1, 2, 4]
  # sizes (n_views, n_rows, n_channels) — ASYMMETRIC (all three differ) to surface axis
  # swaps/transposes that cubic NxNxN sizes hide; one DIVIDING + one NON-DIVIDING per platform
  # to exercise the view- AND slice-PADDING paths (all-odd → non-divisible by 2 and 4 on every
  # axis); plus a GPU 1024-class capacity size where some bugs only appeared (§6).
  sizes:         {"cpu": [(128,112,96),   (129,113,97)],
                  "gpu": [(512,448,384),  (513,449,385),  (1024,1008,992)]}
  single_trial_sizes: ["1024x1008x992"]    # all ops trials=1 here: capacity/memory + does-it-run
                                            # check, NOT a timing ruler → time is informational
  vcd_min_devices:    {"1024x1008x992": 2}  # carry cone_baseline's OOM-skip (1024-class VCD OOMs
                                            # one 80 GB GPU; let it OOM once = the capacity result)
  # vcd
  vcd_iterations: 3
  weight_mode:    "nonconstant"   # deterministic pattern (see §7); "constant" allowed
  weight_seed:    13
  # measurement
  warmup: 1
  trials_by_op: {forward:3, back:3, direct_filter:3, vcd_nonconst:1}   # overridden→1 on single_trial_sizes
  inline: false                   # true = single-process, debuggable (memory not per-config)
  # io / provenance
  out_dir:        <required>      # stable nightly dir, or results/manual/<tag>
  date:           <required>      # stamped by the orchestrator, never in a worker
  run_tag:        ""
  git_commit:     <auto rev-parse HEAD>
  git_dirty:      <auto porcelain>   # asserted empty for nightly; informational for manual
  # comparison / gate
  compare_to_prior: true          # day-over-day
  golden_path:      <committed golden_fingerprint_<plat>.yaml | None>
  gate:             true          # set exit code; manual runs default false
  thresholds:       {see §10}
  correctness_pct_tolerance: 0.01 # % of points allowed beyond tol (lax.map/scatter bug; §7)
  k_sample_tol:     1             # how many of the K fingerprint samples may deviate (§7)
  tests_result:     <path | None> # tests_<date>.yaml ingested into the unified report
```

**Config → worker:** this is the one real change from the `cone_baseline_scaling.py`
pattern.  cone_baseline lets the worker read module-level constants (GEOMETRY, SIZES) because
it is a single fixed config; here geometry/sizes are sweep dimensions and may be overridden by
a launcher, so module globals would not reach the fresh worker process.  Instead the
orchestrator **writes the resolved Config to a temp YAML and passes `--config <path>`** to each
worker, with the sweep coordinates on argv for readability:
`--worker --mode measure --config <tmp.yaml> --geometry cone --op back --size 256x256x256`.
Everything else (sizes for *other* ops, vcd params, thresholds) the worker reads from the
config file.  This is strictly more robust than implicit globals and is what lets one engine
serve nightly + manual.

---

## 5. Inline / debug mode (addresses the run_worker reservation)

Default = isolated-subprocess (load-bearing for the memory ruler: `peak_bytes_in_use` is a
cumulative high-water mark, so one fresh process per `(geom,op,size)` makes the peak equal
*that config's* allocation, and the orchestrator stays JAX-free so it never holds a backend
while a worker measures — `scaling_common.py:46`, `run_measure_loop` docstring).

`inline=True` calls the worker body (`worker_measure(...)` → `build_and_time(n, devs)`, already
plain functions) **directly in the orchestrator process** — no subprocess hop, fully
step-through-able in PyCharm.  The one cost: **memory in inline mode is cumulative across the
sweep, not per-config**, so inline is for debugging logic/correctness; trust the
isolated-subprocess numbers for the actual memory ruler.  The manual launcher exposes `INLINE`
prominently; the nightly always runs `inline=False`.

---

## 6. Measurement — reuse the harness, do not rebuild it

The engine is thin over `scaling_common.py` + the proven structure of
`cone_baseline_scaling.py`, inheriting every ruler fix already landed: per-count device pinning
(`configure_devices`), on-device input pre-placement (`to_device`, "measure the op not the
scatter"), `output_sharded=True` for the filter (avoid the gather-at-exit trap), skip-OOM
descent, throttle sampling, the path/band YAML fields, GPU topology + dev2dev probe.

Changes vs cone_baseline:
- **GEOMETRY becomes a sweep dimension** (outer loop), not a module constant.
- **`vcd_const` → `vcd_nonconst`** (§7).
- Add the **correctness fingerprint** capture (§7) alongside timing/memory in each worker.
- Add the **diff + gate** stage in the orchestrator (§10).
- **Asymmetric sizes** (all three sino dims differ) so a transpose / axis-swap regression
  shows up as a fingerprint mismatch — a cubic NxNxN size silently hides it.
- **Padding sizes** (the all-odd non-dividing size) exercise the view- and slice-padding code at
  n=2/4.  **Cone padding is B5 (not implemented), and the engine does NOT skip it — it lets the
  op run and records ground truth** (Greg's call: validate the harness's failure-capture path +
  track the B5 fix as a datable failure→success transition).  Measured this session at a padded
  count: cone **`forward` RAISES** (`run_measure_loop` captures it as a failure cell and continues
  the descent, since it is not OOM), while **`back` and `direct_filter` tolerate padding** and
  return the padded DEVICE form (49→50 views, 41→42 slices) — real measurements, not garbage.  **No
  allowlist** — the gate fires only on a CHANGE in cell status vs the prior day / golden (§10a), so
  a persistent cone-padding failure is a visible wart that does not alarm, and the B5 fix surfaces
  as a fail→ok improvement.  Parallel padding (P5) is fully exercised now.
- **1024-class GPU capacity size** with `trials=1` for every op (some bugs — the gather-at-exit,
  the n=1 forward memory — only manifested at 1024³).  At single-trial sizes **time is recorded
  but not gated** (1 sample is not a timing ruler); memory, structural flags, and the
  correctness fingerprint ARE still gated.  `vcd_min_devices` skips the 1-GPU VCD that is known
  to OOM (the OOM itself is the capacity result — let it happen once, then raise the floor).

Bounded for "a few minutes": the everyday cross-product is {2 geom} × {4 op} × {≤3 counts} ×
{2 small sizes} plus the single GPU 1024-class capacity sweep (trials=1).  Measure one dry-run
wall-clock before committing the nightly cadence; if the 1024-class size pushes past budget,
make it a *weekly* extra rather than nightly.

---

## 7. Correctness fingerprint (tolerant, not a hash)

A "float-aware hash" can't be a real hash (a 1-ULP change flips it, colliding with the
project's never-exact-equality-for-computed-floats rule).  So per (geometry, op, size, n_dev)
store a small **tolerant fingerprint** of the op's output:

```
fingerprint:
  sum:    <float>
  mean:   <float>
  l2norm: <float>
  min:    <float>
  max:    <float>
  samples: [<value at K fixed deterministic flat indices>]   # K ~ 8–16
  shape:  [...]            # the TRUE recon/sino shape (NOT the padded device form); exact-compared
  dtype:  "float32"        # structural: exact-compared
  padding_zero: true|null  # see "padded outputs" below; null when not padded
```

- **The fingerprint is always on the TRUE shape, never the padded device form** (Greg's call).
  At a non-dividing count `back`/`direct_filter` return the padded device form (e.g. 49→50 views,
  41→42 slices).  Before fingerprinting, the worker (a) **asserts the padded region is exactly 0**
  — `padding_zero` — which is the *allowed* exact check (constructed-zero invariant, per the
  correctness rules) and is itself a real correctness gate: a non-zero pad is a B5 padding-leak
  wart surfaced in the open; then (b) **crops to the true shape** (`recon_shape` / sinogram shape
  from the 1-device base model) and fingerprints the cropped array.  Result: the fingerprint is
  comparable across device counts, across B5, and vs golden — no shape disagreement.  (Cone
  `forward` never reaches this — it RAISES at a padded count and is a failure cell.)

- **Compared with tolerance, and a small fraction of out-of-tolerance points is ALLOWED.**  The
  known jax `lax.map` + scatter rounding bug leaves a handful of points off by ~fp32 ULPs even on
  a correct run, so the gate must tolerate them (the policy `scaling_common.correctness_metrics`
  already encodes via `pct_above_threshold`):
    - **Robust aggregates** {`sum`, `mean`, `l2norm`}: relative-`allclose` at ~1e-5 single-shot
      (`direct_filter`/`forward`/`back`), ~1e-4 iterated (`vcd_nonconst`; measured GPU run-to-run
      noise ~8e-6 rel, scatter-add atomics).  A few bad points among millions barely move a
      sum/mean/norm, so these are the **primary** correctness signal.
    - **Vulnerable extremes** {`min`, `max`, `samples`}: a *single* rounding-bug point can trip
      one, so they are recorded and **soft-flagged**, not hard-gated alone (allow up to
      `k_sample_tol` of the K sampled points to deviate).
    - **The `.npy` deep-diff** (break-glass, §8) uses `correctness_metrics`: HARD only if
      `pct_above_threshold > correctness_pct_tolerance` (config, ~0.01% default) — a few stray
      points pass; a systematic shift fails.
  `shape`/`dtype` are exact-compared (data-movement identities — those never tolerate drift).
- Tiny (a few floats per cell) → lives readably in the dated YAML, diffs cleanly, travels in
  git.  The per-cell fingerprint covers the **whole sweep** (it is cheap).  The full `.npy`
  deep-diff array is a different, heavier artifact captured at **one representative small size
  per (geometry, op)** only (Greg's note: not all sizes) — see §8.  A small array is sufficient
  for diagnosis because the numeric differences come from geometry/pixel patterns, not array
  scale (the same reasoning the existing `*_capture_baseline.py` scripts rely on).

**`vcd_nonconst`:** a deterministic nonconstant weight pattern (e.g. seeded-random or a smooth
radial/row ramp, `weight_seed`) so it is reproducible AND exercises the weighted
gradient/Hessian path that all-ones weights skip.  3 iterations: enough to be a correctness/
integration anchor, not a scaling ruler (the §8a ruler note: few iters under-amortize fixed
per-recon overhead — so VCD time is a SOFT signal only).

---

## 8. Baselines & transport — a dedicated output repo (the webserver is retired)

**Two correctness references, different jobs.**  (1) The **fingerprint golden**
(`golden_<plat>.yaml`, via `capture_golden.py`) is captured on the CURRENT branch and is the
day-to-day drift/accept reference the gate compares against.  (2) The **`.npy` deep-diff baseline**
(via `capture_main_baseline.py`) is captured on the **`main`** branch — a *cross-version* anchor
that answers "does the sharding branch still reproduce released `main` within tolerance?" (a few
lax.map/scatter outliers allowed).  `main` has no sharding, so the `.npy` are single-device at one
small representative size; the capture reuses the engine's input builders + run functions (they
import mbirjax lazily, so a `main`-worktree run uses `main`'s API).  The deferred
`compare_to_baseline.py` re-runs the sharding branch at that size and tolerant-compares.

**Why not the mbirjax repo (Greg's note 5):** every mbirjax branch except `main` is periodically
deleted, and we do not want to push results to `main`.  So results/golden have no stable home in
mbirjax.  **Decision (confirmed): a dedicated git repo — `mbirjax_metrics`,
https://github.com/gbuzzard/mbirjax_metrics (public)** — cloned once at a stable path on each
node.  It is the shared channel for everything the nightly reads/writes:

```
mbirjax_metrics/                    (cloned on the Mac AND the cluster)
  golden/
    golden_fingerprint_<plat>.yaml  # per-cell tolerant fingerprint + perf anchor (a few KB);
                                     # ALSO the de-facto "expected state" — a failure recorded
                                     # here is a known wart (quiet); see §10a (no separate allowlist)
    <geometry>_<op>.npy             # cross-version correctness reference: ONE small single-device
                                     # array per (geom,op), captured from the MAIN branch via
                                     # capture_main_baseline.py (Greg note 3 — tolerant deep diff)
    main_baseline_meta.yaml         # the size/seeds/branch the .npy were captured at
  results/
    cpu/regression_cpu_<YYYYMMDD>.yaml   # the time series, one file per nightly
    gpu/regression_gpu_<YYYYMMDD>.yaml
```

This resolves the earlier samba/webserver awkwardness:
- **Day-over-day needs no network** — today's dated YAML vs *yesterday's*, both already in the
  local clone on that node.  Primary nightly gate.
- **Golden anchor travels in git**, not the mbirjax checkout — survives mbirjax branch churn,
  no push to mbirjax `main`.  Catches slow/persistent drift and the "first nightly is already
  wrong" case.
- **The `.npy` deep-diff array is now SMALL** (one representative size per op, note 3) → it fits
  in the git repo directly.  **So the webserver is no longer needed** — the deep-diff array
  ships with the clone; the on-demand `compare_to_baseline.py` reads it locally.  (A small array
  is sufficient for diagnosis — the numeric difference is geometry/pixel-pattern driven, not
  scale driven; §7.)

**Push mechanics (both nodes can reach the web — confirmed):**
- **Mac:** the nightly does `git -C $REPO pull --rebase` at start (fresh golden) and
  `git add results/cpu/<file> && git commit && git push` at end.  CPU and GPU write to *different
  subdirs* (`results/cpu/` vs `results/gpu/`) so pushes never conflict.
- **Cluster:** the compute nodes **do** have outbound web access, so the batch job pushes
  **inline**, exactly like the Mac — no separate login-node step.

**Credentials (the real wrinkle — unattended push):**
- **Mac:** Greg's periodically-refreshed token in the macOS keychain is visible to the `launchd`
  job (it runs as the same user), so push "just works" until the token expires.
- **Cluster:** PyCharm prompts for the token each login and forgets it next session — useless for
  an unattended job.  Fix: a **dedicated fine-grained PAT with write access to `mbirjax_metrics`
  only**, persisted on the cluster in a chmod-600 file referenced from `regression.env` (path,
  not the token itself — the secret is never committed).  Cleanest wiring is a per-clone
  credential, e.g. `git -C $REPO config credential.helper "store --file=$TOKEN_FILE"`, scoped to
  that one repo so it cannot leak into other git use.
- **Push failure is NON-FATAL.**  An expired/rotated token must never break the measurement: the
  dated YAML is always written to the local clone first; a failed `pull`/`push` is logged as a
  **WARN** (not a HARD gate) and the unpushed file syncs on the next successful run.  The gate
  exit code reflects regressions, not git-transport hiccups.

So: **nightly = day-over-day (local clone) + golden (git), zero web fetch on the read path.**
The dedicated repo is the single source of truth for golden + the result history; no webserver,
no samba dependency.

---

## 9. Dated YAML schema (one file per nightly run = the time series)

`<out-dir>/regression_<plat>_<YYYYMMDD>.yaml`:

```
kind: regression
date: 2026-06-15
platform: cpu|gpu
git_commit: <sha>
git_branch: <branch>
git_dirty: false
device_label, topology, dev2dev_safe, mbirjax_path   # from scaling_common setup
config: { ...resolved Config sans io paths... }
tests:                       # ingested from tests_<date>.yaml (nightly only)
  passed: int
  failed: int
  failures: [nodeids...]     # the full failing set (the 4 B5 cone tests at 4 dev live here)
  new_failures: [...]        # gate output: today's failures MINUS the prior day's (no allowlist)
cells:
  # a measured cell:
  - {geometry, op, size, n_dev, min_ms, mean_ms, mem_mb, mem_kind,
     speedup, is_sharded, n_shard_devices, back_n_bands_per_shard, trials,
     fingerprint: {...}}
  # a FAILED cell (op raised / OOM'd) carries instead:
  - {geometry, op, size, n_dev, failed: true, oom: bool, error: "..."}
  # a deliberately SKIPPED cell (vcd_min_devices OOM-floor) carries:
  - {geometry, op, size, n_dev, skipped: true, reason: "..."}
gate:                        # filled by the diff stage
  result: pass|warn|fail
  hard: [ ...descriptions... ]
  soft: [ ...descriptions... ]
  compared_to: regression_<plat>_<prior>.yaml
  golden: golden_fingerprint_<plat>.yaml
```

The dated files ARE the time series; a rolled-up multi-day view for drift plotting is a
derivable secondary tool (a `replot_from_yaml`-style reader), not a separate source of truth.

---

## 10. Diff & gate — memory/structural/ratio hard, absolute time soft

Carrying the plan's taxonomy forward (it is what actually regressed in the three 2026-06-14
bugs):

**Updated 2026-06-15 (as built — Greg's call): of the perf signals, only MEMORY hard-fails, and
only on GPU.**  Speedup and absolute time are SOFT everywhere — both derive from timings, which are
noisy even on GPU (especially small runs), so neither should fail the gate; memory alone is the
HARD perf gate (deterministic `peak_bytes_in_use`, and it is what catches the gather-bug class —
memory that fails to shard).  On CPU `mem_mb` is coarse whole-process RSS, so even memory is SOFT
there.  Net: **HARD everywhere = correctness + structural + status + vanished-cell; HARD on GPU only
= memory; SOFT everywhere = speedup, absolute time, CPU memory, config add/drop, improvements.**

| Signal | Gate | Threshold (initial, tunable) |
|---|---|---|
| **Correctness** fingerprint | **HARD (all platforms)** | robust aggregates {sum,mean,l2norm} fail `allclose` (§7); a small fraction of out-of-tol points is ALLOWED (lax.map/scatter bug); shape/dtype exact |
| **Structural** is_sharded flip / band-count change / OOM appears / cell expected-by-config-but-absent | **HARD (all platforms)** | any change (see §10a for the expected-vs-absent distinction) |
| **Cell status ok→fail** (passed in prior/golden, now raises/OOMs) | **HARD (all platforms)** | any (persistent fail→fail is a quiet known wart; fail→ok is a WARN "improved"; §10a) |
| **New unit-test failure** (in today's set, absent from the prior day's) | **HARD (all platforms)** | any |
| **Memory** `mem_mb` growth | **HARD on GPU / SOFT on CPU** | > +8% (GPU `peak_bytes_in_use` ~deterministic; CPU is coarse RSS) |
| **Speedup-ratio** drop | **SOFT (warn)** | > 15% — a ratio of noisy timings; noisy even on GPU |
| **Absolute time** `min_ms` | **SOFT (warn)** | > +25% — GPU run-to-run variance is ~1.9× for cone back |
| **Sweep-set change** (geometry / op / size / device-count added or dropped in `config`) | **SOFT (warn)** | any — intentional config change, not a regression (see §10a) |

- **Every reported delta shows BOTH the absolute and the percentage difference vs expected** (e.g.
  `memory 1100 MB vs 1000 MB expected (+100 MB, +10.0%)`), so a reader can judge importance — a big
  % on a tiny absolute is often noise, and vice versa.
- **Compare to:** (a) most-recent-prior dated file [sudden breaks] AND (b) committed golden
  [cumulative drift].
- **Exit code:** non-zero on any HARD fail → cron/slurm surfaces it (MAILTO / `--mail-user`).
  WARN-only → exit 0 but flagged in the report.  Manual runs default `gate=false` (exit 0).
- **CPU caveat:** on CPU `peak_memory_mb` is whole-process RSS, not per-device
  (`scaling_common.py:617`); the memory gate is sharpest on GPU (hence HARD only there).

### 10a. Sweep-set reconciliation — graceful add/drop of a geometry, op, size, or device count

The set of cells (geometry × op × size × n_dev) can legitimately differ between two YAMLs: a
new geometry ports in, an op is added, a size is retuned, a device count is dropped.  The diff
must **never hard-fail on an intentional sweep change**, but must **still hard-fail on a cell
the current run was supposed to produce and didn't** (a crash/OOM masquerading as absence).
The resolved `config:` block recorded in every YAML (§9) is what tells these apart — we
reconcile the *configured* coordinate sets, not just the produced rows:

- **Added** coordinate (in today's `config`, absent from the comparison YAML's `config`): the
  cell has no baseline yet → **WARN** ("new geometry/op/size/n_dev `X`: recorded, not gated this
  run"), record the cell, skip its per-cell gate.  Next run it has a prior and gates normally.
- **Dropped** coordinate (in the comparison YAML's `config`, absent from today's `config`):
  intentionally no longer swept → **WARN** ("dropped from sweep: `X`"), no gate.
- **Deliberate skip** (the coordinate is in today's `config`, but the engine intentionally did
  not attempt it — currently only the `vcd_min_devices` OOM-floor, which avoids an hour of
  near-OOM thrash before an inevitable OOM): **WARN**, never HARD.  The skip row carries
  `{skipped: true, reason: "..."}`.
- **Status TRANSITION on a cell present in both — NO allowlist** (Greg's call; the gate fires on a
  *change*, so a known wart needs no list to mute it).  Comparing today's cell status against the
  reference's (prior day, and golden):
    - **ok → fail**: **HARD** — a new regression (a cell that worked now raises/OOMs).
    - **fail → fail**: **quiet** — a persistent known failure (e.g. cone `forward` at a padded
      count).  The cell is `{failed: true, ...}` and fully visible in the report; it just does not
      alarm, because it is unchanged.  (Measured: only cone `forward` fails at a padded count;
      `back`/`direct_filter` succeed on the padded device form, §6.)
    - **fail → ok**: **WARN** "improved" — prompts a deliberate `--capture-golden` re-baseline so
      golden tracks the new good state.  No list to prune.
    - **ok → ok**: gate the metrics (memory / structural / speedup / correctness, §10).
- **Expected-but-absent** (the coordinate IS in today's `config`, produced **no** row AND has no
  `failed`/`skipped` record): **HARD** — a silently vanished cell (crash before measurement, or a
  sweep that stopped emitting).  "Missing" gates on *config-expected, non-skipped, non-failed* cells.

The same reconciliation applies to the **golden** comparison (golden captured under some config;
today's may differ): compare on the **intersection** of configured coordinates, WARN on the
symmetric difference, apply the status-transition rule on the intersection.  The **golden is the
single source of "what's expected"** — already maintained deliberately via `--capture-golden`, so
it doubles as the accept-list with zero extra machinery: a failure recorded in golden is known
(quiet); a failure NOT in golden nags vs golden every run until fixed or golden is consciously
re-captured.  **Cold start** (no prior, no golden): every cell is "new" → WARN, never HARD, so the
first run cannot red-alarm.  The reconciliation summary (added / dropped / transitions /
expected-absent) goes into the `gate:` block so the run self-documents why its comparison set
changed.

---

## 11. Unit-test integration — new failures vs the prior day (no allowlist)

Layer-1 runs the suite at **`MBIRJAX_NUM_CPU_DEVICES=4`** (Greg's decision — exercises the
view/slice padding paths the 2-device default never reaches), captures the summary line + the full
failing nodeid set into `tests_<date>.yaml`, and `performance_tracking.py` ingests it via
`--tests-result` so one artifact + one exit code tell the whole story.  (`run_tests.sh` itself
defaults to 2 devices via conftest `DEFAULT_MAX_CPU_DEVICES`; the wrapper exports the override
before pytest.)

**No allowlist — same transition logic as the scaling cells (§10a).**  The gate fires on
`today_failures − prior_day_failures` (a *new* failing test) → HARD.  Persistent failures are
quiet visible warts.  At 4 CPU devices the 4 B5 cone-padding failures are **CONFIRMED this session**
(the forward gather assembles the *padded* cylinder → `ValueError: voxel_values must have
shape[0:2] = (num_indices, num_slices)`):

```
tests/geometries/test_projectors.py::TestProjectors::test_adjoint_anisotropic_cone
tests/geometries/test_projectors.py::TestProjectors::test_hessian_anisotropic_cone
tests/geometries/test_vcd.py::TestVCD::test_split_sino
tests/geometries/test_vcd.py::TestVCD::test_vcd_anisotropic_cone
```

These persist day-over-day → never flagged as new (no list to keep in sync).  When B5 lands they
drop out of the failing set — a visible improvement (a *shrinking* failure set is a WARN at most,
never HARD), again with nothing to prune.  **Cold start** (no prior `tests_*.yaml`): the failing
set is reported but not gated HARD — there is nothing to diff against yet.

---

## 12. Scheduling — enable / modify / disable (Mac + cluster), with a single config point

All schedule knobs live in **one committed file**, `dev_scripts/regression/regression.env`
(sourced by every entrypoint), so there is exactly one place to edit:

```
# dev_scripts/regression/regression.env  — the single source of truth
ENABLED=1                                  # master kill-switch (0 = installed but does nothing)
BRANCHES=("greg/conebeam_sharding")        # one line = the one place to change tracks (§13)
SCHEDULE="0 2 * * *"                        # cron/scrontab spec: 02:00 nightly
WEEKLY_CLEAN_DOW=7                          # day-of-week for the full clean install (else fast)
METRICS_REPO="$HOME/mbirjax_metrics"       # clone of github.com/gbuzzard/mbirjax_metrics (§8)
OUT_DIR="$METRICS_REPO/results"            # <plat>/ subdir chosen by the wrapper
TOKEN_FILE=""                              # cluster only: path to chmod-600 file holding the
                                           # mbirjax_metrics write PAT (NOT the token; never commit)
TEST_CPU_DEVICES=4                         # MBIRJAX_NUM_CPU_DEVICES for the test step (§11)
NOTIFY="greg.buzzard@gmail.com"            # MAILTO / --mail-user
```

A README (`dev_scripts/regression/README.md`) documents the whole loop end to end; the helper
scripts make enable/disable a one-liner on each platform:

**Mac — prefer `launchd` over cron.**  A laptop is often asleep at 02:00; plain cron simply
*skips* a missed run, whereas a `launchd` `StartCalendarInterval` agent **runs at the next wake**.
So ship a `com.mbirjax.regression.plist` and two helpers:
- `enable_nightly.sh`  → `launchctl load -w ~/Library/LaunchAgents/com.mbirjax.regression.plist`
  (or, if you prefer cron, writes the `SCHEDULE` line into `crontab`).
- `disable_nightly.sh` → `launchctl unload -w …` (or removes the cron line).
- To **modify**: edit `regression.env` (time → also update the plist `StartCalendarInterval`
  via the helper, which regenerates it from `SCHEDULE`), then re-run `enable_nightly.sh`.
- `ENABLED=0` is the fast kill-switch that needs no `launchctl` — the wrapper exits immediately.

**Cluster — prefer `scrontab` (Slurm's cron) over a login-node crontab.**  Most Slurm sites
provide `scrontab -e`; an `#SCRON` line schedules `nightly_regression.slurm` natively and it
survives login-node rotation.  Helpers mirror the Mac:
- `enable_nightly.sh` (on the cluster) → installs the `#SCRON ${SCHEDULE}` entry (or a
  login-node crontab line if `scrontab` is unavailable).
- `disable_nightly.sh` → `scrontab -r`-style removal (or removes the crontab line).
- The batch job itself: module load cuda/conda, request 4 GPUs, set
  `XLA_PYTHON_CLIENT_PREALLOCATE`/`MEM_FRACTION` (the `build_worker_env` knobs), `--mail-user`
  for notification.  **Git push runs inline from the compute node** (they have outbound web —
  confirmed), authenticating via the `$TOKEN_FILE` credential helper (§8 Credentials).

**Common to both:**
- `git -C $METRICS_REPO pull --rebase` at start (fresh golden), `add/commit/push` the dated YAML
  at end (CPU→`results/cpu/`, GPU→`results/gpu/`, conflict-free).  **Push failure is non-fatal**
  (WARN; the file is already in the local clone and syncs next run — §8).
- **Date** resolved once in the wrapper and passed down (`--date`); never `datetime.now()` in a
  worker that must stay reproducible.
- Notification = cron/launchd emails stdout (`MAILTO=$NOTIFY`) / slurm `--mail-user=$NOTIFY`;
  a HARD-gate non-zero exit makes the email a real alert.

---

## 13. Resolved decisions (this review) + still to confirm

**Resolved:**
1. **Nightly branch → active integration branch, via a single `BRANCHES` list.**  One place to
   change tracks; listing two branches yields the two-track nightly for free.  Default
   `BRANCHES=("greg/conebeam_sharding")`.  The wrapper loops the list (one worktree + run per
   branch); each YAML records the resolved HEAD for its branch.
2. **Golden capture + refresh → manual trigger, fully automated capture.**  A human triggers it
   naming a branch AND commit (and, by convention, the reason in the commit message); the engine's
   `--capture-golden --branch B --commit C` mode then does everything automatically:
   `git worktree add` at that exact commit → clean install → full measurement → write
   `golden/golden_fingerprint_<plat>.yaml` + the representative `.npy` per (geom,op) + commit/push
   to `mbirjax_metrics`.  Never auto-promoted from a nightly; refreshed only by deliberate trigger.
   **Refresh modes (Greg's note F — new geometries / intentional baseline changes):**
   - **Full** (`--capture-golden`): recapture everything (use after a sweeping change).
   - **Selective** (`--capture-golden --only cone` or `--only direct_filter`): recapture just the
     named geometry/op cells, leaving the rest of the golden untouched — for an *intentional*
     change that moves one path's numbers, or to **add a newly-ported geometry** additively
     without disturbing the established anchors.  The commit message records what moved and why
     (so the golden's git history is the audit trail of every deliberate baseline change).
3. **Wrapper home → `dev_scripts/regression/`** (operational subfolder, beside the
   `clean_install_all.sh` / `run_tests.sh` it calls).  **The python engine / manual launcher /
   golden-capture stay in `plans/experiments/sharding/scaling_tests/`** — they are tightly coupled to
   `scaling_common.py` (both `import scaling_common` and the `beta_root()` path derivation
   depend on living in scaling_tests/; see §1 "File-location split").
4. **Output storage → the dedicated repo `mbirjax_metrics`** (https://github.com/gbuzzard/mbirjax_metrics,
   public; Greg created it).  Cloned on both nodes — survives mbirjax branch churn, no push to
   mbirjax `main`, and (with the small representative `.npy`, note 3) makes the webserver
   unnecessary (§8).
5. **Test-step device count → 4 CPU devices** (`MBIRJAX_NUM_CPU_DEVICES=4`); exercises padding;
   the 4 B5 cone failures are confirmed and stay quiet day-over-day (no allowlist, §11).
6. **Cluster push → inline from the compute node** (they have outbound web); credential via a
   chmod-600 `$TOKEN_FILE` PAT (§8 / §12).

**Still to confirm (minor, non-blocking):**
- Exact `$METRICS_REPO` clone path on the cluster (a persistent home/scratch location).
- Whether to capture/track golden separately per branch in `BRANCHES`, or only for the primary
  track (proposal: primary track only — the golden is the stable anchor, not a per-branch thing).

---

## 14. Build order (CPU first, per Greg)

- ✅ **P0** — `performance_tracking.py` engine: Config + `run()`; GEOMETRY sweep; `--inline`;
  dated YAML.  (Done + CPU-validated.)
- ✅ **P1** — correctness fingerprint (true-shape crop + padding-zero check) + `vcd_nonconst`.
- ✅ **P2** — `run_performance_local.py` manual launcher.
- ✅ **P3** — diff + gate (day-over-day + golden; status transitions; exit code).  `tests_<date>.yaml`
  ingest is the wrapper's job (P5, not built).
- ✅ **P4 (engine side)** — `capture_golden.py` (full + `--only` selective refresh, writes
  `results/golden/golden_<plat>.yaml`).  The worktree@commit + clean-install + push wrapping is P5.
  No allowlist file — golden IS the expected state (§10a).
- ✅ **(added) Record book** — `records_<plat>.yaml` (best-ever per cell/metric + the commit).
- ✅ **(added) `main` 1-device baseline** — `capture_main_baseline.py` (time + memory + fingerprint
  at the full sweep sizes + small `.npy`); engine auto-discovers + prints a soft "vs main" note.
- **P5** — `dev_scripts/regression/` wrapper + `regression.env` + `enable/disable_nightly.sh` +
  README; the `BRANCHES` loop + worktree + fast/clean install + tests (`MBIRJAX_NUM_CPU_DEVICES=4`)
  + engine + repo pull/push (non-fatal) + notify; dry-run end-to-end on the Mac via `launchd`;
  measure wall-clock; tune sizes to the "few minutes" budget (decide if the 1024-class size is
  nightly or weekly).
- **P6 (GPU, Greg)** — `nightly_regression.slurm` + `scrontab` entry + `$TOKEN_FILE` credential;
  first golden capture on GPU (inline compute-node push); validate the per-device memory gate and
  speedup-ratio gate at real sizes incl. the 1024-class capacity sweep.
- **P7 (deferred)** — `compare_to_baseline.py` (local `.npy` deep diff against the in-repo
  golden array — no webserver) + a drift time-series reader/plot over `results/<plat>/`.
- **Visual interrogation surface (deferred, details TBD — Greg, 2026-06-15).**  The YAML time
  series + record book want a human-friendly way to explore: a spreadsheet export, a static
  browser page, and/or plots (time/memory/speedup vs date, per cell; record progression; the
  latest gate report).  This is the read/analysis side that complements the write side we are
  building now — scope and tooling to be decided later (likely reads `results/<plat>/*.yaml` +
  `records_<plat>.yaml`, so the file formats already carry everything it needs).

---

## 15. Adjacent task — harden unit-test geometry sizes (Greg's note E)

Independent of the regression tool, several unit tests use **cubic / even** geometry sizes that
hide the same two bug classes the regression sweep now targets:
- **Axis swaps / transposes** are invisible when `n_views == n_rows == n_channels`.
- **Padding paths** are never hit when every axis divides the device count.

Proposal (do as a separate, small PR so it doesn't entangle the tool): audit the geometry/sharding
tests and migrate representative ones to **asymmetric + at least one odd axis** (e.g. the
anisotropic-cone fixtures, the `tests/sharding/` sizes), so the unit suite itself exercises
swap-detection and padding the way the nightly does.  **Gate the migration carefully:** changing a
test's size changes its expected numbers, so each touched test must be re-baselined and re-verified
(tight `allclose`, not the old constants); the cone-padding cases simply keep failing until B5
(visible, expected).  This is complementary to — not a substitute for — the nightly's asymmetric sweep.
