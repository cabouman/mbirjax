# Preprocessing pipeline refactor — DRY, fused, view-sharded

**Status:** PLAN (created 2026-06-29). Scope = `mbirjax/preprocess/`. Relates to status task #18
(shard mar.py + preprocessing).

**Goal.** Make the scan→sinogram preprocessing path (a) avoid the repeated host↔device round-trips
and per-stage host re-allocations, (b) run view-sharded across all devices when available, and
(c) share one code path across the format loaders (nsi, zeiss, zeiss_tct, pymbir) instead of the
current near-duplicate orchestration — **without** collapsing the modular per-step functions.

**Sequencing decision (Greg).** Do the **full conversion on nsi first** (kernel split → fuse →
shard), leaving zeiss / zeiss_tct / pymbir untouched and working; **then** migrate the siblings.
**pymbir migration is OPTIONAL** (it is the outlier — loads a sinogram directly, no scan→sino front).

---

## 1. Current state: what is shared vs format-specific

| step (on the big arrays) | nsi | zeiss | zeiss_tct | pymbir |
|---|---|---|---|---|
| load scans + parse geometry | NSI (nsipro/rtf + tif stack) | Zeiss OLE/txrm | Zeiss tct dirs | ORNL HDF5 (**loads sino directly**) |
| convert → mbirjax params | `convert_nsi_*` | `convert_zeiss_*` | `convert_zeiss_*` | `create_proj_params_dict_ornl` |
| crop | ✓ | ✓ | ✓ | — |
| downsample (if >1) | ✓ | ✓ | — | — |
| compute weights | — | — | ✓ `compute_weight` | — |
| transmission (−log) | ✓ | ✓ | ✓ | — |
| det rotation | ✓ | — | — | ✓ |
| background offset | ✓ per_view | ✓ bg_option | ✓ global | — |
| sino shifts | — | ✓ | — | — |
| zinger | — | ✓ | — | — |
| beam hardening | — | — | — | ✓ |
| return tuple | 3 | 4 (+metadata) | 4 (+weights) | 3 |

**Observations that drive the design**
- The **scan→sino body** (crop → downsample → transmission → corrections) is structurally identical
  across nsi/zeiss/zeiss_tct and is *already half-factored* into `utilities.py`. The duplication is
  (a) the per-stage host↔device round-trip + `np.concatenate` baked into each utility, and (b) the
  orchestration sequence retyped in each `compute_sino_and_params`.
- **Every correction is a per-view sinogram→sinogram transform** (rotation, offset, sino-shifts,
  zinger, bh) → a composable, view-shardable chain. Format differences are just *which steps, in what order*.
- **pymbir is the outlier**: no scan→sino front; it loads a sinogram and runs only [bh, rotation].
  The abstraction must let the scan→sino front be optional and the correction chain run standalone.
- The genuinely format-specific parts — **load/parse, param-convert, return-tuple shape** — stay per
  module; they do not DRY-up.

### The three concerns bundled in each utility today
1. the math (−log, block-average, rotate, per-view offset),
2. view-batching for memory,
3. host↔device transfer + `np.concatenate`.

Concerns 2 and 3 are duplicated across functions and ARE the inefficiency. The refactor **separates**
them: one driver owns 2+3; each step keeps only concern 1.

---

## 2. Target architecture (three layers)

- **Frontend (per-format, stays put):** `load_*` + `convert_*`. Produces `(obj, blank, dark, defective)`
  + params; for pymbir, a raw sinogram + params.
- **Shared core (new, DRY — `preprocess/pipeline.py` + kernels in `utilities.py`):**
  - a **driver** that owns batching/sharding + a single upload / single gather (the only place
    transfer code lives), built on the existing `mbirjax.sharding` infra (`Placement`,
    `run_per_device`, `_shard_sinogram` / `_gather_sinogram`);
  - the existing utilities recast as **pure per-shard kernels** (crop, downsample, transmission,
    rotation, offset, zinger, sino-shift, bh, compute_weight);
  - a thin **composition** entry: `scan_to_sino(scans, cfg)` then a `corrections=[...]` chain, both
    run through the one driver.
- **Backend (per-format, stays put):** assemble the 3- or 4-tuple (metadata / weights).

Each format's `compute_sino_and_params` collapses to **frontend → core(steps for this format) → backend**.
The step list *is* the format's identity:
- nsi: scan→sino, corrections `[rotation(if≠0), offset(per_view)]`
- zeiss: scan→sino, corrections `[offset(bg_option), sino_shifts, zinger(if on)]`
- zeiss_tct: scan→sino (+ `compute_weight`), corrections `[offset(global)]`
- pymbir (optional): no scan→sino; corrections `[bh(if on), rotation(if≠0)]`

### Step contract
Same convention we standardized in the model: **numpy-or-jax in → sharded device form out**; the
driver gathers once at the end (host numpy for the disk path; optionally a sharded array for an
in-memory hand-off to `recon`). Kernels never upload/download/loop internally.

### Modularity
Functions are **not** collapsed — each is split into `kernel` (device-array→device-array) + a thin
**host wrapper** that preserves today's public `host→host` API for standalone use and back-compat.
Separation of concerns *increases* modularity.

---

## 3. Validation strategy

Three tiers, cheapest first:

1. **Kernel unit tests (fast, committable, CI).** Extend the existing synthetic
   `TestNSIPreprocessing` (40×64×128 phantom-derived obj/blank/dark with injected defective pixels)
   to exercise each kernel and the fused pipeline directly on device arrays. Primary gate for
   Phases 1–2.
2. **End-to-end nsi golden vs the real dataset.** Source = `/depot/bouman/data/Lilly/
   Autoinjector_HighRes_Horizontal/` (cluster). Derive a **small committable fixture** by heavy
   subsample/crop and store the expected `(sino, cone_beam_params, optional_params)` as a reference
   (.npz / hdf5). Run `compute_sino_and_params` against it and assert bit-for-bit (Phase 2) /
   value-stable within a documented float-fusion epsilon (Phase 3). Full-resolution run is the heavy
   gate, run on the cluster.
3. **Multi-GPU speedup (Phase 3).** Full Lilly dataset on the cluster; confirm near-linear scaling
   with device count and bounded per-shard device memory. Greg runs cluster jobs.

**Prerequisite:** generate the small golden fixture once from the Lilly dataset before Phase 2.
Greg can copy the dataset locally for CPU dev, or generate the fixture on the cluster and commit only
the small reference. (Large datasets are not committed.)

---

## 4. Staged plan

### Phase 0 — Design lock + safety net
- Finalize the kernel/driver boundary, the step contract, and the per-format step lists (Section 2)
  — confirm the abstraction fits all four formats before code.
- Build the validation harness: extend synthetic kernel tests; generate the small Lilly golden
  fixture and capture current `nsi.compute_sino_and_params` output as the reference.
- **Gate:** golden reference captured; kernel tests cover crop/downsample/transmission/rotation/offset
  on synthetic data and pass against current code.

### Phase 1 — Shared kernel/driver foundation (siblings untouched, single device)
- Add `preprocess/pipeline.py` with the driver (batched, single-device for now). Recast each
  `utilities.py` function into **(pure per-batch kernel) + (thin host wrapper)**; the wrappers keep
  the exact current public API so zeiss / zeiss_tct / pymbir keep calling them **unchanged**.
  Consolidate the three duplicated `for-batch / upload / compute / download / concatenate` loops into
  the one driver.
- **Gate:** every format's public functions byte-identical (siblings still use the wrappers);
  synthetic kernel tests + nsi golden stable.

### Phase 2 — Fuse the nsi path (single upload / single gather, single device)
- Rewire `nsi.compute_sino_and_params` to call the core (`scan_to_sino` + correction chain) through
  the driver, so data stays on-device across crop→downsample→transmission→rotation→offset (no
  per-stage host re-allocation). Free win: guard rotation on `det_rotation != 0` (today it
  round-trips the whole sinogram to rotate by zero).
- **Gate:** nsi golden stable (document any tiny float-fusion epsilon from op reordering); peak host
  memory drops (no per-stage `concatenate`). Siblings still unchanged.

### Phase 3 — Shard the driver; nsi runs multi-device  *(nsi fully converted here)*
- Distribute the driver's view loop across all devices via `mjs` `Placement` / `run_per_device`,
  reusing the recon's sharding mechanism. Isolate the two non-per-view operations — the **global**
  background-offset percentile and the **residual-NaN cleanup** in `interpolate_defective_pixels` —
  as explicit small reductions so the rest shards freely.
- **Gate:** nsi values stable vs Phase 2 (per-view steps are bit-stable; apply the reduction-order
  care used elsewhere to the global-offset path); near-linear multi-GPU speedup on the full Lilly
  dataset (cluster run).

### Phase 4 — Migrate siblings onto the shared core (the DRY payoff)
- One at a time: **zeiss → zeiss_tct**. Each keeps its frontend/backend and just declares its
  correction step list; delete its duplicated orchestration. Add format-specific kernels
  (`sino_shifts`, `zinger`, `compute_weight`) as steps/driver calls.
- **pymbir: OPTIONAL.** If done, wire only the correction chain (`[bh, rotation]`) on the preloaded
  sinogram — validates that the scan→sino front is cleanly optional.
- **Gate:** each migrated format's golden output stable; net line count drops as duplication is removed.

### Phase 5 — Cleanup + docs
- Remove dead per-format scaffolding; document the step contract and how to add a new format/step;
  note the optional in-memory fast path (hand `recon` a sharded sinogram instead of a host
  round-trip — keep the host gather only on the save-to-disk path).

---

## 5. DRY notes / file layout / risks
- **Where shared code lives:** kernels stay in `utilities.py`; the driver + composition go in a new
  `preprocess/pipeline.py` so transfer/sharding scaffolding is in exactly one place.
- **Back-compat:** the public per-step functions survive as host wrappers; the two-stage
  `save_preprocessing` / `load_preprocessing` disk workflow and the `Lilly_*` app scripts keep working.
- **Numerical safety:** view-sharding the per-view steps is bit-stable (no cross-view reductions); the
  only cross-view ops (global offset percentile, residual-NaN cleanup) need the reduction-order care
  we applied to recon.
- **Biggest risk is validation coverage, not the refactor.** The Phase 0 golden fixture is
  load-bearing — without it the migrations are unverifiable. Confirm the small fixture is generated
  before Phase 2.
- **pymbir asymmetry:** no scan→sino front and a `bh` step no other format has — kept optional so it
  doesn't gate the main DRY win.

## 6. Deferred candidates (pre-recon path — on the list, not in this pass)

The same treatment (input/output contract, host↔device round-trips, sharding) applies to **anything
between load and the start of recon that we have not already cleaned**. Captured here so it is not lost;
explicitly **deferred** — outside the nsi-first / siblings sequence above — but the natural follow-on
once the main pipeline is fused and sharded.

- **`auto_crop_sino_conebeam` + `est_crop_width`** (`preprocess/utilities.py`): auto-detect blank
  detector margins, crop the sinogram, and adjust geometry offsets. Runs right after
  `compute_sino_and_params` in `Lilly_recon.py`. Review for input/output contract (currently host
  numpy: `_get_sino_indicator` → `np.any`/`np.argmax`) and whether `_get_sino_indicator` round-trips
  the full sinogram host↔device.
- **`TomographyModel._get_sino_indicator`** (`tomography_model.py`, staticmethod): builds the
  object-support mask from the sinogram. **Needs a closer look before any change** because it is
  **shared across three call sites** — the recon init path (`vcd`/recon setup), denoising
  (`denoising.py`), and the preprocess crop helpers (`est_crop_width`, and another use ~`utilities.py:1189`).
  Cleaning it is therefore *not* a preprocess-local edit: it touches the recon contract and must be
  verified on the recon side too. Treat as its own scoped item, not folded into the nsi pass.

## 7. Open decisions
- Small-fixture format/location (committed .npz vs opt-in cluster path test).
- Whether `compute_weight` (zeiss_tct) becomes a driver kernel now or stays a separate host call until
  Phase 4.
- Whether to expose the in-memory sharded-sino fast path in this work or defer (Phase 5 note).
