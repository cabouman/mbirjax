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
2. **End-to-end nsi golden vs the real dataset (EPHEMERAL).** Source = the Lilly Autoinjector dataset
   (`/depot/bouman/data/Lilly/Autoinjector_HighRes_Horizontal/` on the cluster; Samba-mounted at
   `/Volumes/bouman/...` locally). Capture the current `(sino, cone_beam_params, optional_params)` once
   (`experiments/sharding/collect_nsi_golden.py`, `ds4_sv20`) and verify the refactor against it with
   `--ref`. **The golden is NOT kept/committed** — once the new implementation is verified, the new
   implementation is the gold standard.
3. **Multi-GPU speedup (Phase 3).** Full Lilly dataset on the cluster; confirm near-linear scaling
   with device count and bounded per-shard device memory. Greg runs cluster jobs.

**Cross-platform caveat (load-bearing — learned in Phase 1).** Preprocessing is **not** byte-reproducible
across platforms: borderline pixels (the `ratio > 0` sign threshold in `-log`, `nanmedian` ties,
`dm_pix.rotate` bilinear interpolation at edges) diverge CPU↔GPU, giving localized ~1.0 sino diffs
(observed: max abs diff 1.03 between Mac-CPU and the GPU-captured golden — and *pre-refactor* HEAD showed
the identical 1.03, proving it is platform, not code). Therefore:
- **Same-platform comparison is the gate.** Verify on the SAME device the golden was captured on
  (capture + `--ref` both on the cluster GPU → expect PASS; Phase 1 confirmed PASS there).
- The platform-independent proof of a refactor is **repo-vs-HEAD byte-identical on one platform** (use a
  `git worktree` at HEAD; the kernels run eagerly so fusion stays byte-identical — no host round-trip
  changes float32 values). This is the primary local gate; the cross-platform golden is a sanity check,
  not a bit-exact requirement.

**Env note:** develop with an **editable install** (`pip install -e .`).  A non-editable site-packages
copy silently shadows the repo for any script run from a non-repo cwd (this bit Phase 1 verification).

---

## 4. Staged plan

### Phase 0 — Design lock + safety net
- Finalize the kernel/driver boundary, the step contract, and the per-format step lists (Section 2)
  — confirm the abstraction fits all four formats before code.
- Build the validation harness: extend synthetic kernel tests; generate the small Lilly golden
  fixture and capture current `nsi.compute_sino_and_params` output as the reference.
- **Gate:** golden reference captured; kernel tests cover crop/downsample/transmission/rotation/offset
  on synthetic data and pass against current code.

### Phase 1 — Shared kernel/driver foundation (siblings untouched, single device)  **[DONE 2026-06-29]**
Implemented: `preprocess/pipeline.py::map_view_batches` driver + `_transmission_kernel` /
`_downsample_obj_kernel` / `_rotation_kernel` extracted; three duplicated loops collapsed.  Also fixed a
latent bug (`downsample_view_data` indexed `dark_scan[i]` over `blank_scan.shape[0]` — now two loops).
Gates passed: synthetic repo-vs-HEAD **byte-identical** (worktree); suite green; siblings byte-identical;
cluster GPU `--ref` **PASS** on the real Lilly golden.
- Add `preprocess/pipeline.py` with the driver (batched, single-device for now). Recast each
  `utilities.py` function into **(pure per-batch kernel) + (thin host wrapper)**; the wrappers keep
  the exact current public API so zeiss / zeiss_tct / pymbir keep calling them **unchanged**.
  Consolidate the three duplicated `for-batch / upload / compute / download / concatenate` loops into
  the one driver.
- **Gate:** every format's public functions byte-identical (siblings still use the wrappers);
  synthetic kernel tests + nsi golden stable.

### Phase 2 — Fuse the nsi path (single upload / single gather, single device)  **[DONE 2026-06-29, local; cluster --ref pending]**
Implemented: `utilities.py::scan_to_sino` fuses (downsample) → transmission → (rotation) in one
on-device pass per view-batch (object scan uploaded once, sinogram gathered once); `_downsample_blank_dark`
factored out and shared with `downsample_view_data`; `nsi.compute_sino_and_params` rewired to
crop → `scan_to_sino` → host `correct_background_offset`; `det_rotation != 0` guard falls out of the
conditional composition.  Offset stays a host pass (per_view, cheap; on-device fusion deferred).
Gates passed (Mac CPU): synthetic sequential-vs-fused **byte-identical** for transmission→rotation,
**3.6e-7** (~1 ULP, data-specific) with downsample; full-nsi Phase-2 fingerprint (sum/min/max/mean)
**matches HEAD-on-Mac exactly** on the real Lilly data (ds4_sv20).  Pending: cluster GPU `--ref` (expect
PASS — fusion epsilon ≪ 1e-5).
- Rewire `nsi.compute_sino_and_params` to call the core (`scan_to_sino` + correction chain) through
  the driver, so data stays on-device across crop→downsample→transmission→rotation→offset (no
  per-stage host re-allocation). Free win: guard rotation on `det_rotation != 0` (today it
  round-trips the whole sinogram to rotate by zero).
- **Gate:** nsi golden stable (document any tiny float-fusion epsilon from op reordering); peak host
  memory drops (no per-stage `concatenate`). Siblings still unchanged.

### Phase 3 — Shard the driver; nsi runs multi-device  *(nsi fully converted here)*  **[DONE 2026-06-29, local; cluster speedup pending]**
Implemented: `map_view_batches` gained a `devices` arg — single device → the sequential loop (default,
so the legacy public helpers are unchanged); multiple devices → contiguous in-order view shards run
concurrently via `mjs.run_per_device` (each worker under `jax.default_device`).  `scan_to_sino` defaults
to `jax.devices()` and was made **device-agnostic**: its constants are HOST NumPy that auto-promote to
each batch's device (and `_downsample_blank_dark` now returns a host `flat_indices`).  Every stage is
per-view (defective interpolation uses within-view neighbors; the `per_view` offset is the separate host
pass), so there is **no cross-device communication**.  Gates passed (forced 4 CPU devices): scan_to_sino
**1-device vs 4-device byte-identical** across {ds 1×1 / 2×2} × {rot 0 / ≠0} × {even & uneven shards};
full-nsi 4-CPU-sharded on the real Lilly data **matches HEAD-on-Mac exactly** (sum/min/max/mean);
public helpers still byte-identical to HEAD; suite green.  **Multi-GPU speedup CONFIRMED** (cluster,
isolated `scan_to_sino` via `time_scan_to_sino.py`, ds1/sv1, 1800 views ~20 GB): 1 GPU 55s -> 4 GPUs
19.3s = **2.88x**, byte-identical 1-vs-4.  Sub-4x is host-transfer-bandwidth-bound (~20 GB up + ~20 GB
down shared across the GPUs' PCIe), not a parallelism bug; whole-script `bash time` was misleading
(dominated by import + 4-GPU init + compile + the script's repeated runs).  Optional follow-ups: trim the
driver's double host concatenate to one (workers return batch lists; one concat in view order); the
bigger structural win is the Phase-5 recon fast path (skip the 20 GB host gather, hand recon a sharded
sinogram).  Phase 3 DONE.
- Distribute the driver's view loop across all devices via `mjs` `Placement` / `run_per_device`,
  reusing the recon's sharding mechanism. Isolate the two non-per-view operations — the **global**
  background-offset percentile and the **residual-NaN cleanup** in `interpolate_defective_pixels` —
  as explicit small reductions so the rest shards freely.
- **Gate:** nsi values stable vs Phase 2 (per-view steps are bit-stable; apply the reduction-order
  care used elsewhere to the global-offset path); near-linear multi-GPU speedup on the full Lilly
  dataset (cluster run).

### Phase 4 — Migrate siblings onto the shared core (the DRY payoff)  **[zeiss + zeiss_tct DONE 2026-06-29; pymbir still optional/not done]**
Done: both siblings now call `scan_to_sino` for the shared compute and delete the duplicated
downsample/transmission orchestration:
  * **zeiss**: `crop → scan_to_sino(downsample_factor, det_rotation=0) → offset → sino_shifts → zinger`.
  * **zeiss_tct**: `crop → compute_weight → scan_to_sino(downsample=(1,1), det_rotation=0) → offset`.
The corrections were left UNCHANGED (the cross-view `global` offset stays a cheap host pass — no
reduction machinery; `sino_shifts`/`zinger`/`compute_weight` are per-view/cheap).  So the only change is
the (already-validated) `scan_to_sino` swap; siblings now also shard the transmission across devices.
Verified by an **ephemeral real-format before/after** (loaders run locally via olefile + Samba):
zeiss foam512 `.txrm` (ds1/sv1) **PASS, max abs diff 0.0**; zeiss_tct purdue BGA `.xrm` **PASS, max abs
diff 0.0** (sino + params + metadata/weights all match).  Baselines (`~/Documents/tmp/*.npz`) discarded
after; collect/verify via `experiments/sharding/collect_sibling_baseline.py`.  Coverage note: foam is
ds1, so the zeiss *downsample* branch is covered transitively by `scan_to_sino`'s validation (synthetic
ds2 byte-identical, nsi ds4 exact), not a real-data ds>1 zeiss run.
- **pymbir: OPTIONAL, not done.** If done, wire only the correction chain (`[bh, rotation]`) on the
  preloaded sinogram — validates that the scan→sino front is cleanly optional.
- Optional follow-up: shard `correct_sino_shifts` (currently a per-view Python loop) through the driver.

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
  `compute_sino_and_params` in `Lilly_recon.py`.  **SLOWNESS FIXED 2026-06-29:** `est_crop_width` now
  subsamples views to ≤20 (matching `auto_set_regularization_params`) before `_get_sino_indicator`, so
  the full-array host passes (finiteness/histogram/median/mask) run on ~20 views instead of the whole
  ~20 GB sinogram -- byte-identical crop for a view-stable object, ~17× faster on a 400-view synthetic.
  Full row/column resolution is preserved; the safety_buffer absorbs the small view-sampling
  approximation.  No sharding needed here anymore.
- **`TomographyModel._get_sino_indicator`** (`tomography_model.py`, staticmethod): builds the
  object-support mask from the sinogram. **Needs a closer look before any change** because it is
  **shared across three call sites** — the recon init path (`vcd`/recon setup), denoising
  (`denoising.py`), and the preprocess crop helpers (`est_crop_width`, and another use ~`utilities.py:1189`).
  Cleaning it is therefore *not* a preprocess-local edit: it touches the recon contract and must be
  verified on the recon side too. Treat as its own scoped item, not folded into the nsi pass.

**Related sharding finding (Phase 3, not pre-recon):** `_generate_3d_shepp_logan_sharded`
(`utilities.py`) uses a single-threaded loop + async dispatch (no thread pool).  A GPU run showed no
speedup (and `block_until_ready` made no difference) — single-thread async dispatch does not overlap
devices well.  The preprocessing path's `run_per_device` (thread pool) DID parallelize on GPU
(user+sys ~2.6x with lower wall on the collect-golden `time`).  So the generator should likely switch
to `run_per_device` too — separate revisit.

## 7. Open decisions
- Small-fixture format/location (committed .npz vs opt-in cluster path test).
- Whether `compute_weight` (zeiss_tct) becomes a driver kernel now or stays a separate host call until
  Phase 4.
- Whether to expose the in-memory sharded-sino fast path in this work or defer (Phase 5 note).
