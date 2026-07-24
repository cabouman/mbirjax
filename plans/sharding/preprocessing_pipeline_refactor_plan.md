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
   (`plans/experiments/sharding/collect_nsi_golden.py`, `ds4_sv20`) and verify the refactor against it with
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
after; collect/verify via `plans/experiments/sharding/collect_sibling_baseline.py`.  Coverage note: foam is
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

## 6b. Preprocessing memory diagnosis + fix (2026-06-30)

**Symptom (Greg, cluster).** `Lilly_recon.py` reported `peak_bytes_in_use ≈ 56.875 GB` **per GPU**
right after `scan_to_sino`'s "Sinogram computation complete." print — identical for n=2 and n=4
(GPU 0+1 used at n=2; all 4 at n=4), i.e. ~3× the full object array on **each** device, with **zero**
reduction from sharding.  `bytes_limit ≈ 77.6 GB` (the `XLA_PYTHON_CLIENT_MEM_FRACTION=0.98` pool set
in `tomography_model.py`), so `peak < limit` → genuine in-use, not a pre-allocation artifact.

**Sharding is fine.** `num_allocs` halved 4175 (n=2) → 2130 (n=4): each device runs its 1/n share of
batches.  But the per-device PEAK is fixed at ~56.8 GB regardless of n, because it is **one batch's**
working set (`batch_size=90` is the same on every device) — sharding splits views across devices but
never splits a batch.  `batch_size=10` ablation → ~6.45 GB (linear in `batch_size`), confirming it is
per-batch.

**Root cause (confirmed by Greg, stepping through `get_memory_stats`): `dm_pix.rotate`, NOT
interpolate.**  `interpolate_defective_pixels` tops out <5 GB (with ~8000 all-view dead pixels the NaN
count is ~720K/batch — the <5 GB is batch-sized reshape/copies, not the neighbor gather).  The blowup
is the detector-rotation kernel: a `(1496,1880,90)` block enters at ~1 GB and `dm_pix.rotate` spikes
gpu0 from 10.4 → **56.8 GB**.  Source dive (`dm_pix/_src/augment.py:731` `rotate` →
`affine_transform:707`; `interpolation.py:97`): `rotate` uses a **3×3** matrix (identity 3rd row), so
`affine_transform` builds a `meshgrid` over **all three** axes and `flat_nd_linear_interpolate` does
`2^3 = 8`-corner interpolation.  Each corner array is the full block (~1 GB): `point_indices` (8 GB)
+ `point_weights` (8 GB) + gather `volume[indices]` (8 GB) + `weights*gather` (8 GB) = ~32 GB, plus
~10 GB of meshgrid/coordinate/lower/upper arrays.  Three multiplicative wastes: (1) 8 corners not 4
(the view axis is a redundant identity interpolation dim), (2) every intermediate carries the full
90-view depth with the 2-D rotation grid replicated across all views, (3) **eager** execution (the
kernel is not jitted) → no XLA fusion/buffer-reuse, so all the 8 GB arrays coexist.  `= 8,116,719,616`
B `largest_alloc_size` is exactly one `(8,1496,1880,90)`-class corner array.

**Fixes.**
1. **DONE — `map_view_batches` (`pipeline.py`):** pre-allocate the host output (shape/dtype probed from
   the first batch) and have each worker write its disjoint view-slice **in place** — eliminating the
   per-shard result lists + final `np.concatenate`.  Host footprint ~3× → ~2× (input + output),
   independent of n.  Byte-identical.  (Keeps a full-batch shape probe — see the probe decision under
   item 3.)
2. **DONE — `interpolate_defective_pixels` rewritten as a jittable dense fill (`utilities.py`).**
   (Supersedes the interim chunked-`argwhere` median version.)  Invalid pixels (non-finite + the shared
   `defective_pixel_array`) are marked NaN and filled by **K=3 dense neighborhood-MEAN passes** computed
   with `reduce_window` box sums (values + a validity mask) — no `argwhere`, no data-dependent `while`,
   no 9× neighbor stack, O(N) memory.  Each pass fills the NaN frontier inward one pixel, so K=3 covers
   dead-pixel clusters up to ~5×5; pixels still NaN afterward are set to 0 with a host-side warning via
   `jax.debug.callback(..., ordered=True)`.  **MEAN replaces the old median** (Greg-approved): the test
   showed median is ~0.07 more accurate at the few dead pixels on sinogram gradients (max ~0.21 vs
   ~0.14 vs the true sino, <1% of pixels), recon-negligible; `tests/test_preprocessing.py` atol/nrmse
   relaxed accordingly (99th-pct gate unchanged).  Verified: jittable (jit==eager exactly), no NaN
   escapes, isolated/defective pixels fill correctly, 5×5 fills / 9×9 core warns+zeros.
3. **DONE (un-jitted) — replaced the rotation (Option C).**  `_rotation_kernel` is now a direct **2-D**
   bilinear rotation: it computes the rotated `(rows, cols)` sampling grid **once** (mirroring dm_pix's
   matrix/center/offset) and gathers the **4** neighbors for every view in one shot with shared 2-D
   weights — no per-view coordinate replication, 4 corners not 8, no 3-D `meshgrid`.  Structurally
   ~10× less transient memory (the ~46 GB came from `8 × (rows,cols,views)` corner arrays; now it's
   ~4 corner gathers ≈ a few GB eager) and faster.  **Validation (CPU):** angle=0 is exact; vs
   `dm_pix.rotate` max abs diff ~5.8e-5 (mean ~1e-6) on random N(0,1), concentrated at a ~1-3% of
   pixels near interpolation boundaries, no systematic bias; **end-to-end through `scan_to_sino` the
   diff is ~3.2e-6** (real sinograms are smoother).  No-rotation and interpolate paths stay
   byte-identical.  `tests/test_preprocessing.py` passes.  **Cluster: per-GPU preprocessing peak
   56.8 GB → 5.73 GB (~10×).**
   - **DONE — jitted (`_rotation_kernel_jit = jax.jit(_rotation_kernel)`)**, used in `scan_to_sino` and
     `correct_det_rotation`.  XLA fuses the 4-corner gather + weighted sum + mask into one kernel
     (buffer reuse + no eager per-op dispatch).  `det_rotation` flows as a traced scalar → one compile
     per shape, reused across angles; under `default_device(device)` it lands on the worker's device.
     **Validation (CPU):** jit vs eager ~5.5e-5 (XLA fusion reorder); **jit vs `dm_pix` only ~2.4e-7**
     (the jitted form is *closer* to dm_pix than eager); end-to-end `scan_to_sino` vs the dm_pix golden
     **1.2e-7**; rough CPU time/call 13.6 → 3.4 ms (**~4×**, indicative — real win is GPU); tests pass.
     GPU memory/speed to be confirmed on the cluster.
   - **Probe decision:** keep the full-batch shape probe in `pipeline.py` (do NOT switch to a 1-view
     probe).  C removed the gpu0 rotate spike that motivated it, and with the jitted rotation the
     full-batch probe reuses the workers' compiled shape, whereas a 1-view probe would force an extra
     (1, rows, cols) compile for negligible gain.
4. **DONE — whole `fused_kernel` wrapped in a single `jax.jit` (`scan_to_sino`).**  Now that interpolate
   is jittable, the entire per-batch kernel — (downsample) → transmission → interpolate → rotation — is
   one jitted function (it calls the *eager* `_rotation_kernel`, fused by the outer jit; the standalone
   `_rotation_kernel_jit` stays for `correct_det_rotation`).  Host constants + the do_downsample/
   do_rotation flags + `det_rotation` are compile constants; it compiles once per shape per device under
   each worker's `default_device`.  XLA fuses all stages → one dispatch per batch, buffer reuse.
   **Validation (CPU):** runs single + multi device; **1-dev vs N-dev is now bit-exact (0.0)** for both
   downsample and plain paths (jit fusion removed the prior ~1-ULP cross-device drift); `tests` pass.
   One gotcha fixed: interpolate's `ravel_multi_index` needs `mode='clip'` (default `'raise'` forces a
   concrete bounds check that breaks under jit).  GPU speed/memory to be confirmed on the cluster.

**Verification so far.** Ephemeral pre/post golden on NaN-heavy synthetic data (single + multi CPU
device): `interpolate` and `scan_to_sino` byte-identical (max abs diff 0.0); multi-chunk interpolate
checked at chunk=3/7/50; `tests/test_preprocessing.py` passes.  Cross-device 1-ULP diffs in the
downsample variant are the known pre-existing XLA op-fusion artifact.

**Broader thread (your prompt):** the whole fused kernel is eager; `interpolate`'s `argwhere` (dynamic
shapes) blocks jitting it wholesale, but the rotation/transmission/downsample are jittable — worth
revisiting for the multi-device throughput, separate from this memory fix.

**Data note (Greg to check, not a code issue):** ~8000 detector pixels are dead in all views.

## 6c. Zinger correction — jittable + shared fill + folded into scan_to_sino (2026-06-30)

`interpolate_zinger_pixels` was the twin of the old `interpolate_defective_pixels` (argwhere + while +
nanmedian; not jittable, per-zinger gather).  Rewritten the same way and **folded into `scan_to_sino`**
because large zeiss data is a live target and the path is transfer-bound (folding avoids a whole-sino
host↔device round-trip).

**Shared code.** Factored `_fill_nan_pixels(sino, num_passes=3)` (the K-pass dense nanmean fill + warn +
zero) out of `interpolate_defective_pixels`; both it and zinger now call it.  Added
`_zinger_fill(sino, threshold, num_passes)` (flag non-finite + `value < threshold` → NaN, then
`_fill_nan_pixels`) and `_zinger_threshold(sino_sub, ratio)` (= `-ratio·RMS` over support).  Defective
vs zinger differ ONLY in how pixels are flagged NaN.

**Fold.** `scan_to_sino(..., zinger_pixel_ratio=None)`: when set, a cheap pre-pass runs the kernel
(no zinger) on a ~20-view subsample → threshold; the main fused kernel then appends
`_zinger_fill(threshold)` after rotation — one jitted pass, no extra round-trip.  `zeiss.py` passes
`zinger_pixel_ratio=0.1 if zinger_correction else None` and drops the separate `interpolate_zinger_pixels`
call.  Zinger now runs **before** offset/shifts (Greg: fine, arguably better — removes a zinger before a
sub-pixel shift could smear it).

**Thin wrappers kept (Greg):** `interpolate_zinger_pixels` is now a standalone `map_view_batches`-driven
wrapper (threshold from a subsample → per-batch `_zinger_fill`, memory-bounded + shardable, returns a
NumPy sinogram); `detect_zinger_pixels` keeps its indices API via `_zinger_threshold`.

**Behavior changes (Greg-approved):** median→mean; fixed K=3 + warn/zero instead of raise; ordering
(zinger before offset/shifts).  Detection condition (`value < -ratio·RMS`) unchanged.

**Verification (CPU):** defective path **unchanged** (byte-identical via the refactor); no-zinger
`scan_to_sino` **unchanged** (cross-device 0.0; scan_1dev 1.49e-7 / scan_plain 0.73 vs the old golden,
same as before); zinger standalone + fold: zingers corrected, no NaN escapes, **1-dev vs 4-dev
bit-exact (0.0)**; `detect_zinger_pixels` finds all injected zingers; `tests/test_preprocessing.py`
passes.  **zeiss before/after on a `demo_zeiss.py` recon to be confirmed on the cluster** (same-platform;
the ordering + median→mean shifts are expected and recon-negligible).

**Subsampling DRY'd (Greg).** Factored the view-subsample-before-`_get_sino_indicator` pattern into
`TomographyModel.subsample_views(array, max_views_to_use=20, num_real_views=None)` (host, evenly-spaced),
sitting next to `_get_sino_indicator` (which now carries a note that a subsample should typically be
passed, not the full sinogram).  `_zinger_threshold` now subsamples **internally** (callers pass the full
sino); `detect_zinger_pixels`, `interpolate_zinger_pixels`, `scan_to_sino`'s threshold pre-pass,
`est_crop_width`, and `auto_set_regularization_params` all route through it.

`num_real_views` samples only `array[:num_real_views]` -- for a device-form input whose view axis is
zero-padded (`prepare_sino_for_devices`), it samples the REAL views directly instead of sampling the
padded array and dropping padded slots afterward (which left fewer real views).  `auto_set_reg` uses this
and **drops its old `keep` filter**.  Identical when there's no padding (all tests, verified: `test_qggmrf`
+ `geometries/test_vcd` recons pass); a slightly better estimate only in the padded recon path.

**Preprocessing does NOT pad views** (checked): `map_view_batches` uses `np.array_split` (uneven-but-real
shards, no zero-padding) into a `(num_views,...)` output, so the zinger/crop estimates see only real
views.  View-padding is recon-side (`prepare_sino_for_devices`), which is why only `auto_set_reg` needs
padding-awareness.

## 7. Open decisions
- Small-fixture format/location (committed .npz vs opt-in cluster path test).
- Whether `compute_weight` (zeiss_tct) becomes a driver kernel now or stays a separate host call until
  Phase 4.
- Whether to expose the in-memory sharded-sino fast path in this work or defer (Phase 5 note).
