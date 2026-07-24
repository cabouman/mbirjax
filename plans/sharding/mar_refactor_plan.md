# MAR (`preprocess/mar.py`) refactor — device management, sharding, memory

**Status:** ✅ **COMPLETE / shipped** (2026-07-03).  Phases 1, 2, 2b, 2c, 2d are all done and committed on
`greg/shard_profiling`, and validated at full scale on the cluster (the (1800,1365,1880) and 1024³ MAR
traces that drove the Phase-2d fixes).  Net result: the correction path is view-sharded end-to-end (no
single-device gather), the recon stays on-device through the BH loop (sharded histogram + Otsu), and the
memory footguns found in the full-scale trace are fixed.  **Phase 3 (subsample / speed up the BH model fit)
is DEFERRED and now tracked in `current_plans.md`**, along with the residual open questions below.
Scope = `mbirjax/preprocess/mar.py` (+ `segmentation.py`, `utilities.py`); the scan→sino preprocessing half
of status task #18 was already done.

_(History: created 2026-06-29; revised 2026-07-01, phases reordered so the correction-application sharding
led and the BH-fit subsampling deferred; phases completed through 2026-07-02.  Phase records retained below
as the implementation log.)_

**Context.** The MAR loop `recon_plastic_metal` alternates `correct_sino_plastic_metal` (beam-hardening
correction driven by a segmented recon) with `recon`/`split_sino_recon` (already sharded — leave alone).
The issues are all in the **correction** path.

---

## Current issues

1. **Single-device gather (memory; defeats sharding).** `correct_sino_plastic_metal` does
   `device = ct_model.recon_placement.devices[0]`, and `_est_plastic_metal_sinos_from_recon` then does
   `ct_model.forward_project(jax.device_put(mask * recon, device))`.  When `recon` is a sharded array,
   `device_put(..., device[0])` **gathers the full recon-sized volume onto one GPU** before forward
   projection — the single-device-gather footgun we removed elsewhere — instead of letting
   `forward_project` shard internally.

2. **Correction application on the full sino** (`_correct_plastic_sinogram`, `BH_correction`; batched at
   64) is per-pixel over the whole (flattened) sinogram → should run per view-batch through the shared
   `map_view_batches` driver with **jitted** kernels (memory-bounded + multi-GPU + fused), exactly like
   `scan_to_sino`.

3. **BH model fit runs on the full flattened sinogram.** `_est_plastic_metal_sinos_from_recon` returns
   full-sino-length vectors (`p`, `m_i`); `_compute_entry_for_OSQP` builds full-length `H` columns
   (`_get_column_H`) and dots them on one device, recomputing each column O(num_cols²) times.  The fit is
   statistical (a few polynomial coefficients), so in principle it needs only a subsample — BUT the
   subsample must contain pixels spanning DIVERSE metal path length, which are sparse in a mostly-plastic
   object.  This is the hardest piece and is deferred to Phase 3.

The recon steps in the loop (`split_sino_recon` / `recon`) are already sharded — out of scope.

---

## Staged plan

### Phase 1 — Retire the single-device gather (memory; byte-identical)  **[DONE 2026-07-01]**
- Dropped the `device` param from `_est_plastic_metal_sinos_from_recon` and the
  `device = ct_model.recon_placement.devices[0]` in `correct_sino_plastic_metal`; `forward_project` now
  shards `plastic_mask` / `mask*recon` internally (`_shard_recon`, a no-op if already sharded) — no
  full-volume gather onto one GPU.
- **Verified byte-identical:** `forward_project(device_put(x, dev0))` == `forward_project(x)` on a
  4-device model (max_abs_diff 0.0) — the old `device_put` was a pure gather; and an end-to-end MAR smoke
  (`correct_sino_plastic_metal` on a plastic+metal phantom) produced a finite corrected sino.

### Phase 2 — Shard the correction across devices (memory + parallelism)  **[DONE 2026-07-01]**
Pivoted from `map_view_batches` to **JAX-native view-sharding + SPMD** (Greg): the correction has global
reductions (`Sp` mean floor, scaling inner products, `HtH`/`Hty`, argmins) that `map_view_batches` (a
map, not a reduce) can't express, whereas sharded arrays give those reductions as cross-device
all-reduces *and* run the elementwise steps per-shard in parallel.  Motivation is memory: on a 20+ GB
sino the un-sharded correction is ~(5+m)×S on one device → OOM; sharded it is ~(5+m)×S/n per device.
(Speed is a minor bonus — the correction is a few % of the MAR recon.)
- Projections use `output_sharded=True`; the sinos stay **3-D view-sharded** (no flatten to one device).
- `_compute_entry_for_OSQP`: `jnp.dot` → `jnp.sum(a*b)` (all-reduces on N-D sharded; padded `h_i`=0 → no
  mask needed).  `_correct_plastic_sinogram`: `median(Sp)` → **masked mean** (median needs a global sort
  → would gather; mean all-reduces).  `_estimate_plastic_scaling`: boolean-select → `where`-masked sums.
  `_find_most_violated_constraints`: masked argmin over real views + flat-index reads.  `_get_row_H`:
  index the flattened view.  A tiny per-view `view_mask` (1 on real, 0 on the zero-padded views from
  `prepare_sino_for_devices`) excludes padding from the reductions.  The return `_gather_sinogram`s to a
  host sino cropped to real views (feeds the recon as before).
- **Verified (CPU):** `correct_sino_plastic_metal` on a plastic+metal phantom, **n=1 (unpadded) vs n=4
  (padded/sharded): max_abs 2.2e-4, NRMSE 2.9e-6** — i.e. consistent; the tiny max is the argmin
  constraint-pixel sensitivity to all-reduce ULP (no padding leak: NRMSE ~ULP, no bias).  Both finite,
  padding cropped.  Structurally nothing full-length lands on one device (all view-sharded; only the
  final gather is host).  **Behavior change:** median→mean floor (approved).  **GPU per-device memory +
  a real MAR before/after to be confirmed on the cluster.**

### Phase 2b — Keep the recon on the devices through the loop (sharded segmentation)  **[DONE 2026-07-02]**
Greg's cluster trace: the recon bounced host↔device each BH iteration, and `np.histogram`
(`segmentation.py`, inside `multi_threshold_otsu`) was the op pinning the recon to the host.
- **Sharded histogram:** a histogram is sum-decomposable (integer counts; min/max order-insensitive), so
  `jnp.histogram` on a sharded volume compiles to per-shard partials + all-reduce — verified **0
  all-gathers** (3 all-reduces for counting; the bin determination = a separate masked min/max pass with
  its own all-reduces, NOT part of those 3).  `_sharded_histogram` (new, `segmentation.py`) uses masked
  min/max for the bin range and pushes padded entries to a finite sentinel ABOVE the range (dropped by
  `histogram` range semantics) → bin edges AND counts **bit-identical** to the unpadded host computation
  (no post-hoc count correction; Greg's bin-boundary concern resolved by construction).  Memory: temps
  are shard-bounded (CPU-measured ~2×shard for min/max, ~7×shard worst-case for the histogram; if the
  cluster fingerprint shows the 7× hurting, swap in a manual bucketize+scatter-add at ~1.3×shard).
- **`multi_threshold_otsu(valid_mask=None)`:** module-aware — numpy input → host `np.histogram` as
  before; jax input → `_sharded_histogram` on its own device(s); only the (num_bins,) histogram comes to
  the host for the threshold search.
- **`segment_plastic_metal(valid_mask, num_real_slices)`:** class masks are ANDed with `valid_mask` (a
  threshold interval spanning 0 would otherwise mark the padded zeros and bias
  `compute_scaling_factor`'s denominator); `apply_cylindrical_mask(num_real_slices=...)` applies the
  bottom margin at the REAL bottom slices (`[-b:]` would have zeroed the padding instead — latent bug
  under slice padding, fixed).
- **`_est_plastic_metal_sinos_from_recon`:** `recon = ct_model._shard_recon(recon)` at entry (no-op if
  already sharded) + builds the tiny per-slice `valid_mask` from `recon_placement` — segmentation runs
  on-device and the 1+m projections consume ONE sharded recon instead of re-uploading per mask.
- **`recon_plastic_metal(..., output_sharded=False)`:** user-facing sharding contract.  Loop internals:
  `direct_recon(..., output_sharded=True)`; non-cone `recon_function = partial(recon,
  output_sharded=True)`; **`split_sino_recon` stays host-output by design** (host-split memory halving;
  device-side stitching noted as a follow-up) — both forms converge at the correction's `_shard_recon`
  entry and the exit adapter (`_shard_recon`/`_gather_recon` per the kwarg, each a no-op when already in
  form).
- **Verified (CPU, n=1 vs n=3 with BOTH view and slice padding):** Otsu thresholds **bit-identical**
  (host numpy vs sharded+padded); corrected sino consistent (max 9.9e-4 = argmin-ULP scale, NRMSE
  1.1e-5); `recon_plastic_metal` default returns host ndarray, `output_sharded=True` returns the
  sharded form, gathered == host **exactly (0.0)** (also implicitly verifies `recon(init_recon=
  <sharded>)`); `tests/test_preprocessing.py` + `tests/test_utilities.py` pass (incl. the existing
  cylindrical-mask tests).

### Phase 2c — Otsu threshold search: recursion → vectorized DP  **[DONE 2026-07-02]**
`_recursive_otsu` + `_binary_threshold_otsu` (per-bin Python loops; ~267 ms at B=1024, k=2; another
factor of B for k=3) replaced by `_otsu_thresholds_dp`: the within-class-variance objective is
separable over classes, so the optimum solves the classic 1-D segmentation DP — O(1) interval costs
from centered moment prefix sums (float64; second moment ~1e15 for 1e9 voxels), one vectorized
(B+1)² min-reduction per class, argmin backtrack.  **7.8 ms (34×)**, no recursion, host NumPy only
(jit deliberately NOT used: 1024-bin host problem, float64 safest; a jnp variant is mechanical if
segmentation is ever fused on-device).
- **Verification (pre-swap, parallel implementations):** 90 random-histogram trials + a MAR-like
  phantom hist, refereed by the old scoring function: DP never worse; ALL divergences explained by an
  off-by-one **in the old code** (`_binary_threshold_otsu` reported inclusive indices, but the scorer
  and the recursive outer split used half-open boundaries — inner thresholds were optimized under one
  convention and scored under the other).  DP uses the boundary convention coherently.
- **Threshold values now `bin_edges[t]`** (was `bin_centers[t]`): under the boundary convention the
  left edge is the exact cut (values below it fall precisely in the lower classes).  Half-bin-scale
  value change.
- Old functions + the scorer **deleted** (Greg); docs/build HTML still shows them until the next docs
  build (generated artifacts).
- **Gate:** MAR phantom — thresholds bit-identical host-vs-sharded (n=1, n=3); corrected sino
  **byte-identical** to the pre-swap output (the threshold move landed inside an empty inter-mode
  valley — a tie plateau with zero downstream effect); tests pass.  On real data expect ~bin-scale
  threshold shifts with negligible segmentation change (cluster before/after covers it).

### Phase 2d — Full-scale trace fixes (2026-07-02, from Greg's cluster trace)
Found and fixed while tracing the full-size (1800, 1365, 1880) MAR recon:
- **`u_m` flat-index bug** (`_estimate_BH_model_params`): a flat pixel index applied to the now-3-D
  sinogram grabbed a whole VIEW (TypeError at the constraint concatenate); fixed via
  `reshape(-1)[i]`; the constraint branches are now exercised by a forced-`tolerance` check (the
  original gate never entered them).
- **`split_sino_recon` sharded `init_recon`**: slicing a slice-sharded volume along its sharded axis
  REPLICATES the half onto every device, and on a padded volume the bottom-half slice picked up
  padding zeros (correctness).  Fixed by gathering `init_recon` to host at entry (`_gather_recon`,
  same treatment as sino/weights; preserves the memory-halving design).  Verified byte-identical
  host-vs-sharded init with the VCD partition RNG seeded.
- **cuSolver `jnp.linalg.solve` → host `np.linalg.solve`** (tiny QP); **`jnp.linalg.norm` OOM** →
  per-iteration VCD stats fused into one jitted `_vcd_iteration_stats`; `gen_huber_weights` body
  jitted.  Root cause class: out-of-pool allocations (cuSolver workspaces, NCCL collective buffers)
  starved by the retained BFC pool; default mem fraction now `setdefault('0.94')` (was hard-set 0.98).
  Confirmed on cluster: `pool_bytes` 83.3 GB at `bytes_in_use` 0.64 GB; 0.94 clears the OOM.
- **Otsu histogram at full scale (47 GiB OOM)**: GSPMD does not partition scatter — `jnp.histogram`
  AND a global `.at[idx].add` both all-gather image-sized arrays.  Final design (after Greg review;
  the interim `shard_map` version was dropped per the fbp SPMD-lowering precedent, and the interim
  volume-subsampling was dropped because the metal class can be sparse): `_sharded_histogram` runs a
  **per-device local** bucketize+scatter on each device's own shard block (`addressable_shards`,
  deduped by index), in **≤2^28-element slabs** (int32 exact within a slab; bounds temps), with the
  tiny partials combined **on the host in int64** — exact counts at any scale, no jax x64, and zero
  cross-device collectives (min/max also host-combined; slabs all dispatched before any read so
  devices overlap).  Also fixed on the way: `np.histogram` edges depend on the `range` dtype — all
  paths now pass python-float endpoints, so the numpy-plain, numpy-masked, and sharded paths produce
  IDENTICAL thresholds on identical data (verified).  Gate: counts+edges == `np.histogram` (int64),
  padded==unpadded exact, replicated arrays not double-counted, multi-slab == single-slab, MAR gate
  byte-unchanged.

### Phase 2e — Flat-index >2^31 hazard in the constraint machinery (2026-07-03)
The "int64 ... truncated to int32" UserWarning in `correct_sino_plastic_metal` flagged a REAL silent
bug at full scale, introduced by the sharding refactor's flat-index reads: (1) `lax.argmin` computes
its index labels in int32 (x64 off), so a flat argmin over a >2^31-element sinogram WRAPS — verified:
a minimum planted at flat 2.3e9 returned −1,994,967,296 (exactly 2^32 off) with NO warning; (2) any
scalar read on a >2^31-long flat axis requests int64 indices (`int_dtype_for_dim`), truncated with the
observed warning.  A 4.62e9-pixel sino puts ~53% of positions past 2^31.  Fix: **no flat indices** —
`_argmin_3d` stages the argmin per axis (per-view argmin over the R*C plane, then argmin over the
per-view minima; identical row-major tie-breaking; also returns the min value, eliminating the flat
read) and `_find_most_violated_constraints` now returns **(view, row, col) tuples**; `_get_row_H`,
`u_m`, and `C_p`/`C_m` use the tuples via basic per-axis indexing (every axis << 2^31 → int32-safe).
Gates: `_argmin_3d` == flat argmin over 200 trials incl. crafted ties; phantom corrected sino AND
forced-constraint theta byte-identical; repo audit — these were the only `jnp.argmin/argmax` sites.

### Phase 3 — (DEFERRED → moved to `current_plans.md`) Subsample / speed up the BH model fit
- The OSQP fit is statistical, so it *could* be estimated from a subsample — the analog of
  `est_crop_width` / `detect_zinger_pixels`.  **But a uniform view/stride subsample is wrong here:** the
  BH model is identifiable only from sinogram pixels spanning MULTIPLE levels of metal exposure (varying
  metal path length), and those are **sparse** in a mostly-plastic object with small metal parts.  A
  uniform subsample would be dominated by plastic-only / zero-metal pixels and under-determine the metal
  terms.
- So this needs **metal-thresholded targeted subsampling** — select pixels by their estimated metal path
  length (from `metal_sino_est`) to cover the exposure range (e.g. stratify by metal magnitude / keep
  pixels above a metal threshold plus a plastic sample).  That is a more involved refactor than the other
  subsamplers, hence deferred.
- Independent, cheaper win to fold in here: cache each `H` column once (`_get_column_H`) instead of the
  O(num_cols²) recomputation in `_compute_entry_for_OSQP`.
- **A/B test the estimation IN ISOLATION first:** as a standalone experiment, compare the fitted `theta`
  (and the resulting corrected sino) from the full fit vs the targeted-subsample fit — *before* wiring
  the subsampling into the recon loop.  Sweep the subsample size/strategy to pick the knee.  NOT
  byte-identical by design; gate on the corrected recon within a documented tolerance.

---

## Validation strategy
- **Ephemeral real-data before/after** on a MAR recon (`recon_plastic_metal`, `num_metal>0`), same
  platform — capture the current corrected sino + recon, verify after each phase (Phase 1 ~byte-identical;
  Phase 2 ~byte-identical; Phase 3 tolerance).  Discard after, like the other baselines.
- Watch peak memory (`memory_report`) on a large run — Phase 1 removes the full-volume single-GPU gather;
  Phase 2 bounds the application memory.
- Phase 3's estimation A/B is a standalone experiment (`theta` full-vs-subsample), not a recon gate until
  the sampling strategy is dialed in.

## Open questions (all resolved during Phases 1–2d)
- `forward_project` sharding of a single-device-committed input — **resolved:** verified byte-identical to a
  host/sharded input on a 4-device model (Phase 1); the old `device_put` was a pure gather.
- `segment_plastic_metal` host-vs-device masks — **resolved:** segmentation now runs on-device (Phase 2b),
  masks stay sharded, `mask * recon` never gathers to one device.
- `gen_huber_weights` / `BH_correction` standalone-caller contracts — **resolved** during the Phase-2/2d
  work (kernels jitted; `gen_huber_weights` body jitted).
