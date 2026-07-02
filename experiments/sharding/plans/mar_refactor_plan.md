# MAR (`preprocess/mar.py`) refactor — device management, sharding, memory

**Status:** PLAN (created 2026-06-29; revised 2026-07-01 — phases reordered: the correction-application
sharding is promoted and done with jitted operators; the BH-fit subsampling is deferred to the end,
because it needs metal-exposure-diverse sampling, not a uniform subsample).  Scope =
`mbirjax/preprocess/mar.py` (metal-artifact reduction + beam-hardening correction).  Other half of status
task #18 (the scan→sino preprocessing half is done).

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

### Phase 3 — (DEFERRED — revisit after Phases 1–2) Subsample / speed up the BH model fit
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

## Open questions
- Does `forward_project` shard a single-device-committed input the same as a host/sharded one (Phase 1
  byte-identicality)?  Confirm on the before/after.
- `segment_plastic_metal` (in `segmentation.py`): does it return host or device masks? — affects where
  `mask * recon` lives in Phase 1 (want it sharded/host, not gathered to one device).
- `gen_huber_weights` / `BH_correction` standalone callers — confirm contracts before touching (Phase 2).
