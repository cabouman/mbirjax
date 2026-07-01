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

### Phase 1 — Retire the single-device gather (memory; byte-identical)  **[IN PROGRESS]**
- Drop the `device` param from `_est_plastic_metal_sinos_from_recon` and the
  `device = ct_model.recon_placement.devices[0]` in `correct_sino_plastic_metal`.  Hand `forward_project`
  the `plastic_mask` / `mask * recon` directly (host or sharded, as they arrive) and let it shard
  internally — the host-safe pattern used everywhere else.  Nothing is gathered to one GPU.
- **Byte-identical (expected):** `forward_project` shards its input the same way whether or not it was
  first gathered to one device (float32 — same projection, just no gather spike).  If the sharded
  reduction reorders vs a single-device one, expect ≤ ~1 ULP; document if so.
- **Gate:** ephemeral before/after on a MAR recon (`num_metal>0`) — corrected sino + recon match
  (byte-identical or ~ULP); `memory_report` shows no full-volume single-GPU spike on a large run.

### Phase 2 — Correction application through `map_view_batches` with jitted operators (memory + multi-GPU)
- Run `_correct_plastic_sinogram` / `BH_correction` per view-batch through the shared driver, with the
  per-batch polynomial evaluation wrapped in `jax.jit` (as `scan_to_sino`'s fused kernel is): fuses the
  monomial / `H`-column evaluation, reuses buffers, bounds memory, and shards across devices.  Per-pixel
  (no cross-view coupling) → byte-identical modulo jit-fusion ~ULP.  `theta` (a handful of coefficients)
  and the exponent lists are closed-over compile constants.  Based on the preprocessing work, jitting the
  per-batch operator is what keeps the peak bounded (buffer reuse) — Greg's call.
- **Gate:** before/after ~byte-identical; 1-vs-N device consistent; `memory_report` bounded.

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
