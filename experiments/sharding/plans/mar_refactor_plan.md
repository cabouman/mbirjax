# MAR (`preprocess/mar.py`) refactor — device management, sharding, memory

**Status:** PLAN (created 2026-06-29).  Scope = `mbirjax/preprocess/mar.py` (metal-artifact reduction +
beam-hardening correction).  The other half of status task #18 (the scan→sino preprocessing half is done).

**Context.** The MAR loop `recon_plastic_metal` alternates `correct_sino_plastic_metal` (beam-hardening
correction driven by a segmented recon) with `recon`/`split_sino_recon` (already sharded — leave alone).
The issues are all in the **correction** path.

---

## Current issues

1. **Retired single-device pattern (correctness + OOM).** `correct_sino_plastic_metal` does
   `device = ct_model.recon_placement.devices[0]`, and `_est_plastic_metal_sinos_from_recon` then does
   `ct_model.forward_project(jax.device_put(mask * recon, device))` for the plastic mask and each metal
   mask.  `jax.device_put(mask * recon, device)` forces a **full recon-sized volume onto one GPU** (the
   single-device-gather footgun we removed elsewhere) and defeats `forward_project`'s internal sharding.

2. **BH model fit runs on the full flattened sinogram (memory + no sharding).**
   `_est_plastic_metal_sinos_from_recon` returns `forward_project(...).reshape(-1)` — **full-sino-length**
   vectors (`p`, `m_0`, …) — and the measured sino is flattened too.  `_compute_entry_for_OSQP` builds
   full-length H columns (`_get_column_H`) and `jnp.dot`s them on one device, **recomputing each column
   O(num_cols²) times**.  Several full-sino-sized vectors + the dots on one device → single-device memory
   pressure and slow.  But the fit is **statistical** (a handful of polynomial coefficients via least
   squares), so it does not need every pixel.

3. **Correction application on the full sino** (`_correct_plastic_sinogram`, `BH_correction` — batched
   at 64) is per-pixel over the whole sinogram → a candidate for the shared `map_view_batches` driver
   (memory-bounded + multi-GPU), lower priority than 1–2.

The recon steps in the loop (`split_sino_recon` / `recon`) are already sharded — out of scope.

---

## Staged plan

### Phase 1 — Retire the single-device pattern (correctness + OOM; byte-identical)
- Drop the `device` argument from `_est_plastic_metal_sinos_from_recon` and the
  `device = ...recon_placement.devices[0]` in `correct_sino_plastic_metal`.  Hand `forward_project` a
  host (or sharded) `mask * recon` and let it shard internally (the host-safe pattern used everywhere
  else).  `mask`/`recon` arithmetic stays host or on-device per the inputs; nothing is forced onto one
  GPU.
- **Byte-identical**: `device_put(x, d)` vs letting `forward_project` shard produces the same values
  (float32, just placed differently).
- **Gate:** ephemeral before/after on a MAR recon (Lilly, `num_metal>0`) — sino + recon byte-identical;
  no full-volume single-GPU spike (a `memory_report` check on a large run).

### Phase 2 — Subsample the BH model fit (the big memory/speed win)
- The OSQP fit is statistical, so estimate `HtH`/`Hty` (and the constraint search) from a sinogram
  **subsample** — the exact analog of `est_crop_width` / `detect_zinger_pixels`.  Subsample on a flat
  index stride (or by view) so `p`/`m_i`/`measured` and the H columns are small; apply the resulting
  `theta` correction to the **full** sino (the application is per-pixel and must stay full).
- Bonus: compute each H column **once** (cache), instead of O(num_cols²) recomputation.
- **NOT byte-identical** (a subsampled least-squares fit differs slightly).  **Gate:** the corrected
  recon matches the full-fit recon within a documented tolerance on the Lilly MAR run (the fit is a
  smooth low-order model, so a representative subsample should be very close); sweep the subsample size
  to pick the knee.

### Phase 3 — (optional) Drive the correction application through `map_view_batches`
- Run `_correct_plastic_sinogram` / `BH_correction` per view-batch through the shared driver for
  memory-bounded + multi-GPU application on the full sino.  Per-pixel → byte-identical (eager kernels).
- **Gate:** before/after byte-identical; 1-vs-N device byte-identical.

---

## Validation strategy
- **Ephemeral real-data before/after** on a Lilly MAR recon (`recon_plastic_metal`, `num_metal>0`),
  same platform — capture the current corrected sino + recon, then verify after each phase (Phase 1
  byte-identical; Phase 2 tolerance; Phase 3 byte-identical).  Discard after, like the other baselines.
- Watch peak memory on a large run (`memory_report`) — Phase 1 removes the full-volume single-GPU spike;
  Phase 2 removes the full-sino fit vectors.

## Open questions
- `gen_huber_weights` / `BH_correction` standalone callers — confirm their contracts before touching.
- Whether `segment_plastic_metal` (in `segmentation.py`) returns host or device masks — affects where
  `mask * recon` lives in Phase 1.
