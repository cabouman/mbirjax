# Multi-GPU/CPU reconstruction (all geometries) + retirement cleanup + docs

This PR adds **automatic multi-device reconstruction for every geometry** — `ParallelBeamModel`,
`ConeBeamModel`, `TranslationModel`, `MultiAxisParallelModel` — and the `QGGMRFDenoiser`.  On a
machine with multiple GPUs (or, with no GPU, multiple CPU devices) a single reconstruction is spread
across them with **no change to existing scripts**.  It also retires the pre-sharding device
plumbing (placements are now the single source of device layout) and adds user + developer docs.

## Multi-device reconstruction

`recon()` (and `fbp_recon`/`fdk_recon`/`direct_recon`, `prox_map`, the projectors, and the QGGMRF
denoiser) divides the work across the available devices automatically.  Primary benefit is
**capacity** — peak memory per device drops ~1/N, so larger volumes fit; **speed** improves too
(near-linear at large sizes).  The result is **independent of the device count**: a non-dividing view
or slice axis is zero-padded to equal shares and the padding is kept exactly inert.

Sharding scheme (uniform across geometries): the **sinogram is sharded by view**, the **recon by
slice**, combined with a small amount of banded communication.  Execution uses a per-device thread
pool that keeps shards on-device (no host round-trips, no NCCL collectives in the projectors).

## Public surface

Controls (see the new `usr_multi_gpu` page):
- `model.configure_devices(...)` — explicit device choice (`n`, indices, or device objects); pinned
  across later parameter changes.  (Replaces `configure_sharding`.)
- `use_gpu` — `'automatic'` (default) / `'none'`; `'full'` is accepted as a deprecated synonym.
- `model.device_summary` — reports what was chosen, e.g. `'4 x GPU (sharded)'`.
- `model.prepare_sino_for_devices(sino[, weights])` — distribute a sinogram once for repeated recons.
- `output_sharded=True` on `recon`/`fbp_recon`/`fdk_recon`/`direct_recon`/`back_project`/
  `forward_project`/`denoise` — return the result device-sharded for on-device pipelines (default
  returns an ordinary single-device array, as before).

Data utilities (now sharding-aware):
- `generate_3d_shepp_logan_low_dynamic_range(shape, devices=..., max_block_gb=..., target_max_attenuation=...)`
  — optional **slice-sharded** build across `devices`; single-device build is `lax.map` row-blocked to
  bound memory; `target_max_attenuation` scales the phantom so its forward projection is realistic
  (≈0–8) regardless of array size.
- `generate_demo_data(..., devices=..., target_max_attenuation=...)` — returns device-sharded phantom
  + sinogram when `devices` is given.
- `gen_weights` — now an element-wise pass that **preserves the input's device/sharding** (a sharded
  sinogram yields sharded weights with no gather; see behavior note below).

## Architecture cleanup (the retirement cascade)

Placements (`recon_placement` / `sino_placement`) are the single source of device layout; the
pre-sharding representations and their dead branches are gone:
- Retired `main_device`/`sinogram_device`, the `is_sharded` property, the legacy no-mesh code path,
  the `_supports_sharding` gate, `output_device` (public projector surface), and the `mesh` property
  (kept `shard_devices` as the public device-list accessor).
- `view_indices` → internal `owned_view_indices`; `configure_sharding` → `configure_devices`.
- `recon_shard_axis` / `sinogram_shard_axis` are classmethods (overridable per geometry).

## Documentation

- New user page **`usr_multi_gpu`** (zero-effort path, device subsetting, efficiency tips, what to
  expect, a gentle "sharding" overview) + refreshed `use_gpu`, overview, advanced-features, and FAQ.
- New developer page **`dev_sharding_overview`** (the two shardings, placement, banded forward/back,
  why cone projects whole cylinders, the qGGMRF halos, single- vs multi-device paths, thread-pool
  execution) with the slide diagrams.
- `dev_api` rewritten as guidelines + a per-view-kernel skeleton (the existing geometry classes are
  the canonical reference); Tomography-Model "Device Configuration" section refreshed; prose
  `:meth:` cross-refs qualified so they resolve.

## Behavior changes relative to prerelease

- **Multi-device machines parallelize by default**, for every geometry.  Results agree with the
  single-device result within the usual float tolerance (~1e-4 for iterative recon).
- **CPU** shards across the available CPU devices by default.
- **`gen_weights`** now returns the weights on the **input's device/sharding** (it previously gathered
  to CPU).  Values are unchanged; only the device/sharding of the result differs.
- `use_gpu='sinograms'` (the old hybrid mode) is gone; `'full'` is now a deprecated synonym of
  `'automatic'`.
- Log files are under `~/.mbirjax/logs/`; the JIT cache under `~/.mbirjax/jax_cache` (only if you
  haven't configured your own).

## Validation

- Full test suite green on CPU multi-device (`pytest tests/` runs at 4 virtual CPU devices, which
  exercises the view/slice padding paths).
- Nightly regression on GPU (H100) and CPU, with correctness refs vs `main`, single-vs-multi-device,
  and cross-platform.
- Sharded recon matches single-device bit-for-bit where the math is reorder-free, and within the
  iterated-VCD float tolerance otherwise.

## Scope / known limits

- **Single process only** (no multi-node).
- **Open (under investigation, not a prerelease regression):** a cone-beam 2048³ reconstruction on 8
  GPUs hangs at the first VCD subset update (NCCL "Acquire clique" timeout); clean through 512³/8.
  Cone was single-device in prerelease, so this is new-capability territory, not a regression.  Repro
  tooling is included under `experiments/sharding/cone_deadlock_repro/`.
- **Follow-ons (post-merge):** shard `mar.py`/preprocessing (incl. `gen_weights_mar`); a doc-xref
  cleanup pass (autosummary/undocumented-target refs); revisit the automatic device-count basis
  (recon-slices vs sino-views).

## For testers

Run your existing scripts unchanged on 1/2/4/8 devices and compare results and times; try odd
view/slice counts; check `model.device_summary`.  Test suite: `pytest tests/` (~5–6 min on CPU,
auto 4 virtual CPU devices); override the device count with `MBIRJAX_NUM_CPU_DEVICES=N pytest
tests/sharding/`.
