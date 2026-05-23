# Multi-GPU Sharding: Status and Implementation Plan
*Last updated: 2026-05-23*

---

## Overview

This document records completed work and the forward implementation plan for sharding the
mbirjax reconstruction pipeline across multiple devices (multi-GPU, multi-CPU, or Apple
Silicon multi-core via virtual CPU devices).

The plan has gone through two major design iterations:

**Original plan (Phases 1–5):** Used `shard_map` + `lax.ppermute` for all multi-device
parallelism, including the VCD inner loop.

**Current plan (Steps 0–6):** Uses Python threading with `addressable_shards` (input) and
`make_array_from_single_device_arrays` (output) — "Path G" from the comparison experiments.
This supersedes the shard_map approach after benchmarking revealed that shard_map on GPU
has both correctness issues (diff >> float32 noise at 2 devices) and SPMD compiler overhead.
See `experiments/sharding/fbp_filter_parallelism_comparison.{py,md}` for full analysis.

---

## Architecture: why threading over shard_map

| | shard_map | Path G threading |
|---|---|---|
| GPU correctness | ✗ diff=1.1e-02 at 2-dev | ✓ diff=0.0 |
| GPU speedup (2-dev, size=1024) | 1.54× (but wrong) | 1.89× |
| SPMD overhead | Non-trivial at small sizes | None |
| PCIe traffic | Zero | Zero (Path G) |
| Portability | GPU-specific issues | Same code on GPU/CPU/Metal |

The threading approach keeps each device's data resident throughout the VCD loop.
Because sinogram det-rows and recon slices are in 1-to-1 correspondence in parallel beam,
all operations are embarrassingly parallel along this axis except for a 1-slice halo
exchange at qggmrf boundaries. The halo is ~1 MB per device per VCD iteration — negligible
compared to the compute.

### Consistent-sharding invariant

Once data is sharded at entry, it stays sharded throughout:

| Object | Shape | Sharded axis |
|--------|-------|-------------|
| sinogram | `(views, det_rows, channels)` | det_rows (axis 1) |
| recon / flat_recon | `(rows, cols, slices)` / `(rows×cols, slices)` | slices (axis 2 / 1) |
| weights | `(views, det_rows, channels)` | det_rows (axis 1) |
| error_sinogram | same as sinogram | det_rows (axis 1) |

**No scatter/gather between operations.** The only inter-device data movement in the
entire VCD loop is the 1-slice halo exchange before each qggmrf call. Public API methods
shard at entry and gather at exit; all internal methods receive already-sharded data.

### Cone beam extensibility

Cone beam sharding will require different axes (or a different sharding strategy entirely,
since cone beam projections are not embarrassingly parallel by slice). The design uses
geometry-specific hook methods overridden per subclass, so the base class VCD loop is
geometry-agnostic. See Step 0b.

---

## Completed work

### Phase 1 — Projector allocation fixes (`parallel_beam.py`) ✓

Two 1-line fixes so inner JIT'd projectors derive allocation shapes from input array
dimensions rather than from `projector_params`:

```python
# _sparse_forward_project_view:
sinogram_view = jnp.zeros((voxel_values.shape[1], num_det_channels))

# _back_project_one_view_to_pixel_batch:
det_voxel_cylinder = jnp.zeros((num_pixels, sinogram_view.shape[0]))
```

Zero behavioral change on single-device. Verified against baseline.

**Still needed (Step 2 below):** The outer Python-loop projectors still have hardcoded
shape dependencies: `sparse_forward_project` uses `sinogram_shape[1:]` and
`sparse_back_project` uses `recon_shape[2]`. These must be fixed before sharded
projectors will work, and the docstrings must note that sharded callers pass local shapes.

### Phase 2 — qggmrf halo-aware prior (`qggmrf.py`) ✓

`qggmrf_grad_and_hessian_per_cylinder` accepts explicit `left_val` / `right_val` scalars
(one per cylinder, vmapped). When halos are `None`, mirrors the boundary slice —
identical to prior behavior (reflected BC). When halos are provided, uses actual
neighboring-shard slice values.

```python
def qggmrf_gradient_and_hessian_at_indices(..., left_halo=None, right_halo=None):
    lh_vals = (left_halo[pixel_indices]  if left_halo  is not None
               else flat_recon[pixel_indices, 0])
    rh_vals = (right_halo[pixel_indices] if right_halo is not None
               else flat_recon[pixel_indices, -1])
    cylinder_map = jax.vmap(qggmrf_grad_and_hessian_per_cylinder, in_axes=(0, None, 0, 0))
    gradient, hessian = cylinder_map(flat_recon[pixel_indices], qggmrf_params, lh_vals, rh_vals)
```

Verified: halo=None vs baseline diff=0.0; explicit mirror halos diff=0.0; interior shard
self-consistency diff=0.0.

### Phase 4 — Device management helpers (`tomography_model.py`) ✓

All raw `jax.device_put` / `jax.default_device` / `jnp.zeros(..., device=...)` calls
replaced with helpers that become no-ops when `device is None`:

```python
def _device_put(self, x, device):
    return x if device is None else jax.device_put(x, device)

@contextlib.contextmanager
def _device_context(self, device):
    if device is None: yield
    else:
        with jax.default_device(device): yield
```

Key: `sinogram_device != main_device` evaluates `False` when both are `None`, so the
CPU-to-GPU transfer branch in `vcd_subset_updater` is never taken in sharded mode.

### Phase 1.5 (unplanned) — FBP sharding infrastructure ✓ (partially — see Step 1)

Added `configure_sharding(devices)`, `_maybe_shard(x, axis)`, `_maybe_gather(x)` to
`tomography_model.py`. Added multi-device path to `fbp_filter` in `parallel_beam.py`
using Python threading.

**Current state of fbp_filter multi-device path:** Path E (numpy roundtrips at both ends:
`np.asarray(sinogram)` to scatter, `np.asarray(out)` per device then `jnp.array(concat)`
to gather). This is ~4× the sinogram in PCIe traffic and does not meet the zero-PCIe
target. **Step 1 upgrades this to Path G.**

**`_maybe_gather` bug:** Was `jax.jit(lambda a: a)(x)` — does NOT gather a sharded array.
**Fixed in Step 0a** to `jnp.array(np.asarray(x))`.

---

## Critical JAX bug: `device_put` corrupts non-default GPUs

`jax.device_put(jax_array, NamedSharding)` silently produces garbage on non-default GPUs
in JAX 0.10.0. All three scatter paths are affected (eager device_put, jit with
out_shardings, with_sharding_constraint). Reading from non-default GPUs is fine.

**Workaround:** `jax.device_put(np.asarray(x), sharding)` — materialize to numpy first.
A TODO marks this in `_maybe_shard` for removal once JAX PR #36524 lands.

**Subtle masking:** If both a sharded array and an uncommitted array feeding into an
arithmetic op are corrupted, the errors may cancel and return 0.0. Always diagnose with
`np.asarray(y)` then compare on the Python side.

- Issue: https://github.com/jax-ml/jax/issues/36308
- PR: https://github.com/jax-ml/jax/pull/36524 (open as of 2026-05-23)

---

## Forward plan

### Design principle: output sharding follows the output array type's native sharding

Each geometry defines a **native sharding** for each of its array types (sinogram,
recon). In parallel beam this is the slice/det-row axis; in cone beam it may be something
different. The rule for public API methods is:

- **The output is always in the native sharding for the output array type** — not derived
  from the input sharding. For `forward_project`, the output sinogram is in sinogram
  native sharding. For `back_project`, the output recon is in recon native sharding.
  For parallel beam these happen to live on the same physical axis and same devices
  (slice ↔ det-row), but that is a parallel-beam coincidence, not a general contract.
  Cone beam may have a sinogram native sharding and a recon native sharding that are
  completely different (different axes, potentially different device distributions).

- **Whether the output is returned sharded or gathered** depends on whether the input
  arrived sharded:
  - Input is already a NamedSharding array → output is returned in its native sharding
    (sharded, zero PCIe for pipelined callers)
  - Input is a plain/uncommitted array → shard at entry, compute, **gather at exit**
    (plain array returned, backward-compatible behaviour)

  The single check `isinstance(getattr(x, 'sharding', None), jax.sharding.NamedSharding)`
  on the primary input array drives this decision.

- **Reconstruction methods** (`direct_recon`, `vcd_recon`) always gather at exit by
  default (the final result is nearly always consumed as a plain array), with an optional
  `return_sharded=True` parameter for callers that want to stay on-device.

- **Internal methods** (`sparse_forward_project`, `sparse_back_project` when called from
  within the VCD loop) receive already-sharded data and return sharded data. No
  scatter/gather inside. Their docstrings must state this expectation.
  When called directly by user code in sharded mode, the caller is responsible for
  providing appropriately sharded inputs.

Summary table:

| Method | Primary input | Input sharded? | Output type | Output sharding |
|--------|--------------|----------------|-------------|-----------------|
| `fbp_filter` | sinogram | yes | filtered sinogram | sino native sharding |
| `fbp_filter` | sinogram | no | filtered sinogram | gathered (plain) |
| `forward_project` | recon | yes | sinogram | sino native sharding |
| `forward_project` | recon | no | sinogram | gathered (plain) |
| `back_project` | sinogram | yes | recon | recon native sharding |
| `back_project` | sinogram | no | recon | gathered (plain) |
| `direct_recon` | sinogram | any | recon | gathered (plain); `return_sharded=True` opt-in |
| `vcd_recon` | sinogram | any | recon | gathered (plain); `return_sharded=True` opt-in |
| `sparse_forward_project` | recon shard | sharded (VCD) | sino shard | sino native sharding |
| `sparse_back_project` | sino shard | sharded (VCD) | recon shard | recon native sharding |

---

### Step 0 — Infrastructure fixes (prerequisites; do once before Steps 1–6) ✓ COMPLETE

#### 0a. Fix `_maybe_gather` bug

**File:** `tomography_model.py`

**Change:** Replace `jax.jit(lambda a: a)(x)` with `jnp.array(np.asarray(x))`.

`np.asarray(x)` reads all device shards into a contiguous host array (reliable read path,
unaffected by the JAX scatter bug). `jnp.array(...)` wraps it as an uncommitted JAX array
not pinned to any device, suitable for returning to the user.

**Test:** Create a 2-shard NamedSharding array. Call `_maybe_gather`. Verify the result
has no `sharding` attribute (or is uncommitted), has the correct global shape, and matches
the original data assembled via `np.asarray`.

#### 0b. Add geometry-specific hook methods

**File:** `tomography_model.py` (base class) and `parallel_beam.py` (ParallelBeamModel)

Add five hook methods to `TomographyModel` with `raise NotImplementedError` defaults
(or return `x` unchanged / `(None, None)` when `self.mesh is None`):

```python
def _shard_sinogram(self, sino):
    """Shard sinogram for multi-device compute. Override per geometry."""

def _gather_sinogram(self, sino):
    """Gather sharded sinogram to uncommitted single array. Override per geometry."""

def _shard_recon(self, recon):
    """Shard recon (3D or flat 2D) for multi-device compute. Override per geometry."""

def _gather_recon(self, recon):
    """Gather sharded recon to uncommitted single array. Override per geometry."""

def _extract_halos(self, flat_recon):
    """Return (left_halos_list, right_halos_list) for qggmrf across shard boundaries.
    Each list has length n_devices; boundary entries are None (reflected BC applies).
    Override per geometry."""
```

Implement in `ParallelBeamModel`:

```python
def _shard_sinogram(self, sino):
    return self._maybe_shard(sino, axis=1)    # det_rows axis

def _shard_recon(self, recon):
    axis = 1 if recon.ndim == 2 else 2       # flat_recon vs 3D recon
    return self._maybe_shard(recon, axis=axis)

def _gather_sinogram(self, sino):
    return self._maybe_gather(sino)

def _gather_recon(self, recon):
    return self._maybe_gather(recon)

def _extract_halos(self, flat_recon):
    # Extract one boundary slice per device — O(rows²) data, negligible cost
    shards = sorted(flat_recon.addressable_shards,
                    key=lambda s: s.index[1].start)
    left_halos  = [None] + [np.asarray(s.data[:, 0])  for s in shards[:-1]]
    right_halos = [np.asarray(s.data[:, -1]) for s in shards[1:]] + [None]
    return left_halos, right_halos
```

**Native sharding contract:** Each hook defines the canonical on-device format for its
array type in the current geometry. `_shard_sinogram` applied to an already-correct
NamedSharding array must be a no-op. `_gather_sinogram` must invert it. Public API
methods rely on this contract to decide whether to gather at exit: they inspect the
input, shard if needed, compute, and then gather only if the original input was not
already sharded.

**Future cone beam override:** The sinogram native sharding and recon native sharding
for cone beam may live on entirely different axes and may not correspond to the same
physical dimension or the same device assignment. For example, sinogram might be sharded
along views (axis 0) while recon is sharded along rows (axis 0) — with no geometric
1-to-1 mapping. The threading strategy for `forward_project` and `back_project` in cone
beam would need to handle this (potentially requiring inter-device communication), but
that logic lives entirely inside the `ConeBeamModel` overrides. Zero changes to the base
class or to Steps 1–6 are required.

**Test:** Instantiate `ParallelBeamModel`, call each hook with and without sharding
configured. Verify `_shard_sinogram` returns a NamedSharding array in sharded mode and
the unchanged input in single-device mode. Verify `_extract_halos` returns correct
boundary slices matching `np.asarray(flat_recon)[:, i_boundary]`.

#### 0c. Integrate sharding into `set_devices_and_batch_sizes`

**File:** `tomography_model.py`

**Motivation:** `set_devices_and_batch_sizes` is already the single place where device
configuration is determined from memory budgets. Multi-GPU is a fourth mode in the same
decision tree.

**Add `'sharded'` to the `use_gpu` param** (alongside `'automatic'`, `'full'`,
`'sinograms'`, `'none'`). Name is device-agnostic (works for GPU, CPU, or Metal).

**Memory arithmetic:** With N devices and slice sharding, each device holds `1/N` of the
sinogram and recon. Per-device footprint ≈ `mem_for_all_vcd / N`. Trigger condition:
`mem_for_all_vcd / N < gpu_memory_to_use`, i.e., the problem fits in combined GPU memory.

**Extended decision tree:**

```
N = len(gpus) (or len of explicit device list)
if N > 1 and use_gpu in ['automatic', 'sharded']
        and mem_for_all_vcd / N < gpu_memory_to_use:
    → 'sharded' mode
elif mem_for_all_vcd < gpu_memory_to_use:
    → 'full'   (single GPU, existing logic)
elif mem_for_minimal_sinos < gpu_memory_to_use:
    → 'sinograms'  (existing logic)
else:
    → 'none'   (existing logic)
```

**Avoid circular calls:** Extract a private `_apply_sharding(devices)` that does the
low-level setup (sets `self.mesh`, nulls `main_device`/`sinogram_device`, calls
`_configure_geometry_sharding()` hook). `set_devices_and_batch_sizes` calls
`_apply_sharding`; `configure_sharding` calls `set_devices_and_batch_sizes`. Never the
reverse. `configure_sharding(devices)` stores `self._explicit_shard_devices = devices`
and calls `set_devices_and_batch_sizes()`; that method checks for an explicit request
first.

**Geometry change warning:** If `set_params(geometry_change)` triggers
`set_devices_and_batch_sizes` while `self.mesh is not None` and the new per-device
footprint no longer fits, issue a `warnings.warn` and switch to an appropriate
single-device mode. This prevents silent mode degradation.

**Batch size notes in sharded mode:**
- `view_batch_size_for_vmap`: unchanged (each device sees all views).
- `transfer_pixel_batch_size`: less relevant in Path G (no pixel transfers in VCD), but
  retained for single-device backward compatibility. Add a docstring note.

**Test:**
- No GPUs: `set_devices_and_batch_sizes()` selects `'none'`; `configure_sharding(cpu_devices)` selects `'sharded'` with virtual CPUs.
- 1 GPU, problem fits: `'full'`.
- 1 GPU, tight memory: `'sinograms'`.
- 2 GPUs, problem fits per device: `'sharded'`; verify `self.mesh` is set,
  `main_device is None`, `sinogram_device is None`.
- 2 GPUs, problem too large even split: falls back to single-GPU mode with a logged note.
- Geometry change in sharded mode that breaks per-device fit: issues `warnings.warn`.
- `configure_sharding(None)` then `set_devices_and_batch_sizes()`: mesh cleared, single-device restored with no infinite recursion.

#### 0d. Docstring updates for `sparse_forward_project` and `sparse_back_project`

**File:** `tomography_model.py`

Add a note to both docstrings:

> When called from within a sharded VCD loop, `sinogram` and `flat_recon` are expected
> to be NamedSharding JAX arrays with their slice axis already distributed across mesh
> devices. Each device operates on its local shard independently. If called directly
> by user code in sharded mode, the caller is responsible for providing appropriately
> sharded inputs; no automatic scatter/gather is performed.

---

### Step 1 — `fbp_filter`: Path E → Path G (zero PCIe)

**File:** `parallel_beam.py`

**Change:** Replace the current threading implementation (which does `np.asarray(sinogram)`
to scatter and `np.asarray(out)` + `jnp.array(concat)` to gather — Path E, ~4× sinogram
in PCIe) with Path G: read from `addressable_shards`, keep results on-device, assemble
with `make_array_from_single_device_arrays`.

The output follows the design principle: the filtered sinogram is always assembled in
sinogram native sharding. Whether it is then gathered before returning depends on whether
the input arrived sharded.

```python
def fbp_filter(self, sinogram, ...):
    input_is_sharded = isinstance(getattr(sinogram, 'sharding', None),
                                  jax.sharding.NamedSharding)
    sinogram = self._shard_sinogram(sinogram)   # no-op if already in native sharding

    if self.mesh is not None:
        n_devices = self.mesh.devices.size
        devices = list(self.mesh.devices.flat)
        dev_to_shard = {s.device: s.data for s in sinogram.addressable_shards}
        body_views = (num_views // view_batch_size) * view_batch_size
        tail_views = num_views - body_views
        filter_np = np.asarray(recon_filter)  # float32 numpy; fresh upload per device
        results = [None] * n_devices

        def _worker(i):
            dev = devices[i]
            with jax.default_device(dev):
                out = _apply_fbp_filter_to_shard(
                    dev_to_shard[dev], jnp.array(filter_np),
                    body_views=body_views, tail_views=tail_views,
                    view_batch_size=view_batch_size)
                out = out * float(jnp.pi / num_views)  # scale on-device
                jax.block_until_ready(out)
            results[i] = out                   # stays on device — zero PCIe

        with ThreadPoolExecutor(max_workers=n_devices) as ex:
            list(ex.map(_worker, range(n_devices)))

        filtered = jax.make_array_from_single_device_arrays(
            shape=sinogram.shape[:-1] + (filtered_channels,),
            sharding=sinogram.sharding,        # sinogram native sharding
            arrays=results)
    else:
        # single-device path (unchanged)
        ...

    return filtered if input_is_sharded else self._gather_sinogram(filtered)
```

Note: the `π/num_views` scale is applied on-device inside each worker in the multi-device
path; remove the separate `filtered_sinogram *= jnp.pi / num_views` line from that path.

**Test:**
- Single-device mode: behavior unchanged vs pre-patch.
- Sharded mode, n_dev=1 (virtual CPU or 1 GPU): result matches single-device to float32
  noise (diff < 1e-6).
- Sharded mode, n_dev=2: result matches single-device to float32 noise.
- Verify no `np.asarray(sinogram)` call inside `fbp_filter` when input is a NamedSharding
  array (instrument or grep).
- Timing (optional): confirm ~2× speedup at n_dev=2 on GPU, consistent with comparison
  experiments.

---

### Step 2 — `forward_project` / `sparse_forward_project`: Path G threading

**File:** `parallel_beam.py` (and `tomography_model.py` for outer projectors)

**Changes:**

1. **Fix outer shape dependencies** in `sparse_forward_project`: replace hardcoded use of
   `sinogram_shape[1:]` and `recon_shape[2]` with shapes derived from the actual array
   arguments. This parallels the Phase 1 fix to the inner kernel.

2. **Add Path G threading** to `forward_project` (public API):

   ```python
   def forward_project(self, recon, ...):
       input_is_sharded = isinstance(getattr(recon, 'sharding', None),
                                     jax.sharding.NamedSharding)
       recon = self._shard_recon(recon)            # ensure recon native sharding
       result = self._threaded_forward_project(recon)  # → sinogram native sharding
       return result if input_is_sharded else self._gather_sinogram(result)
   ```

   Each worker thread receives its local recon shard via `addressable_shards`, runs the
   projection for its slice range, and stores the local sinogram shard on-device. Output
   is assembled in sinogram native sharding via `make_array_from_single_device_arrays`.
   For parallel beam this is the same physical axis as the recon native sharding (det-row
   ↔ slice). For cone beam the subclass override handles the mapping.

3. **`sparse_forward_project` when called from VCD:** receives already-sharded `flat_recon`
   and `sinogram`. Add no entry/exit sharding here — add docstring note per Step 0d.

**Test:**
- Single-device mode: result matches pre-patch baseline (bit-identical).
- Sharded mode with sharded recon input (n_dev=2): returns sharded sinogram; result
  matches single-device to float32 noise.
- Sharded mode with plain recon input (n_dev=2): returns plain sinogram; result matches
  single-device to float32 noise.
- Outer shape fix: create a model with non-square geometry and verify `sparse_forward_project`
  produces correct output when called with a local shard of the recon.

---

### Step 3 — `back_project` / `sparse_back_project`: Path G threading

**File:** `parallel_beam.py` (and `tomography_model.py` for outer projectors)

**Changes:** Mirror of Step 2, reversed direction.

1. **Fix outer shape dependencies** in `sparse_back_project`.
2. **Add Path G threading** to `back_project` (public API):

   ```python
   def back_project(self, sinogram, ...):
       input_is_sharded = isinstance(getattr(sinogram, 'sharding', None),
                                     jax.sharding.NamedSharding)
       sinogram = self._shard_sinogram(sinogram)     # ensure sino native sharding
       result = self._threaded_back_project(sinogram)  # → recon native sharding
       return result if input_is_sharded else self._gather_recon(result)
   ```

3. **`sparse_back_project` from VCD:** no entry/exit sharding; docstring note.

**Test:**
- Single-device mode: bit-identical to pre-patch.
- Sharded mode with sharded sinogram input (n_dev=2): returns sharded recon; matches
  single-device to float32 noise.
- Sharded mode with plain sinogram input (n_dev=2): returns plain recon; matches
  single-device to float32 noise.
- **Round-trip test:** `back_project(forward_project(sharded_recon))` returns a sharded
  recon (no intermediate gather). Result matches the equivalent single-device round-trip
  to float32 noise. This confirms zero PCIe across both projectors in pipelined use.

---

### Step 4 — `direct_recon` (FBP path): end-to-end sharded

**Files:** `parallel_beam.py` (`direct_recon`), `tomography_model.py` (if any shared logic)

**Changes:**

`fbp_filter` (Step 1) and `back_project` (Step 3) each handle their own entry/exit
sharding logic. `direct_recon` therefore needs minimal changes: shard the sinogram once
at entry and pass it through. Since the sinogram arrives sharded, `fbp_filter` returns a
sharded filtered sinogram; since that is sharded, `back_project` returns a sharded recon.
`direct_recon` then gathers at exit (reconstruction methods always gather by default).

```python
def direct_recon(self, sinogram, ...):
    sinogram = self._shard_sinogram(sinogram)   # shard once at entry
    filtered = self.fbp_filter(sinogram, ...)   # sees sharded input → returns sharded
    recon    = self.back_project(filtered, ...)  # sees sharded input → returns sharded
    return self._gather_recon(recon)             # gather for user
    # future: honour return_sharded=True kwarg
```

No scatter/gather between `fbp_filter` and `back_project` — data stays on-device
throughout.

**Test:**
- Single-device mode: bit-identical to pre-patch (sharding hooks are no-ops).
- Sharded mode, n_dev=2: FBP reconstruction matches single-device to reconstruction
  tolerance (minor differences from floating-point reordering across devices are allowed).
- Use standard test geometry (shepp-logan or similar from the test suite).
- Verify the returned recon is a plain (ungathered) JAX array regardless of sharding mode.

---

### Step 5 — qggmrf: wire halo exchange into VCD loop

**File:** `tomography_model.py` (`vcd_subset_updater` or its caller)

**Changes:**

`qggmrf_gradient_and_hessian_at_indices` accepts `left_halo`/`right_halo` but
`vcd_subset_updater` does not yet extract or pass them. Add:

1. Add `left_halo` and `right_halo` parameters to `vcd_subset_updater`.
2. In the qggmrf call site inside `vcd_subset_updater`, pass them through.
3. In the VCD outer loop (in `vcd_recon`), before each call to `vcd_subset_updater`,
   call `left_halos, right_halos = self._extract_halos(flat_recon)` and select the
   per-device pair for this iteration.

The halo extraction pattern:
```python
# In vcd_recon, before calling vcd_subset_updater:
if self.mesh is not None:
    left_halos, right_halos = self._extract_halos(flat_recon)
    # left_halos[i], right_halos[i] are numpy arrays or None (boundary devices)
    # Pass to vcd_subset_updater for threading
else:
    left_halos, right_halos = [None], [None]
```

In single-device mode, both are `None` and the existing reflected-BC behavior is unchanged.

**Test:**
- Single-device mode: result identical to pre-patch (halos are None throughout).
- Sharded n_dev=2 vs single-device: qggmrf gradient/hessian at interior boundary voxels
  should match to float32 noise. Specifically: take a known `flat_recon`, compute
  gradient/hessian at the slice on the boundary between device 0 and device 1 using
  (a) single-device, (b) sharded with correct halos. Expect diff ≈ 0.
- Also test that boundary devices receive `None` halos and their reflected-BC results match
  single-device results at the physical boundary.

---

### Step 6 — VCD loop: wire entry/exit sharding

**File:** `tomography_model.py` (`vcd_recon`, `initialize_recon`)

**Changes:**

1. At entry to `vcd_recon` / `initialize_recon`:
   ```python
   sinogram    = self._shard_sinogram(sinogram)
   init_recon  = self._shard_recon(init_recon)
   weights     = self._shard_sinogram(weights) if weights is not None else None
   ```
   (Uses `_shard_sinogram` for weights since weights has the same shape as sinogram.)

2. At exit from `vcd_recon`:
   ```python
   recon = self._gather_recon(recon)
   ```

3. Verify (by inspection and test) that no scatter/gather occurs inside the VCD loop body
   — all `_device_put` calls are already no-ops in sharded mode from Phase 4.

4. The `sinogram_device != main_device` condition in `vcd_subset_updater` (which gates
   the CPU-to-GPU pixel transfer) evaluates `False` when both are `None` — correct
   behavior already ensured by Phase 4. Confirm no other implicit gather is triggered
   inside the loop.

**Test:**
- Single-device mode: reconstruction matches pre-patch baseline to reconstruction
  tolerance.
- Sharded n_dev=2: reconstruction matches single-device reconstruction to reconstruction
  tolerance (allow small differences from floating-point reordering).
- Convergence rate: sharded VCD should converge in the same number of iterations as
  single-device (the update equations are mathematically equivalent; any difference is
  floating-point noise at shard boundaries, not algorithmic).
- Use the existing test cases from `test_vcd.py` extended to run in both modes.
- Extend `test_fbp_fdk.py`, `test_projectors.py`, `test_qggmrf.py` similarly.

---

## Additional work (not blocking, but planned)

### Padding/cropping for divisibility

`num_slices` (= `num_det_rows` in parallel beam) must be divisible by `n_devices`.
Assert this in `_apply_sharding` with a clear error message. Optionally add a utility
that pads sinogram and recon to the next divisible size and crops output after reconstruction.

### Test suite additions

Each step has its own targeted tests (above). For the formal test suite:
- Modify `test_fbp_fdk.py`, `test_projectors.py`, `test_qggmrf.py`, `test_vcd.py` to
  run with `configure_sharding(virtual_cpu_devices)` when real GPUs are unavailable.
  Use `XLA_FLAGS=--xla_force_host_platform_device_count=2` for virtual 2-device CPU.
- Add a new `test_sharding.py` that specifically targets:
  - Halo correctness at shard boundaries.
  - Consistent-sharding invariant (no intermediate gathers in VCD).
  - Portability: same results on GPU, CPU, virtual CPU.

### `use_gpu` / `configure_sharding` user documentation

Update the user-facing docstrings and any tutorial notebooks to describe:
- `set_params(use_gpu='sharded')` for automatic multi-GPU when multiple GPUs are present.
- `model.configure_sharding(jax.devices('gpu'))` for explicit device selection.
- `model.configure_sharding(jax.devices('cpu')[:2])` for multi-CPU or testing.
- Note that `num_slices` must be divisible by `n_devices`.

### Future: cone beam sharding

Implement `_shard_sinogram`, `_shard_recon`, `_extract_halos` overrides in
`ConeBeamModel`. The base class VCD loop and Steps 1–6 infrastructure require zero
changes. The design question (which axes to shard, how large the halos need to be) is
deferred to a separate planning session.

---

## Constraints and portability

| Platform | Device discovery | `configure_sharding` call |
|----------|-----------------|--------------------------|
| Multi-GPU (Linux) | `jax.devices('gpu')` | `configure_sharding(jax.devices('gpu')[:n])` |
| Multi-CPU / testing | Virtual CPU via `XLA_FLAGS` | `configure_sharding(jax.devices('cpu')[:n])` |
| Apple M3 / Metal | Single Metal GPU | `configure_sharding(jax.devices('cpu')[:n])` for threading |

- `num_slices % n_devices == 0` is required. Assert in `_apply_sharding`.
- In `'sharded'` mode, `set_devices_and_batch_sizes` must issue `warnings.warn` if a
  subsequent geometry change causes the per-device footprint to exceed available memory,
  and must switch to an appropriate single-device mode automatically.
- All operations in Steps 1–6 use `jax.default_device` inside each thread, which is
  thread-local in JAX. No global device state is modified.
