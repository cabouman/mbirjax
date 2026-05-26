Let me check a few more things before writing the plan — `fm_hessian` computation and the VCD outer loop setup.

Good — now I have the full picture. Let me write out the complete incremental plan.

Now I have everything I need. Here's the full incremental plan:

---

## Full change plan: multi-GPU sharding along the slice axis

### Context: what gets sharded

All arrays shard along the slice axis:

| Array | Shape | Shard axis |
|-------|-------|-----------|
| `flat_recon` | `(num_pixels, num_slices)` | axis 1 |
| `error_sinogram` | `(num_views, num_det_rows, num_channels)` | axis 1 |
| `fm_hessian` | `(num_pixels, num_slices)` | axis 1 |
| `weights` (if array) | `(num_views, num_det_rows, num_channels)` | axis 1 |
| `prox_input` | `(num_pixels, num_slices)` | axis 1 |

`pixel_indices` (spatial) is replicated on all shards — same set of (row, col) locations on every device.

---

### Phase 1 — Projector: derive allocations from input shapes
**Files:** `mbirjax/parallel_beam.py` (2 lines)  
**Risk:** zero — shapes are identical for single-GPU case

```python
# forward_project_pixel_batch_to_one_view line 173:
# Before: sinogram_view = jnp.zeros((num_det_rows, num_det_channels))
sinogram_view = jnp.zeros((voxel_values.shape[1], num_det_channels))

# back_project_one_view_to_pixel_batch line 215:
# Before: det_voxel_cylinder = jnp.zeros((num_pixels, num_det_rows))
det_voxel_cylinder = jnp.zeros((num_pixels, sinogram_view.shape[0]))
```

`compute_proj_data` reads `num_det_rows` from `projector_params` but never uses it — no change needed there. `recon_shape[:2]` (spatial dims) is used for `unravel_index` and stays global regardless of sharding; same for `num_slices` in `qggmrf_grad_and_hessian_per_slice` which is extracted but also unused.

**Verify:** run an existing reconstruction example, compare `recon` output array element-by-element to a pre-saved baseline — should be bit-identical.

---

### Phase 2 — qggmrf: halo-aware cylinder prior
**Files:** `mbirjax/qggmrf.py`  
**Risk:** low — default `None` halos reproduce current behavior exactly

Modify `qggmrf_gradient_and_hessian_at_indices` to accept optional boundary slices:

```python
def qggmrf_gradient_and_hessian_at_indices(flat_recon, recon_shape, pixel_indices,
                                            qggmrf_params,
                                            left_halo=None, right_halo=None):
    b, sigma_x, p, q, T = qggmrf_params

    # Build cylinders: extend with halos if provided, otherwise mirror boundary (= reflected BC)
    local_cylinders = flat_recon[pixel_indices]          # (N_indices, local_slices)
    lh = left_halo[pixel_indices]  if left_halo  is not None else local_cylinders[:, :1]
    rh = right_halo[pixel_indices] if right_halo is not None else local_cylinders[:, -1:]
    extended = jnp.concatenate([lh, local_cylinders, rh], axis=1)  # (N_indices, local_slices+2)

    cylinder_map = jax.vmap(qggmrf_grad_and_hessian_per_cylinder, in_axes=(0, None))
    gradient, hessian = cylinder_map(extended, qggmrf_params)
    gradient, hessian = gradient[:, 1:-1], hessian[:, 1:-1]  # trim halo positions

    # In-slice neighbors: unchanged
    slice_map = jax.vmap(qggmrf_grad_and_hessian_per_slice, in_axes=(1, None, None, None, 1, 1), out_axes=1)
    gradient, hessian = slice_map(flat_recon, recon_shape, pixel_indices, qggmrf_params, gradient, hessian)
    return gradient, hessian
```

**Verify:** with `left_halo=None, right_halo=None` the mirrored boundary slices produce `delta[0]=0` and `delta[-1]=0`, exactly matching the current `jnp.concatenate((zero, jnp.diff(voxel_cylinder), zero))`. Output should again be bit-identical to baseline.

---

### Phase 3 — Sharding-simulation test (single GPU, sequential blocks)
**Files:** new `mbirjax/vcd_utils.py` helper (or test file)  
**Risk:** none — pure test harness, doesn't touch production code paths

Write a `simulate_sharded_vcd_step` that splits along axis 1, processes blocks sequentially (each block sees the correct halos from the most recent state of adjacent blocks), and reassembles:

```python
def simulate_sharded_vcd_step(updater_fn, flat_recon, error_sinogram, fm_hessian,
                               pixel_indices, n_blocks, ...):
    num_slices = flat_recon.shape[1]
    block_size = num_slices // n_blocks   # require divisibility
    for b in range(n_blocks):
        s, e = b * block_size, (b + 1) * block_size
        left_halo  = flat_recon[:, s-1:s] if s > 0 else flat_recon[:, :1]
        right_halo = flat_recon[:, e:e+1] if e < num_slices else flat_recon[:, -1:]
        local_recon, local_sino, ell1, alpha, ... = updater_fn(
            flat_recon[:, s:e], error_sinogram[:, s:e, :],
            fm_hessian[:, s:e], pixel_indices,
            left_halo=left_halo, right_halo=right_halo, ...)
        flat_recon = flat_recon.at[:, s:e].set(local_recon)
        error_sinogram = error_sinogram.at[:, s:e, :].set(local_sino)
```

**Verify:** run a full reconstruction with `n_blocks=1` (should be bit-identical to Phase 2 baseline), then `n_blocks=2,4` and compare RMSE and convergence curves. Expect near-identical convergence — the only difference is floating-point ordering at shard boundaries due to the prior coupling.

---

### Phase 4 — Clean device-agnostic updater
**Files:** `mbirjax/tomography_model.py`  
**Risk:** medium — new code path, existing single-GPU path kept intact

The existing `vcd_subset_updater` has explicit `jax.device_put(..., self.main_device)` and `jax.device_put(..., self.sinogram_device)` calls (lines 1422, 1454, 1497, 1545, 1554) and a `with jax.default_device(self.main_device):` block. These are incompatible with `shard_map` — within a shard, device placement is managed by JAX, not by explicit `device_put` to named global devices.

Add a new `create_vcd_subset_updater_sharded` that is identical to the current version except:
- No `jax.device_put` calls
- No `with jax.default_device(...)` context
- `pixel_indices_worker` argument removed (no longer two-device distinction)
- `left_halo` / `right_halo` passed through to qggmrf call
- `fm_hessian` and `weights` accepted as explicit arguments rather than closed-over (so they can be passed as shard-local arrays)

The `const_weights` path and `positivity_flag` path both stay, but the `sinogram_device != main_device` transfer branch (line 1423) is removed since all shards are self-contained.

**Verify:** Run this new function on a single GPU (with no `shard_map` yet, just called directly), compare to the Phase 2 baseline. Should be bit-identical modulo removed `device_put` no-ops.

---

### Phase 5 — `shard_map` integration
**Files:** `mbirjax/tomography_model.py`, new `mbirjax/parallel_beam_sharded.py` (or add method to `ParallelBeamModel`)

This is where actual multi-GPU parallelism lands.

**5a. Pre-shard inputs** before calling the VCD loop (in the code around line 1222–1234):

```python
if num_gpus > 1:
    mesh = Mesh(jax.devices()[:num_gpus], ('slices',))
    sharding = NamedSharding(mesh, P(None, 'slices'))
    flat_recon     = jax.device_put(flat_recon, sharding)
    error_sinogram = jax.device_put(error_sinogram, NamedSharding(mesh, P(None, 'slices', None)))
    fm_hessian     = jax.device_put(fm_hessian, sharding)
    if not const_weights:
        weights = jax.device_put(weights, NamedSharding(mesh, P(None, 'slices', None)))
```

**5b. Wrap the updater with `shard_map`**, adding the halo exchange:

```python
from jax.experimental.shard_map import shard_map

@partial(shard_map, mesh=mesh,
         in_specs=(P(None, 'slices'), P(None, 'slices', None), P(None, 'slices'),
                   P(), P(), ...),
         out_specs=(P(None, 'slices'), P(None, 'slices', None), P(), P(), ...))
def sharded_updater(flat_recon, error_sinogram, fm_hessian, pixel_indices, ...):
    n_dev    = jax.lax.psum(1, 'slices')
    dev_id   = jax.lax.axis_index('slices')
    perm_fwd = [(i, (i + 1) % n_dev) for i in range(n_dev)]
    perm_bwd = [(i, (i - 1) % n_dev) for i in range(n_dev)]

    left_halo  = jax.lax.ppermute(flat_recon[:, -1:], 'slices', perm_bwd)
    right_halo = jax.lax.ppermute(flat_recon[:,  :1], 'slices', perm_fwd)
    left_halo  = jnp.where(dev_id == 0,        flat_recon[:, :1],  left_halo)
    right_halo = jnp.where(dev_id == n_dev - 1, flat_recon[:, -1:], right_halo)

    return clean_updater(flat_recon, error_sinogram, fm_hessian, pixel_indices,
                         left_halo=left_halo, right_halo=right_halo, ...)
```

**5c. Update `vcd_partition_iterator`**: the `partition_worker = jax.device_put(partition, self.sinogram_device)` (line 1327) and `subset_worker` distinction disappears for the sharded path — `pixel_indices` is replicated on all devices.

**Verify:** reconstruct a known phantom, compare to single-GPU Phase 2 baseline. Check:
- Per-iteration `ell1` values match to within floating-point tolerance
- Final RMSE matches
- Convergence curves track each other across iterations
- Time-per-iteration scales with `1/num_gpus`

---

### Constraint to document

`num_slices` must be divisible by `num_gpus`. If not, either pad slices (zeroing the extra) or restrict which GPU counts are allowed. For parallel beam, `num_slices == num_det_rows`, so this is straightforward to check at model setup time.

---

### Summary of files touched

| File | Phase | What changes |
|------|-------|-------------|
| `parallel_beam.py` | 1 | 2-line allocation fix |
| `qggmrf.py` | 2 | `left_halo`/`right_halo` args + extend/trim |
| `vcd_utils.py` or test | 3 | simulation harness (test only) |
| `tomography_model.py` | 4, 5 | new `create_vcd_subset_updater_sharded`, pre-shard inputs, `shard_map` wrapper, updated `vcd_partition_iterator` |

Phase 1 and 2 are safe to merge independently — they have no behavioral change on single-GPU. Phase 3 is test infrastructure. Phases 4–5 add the new code path behind a `num_gpus > 1` branch.