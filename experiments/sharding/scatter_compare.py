"""
experiments/sharding/scatter_compare.py
────────────────────────────────────────
Diagnose the 3-pixel rounding discrepancy reported when comparing the
prerelease (unsharded base-class sparse_forward_project) against the
_B-batched sharded path.

Two phases:
  Phase 1 — single-view kernel test: compares the prerelease scatter, the
    _B lax.map scatter, and a Python-loop unrolled scatter, all applied
    to one view with all pixels.  Verifies whether forward_project_pixel_
    batch_to_one_view itself is the source of the discrepancy.

  Phase 2 — full pipeline test: runs complete sparse_forward_project in
    four configurations and cross-compares results:
      (a) unsharded base-class path  (mimics prerelease baseline)
      (b) sharded n_dev=1 path       (what the benchmark measures)
      (a2) unsharded base-class with _B code
      (b2) sharded n_dev=1 with _B code
    If (a) == (b) but (a) != (a2), the issue is _B.
    If (a) != (b), the issue is the sharded vs unsharded code path.
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

from collections import namedtuple
import numpy as np
import jax
import jax.numpy as jnp
import mbirjax

# ── Problem setup ─────────────────────────────────────────────────────────────
N_VIEWS    = 180
N_ROWS     = 128
N_CHANNELS = 256

angles = np.linspace(0, np.pi, N_VIEWS, endpoint=False)
model  = mbirjax.ParallelBeamModel((N_VIEWS, N_ROWS, N_CHANNELS), angles)
model.set_params(no_compile=True, no_warning=True)

recon_shape    = model.get_params('recon_shape')
sinogram_shape = model.get_params('sinogram_shape')
geometry_params = model.get_geometry_parameters()
ProjectorParams = namedtuple('ProjectorParams', ['sinogram_shape', 'recon_shape', 'geometry_params'])
proj_params = ProjectorParams(sinogram_shape, recon_shape, geometry_params)
all_indices    = mbirjax.gen_full_indices(recon_shape, use_ror_mask=model.use_ror_mask)
num_pixels     = all_indices.shape[0]

rng            = np.random.default_rng(42)
recon_np       = rng.standard_normal(recon_shape).astype(np.float32)
flat_recon_np  = np.asarray(
    model.get_voxels_at_indices(jnp.array(recon_np), jnp.array(np.asarray(all_indices))))
# shape: (num_pixels, N_ROWS)

# Pick a single view index (one with a non-trivial angle)
VIEW_IDX = 45
angle    = angles[VIEW_IDX]

voxel_values = jnp.array(flat_recon_np)   # (num_pixels, N_ROWS)

# ── Geometry for this view ────────────────────────────────────────────────────
gp = proj_params.geometry_params
n_p, n_p_center, W_p_c, cos_alpha_p_xy = mbirjax.ParallelBeamModel.compute_proj_data(
    jnp.array(np.asarray(all_indices)), angle, proj_params)
L_max = jnp.minimum(1.0, W_p_c)

local_slices    = voxel_values.shape[1]   # = N_ROWS = 128
num_det_channels = proj_params.sinogram_shape[2]

print(f"num_pixels={num_pixels}  local_slices={local_slices}  "
      f"num_det_channels={num_det_channels}  psf_radius={gp.psf_radius}")
print()

# ── Prerelease approach (single scatter over all slices) ──────────────────────
@jax.jit
def forward_prerelease(voxel_values):
    sinogram_view = jnp.zeros((local_slices, num_det_channels))
    for n_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius + 1):
        n = n_p_center + n_offset
        abs_delta_p_c_n = jnp.abs(n_p - n)
        L_p_c_n = jnp.clip((W_p_c + 1.0) / 2.0 - abs_delta_p_c_n, 0.0, L_max)
        A_chan_n = gp.delta_voxel * L_p_c_n / cos_alpha_p_xy
        A_chan_n *= (n >= 0) * (n < num_det_channels)
        sinogram_view = sinogram_view.at[:, n].add(A_chan_n.reshape((1, -1)) * voxel_values.T)
    return sinogram_view


# ── _B-batch lax.map approach ─────────────────────────────────────────────────
def make_forward_batched(_B):
    @jax.jit
    def forward_batched(voxel_values):
        num_pixels_local = voxel_values.shape[0]
        padded     = ((local_slices + _B - 1) // _B) * _B
        num_batches = padded // _B
        voxel_padded  = jnp.zeros((num_pixels_local, padded)).at[:, :local_slices].set(voxel_values)
        voxel_batched = voxel_padded.T.reshape(num_batches, _B, num_pixels_local)

        def project_slice_batch(voxel_batch):
            sino_batch = jnp.zeros((_B, num_det_channels))
            for n_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius + 1):
                n = n_p_center + n_offset
                abs_delta_p_c_n = jnp.abs(n_p - n)
                L_p_c_n = jnp.clip((W_p_c + 1.0) / 2.0 - abs_delta_p_c_n, 0.0, L_max)
                A_chan_n = gp.delta_voxel * L_p_c_n / cos_alpha_p_xy
                A_chan_n *= (n >= 0) * (n < num_det_channels)
                sino_batch = sino_batch.at[:, n].add(A_chan_n * voxel_batch)
            return sino_batch

        if num_batches == 1:
            sinogram_view = project_slice_batch(voxel_batched[0])[:local_slices]
        else:
            sino_batched  = jax.lax.map(project_slice_batch, voxel_batched)
            sinogram_view = sino_batched.reshape(padded, num_det_channels)[:local_slices]
        return sinogram_view
    return forward_batched


# ── Direct multi-batch without lax.map (unrolled Python loop) ────────────────
def make_forward_batched_nolaxmap(_B):
    @jax.jit
    def forward_batched_nolaxmap(voxel_values):
        num_pixels_local = voxel_values.shape[0]
        padded      = ((local_slices + _B - 1) // _B) * _B
        num_batches = padded // _B
        voxel_padded  = jnp.zeros((num_pixels_local, padded)).at[:, :local_slices].set(voxel_values)
        voxel_batched = voxel_padded.T.reshape(num_batches, _B, num_pixels_local)

        results = []
        for b in range(num_batches):
            voxel_batch = voxel_batched[b]
            sino_batch  = jnp.zeros((_B, num_det_channels))
            for n_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius + 1):
                n = n_p_center + n_offset
                abs_delta_p_c_n = jnp.abs(n_p - n)
                L_p_c_n = jnp.clip((W_p_c + 1.0) / 2.0 - abs_delta_p_c_n, 0.0, L_max)
                A_chan_n = gp.delta_voxel * L_p_c_n / cos_alpha_p_xy
                A_chan_n *= (n >= 0) * (n < num_det_channels)
                sino_batch = sino_batch.at[:, n].add(A_chan_n * voxel_batch)
            results.append(sino_batch)
        return jnp.concatenate(results, axis=0)[:local_slices]
    return forward_batched_nolaxmap


# ── Run and compare ───────────────────────────────────────────────────────────
print("Computing prerelease result …")
ref = np.asarray(forward_prerelease(voxel_values))
print(f"  ref shape {ref.shape}  min={ref.min():.4f}  max={ref.max():.4f}")
print()

for _B in [4, 8, 16, 32, 128]:
    res_laxmap  = np.asarray(make_forward_batched(_B)(voxel_values))
    res_nolaxmap = np.asarray(make_forward_batched_nolaxmap(_B)(voxel_values))

    diff_lax  = np.abs(res_laxmap  - ref)
    diff_nolax = np.abs(res_nolaxmap - ref)
    diff_ll   = np.abs(res_laxmap  - res_nolaxmap)

    print(f"_B={_B:4d}  lax.map vs prerelease:   max_diff={diff_lax.max():.4e}  "
          f"nonzero={np.sum(diff_lax > 1e-5)}")
    print(f"         nolaxmap vs prerelease: max_diff={diff_nolax.max():.4e}  "
          f"nonzero={np.sum(diff_nolax > 1e-5)}")
    print(f"         lax.map vs nolaxmap:   max_diff={diff_ll.max():.4e}")
    print()

# ── Characterise the worst offender for _B=16 ────────────────────────────────
_B = 16
res_laxmap = np.asarray(make_forward_batched(_B)(voxel_values))
diff       = np.abs(res_laxmap - ref)
worst_slice, worst_chan = np.unravel_index(np.argmax(diff), diff.shape)
print(f"Worst error at slice={worst_slice} channel={worst_chan}  "
      f"diff={diff[worst_slice, worst_chan]:.6f}  "
      f"ref={ref[worst_slice, worst_chan]:.6f}")

# Find which pixels project to worst_chan for this view
n_p_np = np.asarray(n_p)
n_p_center_np = np.asarray(n_p_center)
# A pixel contributes to channel c if n_p_center + offset == c for some offset in [-psf_r, psf_r]
hits = np.any(
    np.stack([n_p_center_np + off == worst_chan
              for off in range(-gp.psf_radius, gp.psf_radius + 1)], axis=1),
    axis=1
)
contributing_pixels = np.where(hits)[0]
print(f"\nPixels projecting to channel {worst_chan}: {len(contributing_pixels)}")
if len(contributing_pixels) > 0 and len(contributing_pixels) <= 20:
    for pi in contributing_pixels:
        n_p_val = n_p_np[pi]
        frac    = n_p_val - np.floor(n_p_val)
        print(f"  pixel {pi:6d}: n_p={n_p_val:.6f}  frac={frac:.6f}  "
              f"n_p_center={n_p_center_np[pi]}  voxel_val[slice={worst_slice}]="
              f"{float(voxel_values[pi, worst_slice]):.6f}")
elif len(contributing_pixels) > 20:
    # Just show statistics
    fracs = n_p_np[contributing_pixels] - np.floor(n_p_np[contributing_pixels])
    near_half = np.sum(np.abs(fracs - 0.5) < 0.01)
    print(f"  n_p fractional parts: min={fracs.min():.4f} max={fracs.max():.4f} "
          f"near-half (|frac-0.5|<0.01): {near_half}")

# ── Check whether duplicate n indices are the issue ──────────────────────────
print("\n── Duplicate n_p_center indices per channel ──")
counts = np.bincount(n_p_center_np.clip(0, N_CHANNELS-1), minlength=N_CHANNELS)
print(f"Max pixels per channel: {counts.max()}  "
      f"Channels with >10 pixels: {np.sum(counts > 10)}  "
      f"Channels with >100 pixels: {np.sum(counts > 100)}")
print(f"Channel {worst_chan} has {counts[worst_chan]} pixels with n_p_center={worst_chan}")

# ── Test: does changing ONLY the leading-dim size of the scatter change results?
# Run a pure-Python scatter for a tiny case to see if float32 differences appear
print("\n── Scatter precision micro-test ──")
K = int(counts[worst_chan])
if K > 0:
    # Gather the pixels that project to worst_chan
    px = np.where(n_p_center_np == worst_chan)[0]
    vox_slice = np.asarray(voxel_values[px, worst_slice])  # shape (K,)

    # Compute A_chan for offset=0
    abs_delta = np.abs(n_p_np[px] - worst_chan)
    W_p_c_np  = np.asarray(W_p_c)[px]
    L_max_np  = np.asarray(L_max)[px]
    L_p_c_n   = np.clip((W_p_c_np + 1.0) / 2.0 - abs_delta, 0.0, L_max_np)
    A         = float(gp.delta_voxel) * L_p_c_n / np.asarray(cos_alpha_p_xy)[px]
    weights   = A * vox_slice   # shape (K,)

    # Compute sum in different orders
    s1 = float(np.sum(weights.astype(np.float32)))
    s2 = float(np.sum(weights[::-1].astype(np.float32)))
    s3 = float(np.sum(weights.astype(np.float64)))

    print(f"K={K} pixels project to channel {worst_chan}")
    print(f"  forward sum (float32):  {s1:.8f}")
    print(f"  reversed sum (float32): {s2:.8f}")
    print(f"  float64 sum:            {s3:.8f}")
    print(f"  forward vs reversed:    diff={abs(s1-s2):.4e}")
    print(f"  expected max FP error:  {K * np.max(np.abs(weights)) * 2**-23:.4e}")

# ═════════════════════════════════════════════════════════════════════════════
# Phase 2: Full sparse_forward_project pipeline comparison
# Compare unsharded (base-class) vs sharded-n_dev=1 paths to see if the
# discrepancy comes from the code path rather than the _B scatter itself.
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("Phase 2: Full sparse_forward_project pipeline comparison")
print("═"*70 + "\n")

# Build raw inputs matching the benchmark's rng seed and problem size.
rng2           = np.random.default_rng(42)
sino_np2       = rng2.standard_normal((N_VIEWS, N_ROWS, N_CHANNELS)).astype(np.float32)
recon_np2      = rng2.standard_normal(recon_shape).astype(np.float32)
all_indices2   = mbirjax.gen_full_indices(recon_shape, use_ror_mask=model.use_ror_mask)
all_indices_np2 = np.asarray(all_indices2)
flat_recon_np2 = np.asarray(
    model.get_voxels_at_indices(jnp.array(recon_np2), jnp.array(all_indices_np2)))
# flat_recon_np2: (num_pixels, N_ROWS)

# ── (a) Unsharded base-class path (configure_sharding NOT called) ────────────
model_plain = mbirjax.ParallelBeamModel((N_VIEWS, N_ROWS, N_CHANNELS), angles)
model_plain.set_params(no_compile=True, no_warning=True)
# Do NOT call configure_sharding — this mimics the prerelease baseline.
voxels_plain = jnp.array(flat_recon_np2)
pix_plain    = jnp.array(all_indices_np2)
print("(a) Running unsharded base-class sparse_forward_project …")
result_a = model_plain.sparse_forward_project(voxels_plain, pix_plain)
jax.block_until_ready(result_a)
result_a_np = np.asarray(result_a)
print(f"    shape={result_a_np.shape}  min={result_a_np.min():.4f}  max={result_a_np.max():.4f}")

# ── (b) Sharded n_dev=1 path (configure_sharding called with 1 device) ──────
model_shard = mbirjax.ParallelBeamModel((N_VIEWS, N_ROWS, N_CHANNELS), angles)
model_shard.set_params(no_compile=True, no_warning=True)
devs = jax.devices('cpu')[:1]
model_shard.configure_sharding(devs)
# Pre-shard flat_recon along axis 1 (slices), exactly as the benchmark does.
spec = [None, 'slices']
sharding_spec = jax.sharding.NamedSharding(
    model_shard.mesh, jax.sharding.PartitionSpec(*spec))
voxels_sharded = jax.device_put(flat_recon_np2, sharding_spec)
jax.block_until_ready(voxels_sharded)
pix_shard = jnp.array(all_indices_np2)
print("(b) Running sharded n_dev=1 sparse_forward_project …")
result_b = model_shard.sparse_forward_project(voxels_sharded, pix_shard)
jax.block_until_ready(result_b)
result_b_np = np.asarray(result_b)
print(f"    shape={result_b_np.shape}  min={result_b_np.min():.4f}  max={result_b_np.max():.4f}")

# ── Cross-comparisons ─────────────────────────────────────────────────────────
diff_ab = np.abs(result_a_np - result_b_np)
print(f"\n(a) unsharded vs (b) sharded-1dev:")
print(f"    max_diff={diff_ab.max():.4e}  nonzero_diffs={np.sum(diff_ab > 0)}")
if diff_ab.max() > 0:
    s, r, c = np.unravel_index(np.argmax(diff_ab), diff_ab.shape)
    print(f"    worst: view={s} row={r} channel={c}  "
          f"a={result_a_np[s,r,c]:.6f}  b={result_b_np[s,r,c]:.6f}  "
          f"diff={diff_ab[s,r,c]:.4e}")
