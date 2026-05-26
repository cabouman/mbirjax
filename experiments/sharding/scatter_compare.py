"""
experiments/sharding/scatter_compare.py
────────────────────────────────────────
Diagnose the discrepancy reported when comparing the prerelease
(unsharded base-class sparse_forward_project) against the _B-batched
sharded path.

Two phases:
  Phase 1 — kernel test across all views: compares the prerelease scatter,
    the _B lax.map scatter, and a Python-loop unrolled scatter, applied to
    every view with all pixels.  Verifies whether forward_project_pixel_
    batch_to_one_view itself is the source of any discrepancy.

    Geometry arrays (n_p, n_p_center, W_p_c, cos_alpha_p_xy) are passed as
    explicit function arguments so that a single JIT compilation covers all
    180 view angles with no recompilation.

  Phase 2 — full pipeline test: runs complete sparse_forward_project in
    two configurations and cross-compares results:
      (a) unsharded base-class path  (mimics prerelease baseline)
      (b) sharded n_dev=1 path       (what the benchmark measures)
    If (a) == (b), any remaining discrepancy vs. the saved baseline lives
    in the _B scatter kernel.  If (a) != (b), the code path differs.
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

recon_shape     = model.get_params('recon_shape')
sinogram_shape  = model.get_params('sinogram_shape')
geometry_params = model.get_geometry_parameters()
ProjectorParams = namedtuple('ProjectorParams', ['sinogram_shape', 'recon_shape', 'geometry_params'])
proj_params     = ProjectorParams(sinogram_shape, recon_shape, geometry_params)
all_indices     = mbirjax.gen_full_indices(recon_shape, use_ror_mask=model.use_ror_mask)
num_pixels      = all_indices.shape[0]

rng           = np.random.default_rng(42)
recon_np      = rng.standard_normal(recon_shape).astype(np.float32)
flat_recon_np = np.asarray(
    model.get_voxels_at_indices(jnp.array(recon_np), jnp.array(np.asarray(all_indices))))
# flat_recon_np shape: (num_pixels, N_ROWS)

voxel_values = jnp.array(flat_recon_np)   # (num_pixels, N_ROWS)

# Pre-compute pixel index array — constant across views, so created once.
all_indices_jax = jnp.array(np.asarray(all_indices))

# Constants used inside JIT'd functions as closures.  These are view-independent
# so they don't trigger recompilation when we change the view angle.
gp               = proj_params.geometry_params
local_slices     = voxel_values.shape[1]        # N_ROWS = 128
num_det_channels = proj_params.sinogram_shape[2] # N_CHANNELS = 256

print(f"num_pixels={num_pixels}  local_slices={local_slices}  "
      f"num_det_channels={num_det_channels}  psf_radius={gp.psf_radius}")
print()


# ── Prerelease approach (single scatter over all slices at once) ──────────────
# Geometry arrays are explicit arguments so JAX traces once for all views.
# gp, local_slices, num_det_channels are closed over (view-independent constants).
@jax.jit
def forward_prerelease(voxel_values, n_p, n_p_center, W_p_c, cos_alpha_p_xy):
    """Prerelease scatter: one scatter over all slices simultaneously.

    For each PSF offset n_offset, computes a per-pixel weight A_chan_n and
    scatter-adds A_chan_n[p] * voxel_values[p, :] into column n[p] of the
    sinogram plane.  All slices are processed together via a (local_slices,
    num_pixels) broadcast.
    """
    L_max = jnp.minimum(1.0, W_p_c)
    sinogram_view = jnp.zeros((local_slices, num_det_channels))
    for n_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius + 1):
        n = n_p_center + n_offset                              # (num_pixels,)
        abs_delta_p_c_n = jnp.abs(n_p - n)                    # (num_pixels,)
        L_p_c_n = jnp.clip((W_p_c + 1.0) / 2.0 - abs_delta_p_c_n, 0.0, L_max)
        A_chan_n = gp.delta_voxel * L_p_c_n / cos_alpha_p_xy  # (num_pixels,)
        A_chan_n *= (n >= 0) * (n < num_det_channels)
        # A_chan_n.reshape((1,-1)) * voxel_values.T → (local_slices, num_pixels)
        # scatter-add: for each pixel p, add that column to sinogram_view[:, n[p]]
        sinogram_view = sinogram_view.at[:, n].add(A_chan_n.reshape((1, -1)) * voxel_values.T)
    return sinogram_view


# ── _B-batch lax.map approach ─────────────────────────────────────────────────
def make_forward_batched(_B):
    """Build a JIT'd function that processes slices in batches of _B via lax.map.

    Mirrors the current forward_project_pixel_batch_to_one_view in parallel_beam.py.
    num_batches==1 uses a direct call instead of lax.map to avoid an XLA bug
    where lax.map over a size-1 leading axis can produce wrong results.
    """
    @jax.jit
    def forward_batched(voxel_values, n_p, n_p_center, W_p_c, cos_alpha_p_xy):
        L_max            = jnp.minimum(1.0, W_p_c)
        num_pixels_local = voxel_values.shape[0]
        padded           = ((local_slices + _B - 1) // _B) * _B  # next multiple of _B
        num_batches      = padded // _B
        voxel_padded     = jnp.zeros((num_pixels_local, padded)).at[:, :local_slices].set(voxel_values)
        # Transpose and reshape: (num_batches, _B, num_pixels)
        voxel_batched    = voxel_padded.T.reshape(num_batches, _B, num_pixels_local)

        def project_slice_batch(voxel_batch):
            # voxel_batch: (_B, num_pixels) — the _B slices for this batch
            sino_batch = jnp.zeros((_B, num_det_channels))
            for n_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius + 1):
                n = n_p_center + n_offset
                abs_delta_p_c_n = jnp.abs(n_p - n)
                L_p_c_n = jnp.clip((W_p_c + 1.0) / 2.0 - abs_delta_p_c_n, 0.0, L_max)
                A_chan_n = gp.delta_voxel * L_p_c_n / cos_alpha_p_xy  # (num_pixels,)
                A_chan_n *= (n >= 0) * (n < num_det_channels)
                # A_chan_n * voxel_batch → (_B, num_pixels); scatter to sino_batch[:, n[p]]
                sino_batch = sino_batch.at[:, n].add(A_chan_n * voxel_batch)
            return sino_batch  # (_B, num_det_channels)

        # Bypass lax.map for num_batches==1: XLA can mishandle a size-1 leading axis.
        # num_batches is a Python int (from static shapes), so this branch is static.
        if num_batches == 1:
            sinogram_view = project_slice_batch(voxel_batched[0])[:local_slices]
        else:
            sino_batched  = jax.lax.map(project_slice_batch, voxel_batched)
            sinogram_view = sino_batched.reshape(padded, num_det_channels)[:local_slices]
        return sinogram_view
    return forward_batched


# ── Direct multi-batch without lax.map (unrolled Python loop) ────────────────
def make_forward_batched_nolaxmap(_B):
    """Build a JIT'd function that processes slices in batches of _B via a Python loop.

    Identical math to make_forward_batched but replaces lax.map with a Python
    for-loop that JAX unrolls at trace time.  Used to isolate whether any
    discrepancy is caused by lax.map semantics vs. the batching itself.
    """
    @jax.jit
    def forward_batched_nolaxmap(voxel_values, n_p, n_p_center, W_p_c, cos_alpha_p_xy):
        L_max            = jnp.minimum(1.0, W_p_c)
        num_pixels_local = voxel_values.shape[0]
        padded           = ((local_slices + _B - 1) // _B) * _B
        num_batches      = padded // _B
        voxel_padded     = jnp.zeros((num_pixels_local, padded)).at[:, :local_slices].set(voxel_values)
        voxel_batched    = voxel_padded.T.reshape(num_batches, _B, num_pixels_local)

        results = []
        for b in range(num_batches):
            voxel_batch = voxel_batched[b]       # (_B, num_pixels)
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


# ════════════════════════════════════════════════════════════════════════════════
# Phase 1: Compare kernels across ALL 180 views
# ════════════════════════════════════════════════════════════════════════════════
# Strategy: compute view-dependent geometry once per view and pass it as
# explicit JAX-array arguments.  JAX traces on shapes/dtypes (not values), so
# one compilation covers every angle with no recompilation overhead.
print("=" * 70)
print("Phase 1: Kernel comparison across all views")
print("=" * 70)

B_VALUES     = [4, 8, 16, 32]
fwd_batched  = {_B: make_forward_batched(_B)          for _B in B_VALUES}
fwd_nolaxmap = {_B: make_forward_batched_nolaxmap(_B) for _B in B_VALUES}

# Warm up all JIT compilations on view 0 so the scan loop is pure execution.
print("Compiling JIT functions on view 0 …")
_a0 = angles[0]
_n_p_0, _n_pc_0, _W_0, _cos_0 = mbirjax.ParallelBeamModel.compute_proj_data(
    all_indices_jax, _a0, proj_params)
jax.block_until_ready(forward_prerelease(voxel_values, _n_p_0, _n_pc_0, _W_0, _cos_0))
for _B in B_VALUES:
    jax.block_until_ready(fwd_batched[_B](  voxel_values, _n_p_0, _n_pc_0, _W_0, _cos_0))
    jax.block_until_ready(fwd_nolaxmap[_B](voxel_values, _n_p_0, _n_pc_0, _W_0, _cos_0))
print("  done.\n")

# Accumulate max absolute diff across all views for each (_B, variant) pair.
# tracking[_B]['lax']      = max diff over all views for lax.map variant
# tracking[_B]['lax_loc']  = (view_idx, slice, chan) where max diff occurred
tracking = {
    _B: {'lax': 0.0, 'lax_loc': None, 'nolax': 0.0, 'nolax_loc': None}
    for _B in B_VALUES
}

print(f"Scanning {N_VIEWS} views …")
for view_idx in range(N_VIEWS):
    angle_i = angles[view_idx]
    n_p_i, n_pc_i, W_p_c_i, cos_i = mbirjax.ParallelBeamModel.compute_proj_data(
        all_indices_jax, angle_i, proj_params)

    # Prerelease result for this view (shape: local_slices × num_det_channels)
    ref_i = np.asarray(forward_prerelease(voxel_values, n_p_i, n_pc_i, W_p_c_i, cos_i))

    for _B in B_VALUES:
        lax_i   = np.asarray(fwd_batched[_B](  voxel_values, n_p_i, n_pc_i, W_p_c_i, cos_i))
        nolax_i = np.asarray(fwd_nolaxmap[_B](voxel_values, n_p_i, n_pc_i, W_p_c_i, cos_i))

        diff_lax   = np.abs(lax_i   - ref_i)
        diff_nolax = np.abs(nolax_i - ref_i)

        if diff_lax.max() > tracking[_B]['lax']:
            tracking[_B]['lax'] = float(diff_lax.max())
            s, c = np.unravel_index(np.argmax(diff_lax), diff_lax.shape)
            tracking[_B]['lax_loc'] = (view_idx, int(s), int(c))

        if diff_nolax.max() > tracking[_B]['nolax']:
            tracking[_B]['nolax'] = float(diff_nolax.max())
            s, c = np.unravel_index(np.argmax(diff_nolax), diff_nolax.shape)
            tracking[_B]['nolax_loc'] = (view_idx, int(s), int(c))

# ── Summary table ─────────────────────────────────────────────────────────────
print()
print(f"  {'_B':>4}  "
      f"{'── lax.map vs prerelease ─────────────':^40}  "
      f"{'── nolaxmap vs prerelease ─────────────':^40}")
print(f"  {'':>4}  {'max_diff':>10}  {'view':>5}  {'slice':>6}  {'chan':>6}  "
      f"{'max_diff':>10}  {'view':>5}  {'slice':>6}  {'chan':>6}")
print(f"  {'─'*4}  {'─'*10}  {'─'*5}  {'─'*6}  {'─'*6}  "
      f"{'─'*10}  {'─'*5}  {'─'*6}  {'─'*6}")
for _B in B_VALUES:
    t = tracking[_B]
    ll = t['lax_loc']   or (0, 0, 0)
    nl = t['nolax_loc'] or (0, 0, 0)
    print(f"  {_B:>4}  {t['lax']:>10.2e}  {ll[0]:>5}  {ll[1]:>6}  {ll[2]:>6}  "
          f"{t['nolax']:>10.2e}  {nl[0]:>5}  {nl[1]:>6}  {nl[2]:>6}")
print()

# ── Characterise the worst offender for _B=16 ────────────────────────────────
_B_CHAR = 16
t16 = tracking[_B_CHAR]
if t16['lax_loc'] is not None and t16['lax'] > 0:
    worst_view, worst_slice, worst_chan = t16['lax_loc']
    print(f"── Characterise worst case (_B={_B_CHAR}, lax.map) ──")
    print(f"   view={worst_view}  slice={worst_slice}  chan={worst_chan}  "
          f"max_diff={t16['lax']:.6f}")

    # Recompute geometry for worst view
    angle_w = angles[worst_view]
    n_p_w, n_pc_w, W_p_c_w, cos_w = mbirjax.ParallelBeamModel.compute_proj_data(
        all_indices_jax, angle_w, proj_params)
    ref_w = np.asarray(forward_prerelease(voxel_values, n_p_w, n_pc_w, W_p_c_w, cos_w))
    lax_w = np.asarray(fwd_batched[_B_CHAR](voxel_values, n_p_w, n_pc_w, W_p_c_w, cos_w))
    print(f"   ref[worst]={ref_w[worst_slice, worst_chan]:.6f}  "
          f"lax[worst]={lax_w[worst_slice, worst_chan]:.6f}")

    # Which pixels project to worst_chan for this view?
    n_pc_np = np.asarray(n_pc_w)
    n_p_np  = np.asarray(n_p_w)
    hits = np.any(
        np.stack([n_pc_np + off == worst_chan
                  for off in range(-gp.psf_radius, gp.psf_radius + 1)], axis=1),
        axis=1)
    contrib_px = np.where(hits)[0]
    print(f"   Pixels contributing to channel {worst_chan}: {len(contrib_px)}")
    counts = np.bincount(n_pc_np.clip(0, N_CHANNELS - 1), minlength=N_CHANNELS)
    print(f"   Max pixels per channel: {counts.max()}  "
          f"channels with >10: {np.sum(counts > 10)}  >100: {np.sum(counts > 100)}")

    # ── Scatter precision micro-test ──────────────────────────────────────────
    # W_p_c and cos_alpha_p_xy are view-level scalars, not per-pixel arrays.
    W_p_c_val = float(np.asarray(W_p_c_w))
    cos_val   = float(np.asarray(cos_w))
    L_max_val = min(1.0, W_p_c_val)

    px = np.where(n_pc_np == worst_chan)[0]
    K  = len(px)
    if K > 0:
        vox_slice = np.asarray(voxel_values[px, worst_slice])  # (K,)
        abs_delta = np.abs(n_p_np[px] - worst_chan)
        L_p_c_n   = np.clip((W_p_c_val + 1.0) / 2.0 - abs_delta, 0.0, L_max_val)
        A         = float(gp.delta_voxel) * L_p_c_n / cos_val
        weights   = A * vox_slice   # (K,)

        s1 = float(np.sum(weights.astype(np.float32)))
        s2 = float(np.sum(weights[::-1].astype(np.float32)))
        s3 = float(np.sum(weights.astype(np.float64)))
        print(f"\n   Scatter precision: K={K} pixels at channel {worst_chan}")
        print(f"     forward sum  (float32): {s1:.8f}")
        print(f"     reversed sum (float32): {s2:.8f}")
        print(f"     float64 sum:            {s3:.8f}")
        print(f"     forward vs reversed:    {abs(s1 - s2):.4e}")
        print(f"     expected max FP error:  {K * np.max(np.abs(weights)) * 2**-23:.4e}")
else:
    print(f"_B={_B_CHAR} lax.map: zero diff across all {N_VIEWS} views — "
          f"kernel is identical to prerelease.")
print()

# ═════════════════════════════════════════════════════════════════════════════
# Phase 2: Full sparse_forward_project pipeline comparison
# Compare unsharded (base-class) vs sharded-n_dev=1 paths to see if the
# discrepancy comes from the code path rather than the _B scatter itself.
# ═════════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("Phase 2: Full sparse_forward_project pipeline comparison")
print("═" * 70)
print()

# Build raw inputs matching the benchmark's rng seed and problem size.
rng2            = np.random.default_rng(42)
sino_np2        = rng2.standard_normal((N_VIEWS, N_ROWS, N_CHANNELS)).astype(np.float32)
recon_np2       = rng2.standard_normal(recon_shape).astype(np.float32)
all_indices2    = mbirjax.gen_full_indices(recon_shape, use_ror_mask=model.use_ror_mask)
all_indices_np2 = np.asarray(all_indices2)
flat_recon_np2  = np.asarray(
    model.get_voxels_at_indices(jnp.array(recon_np2), jnp.array(all_indices_np2)))
# flat_recon_np2: (num_pixels, N_ROWS)

# ── (a) Unsharded base-class path (configure_sharding NOT called) ────────────
model_plain = mbirjax.ParallelBeamModel((N_VIEWS, N_ROWS, N_CHANNELS), angles)
model_plain.set_params(no_compile=True, no_warning=True)
# configure_sharding is NOT called — this mimics the prerelease baseline path.
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
sharding_spec = jax.sharding.NamedSharding(
    model_shard.mesh, jax.sharding.PartitionSpec(None, 'slices'))
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

# ── Compare against saved prerelease baseline ─────────────────────────────────
# The baseline was generated on the prerelease branch by sharding_baseline.py
# (configure_sharding commented out, n_dev=1, rng seed 42, same problem size).
# If the inputs match, (a) unsharded should be zero-diff against the baseline.
# Any nonzero diff between (a) and the baseline means the current branch's
# unsharded code path differs from the prerelease — independent of sharding.
print()
_script_dir   = os.path.dirname(os.path.abspath(__file__))
BASELINE_PATH = os.path.join(_script_dir, '..', 'sandboxes', 'sharding_baseline_ref.npz')

if not os.path.exists(BASELINE_PATH):
    print(f"Baseline not found at:\n  {BASELINE_PATH}")
    print("Skipping baseline comparison.")
else:
    baseline = np.load(BASELINE_PATH)
    print(f"Loaded baseline from:\n  {BASELINE_PATH}")
    print(f"  Keys: {list(baseline.keys())}")

    # ── Model-parameter comparison ────────────────────────────────────────────
    # If sharding_baseline.py saved these scalars, compare them to the current
    # branch so we know whether the inputs are structurally identical.
    param_keys = ['recon_shape', 'use_ror_mask',
                  'pixel_batch_size_for_vmap', 'view_batch_size_for_vmap']
    if all(k in baseline for k in param_keys):
        print()
        print("  Model-parameter comparison (prerelease → current):")
        mismatches = []
        for k in param_keys:
            bl_val  = baseline[k]
            cur_val = model_plain.get_params(k) if k == 'recon_shape' else getattr(model_plain, k, None)
            # recon_shape is a tuple; convert for display
            if k == 'recon_shape':
                cur_val = tuple(int(x) for x in cur_val)
                bl_val  = tuple(int(x) for x in bl_val)
            else:
                cur_val = bool(cur_val) if k == 'use_ror_mask' else int(cur_val)
                bl_val  = bool(bl_val)  if k == 'use_ror_mask' else int(bl_val)
            match = '✓' if bl_val == cur_val else '✗ MISMATCH'
            print(f"    {k:<35} baseline={bl_val}  current={cur_val}  {match}")
            if bl_val != cur_val:
                mismatches.append(k)
        if mismatches:
            print(f"\n  WARNING: mismatched params: {mismatches}")
            print("  Input data may differ between baseline and current branch.")
        print()

    # ── Input-data comparison ─────────────────────────────────────────────────
    # If the baseline contains the actual flat_recon and all_indices used to
    # generate it, compare them to what Phase 2 used.  A mismatch here means
    # the projector is being fed different data, not that the projector is wrong.
    if 'flat_recon' in baseline and 'all_indices' in baseline:
        bl_flat   = baseline['flat_recon']
        bl_idx    = baseline['all_indices']
        cur_flat  = flat_recon_np2
        cur_idx   = all_indices_np2
        print("  Input-data comparison:")
        print(f"    flat_recon  shapes:  baseline={bl_flat.shape}  current={cur_flat.shape}")
        print(f"    all_indices shapes:  baseline={bl_idx.shape}   current={cur_idx.shape}")
        if bl_flat.shape == cur_flat.shape:
            flat_diff = np.abs(bl_flat - cur_flat)
            print(f"    flat_recon  max_diff={flat_diff.max():.4e}  "
                  f"nonzero={np.sum(flat_diff > 0)}")
        if bl_idx.shape == cur_idx.shape:
            idx_diff = np.sum(bl_idx != cur_idx)
            print(f"    all_indices mismatched entries: {idx_diff}")
        print()

        # If inputs match, also run Phase 2 (a) with the baseline's own flat_recon
        # to isolate whether any remaining diff is in the projector itself.
        if bl_flat.shape == cur_flat.shape and np.allclose(bl_flat, cur_flat) \
                and bl_idx.shape == cur_idx.shape and np.array_equal(bl_idx, cur_idx):
            print("  Inputs are identical — projecting with baseline inputs to isolate projector diff.")
        else:
            print("  Inputs differ — projecting with baseline's flat_recon to test projector in isolation:")
            _vox_bl = jnp.array(bl_flat)
            _pix_bl = jnp.array(bl_idx)
            _res_bl = model_plain.sparse_forward_project(_vox_bl, _pix_bl)
            jax.block_until_ready(_res_bl)
            _res_bl_np = np.asarray(_res_bl)

    # ── Sinogram comparison against baseline ──────────────────────────────────
    key = 'sparse_forward_project'
    if key not in baseline:
        print(f"  No '{key}' entry in baseline — cannot compare sinograms.")
    else:
        ref_bl = baseline[key]
        print(f"  baseline['{key}'] shape={ref_bl.shape}  "
              f"min={ref_bl.min():.4f}  max={ref_bl.max():.4f}")

        def _report(label, arr):
            if ref_bl.shape != arr.shape:
                print(f"\n  {label}: shape mismatch {arr.shape} vs baseline {ref_bl.shape}")
                return
            d = np.abs(arr - ref_bl)
            # Show the full error distribution, not just max, since many
            # "nonzero" diffs may be float32 rounding noise rather than
            # meaningful errors.
            thresholds = [1e-5, 1e-3, 0.1, 1.0]
            counts = [int(np.sum(d > t)) for t in thresholds]
            print(f"\n  {label} vs baseline:")
            print(f"    max_diff={d.max():.4e}")
            for t, c in zip(thresholds, counts):
                print(f"    diff > {t:.0e}: {c:>8d} elements")
            if d.max() > 1e-3:
                # List all locations with diff > 0.1
                locs = np.argwhere(d > 0.1)
                print(f"    Locations with diff > 0.1 ({len(locs)} total):")
                for idx in locs[:20]:      # cap at 20
                    v, r, c = idx
                    print(f"      view={v:3d} row={r:3d} chan={c:3d}  "
                          f"current={arr[v,r,c]:+.6f}  baseline={ref_bl[v,r,c]:+.6f}  "
                          f"diff={d[v,r,c]:.4e}")
                if len(locs) > 20:
                    print(f"      … and {len(locs)-20} more.")

        _report("(a) unsharded",    result_a_np)
        _report("(b) sharded-1dev", result_b_np)

        # If baseline inputs were available and differed, also report the
        # isolated projector test.
        if 'flat_recon' in baseline and 'all_indices' in baseline:
            bl_flat = baseline['flat_recon']
            cur_flat = flat_recon_np2
            if bl_flat.shape != cur_flat.shape or not np.allclose(bl_flat, cur_flat):
                _report("(c) unsharded with baseline inputs", _res_bl_np)
