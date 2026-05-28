"""
experiments/sharding/minimal_lax_map_repro.py
─────────────────────────────────────────────
Minimal, mbirjax-free reproducer for the lax.scan + scatter-add bug.

Goal: trigger a divergence between
    REF   :  scatter-add applied directly (outside any scan)
    BUGGY :  scatter-add applied inside jax.lax.map
when the input arrays contain an exact-half-integer float "n_p" and
a closed-over int32 "n_pc" = round(n_p).

The structure mirrors mbirjax's PSF-style projector:
    for offset in [-1, 0, 1]:
        n     = n_pc + offset
        abs_d = |n_p - n|
        L     = clip((W+1)/2 - abs_d, 0, L_mx)
        A     = DV * L / cos_a * ((n >= 0) & (n < ND))
        sb    = sb.at[:, n].add(A * vb_batch)

We run several test variants (T1..T4) of increasing aggressiveness:
    T1: single half-integer pixel, others random.       No vmap.
    T2: all n_p exactly at half-integers.               No vmap.
    T3: T2 + jax.vmap over multiple "views".
    T4: production-like n_pc pattern (one duplicate channel).

If none repro, we'll discuss what additional ingredient the production
geometry contributes that this version is missing.
"""

import os, sys
import numpy as np
import jax
import jax.numpy as jnp
# Enable float64 so safe_round_f64 variants below can actually use it.
# All existing tests are explicit float32, so this is benign for them.
jax.config.update("jax_enable_x64", True)

print("=" * 70)
print("  minimal_lax_map_repro.py")
print("=" * 70)
print(f"  JAX version    : {jax.__version__}")
print(f"  default backend: {jax.default_backend()}")
print()

# ── Shapes matching production ────────────────────────────────────────────────
NP    = 2048
N_ROWS = 128
_B    = 16
_NB   = N_ROWS // _B          # = 8
ND    = 256

# PSF/geometry constants (numerically representative of a view near angle≈π/15
# where W_pc≈0.978; these exact values aren't critical, but they put L in a
# regime similar to production).
W_pc  = np.float32(0.978)
L_mx  = np.float32(0.978)
cos_a = np.float32(0.978)
DV    = np.float32(1.0)

def get_2d_ror_mask(recon_shape):
    """
    Get a binary mask for the region of reconstruction.  By default, the mask is the largest possible circle
    inscribed on the longest edge of the 2D recon_shape[0:2].

    Args:
        recon_shape (tuple): Shape of recon in (rows, columns, slices)
        crop_radius_pixels (int): Number of pixels to subtract from the radius before creating the mask.
        crop_radius_fraction (float): Fraction to subtract from the radius before creating the mask.

    Returns:
        A binary mask for the region of reconstruction.
    """
    # Set up a mask to zero out points outside the ROR
    num_recon_rows, num_recon_cols = recon_shape[:2]
    row_center = (num_recon_rows - 1) / 2
    col_center = (num_recon_cols - 1) / 2

    radius = max(row_center, col_center)

    col_coords = np.arange(num_recon_cols) - col_center
    row_coords = np.arange(num_recon_rows) - row_center

    coords = np.meshgrid(col_coords, row_coords)  # Note the interchange of rows and columns for meshgrid
    mask = coords[0]**2 + coords[1]**2 <= radius**2
    mask = mask[:, :]
    return mask


def gen_pixel_partition(recon_shape, use_ror_mask=True):
    """
    Generates a partition of pixel indices into specified number of subsets for use in tomographic reconstruction algorithms.
    The function ensures that each subset contains an equal number of pixels, suitable for VCD reconstruction.

    Args:
        recon_shape (tuple): Shape of recon in (rows, columns, slices)
        use_ror_mask (bool): Flag to indicate whether to mask out a circular RoR

    Raises:
        ValueError: If the number of subsets specified is greater than the total number of pixels in the grid.

    Returns:
        jnp.array: A JAX array where each row corresponds to a subset of pixel indices, sorted within each subset.
    """
    # Determine the 2D indices within the RoR
    num_recon_rows, num_recon_cols = recon_shape[:2]
    max_index_val = num_recon_rows * num_recon_cols
    indices = np.arange(max_index_val, dtype=np.int32)

    # Mask off indices that are outside the region of reconstruction
    if use_ror_mask:
        mask = get_2d_ror_mask(recon_shape)
        mask = mask.flatten()
        indices = indices[mask == 1]

    indices = jnp.sort(indices)

    return jnp.array(indices)

def project_slice_batch(vb_batch, n_p, n_pc):
    """Inner kernel: scatter-add for one slice batch (_B slices)."""
    sb = jnp.zeros((_B, ND), dtype=jnp.float32)
    for no in (-1, 0, 1):
        n     = n_pc + no
        abs_d = jnp.abs(n_p - n)
        L     = jnp.clip((W_pc + 1.0) / 2.0 - abs_d, 0.0, L_mx)
        A     = DV * L / cos_a * ((n >= 0) & (n < ND))
        sb    = sb.at[:, n].add(A * vb_batch)
    return sb


def forward_ref(voxel, n_p, n_pc):
    """REFERENCE: a single big scatter, no scan / no lax.map."""
    sino = jnp.zeros((N_ROWS, ND), dtype=jnp.float32)
    for no in (-1, 0, 1):
        n     = n_pc + no
        abs_d = jnp.abs(n_p - n)
        L     = jnp.clip((W_pc + 1.0) / 2.0 - abs_d, 0.0, L_mx)
        A     = DV * L / cos_a * ((n >= 0) & (n < ND))
        sino  = sino.at[:, n].add(A * voxel.T)
    return sino


def forward_buggy(voxel, n_p, n_pc):
    """BUGGY: lax.map over _NB slice batches."""
    vb_batched = voxel.T.reshape(_NB, _B, NP)
    def body(vb):
        return project_slice_batch(vb, n_p, n_pc)
    return jax.lax.map(body, vb_batched).reshape(N_ROWS, ND)


def compare(name, voxel, n_p, n_pc, use_vmap_axis=None):
    """Run both paths and report max|diff|.  If use_vmap_axis is set,
    add a vmap over that axis of n_p / n_pc / voxel."""
    if use_vmap_axis is None:
        ref_fn   = jax.jit(forward_ref)
        buggy_fn = jax.jit(forward_buggy)
        ref   = ref_fn(voxel, n_p, n_pc)
        buggy = buggy_fn(voxel, n_p, n_pc)
    else:
        ref_fn   = jax.jit(jax.vmap(forward_ref,   in_axes=(None, 0, 0)))
        buggy_fn = jax.jit(jax.vmap(forward_buggy, in_axes=(None, 0, 0)))
        ref   = ref_fn(voxel, n_p, n_pc)
        buggy = buggy_fn(voxel, n_p, n_pc)
    diff = jnp.abs(ref - buggy)
    md   = float(diff.max())
    # Report worst (slice, channel) for context
    if md > 1e-3:
        idx = jnp.unravel_index(jnp.argmax(diff), diff.shape)
        idx_t = tuple(int(i) for i in idx)
        tag = "✗ BUG"
    else:
        idx_t = None
        tag = "✓ ok "
    print(f"  {name:<60s}  max|diff| = {md:.4e}  {tag}"
          + (f"  at {idx_t}" if idx_t else ""))
    return md


# Common voxel data (kept the same across all tests for fair comparison)
np.random.seed(0)
voxel_np = np.random.rand(NP, N_ROWS).astype(np.float32)
voxel    = jnp.array(voxel_np)


# ──────────────────────────────────────────────────────────────────────────────
print("─" * 70)
print("T1: ONE half-integer pixel (n_p[0] = 171.5), rest random")
print("─" * 70)

n_p_np = np.random.uniform(1, ND - 2, NP).astype(np.float32)
n_p_np[0] = np.float32(171.5)
n_pc_np = np.round(n_p_np).astype(np.int32)
# sanity: float32(171.5) rounds to 172 (banker's)
assert n_pc_np[0] == 172, f"n_pc[0]={n_pc_np[0]}, expected 172"
compare("T1  no vmap   ", voxel, jnp.array(n_p_np), jnp.array(n_pc_np))


# ──────────────────────────────────────────────────────────────────────────────
print()
print("─" * 70)
print("T2: ALL n_p exactly at half-integers, n_pc = banker's round")
print("─" * 70)

# Spread half-integer n_p values across the valid channel range
half = np.arange(NP) % (ND - 4) + 2          # integer channels 2..ND-3
n_p_np = (half + 0.5).astype(np.float32)     # all .5 values
n_pc_np = np.round(n_p_np).astype(np.int32)  # banker's rounding
compare("T2  no vmap   ", voxel, jnp.array(n_p_np), jnp.array(n_pc_np))


# ──────────────────────────────────────────────────────────────────────────────
print()
print("─" * 70)
print("T3: T2 + jax.vmap over 2 'views' (different n_pc patterns)")
print("─" * 70)

half0 = np.arange(NP) % (ND - 4) + 2
half1 = (np.arange(NP) + 7) % (ND - 4) + 2
n_p_np  = np.stack([(half0 + 0.5), (half1 + 0.5)]).astype(np.float32)
n_pc_np = np.round(n_p_np).astype(np.int32)
compare("T3  vmap-over-views", voxel,
        jnp.array(n_p_np), jnp.array(n_pc_np), use_vmap_axis=0)


# ──────────────────────────────────────────────────────────────────────────────
print()
print("─" * 70)
print("T4: many pixels collide on a single channel (production-like)")
print("─" * 70)

# All 2048 pixels point at channel 172 with n_p exactly 171.5.
# This creates extreme duplicate-index pressure on the scatter.
n_p_np  = np.full(NP, 171.5, dtype=np.float32)
n_pc_np = np.round(n_p_np).astype(np.int32)
assert n_pc_np[0] == 172
compare("T4  all-collide", voxel, jnp.array(n_p_np), jnp.array(n_pc_np))


# ──────────────────────────────────────────────────────────────────────────────
print()
print("─" * 70)
print("T4b: same as T4 but with +1e-3 perturbation (should be clean)")
print("─" * 70)

n_p_np = np.full(NP, 171.5 + 1e-3, dtype=np.float32)
n_pc_np = np.round(n_p_np).astype(np.int32)
compare("T4b perturbed  ", voxel, jnp.array(n_p_np), jnp.array(n_pc_np))


# ──────────────────────────────────────────────────────────────────────────────
print()
print("─" * 70)
print("T5: production-style n_p from cos/sin of integer pixel grid (no mbirjax)")
print("─" * 70)

# Build a 2D integer pixel grid (32x64 = 2048 pixels), centered.
ROWS, COLS = 32, 64
assert ROWS * COLS == NP
ri, ci = np.meshgrid(np.arange(ROWS), np.arange(COLS), indexing='ij')
yt = (ri.ravel() - (ROWS - 1) / 2.0).astype(np.float32)
xt = (ci.ravel() - (COLS - 1) / 2.0).astype(np.float32)
DCC = (ND - 1) / 2.0           # = 127.5
OFF = np.float32(0.0)
DDC = np.float32(1.0)

# Try several angles; report each.  View 12 of 180 = 12*π/180.
for view_idx in (12, 134, 45, 90):
    angle = np.float32(view_idx * np.pi / 180.0)
    ct = np.cos(angle).astype(np.float32)
    st = np.sin(angle).astype(np.float32)
    x_p = ct * xt - st * yt
    n_p_np  = ((x_p + OFF) / DDC + DCC).astype(np.float32)
    n_pc_np = np.round(n_p_np).astype(np.int32)
    n_half  = int(np.sum(np.abs(n_p_np - np.round(n_p_np)) < 1e-5))
    label = f"T5  view={view_idx:>3d}  half-integer n_p count={n_half:>3d}"
    compare(label, voxel, jnp.array(n_p_np), jnp.array(n_pc_np))


# ──────────────────────────────────────────────────────────────────────────────
print()
print("─" * 70)
print("T6: T5 + vmap over both 'bad views' simultaneously (matches demo shape)")
print("─" * 70)

n_p_list, n_pc_list = [], []
for view_idx in (12, 134):
    angle = np.float32(view_idx * np.pi / 180.0)
    ct = np.cos(angle).astype(np.float32)
    st = np.sin(angle).astype(np.float32)
    x_p = ct * xt - st * yt
    n_p_np  = ((x_p + OFF) / DDC + DCC).astype(np.float32)
    n_pc_np = np.round(n_p_np).astype(np.int32)
    n_p_list.append(n_p_np)
    n_pc_list.append(n_pc_np)
n_p_2v  = jnp.array(np.stack(n_p_list))
n_pc_2v = jnp.array(np.stack(n_pc_list))
compare("T6  vmap over 2 views", voxel, n_p_2v, n_pc_2v, use_vmap_axis=0)


# ──────────────────────────────────────────────────────────────────────────────
print()
print("─" * 70)
print("T7: circular ROR mask (mimics mbirjax use_ror_mask=True default)")
print("─" * 70)

# Default mbirjax recon_shape for (N_VIEWS, N_ROWS, N_CHANNELS=256) is
# typically (256, 256, 128).  pixels are (row, col) indexed.  ROR keeps only
# the disc inscribed in the rectangular grid.
ROWS_3D, COLS_3D = 256, 256
i_grid, j_grid = np.meshgrid(np.arange(ROWS_3D), np.arange(COLS_3D), indexing='ij')
yc = (ROWS_3D - 1) / 2.0
xc = (COLS_3D - 1) / 2.0
r_sq = (i_grid - yc) ** 2 + (j_grid - xc) ** 2
mask = r_sq <= (min(ROWS_3D, COLS_3D) / 2.0) ** 2
in_ror_flat = np.where(mask.ravel())[0]
print(f"  In-ROR pixels: {len(in_ror_flat)} / {ROWS_3D * COLS_3D}  "
      f"({100.0 * len(in_ror_flat) / (ROWS_3D * COLS_3D):.1f}%)")

# Batch 8 (BAD=8 in demo), 2048 pixels
BAD = 8
assert (BAD + 1) * NP <= len(in_ror_flat), "ROR doesn't have enough pixels for batch 8"
pb = in_ror_flat[BAD * NP : (BAD + 1) * NP]
ri_np, ci_np = np.unravel_index(pb, (ROWS_3D, COLS_3D))
yt_ror = (1.0 * (ri_np - (ROWS_3D - 1) / 2.0)).astype(np.float32)
xt_ror = (1.0 * (ci_np - (COLS_3D - 1) / 2.0)).astype(np.float32)
print(f"  Batch {BAD}: row range = {ri_np.min()}..{ri_np.max()}, "
      f"col range = {ci_np.min()}..{ci_np.max()}")

DCC_p = (ND - 1) / 2.0     # = 127.5

for view_idx in (12, 134):
    angle = np.float32(view_idx * np.pi / 180.0)
    ct = np.cos(angle).astype(np.float32)
    st = np.sin(angle).astype(np.float32)
    x_p = ct * xt_ror - st * yt_ror
    n_p_np  = ((x_p + 0.0) / 1.0 + DCC_p).astype(np.float32)
    n_pc_np = np.round(n_p_np).astype(np.int32)

    # How many n_p land exactly on half-integers?  And what's the closest
    # any pixel gets?
    frac     = np.abs(n_p_np - np.round(n_p_np))
    n_half   = int(np.sum(frac < 1e-6))
    closest  = float(np.min(np.abs(frac - 0.5)))   # distance to nearest .5
    # Also: any n_p exactly = X.5?
    diff_5   = np.abs(n_p_np - (np.floor(n_p_np) + 0.5))
    n_exact5 = int(np.sum(diff_5 < 1e-6))

    label = (f"T7  ROR view={view_idx:>3d}  exact-half-int n_p={n_exact5:>3d}  "
             f"closest-to-.5={closest:.2e}")
    compare(label, voxel, jnp.array(n_p_np), jnp.array(n_pc_np))


# ──────────────────────────────────────────────────────────────────────────────
print()
print("─" * 70)
print("T8: production (xt, yt) from local gen_pixel_partition  (mbirjax-free)")
print("─" * 70)

# Hardcoded defaults matching mbirjax ParallelBeamModel((180, 128, 256)).
# DV/DDC/OFF are the parallel-beam defaults (1.0, 1.0, 0.0).  recon_shape
# is (cols, rows, slices) = (256, 256, 128).  use_ror_mask=True is the
# critical ingredient — toggling it to False makes the bug vanish in the demo.
N_VIEWS = 180
angles_full = np.linspace(0, np.pi, N_VIEWS, endpoint=False)
recon_shape = (256, 256, 128)
DV_prod  = 1.0
DDC_prod = 1.0
OFF_prod = 0.0
DCC_prod = (ND - 1) / 2.0
print(f"  recon_shape = {recon_shape}")

all_indices = np.asarray(gen_pixel_partition(recon_shape, use_ror_mask=True))
print(f"  total in-ROR pixels: {len(all_indices)} "
      f"(rectangular would be {recon_shape[0] * recon_shape[1]})")

# Production demo's BAD = 8, PBSZ = NP = 2048
BAD = 8
pb_prod = all_indices[BAD * NP : (BAD + 1) * NP]
ri_p, ci_p = np.unravel_index(pb_prod, recon_shape[:2])
yt_prod = (DV_prod * (ri_p - (recon_shape[0] - 1) / 2.0)).astype(np.float32)
xt_prod = (DV_prod * (ci_p - (recon_shape[1] - 1) / 2.0)).astype(np.float32)
print(f"  Batch {BAD}: row range = {ri_p.min()}..{ri_p.max()}, "
      f"col range = {ci_p.min()}..{ci_p.max()}")

# Compute n_p / n_pc / W_pc / cos_a per view using *production* geometry params
def prod_geom(view_idx):
    angle = np.float32(angles_full[view_idx])
    ct = np.cos(angle).astype(np.float32)
    st = np.sin(angle).astype(np.float32)
    x_p = ct * xt_prod - st * yt_prod
    n_p_ = ((x_p + OFF_prod) / DDC_prod + DCC_prod).astype(np.float32)
    n_pc_ = np.round(n_p_).astype(np.int32)
    cos_a_ = np.float32(max(abs(ct), abs(st)))
    W_pc_ = np.float32((DV_prod / DDC_prod) * cos_a_)
    L_mx_ = np.float32(min(1.0, float(W_pc_)))
    return n_p_, n_pc_, W_pc_, cos_a_, L_mx_

# We need per-view W_pc/cos_a/L_mx now, so compare() in its current form
# (which uses module-level constants) won't work.  Define a local variant.
def compare_per_view(view_idx):
    n_p_np, n_pc_np, W_v, cos_v, L_v = prod_geom(view_idx)
    n_half = int(np.sum(np.abs(n_p_np - np.round(n_p_np)) < 1e-6))

    def fwd_ref(voxel_, n_p_, n_pc_):
        sino = jnp.zeros((N_ROWS, ND), dtype=jnp.float32)
        for no in (-1, 0, 1):
            n     = n_pc_ + no
            abs_d = jnp.abs(n_p_ - n)
            L     = jnp.clip((W_v + 1.0) / 2.0 - abs_d, 0.0, L_v)
            A     = DV_prod * L / cos_v * ((n >= 0) & (n < ND))
            sino  = sino.at[:, n].add(A * voxel_.T)
        return sino

    def fwd_buggy(voxel_, n_p_, n_pc_):
        vb_batched = voxel_.T.reshape(_NB, _B, NP)
        def body(vb):
            sb = jnp.zeros((_B, ND), dtype=jnp.float32)
            for no in (-1, 0, 1):
                n     = n_pc_ + no
                abs_d = jnp.abs(n_p_ - n)
                L     = jnp.clip((W_v + 1.0) / 2.0 - abs_d, 0.0, L_v)
                A     = DV_prod * L / cos_v * ((n >= 0) & (n < ND))
                sb    = sb.at[:, n].add(A * vb)
            return sb
        return jax.lax.map(body, vb_batched).reshape(N_ROWS, ND)

    n_p_j  = jnp.array(n_p_np)
    n_pc_j = jnp.array(n_pc_np)
    ref   = jax.jit(fwd_ref)(voxel, n_p_j, n_pc_j)
    buggy = jax.jit(fwd_buggy)(voxel, n_p_j, n_pc_j)
    diff = jnp.abs(ref - buggy)
    md   = float(diff.max())
    tag  = "✗ BUG" if md > 1e-3 else "✓ ok "
    print(f"  T8  prod-geom view={view_idx:>3d}  exact-half-int n_p={n_half:>3d}  "
          f"W={float(W_v):.4f}  max|diff|={md:.4e}  {tag}")

for view_idx in (12, 134, 0, 45, 90):
    compare_per_view(view_idx)


# ──────────────────────────────────────────────────────────────────────────────
print()
print("─" * 70)
print("T9: production geom + n_p derived in-jit from angle (mirrors demo)")
print("─" * 70)

xt_jnp = jnp.array(xt_prod)
yt_jnp = jnp.array(yt_prod)

def _geom_inline(angle):
    """Replicates the demo's _geom — n_p, n_pc derived inside the jit."""
    ct = jnp.cos(angle)
    st = jnp.sin(angle)
    x_p   = ct * xt_jnp - st * yt_jnp
    n_p   = (x_p + OFF_prod) / DDC_prod + DCC_prod
    n_pc  = jnp.round(n_p).astype(jnp.int32)
    cos_a = jnp.maximum(jnp.abs(ct), jnp.abs(st))
    W_pc  = (DV_prod / DDC_prod) * cos_a
    L_mx  = jnp.minimum(1.0, W_pc)
    return n_p, n_pc, W_pc, cos_a, L_mx


def forward_ref_inline(angle):
    n_p, n_pc, W_pc, cos_a, L_mx = _geom_inline(angle)
    sino = jnp.zeros((N_ROWS, ND), dtype=jnp.float32)
    for no in (-1, 0, 1):
        n     = n_pc + no
        abs_d = jnp.abs(n_p - n)
        L     = jnp.clip((W_pc + 1.0) / 2.0 - abs_d, 0.0, L_mx)
        A     = DV_prod * L / cos_a * ((n >= 0) & (n < ND))
        sino  = sino.at[:, n].add(A * voxel.T)
    return sino


def forward_buggy_inline(angle):
    n_p, n_pc, W_pc, cos_a, L_mx = _geom_inline(angle)
    vb_batched = voxel.T.reshape(_NB, _B, NP)
    def body(vb):
        sb = jnp.zeros((_B, ND), dtype=jnp.float32)
        for no in (-1, 0, 1):
            n     = n_pc + no
            abs_d = jnp.abs(n_p - n)
            L     = jnp.clip((W_pc + 1.0) / 2.0 - abs_d, 0.0, L_mx)
            A     = DV_prod * L / cos_a * ((n >= 0) & (n < ND))
            sb    = sb.at[:, n].add(A * vb)
        return sb
    return jax.lax.map(body, vb_batched).reshape(N_ROWS, ND)


angles_2v = jnp.array(angles_full[[12, 134]].astype(np.float32))

# T9a: vmap over the 2 bad views (matches demo exactly)
ref_v   = jax.jit(jax.vmap(forward_ref_inline))(angles_2v)
buggy_v = jax.jit(jax.vmap(forward_buggy_inline))(angles_2v)
diff_v  = jnp.abs(ref_v - buggy_v)
md_v    = float(diff_v.max())
print(f"  T9a  vmap+in-jit  views=[12,134]   max|diff| = {md_v:.4e}   "
      f"{'✗ BUG' if md_v > 1e-3 else '✓ ok '}")

# T9b: in-jit n_p, single view at a time (no vmap)
for view_idx in (12, 134):
    a = jnp.array(np.float32(angles_full[view_idx]))
    ref_s   = jax.jit(forward_ref_inline)(a)
    buggy_s = jax.jit(forward_buggy_inline)(a)
    md_s    = float(jnp.abs(ref_s - buggy_s).max())
    print(f"  T9b  in-jit no-vmap  view={view_idx:>3d}      max|diff| = {md_s:.4e}   "
          f"{'✗ BUG' if md_s > 1e-3 else '✓ ok '}")


# ──────────────────────────────────────────────────────────────────────────────
print()
print("─" * 70)
print("T10–T13: minimize the T9a reproducer along several axes")
print("─" * 70)

# Reusable factory: build forward_ref / forward_buggy bound to a given
# (xt, yt) pixel set, then vmap over a chosen set of angles.
def make_pair(xt_np, yt_np):
    """Return (jitted_ref_vmap, jitted_buggy_vmap, voxel) functions
    parameterized by an angles tensor."""
    xt_j = jnp.array(xt_np.astype(np.float32))
    yt_j = jnp.array(yt_np.astype(np.float32))
    NP_local = xt_j.shape[0]
    # voxel shape must match the inner kernel
    np.random.seed(0)
    vox = jnp.array(np.random.rand(NP_local, N_ROWS).astype(np.float32))

    def geom(angle):
        ct = jnp.cos(angle); st = jnp.sin(angle)
        x_p   = ct * xt_j - st * yt_j
        n_p   = (x_p + OFF_prod) / DDC_prod + DCC_prod
        n_pc  = jnp.round(n_p).astype(jnp.int32)
        cos_a = jnp.maximum(jnp.abs(ct), jnp.abs(st))
        W_pc  = (DV_prod / DDC_prod) * cos_a
        L_mx  = jnp.minimum(1.0, W_pc)
        return n_p, n_pc, W_pc, cos_a, L_mx

    def fwd_ref(angle):
        n_p, n_pc, W_pc, cos_a, L_mx = geom(angle)
        sino = jnp.zeros((N_ROWS, ND), dtype=jnp.float32)
        for no in (-1, 0, 1):
            n     = n_pc + no
            abs_d = jnp.abs(n_p - n)
            L     = jnp.clip((W_pc + 1.0) / 2.0 - abs_d, 0.0, L_mx)
            A     = DV_prod * L / cos_a * ((n >= 0) & (n < ND))
            sino  = sino.at[:, n].add(A * vox.T)
        return sino

    def fwd_buggy(angle):
        n_p, n_pc, W_pc, cos_a, L_mx = geom(angle)
        vb_batched = vox.T.reshape(_NB, _B, NP_local)
        def body(vb):
            sb = jnp.zeros((_B, ND), dtype=jnp.float32)
            for no in (-1, 0, 1):
                n     = n_pc + no
                abs_d = jnp.abs(n_p - n)
                L     = jnp.clip((W_pc + 1.0) / 2.0 - abs_d, 0.0, L_mx)
                A     = DV_prod * L / cos_a * ((n >= 0) & (n < ND))
                sb    = sb.at[:, n].add(A * vb)
            return sb
        return jax.lax.map(body, vb_batched).reshape(N_ROWS, ND)

    return fwd_ref, fwd_buggy


def run_vmap(label, xt_np, yt_np, view_indices):
    fr, fb = make_pair(xt_np, yt_np)
    angs = jnp.array(angles_full[list(view_indices)].astype(np.float32))
    ref_v   = jax.jit(jax.vmap(fr))(angs)
    buggy_v = jax.jit(jax.vmap(fb))(angs)
    md = float(jnp.abs(ref_v - buggy_v).max())
    tag = "✗ BUG" if md > 1e-3 else "✓ ok "
    print(f"  {label:<60s}  max|diff|={md:.4e}  {tag}")
    return md

# T10 — sanity: use_ror_mask=False with vmap (should NOT bug)
ridx_off = np.asarray(gen_pixel_partition(recon_shape, use_ror_mask=False))
pb_off = ridx_off[BAD * NP : (BAD + 1) * NP]
ri_off, ci_off = np.unravel_index(pb_off, recon_shape[:2])
yt_off = DV_prod * (ri_off - (recon_shape[0] - 1) / 2.0)
xt_off = DV_prod * (ci_off - (recon_shape[1] - 1) / 2.0)
run_vmap("T10 ROR=False  vmap[12,134]   (sanity — should be clean)",
         xt_off, yt_off, (12, 134))

# T11 — does the vmap need both views, or is one enough?
run_vmap("T11a ROR=True  vmap[12]        (single view in vmap)",
         xt_prod, yt_prod, (12,))
run_vmap("T11b ROR=True  vmap[134]       (single view in vmap)",
         xt_prod, yt_prod, (134,))

# T12 — pair view 12 with an irrelevant view; does it still trigger?
run_vmap("T12a ROR=True  vmap[12, 0]    (12 + a clean view)",
         xt_prod, yt_prod, (12, 0))
run_vmap("T12b ROR=True  vmap[12, 12]   (12 paired with itself)",
         xt_prod, yt_prod, (12, 12))

# T13 — smaller recon (still ROR-masked), batch 0 (always exists).
for rs in [(128, 128, 128), (64, 64, 128), (32, 32, 128)]:
    idx_small = np.asarray(gen_pixel_partition(rs, use_ror_mask=True))
    if NP > len(idx_small):
        print(f"  T13  recon={rs}  -- can't even fit one batch of {NP} -- SKIP")
        continue
    pb_s = idx_small[0:NP]
    ri_s, ci_s = np.unravel_index(pb_s, rs[:2])
    yt_s = DV_prod * (ri_s - (rs[0] - 1) / 2.0)
    xt_s = DV_prod * (ci_s - (rs[1] - 1) / 2.0)
    run_vmap(f"T13 ROR=True  recon={rs}  vmap[12]  BAD=0",
             xt_s, yt_s, (12,))

# T14 — scan BAD over 256×256 ROR to see which batches fire.
print()
print("  T14: BAD scan on 256×256 ROR  (which pixel batches trigger?)")
n_total_batches = len(all_indices) // NP
for B in range(0, n_total_batches):
    pb_b = all_indices[B * NP : (B + 1) * NP]
    ri_b, ci_b = np.unravel_index(pb_b, recon_shape[:2])
    yt_b = DV_prod * (ri_b - (recon_shape[0] - 1) / 2.0)
    xt_b = DV_prod * (ci_b - (recon_shape[1] - 1) / 2.0)
    fr, fb = make_pair(xt_b, yt_b)
    a = jnp.array(angles_full[[12]].astype(np.float32))
    ref_v   = jax.jit(jax.vmap(fr))(a)
    buggy_v = jax.jit(jax.vmap(fb))(a)
    md = float(jnp.abs(ref_v - buggy_v).max())
    tag = "✗ BUG" if md > 1e-3 else "✓ ok "
    print(f"    BAD={B:>2d}  rows={ri_b.min():>3d}..{ri_b.max():>3d}  "
          f"max|diff|={md:.4e}  {tag}")


# ──────────────────────────────────────────────────────────────────────────────
print()
print("─" * 70)
print("T15: does safe_round() perturbation kill the bug?")
print("─" * 70)
print()
print("  safe_round(x, scale=1e-4):")
print("    snapped = round(x / scale) * scale     # snap to 1e-4 grid")
print("    return  round(snapped + 0.5*scale)     # nudge past half-integer")
print()


def safe_round(x, scale=1e-4):
    """Round x to int32, robust to last-bit-different recomputations of x.

    The first round snaps any x that lives within scale/2 of a 1e-4 grid
    point to that grid point.  Two upstream chains that compute x to
    last-bit-different float values both land on the same grid point.
    The +0.5*scale nudge then guarantees the final round() never sees an
    exact half-integer tie, so its result is independent of any CSE-
    duplicate chain's floating-point ordering.
    """
    snapped = jnp.round(x / scale) * scale
    return jnp.round(snapped + 0.5 * scale).astype(jnp.int32)


def make_pair_safe(xt_np, yt_np):
    """Same as make_pair, but uses safe_round in place of jnp.round for n_pc."""
    xt_j = jnp.array(xt_np.astype(np.float32))
    yt_j = jnp.array(yt_np.astype(np.float32))
    NP_local = xt_j.shape[0]
    np.random.seed(0)
    vox = jnp.array(np.random.rand(NP_local, N_ROWS).astype(np.float32))

    def geom(angle):
        ct = jnp.cos(angle); st = jnp.sin(angle)
        x_p   = ct * xt_j - st * yt_j
        n_p   = (x_p + OFF_prod) / DDC_prod + DCC_prod
        n_pc  = safe_round(n_p)                  # ← only change vs make_pair
        cos_a = jnp.maximum(jnp.abs(ct), jnp.abs(st))
        W_pc  = (DV_prod / DDC_prod) * cos_a
        L_mx  = jnp.minimum(1.0, W_pc)
        return n_p, n_pc, W_pc, cos_a, L_mx

    def fwd_ref(angle):
        n_p, n_pc, W_pc, cos_a, L_mx = geom(angle)
        sino = jnp.zeros((N_ROWS, ND), dtype=jnp.float32)
        for no in (-1, 0, 1):
            n     = n_pc + no
            abs_d = jnp.abs(n_p - n)
            L     = jnp.clip((W_pc + 1.0) / 2.0 - abs_d, 0.0, L_mx)
            A     = DV_prod * L / cos_a * ((n >= 0) & (n < ND))
            sino  = sino.at[:, n].add(A * vox.T)
        return sino

    def fwd_buggy(angle):
        n_p, n_pc, W_pc, cos_a, L_mx = geom(angle)
        vb_batched = vox.T.reshape(_NB, _B, NP_local)
        def body(vb):
            sb = jnp.zeros((_B, ND), dtype=jnp.float32)
            for no in (-1, 0, 1):
                n     = n_pc + no
                abs_d = jnp.abs(n_p - n)
                L     = jnp.clip((W_pc + 1.0) / 2.0 - abs_d, 0.0, L_mx)
                A     = DV_prod * L / cos_a * ((n >= 0) & (n < ND))
                sb    = sb.at[:, n].add(A * vb)
            return sb
        return jax.lax.map(body, vb_batched).reshape(N_ROWS, ND)

    return fwd_ref, fwd_buggy


# T15a — does safe_round fix the original bug config (BAD=8, vmap[12])?
pb_8 = all_indices[8 * NP : 9 * NP]
ri_8, ci_8 = np.unravel_index(pb_8, recon_shape[:2])
yt_8 = DV_prod * (ri_8 - (recon_shape[0] - 1) / 2.0)
xt_8 = DV_prod * (ci_8 - (recon_shape[1] - 1) / 2.0)
fr_safe, fb_safe  = make_pair_safe(xt_8, yt_8)
fr_van,  fb_van   = make_pair(xt_8, yt_8)
a12 = jnp.array(angles_full[[12]].astype(np.float32))

# Baseline: confirm the bug is present without safe_round
ref_van   = jax.jit(jax.vmap(fr_van))(a12)
buggy_van = jax.jit(jax.vmap(fb_van))(a12)
md_van = float(jnp.abs(ref_van - buggy_van).max())

# With safe_round
ref_safe   = jax.jit(jax.vmap(fr_safe))(a12)
buggy_safe = jax.jit(jax.vmap(fb_safe))(a12)
md_safe = float(jnp.abs(ref_safe - buggy_safe).max())

print(f"  T15a  BAD=8  vmap[12]")
print(f"        vanilla round:  max|diff|(buggy vs ref) = {md_van:.4e}  "
      f"{'✗ BUG' if md_van > 1e-3 else '✓ ok '}  (expected: BUG)")
print(f"        safe_round:     max|diff|(buggy vs ref) = {md_safe:.4e}  "
      f"{'✗ BUG' if md_safe > 1e-3 else '✓ ok '}  (expected: ok)")

# How different is safe_round's reference from vanilla round's reference?
# (Only tie-pixels should differ; everywhere else float-noise.)
md_refdiff = float(jnp.abs(ref_safe - ref_van).max())
print(f"        ref_safe vs ref_vanilla:  max|diff| = {md_refdiff:.4e}  "
      f"(non-tie pixels agree; small diff expected at any tie sites)")

# T15b — BAD scan with safe_round (mirrors T14 but with the fix applied)
print()
print("  T15b: BAD scan with safe_round (expect all ✓ ok)")
n_bug_safe = 0
worst_safe = 0.0
for B in range(0, n_total_batches):
    pb_b = all_indices[B * NP : (B + 1) * NP]
    ri_b, ci_b = np.unravel_index(pb_b, recon_shape[:2])
    yt_b = DV_prod * (ri_b - (recon_shape[0] - 1) / 2.0)
    xt_b = DV_prod * (ci_b - (recon_shape[1] - 1) / 2.0)
    fr, fb = make_pair_safe(xt_b, yt_b)
    a = jnp.array(angles_full[[12]].astype(np.float32))
    ref_v   = jax.jit(jax.vmap(fr))(a)
    buggy_v = jax.jit(jax.vmap(fb))(a)
    md = float(jnp.abs(ref_v - buggy_v).max())
    worst_safe = max(worst_safe, md)
    if md > 1e-3:
        n_bug_safe += 1
        print(f"    BAD={B:>2d}  max|diff|={md:.4e}  ✗ BUG")
print(f"  → with safe_round: {n_bug_safe} of {n_total_batches} batches bug "
      f"(was 1 with vanilla round, see T14).  worst max|diff| = {worst_safe:.4e}")


# ──────────────────────────────────────────────────────────────────────────────
print()
print("─" * 70)
print("T15c–e: alternative round-function fixes — does any kill the bug everywhere?")
print("─" * 70)


def make_pair_with_round(xt_np, yt_np, round_fn):
    """Parametric: replace jnp.round in geom with `round_fn`."""
    xt_j = jnp.array(xt_np.astype(np.float32))
    yt_j = jnp.array(yt_np.astype(np.float32))
    NP_local = xt_j.shape[0]
    np.random.seed(0)
    vox = jnp.array(np.random.rand(NP_local, N_ROWS).astype(np.float32))

    def geom(angle):
        ct = jnp.cos(angle); st = jnp.sin(angle)
        x_p   = ct * xt_j - st * yt_j
        n_p   = (x_p + OFF_prod) / DDC_prod + DCC_prod
        n_pc  = round_fn(n_p)
        cos_a = jnp.maximum(jnp.abs(ct), jnp.abs(st))
        W_pc  = (DV_prod / DDC_prod) * cos_a
        L_mx  = jnp.minimum(1.0, W_pc)
        return n_p, n_pc, W_pc, cos_a, L_mx

    def fwd_ref(angle):
        n_p, n_pc, W_pc, cos_a, L_mx = geom(angle)
        sino = jnp.zeros((N_ROWS, ND), dtype=jnp.float32)
        for no in (-1, 0, 1):
            n     = n_pc + no
            abs_d = jnp.abs(n_p - n)
            L     = jnp.clip((W_pc + 1.0) / 2.0 - abs_d, 0.0, L_mx)
            A     = DV_prod * L / cos_a * ((n >= 0) & (n < ND))
            sino  = sino.at[:, n].add(A * vox.T)
        return sino

    def fwd_buggy(angle):
        n_p, n_pc, W_pc, cos_a, L_mx = geom(angle)
        vb_batched = vox.T.reshape(_NB, _B, NP_local)
        def body(vb):
            sb = jnp.zeros((_B, ND), dtype=jnp.float32)
            for no in (-1, 0, 1):
                n     = n_pc + no
                abs_d = jnp.abs(n_p - n)
                L     = jnp.clip((W_pc + 1.0) / 2.0 - abs_d, 0.0, L_mx)
                A     = DV_prod * L / cos_a * ((n >= 0) & (n < ND))
                sb    = sb.at[:, n].add(A * vb)
            return sb
        return jax.lax.map(body, vb_batched).reshape(N_ROWS, ND)

    return fwd_ref, fwd_buggy


def bad_scan_with(round_fn, label):
    """Sweep BAD = 0..23 with the given round_fn, report any bugs."""
    bad = []
    worst = 0.0
    for B in range(0, n_total_batches):
        pb_b = all_indices[B * NP : (B + 1) * NP]
        ri_b, ci_b = np.unravel_index(pb_b, recon_shape[:2])
        yt_b = DV_prod * (ri_b - (recon_shape[0] - 1) / 2.0)
        xt_b = DV_prod * (ci_b - (recon_shape[1] - 1) / 2.0)
        fr, fb = make_pair_with_round(xt_b, yt_b, round_fn)
        a = jnp.array(angles_full[[12]].astype(np.float32))
        ref_v   = jax.jit(jax.vmap(fr))(a)
        buggy_v = jax.jit(jax.vmap(fb))(a)
        md = float(jnp.abs(ref_v - buggy_v).max())
        worst = max(worst, md)
        if md > 1e-3:
            bad.append((B, md))
    if not bad:
        print(f"  {label:<55s}  ✓ all 24 batches clean (worst {worst:.2e})")
    else:
        details = ", ".join(f"BAD={B}({md:.3e})" for B, md in bad)
        print(f"  {label:<55s}  ✗ {len(bad)} bad: {details}")
    return len(bad), worst


# ── Define variants ──────────────────────────────────────────────────────

# Baseline (no fix): vanilla jnp.round
def round_vanilla(x):
    return jnp.round(x).astype(jnp.int32)

# T15c: smaller scale safe_round.  At scale 1e-6, x / scale is ≈ 1.7e8;
# float32 ULP at that magnitude (~16) is so much larger than 0.5 that the
# first-round tie set is unrepresentable, so the first round can't break
# ties inconsistently.  The quantization step then rounds to ~16-unit grid
# (effectively a no-op for the scale-1e-6 perturbation), but the +0.5e-6
# offset is still the perturbation that pushes past the FINAL round's tie.
def round_safe_1e6(x):
    scale = 1e-6
    snapped = jnp.round(x / scale) * scale
    return jnp.round(snapped + 0.5 * scale).astype(jnp.int32)

# T15d: optimization_barrier alone — addresses the (hypothesized) root cause
# directly by preventing XLA from CSE-duplicating the round computation.
def round_barrier(x):
    n_pc = jnp.round(x).astype(jnp.int32)
    return jax.lax.optimization_barrier(n_pc)

# T15e: safe_round (1e-4) + optimization_barrier — belt and suspenders.
def round_safe_plus_barrier(x):
    scale = 1e-4
    snapped = jnp.round(x / scale) * scale
    n_pc = jnp.round(snapped + 0.5 * scale).astype(jnp.int32)
    return jax.lax.optimization_barrier(n_pc)


# ── Run all variants over the 24-batch scan ──────────────────────────────
print()
print("  (worst max|diff| over all 24 batches; 'clean' threshold = 1e-3)")
print()
bad_scan_with(round_vanilla,           "T14 (vanilla, baseline)")
bad_scan_with(safe_round,              "T15a/b (safe_round, scale=1e-4)")
bad_scan_with(round_safe_1e6,          "T15c (safe_round, scale=1e-6)")
bad_scan_with(round_barrier,           "T15d (jnp.round + optimization_barrier)")
bad_scan_with(round_safe_plus_barrier, "T15e (safe_round + optimization_barrier)")


# T15f: safe_round in float64.  Idea: cast to float64 internally, so the
# arithmetic inside safe_round has ~16 digits of precision instead of ~7.
# This won't add information to a float32 input (the value is the same after
# cast) but it lets the snap-and-perturb steps avoid the float32 ULP issues
# that turned T15c into a no-op, and it spreads any tie-condition on a
# much finer grid.
def round_safe_f64(scale):
    def _r(x):
        x64 = x.astype(jnp.float64)
        snapped = jnp.round(x64 / scale) * scale
        return jnp.round(snapped + 0.5 * scale).astype(jnp.int32)
    return _r


# Cast n_p input to float64 then do safe_round at three scales.
bad_scan_with(round_safe_f64(1e-4),    "T15f safe_round_f64 (scale=1e-4)")
bad_scan_with(round_safe_f64(1e-6),    "T15g safe_round_f64 (scale=1e-6)")
bad_scan_with(round_safe_f64(1e-8),    "T15h safe_round_f64 (scale=1e-8)")


# ── T15i: precompute geometry outside the jit ────────────────────────────
# If this is clean across all batches, the bug requires `round` to live
# inside the vmap+lax.map boundary, and the fix is to precompute the
# (n_p, n_pc, W_pc, cos_a, L_mx) tuple in numpy and pass it in.

def make_pair_precomputed():
    """fwd_ref / fwd_buggy that take precomputed geometry as JAX inputs.
    No cos/sin/round inside.  Voxel data is local because NP depends on
    the pixel batch."""
    def make(NP_local):
        np.random.seed(0)
        vox = jnp.array(np.random.rand(NP_local, N_ROWS).astype(np.float32))

        def fwd_ref(n_p, n_pc, W_pc, cos_a, L_mx):
            sino = jnp.zeros((N_ROWS, ND), dtype=jnp.float32)
            for no in (-1, 0, 1):
                n     = n_pc + no
                abs_d = jnp.abs(n_p - n)
                L     = jnp.clip((W_pc + 1.0) / 2.0 - abs_d, 0.0, L_mx)
                A     = DV_prod * L / cos_a * ((n >= 0) & (n < ND))
                sino  = sino.at[:, n].add(A * vox.T)
            return sino

        def fwd_buggy(n_p, n_pc, W_pc, cos_a, L_mx):
            vb_batched = vox.T.reshape(_NB, _B, NP_local)
            def body(vb):
                sb = jnp.zeros((_B, ND), dtype=jnp.float32)
                for no in (-1, 0, 1):
                    n     = n_pc + no
                    abs_d = jnp.abs(n_p - n)
                    L     = jnp.clip((W_pc + 1.0) / 2.0 - abs_d, 0.0, L_mx)
                    A     = DV_prod * L / cos_a * ((n >= 0) & (n < ND))
                    sb    = sb.at[:, n].add(A * vb)
                return sb
            return jax.lax.map(body, vb_batched).reshape(N_ROWS, ND)

        return fwd_ref, fwd_buggy
    return make

def bad_scan_precomputed(label, view_idx=12):
    """Like bad_scan_with but with all geometry computed in numpy and
    passed in as JAX arrays."""
    make = make_pair_precomputed()
    bad, worst = [], 0.0
    angle = np.float32(angles_full[view_idx])
    ct = np.cos(angle).astype(np.float32)
    st = np.sin(angle).astype(np.float32)
    cos_a_np = np.float32(max(abs(ct), abs(st)))
    W_pc_np  = np.float32((DV_prod / DDC_prod) * cos_a_np)
    L_mx_np  = np.float32(min(1.0, float(W_pc_np)))
    for B in range(0, n_total_batches):
        pb_b = all_indices[B * NP : (B + 1) * NP]
        ri_b, ci_b = np.unravel_index(pb_b, recon_shape[:2])
        yt_b = (DV_prod * (ri_b - (recon_shape[0] - 1) / 2.0)).astype(np.float32)
        xt_b = (DV_prod * (ci_b - (recon_shape[1] - 1) / 2.0)).astype(np.float32)
        x_p_np  = ct * xt_b - st * yt_b
        n_p_np  = ((x_p_np + OFF_prod) / DDC_prod + DCC_prod).astype(np.float32)
        n_pc_np = np.round(n_p_np).astype(np.int32)
        # Build vmap-batched inputs (size-1 vmap, mirroring T11a)
        n_p_v  = jnp.array(n_p_np[None, :])
        n_pc_v = jnp.array(n_pc_np[None, :])
        W_v    = jnp.array(np.array([W_pc_np]))
        ca_v   = jnp.array(np.array([cos_a_np]))
        L_v    = jnp.array(np.array([L_mx_np]))

        fr, fb = make(NP)
        ref_v   = jax.jit(jax.vmap(fr))(n_p_v, n_pc_v, W_v, ca_v, L_v)
        buggy_v = jax.jit(jax.vmap(fb))(n_p_v, n_pc_v, W_v, ca_v, L_v)
        md = float(jnp.abs(ref_v - buggy_v).max())
        worst = max(worst, md)
        if md > 1e-3:
            bad.append((B, md))
    if not bad:
        print(f"  {label:<55s}  ✓ all 24 batches clean (worst {worst:.2e})")
    else:
        details = ", ".join(f"BAD={B}({md:.3e})" for B, md in bad)
        print(f"  {label:<55s}  ✗ {len(bad)} bad: {details}")

bad_scan_precomputed("T15i precomputed geom (round outside the jit)")


# ── T15j: precompute n_pc only, leave the rest of geom in-jit ────────────
# This tests whether the minimal change is sufficient: keep cos/sin/n_p/
# W_pc/cos_alpha/L_max derivation inside the jit (as in the current
# production code), and only move the integer scatter index n_pc out.

def make_pair_npc_only(xt_np, yt_np):
    xt_j = jnp.array(xt_np.astype(np.float32))
    yt_j = jnp.array(yt_np.astype(np.float32))
    NP_local = xt_j.shape[0]
    np.random.seed(0)
    vox = jnp.array(np.random.rand(NP_local, N_ROWS).astype(np.float32))

    def _geom_inline_no_round(angle):
        """Same as _geom_inline but n_pc is NOT computed here."""
        ct = jnp.cos(angle); st = jnp.sin(angle)
        x_p   = ct * xt_j - st * yt_j
        n_p   = (x_p + OFF_prod) / DDC_prod + DCC_prod
        cos_a = jnp.maximum(jnp.abs(ct), jnp.abs(st))
        W_pc  = (DV_prod / DDC_prod) * cos_a
        L_mx  = jnp.minimum(1.0, W_pc)
        return n_p, W_pc, cos_a, L_mx

    def fwd_ref(angle, n_pc):
        n_p, W_pc, cos_a, L_mx = _geom_inline_no_round(angle)
        sino = jnp.zeros((N_ROWS, ND), dtype=jnp.float32)
        for no in (-1, 0, 1):
            n     = n_pc + no
            abs_d = jnp.abs(n_p - n)
            L     = jnp.clip((W_pc + 1.0) / 2.0 - abs_d, 0.0, L_mx)
            A     = DV_prod * L / cos_a * ((n >= 0) & (n < ND))
            sino  = sino.at[:, n].add(A * vox.T)
        return sino

    def fwd_buggy(angle, n_pc):
        n_p, W_pc, cos_a, L_mx = _geom_inline_no_round(angle)
        vb_batched = vox.T.reshape(_NB, _B, NP_local)
        def body(vb):
            sb = jnp.zeros((_B, ND), dtype=jnp.float32)
            for no in (-1, 0, 1):
                n     = n_pc + no
                abs_d = jnp.abs(n_p - n)
                L     = jnp.clip((W_pc + 1.0) / 2.0 - abs_d, 0.0, L_mx)
                A     = DV_prod * L / cos_a * ((n >= 0) & (n < ND))
                sb    = sb.at[:, n].add(A * vb)
            return sb
        return jax.lax.map(body, vb_batched).reshape(N_ROWS, ND)

    return fwd_ref, fwd_buggy


def bad_scan_npc_only(label, view_idx=12):
    bad, worst = [], 0.0
    angle = np.float32(angles_full[view_idx])
    ct = np.cos(angle).astype(np.float32)
    st = np.sin(angle).astype(np.float32)
    for B in range(0, n_total_batches):
        pb_b = all_indices[B * NP : (B + 1) * NP]
        ri_b, ci_b = np.unravel_index(pb_b, recon_shape[:2])
        yt_b = (DV_prod * (ri_b - (recon_shape[0] - 1) / 2.0)).astype(np.float32)
        xt_b = (DV_prod * (ci_b - (recon_shape[1] - 1) / 2.0)).astype(np.float32)
        x_p_np  = ct * xt_b - st * yt_b
        n_p_np  = ((x_p_np + OFF_prod) / DDC_prod + DCC_prod).astype(np.float32)
        n_pc_np = np.round(n_p_np).astype(np.int32)

        fr, fb = make_pair_npc_only(xt_b, yt_b)
        a_v   = jnp.array(np.array([angle]))
        npc_v = jnp.array(n_pc_np[None, :])
        ref_v   = jax.jit(jax.vmap(fr))(a_v, npc_v)
        buggy_v = jax.jit(jax.vmap(fb))(a_v, npc_v)
        md = float(jnp.abs(ref_v - buggy_v).max())
        worst = max(worst, md)
        if md > 1e-3:
            bad.append((B, md))
    if not bad:
        print(f"  {label:<55s}  ✓ all 24 batches clean (worst {worst:.2e})")
    else:
        details = ", ".join(f"BAD={B}({md:.3e})" for B, md in bad)
        print(f"  {label:<55s}  ✗ {len(bad)} bad: {details}")

bad_scan_npc_only("T15j precompute n_pc only (rest of geom in-jit)")


print()
print("=" * 70)
print("Done.")
print("=" * 70)
