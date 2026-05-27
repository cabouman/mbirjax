"""
experiments/sharding/vmap_lax_map_demo.py
──────────────────────────────────────────
Annotated reproducer for a JAX/XLA bug:

  jax.lax.scan (= jax.lax.map) + scatter-add produces incorrect results
  for specific integer scatter-index patterns from the mbirjax pixel grid.

CONTEXT
───────
mbirjax's parallel-beam CT forward projector uses jax.lax.map to iterate
a scatter-add kernel over batches of reconstruction slices.  With the
production geometry (integer pixel grid, N_ROWS=128, N_CHANNELS=256), two
specific view angles (views 12 and 134 of 180) produce sinogram values
that differ from the correct answer by ≈1.7 intensity units.

SYMPTOMS
────────
• Error ≈ 1.7 at views 12 and 134.
• Antisymmetric: energy is swapped between the channel just below and
  just above each pixel's nearest detector channel (n_pc-1 ↔ n_pc+1).
• Only PSF offsets ±1 are affected; offset 0 is always correct.
• The bug does NOT require jax.vmap: it appears even when lax.map is
  called directly (one view at a time) with concrete numpy geometry.

ROOT CAUSE
──────────
The bug lives strictly below JAX's JAXPR level.  Evidence:

  1. The scan body JAxPRs for the "angle-derived geometry" and
     "explicit geometry" variants are bitwise identical (115 lines each),
     yet one variant is buggy.  → Bug is in XLA compilation, not JAXPR.

  2. jax.vmap(psb)(db) [same semantics as lax.map] is correct.
     → Bug is specific to lax.scan's lowering of scatter-add.

  3. The XLA scatter-add kernel appears to mis-handle the case where
     the integer scatter index (n_pc ± 1) hits specific channel values
     that arise from the production integer pixel grid.

FIX
───
Replace  jax.lax.map(project_slice_batch, voxel_batched)
with     jax.vmap(project_slice_batch)(voxel_batched).

Both map project_slice_batch over the leading axis of voxel_batched;
jax.vmap uses a different XLA lowering path that produces correct results.

DEPENDENCIES
────────────
  • mbirjax (for parallel-beam geometry parameters)
  • experiments/sharding/sharding_baseline_ref.npz  (pixel batch data)

Run:
  conda run -n mbirjax python experiments/sharding/vmap_lax_map_demo.py
"""

import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
_sd = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_sd, '../..', '..')))

import numpy as np
import jax
import jax.numpy as jnp
import mbirjax
from collections import namedtuple

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

print("=" * 70)
print("  vmap_lax_map_demo.py — lax.scan + scatter-add bug reproducer")
print("=" * 70)
print(f"  JAX version : {jax.__version__}")
print()

# ── mbirjax model and geometry ────────────────────────────────────────────────
N_VIEWS = 180;  N_ROWS = 128;  N_CHANNELS = 256
angles_np  = np.linspace(0, np.pi, N_VIEWS, endpoint=False)
angles_jax = jnp.array(angles_np.astype(np.float32))
angles_jax = np.array((angles_jax[12], angles_jax[134]))

model = mbirjax.ParallelBeamModel((N_VIEWS, N_ROWS, N_CHANNELS), angles_np)
model.set_params(no_compile=True, no_warning=True)
PP = namedtuple('PP', ['sinogram_shape', 'recon_shape', 'geometry_params'])
recon_shape = model.get_params('recon_shape')
pp = PP(model.get_params('sinogram_shape'),
        recon_shape,
        model.get_geometry_parameters())
gp    = pp.geometry_params
PSFSZ = int(gp.psf_radius)           # PSF half-width (= 1 for production)
DV    = float(gp.delta_voxel)
DDC   = float(gp.delta_det_channel)
OFF   = float(gp.det_channel_offset)
ND    = N_CHANNELS
_B    = 16                            # slice batch size (_B in parallel_beam.py)
_NB   = N_ROWS // _B                  # number of lax.map steps

PBSZ = 2048  # Pixel batch size for vmap
all_indices_np = gen_pixel_partition(recon_shape, use_ror_mask=True)
flat_recon_np = np.random.rand(len(all_indices_np), N_ROWS)

print(f"  PSF radius : {PSFSZ}")
print(f"  DV={DV}, DDC={DDC}, OFF={OFF}")
print(f"  _B={_B}, _NB={_NB}, ND={ND}")
print()

# ── Select pixel batch 8 (contains views 12/134 with the bug) ─────────────────
BAD = 8
vb_all = jnp.array(flat_recon_np [BAD*PBSZ:(BAD+1)*PBSZ])   # (NP, N_ROWS) voxel values
pb     = jnp.array(all_indices_np[BAD*PBSZ:(BAD+1)*PBSZ])   # (NP,)        pixel indices
NP     = vb_all.shape[0]

# Convert flat pixel indices to (y, x) physical coordinates
_rs = pp.recon_shape
ri_np, ci_np = np.unravel_index(np.asarray(pb), _rs[:2])
yt_arr = jnp.array((DV * (ri_np - (_rs[0]-1)/2.0)).astype(np.float32))  # (NP,)
xt_arr = jnp.array((DV * (ci_np - (_rs[1]-1)/2.0)).astype(np.float32))  # (NP,)
DCC = (ND - 1) / 2.0

# Reshape voxels for lax.map: lax.map steps over the NB leading axis,
# delivering one (_B, NP) slice batch per step.
# vb_all.T has shape (N_ROWS, NP); reshape gives (NB, _B, NP).
vb_batched = vb_all.T.reshape(_NB, _B, NP)   # (_NB, _B, NP)

print(f"  Pixel batch: NP={NP}, voxel range=[{float(vb_all.min()):.3f}, {float(vb_all.max()):.3f}]")
print()

# ── Geometry helper ───────────────────────────────────────────────────────────
def _geom(angle):
    """
    Compute per-pixel projection geometry for one view angle.
    All returned values are derived from the vmapped `angle`; they carry
    the vmap batch dimension and become scan-body constants under lax.map.
    """
    ct = jnp.cos(angle);  st = jnp.sin(angle)
    x_p  = ct * xt_arr - st * yt_arr           # projected pixel position (NP,)
    n_p  = (x_p + OFF) / DDC + DCC           # fractional channel index (NP,)
    n_pc = jnp.round(n_p).astype(jnp.int32)    # nearest channel          (NP,)
    cos_a = jnp.maximum(jnp.abs(ct), jnp.abs(st))
    W_pc  = (DV / DDC) * cos_a                 # PSF full-width (scalar)
    L_mx  = jnp.minimum(1.0, W_pc)             # PSF amplitude clamp
    return n_p, n_pc, W_pc, cos_a, L_mx

# ══════════════════════════════════════════════════════════════════════════════
# Three implementations
# ══════════════════════════════════════════════════════════════════════════════

def forward_buggy(angle):
    """
    BUGGY: lax.map over slice batches.

    project_slice_batch closes over n_p, n_pc, W_pc, cos_a, L_mx.
    Under jax.lax.map (= lax.scan with empty carry), XLA compiles the
    scatter-add incorrectly for the specific n_pc integer patterns that
    arise from the mbirjax production pixel grid.  The bug appears for
    PSF offsets ±1 (the adjacent-channel scatter), not for offset 0.
    """
    n_p, n_pc, W_pc, cos_a, L_mx = _geom(angle)

    def project_slice_batch(vb_batch):   # vb_batch: (_B, NP)
        sb = jnp.zeros((_B, ND))
        for no in range(-PSFSZ, PSFSZ+1):
            n     = n_pc + no
            abs_d = jnp.abs(n_p - n)
            L     = jnp.clip((W_pc + 1.0)/2.0 - abs_d, 0.0, L_mx)
            A     = DV * L / cos_a * ((n >= 0) & (n < ND))
            # Scatter: for each pixel p, add A[p]*vb_batch[:,p] into column n[p].
            # This scatter-add is mis-compiled by XLA for specific n_pc patterns.
            sb = sb.at[:, n].add(A * vb_batch)
        return sb

    return jax.lax.map(project_slice_batch, vb_batched).reshape(N_ROWS, ND)


def forward_ref(angle):
    """
    REFERENCE: no inner batching — all slices projected in one scatter call.
    Always correct.  Equivalent to the prerelease code path (Option A fix).
    """
    n_p, n_pc, W_pc, cos_a, L_mx = _geom(angle)
    sino = jnp.zeros((N_ROWS, ND))
    for no in range(-PSFSZ, PSFSZ+1):
        n     = n_pc + no
        abs_d = jnp.abs(n_p - n)
        L     = jnp.clip((W_pc + 1.0)/2.0 - abs_d, 0.0, L_mx)
        A     = DV * L / cos_a * ((n >= 0) & (n < ND))
        sino  = sino.at[:, n].add(A * vb_all.T)
    return sino


def forward_fixed(angle):
    """
    FIX: replace jax.lax.map with jax.vmap.

    Semantically identical to forward_buggy — both iterate project_slice_batch
    over the NB leading axis of vb_batched — but jax.vmap uses a different XLA
    lowering path for the scatter-add that produces correct results.
    """
    n_p, n_pc, W_pc, cos_a, L_mx = _geom(angle)

    def project_slice_batch(vb_batch):   # identical body to forward_buggy
        sb = jnp.zeros((_B, ND))
        for no in range(-PSFSZ, PSFSZ+1):
            n     = n_pc + no
            abs_d = jnp.abs(n_p - n)
            L     = jnp.clip((W_pc + 1.0)/2.0 - abs_d, 0.0, L_mx)
            A     = DV * L / cos_a * ((n >= 0) & (n < ND))
            sb    = sb.at[:, n].add(A * vb_batch)
        return sb

    return jax.vmap(project_slice_batch)(vb_batched).reshape(N_ROWS, ND)  # ← key change


# ── Run all three over all views ──────────────────────────────────────────────
ref_out   = jax.jit(jax.vmap(forward_ref)  )(angles_jax)
buggy_out = jax.jit(jax.vmap(forward_buggy))(angles_jax)
fixed_out = jax.jit(jax.vmap(forward_fixed))(angles_jax)

# ══════════════════════════════════════════════════════════════════════════════
# EVIDENCE 1 & 2: correctness
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print("EVIDENCE 1 & 2 — correctness check")
print("─" * 70)

diff_br = jnp.abs(buggy_out - ref_out)
diff_fr = jnp.abs(fixed_out - ref_out)
bug_max = float(diff_br.max())
fix_max = float(diff_fr.max())
bad_views = np.where(np.asarray(diff_br).max(axis=(1, 2)) > 0.01)[0].tolist()

print(f"  buggy vs reference : max|diff| = {bug_max:.4e}  bad_views={bad_views}  "
      f"{'✗ BUG' if bug_max > 0.01 else '?'}")
print(f"  fixed vs reference : max|diff| = {fix_max:.4e}  "
      f"{'✓ FIX CONFIRMED' if fix_max < 1e-5 else '✗'}")

# Show the antisymmetric error pattern at the worst view
for worst_view in [0, 1]:
    # worst_view = int(np.argmax(np.asarray(diff_br).max(axis=(1, 2))))
    signed_err = (buggy_out[worst_view] - ref_out[worst_view]).sum(axis=0)   # sum over slices
    err_ch     = int(jnp.argmax(jnp.abs(signed_err)))
    lo, hi     = max(0, err_ch - 3), min(ND, err_ch + 4)
    print(f"\n  Signed error (summed over slices) at view {worst_view}, "
          f"channels {lo}–{hi-1}:")
    start_channel = -1
    for ch in range(lo, hi):
        e   = float(signed_err[ch])
        tag = "← missing" if e < -0.01 else ("→ excess" if e > 0.01 else "")
        if np.abs(e) > 0.01 and start_channel < 0:
            start_channel = ch
        print(f"    ch {ch:3d}: {e:+.4f}  {tag}")
    print()
    print("  The error is antisymmetric: energy swapped between n_pc-1 and")
    print("  n_pc+1 channels.  Both offsets +1 and -1 are independently buggy.")

    n_p, n_pc, W_pc, cos_a, L_mx = _geom(angles_jax[worst_view])
    print('Projections of centers of pixels')
    for j in range(3):
        a = np.where(n_pc == start_channel + j)[0]
        print('Channel {}:'.format(start_channel + j))
        with np.printoptions(precision=17, floatmode='unique'):
            print(n_p[a])
print()


# ══════════════════════════════════════════════════════════════════════════════
# EVIDENCE 3: offset isolation
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print("EVIDENCE 3 — offset isolation (which PSF offsets trigger the bug?)")
print("─" * 70)

def _make_fwd_buggy(offsets):
    def fwd(angle):
        n_p, n_pc, W_pc, cos_a, L_mx = _geom(angle)
        def psb(vb_batch):
            sb = jnp.zeros((_B, ND))
            for no in offsets:
                n     = n_pc + no
                abs_d = jnp.abs(n_p - n)
                L     = jnp.clip((W_pc+1.0)/2.0 - abs_d, 0.0, L_mx)
                A     = DV * L / cos_a * ((n >= 0) & (n < ND))
                sb    = sb.at[:, n].add(A * vb_batch)
            return sb
        return jax.lax.map(psb, vb_batched).reshape(N_ROWS, ND)
    return fwd

def _make_fwd_ref(offsets):
    def fwd(angle):
        n_p, n_pc, W_pc, cos_a, L_mx = _geom(angle)
        sino = jnp.zeros((N_ROWS, ND))
        for no in offsets:
            n     = n_pc + no
            abs_d = jnp.abs(n_p - n)
            L     = jnp.clip((W_pc+1.0)/2.0 - abs_d, 0.0, L_mx)
            A     = DV * L / cos_a * ((n >= 0) & (n < ND))
            sino  = sino.at[:, n].add(A * vb_all.T)
        return sino
    return fwd

for offsets in [[0], [-1], [1], [-1, 0, 1]]:
    ob  = jax.jit(jax.vmap(_make_fwd_buggy(offsets)))(angles_jax)
    ref = jax.jit(jax.vmap(_make_fwd_ref(offsets))  )(angles_jax)
    md  = float(jnp.abs(ob - ref).max())
    bv  = np.where(np.asarray(jnp.abs(ob-ref)).max(axis=(1,2)) > 0.01)[0].tolist()
    status = '✗ BUG' if md > 0.01 else '✓ OK '
    print(f"  offsets {str(offsets):<12}: max|diff|={md:.4e}  bad_views={bv}  {status}")

print()
print("  offset=0 is always correct.  offsets ±1 each independently trigger")
print("  the bug, confirming the issue is in the adjacent-channel scatter.")
print()


# ══════════════════════════════════════════════════════════════════════════════
# EVIDENCE 4: JAXPR scan-body comparison
#
# We compare the scan body of:
#   fn_1A — geometry passed as explicit vmap args (in_axes=0 for each array)
#   fn_1B — geometry derived from vmapped angle  (= forward_buggy)
#
# If the bodies are bitwise identical, the bug cannot be in the JAXPR; it
# must be in XLA's lowering of the batched lax.scan → scatter-add.
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print("EVIDENCE 4 — JAXPR scan-body comparison (confirms XLA-level bug)")
print("─" * 70)

# Pre-compute geometry arrays for all views (needed by fn_1A)
def _geom_np(angle_f):
    ct, st = np.cos(angle_f), np.sin(angle_f)
    xp = ct*np.asarray(xt_arr) - st*np.asarray(yt_arr)
    np_ = (xp + OFF)/DDC + DCC
    npc = np.round(np_).astype(np.int32)
    ca  = max(abs(ct), abs(st))
    Wp  = np.float32((DV/DDC)*ca)
    Lm  = np.float32(min(1.0, float(Wp)))
    return np_.astype(np.float32), npc, Wp, np.float32(ca), Lm

n_p_all  = jnp.array(np.stack([_geom_np(float(a))[0] for a in angles_np]))  # (NV,NP)
n_pc_all = jnp.array(np.stack([_geom_np(float(a))[1] for a in angles_np]))  # (NV,NP)
W_all    = jnp.array(np.array( [_geom_np(float(a))[2] for a in angles_np])) # (NV,)
cos_all  = jnp.array(np.array( [_geom_np(float(a))[3] for a in angles_np])) # (NV,)
Lmx_all  = jnp.array(np.array( [_geom_np(float(a))[4] for a in angles_np])) # (NV,)


def fn_1A(n_p, n_pc, W_pc, cos_a, L_mx):
    """Explicit geometry as vmap args — scan body identical to fn_1B."""
    def psb(vb_batch):
        sb = jnp.zeros((_B, ND))
        for no in range(-PSFSZ, PSFSZ+1):
            n     = n_pc + no
            abs_d = jnp.abs(n_p - n)
            L     = jnp.clip((W_pc+1.0)/2.0 - abs_d, 0.0, L_mx)
            A     = DV * L / cos_a * ((n >= 0) & (n < ND))
            sb    = sb.at[:, n].add(A * vb_batch)
        return sb
    return jax.lax.map(psb, vb_batched).reshape(N_ROWS, ND)


# fn_1B is identical to forward_buggy — geometry derived from vmapped angle
fn_1B = forward_buggy

jpr_1A = jax.make_jaxpr(jax.vmap(fn_1A, in_axes=(0,0,0,0,0)))(
    n_p_all, n_pc_all, W_all, cos_all, Lmx_all)
jpr_1B = jax.make_jaxpr(jax.vmap(fn_1B))(angles_jax)


def find_scan_eqn(jpr):
    for eqn in jpr.jaxpr.eqns:
        if str(eqn.primitive) == 'scan':
            return eqn
    return None


scan_1A = find_scan_eqn(jpr_1A)
scan_1B = find_scan_eqn(jpr_1B)

if scan_1A and scan_1B:
    body_str_1A = str(scan_1A.params['jaxpr'])
    body_str_1B = str(scan_1B.params['jaxpr'])
    identical   = (body_str_1A == body_str_1B)
    n_lines     = len(body_str_1A.splitlines())

    print(f"  fn_1A scan body: {n_lines} lines  "
          f"(params: length={scan_1A.params.get('length')}, "
          f"num_consts={scan_1A.params.get('num_consts')})")
    print(f"  fn_1B scan body: {len(body_str_1B.splitlines())} lines  "
          f"(params: length={scan_1B.params.get('length')}, "
          f"num_consts={scan_1B.params.get('num_consts')})")
    print(f"  Bodies identical: {identical}")
    print()

    if identical:
        print("  ✓ CONCLUSION: scan bodies are bitwise identical.")
        print("    fn_1A uses geometry arrays passed as vmapped args.")
        print("    fn_1B derives geometry from the vmapped angle.")
        print("    Both trace to the exact same 115-line scan body JAXPR.")
        print("    The bug is therefore in XLA's lowering of the batched")
        print("    lax.scan, NOT in JAX's JAXPR generation.")

    # Print the scan invars so the reader can see the batched shapes
    print()
    print("  fn_1B scan invars (closed-over constants, batched over NV views):")
    for v in scan_1B.invars:
        print(f"    {v.aval}")
    print()
    print("  int32[NV,NP] is n_pc (scatter index), batched over all views.")
    print("  XLA receives a 2-D scatter index inside the scan and compiles")
    print("  the scatter-add incorrectly for certain integer index patterns.")
else:
    print("  ERROR: could not find scan equation in JAXPR (unexpected)")

print()


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print("  Bug  :  jax.lax.scan scatter-add produces wrong results for the")
print("          specific integer scatter-index patterns arising from the")
print("          mbirjax production pixel grid.  Visible as ≈1.7-unit errors")
print("          at views 12 and 134 with antisymmetric channel swaps.")
print()
print("  Cause:  XLA mis-compiles the scatter-add for specific n_pc patterns.")
print("          The JAXPR is correct (scan bodies bitwise identical for")
print("          derived-geometry and explicit-geometry variants).")
print("          Note: the bug also appears WITHOUT jax.vmap — running one")
print("          view at a time with numpy-materialized geometry still hits it.")
print()
print("  Fix  :  Replace  jax.lax.map(f, xs)  with  jax.vmap(f)(xs).")
print("          Both iterate f over the leading axis of xs; vmap uses a")
print("          different XLA lowering path that produces correct results.")
print()
print("  Alt  :  Remove inner slice batching entirely (prerelease / Option A):")
print("          scatter all slices in one call per view.")
print()
print(f"  JAX  :  {jax.__version__}")
print()

mbirjax.slice_viewer(ref_out, 10 * (buggy_out-ref_out), slice_axis=0,
                     title='Reference projection (left) and 10*(ref - buggy) (right)')