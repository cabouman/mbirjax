"""
jax_rounding_bug_v2.py — direct probe of the CSE-failure hypothesis.

Hypothesis (from `minimal_lax_map_repro.md` §2 + discussion):
  In the buggy vmap(lax.map(...)) setup, XLA fails to deduplicate
  `jnp.round(n_p)` and produces multiple copies of `n_pc`.  For `n_p`
  values exactly at half-integers, those copies can disagree by 1
  (last-bit float diff in the round input flips the result).  The
  scatter destination and the L-weight computation then disagree on
  what `n_pc` is, and contributions land in the wrong channel.

Probe:
  Inside the bug-triggering lax.map body (T9a setup from
  minimal_lax_map_repro), in addition to running the buggy scatter,
  return the three integer arrays  `n_pc - 1`,  `n_pc + 0`,  `n_pc + 1`
  out of the lax.map.  If XLA was forced to use a single consistent
  `n_pc`, the three returned arrays will differ from each other by
  exactly 1 everywhere.  If XLA used three separately-rounded copies,
  the differences could be 0 or 2 at exactly the half-integer pixels.

Caveat: XLA may CSE the three additions trivially when they appear as
direct outputs, even though it doesn't CSE the analogous sites that
feed the scatter / L-weight in the real bug.  If we see all-1 step
sizes here, that's compatible with the bug being real but only
manifesting when n_pc is consumed in different fused subgraphs.
"""

import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import numpy as np
import jax
import jax.numpy as jnp

print("=" * 70)
print("  jax_rounding_bug_v2.py — CSE-failure probe")
print("=" * 70)
print(f"  JAX version    : {jax.__version__}")
print(f"  default backend: {jax.default_backend()}")
print()

# ── Setup matching minimal_lax_map_repro.py T9a (the bug-triggering case) ──
NP, N_ROWS, _B, ND = 2048, 128, 16, 256
_NB = N_ROWS // _B
DV_prod, DDC_prod, OFF_prod = 1.0, 1.0, 0.0
DCC_prod = (ND - 1) / 2.0
recon_shape = (256, 256, 128)
N_VIEWS = 180
angles_full = np.linspace(0, np.pi, N_VIEWS, endpoint=False)


def get_2d_ror_mask(recon_shape):
    nr, nc = recon_shape[:2]
    rc, cc = (nr - 1) / 2, (nc - 1) / 2
    radius = max(rc, cc)
    coords = np.meshgrid(np.arange(nc) - cc, np.arange(nr) - rc)
    return coords[0] ** 2 + coords[1] ** 2 <= radius ** 2


def gen_pixel_partition(recon_shape):
    nr, nc = recon_shape[:2]
    indices = np.arange(nr * nc, dtype=np.int32)
    mask = get_2d_ror_mask(recon_shape).flatten()
    return np.sort(indices[mask == 1])


BAD = 8
all_indices = gen_pixel_partition(recon_shape)
pb = all_indices[BAD * NP : (BAD + 1) * NP]
ri, ci = np.unravel_index(pb, recon_shape[:2])
yt_prod = (DV_prod * (ri - (recon_shape[0] - 1) / 2.0)).astype(np.float32)
xt_prod = (DV_prod * (ci - (recon_shape[1] - 1) / 2.0)).astype(np.float32)
xt_jnp = jnp.array(xt_prod)
yt_jnp = jnp.array(yt_prod)

np.random.seed(0)
voxel = jnp.array(np.random.rand(NP, N_ROWS).astype(np.float32))


def _geom_inline(angle):
    ct = jnp.cos(angle)
    st = jnp.sin(angle)
    x_p   = ct * xt_jnp - st * yt_jnp
    n_p   = (x_p + OFF_prod) / DDC_prod + DCC_prod
    n_pc  = jnp.round(n_p).astype(jnp.int32)
    cos_a = jnp.maximum(jnp.abs(ct), jnp.abs(st))
    W_pc  = (DV_prod / DDC_prod) * cos_a
    L_mx  = jnp.minimum(1.0, W_pc)
    return n_p, n_pc, W_pc, cos_a, L_mx


def forward_buggy_with_dests(angle):
    """T9a's buggy forward, plus return n_pc-1, n_pc+0, n_pc+1 from the lax.map."""
    n_p, n_pc, W_pc, cos_a, L_mx = _geom_inline(angle)
    vb_batched = voxel.T.reshape(_NB, _B, NP)

    def body(vb):
        sb = jnp.zeros((_B, ND), dtype=jnp.float32)
        # Real bug-triggering scatter chain — unchanged from T9a.
        for no in (-1, 0, 1):
            n     = n_pc + no
            abs_d = jnp.abs(n_p - n)
            L     = jnp.clip((W_pc + 1.0) / 2.0 - abs_d, 0.0, L_mx)
            A     = DV_prod * L / cos_a * ((n >= 0) & (n < ND))
            sb    = sb.at[:, n].add(A * vb)
        # Diagnostic outputs: the three offset destinations.
        return sb, n_pc + (-1), n_pc + 0, n_pc + 1

    sb_stack, dn1, d0, dp1 = jax.lax.map(body, vb_batched)
    return sb_stack.reshape(N_ROWS, ND), dn1, d0, dp1


def forward_ref_inline(angle):
    """Reference (no lax.map) — to confirm the bug is present in this setup."""
    n_p, n_pc, W_pc, cos_a, L_mx = _geom_inline(angle)
    sino = jnp.zeros((N_ROWS, ND), dtype=jnp.float32)
    for no in (-1, 0, 1):
        n     = n_pc + no
        abs_d = jnp.abs(n_p - n)
        L     = jnp.clip((W_pc + 1.0) / 2.0 - abs_d, 0.0, L_mx)
        A     = DV_prod * L / cos_a * ((n >= 0) & (n < ND))
        sino  = sino.at[:, n].add(A * voxel.T)
    return sino


# ── Run on view 12, with the same vmap wrapper that triggers the bug ──────
view_idx = 12
angles_jax = jnp.array(angles_full[[view_idx]].astype(np.float32))

sino_b, dn1, d0, dp1 = jax.jit(jax.vmap(forward_buggy_with_dests))(angles_jax)
sino_r = jax.jit(jax.vmap(forward_ref_inline))(angles_jax)

# Sanity: confirm the bug is actually triggered in this code path
md = float(jnp.abs(sino_b - sino_r).max())
status = "BUG present" if md > 1e-3 else "no bug — test below may not be diagnostic"
print(f"Sanity check:  buggy vs ref  max|diff| = {md:.4e}   ({status})")
print()

# ── Check the step sizes between offset destinations ──────────────────────
dn1_np = np.asarray(dn1[0])   # shape (_NB, NP)
d0_np  = np.asarray(d0[0])
dp1_np = np.asarray(dp1[0])

step_low  = d0_np  - dn1_np   # expected 1 everywhere
step_high = dp1_np - d0_np    # expected 1 everywhere

print(f"step_low  = (n_pc+0) - (n_pc-1):  unique values = "
      f"{sorted(np.unique(step_low).tolist())}")
print(f"step_high = (n_pc+1) - (n_pc+0):  unique values = "
      f"{sorted(np.unique(step_high).tolist())}")

n_bad_low  = int(np.sum(step_low  != 1))
n_bad_high = int(np.sum(step_high != 1))
print(f"# cells with step_low  != 1:  {n_bad_low}")
print(f"# cells with step_high != 1:  {n_bad_high}")

if n_bad_low > 0 or n_bad_high > 0:
    print("\n✗ INCONSISTENCY DETECTED — three n_pc copies disagree.")
    bad_mask = (step_low != 1) | (step_high != 1)
    bb, pp = np.where(bad_mask)
    print(f"  Total inconsistent cells: {bad_mask.sum()}.  First 10:")
    # Also show the n_p value at those pixels to confirm half-integer
    n_p_np = np.asarray(_geom_inline(angles_jax[0])[0])
    for b, p in list(zip(bb, pp))[:10]:
        print(f"    batch={b:>2d}  pixel={p:>4d}  "
              f"dn1={dn1_np[b,p]:>4d}  d0={d0_np[b,p]:>4d}  dp1={dp1_np[b,p]:>4d}  "
              f"n_p={n_p_np[p]:.6f}")
else:
    print("\n✓ All three destinations differ by exactly 1 — XLA used a")
    print("  consistent n_pc here.  Two interpretations:")
    print("    (a) returning these values forced CSE that wouldn't otherwise win, or")
    print("    (b) the inconsistency happens elsewhere (e.g., between the")
    print("        scatter-index path and the L-weight path), not between the")
    print("        three offset increments.")

print()
print("=" * 70)
print("Done.")
print("=" * 70)
