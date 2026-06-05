# `lax.map` + scatter-add bug — investigation notes

*Companion to `minimal_lax_map_repro.py` and `vmap_lax_map_demo.py`.*
*Investigated 2026-05-25/26.  JAX 0.10.1, CPU backend, Apple Silicon (arm64).*

---

## 1. Symptom (production)

In mbirjax's parallel-beam forward projector
(`forward_project_pixel_batch_to_one_view`), an inner `jax.lax.map` over
`_NB` slice batches produces incorrect sinogram values for specific view
angles when called via the outer `vmap`-over-views machinery in
`sparse_forward_project`.

Concretely, with the production geometry
`ParallelBeamModel((180, 128, 256))` and pixel batch 8 of
`gen_full_indices(recon_shape)`:

- max|diff| vs the prerelease scatter ≈ **1.7** (visible reconstruction
  artifact, not float-noise).
- The error appears only at angles **12** and **134** of 180.
- The error is antisymmetric: excess at channel `n_pc − 1`, zero at
  `n_pc`, deficit at `n_pc + 1`.
- Visible only for PSF offsets ±1; offset 0 is always clean.

---

## 2. Root cause (as far as we've reduced it)

The bug lives **strictly below JAX's JAxPR level**.

- The scan-body JAxPRs for "geometry derived from `angle`" and "geometry
  passed as explicit `vmap` args" are *bitwise identical* (115 lines).
- Replacing `jax.lax.map(f, xs)` with `jax.vmap(f)(xs)` (same semantics,
  different XLA lowering) eliminates the bug.
- Replacing `lax.map` with a Python for-loop also eliminates the bug.

The HLO dive (`/tmp/hlo_demo/`, see "HLO findings" below) shows the
scatter dim-numbers, bounds checks, and arithmetic are identical between
the reference and buggy paths.  The only structural difference is that
the buggy path's scatter is wrapped inside a `while` loop (`lax.map`'s
lowering) and operates on a `[2, 16, 256]` per-batch slab instead of the
reference's `[2, 128, 256]` whole-view buffer.  We did not find a smoking
gun in the HLO or LLVM IR.

**The algorithm is continuous in `n_p`.** For an `n_p` exactly at a
half-integer, the PSF clipping makes the result independent of which
side `round()` picks — both `n_pc = round_up` and `n_pc = round_down`
deposit the same total energy into the same two adjacent channels.
So the observed ~1.7-magnitude output difference is *not* a sensitivity
artifact at a discontinuous boundary; it is a real bug.

**The observed error signature** — excess at `n_pc − 1`, zero at
`n_pc`, deficit at `n_pc + 1`, with magnitude consistent with one
pixel's L-weight (~0.5) per affected pixel — is what you would see if
the integer scatter destination `n = n_pc + offset` and the float
`abs_d = |n_p − n|` (which feeds the L-weight) were computed against
**different copies of `n_pc`** that disagree by 1.  That can happen
when XLA fails to deduplicate `jnp.round(n_p)` and the duplicate
chains' upstream float arithmetic produces last-bit-different results
that flip the rounding at exact half-integers.  A similar JAX bug was
observed in this codebase in the past (now resolved by upstream
changes); the symptom matches.

**What we directly verified vs. what we did not.**  `jax_rounding_bug_v2.py`
(in the `jax rounding bug/` subdir) reproduces the bug in this setup
(`max|diff| ≈ 0.5`) and additionally returns the three offset destinations
`n_pc − 1`, `n_pc`, `n_pc + 1` from inside the `lax.map` body.  Those three
arrays differ from each other by exactly 1 everywhere — so within a single
offset block, the three offset-additions of `n_pc` are mutually consistent.
That rules out the most aggressive form of the CSE-failure story
(separately-rounded `n_pc` per offset block) but is compatible with a
narrower form: the integer path that produces the scatter index and the
float path that produces the L-weight could still draw from different
`n_pc` copies inside a single offset block.  We did not find a way to
directly probe that without modifying the buggy chain itself, and we
stopped digging at that point.

**Status of the root-cause claim:** the empirical signature and the
mechanism are consistent, but our probe is suggestive rather than
conclusive.  The fix below works regardless.

---

## 3. The minimal-failing configuration

The full reduction is in `minimal_lax_map_repro.py`.  Required
ingredients (anything else removed):

| Ingredient | Required? | Evidence |
|---|---|---|
| `use_ror_mask=True` (circular pixel mask) | **yes** | T10 (False) → clean |
| Pixel **batch 8** of the 256×256 ROR-masked recon | **yes** | T14: 23 of 24 batches are clean |
| `vmap` outer wrapper (size ≥ 1) | **yes** | T9b (jit only, no vmap) → clean |
| `lax.map` inner slice batching | **yes** | replacing with `vmap` → clean (`forward_fixed` in demo) |
| PSF kernel with offsets ±1 around `n_pc` | **yes** | offset 0 alone → clean |
| `n_p` derived in-jit from `angle` via `cos/sin` (vs precomputed numpy) | **yes in my setup** | T8 (precomputed) → clean; T9b (in-jit) → clean *without* vmap; T9a (in-jit) **bugs** with vmap |

**Smallest reproducer:** `vmap` of size 1 over view 12 with the
256×256/ROR/batch-8 pixel cloud reproduces the bug at full magnitude
(`max|diff| ≈ 0.5`).  No `_geom`-related randomness, no second view,
no production model — just the local `gen_pixel_partition` (≈30 lines,
mbirjax-free) and the PSF scatter chain.

---

## 4. Workaround

**Replace `jax.lax.map(f, xs)` with `jax.vmap(f)(xs)`.**  Both iterate
`f` over the leading axis of `xs`; `vmap` lowers via a different XLA
path that does not exhibit the bug.

In `parallel_beam.py`, line 541
(`forward_project_pixel_batch_to_one_view`, the `else` branch of the
`num_batches == 1` guard):

```python
# BEFORE
sino_batched = jax.lax.map(project_slice_batch, voxel_batched)

# AFTER
sino_batched = jax.vmap(project_slice_batch)(voxel_batched)
```

**Tradeoffs:**
- `lax.map` was chosen for cache locality (sequential execution, one
  `(_B, ND)` scatter target in L1 at a time).  `vmap` may materialize
  all `num_batches × _B × ND` outputs simultaneously and lose that
  benefit — but in our case `num_batches × _B × ND = 8 × 16 × 256` ≈
  512 KB per view, negligible.
- Memory:  fine on all expected backends.
- Correctness: confirmed in `vmap_lax_map_demo.py` (`forward_fixed` →
  `max|diff| = 0`).

**Alternative:** remove inner slice batching entirely (the prerelease
behaviour — one big scatter per view).  This is what the
`num_batches == 1` guard already does for the small-recon case.

---

## 5. Test results from `minimal_lax_map_repro.py`

```
T1   single half-integer n_p, no vmap                                    ok
T2   all n_p exactly at half-integers, no vmap                           ok
T3   T2 + vmap over 2 synthetic 'views'                                  ok
T4   all 2048 pixels collide on channel 172 (n_p = 171.5)                ok
T4b  T4 + 1e-3 perturbation                                              ok
T5   cos/sin chain on rectangular integer grid, four angles              ok
T6   T5 + vmap over [12, 134] with rectangular grid                      ok
T7   circular ROR mask (radius 128, slightly larger than mbirjax's)      ok
T8   production (xt,yt) from gen_pixel_partition, no vmap                ok
T9a  production (xt,yt) + in-jit n_p + vmap over [12, 134]               BUG  max|diff|=0.4995
T9b  same as T9a but no vmap                                             ok
T10  use_ror_mask=False with vmap[12,134]                                ok
T11a vmap[12]   (single view in vmap)                                    BUG  max|diff|=0.4995
T11b vmap[134]  (single view in vmap)                                    BUG  max|diff|=0.4995
T12a vmap[12, 0]   (paired with a clean view)                            BUG
T12b vmap[12, 12]  (paired with itself)                                  BUG
T13  smaller ROR-masked recons (128² and 64²) with batch 0               ok
T14  BAD scan on 256×256 ROR vmap[12]:  only BAD=8 (rows 91..99) bugs    23/24 clean, 1 BUG
```

`run_vmap` reports the maximum absolute difference between the
reference scatter and the `lax.map` scatter on the same inputs;
`max|diff| < 1e-3` is treated as "clean" (float-summation noise).

---

## 6. Things ruled out

- **Half-integer `n_p` values alone.**  T2 (all 2048 pixels at exact
  half-integers) and T4 (all colliding on channel 172) are both clean.
- **Pure duplicate-index pressure.**  T4 puts maximum scatter-add
  pressure on one channel: clean.
- **The `cos/sin` derivation chain alone.**  T5 with a rectangular
  integer grid, including view 90 where all 2048 `n_p` land on exact
  half-integers, is clean.
- **The combination "ROR + cos/sin + production geometry params"**, as
  long as `vmap` is absent.  T8 / T9b clean.
- **Recon size (independent of pixel layout).**  T13 with 128×128 and
  64×64 ROR-masked recons (batch 0) is clean.  But this is confounded
  with pixel set — smaller discs have entirely different pixel batches.

---

## 7. Things confirmed required

- **`use_ror_mask=True`.**  This restricts `gen_pixel_partition` to a
  circular subset of pixels, producing a specific `(xt, yt)` cloud per
  batch.  Toggling it to False in the demo (and in T10) eliminates the
  bug.
- **The specific pixel batch.**  Of 24 batches in the 256×256 ROR,
  *only* batch 8 (rows 91..99) triggers the bug for view 12.  Other
  batches are clean to float-noise level.
- **An outer `vmap` wrapper.**  `vmap` of size 1 is sufficient.  Plain
  `jit` without `vmap` does not trigger the bug.
- **The PSF kernel with offsets ±1.**  Offset 0 alone is clean.

---

## 8. HLO findings (summary)

Dump command:
```bash
XLA_FLAGS="--xla_dump_to=/tmp/hlo --xla_dump_hlo_as_text" \
  python experiments/sharding/vmap_lax_map_demo.py
```

Three relevant modules:
- `module_0038.jit_forward_ref` — reference path
- `module_0040.jit_forward_buggy` — `lax.map` path
- `module_0042.jit_forward_fixed` — `vmap` path (fix)

**HLO-level observations:**
- Scatter dim-numbers are identical between ref and buggy
  (`update_window_dims={1,2,3}`, `inserted_window_dims={}`,
  `scatter_dims_to_operand_dims={0,2}`, `index_vector_dim=1`).
- Both paths recompute `round-nearest-even(n_p) → s32` for each PSF
  offset (XLA does not hoist or share these).
- Bounds checks (`n >= 0`, `n < ND`) are structurally identical.
- The fractional `(n_p − n_pc)` is computed identically in both paths
  by round-tripping `n_pc` through `s32 → f32`.

**Structural differences:**
- Reference scatter operates on `f32[2, 128, 256]` per view.
- Buggy scatter operates on `f32[2, 16, 256]` per batch, inside a
  `while` loop of 8 iterations.  Indices are constant across iterations
  (the same `s32[4096, 2]` is reused); only the per-batch update slab
  changes.

**LLVM IR observations:**
- The buggy scatter kernel processes 16-wide SIMD-vectorized fadds per
  scatter index.
- The reference scatter kernel has the same structure but with a
  16-wide inner × 8-iteration outer loop to cover 128 slices.
- Both look correct in isolation.  No obviously wrong arithmetic.

We did **not** find a smoking gun in the HLO or LLVM dumps.

---

## 9. Open questions

1. **What is numerically special about batch 8's `(xt, yt)` cloud** that
   makes the XLA scatter codegen go wrong for it (and not for the
   other 23 batches at the same recon size)?  Some hypotheses worth
   testing:
   - A specific `n_pc` value or duplicate-collision pattern that only
     occurs at this row range.
   - A bit-pattern coincidence in the resulting `n_p` array (the float
     representation, not the value).
   - An interaction between the `_NB = 8` scan length and the row
     layout (the batch row count matches the scan length).
2. **Does any other scan-shaped outer wrapper trigger it?**  E.g.
   `lax.map(f, angles)` of length 1 instead of `vmap`.  This would
   widen or narrow the upstream bug story.
3. **Is this CPU-specific?**  All testing so far is CPU only
   (Apple Silicon arm64).  Whether GPU or TPU backends are affected
   has not been checked.

These are not blocking the production fix.  They are useful for an
upstream bug report.

---

## 10. Files

- `experiments/sharding/vmap_lax_map_demo.py` — annotated demo,
  reproduces the bug with `max|diff| ≈ 1.68` and shows the fix.
- `experiments/sharding/minimal_lax_map_repro.py` — minimization
  script (this document's companion).  Contains a self-contained
  `gen_pixel_partition` so it has no mbirjax dependency.
- `experiments/sharding/laxmap_isolation_test.py` — earlier in-tree
  isolation test that first established the bug exists.
- `mbirjax/parallel_beam.py` line 541 — the production `lax.map` call
  that needs the workaround applied.
