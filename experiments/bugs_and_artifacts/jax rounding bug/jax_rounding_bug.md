# JAX/XLA rounding bug — overview, resolution, and implementation plan

*Companion to `jax_rounding_bug_v2.py` (this directory) and
`minimal_lax_map_repro.{py,md}` (in `./lax_map_scatter_bug/`).
Investigation 2026-05-25/26; JAX 0.10.1.*

---

## Table of contents

- [1. The problem](#1-the-problem)
- [2. What we tried (summary)](#2-what-we-tried-summary)
- [3. The resolution](#3-the-resolution)
- [4. Implementation plan](#4-implementation-plan)
    - [4.1 Where the precompute lives](#41-where-the-precompute-lives)
    - [4.2 Geometry-specific computation: subclass overrides](#42-geometry-specific-computation-subclass-overrides)
    - [4.3 Compatibility with future sharding](#43-compatibility-with-future-sharding)
    - [4.4 API changes — places to update](#44-api-changes--places-to-update)
    - [4.5 Other call sites with the bug precondition](#45-other-call-sites-with-the-bug-precondition)
    - [4.6 Testing](#46-testing)
    - [4.7 Open questions for the design discussion](#47-open-questions-for-the-design-discussion)
- [5. Diagnostic: half-integer preflight check](#5-diagnostic-half-integer-preflight-check)
- [6. Status and decision](#6-status-and-decision)
- [7. Files](#7-files)

---

## 1. The problem

The mbirjax forward projector (`forward_project_pixel_batch_to_one_view`
in `parallel_beam.py`) computes an integer scatter destination
`n_p_center = jnp.round(n_p).astype(int32)` *inside* the jit boundary,
where `n_p` is the continuous projected channel coordinate derived
from `cos`/`sin` of the view angle.  Under certain combinations of
outer wrappers — specifically `jax.vmap` (over views) wrapping
`jax.lax.map` (over slice batches) wrapping a scatter-add — XLA's
optimized HLO produces incorrect results for specific pixel-cloud /
view-angle combinations.

**Symptom:** an antisymmetric channel error in the sinogram —
excess at `n_pc − 1`, deficit at `n_pc + 1` — of magnitude
≈ 0.5 per affected pixel, totaling ≈ 1.7 intensity units summed
across the affected pixel cluster.  In the production demo this fires
at views 12 and 134 (of 180) when the ROR-masked pixel batch 8
(rows 91..99 of a 256×256 recon) is being projected.

**The algorithm is mathematically continuous** in `n_p`: the PSF's
boundary clipping makes the projection independent of which side
`round()` picks at exact half-integers.  So this is a real
implementation bug, not a numerical-sensitivity artifact.

**The error pattern looks exactly like** the scatter destination
`n = n_pc + offset` and the L-weight's `abs_d = |n_p − n|` disagreeing
on what `n_pc` is — as if `n_pc` were computed twice in XLA's lowering
and the two copies diverged at exact half-integers.  However direct
probes (returning `n_pc + {−1, 0, +1}` from the `lax.map` body)
showed those three values are internally consistent, and an
`optimization_barrier` placed on `n_pc` did **not** fix the bug.  So
the precise XLA failure mechanism is **not** "CSE-failed round" of the
kind we (and presumably the upstream JAX/XLA team) have seen before.
It remains undetermined.

**What we know with certainty** is the precondition for the bug:
`jnp.round` of a continuous, in-jit-derived `n_p` must live inside
the `vmap → lax.map → scatter` chain.  Take any one of those three
nesting levels away — or precompute `n_pc` on the host so the `round`
doesn't appear inside the chain at all — and the bug vanishes.

The full investigation, with all 15 test variants and the HLO
post-mortem, is in `./lax_map_scatter_bug/minimal_lax_map_repro.md`.

---

## 2. What we tried (summary)

Run by `./lax_map_scatter_bug/minimal_lax_map_repro.py`, which sweeps
all 24 ROR-masked pixel batches of a 256×256 recon under `vmap` of
size 1 over view 12, with `lax.map` over `_NB = 8` slice batches.

| Fix attempt | Result                                                                    |
|---|---------------------------------------------------------------------------|
| Vanilla `jnp.round` (baseline)                                          | BAD=8 fires, magnitude 0.4995                                             |
| `safe_round(scale=1e-4)`  (snap-then-perturb past half-integer)         | Moves bug to BAD=18                                                       |
| `safe_round(scale=1e-6)`                                                | Sub-ULP perturbation → no-op → BAD=8 fires                                |
| `jnp.round` + `jax.lax.optimization_barrier(n_pc)`                      | BAD=8 fires (barrier ineffective)                                         |
| `safe_round + optimization_barrier`                                     | BAD=18 fires (same as safe_round alone)                                   |
| `safe_round_f64` (cast to float64 inside, scales 1e-4 / 1e-6 / 1e-8)    | Same pattern as float32 — float64 changes nothing                         |
| **Precompute full geometry tuple, pass in as JAX inputs (T15i)**        | **✓ All 24 batches bit-exact (`max abs(diff) = 0.00e+00)`**               |
| **Precompute `n_pc` only, leave rest of geom in-jit (T15j)**            | **✓ All 24 batches clean (worst `max abs(diff) = 6.3e-05`, float noise)** |

`lax.map → vmap` (the other workaround we considered) also fixes the
bug, but at the cost of restricting the code's expressiveness.

---

## 3. The resolution

**Precompute the integer scatter index (`n_p_center` in parallel beam,
`(n_p_center, m_p_center, k_m_center)` in cone beam) on the host, and pass it
into the jit'd projector as an input.**  Leave the rest of the
geometry derivation (`cos`/`sin`/`n_p`/`W_pc`/`cos_alpha`/`L_max`)
inside the jit unchanged.

T15j confirms this is the smallest change that works.  The bug
requires the `jnp.round` to live inside the `vmap + lax.map + scatter`
chain; precomputing `n_pc` removes that single ingredient.

**Why this is preferable to `lax.map → vmap`:**

- No structural restriction on the code.  Future contributors can use
  `lax.map` (and `lax.scan`, `lax.fori_loop`, etc.) freely.  The rule
  becomes a localized API contract — "the projector takes the integer
  scatter destinations as inputs" — rather than a folk wisdom about
  which JAX primitives to avoid.
- Preserves the cache-locality benefit of the inner `_B` slice
  batching in `lax.map`.
- Generalizes cleanly to cone beam, which has additional integer
  scatter indices and where switching the inner loop to `vmap` would
  be more expensive (more memory pressure from vectorizing across
  slice / channel batches).

**Memory cost:** for parallel beam, per call to the jit'd projector,
one extra `int32[num_views_in_batch, num_pixels_in_batch]`.  At
typical sizes that's tens to hundreds of KB — negligible.  Cone beam
adds a second `int32` array of similar size.

---

## 4. Implementation plan

> **Deferred (2026-05-26).**  This rewire is **not** being implemented
> at this time — see §6 for the decision and rationale.  The plan is
> kept here, ready to execute, in case the bug ever proves to matter
> in practice.

This section is an outline, not a final spec.  It identifies the
moving parts and the questions that need to be answered before writing
code.

### 4.1 Where the precompute lives

`sparse_forward_project` and `sparse_back_project` already iterate
over (pixel-batch × view-batch) tiles with a Python `for` loop on the
host.  That is the natural place to precompute the integer scatter
indices: outside the jit boundary, in plain numpy (or JAX with
`jax.disable_jit()` if we want JAX's `cos`/`sin` for bit-exact match,
but numpy is simpler), once per tile, before invoking the jit'd
projector kernel for that tile.

Same story on the back-projection side — the integer indices used in
the gather/scatter are the same geometric quantity.

### 4.2 Geometry-specific computation: subclass overrides

The integer scatter indices are geometry-specific:

- **Parallel beam:** `n_p_center(pixel_indices, angle, params) → int32`
  — one int per pixel per view.
- **Cone beam:** `(n_p_center, m_p_center, k_m_center)(pixel_indices, angle, params)
  → (int32, int32)` — two ints per pixel per view (channel and row).

Proposed structure: a base-class method on `TomographyModel`:

```python
def precompute_scatter_indices(self, pixel_indices, angles):
    """Compute the integer scatter destinations needed by
    forward/back projection.  Geometry-specific; must be overridden
    by each subclass.

    Returns a geometry-specific object (e.g. a NamedTuple) whose
    fields are passed individually into the jit'd projector
    kernels as additional arguments.

    Implementations run on the host (numpy) so that the round
    operation does not appear inside any jit boundary.
    """
    raise NotImplementedError
```

`ParallelBeamModel.precompute_scatter_indices` returns one int32
array `n_p_center` of shape `(num_views, num_pixels)`.
`ConeBeamModel.precompute_scatter_indices` returns the
`(n_p_center, m_p_center)` pair.

The jit'd projector kernels (`forward_project_pixel_batch_to_one_view`,
its back-projection sibling, and the cone-beam analogs) take these
arrays as new required arguments and stop calling
`compute_proj_data` for the integer part.  `compute_proj_data`
either splits in two (a float-only version stays in-jit, an
integer-only version moves out) or is wrapped at the call sites.

### 4.3 Compatibility with future sharding

In the planned final pipeline (see `.claude/status.md`), the sinogram
is sharded along `det_rows` and the recon along `slices`.  Critically:

- The integer scatter indices depend only on `(pixel_coords, view_angles,
  geometry_params)`.
- They do **not** depend on `det_rows` (parallel beam) or on
  individual slices.

So precomputed indices are **invariant under the sharding axes for
both sinogram and recon**.  Each device that owns a slice of the
recon needs the indices for the pixel batches it is projecting — but
those indices are identical across devices for the same pixel set.

**Storage flexibility.**  The precomputed indices do *not* have to be
numpy arrays.  JAX arrays — including `NamedSharding` arrays
distributed across devices — are fine, as long as the values are
**concrete** (materialized in memory, host or device) by the time they
arrive at the projector's jit boundary.  The bug requires the round
operation to be *traced* inside the projector's jit; if the indices
arrive as a concrete `jax.Array` (already-computed values rather than
tracers), the round chain doesn't exist inside the jit and the
precondition is broken.  This means we can use whatever storage and
sharding strategy fits naturally with how the indices are produced and
consumed.

Two reasonable approaches:

1. **Compute once on the host, replicate to every device.**  Simplest.
   The indices are small (an int32 array of `num_views × num_pixels`)
   so the replication cost is negligible.
2. **Compute on each device for its local pixel set.**  Slightly more
   complex but avoids the host→device transfer.  Each device's
   threaded worker (Path G threading, per the sharding plan) computes
   its own indices and ships them to its device alongside its
   sinogram/recon shard.

The choice can be deferred — the API (a function that returns the
indices) is the same either way.

### 4.4 API changes — places to update

Approximate (need to confirm exact signatures when implementing):

- `tomography_model.py`:
  - Add abstract `precompute_scatter_indices(pixel_indices, angles)`.
  - Modify `sparse_forward_project` / `sparse_back_project` to call
    `precompute_scatter_indices` once per (pixel_batch, view_batch)
    tile before invoking the jit'd projector, and to pass the result
    into the jit'd kernel as an additional argument.
- `parallel_beam.py`:
  - Override `precompute_scatter_indices` to return `n_p_center`.
  - Modify `forward_project_pixel_batch_to_one_view` and its
    back-projection sibling to take `n_p_center` as an argument and
    drop the in-jit `jnp.round`.
  - Audit `compute_proj_data`: the float outputs (`n_p`, `W_p_c`,
    `cos_alpha_p_xy`) stay in-jit; the integer output
    (`n_p_center`) leaves and is computed by the new host-side method.
- `cone_beam.py`:
  - Same pattern, returning both `n_p_center` and `m_p_center`.
  - Three known `jnp.round`-of-continuous-geometry sites at lines
    401, 625, 688 (per earlier discussion).  Each becomes an input
    rather than an in-jit computation.

### 4.5 Other call sites with the bug precondition

The same precondition (`jnp.round` of an in-jit-derived continuous
projection coordinate, used to compute a scatter destination) exists
at multiple sites across the geometry kernels.  We take the operating
position that **any of these could exhibit the bug under the right
combination of input layout, view angles, and outer wrappers** —
some hit cleanly today, others may not, others may start hitting
after a future JAX/XLA update.  Rather than verify each site
individually, the plan is to apply the precompute pattern uniformly
to all of them.

Full inventory:

| File | Line | Function | Notes |
|---|---|---|---|
| `parallel_beam.py` | (the original failure) | `forward_project_pixel_batch_to_one_view` | `n_p_center` closed over outside the `lax.map` body |
| `cone_beam.py` | 401 | (per earlier audit) | |
| `cone_beam.py` | 625 | (per earlier audit) | |
| `cone_beam.py` | 688 | (per earlier audit) | |
| `multiaxis_parallel.py` | 251 | `forward_vertical_fan_one_pixel_to_one_view` | `m_center` inside `lax.map` body, varies per iter |
| `multiaxis_parallel.py` | 303 | `forward_horizontal_fan_pixel_batch_to_one_view` | round at top of function, no inner `lax.map` |
| `multiaxis_parallel.py` | 365 | `back_horizontal_fan_one_view_to_pixel_batch` | round at top of function |
| `multiaxis_parallel.py` | 436 | `back_vertical_fan_one_view_to_one_pixel` | inside `lax.map` body |
| `translation_model.py` | 332 | inside `create_det_column_rows` (a `lax.map` body) | |
| `translation_model.py` | 553 | `compute_vertical_data_single_pixel` | |
| `translation_model.py` | 605 | `compute_horizontal_data` | |

The sites span two structural variants:
- Round of a value **closed over** outside the `lax.map` (parallel_beam
  pattern; confirmed bug-triggering).
- Round of a value **computed inside** the `lax.map` body, varying per
  scan iteration (multiaxis vertical fans; translation_model
  `create_det_column_rows`).  Whether these trigger the same XLA
  mis-optimization is not yet directly tested, but the precondition
  is in the same family.

Also includes sites where the round is at the top of the function with
no inner `lax.map` (multiaxis horizontal fans).  Those are safer as
written, but precomputing the integer there too costs nothing,
removes structural ambiguity, and keeps the API uniform across
geometries — so they get the same treatment in the rewire.

### 4.6 Testing

- The bug demo (`vmap_lax_map_demo.py`) and minimization script
  (`minimal_lax_map_repro.py`) both already reproduce the original
  failure; they should serve as regression tests for the fix.
  Specifically, T15j is the test that exercises the proposed fix
  pattern.
- Re-run the standard parallel-beam test suite against the prerelease
  baseline.  The 3-pixel +x/0/−x error pattern that originally
  motivated this investigation should disappear.
- Equivalent cone-beam regression tests once the cone-beam sites are
  patched.

### 4.7 Open questions for the design discussion

1. **Naming / structure of the precomputed object.**  A `NamedTuple`
   keeps the parallel-vs-cone API uniform (each geometry packages its
   own fields, the caller doesn't care).  Alternatives: a dict, or
   separate positional arguments.  The latter is cleaner for jit
   tracing but uglier when the field count differs across geometries.

2. **Caching across iterations.**  Within a single VCD reconstruction,
   the indices for a given (pixel-batch, view) are constant.
   Recomputing them every iteration is wasteful.  Should they be
   cached on `self`?  If so, invalidation rules — they're invalidated
   by any geometry-parameter change (angles, voxel size, detector
   offsets, …).  The caching itself looks straightforward; the cost
   is the invalidation hooks in `set_params` and the increased state
   on the model object.  Probably worth doing but not blocking.

3. **The "concrete inputs" contract.**  The precomputed indices must
   reach the jit'd projector as concrete `jax.Array` or numpy values
   — not as tracers from an outer jit.  In current production
   `sparse_forward_project` / `sparse_back_project` are plain Python
   loops, so this is satisfied automatically.  If a future change
   wraps either function in `jit`, the precompute logic disappears
   into the outer trace and the bug returns silently.  Worth
   documenting in the projector's docstring and possibly adding a
   runtime assertion that the integer-index argument is not a tracer.

4. **Uniform geometry-specific interface.**  Each geometry has more
   than one integer scatter index it needs to precompute (e.g.
   cone-beam has `n_p_center` and `m_p_center`; translation_model has
   `n_p_center`, `m_p_center`, and `k_m_center` for the various
   inside-the-`lax.map`-body sites).  The `precompute_scatter_indices`
   method needs to return a geometry-specific container holding all
   of them, and each downstream projector kernel needs to know which
   fields to consume.  This is mostly mechanical but does require
   touching each kernel signature.  Worth designing the container
   (NamedTuple per geometry?) so the kernel call sites are clean.

5. **Host-side compute parity with the device-side `jnp.cos` /
   `jnp.sin`.**  When the precompute runs in `numpy`, the float
   arithmetic may differ from `jnp` in the last bit or two.  The
   result is still an integer (via `round`), so this matters only at
   exact half-integer boundaries — exactly the boundaries the
   precompute is meant to handle deterministically.  Worth a quick
   sanity test that the integer output matches between
   `numpy.round(numpy.cos(angle) * xt - ...)` and the equivalent
   `jnp` chain, on a representative set of (angle, pixel) values.
   Likely fine, but cheap to verify.

6. **`fbp_filter` / `qggmrf` / other non-geometry code paths.**  No
   `jnp.round` inside jit was identified outside the geometry
   kernels.  Worth a periodic re-audit when new geometry or kernel
   code is added, since the pattern is easy to introduce
   accidentally.

---

## 5. Diagnostic: half-integer preflight check

An alternative to the §3–§4 precompute rewire was considered: a
preflight pass that computes all `n_p` values (and the cone /
translation analogs), flags any that land *exactly* on a half-integer,
and perturbs the corresponding view angle by a small amount
(e.g. +1e-5 rad) so the existing in-jit code runs on an angle that
doesn't produce the tie.  The perturbed angle is then used unchanged
by the existing projector code — no kernel rewiring.

**As a silent auto-fix we would *not* use it**, for reasons worth
recording so they aren't re-derived later:

1. **It bets correctness on "exact half-integer is the only trigger,"
   and we never established the XLA mechanism.**  CSE-failure was ruled
   out, `optimization_barrier` didn't help, float64 didn't help.  The
   failure mode is a *silent wrong answer*, so relying on a complete
   characterization of a bug we couldn't explain is the riskiest kind
   of bet.  The precompute fix is robust regardless of the true
   trigger set; the preflight is only as good as our model of it.

2. **It must reproduce XLA's in-jit `n_p` bit-for-bit, or it misses
   the trigger.**  The bug fires on whatever `n_p` XLA computes inside
   the jit.  A host-side preflight that computes `n_p` even 1 ULP
   differently (different FMA fusion, op order, or backend) can fail
   to flag a value that XLA rounds to an exact tie internally —
   a silent wrong answer despite the check.  This parity is hard to
   guarantee across backends and JAX versions.

3. **Whack-a-mole across pixels at one angle.**  Perturbing the angle
   moves every pixel's `n_p` by a different amount; clearing the
   flagged pixel can land a *different* pixel on a tie.  Requires an
   iterate-recheck-reperturb loop with a probabilistic bound.

4. **Adjoint consistency.**  Forward and back projectors must stay
   adjoint in the VCD loop.  Perturbed angles would have to be used
   consistently everywhere geometry enters — effectively a global
   mutation of the model's angle set, not a local preflight.

5. **Silent modification of user-specified angles.**  The model would
   no longer use exactly the angles the user passed.  Harmless for
   almost everyone, but spooky for precise geometry work or cross-tool
   comparison.

6. **Multiple rounded quantities.**  Cone beam and translation_model
   round more than one quantity; one angle perturbation would have to
   clear all of them jointly.

**As a non-mutating *warning* it is pure upside**, and that is the
role it would be given if built:

- Scan `n_p` (and the cone / translation analogs) for exact
  half-integer values; emit a warning identifying the
  (geometry, view, pixel) so a developer knows they are in the danger
  zone.
- Do **not** auto-perturb — just report.
- After any future precompute rewire, the same scan becomes a
  regression tripwire.

**Open question if we build even the warning:** how stable is the set
of exact-half-integer `n_p` across backends?  If CPU and GPU disagree
on which pixels land exactly on 0.5 (different cos/sin/FMA), a warning
computed on one backend doesn't predict the bug on another.  Worth a
quick cross-backend test before relying on it.

**Status:** on the back burner; not built.  Recorded here so the
option and its caveats aren't re-discovered from scratch.

---

## 6. Status and decision

**Decision (2026-05-26): do *not* implement the precompute rewire at
this time.**  Rationale:

- Whatever the precise trigger is, it is **rare** — one specific pixel
  cloud at one or two specific view angles, requiring an *exact*
  half-integer `n_p`.  Real reconstructions with measured or otherwise
  non-idealized angles are very unlikely to hit it; and when they do,
  the error is a localized, near-antisymmetric perturbation of ~0.5
  per affected pixel that largely averages out across views.  The
  practical impact on a real reconstruction is expected to be
  negligible.
- The precompute rewire (§4) is a real, multi-file change with an API
  design that needs maintainer coordination.  Given the low practical
  risk, it is not the best use of current resources.

**What we are doing instead:**

- Keep the diagnosis, reproducer, and rewire plan fully documented
  (here and in `./lax_map_scatter_bug/`) so the work isn't lost.
- Treat the precompute approach (§3–§4) as the known, robust fix,
  ready to implement if the bug ever proves to matter in practice.
- Keep the preflight *warning* (§5) as a possible future diagnostic.
- Continue watching for other code structures in which the bug may
  appear, and extend the §4.5 inventory as new sites are found.

**What is known and tested:**

- The bug's precondition and the precompute fix are confirmed in
  `./lax_map_scatter_bug/minimal_lax_map_repro.py` (T15i / T15j).
- The precise XLA mechanism remains **undetermined**.
- Earlier `lax.map → vmap` and `safe_round` workarounds were
  considered and not adopted — the former restricts code
  expressiveness, the latter only shifts the bug to a different
  pixel batch.

---

## 7. Files

- `jax_rounding_bug_v2.py` (this directory) — direct CSE-failure probe.
  Inconclusive but informative.
- `./lax_map_scatter_bug/minimal_lax_map_repro.py` — T1–T15j
  systematic minimization, includes the working fixes.
- `./lax_map_scatter_bug/minimal_lax_map_repro.md` — investigation
  notes.
- `./lax_map_scatter_bug/vmap_lax_map_demo.py` — annotated demo of
  the original bug.
- `mbirjax/parallel_beam.py`, `mbirjax/cone_beam.py`,
  `mbirjax/multiaxis_parallel.py`, `mbirjax/translation_model.py` —
  the geometry kernels with `jnp.round`-inside-jit sites (see §4.5
  inventory); to be patched only if/when the rewire is undertaken.
