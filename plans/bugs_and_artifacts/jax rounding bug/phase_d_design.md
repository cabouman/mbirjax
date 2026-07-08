# Phase D design: the `n_p_center` concrete-input precompute

*Drafted 2026-07-08 (branch `greg/kernel_investigation`); IMPLEMENTED as designed the same
day — see §10 (as built) at the end.  Supersedes §4 of `jax_rounding_bug.md` as the
implementation plan (that document remains the authoritative record of the bug itself, which
is still present in JAX); this one reflects the June module-jit refactor and the July
TilePolicy/kernel work.*

## 1. Goal and the fix principle

Remove the confirmed rounding-bug precondition — `jnp.round` of an in-jit-derived continuous
projection coordinate feeding a scatter/gather destination inside a `vmap → (lax.map) →
scatter` chain — by making the integer center indices **concrete inputs** to the projector
programs (T15j: the verified minimal fix).  Concreteness is what breaks the precondition,
not host-origin: values computed in a *separate small jit* and passed in are just data in
the projector's compiled program — the round never appears inside the chain.

## 2. What changed since the 2026-05 plan (§4 of the bug doc)

- **The drivers are module-level jits with the batching INSIDE** (June refactor):
  `projectors._jit_sparse_forward_project` / `_jit_sparse_back_project` batch pixels and
  views inside one compiled program.  The old plan's "precompute per tile in the host loop"
  (§4.1) has no host loop to live in — the precompute must either enter as a driver
  argument or the view loop must move back out to the public wrappers.
- **TilePolicy + kernel-algorithm flags** (July): the channel reduction is
  provenance-agnostic (`channel_scatter_reduce` takes `n` however produced), and the sorted
  GPU path already uses `lax.sort_key_val` keys-AS-ids, immune to this bug class by
  construction.  Phase D does not touch the reduction; once `n_pc` is concrete, cached
  per-(view, pixel-batch) sort permutations become an optional later GPU refinement.
- **The multiaxis/translation forward fans now share the parallel/cone kernel structure**
  (this session), so a uniform kernel-signature change covers all four geometries the same
  way.

## 3. Site inventory, classified by SHAPE (the load-bearing new fact)

The 2026-05 plan (§4.5) inventoried the sites; what it did not make explicit is that they
split into two classes with very different feasibility:

**Class H — horizontal fans, round a (views × pixels) quantity.  Materializable.**

| site | function | consumers |
|---|---|---|
| `parallel_beam.py` `compute_proj_data` | `n_p_center` | fwd + back kernels |
| `cone_beam.py` `compute_horizontal_data` | `n_p_center` | fwd hfan + back hfan |
| `multiaxis_parallel.py` fwd hfan / back hfan | `n_p_center` (×2 sites) | fwd + back |
| `translation_model.py` `compute_horizontal_data` | `n_p_center` | fwd hfan + back hfan |

**Class V — vertical fans, round a (views × pixels × slices) or (views × pixels ×
det_rows) quantity.  NOT materializable at scale** (~10¹² elements at 1024³):

| site | function | round input shape |
|---|---|---|
| `cone_beam.py` fwd vertical (`create_det_column_rows`) | `k_m_center` | (V, P, det_rows) |
| `cone_beam.py` `compute_vertical_data_single_pixel` | `m_p_center` | (V, P, slices) |
| `multiaxis_parallel.py` fwd vertical | `m_center` | (V, P, slices) |
| `multiaxis_parallel.py` back vertical band | `m_center` | (V, P, slices) |
| `translation_model.py` fwd vertical | `k_m_center` | (V, P, det_rows) |
| `translation_model.py` `compute_vertical_data_single_pixel` | `m_p_center` | (V, P, slices) |

The ORIGINAL confirmed production failure is Class H (parallel forward).  The 2026-06-03
observation in cone `forward_vertical_fan_one_pixel_to_one_view` is Class V — and it
persisted with `vmap` replacing `lax.map`, so the "remove one nesting level" escape is not
available there either.  **Phase D as proposed here fixes Class H completely and leaves
Class V as a documented, monitored risk** (options in §7).  This is a scope reduction
relative to the old plan's "uniform treatment", forced by the shapes.

## 4. Memory budget for Class H

`n_pc` is one int per (view, pixel).  P = full-grid pixels (rows × cols):

| problem | V × P | int32 | int16 | per-128-view chunk (int32) |
|---|---|---|---|---|
| 1024³-class full grid | 1024 × 1.03e6 | 4.2 GB | 2.1 GB | 0.53 GB |
| 513³-class full grid | 513 × 2.3e5 | 0.47 GB | 0.24 GB | 0.12 GB |
| VCD subset at 1024³ (1/64 of pixels) | 1024 × 1.6e4 | 66 MB | 33 MB | 8 MB |
| translation cells (15 views, 40×256 recon) | 15 × 1.0e4 | 0.6 MB | — | — |

The only problematic case is the full-grid call at capacity sizes — the same cells that
already run within a few GB of the memory gates.  Everything VCD-sized is trivial.

## 5. Proposed design (Class H)

**5.1 The per-geometry center computation.**  One module-level jitted function per geometry
(shared jit cache, same pattern as the kernels):

```python
compute_scatter_centers(view_params_chunk, pixel_indices, projector_params) -> int32 (V_chunk, P)
```

executed EAGERLY in the public wrappers (its output is a concrete device array), reusing the
existing float geometry helpers (`compute_proj_data` / `compute_horizontal_data`) so fwd and
back consume the SAME centers (adjointness preserved, and ties become deterministic: one
authoritative computation instead of two optimizer-divergent copies).  Device jnp, not
numpy: one code path for CPU/GPU, negligible cost (a few flops per element in a tiny
program), and it sidesteps the numpy-vs-jnp last-bit parity question (§4.7-5 of the old
plan) entirely.

**5.2 Kernel signature.**  The per-view kernels gain one traced argument:

```python
forward_project_pixel_batch_to_one_view(voxel_values, pixel_indices, view_params,
                                        n_pc_row,           # (num_pixels,) int, THIS view
                                        projector_params)
```

(back kernels likewise).  Inside, the horizontal fans use `n_pc_row` instead of rounding;
`compute_proj_data`/`compute_horizontal_data` keep producing the float `n_p` (the weights
still need the fractional coordinate) and simply stop producing the integer.  Uniform
across all four geometries.  No geometry-specific container is needed (§4.7-1/4 of the old
plan asked): every Class-H site is exactly one `(V, P)` int array, so a plain array argument
is cleaner than a NamedTuple.

**5.3 Driver plumbing.**  `_jit_sparse_forward_project` / `_jit_sparse_back_project` /
`_jit_sparse_back_project_band` gain the `n_pc` traced argument.  Batching detail: the
drivers batch pixels and views on LEADING axes via the shared helpers
(`sum_function_in_batches` / `concatenate_function_in_batches`), so the two ops want
opposite layouts — forward batches pixels outer / views inner (wants `(P, V)` joining the
pixel-batched operands, view axis mapped by `vmap in_axes=1`), back batches views outer /
pixels inner (wants `(V, P)`).  Passing each op its natural layout from the wrapper (one
transpose at most, on an int array) avoids touching the batching helpers; the exact choice
is an implementation detail to settle at review.

**5.4 Chunking policy (the hybrid).**  Materializing the full `(V, P)` array is fine except
at full-grid capacity sizes (§4).  Proposal:

- **Small/medium (V × P ≤ threshold):** one driver call, full `n_pc` input — today's call
  structure, zero restructuring.  Covers ALL VCD work and every CPU cell.
- **Large:** the public wrapper loops over VIEW CHUNKS (size = the op's view batch, e.g.
  128), computing that chunk's `n_pc` eagerly and calling the SAME jitted driver on the view
  subset; forward concatenates the per-chunk view outputs, back accumulates the sum.  Same
  compiled program for every full chunk (one extra trace for a ragged tail).  This bounds
  the resident `n_pc` at ~0.5 GB (1024³) while splitting one executable into ~8 dispatches —
  overhead expected negligible at these sizes, to be MEASURED (the batching episode's lesson:
  driver-level restructurings must be validated end-to-end).
- Threshold: start at 256 MB int32 (V × P = 6.7e7); sweep, don't guess, before landing.

**5.5 The concreteness contract.**  The wrappers assert the computed centers are not
tracers (guards the "future outer jit swallows the wrapper" failure mode, §4.7-3), and the
kernel docstrings state the contract: *the integer centers are inputs; do not recompute them
in-jit*.

**5.6 Caching.**  Recomputation is cheap (one tiny jit call per driver call); per-(angles,
pixel-set) caching is NOT proposed for Phase D (invalidation hooks cost more than the
recompute).  The later sort-permutation cache (GPU refinement) can revisit this.

## 6. Verification plan

- The original bug demos (`vmap_lax_map_demo.py`, `minimal_lax_map_repro.py` T15j, in
  `plans/experiments/bugs_and_artifacts/jax rounding bug/lax_map_scatter_bug/`) as
  regression anchors — the production-shaped demo must show the antisymmetric ±1-channel
  signature GONE.
- Kernel-equality tests old-vs-new per geometry: values equal EXCEPT at exact .5 ties
  (fraction-deviating gate, never max-error — tie flips are the fix working, not a
  regression).  Both coeff_powers on the back kernels.
- NOTE: unlike the fan rollouts, Phase D CHANGES the CPU compiled program (the round leaves
  it) — there is no HLO-parity shortcut.  The gate is the full CPU + sharding suites, the
  fingerprint watch on the nightly, and end-to-end A/B timing (expect ~neutral; the round is
  a negligible share of kernel time, and the added input is small except where chunked).
- Memory A/B at the capacity cells (the chunked path must not move the 1024³ peaks).

## 7. Class V options (not in Phase D scope — for discussion)

1. **Accept + monitor** (proposed): the one observed Class-V instance was input-sensitive
   and vanished at production shapes; the nightly fingerprint gates + the known
   antisymmetric signature are the tripwire.  Cost: a latent, JAX-version-dependent risk.
2. Consistency-by-construction rewrites (make the scatter index and the weight distance
   provably one tensor): no known formulation — `optimization_barrier` is already proven
   ineffective, and removing `lax.map` did not help the cone instance.
3. Algorithmic restructuring of the vertical fans (e.g. band-anchored integer walks):
   changes values and kernels substantially; only worth designing if a Class-V production
   hit is ever confirmed.

## 8. Rollout order and interaction with the current branch

1. Land the multiaxis/translation fan rollout first (this session — kernel signatures then
   stable across all four geometries).
2. Phase D on top, geometry by geometry: parallel (the confirmed site, and the demo to
   verify against) → cone → multiaxis → translation.  Each step: kernel-equality tests →
   suites → nightly watch.

## 9. Open questions for review

1. **int16 vs int32 + chunking:** chunking alone (int32) is proposed; int16 would let the
   full 1024³ grid materialize in 2.1 GB with NO wrapper loop — simpler control flow, but a
   permanent 2 GB tax at capacity vs a 0.5 GB chunked transient.  Preference?
2. **Threshold value** for the hybrid (start 256 MB, sweep) — acceptable?
3. **Class V acceptance** (§7, option 1) — agreed, or should option 3 get design time now?
4. **Layout plumbing** (§5.3): pass per-op natural layouts vs extend the batching helpers
   to batch a chosen axis — either is contained; pick at implementation review.

## 10. As built (2026-07-08)

Implemented per §5 with these resolutions: int32 + hybrid chunking (§9-1); threshold
`projectors.N_PC_SINGLE_CALL_MAX_BYTES = 256 MB` initial, chunk size = the op's view batch
(§9-2); Class V accepted + monitored (§9-3); per-op natural layouts, batching helpers
untouched (§9-4 — they already batch tuples, so forward passes pixels-major (P, V) and
back/band views-major (V, P), one batch-sized transpose each inside the drivers).

Key pieces: per-geometry `compute_channel_coordinate` (the [0] of the existing float chain;
multiaxis extracted its inline chain into one shared method used by BOTH hfans and the
precompute); `projectors._jit_compute_scatter_centers` (vmapped round, `out_axes` picks the
layout); kernels take `n_p_centers` per view; wrappers compute centers eagerly with a
tracer-guard assert and never chunk a multi-device view axis (back chunking slices the
sinogram eagerly, single-device only).

Verification highlights (full record in `plans/projector_kernels/fwd_back_findings.md`):
the minimized repro on jax 0.10.1 still fires and T15j (this pattern) is clean on all 24
batches; the compiled parallel fwd AND back programs contain ZERO round ops; value gates
bitwise except a benign multiaxis-back fusion-context ULP (hfan-only probe bitwise);
`tests/test_scatter_centers.py` pins the centers, the chunked path, and the outer-jit
refusal.

**Post-implementation episode (same day): the eager-gather VCD regression.**  The first GPU
round showed VCD +35% at 200³.  Attribution chain: repeat-trials (real, tight) → wrapper
micro-bench (flat!) → dispatch-count probe (exonerated) → exact-shape sweep (flat) → device
trace (device time DOWN — the cell is ~95% HOST-bound) → cProfile (the answer: ONE eager
`view_params[asarray(owned)]` gather per projector call, ~1 ms host each, 547×/recon; the
micro-bench had used the empty-default `owned_view_indices` path).  Fix: `owned_view_indices`
passes THROUGH to the jitted programs (the drivers' historical in-jit gather restored; the
centers jit gained the same in-jit gather), leaving the wrappers with ZERO eager array ops on
the single-call path — now an explicit performance contract in create_projectors.  After the
fix: VCD 200³ neutral-to-better (2092 → 1950 ms), 1024³ fwd/back −3%, memory unchanged.
Remaining accepted costs: mid-size fwd cells +~0.5 GB (materialized centers + chunk concat;
memory-gate ack) and 513-class back +8% (the chunked accumulation).
