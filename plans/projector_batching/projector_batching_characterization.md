# Sparse-projector batching machinery — characterization

**Written 2026-07-03** (branch `greg/performance_improvements`), as the pre-refactor
characterization for current_plans.md (then post_shard_plans) item 3 (simplify the batching machinery) and item 1
(size-adaptive `view_batch_size_for_vmap`).  Everything below was verified against the code on
this date; line numbers will drift.  Companion artifacts in plans/experiments/projector_batching/:
`batching_census.py` (ragged-tail cost census, pure numpy) and `transient_memory_probe.py`
(compiled `memory_analysis()` vs `view_batch_size`, CPU).

## 1. Structure: two tiers of batching

Batching happens at TWO tiers, and only the lower one is the refactor's focus:

**Host tier (`mbirjax/tomography_model.py`)** — Python-level loops that carve work up
*before* calling the jitted drivers (each also sets communication/gather granularity, so they
are not pure batching):

| site | axis / unit | tail handling today |
|---|---|---|
| `_forward_project_to_view_shards` worker (~L1358) | pixel batches of 2048 (cone gather-forward, n_dev>1) | ragged last batch → separate driver executable |
| `_sparse_back_project_single_device` (~L1547) | view + pixel transfer batches (n=1 GPU short-circuit) | `array_split` → balanced (±1) |
| `_forward_project_all_bands` (~L1466) | slice-bands (parallel-beam banded forward) | `_balanced_slice_bounds` → balanced (±1) |
| `_back_project_all_bands` (~L1806) | slice-bands (sharded band back) | `_balanced_slice_bounds` → balanced (±1) |

The band loops are already balanced; the two ad-hoc sites are follow-on candidates (§6), to
be taken up only if the driver-tier work shows a win that transfers.

**Driver tier (`mbirjax/projectors.py`)** — the refactor target.  Two generic helpers,
`concatenate_function_in_batches` and `sum_function_in_batches` (used nowhere else in the
package), batch inside three module-level jitted drivers (shared jit cache across model
instances; batch sizes and kernels are **static** args, view params / data are traced):

| driver | outer axis (`sum_function_in_batches`, `lax.scan`) | inner axis (`concatenate_function_in_batches`, `lax.map` + `vmap`) |
|---|---|---|
| `_sparse_forward_project` (~L332) | **pixels** (summed; carry = owned sinogram) | **views** (concatenated; vmap width = `view_batch_size`) |
| `_sparse_back_project` (~L367) | **views** (summed; carry = `(num_pixels, num_slices)` block) | **pixels** (concatenated; vmap over views inside) |
| `_sparse_back_project_band` (~L407) | same as back | same as back; banded kernel, `g0` traced, `num_band_slices` static |

Axis semantics matter for tail strategies: on a **concatenate** axis (views in forward,
pixels in back) any tail handling is value-exact; on a **sum** axis (pixels in forward, views
in back) overlap/recompute needs a mask or it double-counts, and batch size changes the
reduction order (fp-equivalence gate, never byte-exact — project rule).

**Ragged-tail mechanics today** (both helpers, ~L164–199 / ~L275–298): the remainder `n mod B`
runs FIRST as a separately-inlined odd-shaped batch, then `lax.map`/`lax.scan` covers the full
batches.  Inside one jit this is an extra inlined kernel in the HLO (compile size / fusion
variants), *not* a separate executable.  Distinct **outer** shapes (subset size, view count)
do create separate executables.  The `lax.map` at ~L188 is the safe no-`batch_size` form
(jax#27591 concerns `batch_size=`; still worth collapsing to `scan` for uniformity).

## 2. What shapes the drivers actually see (call sites, verified)

- **Driver view axis, all sharded paths:** `n = views_per_dev = ceil(num_views / n_dev)`
  (view-padded), selected via `owned_view_indices`.
- **Cone sharded gather-forward** (`_forward_project_to_view_shards`, tomography_model
  ~L1312): host Python loop pre-batches pixels at `pixel_batch_size_for_vmap` (2048), one
  driver call per host batch → the driver's internal pixel batching is trivial (`n == B`);
  the host tail is a **separate executable** (full extra trace+compile).  Side effect: driver
  shapes are *normalized* across VCD granularity levels (always 2048 + one tail size per
  level).  At `n_dev == 1` it instead makes ONE monolithic call with all pixels — the host
  loop measurably defeats XLA remat (1024³: ~16 GB one-shot vs ~32 GB looped; comment ~L1347).
- **Parallel-beam banded forward** (worker ~L1492): one driver call per slice-band with the
  FULL pixel set (`n =` subset size) and band-length `L` slices; internal pixel batching real.
- **Sharded band back** (~L1638): host double loop (slice-owner × band) →
  `sparse_back_project_band` per band; `n_pixels =` subset size, `num_band_slices` static.
- **n=1 GPU back short-circuit** (`_sparse_back_project_single_device`, ~L1525): host loops
  over `jnp.array_split` view batches (±1 balanced!) and `transfer_pixel_batch_size` =
  100·2048 pixel batches, with a `block_until_ready` per pixel batch (serializes dispatch).
  This is the **only** remaining consumer of `transfer_pixel_batch_size` — the last vestige
  of the old host-resident-recon "big transfer then batch" design.  `transfer_view_batch_size`
  merely aliases `view_batch_size_for_vmap` (~L1543), so the vmap knob silently controls this
  host loop too.
- **VCD subsets:** partitions are exactly equal-sized (`gen_pixel_partition` pads by repeating
  random indices, vcd_utils ~L194–207), so the pixel axis has ONE shape per granularity level.
  Default sequence `[0,2,4,6,7]` over granularity `[1,2,4,8,...,256]` → subsets {1,4,16,64,128},
  subset size = `ceil(ror_pixels / num_subsets)`, `ror_pixels` = inscribed-ellipse mask count.

**Balanced tiling already exists in-tree** on two axes: `_balanced_slice_bounds` (~L1966;
slice-band axis: fewest bands, lengths ±1, no overlap — at most 2 compiled band lengths) and
the `array_split` above.  The ragged-tail problem is specifically the fixed-`B` batching
*inside* the drivers.

## 3. Census results (batching_census.py, 2026-07-03)

86 realistic axis instances (recon 256–2048, views 256–3142, 1–8 devices, default granularity).
Strategies: `ragged` (status quo), `pad`/`overlap` (fixed B, waste = pad or recompute),
`balanced` (B\* = ceil(n/ceil(n/B)), residual < num_batches handled by tiny overlap).

- **pad/overlap at fixed B is not viable:** waste max **70.7%** (views=1200, n_dev=8 → n=150,
  B=128), p90 **30%**.  The bad regime is exactly the important one: high device counts make
  `views_per_dev` land just above a multiple of B.
- **balanced is essentially free:** waste max **0.76%**, median 0 (residual is exactly 0
  whenever `num_batches | n`).  One kernel shape per axis instead of two.
- **43/86 axes carry an odd tail today** (extra inlined kernel each; no wasted FLOPs).
- **Cost of balanced:** B\* shrinks below B_max by median 3.7%, **max 41%** (n=150 → two
  batches of 75).  Memory-safe (smaller transient); the open risk is only whether per-item
  time is flat down to B\* ≈ B/2 (the vmap-width knee — measure, see §5).

## 4. Transient memory (transient_memory_probe.py CPU + gpu_knee_probe.py H100)

`compiled.memory_analysis()` on the drivers, sweeping `view_batch_size` at `pixel_batch_size`
2048.  CPU: cone 128³/384 views, vb ∈ {16..128}.  GPU (H100 80GB, jax 0.10.1, 2026-07-04,
run by Greg): cone N ∈ {256, 512}, 3840 views, vb ∈ {16..256}.

- **Forward:** CPU temp ≈ **7.2** buffers of `[vb × 2048 × det_rows]` f32 + ~1.2× owned-sino
  constant.  GPU: k ≈ 1.04 (N=256) / 1.27 (N=512); subtracting the view-batch sino stack
  (vb·N²·4B = N/2048 in these units: 0.125 / 0.25) both sizes give ≈ **1.0 coexisting kernel
  buffer** — GPU fusion is far tighter than CPU.  GPU constant = 1.00× sino exactly: the scan
  CARRY (the accumulator), expected and irreducible in the v1 structure.
- **Back:** CPU ≈ **1.5** buffers + small constant; GPU ≈ **1.9 / 1.8** buffers + a constant
  of **1.00× sino IN TEMP** — see below.
- **Hidden copies are backend-dependent.**  CPU: none (back's input reshape and forward's
  output concatenate are aliased away by buffer assignment).  GPU: back's `sum_function_in_
  batches` input reshape **materializes a full view-shard-sized copy** (the only sino-scale
  object in that driver besides the input arg; slope terms accounted for).  v1 back on GPU
  therefore carries ~1 extra owned-sino-shard of compiled temp reservation that the v2 windowed-`dynamic_slice`
  mechanic eliminates — a concrete memory win to verify in the post-refactor re-probe.
- **The isolated-driver k is far below the full-path history — provenance matters here.**
  The only prior GPU datum is the code comment "~18.8 GiB just for that intermediate" at
  1024³/vb=512 (tomography_model ~L131, added in 2b962fe 2026-07-03; no independent record
  of the measurement exists in the repo).  Dividing 18.8 by one [512 × 2048 × 1024] f32
  buffer (4.29 GB) gives a DERIVED k ≈ 4.4 (≈4.7 if the comment's GiB is literal) — an
  inference, not a measurement.  Greg's reconstruction of the original analysis (2026-07-04,
  plausible but not guaranteed): the 18.8 GiB was the forward-projection intermediate of the
  coarse VCD subset at 1024³ — `view_batch(512) × pixel_batch(2048) × slices(1008) ×
  footprint(~5)` floats — i.e. a KERNEL-INTERNAL buffer with a ~5× detector-footprint factor
  on the `[vb × pb × slices]` base.  The probe's k ≈ 1.3 at N ≤ 512 with one pixel batch is
  not directly comparable: the footprint intermediate may be fused away at these sizes/this
  call pattern and materialize only at 1024³ / full-path (XLA fusion is shape-dependent).
  Consequences: the adaptive-knob unit may need the footprint factor at large N, and the
  1024³ REAL-PATH A/B in the refactor rollout is load-bearing for the knob budget, not a
  formality.

The memory model for the adaptive knob is `vb_max ≈ budget / (k · pixel_batch · det_rows ·
4B)` with k ≈ 1.3–1.9 GPU / 7.2 CPU (current lowering; re-derive at 1024³ real path) — and on
GPU the sino-sized constants dominate the vb term, so the knob has more headroom than the
hardwired 128 assumed.

## 5. Prior measured facts that bear on the refactor

- Band-size sweep (H100×4, 512³/1024³): **time flat across band length on GPU** — smaller
  streaming units were a free memory win (`_slice_band_length` docstring, ~L1907).
- **View-vmap width sweep (H100, gpu_knee_probe.py, 2026-07-04): NO knee down to B/2.**
  Warm time vs vb, both drivers, N ∈ {256, 512}, tail-free (3840 views): vb=96..256 flat
  within 1%; vb=80 ≤1.02×; vb=64 ≤1.04×; even vb=16 only 1.10–1.16×.  Balanced B\* is
  confined to (B_max/2, B_max] by construction, so balancing costs ≤~4% worst case and
  ≤~2% typically — **no floor logic needed on GPU**.  (CPU has a real cliff at vb=16 for
  back, ~3×, but B\* ≥ 64 at B_max=128 clears it.)
- `view_batch_size` 512→128: ~4× transient reduction for ~2% time (comment ~L131) —
  consistent with the sweep above.
- Platform-divergent back kernels (band ~8× faster on CPU, ~2.25× slower on GPU) — the n=1
  GPU short-circuit must survive the refactor.

## 6. Refactor implications and open questions

Supported by the above:

1. **Balanced batching per axis** (the `_balanced_slice_bounds` policy moved inside the
   helpers) eliminates both the odd-tail kernel and the pad/overlap waste; the residual
   (< num_batches items) is the only tail left, handled by a tiny overlap on concatenate axes
   and a mask (or the existing odd-batch mechanics, now near-full-size) on sum axes.
2. The adaptive `view_batch_size` policy composes cleanly on top: size-derived `vb_max` from
   §4's formula, then B\* = ceil(n/ceil(n/vb_max)) per driver call.  Device-count dependence
   is acceptable (Greg, 2026-07-03: fp-equivalence is the gate, not byte identity).
3. `transfer_pixel_batch_size` and the n=1 GPU back host loop are refactor candidates: the
   monolithic-call lesson from the cone n_dev=1 forward (~L1347) suggests the same treatment.

Open (measure before/during the refactor):

- ~~GPU vmap-width knee~~ / ~~GPU k coefficient~~ / ~~hidden-copy check~~ — **ANSWERED
  2026-07-04** (H100 run, results in §4/§5): no knee, k ≈ 1.3/1.9, and one real hidden copy
  found (v1 back's input reshape on GPU) that v2 removes.
- ~~Whether collapsing scan/map/vmap changes XLA's buffer reuse~~ — **MEASURED 2026-07-04**,
  and it cut both ways: the windowed scan removed the back driver's sino-sized GPU temp
  (H100 v1-vs-v2: const 1.00× → 0.00× sino, ~3% faster) but was 1.5–1.6× SLOWER on CPU for
  the concatenate axes, where v1's map+inline-peel mechanics were kept (the v2 hybrid; full
  numbers in `batching_refactor_design.md` §1 and §4-gates).
- **1024³ real-path memory A/B** (part of the rollout gates; `full_path_ab.py` in this
  directory, cluster): reconciles the derived full-path k ≈ 4.4 (see §4 — an inference from
  the L131 comment, not a measurement) vs the isolated-driver ≈1.3, anchors the
  adaptive-knob budget, and is the fp-equivalence + memory gate for flipping the default to
  v2.
