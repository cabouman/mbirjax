<!-- Appendix to plans/projector_kernels/gpu_headroom_plan.md.
Produced 2026-07-12 by a parallel research agent during the headroom-investigation kickoff
(five-agent workflow; this file is one agent's report, reproduced verbatim).
Claims marked "verified" were checked against the repo / the pinned jax 0.10.1 env / cited
sources by that agent; quantitative traffic models are PRE-E0 estimates pending the HLO/ncu
verification pass. -->

# XLA:GPU lowering of the mbirjax reduction patterns — research findings

**Bottom line:** `segment_sum(indices_are_sorted=True)` already hits a genuine specialized sorted-scatter emitter in XLA:GPU (register accumulation per warp, atomics only at run flushes) — the scatter fusion itself likely has <2x headroom. The recoverable no-custom-kernel headroom is *around* the scatter: the per-view CUB sort and the gather traffic that build the sorted updates. The single biggest XLA-level lever is that our sort permutation is **geometry-only** (depends on view params + pixel indices, not voxel values) and can be hoisted out of the compiled programs entirely, the same way the Phase D `n_pc` centers were. No relevant XLA/JAX improvements have shipped after the 0.10.1 pin (verified through today); staying pinned is correct.

---

## 1. How XLA:GPU lowers each pattern

All emitter facts below are from current `openxla/xla` source (paths verified 2026-07-11); local HLO evidence is from the pinned jax/jaxlib **0.10.1** in the `mbirjax` env (probe script: `/private/tmp/claude-501/-Users-gbuzzard-Documents-PyCharm-Projects-Research-mbirjax/a7f368bb-e9f2-4bda-9ccf-635f8925c89d/scratchpad/hlo_probe.py`).

### (a) `jnp .at[idx].add` with duplicate indices (our CPU/atomic path)
- Emitted HLO: `stablehlo.scatter` with `indices_are_sorted = false, unique_indices = false` (verified locally for `_channel_reduce_scatter_add`, both unbatched and vmapped).
- GPU lowering: the MLIR scatter emitter `xla/backends/gpu/codegen/emitters/scatter.cc`. With non-unique, non-sorted indices it selects **`ScatterWithDistributedUpdates`**: a grid-stride loop over the updates tensor, one **`AtomicRMWOp` per update element** (f32 → hardware `atomicAdd`; the reducer is inlined into the atomic body). Nondeterministic accumulation order (hence f32 non-reproducibility) is inherent to this path ([openxla determinism doc](https://openxla.org/xla/determinism)).
- With `unique_indices=True` the emitter **bypasses atomics entirely** (plain insert) — but produces silently wrong results if the promise is false.
- Under `xla_gpu_deterministic_ops`/`xla_gpu_exclude_nondeterministic_ops`, scatter falls back to a serialized while-loop expansion — the known catastrophic path ([jax#17844](https://github.com/jax-ml/jax/issues/17844): vmapped scatter_add up to 377x slower).

### (b) `jax.ops.segment_sum(..., indices_are_sorted=True)` (our GPU path)
**Yes — it emits a specialized sorted-segment reduction, not a generic scatter.** Chain of evidence:

1. JAX passes the flag through: local HLO shows `stablehlo.scatter ... indices_are_sorted = true, unique_indices = false`.
2. **The flag survives our vmap over views.** `_scatter_batching_rule` (jax 0.10.1, `jax/_src/lax/slicing.py` ~line 2857) passes `indices_are_sorted=indices_are_sorted` through unchanged and uses the modern `input_batching_dims`/`scatter_indices_batching_dims` mechanism (no iota-concat). Verified locally: the vmapped segsum scatter keeps `indices_are_sorted = true`. (Contrast: **gather**'s batching rule *does* reset `indices_are_sorted=False`/`unique_indices=False` when only indices are batched — jax `slicing.py` line ~2244.)
3. XLA:GPU's `CreateScatterFusion` (in `scatter.cc`) selects **`ScatterWithDistributedIndices`** when `indices_are_sorted() && !unique_indices() && num_slices > max_active_warps`, where `max_active_warps = 4 * core_count` (H100: 528). Our flattened `num_slices` (view-batch × T × P ≈ 3.1M) passes easily.
4. The algorithm (from `scatter.h`/`scatter.cc`): each warp owns `num_indices_per_warp = ceil(num_slices / max(max_active_warps, num_possible_valid_indices))` consecutive sorted updates, accumulates runs of equal indices in a **register vector accumulator**, and flushes to global memory (still via `AtomicRMWOp`, to handle runs straddling warp boundaries) only when the index changes. For our shapes `num_possible_valid_indices = V·C` and `num_indices_per_warp ≈ T·P/C` = the **collision ratio (~24 at 1024³)** — i.e., XLA's heuristic literally rediscovers the quantity your `SORTED_CHANNEL_REDUCE_MIN_COLLISION_RATIO` guard encodes, and each warp amortizes ~24 updates per atomic flush. This matches the measured 2-3x win over the atomic path and the observed cliff when the ratio is small.
- Caveats: index-read vectorization only applies when `index_vector_length == 1`; our vmapped scatter has a 2-vector index (view, channel) after simplification, so index reads are not vectorized (see lever 4 below). Sharp edges exist in adjacent paths: [jax#26227](https://github.com/jax-ml/jax/issues/26227) (sorted segment_ids 100x slower, f16/f8 only — we are f32, unaffected, worth monitoring).

### (c) `lax.sort_key_val` → **CUB DeviceRadixSort**
`xla/backends/gpu/transforms/sort_rewriter.cc` rewrites HLO sort to a CUB radix-sort custom call when: 1-2 operands; simple key comparator (or the canonical NumPy-order pattern); **sort dimension is the minor one** (batch dims allowed — the H100/A100 heuristic tables take `batch_size` explicitly); supported types — for pairs, `S32/U32/U8/U16/U64/F32` keys with 16/32/64-bit values. Our `(views, T·P)` s32-key/s32-iota sort qualifies (int32 key-pair instantiations completed upstream 2026-03-17, well before the 0.10.1 pin), and at batch 128 × 24576 elements the H100 heuristic (`batch_size > 66 && num_elements > 2^11`) selects CUB. Controlled by `xla_gpu_enable_cub_radix_sort` (**default true**). The fallback when ineligible is the generic sort emitter, explicitly described in-source as "slower at runtime". So: the sort is already the best library sort XLA has; the only way to beat it is to not sort at runtime at all (lever 1).

### (d) Large gathers with computed row indices
Gathers are lowered via `gather_simplifier` into canonical form and emitted inside **loop fusions**: each thread computes its output element(s) independently; output-side coalescing is the only structured locality; **no shared-memory tiling exists for gather** — shared memory is used only by the transpose and reduction emitters ([openxla emitters doc](https://openxla.org/xla/emitters)). Consequences:
- Any input-locality benefit from sorted/clustered indices is a **hardware L1/L2 cache effect**, not a compiler path — there is no `indices_are_sorted` fast path for gather codegen (the flag exists on the HLO but the GPU emitter has no equivalent of the scatter strategy split).
- Our row gathers (`values[order % P]`, 64-float = 256 B rows) are internally coalesced per row; the randomness is at row granularity. The (P × cols) source tile (~2 MB at 8192×64) fits comfortably in H100's 50 MB L2, which is the *only* cross-view reuse mechanism available (see §3).

**Is the sorted segment_sum near-optimal?** Within the scatter fusion: yes, roughly — register-accumulated runs with one atomic per run is close to what a hand kernel would do for the *reduction step*. What XLA cannot do, and a custom kernel could: (i) skip the runtime sort, (ii) avoid materializing the sorted 24576×cols updates stream between sort and scatter fusion (the sort is an unfusable kernel boundary; the gathers/multiply after it can fuse into the scatter fusion, but the sort's output round-trips HBM), (iii) keep a C-sized accumulator tile in shared memory and stream pixels. The "10x above compute-bound" gap is dominated by (i)+(ii)+gather traffic, not by scatter-fusion inefficiency.

---

## 2. Flag inventory (jax/jaxlib 0.10.1; defaults from `xla/debug_options_flags.cc`)

| Flag | Default | Relevance | Risk |
|---|---|---|---|
| `xla_gpu_enable_cub_radix_sort` | **true** | High (verify it's not disabled in any env wrapper; losing CUB = "slower at runtime" generic sort) | None to keep on |
| `xla_gpu_enable_scatter_determinism_expander` | **false** | Low for perf — it's a determinism pass (sort + log2(n) segmented prefix-scan + unique-index scatter, `xla/backends/gpu/transforms/scatter_determinism_expander.cc`); our hand-written sorted segsum is the same idea with fewer passes. Only matters if run-to-run determinism is ever required — then enable it to avoid the while-loop fallback | Known 64-bit issue ([jax#27324](https://github.com/jax-ml/jax/issues/27324)) |
| `xla_gpu_deterministic_ops` / `xla_gpu_exclude_nondeterministic_ops` | false | Known; avoid — triggers serialized scatter ([jax#17844](https://github.com/jax-ml/jax/issues/17844)) unless the expander covers the op | High perf risk |
| `xla_gpu_enable_command_buffer` | **FUSION, CUBLAS, CUBLASLT, CUDNN, CONDITIONAL, DYNAMIC_SLICE_FUSION already on** (min graph size 5) | Low for the 1024³ kernels (multi-second kernels, launch overhead negligible). **Medium for the §3 VCD host-dispatch pool** — worth checking whether the VCD loop's small ops are being captured (`xla_gpu_graph_min_graph_size`) | Low; occasional capture bugs |
| `xla_gpu_experimental_enable_fusion_block_level_rewriter` | false | Low: redirects existing fusions to the Triton emitter; Triton emitters do not cover scatter, and our hot fusions are scatter/gather-rooted | High (experimental; can perturb every fusion) |
| `xla_gpu_experimental_enable_triton_heroless_priority_fusion` | false | Low, same reasoning | High |
| `xla_gpu_enable_triton_gemm` / `xla_gpu_triton_gemm_any` | true/true | None (no GEMMs in the projectors) | — |
| `xla_gpu_autotune_level` | 4 | None — autotunes GEMM/conv only, not scatter/gather/sort | — |
| `xla_gpu_enable_latency_hiding_scheduler` | false | None for single-GPU kernels; possibly minor for multi-device band transpose overlap | Low |

There is **no** flag that changes scatter/gather codegen strategy directly; the strategy split in §1b is driven solely by the HLO `indices_are_sorted`/`unique_indices` bits and shape heuristics. One diagnostic worth doing once on H100: `--xla_dump_to` and confirm the fwd scatter fusion's launch dims match the distributed-indices path (blocks ≈ `num_slices·warps_per_slice / (num_indices_per_warp·4)`), so we know the fast path is actually active at production shapes rather than inferring it.

---

## 3. Patterns to coax locality out of XLA — honest assessment

**Confirmed: XLA has no inter-vmap-lane data reuse.** vmap lowers to batch dimensions on single HLO ops; loop-fusion threads compute output elements independently with "no cross-thread data sharing except through hardware caches" ([emitters doc](https://openxla.org/xla/emitters)); per-lane gathers with different index vectors are not CSE-able. The 128-view vmap re-reads the shared voxel tile once per view — the only reuse is that the ~2 MB tile stays L2-resident. A shared-memory-resident accumulator/tile across views is exactly what only Pallas/CUDA can express.

Ranked no-custom-kernel levers:

1. **Hoist the channel sort out of the compiled programs (expected 1.2–2x on GPU fwd kernels; low-medium effort).** The sort permutation `order` and the sorted segment ids depend only on geometry (`view_params`, `pixel_indices`, the already-eager `n_pc` centers) — never on voxel values. Precompute them in a separate eager jit exactly like the Phase D `n_pc` idiom and pass concrete `(order, sorted_n)` arrays; the in-kernel work collapses to gather + multiply + sorted scatter, and the CUB sort cost is paid once per geometry instead of once per projector call (and is shared across all VCD iterations). Bound the win first: profile what fraction of the 8.2 s fwd kernel is the sort kernels (ncu or `--xla_dump_to` + nsys). Two acknowledged hazards, both already familiar in-repo: (i) this reintroduces an argsort-then-regather shape — but computed eagerly outside the projector programs, which is precisely the mitigation `projectors.py` already uses for `n_pc` (the in-jit-round hazard doesn't apply to a separate concrete-output jit); (ii) memory: `(V, T·P)` int32 order arrays want the same `N_PC_SINGLE_CALL_MAX_BYTES`-style chunking.
2. **Flatten the batched scatter to 1-D segment ids (expected 1.05–1.3x; trivial to try).** Instead of vmapping segment_sum (which yields a 2-vector (view, channel) scatter index after simplification), compute `flat_ids = view_id·C + sorted_n` and do one `segment_sum(..., num_segments=V·C)` over the flattened batch. `index_vector_length == 1` is the emitter's precondition for **vectorized index reads** (`scatter.cc`: `if (description.index_vector_length == 1 && ...) indices_vector_size = max_vectorized_indices`), which the current 2-D form forfeits. Same math, same sortedness (view-major lexicographic), cheap A/B.
3. **Bucketed fixed-K gather formulation for the forward fan (expected: converts fwd to ~back-kernel behavior; medium-high effort).** Precompute, per (view, channel), the contributor (pixel, tap) list padded to fixed K (K_mean = collision ratio ≈ 24 at 1024³; padding overhead is the K_max/K_mean ratio — favorable for parallel beam where per-channel counts are smooth, bad for wide-psf translation shapes). The kernel becomes gather + reshape-sum: no sort, no atomics, no collision cliff, tables are geometry-only. Honest ceiling: the back kernel (pure stacked gather) is 10.9 s vs fwd 8.2 s at 1024³ — gather-formulation is not intrinsically faster than the current fwd; its value is robustness (kills the ratio<4 cliff class) plus composing with lever 1's precomputation. Not a path to the 10x.
4. **Nearly-sorted pixel ordering outside the kernel: only helps the gather side, not scatter.** XLA's scatter fast path keys on the *flag* (exact sortedness), not on empirical near-sortedness; for the atomic path, clustering duplicate indices *increases* same-address contention within warps (consistent with your measured collision cliff). For gathers, sorted/clustered indices improve L1/L2 hit rates — a hardware effect XLA neither creates nor blocks. Recoloring the pixel partition (e.g., channel-coherent pixel order per batch) so both fwd gathers and the back stacked-gather read nearly-contiguous rows is a legitimate, XLA-invisible lever — but it interacts with VCD partition semantics, so expected win is uncertain (1.0–1.5x); worth a cheap ncu A/B on L2 hit rate before any real work.
5. **One-hot / block-dense matmul reformulation: reject.** Naive one-hot is a ~C× FLOP blowup (hundreds of PF at 1024³). Tile-wise block-dense (cuSPARSE Blocked-ELL-style) only pays off with tensor cores, i.e., TF32/BF16 — which violates the repo's float32 bitwise-correctness gates; in fp32 CUDA cores there is no FLOP advantage and XLA would materialize the one-hot blocks through HBM anyway. The trick's documented wins are TPU-specific, where scatter is far weaker than XLA:GPU's sorted path.
6. **`unique_indices=True`: no safe application.** It removes atomics (`scatter.cc` line 334) but our reductions are non-unique by definition, and a false promise silently corrupts results. Only usable if a future formulation writes one value per (view, channel) bin — which lever 3 achieves without any scatter at all.

---

## 4. Newer-JAX check (post-0.10.1)

- Releases: 0.10.0 (2026-04-16), 0.10.1 (2026-05-20), **0.10.2 (2026-06-17, latest as of 2026-07-11; no 0.10.3 yet)**. The 0.10.2 changelog contains **nothing** scatter/sort/gather/GPU-kernel related (only scipy.linalg additions and `ShapeDtypeStruct.like`), so the known 4x cone regression is an undocumented XLA-pin side effect and **remains unfixed upstream** — no issue matching it surfaced in searches.
- `openxla/xla` commit history on the scatter emitter since the 0.10.1 pin: only refactors plus "Support Variadic Scatter in GPU MLIR codegen" (2026-06-24) — multi-operand scatters, not ours; no performance work. Sort rewriter: only "deviceless CUB mode" integration (May 2026), irrelevant at runtime. The pieces we rely on (distributed-indices scatter emitter, int32 key-pair CUB sort from 2026-03-17, relaxed scatter simplifier from 2026-03-17) are all **already in the 0.10.1 pin**.
- **Verdict: a toolchain bump buys nothing today and re-imports the 0.10.2 regression. Stay pinned; re-check when 0.10.3/0.11 release notes mention XLA GPU emitters.**

---

## Sources

- [xla/backends/gpu/codegen/emitters/scatter.cc](https://github.com/openxla/xla/blob/main/xla/backends/gpu/codegen/emitters/scatter.cc) and [scatter.h](https://github.com/openxla/xla/blob/main/xla/backends/gpu/codegen/emitters/scatter.h) — strategy selection, atomics, distributed-indices algorithm, heuristic constants
- [xla/backends/gpu/transforms/sort_rewriter.cc](https://github.com/openxla/xla/blob/main/xla/backends/gpu/transforms/sort_rewriter.cc) — CUB eligibility, type tables, H100/A100 batch-size heuristics
- [xla/backends/gpu/transforms/scatter_determinism_expander.cc](https://github.com/openxla/xla/blob/main/xla/backends/gpu/transforms/scatter_determinism_expander.cc) — sort + log-step prefix-scan algorithm
- [xla/debug_options_flags.cc](https://github.com/openxla/xla/blob/main/xla/debug_options_flags.cc) — flag defaults quoted above
- [XLA:GPU Emitters](https://openxla.org/xla/emitters), [Determinism (GPU)](https://openxla.org/xla/determinism), [XLA Flags Guidance](https://openxla.org/xla/flags_guidance), [xla.proto](https://github.com/openxla/xla/blob/main/xla/xla.proto)
- [jax#17844](https://github.com/jax-ml/jax/issues/17844) (deterministic vmapped scatter 377x), [jax#26227](https://github.com/jax-ml/jax/issues/26227) (sorted segment_ids f16 100x), [jax#27324](https://github.com/jax-ml/jax/issues/27324) (expander 64-bit issue)
- [JAX CHANGELOG](https://github.com/jax-ml/jax/blob/main/CHANGELOG.md); jax 0.10.1 installed source `jax/_src/lax/slicing.py` (batching rules); local HLO probe at `/private/tmp/claude-501/-Users-gbuzzard-Documents-PyCharm-Projects-Research-mbirjax/a7f368bb-e9f2-4bda-9ccf-635f8925c89d/scratchpad/hlo_probe.py`
