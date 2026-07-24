<!-- Appendix to plans/projector_kernels/gpu_headroom_plan.md.
Produced 2026-07-12 by a parallel research agent during the headroom-investigation kickoff
(five-agent workflow; this file is one agent's report, reproduced verbatim).
Claims marked "verified" were checked against the repo / the pinned jax 0.10.1 env / cited
sources by that agent; quantitative traffic models are PRE-E0 estimates pending the HLO/ncu
verification pass. -->

# Pallas Assessment for mbirjax Custom GPU Projector Kernels (jax 0.10.1 pin)

## Version ground truth (verified locally)

- Pin: `jax>=0.10,!=0.10.2` + matching jaxlib (`pyproject.toml` lines 33–34; 0.10.2 excluded for an XLA GPU-forward-projection regression). Installed in the `mbirjax` env: **jax 0.10.1 / jaxlib 0.10.1**.
- Installed Pallas submodules: `fuser, mosaic_gpu, ops, tpu, tpu_sc, triton`. `jax.experimental.pallas.triton` exports `atomic_add/max/min/cas/xchg/and/or/xor, load, store`; `mosaic_gpu` exports `SMEM, GMEM, TMEM, ACC, Barrier, WGMMAAccumulatorRef`, TMA/async-copy ops, warp-specialization machinery. `jax.ffi` is present.
- In-tree GPU example kernels at this version (`pallas/ops/gpu`): `paged_attention`, `decode_attention`, `ragged_dot_mgpu`, `hopper_matmul_mgpu`, `blackwell_matmul_mgpu`, attention, layer/rms norm.

## Q1 — Maturity, backend, wheels, churn

- **Two GPU backends at 0.10.1.** Mosaic GPU is the **default GPU lowering path since jax 0.9.0**; the Triton backend remains available via `compiler_params=pltriton.CompilerParams()` but is "maintained on a best-effort basis and not recommended for new use" ([Pallas changelog](https://docs.jax.dev/en/latest/pallas/CHANGELOG.html), [Pallas-for-beginners overview](https://huggingface.co/blog/ariG23498/pallas-for-beginners)).
- **No extra install.** Pallas compiles "exclusively via XLA" on GPU — the old lowering through Triton Python APIs was removed, so the standard `jax[cuda12]` wheels suffice; no separate `triton` pip package ([JAX changelog](https://docs.jax.dev/en/latest/changelog.html); historical install pain in [issue #18603](https://github.com/jax-ml/jax/issues/18603) is resolved by this).
- **Churn is real and documented.** Official status: "Pallas is experimental and is changing frequently… you may be broken by changes" ([docs index](https://docs.jax.dev/en/latest/pallas/index.html), [quickstart](https://docs.jax.dev/en/latest/pallas/quickstart.html)). Concrete breaking changes bracketing the pin: 0.9.1 removed `backend=` from `pallas_call`; 0.10.0 refactored `kernel()` (`out_shape`→`out_type`); 0.10.2 breaks `pl.program_id`/`num_programs` inside Mosaic-GPU kernels (→ `axis_index`/`axis_size`) ([changelog](https://docs.jax.dev/en/latest/pallas/CHANGELOG.html)). Churn is also **performance**, not just API: [triton-lang #9640](https://github.com/triton-lang/triton/issues/9640) — a Pallas-Triton kernel with small 8×8 blocks regressed **5–10×** going jax 0.6.2→0.8.0 because Triton 3.x dropped the heuristics that coalesced Pallas's per-element pointer tensors into block loads.

## Q2 — Capability check vs. our kernel needs

| Need | Triton backend (0.10.1) | Mosaic GPU (0.10.1) |
|---|---|---|
| SMEM scratch accumulator | **NO** — verified in installed source: `jax/_src/pallas/triton/lowering.py:321` raises `"scratch memory not implemented in the Triton backend"`. Accumulate in register block-arrays; Triton manages SMEM implicitly | **YES** — explicit `plgpu.SMEM` scratch via `scratch_shapes`/`run_scoped`; SMEM/GMEM/TMEM/registers all first-class ([MGPU reference](https://docs.jax.dev/en/latest/pallas/gpu/reference.html)) |
| Atomic add fp32 | **YES** to GMEM refs (`plt.atomic_add`, plus max/min/cas/xchg) | **YES** since 0.9.2 (add/max/min/and/or/xor per changelog); accumulation idiom is per-block SMEM arrays + barriers rather than SMEM atomics |
| Sorted/segmented reduction in-kernel | No primitive; hand-write with masks + block reduce/scan. `ragged_dot_mgpu` in-tree is the existence proof for segmented patterns | Same; more control, more code |
| Gather with computed indices | **YES** — ref indexing lowers to pointer arithmetic (this is where #9640's per-element-pointer cliff lives at odd shapes) | **YES** but layout-constrained on Hopper (manual loads); hardware TMA-gather along dim 0 is **Blackwell-only** ([MGPU reference](https://docs.jax.dev/en/latest/pallas/gpu/reference.html)) |
| Grid (pixel-tile × view-tile) + register accumulation over a loop | Core model: `grid` + `BlockSpec` index maps + `fori_loop` carry | Same, plus explicit pipelining (`emit_pipeline`) |
| H100 TMA / warp specialization | Not exposed (whatever Triton auto-generates) | **YES** — TMA async copies, `emit_pipeline_warp_specialized`, `plgpu.ACC` wgmma accumulators ([MGPU reference](https://docs.jax.dev/en/latest/pallas/gpu/reference.html)) |

Verified locally (interpret mode, CPU, this macOS machine): scratch (`pl.ANY`) accumulator, computed-index gather, `plt.atomic_add`, `jit`+`vmap` composition, and a `custom_vjp` wrapping a pallas fwd/adjoint pair all work at 0.10.1. Gotcha found: `atomic_add` into an **uninitialized** output ref yields NaN — outputs must be explicitly zeroed.

## Q3 — Composition

- **jit / vmap**: documented and tested — `pallas_call` under `jit` is a plain XLA custom call; `vmap` has a first-class rule (adds a grid dimension) ([quickstart](https://docs.jax.dev/en/latest/pallas/quickstart.html), [design doc](https://docs.jax.dev/en/latest/pallas/design/design.html)).
- **Autodiff**: reverse-mode through kernels is deliberately not the path — the design doc notes transposition turns disjoint-parallel writes into overlapping-parallel writes ("slow with atomics") and states "`jax.custom_vjp` is a viable escape hatch" ([design doc](https://docs.jax.dev/en/latest/pallas/design/design.html)). **Our existing fwd/adjoint pair under `custom_vjp` is exactly the endorsed pattern** (confirmed working locally).
- **Donation**: `pallas_call` takes `input_output_aliases` (verified in the 0.10.1 signature); outer `jit` donation composes as usual.
- **Per-device thread-pool dispatch**: a pallas kernel is a single-device XLA op — no GSPMD interaction; drops into the existing per-device jit exactly like current code. Two caveats: pallas calls are opaque to XLA fusion ([GPU perf tips](https://docs.jax.dev/en/latest/gpu_performance_tips.html)), and they are **not captured in CUDA command buffers/graphs by default** ([issue #27988](https://github.com/jax-ml/jax/issues/27988)) — directly relevant to the VCD ~95%-host-dispatch bottleneck at interactive sizes; a pallas kernel could *worsen* dispatch-bound cases.
- **Interpret mode**: `interpret=True` runs the grid as a sequential scan on any backend incl. CPU — "develop GPU or TPU kernels on any XLA-supported platform (even CPU!)" ([design doc](https://docs.jax.dev/en/latest/pallas/design/design.html)). Deterministic and race-blind: good for CPU-CI value-correctness of race-free kernels, **not** a race/atomics-ordering检测 — GPU runs still need the float-correctness gates.

## Q4 — Evidence for and against

**For:**
- Mosaic GPU flash attention matches FlashAttention-3 performance on H100 in ~200 lines of Python ([announcement thread](https://www.threads.com/@sung.kim.mw/post/C9tQc3oy6-e?hl=en); example in-tree).
- Fused scatter→expert-FFN→gather MoE Pallas kernel: prefill 5.16→2.42 ms by hand-scheduling what "XLA cannot reliably place… onto one hand-scheduled pipeline" ([LMSYS SGLang-JAX blog, June 2026](https://www.lmsys.org/blog/2026-06-17-ling-2-6-tpu/)) — TPU, but the same fuse-scatter-compute-gather class as our kernels.
- In-tree gather-heavy GPU kernels ship at this version (`paged_attention`, `decode_attention`, `ragged_dot_mgpu`) — Google runs this pattern in production.
- Honest note: public GPU benchmarks of Pallas beating XLA on *segment-sum-like* scatter specifically are thin. The strongest case here is internal: `fwd_noscatter` = 3–4% and ncu's 96%-mem-pipe/8%-HBM profile say the gap is precisely the access pattern a hand-tiled SMEM-accumulator kernel targets (the standard CUDA tomography-projector design).

**Failure modes:**
- Odd/small shapes → per-element pointer codegen → uncoalesced loads (Triton, [#9640](https://github.com/triton-lang/triton/issues/9640)); our T=3 taps and arbitrary detector counts are "odd shapes".
- Mosaic GPU rigidity: quickstart examples themselves fail on the MGPU backend ("copies… divisible by the warpgroup size", unimplemented primitives) ([issue #32123](https://github.com/jax-ml/jax/issues/32123)).
- Register spills are a documented first-class hazard in MGPU ([reference](https://docs.jax.dev/en/latest/pallas/gpu/reference.html)); "Pallas is very slow" reports exist when the model is misused ([#19350](https://github.com/jax-ml/jax/issues/19350)).

## Q5 — Portability and the FFI alternative

- **Arch matrix at the 0.10.1 pin**: Mosaic GPU = **Hopper + Blackwell only**; Ampere support (mma via `cp.async`) lands in **0.10.2 — the exact version this project excludes**. Triton backend covers Ampere and later (A100, Ada/L40S, consumer) best-effort. ([changelog](https://docs.jax.dev/en/latest/pallas/CHANGELOG.html), [quickstart](https://docs.jax.dev/en/latest/pallas/quickstart.html))
- **Autotuning**: none built into Pallas — block sizes / `num_warps` / `num_stages` are hand-picked per arch via `CompilerParams`; plan on manual sweeps per GPU class.
- **Pure-python wheels**: Pallas kernels are Python source compiled at runtime by the XLA bundled in jaxlib — **zero packaging burden; the only option meeting the hard constraint.**
- **jax.ffi / hand CUDA**: requires building a shared library against XLA's FFI headers (CMake/nanobind), per-platform binary wheels or user-side compilation, CUDA-stream handlers registered per platform, and extra plumbing for vmap/grad ([FFI docs](https://docs.jax.dev/en/latest/ffi.html)). The docs themselves steer libraries toward Pallas for lower "development and maintenance cost". FFI **breaks the pure-python constraint**; only viable as an optional plugin package, doubling release surface. [jax-triton](https://github.com/jax-ml/jax-triton) (raw `triton_call`, v0.3.1 Feb 2026, alive) is middle ground but re-adds a fragile external `triton` dependency.

## Verdict

**Viable at 0.10.1 — as an opt-in accelerated path, not a replacement.** Pallas is the only vehicle compatible with pure-python wheels; `custom_vjp` over our existing fwd/adjoint pair is the officially endorsed autodiff pattern; interpret mode slots into CPU CI (race-blind — keep GPU float-gates). But the pin forces an awkward backend choice: **Mosaic GPU** has everything the kernels need (explicit SMEM accumulators, TMA, warp specialization, atomics) yet is H100/B200-only at 0.10.1; **Triton** covers all user GPUs and has `atomic_add`, but has *no SMEM scratch* (verified in source), is best-effort-maintained, and has demonstrated cross-version 5–10× perf cliffs. Practical shape: flag-gated Pallas kernels (like the campaign's TilePolicy flags), enabled only on supported archs, XLA kernels as the always-correct fallback.

**Materially better jax version**: the first good release **after 0.10.2** (0.10.3/0.11.x) — 0.10.2 adds Ampere support to Mosaic GPU, collapsing the backend dilemma, but is excluded for the unrelated XLA forward-projection regression; re-test per the existing pyproject protocol. If targeting that horizon, write MGPU kernels against `axis_index`/`axis_size` (0.10.2 breaks `program_id` in MGPU).

**Top 3 risks**
1. **Churn in an experimental namespace**: three breaking API changes within 0.9.0→0.10.2 plus silent performance regressions across upgrades (#9640) — the pin-and-retest burden lands on a small-team library that already excludes 0.10.2 for a different regression.
2. **Arch fragmentation**: the capable backend doesn't run on A100/L40S/consumer at the pin; the portable backend lacks the SMEM-accumulator design the workload wants — risk of maintaining two kernel variants or shipping H100-only wins.
3. **The workload sits in both backends' documented weak spots**: computed indices, T=3 taps, arbitrary detector counts (per-element pointer codegen, warpgroup-divisibility copy limits, register spills) — and it's where public Pallas-beats-XLA evidence is thinnest; budget for recovering only part of the ~10× without an algorithmic re-tiling, and note pallas calls skip CUDA command buffers by default (#27988), which could hurt the dispatch-bound VCD path.

Sources: [Pallas changelog](https://docs.jax.dev/en/latest/pallas/CHANGELOG.html) · [Mosaic GPU reference](https://docs.jax.dev/en/latest/pallas/gpu/reference.html) · [Pallas design doc](https://docs.jax.dev/en/latest/pallas/design/design.html) · [Pallas quickstart](https://docs.jax.dev/en/latest/pallas/quickstart.html) · [Pallas index](https://docs.jax.dev/en/latest/pallas/index.html) · [pallas.triton module docs](https://docs.jax.dev/en/latest/jax.experimental.pallas.triton.html) · [JAX FFI docs](https://docs.jax.dev/en/latest/ffi.html) · [GPU perf tips](https://docs.jax.dev/en/latest/gpu_performance_tips.html) · [triton-lang #9640](https://github.com/triton-lang/triton/issues/9640) · [jax #32123](https://github.com/jax-ml/jax/issues/32123) · [jax #27988](https://github.com/jax-ml/jax/issues/27988) · [jax #19350](https://github.com/jax-ml/jax/issues/19350) · [jax #18603](https://github.com/jax-ml/jax/issues/18603) · [jax-triton](https://github.com/jax-ml/jax-triton) · [LMSYS SGLang-JAX MoE kernel blog](https://www.lmsys.org/blog/2026-06-17-ling-2-6-tpu/) · [Pallas-for-beginners (HF blog)](https://huggingface.co/blog/ariG23498/pallas-for-beginners) · [MGPU FA3 announcement](https://www.threads.com/@sung.kim.mw/post/C9tQc3oy6-e?hl=en) · local verification: `/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirjax/pyproject.toml`, `/Users/gbuzzard/miniforge3/envs/mbirjax/lib/python3.11/site-packages/jax/_src/pallas/triton/lowering.py:321`
