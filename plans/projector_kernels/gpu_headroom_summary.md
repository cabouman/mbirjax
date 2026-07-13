# GPU headroom campaign — plain-language summary

(Reader-facing companion to `gpu_headroom_plan.md` (the plan) and
`gpu_headroom_findings.md` (the measured record).  Written 2026-07-12 after the E3
kernel spike; update as milestones land.)

## What a Pallas kernel is (background for everything below)

Pallas is jax's built-in kernel language: GPU kernels written in Python with
numpy-style syntax, compiled through Triton (NVIDIA's kernel compiler) instead of XLA.
It ships inside jax — no new dependency, no CUDA build step — and the same kernel runs
in a slow "interpret mode" on CPU, which is how the correctness tests run in CI
without a GPU.

**The core inversion.**  With ordinary jax/XLA you write whole-array math and the
compiler decides the loop structure, the fusion, and how memory is touched.  Pallas
flips that: you write ONE small program plus a *grid*, the program runs once per grid
point (thousands of instances concurrently), and *you* decide what each instance
reads, computes, and writes.  Our back kernel's grid is (row-chunk, pixel): each
program instance owns one pixel's chunk of output rows, loops over all views and psf
taps, and writes its result once.  Programs see memory through *refs* (declared views
into device arrays); indexing a ref is a load, assigning into one is a store, and when
two programs might write the same location the write must be an *atomic add*
(hardware-serialized) — which is exactly the collision cost the forward kernel's
two-phase design minimizes.

**Why it beats XLA here** — not because XLA generates bad code (E0 showed its fusions
are near-optimal within their model), but because Pallas can express two things the
whole-array model structurally cannot:

1. *Registers across a reduction.*  The back kernel keeps its accumulator in registers
   while looping over all views — the view sum never touches memory.  XLA's fusion
   materializes per-view partials into the reduction.  This is most of the 9×.
2. *Scheduling for cache residency.*  The grid is ordered so that all
   concurrently-running programs gather from the same ~33 MB slice of the sinogram,
   which then lives in L2 — the transaction-bound gathers (the E2a-confirmed limiter)
   become cache hits.  XLA offers no handle on which programs run near each other.

The forward kernel adds a third trick: restructuring a scatter (colliding atomic adds)
into a gather over pre-sorted contributor streams, walked segment by segment.

**What it costs.**  Everything is static and explicit.  All shapes are compile-time
constants — any data-dependent launch shape recompiles per call (the increment-2
0.68× episode).  The Triton backend at our jax pin has sharp edges: no in-kernel
gather of block-loaded arrays (ref-level indexing is the one supported idiom),
power-of-two block shapes, no shared-memory scratch.  Performance is per-architecture
(constants measured on H100; the allowlist refuses anything unmeasured).  And it is a
second implementation of physics we already have — hence the update/retire protocol in
`_pallas_kernels.py`, the float-tolerance and adjoint gates against the XLA path, and
the XLA path staying compiled-in everywhere as the permanent fallback (inspect what a
run will actually use via `model.get_compute_config()`).

One-sentence version: XLA is a compiler you trust with the loop structure; Pallas is
you taking responsibility for the loop structure — worth it precisely where the win
lives in memory choreography the compiler can't see.

## Where the campaign stood before E3

The attribution rounds (E0–E2a) reduced the "GPU kernels are ~10× above compute bounds"
claim to two sharp facts: forward projection's device time is almost entirely ONE
XLA-generated kernel (the sorted-reduce fusion, 86–88%), and back projection's is one
other (the gather-reduce fusion, 98%).  Both run at ~1% of the GPU's arithmetic peak,
limited by memory access patterns.  Every cheap remedy was tested and closed: XLA
already fuses everything; the runtime sort is only 2–3%; cache-friendlier tile shapes
don't help.  Conclusion: the remaining factor is reachable only by writing the kernels
ourselves.

## E3: the first custom kernel (forward horizontal fan)

**The operation.**  Per view, each pixel's column of voxel values contributes to its
~3 neighboring detector channels with trapezoid weights:
`out[channel, :] += weight × values[pixel, :]`.  XLA sorts the contributions by channel
at runtime, then scatter-adds.

**Our kernel flips scatter to gather**, using two things already engineered: channel
targets are computable BEFORE the kernel runs (the concrete-centers machinery from the
rounding-bug fix), and VCD's pixel subsets are fixed for a whole recon.  A precompute
builds, per view, the contributor list sorted by channel plus each channel's starting
offset.  The kernel is then one small GPU program per (view, channel): walk the
contributor list, fetch each pixel's row, multiply–add in on-chip registers, write the
finished output row once.  No runtime sort, no write collisions, no atomics in the
common path.  Written in Pallas (pure Python, compiles through the jax we already
ship — no CUDA build step; the Triton backend, after a 4-round probe established that
ref-level indexing is the one supported gather idiom at our jax pin).

**Parallel-beam results (H100, 1024³-class cell, values ≤6e-7 relative):**

| round | what changed | subset batch (VCD shape) | raster batch (full grid) |
|---|---|---|---|
| v1 | plain design | **1.96×** vs XLA | 1.42× (channel skew stalls programs) |
| v2 | split long channels (≤CAP pieces, atomic adds) | 1.78× (zeros-init tax) | **1.59×** |
| v3 | two-phase: store first pieces, atomic-add leftovers | **2.13×** | 1.41× (extra launch + empty channels) |

**Verdict: the spike bar (≥1.5–2×) is met by a variant policy** — v3 for uniform
batches, v2 for skewed ones — chosen for free from statistics the precompute already
produces (the established TilePolicy pattern).  Sum order changes, weights don't:
adjointness is preserved by construction, and the usual relative-error gates apply.

**Cone hfan, first round (same day): values pass, performance does not yet transfer**
(subset 1.14×, raster 0.84×).  Three identified reasons, all tuning-addressable: cone's
input is PER-VIEW (the vertical fan's output, ~2 GB total) so parallel's always-L2-hot
shared values tile becomes a streaming working set; rows are 4× wider (4 KB vector work
per tap moved by a single warp at the parallel-tuned num_warps=1); and segments are
half as long (~12 taps), doubling per-program overhead weight.  Cone v2 = row-chunked
grid + warp sweep + view-major scheduling.

**Honest caveats.**  (1) The stream precompute (~145 ms per 128-view/8k-pixel batch on
the H100) amortizes to nothing across VCD iterations but must be charged to one-shot
projections.  (2) The isolated bench flatters the XLA baseline relative to its in-scan
production context — the real end-to-end number comes from the E4 model-level A/B
(compose, don't extrapolate).  (3) This kernel covers FORWARD; back projection's
98%-fusion is the follow-on with the same machinery (agreed: integrate both together).

## Q&A (Greg, 2026-07-12)

**How H100-specific are these results?**  Three layers.  The DESIGN (CSR segment walk,
register accumulation) is architecture-generic — it is the standard sparse-matrix /
segmented-reduction structure.  The BACKEND (Pallas-Triton) runs on Ampere and newer
(A100, L40S, consumer) — one reason it was chosen over Mosaic GPU, which is Hopper-only
at our pin.  The NUMBERS are H100-measured: they depend on L2 size (50 MB — the
values-tile residency argument), SM count, atomic throughput, and the XLA baseline's
own H100 autotuning; expect qualitatively similar but re-measured wins elsewhere, gated
per-arch in TilePolicy exactly like the existing platform-conditional kernels (and
num_warps retuned — cone just demonstrated why).  The larger portability risk is
VERSION, not architecture: the Triton backend is maintained best-effort upstream with
documented cross-version performance cliffs — the pin-and-retest discipline applies.

**How big is the power-of-2 padding?**  Only the ROW/BAND axis (the kernel's vector
width) needs padding — pixel batches and stream lengths are scalar-indexed and
unconstrained.  Typical cost ≈ 1–2%: parallel band 252→256 (+130 KB on an 8 MB tile),
cone rows 1008→1024 (+33 MB on 2.06 GB), output the same factor plus one scratch row.
The pathological case is a row count just past a power of two (e.g. 513→1024 ≈ 2×);
two mitigations exist — for parallel the band size is OUR knob (prefer power-of-2
bands on the pallas path), and for cone's data-fixed row counts the row-chunked grid
that cone tuning needs anyway also removes the padding blowup (pad only to the chunk
multiple).

**Is the precompute per partition, and what does it cost?**  The streams depend on
(pixel subset, views), so yes: per (subset, view-chunk), fixed across iterations
because partitions are fixed per recon.  Consequences: the repeated fine-tail
iterations (11–45 at granularity 128 in production sequences) amortize it away; the
one-shot coarse iterations do NOT — but the natural policy makes this moot: keep XLA
for one-shot coarse subsets (which are also the skewed case where the kernel is
weakest) and use the pallas path for the repeated fine tail (where it is strongest,
2.13×).  Memory (CORRECTION to an earlier statement of "3×"): the streams are 6× the
`n_p_centers` bytes — two arrays (weights f32 + pixel ids i32) × T=3 taps per (view,
pixel) = 24 B/(view·pixel) vs the centers' 4.  Concretely at the 1024³ n=1 cell:
full-grid calls, chunked at 128 views → 2.37 GB resident during the call ≈ 0.58× the
sinogram (freed after; same transient class as the Phase B/D memory acks); a
granularity-128 VCD subset (6.3k pixels × all views) → 156 MB ≈ 4% of the sinogram —
recomputed per call or host-cached and streamed (~6 ms over PCIe); caching all 128
subsets on-device (19 GB) is out at capacity.  A known compression (store per-pixel
geometry, derive tap weights in-kernel) would roughly halve this if it ever matters.

## The composed preview (added 2026-07-12, post-Q&A)

Wiring the back kernel into a full production-shaped call (all pixels × all views,
every cost charged) first measured 3.54× — and the shortfall taught the campaign's
final lesson twice over: the limiter was a bench artifact (a per-call-constructed jit
re-TRACING the weight builder every chunk; the same bug inflated the earlier
"precompute oddity").  With the builder hoisted — the production structure — the
composed result is **9.07× end-to-end for one-shot back projection (10.5 s → 1.16 s),
9.11× for the Hessian**, kernel-bound at 96%, values unchanged.  VCD fine-tail calls,
which amortize their plans across iterations, sit closer to the kernel-level 16–26×.

## E4 agreements (Greg, 2026-07-12)

1. Precompute placement: alongside `n_p_centers` in the public wrappers (same eager
   separate-jit pattern, same chunking rule) — details to be presented with the E4
   design.
2. Policy: TilePolicy flag, measured guards, segment-stat variant choice, XLA fallback
   everywhere else.
3. Gates: kernel-equality (rel 1e-5), poison-the-padding, explicit adjoint test,
   model-level A/B + memory gates + VCD guard cells.
4. **Ship forward + back together** — E4 integration waits for the back-projection
   kernel.
