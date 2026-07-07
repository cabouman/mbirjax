# Parallel-beam forward vs back kernel: attribution and alternatives

**Written 2026-07-07** (branch `greg/performance_improvements`).  Companion to
`fwd_back_kernel_ab.py` (this directory), which produced every number below.  Question: why is
`parallel_beam.forward_project_pixel_batch_to_one_view` slower than
`back_project_one_view_to_pixel_batch` (nightly model-level: fwd/back ≈ 1.4–1.9× GPU, ≈ 1.5× CPU)?

**Answer in one line:** forward's psf loop is a **scatter-add with duplicated channel indices**
(~3·P/C pixels collide per channel); back's is a conflict-free **gather** + dense accumulate.  The
`fwd_noscatter` control (identical arithmetic, no channel scatter) runs at 1–6% of the forward
kernel on both platforms, so the scatter is essentially the whole kernel-level story.

## Method

Kernel-level timings use the PRODUCTION kernel shape: a 2048-pixel batch (the driver's
`pixel_batch_size_for_vmap` scan unit) vmapped over 128 view angles
(`fwd_view_batch_size_for_vmap`), on one device.  Back is timed as its driver composes it
(vmap over views, then the view sum).  A driver-level ground truth
(`projector_functions.sparse_forward_project` / `sparse_back_project` on the full pixel grid)
anchors the kernel numbers to reality.  Values are checked with a robust
fraction-of-deviating-elements metric (see "Numerics note").

## Variants and results

All variants are value-equal reformulations of the forward kernel (same `n`, `A` from the same
geometry code; only the channel reduction differs), except `fwd_noscatter`.

| variant | what it does | CPU (M3) | GPU (H100) |
|---|---|---|---|
| `fwd_asis` | library kernel: 3 sequential `.at[n,:].add` scatters, one per psf tap | 2.7–3.4× back | 1.07× back (448-size), 1.29× (1024-size) |
| `fwd_1scatter` | ONE stacked scatter: (3P,) indices, (3P,S) updates | ~1.2× worse than as-is | ~1.45× worse than as-is |
| `fwd_segsum` | `jax.ops.segment_sum` over the stacked indices (unsorted; lowers to scatter) | ≈ `fwd_1scatter` | ≈ `fwd_1scatter` |
| `fwd_sortsegsum` | argsort the stacked indices IN-kernel, `segment_sum(indices_are_sorted=True)` | **2–4× WORSE** | **2.3–3.2× BETTER** (0.34–0.57× back) |
| `fwd_matmul` | dense (C,P) one-hot weight matrix @ (P,S) voxels — C/3× redundant flops | ties at C=160, loses at C=384 | **5.8× better — but TF32 only**; forced float32 it loses at 1024-size (1.93× back) |
| `fwd_gathertable` | host-precomputed per-view channel→pixel tables padded to (C,K); kernel = pure gather + reshape-reduce | 19× worse | 16× worse |
| `fwd_noscatter` | lower bound: all compute, psf reduce, NO channel scatter (wrong values) | 1–6% of as-is | 3–4% of as-is |

Notes per variant:

- **`fwd_1scatter` / `fwd_segsum`:** merging the 3 taps into one scatter does NOT help — the cost
  is the scattered accumulation itself, not the kernel-launch count.
- **`fwd_sortsegsum` — the GPU winner.**  With sorted indices XLA lowers the segment reduction to
  an efficient contiguous reduce instead of atomics; the win *includes* the in-kernel argsort
  (~6K elements/view at production shape).  On CPU the same path is far slower than the plain
  scatter — XLA's CPU sort+segment lowering is poor — so this is a **platform-conditional** kernel
  (precedent: back projection already splits monolithic-vs-band by platform).
- **`fwd_matmul`:** the spectacular GPU number rides TF32 tensor cores (JAX default matmul
  precision on H100).  Forced to full float32 it is no longer a win at scale.  Using it would be
  a *precision-policy* decision, not a drop-in optimization.
- **`fwd_gathertable`:** dead end as padded tables — per-channel pixel counts skew ~17× above the
  mean (K = 19439 vs mean ~1150 at the 448-size full grid), so padding waste dominates.  The
  unpadded form of this idea IS `fwd_sortsegsum`.

## The three tiers of the gap (GPU)

| tier | fwd/back | where the extra time lives |
|---|---|---|
| kernel (production shape) | 1.07–1.29× | the channel scatter |
| raw driver (full grid) | 1.06× (448) → **1.56×** (1024) | forward's ~480-step pixel scan accumulating into a ~512 MB (V,C,S) carry; back's driver has no equivalent |
| nightly model level | 1.4–1.9× | banded forward vs back's n=1 monolithic short-circuit; ragged view batches at odd view counts |

On CPU the kernel tier dominates outright (kernel 2.7–3.4×, driver 2.4–3.1×); the nightly's
smaller 1.5× is because the deployed CPU paths use band kernels on both sides.

## Strategy ranking (assessment of 2026-07-07; no library changes yet)

1. **GPU platform-conditional sorted-segment-sum reduction** — verified 2.3–3.2× kernel win,
   exact values, pure XLA.  Structure: keep ONE shared kernel body (geometry → `n`, `A`) and
   platform-dispatch only the ~10-line channel reduction.  The same primitive drops into the
   cone/multiaxis/translation horizontal fans (same scatter structure).
2. **Driver tier:** revisit `pixel_batch_size_for_vmap` for parallel forward (fewer scan steps →
   less carry traffic; 2048 was tuned for cone's gather path).
3. **Model tier:** quantify banded-forward overhead at n=1 vs a monolithic short-circuit
   mirroring back's.
4. **TF32 matmul:** only if a reduced-precision forward is ever acceptable to the correctness
   gates (doubtful).
5. **CPU:** accept for now — the scatter is XLA-CPU-inherent; the honest alternative is an
   algorithmic rewrite (shear/resample forward), a much larger project.

## Numerics note (ties to the known JAX rounding bug)

Early full-grid runs showed two *value-equal* compiled programs (`fwd_asis` vs the stacked
variants) differing by ~4e-3 relative on isolated elements: 2 of 32 views, channels `c−1`/`c+1`
around a rounding tie, `c` itself clean, ~0.3–0.5 magnitude per affected cluster.  That is the
**antisymmetric ±1-channel signature of the known rounding bug**
(`experiments/bugs_and_artifacts/jax rounding bug/jax_rounding_bug.md`): the projection is
mathematically continuous at exact .5 ties (PSF clipping), so a compilation-dependent value
change means the scatter destination `n` and the weight's `|n_p − n|` disagreed about
`n_p_center` — the same family as the vmap→map→scatter mis-optimization documented there, here
surfaced by comparing two differently-optimized programs.  Consistent with that bug's known
input sensitivity, the effect vanished at production shapes (frac > 1e-4 = 0 everywhere).
The planned host-side precompute of `n_p_center` (that doc, §3–4) removes the ambiguity at the
root and would also let the sort permutation be precomputed/cached per (view, pixel batch).
The bench therefore checks values with a fraction-deviating metric, never max-error.
