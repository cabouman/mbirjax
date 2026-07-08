**Q1: Why is forward ~9× slower than back on CPU (and GPU is more comparable at ~2×)?**

The root cause is **scatter vs. gather asymmetry**, and it's particularly severe on CPU.

*Forward projection* (scatter-add): For each pixel `p`, the code computes which detector channels it projects onto and adds to those positions in the sinogram buffer — `sinogram.at[n].add(weight * voxel)`. The critical problem is **write conflicts**: many pixels project to the same channel, so you have repeated updates to the same memory location. XLA's scatter kernel cannot vectorize conflicting writes — it must either serialize them or use atomic operations. On CPU, there is no hardware atomic-add unit for FP32; instead XLA uses a sequential reduction loop. Also, the scatter pattern is random (different pixels project to different channels depending on geometry), so the write addresses are unpredictable, thrashing the cache lines of the output sinogram buffer.

*Back projection* (gather): For each pixel `p`, the code reads from the sinogram at column `n` — `sinogram[:, n]`. Reading from random locations causes cache misses, but **reads never conflict** — any number of threads or vectorized lanes can read the same location simultaneously. XLA can vectorize gather, pipeline cache-miss latency with computation, and use SIMD loads freely.

This asymmetry is not surprising: it's why databases, graph algorithms, and linear solvers on CPU all treat scatter as a much heavier operation than gather.

*Why GPU is more symmetric:* GPUs have hardware warp-level scatter reduction (warp shuffle + atomic scatter in shared memory), HBM bandwidth is symmetric for reads and writes, and the GPU's massively parallel scatter can amortize conflicts across thousands of threads at once. The 2× ratio on GPU (39s vs. 20s) likely reflects a real remaining advantage for gather (random HBM reads have some latency vs. writes) plus differences in total work, not the conflict penalty.

*Possible improvements for forward on CPU:*

1. **Batch-based `lax.map` over pixel batches** (the `gp.entries_per_cylinder_batch` approach): Instead of scattering all `num_pixels` at once into a `(local_slices, channels)` sinogram buffer, map over groups of `batch_size` pixels. Each call scatters only `batch_size` pixels → fewer simultaneous write conflicts per call → each scatter kernel operates on a smaller conflict set, which XLA might handle with a more efficient algorithm. This is the `lax.map`-over-batch-indices pattern from cone beam. Whether this actually helps depends on whether XLA detects and exploits the reduced conflict density — not guaranteed, but worth trying.

2. **Sorting pixels by projected channel**: If pixels are sorted so that nearby pixels project to nearby channels, scatter writes become more sequential and cache-friendly. This is a preprocessing cost but could significantly reduce cache thrashing for repeated forward projections.

3. **Transposing the loop**: Instead of iterating over pixels and scattering to channels, iterate over channel bins and gather which pixels contribute. This converts scatter to gather entirely. It would require reorganizing the geometry lookup and may not be straightforward in the current design, but it's the principled fix.

For now, option 1 (batch lax.map) is the lowest-effort experiment with a reasonable chance of modest improvement.

---

**Q2: Why is back-projection CPU scaling poor, and can it be improved?**

*Why the cache cap helped but wasn't sufficient:*

The 1 MB cap improved 4-dev from 1.32× to 1.93×, which confirms the diagnosis: L3 cache thrashing was the primary bottleneck at 2–4 devices. At 8 devices the speedup actually drops (1.18× → 1.22× is negligible), which points to a different bottleneck: **memory bandwidth saturation**.

8 virtual CPUs on one physical machine share a single memory bus. At 8 threads, the aggregate sinogram read bandwidth is roughly `8 threads × view_batch × local_slices × channels × 4 bytes`. Even with the 1MB cap, 8 concurrent threads issuing cache-miss-heavy gather reads can saturate the off-chip memory bus. Once you hit bandwidth saturation, adding more cores gives no speedup (or even slight regression due to bus contention overhead).

The breakeven point for "enough threads to saturate the bus" on a typical machine is usually 4–6 physical cores, which aligns with the data: 4-dev scales well, 8-dev doesn't.

*Possible improvements:*

1. **Batch-based `lax.map` over pixel batches in `back_project_one_view_to_pixel_batch`**: This was the intended change. The current kernel gathers from the sinogram into a `(num_pixels, local_slices)` output cylinder in one shot. Breaking it into batches of `gp.entries_per_cylinder_batch` pixels reduces the peak output buffer size from `num_pixels × local_slices × 4` bytes to `batch_size × local_slices × 4` bytes. If the batch output fits in L1/L2 cache, subsequent gather operations reuse cached output addresses rather than evicting them. This could improve single-thread performance and reduce the per-thread footprint that causes cross-thread cache contention.

2. **Reduce memory-bus pressure by computing fewer gather indices**: At 8 devices, sinogram shard per thread is `n_views × (n_rows/8) × n_channels`. For the test case, that's `180 × 16 × 256 × 4 = 2.9 MB` — even the shard doesn't fit in L2. The batch-lax.map approach won't shrink the sinogram read; it only shrinks the output cylinder. This limits its effectiveness for the bandwidth bottleneck.

3. **Reduce view_batch further at high device counts**: The current 1 MB cap is per-thread. At 8 threads, a 128 KB cap (so total = 1 MB across all threads) might reduce bus pressure. Could be parameterized as `_CACHE_TARGET_BYTES / n_devices`.

*Practical recommendation:*

Apply the batch lax.map to back_project (the intended change) — it's likely to help single-thread performance by reducing the output cylinder working set. For multi-device scaling: CPU scaling at 2–4 real physical cores is ~1.7–2×, which is respectable. Virtual CPU scaling at 8+ hits bus saturation regardless of algorithmic changes. For GPU, the 2× at 2-GPU result is the number that matters, and the architecture is fundamentally better suited to both forward and back projection.

Given that forward and back are always called in pairs, and forward dominates at 9:1 on CPU, optimizing back scaling further may not move the needle on total reconstruction time. The batch lax.map for back is still worth doing because it's the "correct" structure for the kernel (matching cone beam) and may improve single-thread performance, but I wouldn't expect 8-dev CPU scaling to improve dramatically.