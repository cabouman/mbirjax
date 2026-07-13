# Forward-kernel dispatch guard: P × band sweep (fwd_guard)

**Status: sweeps 1–2 COMPLETE (jobs 13497787 / 13497833, 2026-07-13) — no knee
found ANYWHERE, pallas ≥3.1× through full grid; the band=2048 XLA cliff is NOT the
2^31 boundary (views ablation refuted it).  Sweep 3 (small-band corner) running.
Guard proposal drafted below, pending sweep 3 + Greg approval.**

## Question

The pallas forward path currently dispatches on `pixel_count <= tiles.fwd_pixel_batch`
(= 8192) in `sparse_forward_project_public` — a constant inherited from the XLA
driver's pixel-tile size, not a measured property of the pallas kernel.  What is the
measured speedup-vs-P knee per band width, and what band-aware guard should replace
the constant?

Prior physics conjecture: the kernel's win rests on the shared values tile
(P × band × 4 B) staying L2-hot across views (H100 L2 = 50 MB), predicting knees at
P ≈ 24.4k / 12.2k / 6.1k for bands 512 / 1024 / 2048.

## Method

`plans/experiments/projector_kernels/fwd_guard_sweep.py` (+ `.slurm`), modeled on
`e4_ab_back.py`: each (band, P, impl) cell in an isolated subprocess (JAX-free
orchestrator, honest `peak_bytes_in_use`); one H100; sino (views=1024, rows=band,
channels=1024) so band = num recon slices = the values second dim; P ∈ {2048, 4096,
6144, 8192, 12288, 16384, 24576, 49152} (8192 = the current guard, an exact point);
warmup 2 / median of 5 timed trials; values gate rel-max ≤ 1e-5 per cell pair.
The pallas cell calls `_pallas_kernels.forward_project_subset(model, values, idx)`
directly (bypasses the wrapper guard — no library edit); the XLA cell runs the
public wrapper under `MBIRJAX_DISABLE_PALLAS=1` (the production fallback, with its
own 8192 pixel batching).  Identical seeded idx/values per pair.

## Sweep 1 results (job 13497787, H100 80GB, jax 0.10.1)

All 48 cells value-gated PASS (worst rel-max 7.6e-7, gate 1e-5).  Speedup = XLA
wall / pallas wall, medians of 5:

| band | P: 2048 | 4096 | 6144 | 8192 | 12288 | 16384 | 24576 | 49152 |
|---|---|---|---|---|---|---|---|---|
| 512  | 3.13 | 3.08 | 4.24 | 4.82 | 4.92 | 5.43 | 5.76 | **5.95** |
| 1024 | 5.94 | 6.31 | 6.00 | 5.81 | 6.39 | 5.95 | 6.01 | **6.05** |
| 2048 | 61.6 | 100.9 | 122.6 | 136.3 | 149.8 | 160.2 | 155.7 | **167.9** |

Walls (ms), XLA → pallas, selected: b512 P49152: 328 → 55; b1024 P49152: 587 → 97;
b2048 P49152: 31,507 → 188.  Full grids in the job log and
`fwd_guard_sweep_results.json` (scratch).

### Read 1: NO knee anywhere in the swept range — the L2 conjecture is falsified

Speedup is monotonically non-decreasing in P for every band; at band=512 the tile is
96 MB (~2× L2) at the top point and the win is still RISING (5.95×).  Mechanism, from
linear fits wall ≈ a + b·P per band: the pallas slope is near-proportional to band
(b ≈ 1.04 / 1.84 / 3.59 µs/pixel for 512/1024/2048 — i.e. ~1.8–2.0 ns per
pixel·band-element, near the HBM traffic bound of ~V·P·band·4 B per call), so past
L2 the kernel degrades into an orderly HBM-streaming regime — but the XLA path pays
the SAME traffic plus its sort/scatter overhead (b ≈ 6.5 / 11.2 / 644 µs/pixel), so
the ~6× constant-factor gap survives L2 exhaustion.  The L2 knee exists only as a
mild slope change, invisible in the ratio.  Consequence: the guard cannot be derived
from L2 capacity; the remaining question is whether the gap persists to full-grid P
(~823k at this geometry) — sweep 2.

### Read 2: the band=2048 XLA column is a separate pathology — 2^31 hypothesis REFUTED

Band=2048's XLA per-pixel slope (644 µs/px) is **58×** band=1024's — categorical at
every P, while pallas scales smoothly (1.94× slope for 2× band).  First hypothesis:
at views=1024, band=2048, channels=1024 the sinogram element count is EXACTLY 2^31,
suggesting 64-bit-index slow paths (the lessons.md §4 class, a silent PERFORMANCE
face).  Sweep 2's views=512 ablation REFUTED it: at 2^30 elements (band unchanged)
the XLA wall halved exactly (5063 → 2517 ms, ratio 0.497; the band=1024 control also
halved, 114.5 → 57.4 ms), so the per-work cliff is fully intact below 2^31.  The
cliff is genuinely BAND-categorical in the XLA forward program (present at
band=2048, absent at ≤1024, at any views/element count).  Most plausible mechanism:
a fusion/materialization threshold in the sorted-channel-reduce lowering (a
band-wide per-tap intermediate that stays fused at ≤1024 columns and materializes at
2048); the discriminator would be an HLO dump of the two programs — DEFERRED, it
does not affect the guard (pallas wins the column by 62–168×).  It DOES matter for
whoever runs the XLA fallback (multi-device, non-allowlisted arch, kill switch) on
≥2048-slice parallel-beam problems — flagged as follow-up.

### Memory

Pallas peaks stay flat at ~2 sinograms (4.0 / 8.0 / 16.0 GB for bands 512/1024/2048;
+0.1–0.4 GB across the whole P range — the streams are per-view-chunk transients).
XLA peaks match pallas at small P but jump ~+50% in the P=12288/16384 window (e.g.
24.3 vs 16.1 GB at band=2048) — the single-call n_pc window before the wrapper's
view-chunking threshold (256 MB) kicks in at P ≥ ~24576.  No cell approached the
80 GB card.

## Sweep 2 results (job 13497833): the win persists to FULL GRID

`fwd_guard_sweep2.py`: P extension at bands 512/1024, band=2048 single point, views
ablation.  All 13 pairs value-gated PASS:

| band | P: 98304 | 196608 | 393216 | 786432 | full (821,904) |
|---|---|---|---|---|---|
| 512  | 5.79 | 5.68 | 4.70 | 3.73 | **3.76** |
| 1024 | 5.72 | 4.84 | 3.69 | 3.09 | **3.17** |
| 2048 | 138.3 | — | — | — | — |

Full-grid production walls (views=1024, channels=1024): band=512: 4.85 s → 1.29 s;
band=1024: 8.81 s → 2.78 s (the E4 fwd_full XLA wall at the 1024³-class cell was
7.97 s at rows=1008/channels=992 — consistent).  The margin erodes gently from ~6×
(P ≤ 200k) to ~3.1–3.8× at full grid but never approaches 1×.  Two acks: (1) pallas
full-grid peak is +2.6 GB (band 1024) / +4.9 GB (band 512) over XLA — the streams
(view_chunk × 3P × 8 B) + values-tile transients; bounded and small next to the 80 GB
card, but it scales with P·band, worth restating at the next detector-size jump.
(2) rel-max grows with P (6.1e-6 at band=512 full) — atomic-add summation-order noise
scaling with collisions per channel (P/C ≈ 800 at full grid), same magnitude as the
documented same-executable GPU run-to-run noise (~8e-6); the 1e-5 single-shot gate
still passes but sits close — iterated/full-pipeline comparisons should use the
existing 1e-4/1e-3 tiers, as always.

## Sweep 3 (running): the small-band corner + pad path

`fwd_guard_sweep3.py`: bands {128, 256} × P {2048, 8192, 24576} (views/channels
1024 — single-variable vs sweep 1's band=512 column); band=768 at P=8192 (pad to
1024, 33% wasted columns — every sweep-1/2 band was a power of two, so the driver's
`vals_pad` path was unmeasured); and a small-problem pair at production aspect,
sino (256, 256, 256), P ∈ {2048, full≈51k} — the op-level mirror of E4's vcd_guard.

## Proposed guard (draft — pending sweep 3 + Greg approval)

**Drop the pixel-count clause entirely: dispatch pallas for every single-device GPU
parallel-beam forward call when `tiles.fwd_pallas` is set.**  Rationale: pallas
measured ≥3.1× at EVERY point in a 3-decade P range (2048 → full grid) × bands
512–2048, with the minimum at full grid — there is no measured regime where the XLA
path wins, so any pixel-count threshold would be an unmeasured complication.  The
"likely formula" from the task framing (min of an L2 cap and a measured knee) is
moot: neither the L2 cap nor a knee exists in the data (Read 1).

Concretely (post-approval):
- `projectors.py sparse_forward_project_public`: the dispatch condition becomes
  `getattr(tm.tiles, 'fwd_pallas', False)` alone; comment rewritten to cite this doc.
  `tiles.fwd_pixel_batch` keeps its meaning as the XLA path's pixel batching (the
  fallback still uses it).
- `parallel_beam.py _select_tile_policy`: comment update only (`fwd_pallas` is no
  longer "subset-sized calls only"); the flag itself already carries the
  n_devices == 1 + is_available() gating, which is unchanged.
- `tests/test_pallas_kernels.py`: extend so a call with P > fwd_pixel_batch
  dispatches pallas and matches XLA at the 1e-5 gate (interpret mode on CPU CI).
- Validation after the diff: rerun E4's six-cell A/B — fwd_full flips from a
  "policy keeps XLA (1.00×, bitwise)" check to a "pallas ≈3× and value-gated" cell;
  vcd_guard should not regress (expected: same or slightly better than 1.06×, since
  coarse-granularity forward calls now also go pallas).

**Behavioral change requiring explicit sign-off: full-grid forward projection
becomes atomics-nondeterministic run-to-run** (~1e-6..6e-6 rel), where the XLA
sorted-reduce was deterministic per executable.  Precedent: full-grid BACK
projection already shipped exactly this trade in E4 increment 1 (atomic adds, 9.1×),
and the float-gate policy (lessons.md §2) was designed for it.  MBIRJAX_DISABLE_PALLAS
remains the escape hatch.

## Open follow-ups (not this session's scope)

1. **XLA forward band≥2048 cliff** (Read 2): ~50× per-work penalty on the XLA path
   at ≥2048 slices, band-categorical, NOT the 2^31 boundary (ablated), values
   correct.  Hits XLA-fallback users (multi-device, non-H100, kill switch) on
   2048-row parallel-beam problems today.  Next discriminator: GPU HLO dump of the
   band=1024 vs band=2048 forward programs (fusion/materialization threshold is the
   leading hypothesis).
2. Whether the multi-device banded forward (wave 2, the other session) inherits the
   no-guard conclusion — its band is `fwd_slice_band` = 256, BELOW sweep 3's corner
   cells; sweep 3's band-256 column is the relevant evidence.

## Library-change requests for the kernel session (none applied here)

None so far: the sweep needed no `_pallas_kernels.py` changes (the driver's
shape-static design held from P=2048 to full grid; grid segment counts at full-grid
P≈823k are ~38k < the 65535 CUDA grid-dim bound, checked before sweep 2).

## Library-change requests for the kernel session (none applied here)

None so far: the sweep needed no `_pallas_kernels.py` changes (the driver's
shape-static design held to P=49152; grid segment counts at full-grid P≈823k are
~38k < the 65535 CUDA grid-dim bound, checked before sweep 2).
