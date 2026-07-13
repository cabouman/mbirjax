# Forward-kernel dispatch guard: P × band sweep (fwd_guard)

**Status: sweep 1 COMPLETE (job 13497787, 2026-07-13) — no knee found in range;
sweep 2 (full-grid extension + views ablation) running.  Guard proposal pending
sweep 2 + Greg approval.**

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

### Read 2: the band=2048 XLA column is a separate pathology, likely the 2^31 boundary

Band=2048's XLA per-pixel slope (644 µs/px) is **58×** band=1024's — categorical at
every P, while pallas scales smoothly (1.94× slope for 2× band).  At views=1024,
band=2048, channels=1024 the sinogram element count is EXACTLY 2^31; the hypothesis
is XLA switching the forward program's gather/scatter/sort onto 64-bit-index slow
paths (the lessons.md §4 boundary class — here a silent PERFORMANCE face, values
still gate PASS at 2e-7..7e-7).  Sweep 2's discriminator: views=512 (element count
2^30, band unchanged) — if per-work cost snaps back in line, it's the index space,
not the band.  Either way the guard conclusion is unaffected (pallas wins the column
by 62–168×); but the attribution matters for the XLA fallback path's users
(multi-device, non-allowlisted arch), where crossing 2^31 sino elements would mean a
~50× forward slowdown TODAY.

### Memory

Pallas peaks stay flat at ~2 sinograms (4.0 / 8.0 / 16.0 GB for bands 512/1024/2048;
+0.1–0.4 GB across the whole P range — the streams are per-view-chunk transients).
XLA peaks match pallas at small P but jump ~+50% in the P=12288/16384 window (e.g.
24.3 vs 16.1 GB at band=2048) — the single-call n_pc window before the wrapper's
view-chunking threshold (256 MB) kicks in at P ≥ ~24576.  No cell approached the
80 GB card.

## Sweep 2 (running)

`fwd_guard_sweep2.py`: P ∈ {98304, 196608, 393216, 786432, full≈823k} at bands
512/1024 (both impls, same gates); band=2048 single point P=98304; views ablation
(views=512, P=8192) at bands 1024 (control) and 2048 (discriminator).

## Proposed guard

(pending sweep 2 — if the win persists to full grid, the natural proposal is to
drop the pixel-count guard entirely for the single-device parallel forward path,
i.e. dispatch pallas for ALL P when `fwd_pallas` is set, keeping
MBIRJAX_DISABLE_PALLAS as the escape hatch; note this would change full-grid forward
from the deterministic sorted-reduce to atomic adds — run-to-run noise at the ~1e-6
relative level, within the existing float gates, but it needs Greg's explicit sign-off.
If the win erodes, guard at the measured knee.)

## Library-change requests for the kernel session (none applied here)

None so far: the sweep needed no `_pallas_kernels.py` changes (the driver's
shape-static design held to P=49152; grid segment counts at full-grid P≈823k are
~38k < the 65535 CUDA grid-dim bound, checked before sweep 2).
