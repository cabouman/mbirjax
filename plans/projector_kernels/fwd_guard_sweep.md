# Forward-kernel dispatch guard: P × band sweep (fwd_guard)

**Status: SHIPPED (Greg approved 2026-07-13; commit 8ea8f7a).  Sweeps 1–3 (jobs
13497787 / 13497833 / 13500938), 70 cells, all value-gated PASS: pallas wins EVERY
cell (min 1.34×, ≥3× everywhere P ≥ 24576, through full grid); no knee, no L2 cap.
The pixel-count guard is REMOVED; full CPU suite green (299 passed, 2 skipped, 72
subtests); GPU validation = the E4 six-cell A/B rerun (job 13504454), table below.**

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

## Sweep 3 results (job 13500938): the small-band corner + pad path — all wins

`fwd_guard_sweep3.py`, 9 pairs, all value-gated PASS (worst rel-max 1.67e-6):

| cell | XLA → pallas (ms) | speedup |
|---|---|---|
| band=128, P=2048 (v=1024, c=1024) | 5.1 → 3.7 | **1.38×** (the global minimum) |
| band=128, P=8192 | 15.8 → 5.3 | 2.97× |
| band=128, P=24576 | 52.2 → 12.4 | 4.22× |
| band=256, P=2048 | 7.5 → 4.5 | 1.70× |
| band=256, P=8192 | 20.7 → 7.1 | 2.90× |
| band=256, P=24576 | 66.0 → 17.5 | 3.76× |
| band=768 (pad→1024), P=8192 | 84.0 → 18.5 | 4.55× |
| sino (256,256,256), P=2048 | 1.35 → 1.01 | 1.34× |
| sino (256,256,256), P=full≈51k | 30.5 → 8.7 | 3.52× |

Reads: the pallas fixed cost (~3.5–4 ms/call at views=1024, ~1 ms at views=256 —
it is per-view-chunk, 8 chunks vs 2) compresses the margin at the smallest calls
but never inverts it; the pad path is healthy (band=768 padded to 1024 costs
18.5 ms vs true band=1024's 19.7 — the wasted columns are effectively free, while
the XLA path at 768 is 84 ms); the production-aspect small problem wins at both the
fine tail and full grid.  Relevant to wave 2: the multi-device banded forward runs
band = `fwd_slice_band` = 256 — this column says pallas wins there at ≥1.7×
(op-level; the per-owner glue is wave 2's own gate to run).

## Implementation + GPU validation (2026-07-13, commit 8ea8f7a, job 13504454)

The diff (approved by Greg): `projectors.py sparse_forward_project_public` dispatches
on `fwd_pallas` alone (pixel-count clause deleted; `fwd_pixel_batch` remains the XLA
fallback's batching); comment updates in `parallel_beam._select_tile_policy` and the
base policy; `get_compute_config` drops the now-meaningless `fwd_pallas_max_pixels`
key (and its `test_tile_policy` assertion); new
`test_pallas_kernels.test_wrapper_dispatches_pallas_above_pixel_batch` forces a call
above a lowered `fwd_pixel_batch` and asserts both the dispatch and the 1e-5 value
gate (driver spied to run interpret mode on CPU CI).  The kernel session aligned
`_pallas_kernels.py`'s docstrings in the same tree (its own commit).  Full CPU suite:
299 passed, 2 skipped, 72 subtests.

E4 six-cell A/B rerun (flag-on vs kill-switch, fresh subprocesses, H100):

| cell | XLA → pallas | speedup | vs pre-change | values |
|---|---|---|---|---|
| back full-grid | 10.48 → 1.16 s | 9.08× | unchanged | 5.5e-7 PASS |
| back subset | 198 → 24 ms | 8.25× | unchanged | 4.4e-7 PASS |
| Hessian full-grid | 10.48 → 1.16 s | 9.08× | unchanged | 5.1e-7 PASS |
| fwd subset (6,026 px) | 72 → 28 ms | 2.57× | unchanged | 2.5e-7 PASS |
| **fwd full-grid** | 7.99 → 2.35 s | **3.39×** | was 1.00× (guard kept XLA, bitwise) | 5.5e-6 PASS |
| VCD guard (5 it, 256³) | 17.16 → 16.13 s | 1.06× | unchanged | 2.0e-6 PASS |

Reads: fwd_full lands inside the sweep-2 prediction band (3.2–3.8×) with the
predicted atomics-level rel (~5e-6, under the 1e-5 gate — the cell's meaning has
permanently flipped from "guard keeps XLA, rel 0" to "pallas, value-gated"; +4.2 GB
peak, the acked streams/values transients).  vcd_guard holds at 1.06× — at 256³ VCD
is host-dispatch-bound (current_plans §3 pool 1), so the added coverage pays off at
LARGE sizes (full-grid forward is the 3.39× itself), not in the interactive cell.
Every pre-existing cell is unchanged, confirming the diff touched nothing else.

## The approved guard decision (record)

**Drop the pixel-count clause entirely: dispatch pallas for every single-device GPU
parallel-beam forward call when `tiles.fwd_pallas` is set.**  Rationale: pallas
measured faster at EVERY one of 70 cells across P ∈ [2048, full grid ≈ 823k] ×
bands 128–2048 × two problem aspects, minimum 1.34× (the smallest fine-tail call),
≥3× wherever P ≥ 24576 — there is no measured regime where the XLA path wins, so
any pixel-count threshold would be an unmeasured complication.  The "likely
formula" from the task framing (min of an L2 cap and a measured knee) is moot:
neither the L2 cap nor a knee exists in the data (Read 1).

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

## Follow-up: the XLA forward cliff — RESOLVED (job 13505779, `fwd_guard_cliff.py`)

The Read-2 cliff is fully localized.  Grid: band ∈ {1024, 1152, 1280, 1536, 1792,
2048, 3072, 4096} × channel reduction ∈ {sorted segment-sum, scatter-add}, at fixed
P=8192, views=1024, channels=1024, XLA path (`MBIRJAX_DISABLE_PALLAS=1`); the
reduction is forced via `model.tiles._replace(sort_by_channel=…)` + a projector
rebuild.  Per-band-element cost vs band=1024 (a flat curve = linear scaling, a jump
= cliff):

| band | 1024 | 1152 | 1280 | 1536 | 1792 | 2048 | 3072 | 4096 |
|---|---|---|---|---|---|---|---|---|
| **sorted** | 1.00 | 1.13 | 1.01 | **18.5** | 24.2 | 22.1 | 42.8 | 59.5 |
| **scatter-add** | 1.00 | 1.01 | 1.01 | 1.01 | 1.01 | 1.01 | 1.01 | 1.00 |

sorted-vs-scatter values agree at 2–3e-7 everywhere (apples-to-apples).  Three
findings:

1. **Cause: the sorted segment-sum reduce alone** (`_channel_reduce_sort_segsum` in
   projectors.py).  Scatter-add (`_channel_reduce_scatter_add`) is perfectly linear
   in band from 1024 to 4096 — no cliff.  So the base forward kernel and everything
   upstream are fine; the cliff is entirely in `lax.sort_key_val` + `segment_sum`
   over wide (`psf_width·P`, band) rows.
2. **Threshold: onset between band 1280 and 1536, NOT 2048.**  1280 is still linear
   (1.01×); 1536 is already 18.5×.  So it is a mid-range limit (~1300–1500 columns),
   not a power-of-two tiling boundary — most consistent with a register/vectorization
   width ceiling in the segmented-reduce lowering (1536 f32 = 6 KiB per reduced row).
3. **Not materialization.**  Peak memory is identical between the two reductions
   (16.17 vs 16.13 GB at band=2048) and grows smoothly (linear in band, = the
   sinogram) for both — no jump at the threshold.  The cliff is a compute-pattern
   (tiling/vectorization) collapse, not a temp blow-up.  No HLO dump is needed;
   this + the flat scatter-add curve already pin it.

**The crossover (fix-design curve):** sorted is ~2.7–2.9× FASTER than scatter for
band ≤ 1280 (which is why it is the GPU default), and 6–20× SLOWER for band ≥ 1536.
Clean crossover in (1280, 1536].

### Fix APPLIED (Greg approved 2026-07-13; commit 726b607)

**Cap the sorted reduce by column width.**  `SORTED_CHANNEL_REDUCE_MAX_COLS = 1280`
(the last measured-safe point, leaving the unmeasured 1281–1535 gap on the safe side)
sits next to the existing `_MIN_COLS` — together they define the COLUMN-count WINDOW
in which the sorted form wins (wide enough to amortize the sort, narrow enough to
avoid the segment-sum collapse).  The gate is at the ONE reduction site
`channel_scatter_reduce`:

```python
if use_sorted and values.shape[1] <= SORTED_CHANNEL_REDUCE_MAX_COLS:
    return _channel_reduce_sort_segsum(...)
return _channel_reduce_scatter_add(...)
```

`values.shape[1]` (= num_cols = the reduce's row width) is static at trace time, so
this is a compile-time branch, value-equal (both reduces agree at 2e-7).  Why the
reduce site and not `_select_tile_policy`: the policy bakes ONE `sort_by_channel`
flag, but the single-device XLA forward feeds it the FULL num_slices while the
multi-device banded path feeds ≤256 — only a per-call `values.shape[1]` check sees
the width each caller actually passes, so it protects every caller and every
geometry automatically (cone/multiaxis/translation forward share this reduce).  It
touches only `projectors.py` (my territory, not `_pallas_kernels.py`), plus a test.

**Exposure this closes:** single-device on a non-allowlisted GPU or under the kill
switch (pallas off), parallel-beam recon with ≥~1536 slices — the large-recon
fallback regime; e.g. at band=2048 the forward reduce drops 5053 → 662 ms (7.6×).
Everything on the shipped hot paths is unaffected: single-device H100 forward is
pallas; the multi-device banded forward feeds 256 cols (< the cap), so it keeps
sorted unchanged.

**Validation:** new `tests/test_channel_reduce.py::test_wide_columns_fall_back_to_scatter`
spies on the routing — at exactly 1280 cols the sorted path still runs (inclusive
cap), at 1281 it routes to scatter — plus a value-correctness check either way.  An
end-to-end CPU check (`sort_by_channel` forced on, 1300-slice recon) gave the forward
bitwise-identical to `sort_by_channel` off (rel 0), confirming the cap engages through
the real kernel and `values.shape[1]` is num_cols in the live call path.  Full CPU
suite: 303 passed, 2 skipped, 72 subtests.

### Remaining follow-up

Whether the multi-device banded forward (wave 2, the other session) inherits the
no-guard conclusion — its band is `fwd_slice_band` = 256, BELOW sweep 3's corner
cells; sweep 3's band-256 column is the relevant evidence.

## Library-change requests for the kernel session (none applied here)

None: the sweeps needed no `_pallas_kernels.py` changes (the driver's shape-static
design held from P=2048 to full grid; grid segment counts at full-grid P≈823k are
~38k < the 65535 CUDA grid-dim bound).  The proposed cliff fix is also outside
`_pallas_kernels.py` (it lives in `projectors.channel_scatter_reduce`).
