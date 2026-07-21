# A100 tuning campaign — measured record (2026-07-20)

Companion to `gpu_headroom_findings.md` (the H100 campaign that produced the Pallas
kernels).  That campaign measured every tuning constant on H100 (sm90); the A100
(sm80) entry was later added to `_ARCH_ALLOWLIST` on the strength of a single
uncontrolled A/B (111 s → 74 s on a BGA cone recon), with **no sm80 correctness
evidence and no sm80 tuning data**.  This document is that evidence.

**Headline: every shipped Pallas constant survives the move from H100 to A100
unchanged — and the same sweep re-run on H100 confirms them there too** (§8).  Nothing
beat the incumbents by the pre-registered margin on either architecture, and the sweep
demonstrably *could* have detected a change: it resolved effects of +7.9%, +9.1%,
+12.5%, +14.7% and +81% against noise floors of 0.0–2.3%.

## Provenance — exactly what was measured, where

Recorded because the branches moved during the campaign and "the branch" is not a
sufficient reference.  **The measured code is NOT the commit this document sits on.**

| what | where | commit measured |
|---|---|---|
| all A100 runs (§1–§7) | gilbreth `~/PycharmProjects/mbirjax` | **`9b997e2`** "Update to cuda13." |
| all H100 runs (§8) | gautschi `~/PycharmProjects/mbirjax` | **`ea9e2ef`** "Add H200." |
| this document | `greg/gpu_headroom` | later (`3f36112`+) |

`ea9e2ef` is two commits ahead of `9b997e2` (an H200 allowlist entry and a gautschi
install-script tweak); **neither touches `_pallas_kernels.py`**, so the kernels under
test were byte-identical on both clusters.  Both clusters ran jax 0.10.1 / Python
3.11.15, so there is no toolchain confound with the original H100 record either.

Clusters: gilbreth `bouman` / `a100-40gb` (SXM4 pinned, see trap 3); gautschi `bouman` /
`ai` (H100 80GB HBM3).  Scripts: `plans/experiments/projector_kernels/a100/` (A100
harnesses) and `.../h100/` (`h100_boundary_*.slurm`, plus the original 2026-07 campaign
scripts).  Raw results: `/scratch/gilbreth/buzzard/a100_tuning/results/` and
`/scratch/gautschi/buzzard/h100_tuning/results/` (purge-eligible scratch).

---

## 1. Correctness on sm80 (job 11343163, `a100_gate.slurm`)

`availability()` → `True | available (NVIDIA A100-SXM4-40GB)`: the probe kernel
compiles on sm80, so the gates ran **compiled**, not in interpret mode.

* `tests/test_pallas_kernels.py`: **35 passed, 0 failed** (45 s) — parallel and cone,
  back and forward, n=1 and per-owner band drivers, both `coeff_power`s, all four
  adjoint identities, the `bp_psf_radius` fallback guard.
* `tests/geometries/test_projectors.py` + `test_banded_projectors.py` +
  `test_tile_policy.py`: 33 passed, 3 skipped, 14 subtests passed.
* Production-scale value gate at sinogram (752, 720, 688):
  `back_grad 3.06e-06` (tol 1e-5), `back_hess 7.08e-06` (1e-4),
  `fwd_nrmse 1.22e-05` (2e-5), `fwd_maxrel 4.33e-05` (3e-4) — **PASS**.
  For comparison E5 measured `6.5e-6` gradient on H100: **sm80 accuracy is not worse.**

**A trap worth recording.** `test_pallas_kernels.py` chooses its mode with
`_interpret() = not is_available()`.  On a node where availability is False the whole
suite silently runs the interpret-mode reference implementation and passes, proving
nothing about the compiled kernels.  A green suite is only meaningful together with
`availability() == True`; `a100_gate.slurm` asserts this and aborts otherwise.

## 2. The BGA baseline A/B, controlled (job 11342941, `a100_baseline_ab.py`)

Greg's original observation, re-measured with controls: same script settings
(`dataset_index=4`, downsample 3, view-subsample 5, sharpness 1.5, snr_db 35), one
isolated subprocess per cell, interleaved variants, a discarded warmup, and
`stop_threshold_change_pct=0` so both variants run **exactly** 15 iterations.

Sinogram (481, 322, 510) → recon (510, 510, 456) — note 456 = 322 + 67 + 67, the
flash-remediation axial padding growing the slice axis 42%.

| variant | time (min) | time (med) | peak | flags |
|---|---|---|---|---|
| pallas | 69.98 s | 71.66 s | **4.698 GB** | fwd+back True |
| xla | 101.56 s | 102.07 s | 5.015 GB | both False |

**pallas is −29.8% time AND −6.3% memory — it Pareto-DOMINATES.**  Both arms ran 15
iterations and converged to the same solution (final `fm_rmse` 12.7610 vs 12.7577,
agreeing to 2.6e-4).  Run-to-run time spread: pallas 2.6%, xla 0.9%.

**A prediction of mine that was wrong, recorded because it was stated confidently.**
I argued the pallas back path *should* cost extra memory — it allocates a channel-major
sinogram copy (rows padded to a power of two), a `(V, T, P)` weights array per view
chunk, and holds two output buffers while accumulating — and warned that if the cost
exceeded ~11% the allowlist commit would be a bad trade at Greg's stated exchange rate.
It is measurably **leaner**, not fatter: those allocations are smaller than the XLA
path's own transients (the sorted scatter/segment-sum machinery it replaces).

## 3. Why kernel tuning is worth anything: projector share vs recon size

Derived from the existing nightly GPU records (`mbirjax_metrics/results/gpu/
greg_gpu_headroom/records_gpu.yaml`), no new GPU time.  Estimated as
`iterations x (forward + back) / vcd_time` — each VCD iteration sweeps every pixel once,
so one iteration ≈ one full-grid-equivalent forward + back.  Approximate (ignores
per-subset call overhead and the direct_recon init) but directionally solid:

| geometry (n=1) | 200x208x160 | 512x448x384 | 1024x1008x992 |
|---|---|---|---|
| cone | 10% | 51% | **62%** |
| parallel | 3% | 26% | **47%** |

**Kernel tuning is nearly pointless at the 200³ class and dominant at the 1024 class.**
This is the number that should decide where any future kernel campaign measures.
(These are H100 nightlies; the trend should carry, the absolute values needn't.)

**The 40 GB cliff.**  From the same records: cone 1024x1008x992 VCD needs **47.6 GB**
and parallel **42.6 GB** at n=1 — so **neither fits a single A100-40GB**; both need
n≥2 (cone 22.7 GB at n=2).  Isolated projector calls at that size *do* fit
(fwd 15.9 GB, back 17.1 GB), which is why the sweep below could run at n=1.

## 4. The cone constant sweep (job 11343171, `a100_cone_sweep.py`)

Cell: sinogram (752, 720, 688) → recon (688, 688, 810), chosen to fit one A100-40GB
with headroom (full VCD ≈ 14.3 GB measured) and with all axis lengths different.

Only four module constants can move a cone workload: `CONE_LC`, `CONE_NUM_WARPS`,
`CONE_FWD_NUM_WARPS`, `FWD_SEGMENT_CAP`.  `ROW_CHUNK` / `NUM_WARPS` / `FWD_NUM_WARPS`
are read only in the parallel builders and are **inert** here;
`BACK_VIEW_CHUNK_CAP` is a **clamp, not a knob**, at n=1 (see §6).

| knob | A100 result | H100 record |
|---|---|---|
| `CONE_LC` | **128 best**; 32 → +19%, 256 → +6.7%, 512 → +14% | 128 (sweep stopped there) |
| `CONE_NUM_WARPS` | **1 best**; 2 → +7.9%, 4 → +14.7% | 1 |
| `CONE_FWD_NUM_WARPS` | **2 best**; 1 → +12.5%, 4 → +0.6% (tied) | 2 (+23% for warps=1) |
| `FWD_SEGMENT_CAP` | **no measurable effect** 32→256 (0.4% span = the noise floor) | "caps ≥64 flat" |

Noise floors from repeated shipped-constant anchor cells: 0.4–0.7%.

**A strongly non-additive interaction.**  `CONE_LC=64` alone costs +1.6% and
`CONE_NUM_WARPS=4` alone +14.7%, but **together +81%**.  Consistent with register
pressure (more warps competing for the register file while a smaller slice chunk gives
less work to amortize it).  A one-factor-at-a-time sweep cannot see this — the 2-D
stage was worth its cost.

**The predicted A100 L2 effect appeared and confirmed the incumbent.**  The cone-back
sinogram block is shared through L2 with a per-slice-chunk working set of roughly
`view_chunk x channels x (lc + 2*bp) x 4 B` ≈ **34.5 MB** at shipped settings — inside
H100's 50 MB but 86% of A100's 40 MB, and ~68 MB at `lc=256`.  The measured +6.7%
penalty at `lc=256` is what that model predicts, and it pushes *toward* the shipped 128.

## 5. Boundary cases: all four closed

The H100 campaign left four constants sitting at the edge of their swept range — the
pattern that usually means a missed optimum.  All are now resolved:

| constant | untested on H100 | A100 result | verdict |
|---|---|---|---|
| `CONE_LC=128` | 256, 512 | +6.7%, +14% worse | curve turns over just past the boundary |
| `CONE_FWD_NUM_WARPS=2` | 4 | +0.6% (tied) | nothing beyond 2 |
| `FWD_SEGMENT_CAP=64` | 256 | flat | not a lever at all |
| `ROW_CHUNK=256` | 512 | **operating-point dependent** — see §7 | 256 is the robust compromise |

All four are now **also confirmed on H100** (§8), so the boundary question is closed on
both architectures.

## 6. `back_view_batch` — an A100-only memory lever, CLOSED (do not adopt)

`BACK_VIEW_CHUNK_CAP` cannot act at n=1: the driver takes
`min(tiles.back_view_batch, BACK_VIEW_CHUNK_CAP, num_views)` and the base policy already
pins `back_view_batch = _BACK_VIEW_CAP_SINGLE = 128`, so every cell with CAP ≥ 128 is
bit-identical to shipped.  It also applies to the **cone forward** driver
(`_pallas_kernels.py` line ~874), not just back.  The *effective* view chunk is the
knob, and it was swept via `tiles._replace(back_view_batch=...)`.

End-to-end full-VCD A/Bs at the §4 cell, 8 iterations, `stop_threshold_change_pct=0`
(jobs 11343553 cone, 11343823 parallel):

| geometry | vc=64 | vc=32 |
|---|---|---|
| cone | +0.7% time, **−6.3% peak** (trade, ratio 8.8x) | +1.6% time, −7.0% peak (4.3x) |
| parallel | −1.6% time, −3.2% peak (**dominates**) | −1.5% time, **−10.8% peak** (dominates) |

**Attenuation is the honest headline.**  The kernel-level parallel win was −19.7%; it
composes to **−1.6%** of recon wall — a 12x dilution, because back projection is only
~13–15% of recon time (from the nightly: parallel 1024³ fwd 2352 ms vs back 1153 ms,
projectors ~47% of VCD).  Kernel and recon numbers agree arithmetically, which is
reassuring about both.

So the accurate claim **on A100** is: `back_view_batch=32` buys ~11% recon peak memory
for free on parallel (the −1.5% time is within run-to-run variation).  Not a speed lever.

### CLOSED — the lead is A100-specific and would REGRESS H100

The H100 re-run (§8) settles this and reverses the recommendation.  On H100 the same
configurations are **slower at the operating point that dominates a recon**:

| parallel back, vs shipped | 2,896 px (12 of 15 iterations) | 92,649 px |
|---|---|---|
| `VIEW_CHUNK=64` — A100 | −4.8% | −3.9% |
| `VIEW_CHUNK=64` — **H100** | **+3.2%** | **+4.3%** |
| `VIEW_CHUNK=32` — A100 | −1.6% | −11.1% |
| `VIEW_CHUNK=32` — **H100** | **+28.0%** | **+10.5%** |

Adopting the A100 finding as a default would have cost **+28% on the primary platform**
at the count 12 of 15 iterations run.  So this is not "a memory win pending validation";
it is an **architecture-specific effect that would need per-arch constants to exploit
safely**, for ~11% recon peak memory and no time.  Given that the shipped
`_BACK_VIEW_CAP_SINGLE = 128` also governs the XLA path (where it was chosen on
vmap-transient grounds — *"wider vmaps raise peak memory ~25%"*), **the recommendation is
to leave it alone.**

This is the campaign's clearest cautionary result: a dominant-looking single-architecture
win that reverses sign on the other architecture.  Any future single-arch tuning finding
should be cross-checked before it is believed.

## 7. Pixel count is the real axis (job 11343876, `a100_pixel_count_sweep.py`)

Greg's correction, and it materially changed the conclusions.  "Subset" and "full grid"
are not intrinsic categories — a wider recon's 128-subset piece can exceed a narrow
recon's whole grid.  The variable is the **absolute pixel count**, and it matters for a
structural reason: **the Pallas back path does no pixel batching.**  The whole index set
becomes one kernel grid dimension —

```
grid=(rows_padded // rc, num_pixels)   # parallel  (_make_back_call)
grid=(l_padded   // lc, num_pixels)    # cone      (_make_cone_back_call)
```

— and `num_pixels` is in each builder's `functools.cache` key, so **every distinct count
compiles its own kernel**.  (`fwd_pixel_batch` / `back_pixel_batch` are XLA-path knobs;
the Pallas drivers ignore them.)  Small counts are occupancy-starved on 108 SMs; large
ones saturate.

The counts a real 15-iteration recon visits at recon (688, 688, 810):

| pixels | source | occurrences |
|---|---|---|
| 2,895 | 128 subsets | **12 of 15 iterations** |
| 5,791 | 64 subsets | 1 |
| 23,163 | 16 subsets | 1 |
| 92,649 | 4 subsets | 1 |
| 473,344 | Hessian — `arange(rows*cols)`, the UNMASKED rectangle, `coeff_power=2` | 1 |

Parallel back, relative to shipped (negative = faster):

| config | 2,896 | 5,791 | 23,163 | 92,649 | 473,344 |
|---|---|---|---|---|---|
| `VIEW_CHUNK=64` | −4.8% | −15.7% | −13.5% | −3.9% | −18.5% |
| `VIEW_CHUNK=32` | −1.6% | −19.7% | **−28.9%** | −11.1% | −14.0% |
| `ROW_CHUNK=512` | −4.8% | −9.8% | −4.8% | **+9.0%** | **+12.9%** |

Three corrections to the two-point sampling of §4/§6:

1. **The `ROW_CHUNK=512` crossover is between 23,163 and 92,649 pixels** — inside the
   gap the two-point sampling skipped.  Iteration 0 (92,649 px) is on the bad side,
   iterations 1–14 on the good side; the shipped 256 is the right compromise.
2. **The largest effects are in the MIDDLE, not at the ends** — `VIEW_CHUNK=32` peaks
   at −28.9% at 23,163 px, a point never measured before.
3. `VIEW_CHUNK=64/32` is faster at **every** real operating point *on A100*, so §6's
   effect is structural rather than an endpoint artifact — but see §6: it reverses on
   H100.

**Lesson: sample the operating points a real recon visits, not two labels that bracket
them.**  Endpoint sampling both understated the effect and hid a sign change.

### Three caveats on the table above

**(a) This sweep has NO anchor cells, hence no noise floor** — unlike §4, which re-ran a
shipped-constants anchor every 6 cells.  `a100_pixel_count_sweep.py` uses only
min-of-10-trials across 2 randomized passes.  Reproducibility was therefore checked
after the fact from the saved per-pass records: **median pass-to-pass spread 0.2%, max
6.9%** (the worst cell being `ROW_CHUNK=512` at 23,163 px).  The numbers are
reproducible; the omission was still a lapse and the harness should grow anchors before
it is reused.

**(b) The Hessian column is a DIFFERENT KIND of index set and is not a continuation of
the curve.**  The four subset columns are sorted uniform random samples of the ROR
(`gen_pixel_partition` permutes the ROR indices and reshapes into equal subsets — there
is no spatial tiling; `gen_pixel_partition_grid`, which does tile, is theoretical and
not used in practice).  Their sampling *density* rises with size — mean index gap ~128
at 2,895 pixels, ~4 at 92,649.  The Hessian set is `arange(rows*cols)`: dense,
contiguous, and including pixels **outside** the ROR, so it has near-ideal gather
locality and sits in its own regime.  Read the first four columns as a series; read the
fifth separately.

**(c) The `ROW_CHUNK=512` swing is reproducible but UNEXPLAINED.**  Within the
homogeneous subset series it is a monotone improvement then one sharp crossover
(−3.3%, −17.3%, −36.2%, **+72.3%** on H100; −4.8%, −9.8%, −4.8%, **+9.0%** on A100),
reproducible to 0.2% pass-to-pass.  A ~108-point swing across a 4x size change with no
identified mechanism is extraordinary and should be treated with suspicion (Greg's
instinct, and correct).  It does not affect any conclusion — `ROW_CHUNK=512` loses at
92,649 px on both architectures, so the shipped 256 stands regardless — and locating the
mechanism would need profiling that was judged not worth the cost.

## 8. H100 cross-check (gautschi jobs 13939582 cone, 13939583 parallel)

Every boundary cell in §5 was closed on A100 but remained **untested on the architecture
that actually ships the constants**.  The same harness was re-run on H100 80GB HBM3
(`h100_boundary_*.slurm`, commit `ea9e2ef`).

**Pre-registered prediction, and it held.**  The cone-back L2 working set is roughly
`view_chunk x channels x (lc + 2*bp) x 4 B` ≈ 34.5 MB at `lc=128` and ~68 MB at
`lc=256`; 68 MB exceeds H100's 50 MB L2 just as it exceeds A100's 40 MB, so `lc=256`
should lose on H100 too.  Measured (noise floor 0.6%):

| `CONE_LC` | H100 | A100 |
|---|---|---|
| 32 | +21.7% | +19% |
| 64 | −0.7% (inside noise) | +1.6% |
| **128 (shipped)** | — | — |
| 256 | **+9.1%** | +6.7% |
| 512 | +14.7% | +14% |

Every cone stage returned *"NOT a real improvement; the incumbent stands"* (noise floors
0.0–0.6%): `CONE_LC=128`, `CONE_NUM_WARPS=1`, `CONE_FWD_NUM_WARPS=2` and
`FWD_SEGMENT_CAP=64` are confirmed optimal on **both** architectures.  The value gate
passed identically.  The positive control measured pallas 0.0143 s vs XLA 0.0498 s
(**3.48x**), so the kernels were live and distinguishable.

One honest wrinkle for the L2 model: the `lc=256` penalty is *larger* on H100 (+9.1%)
than on A100 (+6.7%) despite H100 having 25% more L2.  The model predicted the sign
correctly but not the ordering of magnitudes, so treat it as a useful heuristic rather
than a quantitative account.

The parallel half of the re-run is the source of §6's reversal.

## 9. Thermal / power headroom — the kernels are not throttling in these runs

Greg's field observation: since the Pallas kernels shipped, nightly runs leave the GPUs
slightly hot and slightly throttled — evidence that at large recons the kernels are near
a hardware bound rather than a software one, which independently corroborates the null
result above.

Telemetry sampled on gautschi h002 **during** the §8 cone sweep, however, shows no
throttling at all: SM clock pinned at 1980/1980 MHz on every sample, power 124–422 W
against a **700 W** limit, 39–48 °C, `clocks_throttle_reasons.active = 0x0` throughout.

The two observations are compatible and the distinction matters: this sweep is
**bursty** — one subprocess per cell, so wall time is dominated by Python/JAX startup and
compile, with utilization at 0% on most samples and short kernel bursts between.  Low
duty cycle means no thermal accumulation.  Sustained back-to-back production recons are a
different load.

Consequences: (1) the tuning numbers here measure **un-throttled** kernel speed, which is
the correct basis for choosing constants (relative comparisons hold), but absolute
speedups under sustained load may be lower; (2) the single 99%-utilization sample at
421 W of 700 W suggests these kernels do not saturate H100 *power* even at full tilt,
consistent with the earlier `ncu` finding that the accumulate kernel is memory-access-
*pattern*-bound rather than bandwidth- or compute-bound; (3) if the practical ceiling is
thermal, then the remaining productive axes are memory, host dispatch (`current_plans`
§3: ~95% of a 200³ VCD is host time) and algorithmic work — not kernel micro-tuning.
It also qualifies the "GPU kernels ~10x above compute-only bounds" note in
`current_plans` §3: that gap is against a theoretical roofline, and the *achievable*
bound on thermally-capped parts is lower.

Not measured: telemetry during a sustained large recon, which is what would turn "close
to the hardware bound" from an inference into a number.

## 10. Measurement traps found (each would have produced a confident wrong answer)

1. **Two-level cache staleness.**  `_make_cone_fwd_chunk_fn` caches a `jax.jit` closure
   holding already-built `pallas_call` objects; clearing only `_make_cone_fwd_phase`
   returns the OLD binary.  A hand-written clear list missed it and would have made the
   `CONE_FWD_NUM_WARPS` axis — the sharpest knob in the whole record — perfectly flat.
   Fix: clear every `cache_clear` in the module by introspection (7 caches found) and
   ASSERT `cache_info().misses` increased per cell.
2. **Patching without clearing is a silent no-op.**  Verified locally: setting
   `ROW_CHUNK` without `cache_clear()` leaves the old kernel in place with no error.
3. **The mixed-hardware partition.**  `a100-40gb` silently contains BOTH A100-SXM4
   (features `N`/`nvlink`, 4 GPU/node, 400 W) and A100-PCIe (`G`, 2 GPU/node, 250 W).
   Timings are not comparable across them.  Every timing job pins `--constraint=N`.
4. **Input distribution vs calibrated tolerances.**  A gate fed `standard_normal`
   reported `back_grad_rel` 4.9e-5 against a 1e-5 tolerance — an apparent correctness
   failure on a shipped kernel.  Every calibrating experiment (`test_pallas_kernels.py`,
   E5, E6) uses `rng.random` (uniform positive); signed zero-mean data makes the view sum
   cancel toward zero while terms stay O(1), inflating every relative-to-max metric.
   With positive inputs: 3.06e-6.  **Suspect the ruler.**
5. **A positive control built on the wrong mechanism.**  `MBIRJAX_DISABLE_PALLAS` flips
   only the TilePolicy DISPATCH flags; a harness that calls the driver function directly
   is unaffected, so the control aborted every run.  The real control is pallas-driver
   vs XLA-path.  (The guard was right to fire even though the guard was wrong.)
6. **Pooling different ops into one noise floor.**  A stage mixing back and forward
   cells reported a 194.8% anchor spread; per-op it is 0.0%.  Group by op, always.
7. **`np.savez` appends `.npz`** to any path lacking it, breaking an atomic-rename
   temp-file idiom.  Caught by the cheap probe before the long sweep.

## 11. Open / next

* ~~**H100 boundary re-run**~~ — DONE (§8): prediction held, all constants confirmed on
  H100 as well.
* ~~**`back_view_batch`**~~ — CLOSED (§6): A100-specific, would regress H100 by +28% at
  the dominant operating point.  Do not adopt.
* **Cone pixel-count sweep** — cone got only the two-point treatment; §7 shows why that
  is not enough.
* **Multi-device (n≥2)** — untouched here.  `BACK_VIEW_CHUNK_CAP` binds only there, and
  it is the constant with an explicit memory rationale in its comment.
* **A100 remains untuned for `translation` and `multiaxis_parallel`.**
