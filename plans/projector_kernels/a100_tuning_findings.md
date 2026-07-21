# A100 tuning campaign — measured record (2026-07-20)

Companion to `gpu_headroom_findings.md` (the H100 campaign that produced the Pallas
kernels).  That campaign measured every tuning constant on H100 (sm90); the A100
(sm80) entry was later added to `_ARCH_ALLOWLIST` on the strength of a single
uncontrolled A/B (111 s → 74 s on a BGA cone recon), with **no sm80 correctness
evidence and no sm80 tuning data**.  This document is that evidence.

Cluster: **gilbreth** (`bouman` account, `a100-40gb` partition), jax 0.10.1 / Python
3.11.15 — identical to gautschi, so no toolchain confound with the H100 record.
Scripts: `plans/experiments/projector_kernels/a100_*.py` + `.slurm`.

**Headline: every shipped Pallas constant survives the move from H100 to A100
unchanged.**  Nothing beat the incumbents by the pre-registered margin, and the sweep
demonstrably *could* have detected a change — it resolved effects of +7.9%, +12.5%,
+14.7% and +81% against noise floors of 0.0–2.0%.

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

## 6. `back_view_batch` — a real memory lever, and the campaign's one open lead

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

So the accurate claim is: **`back_view_batch=32` buys ~11% recon peak memory for
free** on parallel (the −1.5% time is within run-to-run variation).  Not a speed lever.

**NOT recommended as a default change** without: the XLA path re-measured (the current
128 was chosen on XLA-vmap grounds — *"wider vmaps raise peak memory ~25%"* — and not
everyone runs Pallas), multiple shapes and geometries, the n≥2 path where
`_BACK_VIEW_CAP_SHARDED = 512` and the cap actually binds, and a dashboard re-baseline
(every memory gate shifts at once).

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
3. `VIEW_CHUNK=64/32` is faster at **every** real operating point, so §6's effect is
   structural rather than an endpoint artifact.

**Lesson: sample the operating points a real recon visits, not two labels that bracket
them.**  Endpoint sampling both understated the effect and hid a sign change.

## 8. Measurement traps found (each would have produced a confident wrong answer)

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

## 9. Open / next

* **H100 boundary re-run** (submitted to gautschi, `h100_boundary_*.slurm`): every cell
  in §5 is still untested on the architecture that ships the constants.  Pre-registered
  prediction: `lc=256` is worse on H100 too (~68 MB working set vs 50 MB L2).  If it
  instead wins, the L2 model is wrong and a real speedup has been sitting unclaimed on
  the primary platform.
* **`back_view_batch`** — §6's lead, gated on the work listed there.
* **Cone pixel-count sweep** — cone got only the two-point treatment; §7 shows why that
  is not enough.
* **Multi-device (n≥2)** — untouched here.  `BACK_VIEW_CHUNK_CAP` binds only there, and
  it is the constant with an explicit memory rationale in its comment.
* **A100 remains untuned for `translation` and `multiaxis_parallel`.**
