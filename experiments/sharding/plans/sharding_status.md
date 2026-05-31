# Sharding status (beta branch `greg/parallel_sharding`)

*Short living status. Detailed checklist: `sharding_implementation_plan.md`.*

---

## HANDOFF (2026-05-31) — Phase F1 COMPLETE

**The sharded FBP filter is done, promoted, and validated on the H100.**  The
per-device kernel is now `tomography_utils.apply_row_filter(block, filter_arr)`
— a geometry-neutral `lax.scan` over overlapping B-row windows
(`ROW_FILTER_BATCH = 1024`) that any geometry can call.  Under view-sharding the
ramp filter is per-row, so each device filters its own view-shard locally with
zero cross-device communication.

**Measured on H100 (honest harness):**
- Memory at the **2× (input+output) floor**, scaling ~1/n across devices, and
  divisibility-agnostic (the overlapping-window scan needs no padding/concat).
- **~10× faster** than a small batch; **B=1024** is the throughput knee (benefit
  past it shrinks as channels grow while the work area grows — hence the c-aware
  note on `ROW_FILTER_BATCH`).
- **1624³ runs on a single H100**; cross-platform correctness (GPU vs CPU
  prerelease baseline) = 5.6e-9.

**How we got here (highlights):** jit fix (kernels had lost `@jax.jit`) → the
memory blowup root cause was per_view's FFT work area = `view_batch_size *
n_rows` (geometry-bound; OOM'd at 1624³) → replaced with the row-batched scan and
folded `pi/num_views` into the f32 filter (killed an f64 2× OOM) → found the
harness over-reported memory by one shard (it held the prior result in the timing
loop) → GPU B-sweep picked B=1024.

**Cleanup done:** removed the `_FBP_FILTER_KERNEL` switch and the per_view/flat
kernels; moved the single kernel to `tomography_utils`; `view_batch_size` is gone
from internals and **deprecated** (DeprecationWarning) on the user-facing
`direct_recon`/`direct_filter`/`fbp_recon`/`fbp_filter`.  Tests: `test_sharding_*`
+ `test_fbp_fdk` 28/28.

**Next: Step 5 = Phase D** (sharded back projection, reduce-scatter) — see the
implementation plan.

**Deferred (noted, not blocking):**
- cone_beam / translation_model / multiaxis_parallel still use `view_batch_size`
  functionally — migrate them to `apply_row_filter` later (then drop
  `view_batch_size` from `vcls.py:167`).
- A c-aware `ROW_FILTER_BATCH ≈ budget / fft_len(c)` could reclaim small-c
  throughput; fixed 1024 is the pragmatic default for now.
- `tomography_model.py:473` "not enough GPU memory" warning (a TomographyModel
  heuristic, noisy under preallocate=true) — clean up later.

**jax bug still respected:** #27591 (lax.map wrong for large `batch_size`).  The
kernel uses vmap for the B-way parallelism and scan with no `batch_size`, so it
is immune.

---

## Where we are

- **2026-05-29:** Beta worktree created from `prerelease`.  Has **no** sharding
  infrastructure yet (clean slate — confirmed `configure_sharding`,
  `_maybe_shard`, `_shard_*`, `_extract_halos` all absent).  Implementation plan
  drafted.
- **2026-05-29:** **Phase 0 (scaffolding migration) complete.**  `_device_setup.py`
  + `conftest.py` device-count policy (env override → macOS P-cores → Linux
  affinity → fallback, capped at 8); `_device_setup` wired as first import in
  `__init__.py`; `device_put_check.py` recreated under `parallel_performance/`;
  `test_sharding_scaffolding.py` passes 4/4 (8 virtual CPU devices on M3 Max,
  env override honored, existing suite still collects).
- **2026-05-29:** **Phase A (primitives) complete.**  New subpackage
  `mbirjax/_sharding/` (`transfer.py`: `is_dev2dev_safe`, `move_shard`;
  `thread_execution.py`: `run_per_device`, `assemble_sharded`) +
  `TomographyModel.configure_sharding(devices=None)` (1-D mesh axis `'devices'`,
  empirical `dev2dev_safe` probe, O4 divisibility assert on views+slices).
  Fully additive — `main_device`/`sinogram_device` untouched.
  `test_sharding_primitives.py` 10/10.
- **2026-05-30:** **Phase B (sharding hooks) complete.**  Base-class hooks on
  `TomographyModel`: axis declarations `sinogram_shard_axis()`→0,
  `recon_shard_axis()`→-1 (last axis = slices, works for 3-D and flat recon);
  `_shard_on_axis`/`_gather_to_host` mechanism (uses the `dev2dev_safe` probe so
  the host-bounce is only paid where needed); `_shard_sinogram`/`_gather_sinogram`/
  `_shard_recon`/`_gather_recon`; `_extract_halos` (rank-agnostic). Divisibility
  asserts now read the axis hooks. Parallel beam inherits with no overrides;
  fully additive. `test_sharding_hooks.py` 8/8; full sharding suite 22/22;
  projectors regression unchanged.  **Next: Phase F1 (FBP filter — view-sharded,
  zero cross-device comms).**

## Worktrees

| Path | Branch | Role |
|------|--------|------|
| `…/Research/mbirjax/` | `greg/parallel_tests` | Research / prior art. Tag `research-snapshot-2026-05-29`. To be deleted eventually — migrate first. |
| `…/Research/mbirjax_sharding/` | `greg/parallel_sharding` | Beta (this tree). New work. |

## Design in one line

Sinogram sharded by **view** (stationary); recon by **slice** (moves as voxel
batches: all-gather forward / reduce-scatter back, pipelined).  Threading + one
probed transfer primitive.  Trivial-sharding unifies single/multi-device.

## Verified hardware facts (2026-05-29, JAX 0.10.1)

- **H100 2×:** direct device-to-device `device_put` correct → no host bounce.
- **L40S 2×:** device-resident cross-device transfer **silently zeros** the
  non-default shard → host-bounce required.  Hence the empirical `_d2d_safe`
  probe in the transfer helper.

## Phase tracker (execution order)

direct_recon is pulled early (depends on filter + back, not forward) to reach a
usable, stress-testable FBP pipeline before forward projection / VCD.

- [x] Step 1 — Phase 0: scaffolding migration
- [x] Step 2 — Phase A: primitives (transfer, threading, mesh/trivial-sharding)
- [x] Step 3 — Phase B: sharding hooks (new view/slice axes)
- [x] Step 4 — Phase F1: FBP filter (view-sharded, zero comms) — DONE; kernel is
      `tomography_utils.apply_row_filter`, 2× memory floor, B=1024, H100-validated
- [ ] Step 5 — Phase D: back projection (reduce-scatter)
- [ ] Step 6 — Phase F2: direct_recon (first usable pipeline; stress test)
- [ ] Step 7 — Phase C: forward projection (all-gather) + adjoint test
- [ ] Step 8 — Phase E: VCD integration + halos
- [ ] (later) cone beam

## Scaling notes (IN PROGRESS — 2026-05-30, M3 Max virtual CPU)

All numbers below are CPU/virtual-device on one M3 Max and are PRELIMINARY.
Authoritative perf data will come from the H100 cluster.  Treat single sweeps
with caution: run-to-run variability here has been large.

**Reference points**
- User's own `sharding_scaling` run, research branch, size 180×128×256:
  fbp_filter 1.0 / 1.87 / 3.37 / 5.81× at 1/2/4/8 devices.

**ROOT CAUSE FOUND (2026-05-30): the beta fbp_filter kernel lost `@jax.jit`.**
Research's per-device kernel `_apply_fbp_filter_to_shard` is decorated
`@partial(jax.jit, static_argnames=(...))`; the beta kernels
`_apply_fbp_filter_per_view` / `_apply_fbp_filter_flat` I wrote during the port
have **no jit** → they run eager (each lax.map/fftconvolve/reshape dispatched
op-by-op through Python), which on the CPU backend is far slower and barely
scales.

How we got here (each step ruled something out, cleanly):
- 3-way A/B (research vs beta per_view vs beta flat), same harness/sizes/seed:
  research scaled ~6.5×, both beta kernels ~2× — so NOT the kernel structure.
- view_batch_size sweep on beta: flat across vbs — NOT the lax.map batching.
- block-in-thread probe: ≤3% — NOT where we block (Greg predicted this).
- shard-axis microbench (pure JAX, no mbirjax, view vs row axis, same per-row
  conv): view and row IDENTICAL to ~2% — **shard axis is NOT the cause.**  But
  this isolated test only scaled ~2× too (it had no jit either), which pointed
  the finger at the one thing it shared with beta and not research: no jit.
- jit toggle in that same pure-JAX probe (eager vs jitted kernel, both axes):

    256×256×256  1→8 dev:  eager ~2.5× (200 ms)   jitted ~6.8× (66 ms)
    512×128×128  1→8 dev:  eager ~1.3× (187 ms)   jitted ~5.5–6.7× (31 ms)

  Jitted matches research's numbers (research was 69 ms / 32 ms at 8 dev); eager
  matches beta's.  Shard axis irrelevant in both eager and jitted columns.

**Conclusion:** add `@jax.jit` (static `view_batch_size`) to the beta kernel(s)
and the gap closes — independent of per_view-vs-flat and independent of shard
axis.  This is a one-decorator fix, confirmed in isolation before touching the
library.

**FIX APPLIED + CONFIRMED IN THE LIBRARY (2026-05-30).**  Added
`@partial(jax.jit, static_argnames='view_batch_size')` to both
`_apply_fbp_filter_per_view` and `_apply_fbp_filter_flat`.  Re-ran the real
3-way (`fbp_filter_research_vs_beta.py`).  **MEASURED min ms** (read from the
harness's merged combined table — these are the real numbers):

  size ndev      research  per_view     flat
  256³   1        462.57    461.19   445.60
  256³   2        208.01    235.98   229.22
  256³   4        124.30    123.17   120.16
  256³   8         70.01     67.46    62.53
  512×128 1       204.58    200.30   177.04
  512×128 2       107.02    105.54   108.44
  512×128 4        56.38     56.28    56.34
  512×128 8        31.98     31.23    30.03

- **The jit fix works.**  per_view at 256³/8-dev went from ~190 ms (eager, before
  the fix) to 67 ms — now matches research (70) and scales ~6.8×.  This is the
  whole win; the eager→jit change was the entire research-vs-beta gap.
- **All three are within a few % at every config now.**  flat is marginally
  fastest at most points (e.g. 256³/8: flat 62.5 vs per_view 67.5 vs research
  70; 512×128/1: flat 177 vs ~200); per_view ≈ research.  No regressions.
- **flat lean is supported but NOT yet locked.**  flat is fastest-or-tied here
  AND cleaner/axis-agnostic, but (a) its lax.map batch is capped at 128 rows for
  #27591 — the batch-sizing for arbitrary (rows, channels) still wants the
  deliberate design noted in the handoff, and (b) GPU is the deciding hardware.
  Default stays "per_view" (proven, == research) until GPU confirms flat.
- Tests: `test_sharding_fbp.py` + `test_fbp_fdk.py` pass 6/6 with the jit.

**flat kernel + jax bug #27591 (https://github.com/jax-ml/jax/issues/27591):**
`lax.map` can return WRONG results for large `batch_size` (their repro: 128 OK,
512 garbage).  The flat kernel maps over views*rows rows, so a naive batch of
`view_batch_size*n_rows` could be tens of thousands → unsafe.  The flat kernel
now caps its map batch at `_FLAT_MAP_BATCH = 128`.  The per_view kernel batches
≤ view_batch_size (≤128) *views*, already safe.  **Open design question for the
parallel convolves (see handoff):** flattening gives freedom to choose ANY 2-D
(rows, channels) decomposition for the vmap+lax.map; pick row-batch ≈128 to stay
clear of #27591 while keeping good parallel width.

**Implications now settled:**
- View-axis sharding is NOT inherently slow — the earlier "view-sharding has a
  CPU scaling cost" worry is withdrawn; it was the missing jit.
- per_view vs flat remains genuinely close (~7%); decide it later (GPU + memory),
  but it is a minor optimization, not the scaling lever.

**Next steps:**
1. Add `@partial(jax.jit, static_argnames='view_batch_size')` to the beta
   kernels; re-run `fbp_filter_research_vs_beta.py` to confirm beta now matches
   research; ensure `test_sharding_fbp.py` / `test_fbp_fdk.py` still pass.
2. Then take the (now-jitted) comparison to the H100 cluster.
3. per_view-vs-flat decision after that.

Library default remains `_FBP_FILTER_KERNEL = "per_view"`.

## Blocked / open

- O1 heterogeneous CPU-recon/GPU-sino placement — first touchpoint Phase D,
  deferrable (first direct_recon cut is homogeneous).
- O2 trivial-vs-gathered return contract — decide at Phase A.3 (leaning: user
  API gathers, internal stays trivially sharded).
- O3 sparse-view regime — noted, no action.
- O4 divisibility (both views and slices must divide n_devices) — assert at
  Phase A.3; optional pad/crop usability layer later.
