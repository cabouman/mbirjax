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

## Targets (scope)

**CPU is a first-class target, not just dev/CI.**  Users run on CPU, and CPU
sharding is a *real* performance path — the view-sharded fbp_filter measured
~5.4–6.5× at 8 virtual CPU devices (the per-device threading spreads the
embarrassingly-parallel view-shards across cores; the GIL is released during XLA
execution).  So **every phase (back projection, VCD, …) should be validated and
tuned on CPU as well as GPU**, not GPU-only.  Caveats: CPU memory is
whole-process RSS (not the per-device floor) and grows with device count, so CPU
sharding trades some memory for speed.  Multi-**node** scaling is out of scope —
all sharding here is single-process (one machine's devices); spanning nodes would
need multi-host JAX (`jax.distributed`).

## Verified hardware facts (2026-05-29, JAX 0.10.1)

- **H100 2×:** direct device-to-device `device_put` correct → no host bounce.
- **L40S 2×:** device-resident cross-device transfer **silently zeros** the
  non-default shard → host-bounce required.  Hence the empirical `_d2d_safe`
  probe in the transfer helper.
- **CPU (virtual devices):** view-sharded fbp_filter scales ~5.4–6.5× at 8
  devices — CPU sharding is a genuine speedup, not just a test path (see Targets).

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

## Jax lax.map bug note

**flat kernel + jax bug #27591 (https://github.com/jax-ml/jax/issues/27591):**
`lax.map` can return WRONG results for large `batch_size` (their repro: 128 OK,
512 garbage).  The flat kernel maps over views*rows rows, so a naive batch of
`view_batch_size*n_rows` could be tens of thousands → unsafe.  The flat kernel
now caps its map batch at `_FLAT_MAP_BATCH = 128`.  The per_view kernel batches
≤ view_batch_size (≤128) *views*, already safe.  **Open design question for the
parallel convolves (see handoff):** flattening gives freedom to choose ANY 2-D
(rows, channels) decomposition for the vmap+lax.map; pick row-batch ≈128 to stay
clear of #27591 while keeping good parallel width.


## Blocked / open

- O1 heterogeneous CPU-recon/GPU-sino placement — first touchpoint Phase D,
  deferrable (first direct_recon cut is homogeneous).
- O2 trivial-vs-gathered return contract — decide at Phase A.3 (leaning: user
  API gathers, internal stays trivially sharded).
- O3 sparse-view regime — noted, no action.
- O4 divisibility (both views and slices must divide n_devices) — assert at
  Phase A.3; optional pad/crop usability layer later.
