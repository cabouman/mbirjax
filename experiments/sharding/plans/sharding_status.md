# Sharding status (beta branch `greg/parallel_sharding`)

*Short living status. Detailed checklist: `sharding_implementation_plan.md`.*

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
- [ ] Step 4 — Phase F1: FBP filter (view-sharded, zero comms — low-hanging)
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

**Measurements taken so far (each a single run unless noted) — NOT yet
reconciled:**
- A same-size A/B (`/tmp/ab_fbp.py`, separate `PYTHONPATH` processes, 256×128×256,
  warmup 2 / 6 trials) showed: research 1.0/1.9/3.7/6.4× (good); beta library
  fbp_filter 1.0/2.0/2.0/1.7× (plateau).
- A later controlled comparison of locally-defined kernels
  (`fbp_filter_flatten_experiment.py`, same size/method) showed BOTH a per-view
  kernel (1.0/1.86/3.52/6.45×) and a flattened kernel (1.0/1.87/3.57/6.72×)
  scaling well — i.e. did NOT reproduce the beta plateau.

**Honest status: NOT resolved.**  The two runs above disagree about whether the
library fbp_filter plateaus.  Possible reasons (untested): run-to-run noise;
machine state during the slow run; a real difference between the library call
path and the locally-defined per-view kernel used in the second experiment.  We
have not isolated which.  **No conclusion should be drawn yet.**

**Flattening is still an open, promising candidate** (per Greg — to be explored
further, not dismissed).  Why it could help: the FBP filter is genuinely
per-detector-row, so flattening (views, rows, channels)→(views*rows, channels)
exposes views*rows independent rows regardless of how few views a device holds.
That should matter most at high device counts / few-view scans, and on GPU where
granularity drives occupancy.  In the one CPU run measured it was marginally
ahead at every device count; whether that is real needs repeats + GPU + varied
sizes.

**Code currently in `parallel_beam.py` (on disk, 6/6 fbp tests pass):** library
kernel is **per-view** (`_apply_fbp_filter_to_views`: lax.map over views, vmap
over rows); default `view_batch_size` None (adaptive `min(vmap_batch, 128)`);
body/tail split computed per-shard (`shard.shape[0]`).  No flatten applied to
the library yet.  The flattened variant exists only in the experiment script.

**Next investigation steps (deliberate):** (1) re-run the A/B several times to
quantify variance; (2) compare the *library* fbp_filter call directly against
the flattened kernel (not just a local per-view reimpl) so the comparison is
apples-to-apples; (3) vary device count / size, especially small per-device
view counts; (4) repeat on GPU.  Decide on flatten only after that.

## Blocked / open

- O1 heterogeneous CPU-recon/GPU-sino placement — first touchpoint Phase D,
  deferrable (first direct_recon cut is homogeneous).
- O2 trivial-vs-gathered return contract — decide at Phase A.3 (leaning: user
  API gathers, internal stays trivially sharded).
- O3 sparse-view regime — noted, no action.
- O4 divisibility (both views and slices must divide n_devices) — assert at
  Phase A.3; optional pad/crop usability layer later.
