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
  env override honored, existing suite still collects).  **Next: Phase A
  (primitives — transfer helper + `_d2d_safe` probe, threading helper,
  `configure_sharding` + trivial-sharding).**

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
- [ ] Step 2 — Phase A: primitives (transfer, threading, mesh/trivial-sharding)
- [ ] Step 3 — Phase B: sharding hooks (new view/slice axes)
- [ ] Step 4 — Phase F1: FBP filter (view-sharded, zero comms — low-hanging)
- [ ] Step 5 — Phase D: back projection (reduce-scatter)
- [ ] Step 6 — Phase F2: direct_recon (first usable pipeline; stress test)
- [ ] Step 7 — Phase C: forward projection (all-gather) + adjoint test
- [ ] Step 8 — Phase E: VCD integration + halos
- [ ] (later) cone beam

## Blocked / open

- O1 heterogeneous CPU-recon/GPU-sino placement — first touchpoint Phase D,
  deferrable (first direct_recon cut is homogeneous).
- O2 trivial-vs-gathered return contract — decide at Phase A.3 (leaning: user
  API gathers, internal stays trivially sharded).
- O3 sparse-view regime — noted, no action.
- O4 divisibility (both views and slices must divide n_devices) — assert at
  Phase A.3; optional pad/crop usability layer later.
