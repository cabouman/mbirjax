# Sharding implementation plan (beta branch `greg/parallel_sharding`)

*Created 2026-05-29.  Living checklist for the view-sharded reimplementation of
multi-device sharding, built fresh on `prerelease`.*

**Companion docs**
- `experiments/sharding/plans/sharding_status.md` — short living status (what
  phase we're in, what's blocked).
- `experiments/sharding/parallel_performance/fbp_parallel_options.md` —
  shard_map-vs-threading parallelism comparison (migrated from research
  `.claude/parallel_options.md`); background for Phase F1.
- Research-branch prior art: `…/Research/mbirjax/.claude/status.md` (slice-axis
  design + Step 0–6 history) and the auto-memory `sharding_progress.md`.  The
  research branch (`greg/parallel_tests`, tag `research-snapshot-2026-05-29`)
  holds the original implementation we are porting from.  **It will eventually
  be deleted — migrate anything worth keeping before then** (see §Migration and
  the standalone migration note at the end).

---

## 0. Design summary (the scheme we are building)

- **Uniform sharding across geometries:** sinogram-like objects sharded **by
  view** (axis 0); recon-like objects sharded **by slice**.  This uniform scheme
  is simple to maintain and suits cone beam.  The axis choice for each array type
  is not hardcoded throughout the code: it is declared in one place per geometry
  (see the axis-declaration hooks in Phase B) and everything axis-dependent
  (divisibility checks, PartitionSpecs, shard/gather) derives from it.  Keeping
  the choice **explicit and located** means a geometry could, in principle,
  declare a different axis later without hunting down scattered assumptions.
- **Sinogram stationary, recon moves.**  In the VCD loop the sinogram, weights,
  and error-sinogram stay view-sharded and never re-shard.  Forward projection
  **all-gathers** voxel cylinders (recon slices) so each device has the full
  cylinders for its views; back projection **reduce-scatters** the partial
  cylinders back to slice owners.  Transfers happen in pixel-batches that
  overlap the previous batch's compute.
- **Threading (Path G), not shard_map.**  Manual collectives over
  `addressable_shards`.  Compiler-inserted collectives (shard_map/GSPMD) were
  rejected on the research branch for SPMD overhead and L40S correctness.
- **One transfer primitive.**  All cross-device movement goes through a single
  helper that picks direct-d2d vs host-bounce from an empirical startup probe
  (`self._d2d_safe`).  See §A.1 and the hardware findings below.
- **Trivial-sharding unification.**  Single-device is a trivial `NamedSharding`
  over a 1-device mesh, so the multi-device and single-device code paths are the
  same.  `mesh is None` special-casing goes away.

### Hardware findings that constrain the design (verified 2026-05-29, JAX 0.10.1)
- **H100 (2×):** all cross-device `device_put` paths correct, incl. direct d2d.
- **L40S (2×):** transferring a **device-resident** array across devices is
  **silently zeroed** (non-default shard → zeros, no error).  Host-sourced
  (`device_put(np.asarray(x), …)`) and pre-placed-shard assembly
  (`make_array_from_single_device_arrays`) are correct.
- Probe script: `experiments/sharding/parallel_performance/device_put_check.py`
  (migrate from cluster scratch — see Migration).  Implication baked into §A.1.

---

## Execution order

Phases carry **stable letter IDs** (used in cross-references); **step numbers**
give execution order.  The order front-loads the low-risk FBP filter and reaches
a usable, stress-testable `direct_recon` before forward projection / VCD.

| Step | Phase | What | Why here |
|------|-------|------|----------|
| 1 | **0** | Scaffolding migration | nothing testable without it |
| 2 | **A** | Primitives (transfer, threading, mesh / trivial-sharding) | everything builds on these |
| 3 | **B** | Sharding hooks (new view/slice axes) | filter + projectors need them |
| 4 | **F1** | FBP filter | low-hanging: view-sharded, **zero cross-device comms** |
| 5 | **D** | Back projection (reduce-scatter) | the harder collective; needed for direct_recon |
| 6 | **F2** | `direct_recon` (FBP recon) | **first usable, stress-testable pipeline**; prereq for VCD |
| 7 | **C** | Forward projection (all-gather) | only needed by VCD; adjoint test lands here |
| 8 | **E** | VCD integration + halos | needs C + D |

Rationale for the order (vs. plain alphabetical): `direct_recon` depends on the
FBP filter and back projection but **not** forward projection, so we can reach a
complete, verifiable FBP reconstruction early and exercise the primitives and
the back-projection reduce-scatter under realistic load before building forward
projection and VCD.

---

## Cross-cutting principles (hold across all phases)

1. **Empirical probes, not hardware allowlists.**  d2d-safety, device counts,
   memory — detect at runtime so we auto-adapt to new hardware and to the
   eventual upstream `device_put` fix.
2. **One transfer primitive, one threading primitive.**  All inter-device
   movement → the transfer helper (§A.1).  All per-device fan-out → the
   threading helper (§A.2).  The d2d-vs-host-bounce branch lives in exactly one
   place.
3. **Trivial-sharding regression gate.**  Every phase that touches a projector
   or recon path must pass a "single-device trivial-sharding output is
   bit-exact vs. plain prerelease" check before any multi-device work on that
   piece.  Cheap because trivial sharding unifies the paths.
4. **No silent fallbacks.**  When the L40S host-bounce path (or any degraded
   path) is taken, log it once so it is observable, not invisible.
5. **Native-sharding / API contract:** user-facing `forward_project` /
   `back_project` / `direct_recon` accept either plain or sharded input and
   **return gathered (host/plain) arrays**.  Internal `sparse_forward_project` /
   `sparse_back_project` **expect sharded input and return sharded output** — no
   gather inside.
6. **Sharding primitives are internal, accessed with the `mjs` prefix.**  The
   `mbirjax._sharding` subpackage is internal (leading underscore; not part of
   the public API and not re-exported at the top level).  Everywhere — consumers
   *and* in-package modules — use `import mbirjax._sharding as mjs`, then call
   `mjs.move_shard(...)`.  Importing the **submodule directly** is safe even
   mid-import of `mbirjax`: it forces `mbirjax._sharding` to load, and that
   subpackage pulls in only jax/numpy/warnings, nothing that loops back into the
   partially-initialized `mbirjax`.  (The thing that *would* be fragile mid-import
   is aliasing the *top-level* package — `import mbirjax as mj` then reaching
   `mj._sharding.move_shard` as an attribute before `__init__` has bound it; the
   submodule alias avoids that.)  Verified: `import mbirjax` succeeds with
   `tomography_model.py` using the `mjs` alias.

---

## Migration (research → beta; do as part of Phase 0 unless noted)

| Item (research branch) | Action | Notes |
|---|---|---|
| `tests/conftest.py` (CPU detect, `preferred_devices`) | migrate ~as-is (verify) | design-agnostic; needed before any test runs |
| `mbirjax/_device_setup.py` (sets `XLA_FLAGS` / virtual CPU devices before JAX init) | migrate ~as-is (verify) | wire as **first** import in `mbirjax/__init__.py` |
| `device_put_check.py` (cluster scratch `~/PycharmProjects/`) | migrate into `experiments/sharding/parallel_performance/` | version-control the probe; it validates §A.1 |
| `tests/test_sharding_step0–3.py` | migrate as **reference only** | test old API/axes; harvest assertions into new phase tests, don't run wholesale |
| `experiments/sharding/parallel_performance/*` (incl. `fbp_parallel_options.md`) | already migrated | fbp comparison still valid; forward/back perf numbers must be **re-measured** under view-sharding |
| heterogeneous CPU-recon/GPU-sino design (`main_device`/`sinogram_device`) | redesign, not copy | see Open Question O1 |
| `jax_rounding_bug` docs (`experiments/bugs_and_artifacts/`) | **leave on research branch** | beta inherits the bug knowingly; nothing to port |
| `.claude/status.md` | do **not** copy verbatim | research-specific; beta uses `sharding_status.md` |

"Migrate ~as-is (verify)" = copy, then check against prerelease — prerelease may
have diverged in ways that need small adjustments; do not copy blindly.

---

## Step 1 — Phase 0: Scaffolding migration (no new design)  ✅ COMPLETE (2026-05-29)

**Goal:** make the beta worktree testable for sharding at all.

- [x] Migrate `mbirjax/_device_setup.py`; add `from . import _device_setup` as
      the **first** line of `mbirjax/__init__.py` (before any JAX import).
      Device-count policy: env `MBIRJAX_NUM_CPU_DEVICES` → macOS P-cores →
      Linux affinity → fallback, capped at 8 (see `_device_setup.py`).
- [x] Migrate `tests/conftest.py` (CPU detection + `preferred_devices(n)`).
- [x] Recreate `device_put_check.py` into
      `experiments/sharding/parallel_performance/` (cluster scratch copy was
      gone; recreated with H100/L40S verified-results header).
- [x] Sanity: `import mbirjax` spins up 8 virtual CPU devices on M3 Max;
      `jax.devices()` shows them; `MBIRJAX_NUM_CPU_DEVICES=3` override honored.
- **Test:** [x] `test_sharding_scaffolding.py` — 4/4 pass (≥2 devices, XLA flag
      present, `preferred_devices(2)` returns 2, over-request returns None).
      Existing suite still collects unchanged.

---

## Step 2 — Phase A: Foundations (primitives, no projector changes)  ✅ COMPLETE (2026-05-29)

**Goal:** the two reusable primitives + mesh setup, each unit-tested in
isolation with tiny arrays.

Implemented additively in a new subpackage `mbirjax/_sharding/`
(`transfer.py`, `thread_execution.py`, `__init__.py`) plus
`TomographyModel.configure_sharding`.  Existing single-device code paths
(`main_device`/`sinogram_device`) were left untouched — they retire per-piece
as the projectors are ported (Phases C–F).  Tests in
`tests/test_sharding_primitives.py` (10) pass; scaffolding still green.

### A.1 Transfer helper (the one cross-device primitive)
- [x] Empirical probe `is_dev2dev_safe(devices)` (move known array dev0→dev1,
      verify readback); cached as `self.dev2dev_safe` in `configure_sharding`.
- [x] `move_shard(x, target_device, dev2dev_safe=True)` chooses:
      - safe → direct `device_put(x, target_device)`.
      - not safe → host bounce `device_put(np.asarray(x), target_device)`.
      Warns once when the host-bounce path is first taken (principle 4).
- [x] Endpoints are **arbitrary devices** (GPU/GPU *and* CPU/GPU) — no two-GPU
      assumption (feeds Open Question O1).
- **Test:** [x] both branches exercised (force `dev2dev_safe` True/False);
      round-trip values preserved; warning asserted on host-bounce.

### A.2 Threading / per-device execution helper
- [x] `run_per_device(devices, worker_fn)` — one thread per device, each under
      `jax.default_device(device)`, returns on-device results in device order.
- [x] `assemble_sharded(per_device_arrays, global_shape, sharding)` — wraps
      `make_array_from_single_device_arrays` (kept separate so fan-out and
      assembly compose).
- [x] Block discipline: `run_per_device` does NOT `block_until_ready`, so the
      next batch's transfer can overlap current compute (documented).
- **Test:** [x] fan a trivial op across N devices, check device-order results,
      results land on correct devices, assembly matches reference.

### A.3 Mesh + `configure_sharding` + trivial-sharding unification
- [x] `configure_sharding(devices=None)` builds a 1-D `Mesh` (axis `'devices'`),
      runs the probe, sets `use_gpu='sharded'`.  Additive: does not alter
      `main_device`/`sinogram_device` or `set_devices_and_batch_sizes`.
- [x] Single-device = trivial mesh (`devices=None` → 1 device); same code path.
- [ ] Decide & document the trivial-vs-gathered boundary (Open Question O2):
      internal stays trivially-sharded, user-facing gathers.  *(Deferred to the
      first projector phase that returns to a user — Phase F1/D — where the
      gather actually happens; the mesh plumbing is in place now.)*
- [x] Divisibility assert (Open Question O4): both `num_views % n_devices == 0`
      and `num_slices % n_devices == 0`, with a clear error.
- **Test:** [x] `configure_sharding` with 1 and 2 (virtual CPU) devices; mesh
      size, `dev2dev_safe` cached; non-divisible view count raises clearly.

---

## Step 3 — Phase B: Sharding hooks under the new axes (parallel beam)  ✅ COMPLETE (2026-05-30)

**Goal:** geometry hooks that place objects on the correct axes, with the axis
choice declared in one overridable place rather than hardcoded throughout.

All hooks implemented in the **base class** `TomographyModel` (uniform scheme;
parallel beam inherits with no overrides).  Additive — no existing path changed.
`tests/test_sharding_hooks.py` (8) pass; full sharding suite 22/22; projectors
regression unchanged.

- [x] **Axis-declaration hooks** on `TomographyModel` (simple ints, overridable
      per geometry): `sinogram_shard_axis()` → 0 (view); `recon_shard_axis()` →
      **-1** (last axis = slices; one value works for both 3-D `(rows,cols,
      slices)` and flat `(rows*cols, slices)` recon).  Single source of truth.
- [x] Retrofit `configure_sharding`'s divisibility asserts to read these hooks
      (`shape[axis % ndim]`) instead of hardcoding views/slices.
- [x] `_shard_sinogram` / `_shard_recon` build their PartitionSpec from the axis
      hooks via the shared `_shard_on_axis(x, axis)` helper.
- [x] `_gather_sinogram` / `_gather_recon` inverses via `_gather_to_host`
      (uncommitted single-device output).
- [x] `_shard_on_axis` uses the `dev2dev_safe` probe through `move_shard`, so the
      numpy host-bounce is paid only on hardware that needs it (was unconditional
      on the research branch).  Re-sharding an already-correct array is a no-op.
- [x] `_extract_halos` (slice-boundary; ports cleanly — recon still
      slice-sharded; uses `[..., -1]`/`[..., 0]` so it is rank-agnostic).
- [x] Native-sharding contract documented in the hook docstrings.
- **Test:** [x] `test_sharding_hooks.py` — axis hooks, no-mesh no-ops, shard/
      gather round-trips (sino view-axis, flat + 3-D recon slice-axis), reshard
      no-op, halos match known boundary slices.  Single file.

---

## Step 4 — Phase F1: FBP filter (view-sharded, zero cross-device comms)

**Goal:** the low-hanging first real piece.  Under view-sharding the ramp filter
acts per-view, so each device filters its own views entirely locally — no
cross-device transfer at all.  Good confidence-builder for the B hooks + A
primitives.  Background: `parallel_performance/fbp_parallel_options.md`.

- [x] `fbp_filter` threading under the **view** axis: each device filters its
      view-shard locally via §A.2; assemble the view-sharded filtered sinogram.
- [x] Plain or sharded input accepted (`_shard_sinogram` no-op on either).
      (`fbp_filter` is the internal sharded-contract method → returns
      view-sharded, no gather; the user-facing gather is the O2 boundary,
      tracked separately.)
- **Tests:** [x]
  - [x] single-device trivial-sharding == prerelease, bit-exact.
  - [x] 2-device (virtual CPU) == single-device to float noise.
  - [x] no cross-device transfer in the view-sharded path (the filter is
        per-row, per-view — embarrassingly parallel).

### F1 follow-up: row-batched kernel — DONE (2026-05-31)

The per-device kernel's memory was the last F1 item (per_view's FFT batch =
`view_batch_size × n_rows` grew with geometry → OOM at 1624³).  Resolved:

- **Kernel:** `tomography_utils.apply_row_filter(block, filter)` — a `lax.scan`
  over overlapping B-row windows written in place via `dynamic_update_slice` (no
  padding copy, no concat; the clamped last window overlaps and recomputes a few
  rows, idempotent for a per-row filter).  `vmap` supplies the B-way parallelism
  and the scan has no `batch_size`, so jax#27591 doesn't apply.  Geometry-neutral
  — cone beam and the rest reuse it.
- **B = `ROW_FILTER_BATCH = 1024`** (GPU B-sweep knee).  Work area ~
  `B × fft_len(c)`; a c-aware `B ≈ budget/fft_len(c)` is noted for later (only
  ever *larger* than 1024 for small c).
- **Also:** folded `pi/num_views` into the f32 filter (no f64 post-multiply).
- **Cleanup:** removed the `_FBP_FILTER_KERNEL` switch + per_view/flat kernels and
  the `_split_body_tail`/`_FLAT_MAP_BATCH` machinery; `view_batch_size` deprecated
  on the user-facing methods (DeprecationWarning), gone from internals.

**Measured (H100):** 2× (input+output) memory floor, ~1/n scaling,
divisibility-agnostic, ~10× faster than a small B, 1624³ on a single GPU,
cross-platform correctness 5.6e-9.  **CPU:** ~5.4–6.5× at 8 virtual devices — a
real speedup (see `sharding_status.md` §Targets).  Tests: `test_sharding_*` +
`test_fbp_fdk` 28/28.

---

## Step 5 — Phase D: Back projection (reduce-scatter)  ✅ COMPLETE (2026-06-01, GPU-validated)

**Done.**  Reduce-scatter back projection with slice-band **streaming** (no
full-cylinder partial): peak/shard 11×→3.2× at 1024³/4-dev, single GPU 1024³
28→~12 GB (streams by default), near-ideal scaling (3.92× on 4), no time cost.
Device+compute-bounded band default (`_slice_band_length`); reused thread pool;
memory-bandwidth-bound.  Remaining frontier = sino+recon floor (host streaming,
deferred).  Detail below + in `sharding_status.md`.


**Goal:** `sparse_back_project` with the harder collective — sum partial
cylinders across devices, scatter to slice owners.  This is where
summation-order and (later) adjoint-correctness live, so we tackle it before
forward.

**Design (settled with Greg):** the reduce-scatter splits into a geometry-neutral
communication pattern + a per-device compute that *can* restrict to a slice band.
Key realization (Greg): each device holds **complete views**, so it can
back-project **any** output slice band from purely local data — true for any
geometry.  Partitioning the **output** cylinder means each voxel is computed
once (no redundant compute, parallel *or* cone beam).  The memory win (your
point #2) comes from *streaming* slice bands and freeing — deferred (see below).
Overlapping-tail trick (F1) is reusable for slice sub-tiling *only* where the
combine is SET (the complete reduced value), never ADD; deferred with sub-tiling.

- [x] Per device: back-project local views into partial cylinders spanning all
      slices; reduce-scatter partials to slice owners (via §A.1/§A.2).
      Two-phase: Phase 1 `run_per_device` per-device partials (reuse the existing
      jitted projector unchanged — **no kernel edit for parallel beam**); Phase 2
      naive all-to-all (`move_shard` each owner's band to it, owner sums).
      `_sparse_back_project_sharded` in `tomography_model.py`; single-device body
      extracted to `_sparse_back_project_single_device`.  Loops over the mesh —
      not hardcoded to 2 (validated 1/2/4/8 virtual CPU).
- [ ] Budget the transient full-cylinder partial buffer (per device, per
      pixel-batch) — note memory cost, larger for cone beam later.
      **Deferred** (Greg): no sub-tiling now; the full-partial transient is the
      same order as the single-device working set.  Revisit via scaling tests;
      the streaming/overlap-tail seam (1/n transient) is the obvious next layer.
- [x] `back_project` (user-facing): shard sinogram at entry, gather recon at exit.
- **Tests:** (`tests/test_sharding_back_projection.py`, 7/7)
  - [x] single-device trivial-sharding == plain prerelease, bit-exact (n_dev=1).
  - [x] 2-device == single-device to float noise (~1e-7 rel); also 4/8.
  - [x] internal `sparse_back_project` returns slice-sharded (no gather); public
        `back_project` returns gathered/plain; coeff_power=2 (Hessian) matches;
        view subset rejected (NotImplementedError) in sharded mode.
  - **Deviation from plan:** added a dedicated `test_sharding_back_projection.py`
    (matching the `test_sharding_fbp.py` precedent) rather than threading
    sharded mode into `test_projectors.py` — that file is multi-geometry
    (cone/helical/translation), and Phase D sharding is parallel-beam-only.
  - *(The `back(forward(x))` adjoint round-trip is a **Phase C** gate, when the
    sharded forward projector exists.)*

**Open notes carried forward:**
- `compute_hessian_diagonal` now returns a **slice-sharded** (recon-native)
  array in sharded mode (the reshape preserves the slice sharding) — correct
  values, and arguably the right sharding for VCD.  Whether it should gather
  is a Phase-E contract decision; left sharded for now.
- `_sparse_back_project_sharded` defensively `_shard_sinogram`s its input (a
  no-op when already sharded) so un-ported callers (e.g. `compute_hessian_diagonal`
  passing plain weights) work; revisit when callers shard at entry uniformly.
- **GPU-validated (H100, 2026-06-01).** Slice-band **streaming** added (no
  full-cylinder partial; row-sliced per-band reduce-scatter; balanced bands).
  Multi-device peak/shard **11× → 3.2×** at 1024³/4-dev at no time cost;
  near-ideal scaling (3.92× at 4 dev).  Default band is budget/compute-bounded
  (`_slice_band_length`), and **single device streams by default** too (compute
  cap) — 1024³ on one GPU 28 GB → 10.5 GB (0.37×), time flat.  Persistent thread
  pool (`device_pool`) removes per-band pool churn.  See `sharding_status.md` for
  the full findings.
- **Above-floor residual is compute/working-set, NOT assembly (tried & reverted,
  2026-06-01).** The single-device sweep plateaus at ~**1.44× the sino+recon
  floor** for all small bands — a band-independent ~3 GB transient.  We
  hypothesized this was the per-owner `jnp.concatenate(band_list)` doubling a
  recon shard during assembly, and replaced it with a preallocated, donated
  `dynamic_update_slice` write (the F1 `apply_row_filter` pattern on the recon
  slice axis).  **Measurement disproved it:** peak got *worse* everywhere
  (1024³/4-dev 3435 → 4840 MB, +41%; n=1 also up slightly), so the concatenate
  was never the binding peak — it happens after the compute phase, at a
  lower-memory moment.  The residual is the live recon + per-band compute working
  set, not assembly.  **Reverted** (`017e37c` undone).  Lesson: don't port the F1
  trick without confirming the *cause* of the plateau by measurement.
- **Frontier (back projection): host↔device streaming of sino/recon.** Below the
  ~7.3 GB floor (1024³) the full sinogram (input) and recon (output) are the
  irreducible on-device residents; only loading per-band sino rows / evicting
  written recon slices to host beats it.  Bigger change (overlaps the O1
  heterogeneous-placement design); deferred.

---

## Step 6 — Phase F2: `direct_recon` (FBP reconstruction)

**Goal:** first **usable, end-to-end, stress-testable** sharded pipeline:
shard sinogram → `fbp_filter` (F1) → `back_project` (D) → gather recon.
Exercises the primitives and the back-projection reduce-scatter on a complete
reconstruction with a prerelease baseline.

- [ ] `direct_recon`: shard sinogram once at entry, pass through filter →
      back_project, gather at exit.  No re-shard between filter and back_project
      (both view-in / the back-projector owns the view→slice movement).
- [ ] First cut is **homogeneous**: recon slice-sharded across the projection
      devices (recon fits across the mesh).  Heterogeneous CPU-storage is O1,
      deferred.
- **Test data note:** generate test sinograms with prerelease's **plain**
  `forward_project` (the sharded forward projector C does not exist yet) or a
  phantom, then shard and run.
- **Tests:**
  - [ ] single-device trivial-sharding == prerelease `direct_recon`, to
        reconstruction tolerance.
  - [ ] 2-device == single-device to reconstruction tolerance.
  - [ ] adapt `tests/test_fbp_fdk.py` to both modes.
  - [ ] **stress test:** larger phantom across 2 devices; confirm correctness +
        watch memory (the back-projection partial buffer).

---

## Step 7 — Phase C: Forward projection (stationary sinogram, moving recon)

**Goal:** `sparse_forward_project` with all-gather of voxel cylinders, pipelined.
Only needed for VCD, so it comes after the FBP pipeline is usable.

- [ ] Shard at the **batching layer** (`sparse_forward_project`), not by
      overriding top-level `forward_project` (research insight; survives the
      axis change).
- [ ] Per device: all-gather the voxel-cylinder pixel-batch (via §A.1/§A.2),
      project its local views, write its view-shard of the sinogram locally
      (sinogram never moves).
- [ ] Pipeline: prefetch next pixel-batch's gather during current projection.
- [ ] `forward_project` (user-facing): accept plain or sharded recon, gather
      sinogram at exit (contract §5).
- **Tests:**
  - [ ] single-device trivial-sharding == plain prerelease, bit-exact.
  - [ ] 2-device (virtual CPU) == single-device to float noise.
  - [ ] adapt the existing forward tests in `tests/test_projectors.py` to both
        modes.
  - [ ] **adjoint round-trip** `back(forward(x))` (sharded): adapt the existing
        adjoint tests in `tests/test_projectors.py`; this is the gate for the
        all-gather / reduce-scatter pair being correct adjoints.

---

## Step 8 — Phase E: VCD integration

**Goal:** end-to-end VCD with sharded entry/exit and halo exchange.

- [ ] Entry: shard sinogram (view) / recon (slice) / weights (view) once.
- [ ] Halo exchange wired into the VCD loop using `_extract_halos` (recon is
      slice-sharded, so qggmrf halo logic from research should largely carry
      over — verify).
- [ ] Exit: gather recon.
- [ ] Confirm no accidental re-shard or gather inside the loop body.
- **Tests:**
  - [ ] single-device == prerelease VCD to reconstruction tolerance.
  - [ ] 2-device == single-device to reconstruction tolerance; same iteration
        count (update equations are mathematically identical).
  - [ ] adapt `tests/test_vcd.py` / `tests/test_qggmrf.py` to both modes.

---

## Open design questions (decide before the phase that needs them)

**O1 — Heterogeneous CPU-recon / GPU-sinogram (first touchpoint Phase D;
decision deferrable).**
The current research/prerelease mechanism (`main_device` + `sinogram_device`,
any of CPU/CPU, CPU/GPU, GPU/GPU) supports a large recon on a big-memory CPU
with projection on a moderate-memory GPU, transferring voxel batches per
projection.  Because back projection (D) materializes the recon, the "where does
the recon live" question first appears there — but the **first cut of
`direct_recon` is homogeneous** (recon slice-sharded across the projection
devices), so the heterogeneous decision can be deferred past F2.
**Recommendation:** the new transfer helper (§A.1) already *is* this — two
endpoints, move batches between them.  Do **not** force CPU-recon / GPU-sino into
one `NamedSharding` mesh (JAX meshes want homogeneous devices and get fragile
otherwise).  Model it as an explicit placement policy — "recon storage device"
vs "projection device(s)" — that the transfer helper understands, with the
multi-GPU sharded mesh as one case and CPU/GPU heterogeneous as another, both
behind the same `move_shard`.  Keep §A.1's interface endpoint-agnostic now
(already a checklist item); design the placement-policy object when a
recon-too-big-for-GPU need forces it.

**O2 — Trivially-sharded vs gathered return values (needed by Phase A.3).**
"Trivially sharded" = a `jax.Array` with a 1-device `NamedSharding` (still a
device array carrying sharding metadata).  "Gathered" = a plain/uncommitted
array (or numpy) with no sharding.  **Recommendation (per Greg, 2026-05-29):**
user-facing `forward_project`/`back_project`/`direct_recon` return **gathered**
arrays; internal `sparse_*` keep **trivial sharding** so the code paths unify.
Confirm gathered output still behaves like a plain array for downstream user
code.

**O3 — Sparse-view regime (note only, no design now).**
"Move the recon" is bandwidth-optimal only when views are many (recon < sino).
In sparse-view (recon > sino) it moves more data than a stationary-recon scheme
would.  Acceptable for current targets; **recorded so it is a conscious limit,
not a surprise.**  Revisit only if a sparse-view multi-GPU need appears.

**O4 — Divisibility: padding / cropping to shard (needed by Phase A.3 / B).**
The new scheme shards on **two** axes, so **both** must divide the device count:
sinogram by **view** (`num_views % n_devices == 0`) and recon by **slice**
(`num_slices % n_devices == 0`).  The research branch had only the slice
constraint; view divisibility is new.  Options per axis:
  - **Assert** (cheap, strict): require divisibility, error clearly otherwise.
  - **Pad/crop** (flexible): pad the sharded axis up to the next multiple of
    `n_devices` before sharding, crop the output back after gather.  View
    padding = appended zero/duplicate views (must not corrupt the
    reconstruction — zero-weight or excluded); slice padding = extra recon
    slices discarded at the end.
**Recommendation:** start with a clear **assert** in `configure_sharding`
(Phase A.3) so early phases aren't blocked on padding machinery; add optional
**pad/crop** later as a usability layer once the core pipeline works.  Note that
padded views/slices interact with the projector and weights, so padding is not a
pure wrapper — it must be consistent across sinogram, weights, and recon.

---

## Cone beam (later — out of scope for this plan)

Once parallel beam is proven end-to-end, repeat Phases B–E (and F1/F2) for cone
beam: geometry-specific `_shard_*`/`_extract_halos` overrides, the additional
integer scatter indices, and larger per-device partial buffers.  The Phase A
primitives (transfer, threading) and the view/slice axis scheme are shared
unchanged — that uniformity is the whole point of the new design.

---

## Final cleanup sweep (do near the end, before merge)

- [ ] **Import-order sweep: `import mbirjax` before `jax`** in all user-facing
      files (tests, examples, demos, experiment scripts) so
      `mbirjax._device_setup` runs before JAX initializes its backends.  We
      follow this in every new file we create; this sweep catches pre-existing
      files and any that slipped through.
      Exemptions (do NOT change these):
        - Files *inside* the `mbirjax` package (e.g. `_sharding/*.py`,
          `tomography_model.py`) — they execute mid-`import mbirjax`, so they
          just `import jax`; importing mbirjax there would be circular.
        - `tests/conftest.py` — deliberately sets `XLA_FLAGS` itself before its
          own `import jax`, independent of mbirjax import resolution.
        - `experiments/.../device_put_check.py` — intentionally mbirjax-free
          standalone probe.
- [ ] **Default single-device sharding (auto-configure a trivial 1-device mesh).**
      *(Greg, 2026-06-01: leave for later, but tracked here.)*  Today the sharded
      path — and therefore single-device **slice-band streaming** — only runs
      after an explicit `configure_sharding(...)`; a plain single-GPU
      `back_project` still uses the old non-streaming path.  Wiring a trivial
      1-device mesh by default (constructor? first use?) would give every
      single-GPU user the streaming memory win automatically (1024³: ~28 GB →
      ~10 GB, no time cost — the "stretch the recon per GPU" lever), and is the
      enabler for collapsing the mesh-None paths below.  **Motivation is now
      concrete** (measured single-device benefit), not just cleanup.  Risks: it
      routes all single-device projection through the sharded path, which does
      not yet handle the O1 heterogeneous CPU-recon/GPU-sino placement
      (`main_device`/`sinogram_device`), so that must be reconciled first.
- [ ] **Collapse the single-device (mesh-None) code paths.**  Once every
      projector/recon method is ported *and* the trivial mesh is auto-configured
      (above), single-device becomes just the trivial-1-device-mesh case, so the
      `if self.mesh is None: ...` fallbacks (e.g. in `fbp_filter`) and the old
      `main_device`/`sinogram_device` machinery can be removed, leaving one
      sharded path.  Deferred until enough methods are ported that always-on
      trivial sharding is safe end-to-end.
- [ ] **Public shard/gather utilities.**  Add thin public wrappers
      (`shard_sinogram`, `shard_recon`, `gather_sinogram`, `gather_recon`) over
      the `_shard_*`/`_gather_*` hooks so callers/tests can pre-shard inputs to
      internal sharded-contract functions (like `fbp_filter`) directly, without
      reaching into private methods.
- [ ] (add other end-of-project cleanups as discovered)

---

## Future project: simplify the sparse-projector batching machinery

*(Separate, deliberately-scoped refactor — NOT part of the sharding work; tracked
here so it isn't lost.  Greg has wanted this for a while.)*

The geometry-agnostic projector core in `projectors.py` layers several batching/
mapping helpers that are hard to follow: `sum_function_in_batches` (lax.scan over
view batches), `concatenate_function_in_batches` (lax.map over pixel batches),
and a `vmap` over the per-view geometry kernel — see `back_project_one_view_to_
pixel_batch` and the forward analogue (`.claude/back_projection_overview.md`
walks the chain).  Goals of a rewrite:
- **Clarity** — collapse the nested scan/map/vmap layers into something
  readable (the F1 `apply_row_filter` single-`lax.scan`-with-`vmap` is the
  template).
- **Remove lax.map fragility** — `concatenate_function_in_batches` uses
  `lax.map`, which has the jax#27591 large-`batch_size` bug; a scan + vmap avoids
  it.
- **Possible memory win for forward / single-device** — the
  `concatenate_function_in_batches` assembles results into a list then
  concatenates (a transient doubling); a preallocated in-place write could help
  *those* paths (note: it did **not** help the sharded back-projection peak —
  measured worse, reverted — because there the concat is not the binding peak).

**Development note:** This would be started with a prototype in ParallelBeam
only by overriding the default projector functions to a class-specific
function that would generalize to all geometries.

**Why it's a separate project, not bundled:** this core is shared by forward +
back projection, every geometry, and the single-device path, so it is
high-blast-radius and needs full re-validation (bit-exact across both directions
and all geometries).  Do it on its own, motivated by clarity + the jax#27591
removal, with the projector regression suite as the gate.

## Other notes

Track here anything we still want off the research branch (beyond the §Migration
table above), so nothing is lost when it is deleted:
- [ ] `_device_setup.py`, `conftest.py` (Phase 0).
- [ ] `device_put_check.py` (Phase 0).
- [x] `fbp_parallel_options.md` (migrated to `parallel_performance/`).
- [ ] Harvest assertions from `test_sharding_step0–3.py`.
- [ ] Decide whether any of `experiments/bugs_and_artifacts/` (jax rounding bug
      docs, center-slice-noise notes) should live on a longer-lived branch
      rather than only on `greg/parallel_tests`.
- [ ] (add items as discovered)
