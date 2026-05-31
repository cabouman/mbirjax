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
6. **Sharding primitives are internal, accessed with a prefix.**  The
   `mbirjax._sharding` subpackage is internal (leading underscore; not part of
   the public API and not re-exported at the top level).  Follow the codebase's
   prefix convention rather than bare `from ... import name`:
     - **Consumers** (tests, examples, experiment scripts):
       `import mbirjax._sharding as mjs`, then call `mjs.move_shard(...)`.
     - **In-package modules** (`tomography_model.py`, projector code): aliasing
       mbirjax would be circular mid-import, so use
       `from mbirjax._sharding import move_shard, ...` (direct import is the only
       option inside the package).

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

- [ ] `fbp_filter` threading under the **view** axis: each device filters its
      view-shard locally via §A.2; assemble the view-sharded filtered sinogram.
- [ ] User-facing entry: plain or sharded input; gather at exit (contract §5).
- **Tests:**
  - [ ] single-device trivial-sharding == plain prerelease, bit-exact.
  - [ ] 2-device (virtual CPU) == single-device to float noise.
  - [ ] confirm (instrument/grep) no cross-device transfer occurs in the
        view-sharded path.

### F1 follow-up: row-batched fbp_filter kernel (settles the reshaping item)

**Status (2026-05-30):** jit fix landed; the per-device kernel scales (~6.5×).
The remaining F1 item is the per-device kernel's **memory**.  GPU evidence (H100,
`fbp_filter_scaling.py`): the default `per_view` kernel's FFT batch is
`view_batch_size × n_rows` rows simultaneously, so at 1624³/4-dev it allocated a
**~27–35 GB cuFFT work area** (vs a ~4 GB shard) and **OOM'd at 1 device**.  The
batch — and thus the work area — scales with geometry.

**Design (the reshape that decouples memory from geometry):**
```
reshape (v_local, r, c) → (v_local·r, c)
pad rows up to a multiple of B → (M·B, c),  M = ceil(v·r / B)
reshape → (M, B, c)
lax.map(lambda chunk: vmap(convolve_row)(chunk), axis 0)   # NO batch_size arg
reshape → (M·B, c), crop to (v·r, c), reshape → (v_local, r, c)
```
- **Why not the existing `flat` kernel:** it bounds the batch via
  `lax.map(batch_size=128)`, which is the jax#27591-unsafe path (hence the 128
  cap) and needs the `_split_body_tail` hack for `lax.map`'s contaminating
  internal padding.  The reshape above uses `lax.map` with **no `batch_size`**
  (scan over `M`, `vmap` supplies the `B`-way parallelism), so **#27591 does not
  apply** and `B` is a free knob.  Our **own zero-pad + crop** replaces body/tail
  (each row convolves independently; zero rows give benign, cropped output).
- **Memory:** FFT work area `≈ B × fft_len(c)`, **bounded by `B` alone** —
  independent of `v`, `r`, and device count.  Concrete win to verify: B=256 →
  ~tens of MB work area, so **1624³ should run on a SINGLE H100** (no OOM).

**Decisions (Greg, 2026-05-30):**
- Keep the public `view_batch_size` arg **for now, mark deprecated soon**; it no
  longer governs the FFT batch.  A module constant `_FBP_ROW_BATCH` (the `B`)
  does; **default B = 256**.
- **B must become c-aware:** the work area is `~B × fft_len(c)` where
  `fft_len(c) ≳ 3c−2`, so to hold a fixed per-device memory budget, `B` should
  **decrease as `c` grows** (`B ≈ budget / fft_len(c)`).  Start with the fixed
  default; the GPU sweep sets the budget / the `B(c)` relationship.

**Step sequence:**
1. Implement `"row_batch"` as a third option behind `_FBP_FILTER_KERNEL` (jitted,
   `B` static).  Keep per_view/flat for A/B.
2. CPU correctness: single-device == prerelease baseline (bit-exact); 2-dev ==
   1-dev; non-divisible `v·r` clean (zero-pad/crop).  Adapt
   `tests/test_sharding_fbp.py` + `test_fbp_fdk.py`.
3. GPU (Greg runs): (a) confirm **1624³ no longer OOMs at 1 device**; (b) **sweep
   B** (now free of #27591) for the throughput/memory sweet spot + the `B(c)`
   budget; (c) confirm the memory-fraction curve moves toward ideal.
4. Make `row_batch` the default; **remove the `_FBP_FILTER_KERNEL` switch, the
   losing kernels, and the `_split_body_tail`/`_FLAT_MAP_BATCH` machinery**.
5. Update `sharding_status.md` (close the reshaping item) → then Phase D.

**Risks:** per-`B`/shape cuFFT plan compilation (fine for a sweep); ≤`B−1` padded
rows wasted per shard (negligible when `v·r ≫ B`); small `B` may underfill a GPU
(the sweep + #27591-freedom let us go bigger); assert `mode='valid'` keeps output
length `c`.

---

## Step 5 — Phase D: Back projection (reduce-scatter)

**Goal:** `sparse_back_project` with the harder collective — sum partial
cylinders across devices, scatter to slice owners.  This is where
summation-order and (later) adjoint-correctness live, so we tackle it before
forward.

- [ ] Per device: back-project local views into partial cylinders spanning all
      slices; reduce-scatter partials to slice owners (via §A.1/§A.2).
- [ ] Budget the transient full-cylinder partial buffer (per device, per
      pixel-batch) — note memory cost, larger for cone beam later.
- [ ] `back_project` (user-facing): gather recon at exit.
- **Tests:**
  - [ ] single-device trivial-sharding == plain prerelease, bit-exact
        (standalone gate — no forward projector exists yet to pair with).
  - [ ] 2-device == single-device to float noise.
  - [ ] adapt the existing back-projection tests in `tests/test_projectors.py`
        to run in both modes rather than adding a parallel file.
  - *(The `back(forward(x))` adjoint round-trip is a **Phase C** gate, when the
    sharded forward projector exists.)*

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
- [ ] **Collapse the single-device (mesh-None) code paths.**  Once every
      projector/recon method is ported, single-device should be just the
      trivial-1-device-mesh case, so the `if self.mesh is None: ...` fallbacks
      (e.g. in `fbp_filter`) and the old `main_device`/`sinogram_device`
      machinery can be removed, leaving one sharded path.  Deferred until enough
      methods are ported that always-on trivial sharding is safe end-to-end
      (needs the auto-configure-trivial-mesh decision settled — see below).
      Requires deciding where the trivial mesh gets configured by default
      (constructor? first use?).
- [ ] **Public shard/gather utilities.**  Add thin public wrappers
      (`shard_sinogram`, `shard_recon`, `gather_sinogram`, `gather_recon`) over
      the `_shard_*`/`_gather_*` hooks so callers/tests can pre-shard inputs to
      internal sharded-contract functions (like `fbp_filter`) directly, without
      reaching into private methods.
- [ ] (add other end-of-project cleanups as discovered)

---

## Note: research-branch content to migrate before `greg/parallel_tests` is deleted

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
