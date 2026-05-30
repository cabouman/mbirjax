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
  view** (axis 0); recon-like objects sharded **by slice**.  Parallel beam loses
  its slice↔det-row embarrassing parallelism but gains a uniform scheme that
  suits cone beam.  Parallel beam is the training ground; slice-sharding for it
  is a later optimization.
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

## Step 2 — Phase A: Foundations (primitives, no projector changes)

**Goal:** the two reusable primitives + mesh setup + trivial-sharding unify,
each unit-tested in isolation with tiny arrays.

### A.1 Transfer helper (the one cross-device primitive)
- [ ] Startup probe at `configure_sharding`: run a Test-C-equivalent (move a
      known nonzero array dev0→dev1, read back) and cache `self._d2d_safe`.
- [ ] `move_shard(array_or_shard, target_device)` chooses:
      - `_d2d_safe` → direct `device_put(x, target_device)`.
      - not safe → host bounce `device_put(np.asarray(x), target_device)`.
      Log once (principle 4) when the host-bounce path is selected.
- [ ] Designed so endpoints are **arbitrary devices** (GPU/GPU *and* CPU/GPU) —
      do not bake in a two-GPU assumption (feeds Open Question O1).
- **Test:** [ ] unit test on whatever devices are present (virtual CPU on
      laptop, real GPUs on cluster): round-trip values preserved; both branches
      exercised by forcing `_d2d_safe` True/False.

### A.2 Threading / per-device execution helper
- [ ] `run_per_device(mesh, worker_fn) -> assembled` — one thread per device,
      each under `jax.default_device(device)`, returns an on-device result;
      assemble via `make_array_from_single_device_arrays`.
- [ ] Block discipline: avoid premature `block_until_ready` so the next batch's
      transfer overlaps current compute (call this out in the docstring).
- [ ] Workers obtain peer shards via the §A.1 transfer helper (the two compose;
      threading owns fan-out, transfer owns placement).
- **Test:** [ ] unit test: fan out a trivial per-shard op across N devices,
      assemble, compare to single-device reference.

### A.3 Mesh + `configure_sharding` + trivial-sharding unification
- [ ] `configure_sharding(devices)` builds the mesh, runs the §A.1 probe, sets
      `use_gpu='sharded'`.
- [ ] Single-device path = trivial `NamedSharding` over a 1-device mesh (no
      `mesh is None` branches).
- [ ] Decide & document the trivial-vs-gathered boundary (Open Question O2):
      internal stays trivially-sharded, user-facing gathers.
- [ ] Divisibility assert (Open Question O4): both `num_views % n_devices == 0`
      and `num_slices % n_devices == 0`, with a clear error.  (Pad/crop is a
      later usability layer.)
- **Test:** [ ] `configure_sharding` with 1 and 2 (virtual CPU) devices; mesh
      shape, `_d2d_safe` cached, trivial sharding is a no-op round-trip;
      non-divisible view or slice count raises a clear error.

---

## Step 3 — Phase B: Sharding hooks under the new axes (parallel beam)

**Goal:** geometry hooks that place objects on the correct (new) axes.

- [ ] `_shard_sinogram` → **view** axis (axis 0).  (Research used slice/det-row;
      this is the key change.)
- [ ] `_shard_recon` → **slice** axis (axis 2 of 3D, axis 1 of flat).
- [ ] `_gather_sinogram` / `_gather_recon` inverses (host/plain output).
- [ ] `_extract_halos` (slice-boundary; unchanged in spirit from research since
      recon is still slice-sharded — confirm it ports cleanly).
- [ ] Native-sharding contract table for the new axes (forward: recon in →
      sinogram out; back: sinogram in → recon out), written into the docstrings.
- **Test:** [ ] hook round-trips (`_shard_*` then `_gather_*` == identity to
      float noise); halos match `np.asarray(recon)[:, boundary]`.  Fold into a
      single `test_sharding_hooks.py` rather than one file per hook.

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
