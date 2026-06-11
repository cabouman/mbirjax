# Sharding implementation plan v2 — placement architecture (beta `greg/parallel_sharding`)

*Created 2026-06-02.  This is the **current forward plan**.  It supersedes
`sharding_implementation_plan.md` for forward planning; that doc is retained as
the **completed-work record** (Phases 0/A/B/F1/D/F2 case studies) plus the
still-valid **cross-cutting principles, verified hardware facts, and resolved
open questions (O1–O4)** — read it for history and principles, this doc for
where we are going next.*

---

## Why this plan exists (direction change, 2026-06-02)

Phases F1 (FBP filter), D (back projection), and F2 (`direct_recon`, with the
match-input contract) are **done and GPU-validated**.  In designing the device
configuration UX we converged on a cleaner target architecture than the original
plan assumed, and decided to **re-open back projection** to build on it rather
than retrofit later.  The two ideas:

1. **Placements, not device scalars.**  Replace `main_device` / `sinogram_device`
   with `recon_placement` / `sino_placement` (each a `Sharding`).  Every device
   mode becomes a *placement pair*; single-device is a trivial 1-shard placement.
   Sharding is then **always on** (trivially when degenerate) — one code path.
2. **One movement interface.**  All recon↔compute data motion goes through an
   adjoint pair, `move_cylinders_to_sino` / `sum_cylinders_to_recon`, built on the
   existing `move_shard` primitive, with uniform **pixel-batched** streaming.

What does **not** change: F2's user-facing **match-input** contract and the
`direct_recon` scaling driver are at the public-API layer, independent of back
projection's internals — re-opening D does not disturb them (just re-run the
driver against the new internals).

---

## 0. Design summary

### 0a. Placements

- `recon_placement` and `sino_placement` replace `main_device` /
  `sinogram_device`; each is a `Sharding` (a mesh + shard devices; a single
  device is the trivial 1-shard case).
- Every mode is a placement pair:

  | mode | recon_placement | sino_placement |
  |---|---|---|
  | CPU-only (`none`) | trivial on CPU | trivial on CPU |
  | single-GPU (`full`) | trivial on GPU | trivial on GPU (same) |
  | hybrid (big recon) | trivial on CPU | trivial on GPU |
  | multi-GPU (`sharded`) | slice-sharded on GPU mesh | view-sharded on **same** GPU mesh |

- **Sharding is always on**; single-device is the degenerate trivial placement,
  so there is no separate non-sharded path (the trivial-sharding unification).
- **Scope.**  Support homogeneous multi-GPU and **recon-on-CPU / sino-on-GPU**
  (the big-recon case).  **Out of scope:** sino-on-CPU / recon-on-GPU (would only
  serve host-streaming a too-big *sinogram*; rare in CT, and Greg's call is we
  won't support it).  Multi-*node* is out of scope (single process only).
- The hybrid case uses **two separate single-device meshes** (CPU mesh for recon,
  GPU mesh for sino) — never one mixed mesh — which sidesteps the JAX
  heterogeneous-mesh fragility that O1 worried about.

### 0b. Movement interface (the one thing that crosses placements)

Only **voxel cylinders** move between placements; the sinogram is written
locally on its view-shard and never moves during projection.

- `move_cylinders_to_sino(cylinders)` — forward: gather the full cylinders (all
  slices) for a pixel batch onto each sino device.
- `sum_cylinders_to_recon(partials)` — back (adjoint): reduce per-sino-device
  partial cylinders over devices and scatter slice-ranges into the recon
  placement.
- **Take cylinders directly** (no pixel indexing inside): the pixel axis is
  unsharded, so the caller does `flat_recon[pix]` (a clean slice-sharded
  sub-array) and the write-back; the primitives deal only with the slice/device
  structure.  Keeps them pure and unit-testable.
- **Built on `move_shard`**, looping over the placements' shards: `N×N` for
  homogeneous multi-GPU, `1×1` for single-device/hybrid — **no mode branch**, the
  shard counts come from the placements.  `move_shard` to the same device must be
  a true **no-op** so the single-device path carries zero overhead.
- **Adjoint pair** (gather+concat ⊣ slice+sum) → keeps the projectors adjoint →
  the forward/back adjoint round-trip is the correctness gate.
- **As built (P2/P3):** the side-by-side measurement chose **slice-banding** over
  pixel-batched cylinders, so `move_cylinders_to_sino` / `sum_cylinders_to_recon`
  were **removed** and replaced by the banded adjoint pair `sum_band_to_owner`
  (reduce-scatter, back) / `broadcast_band_to_views` (all-gather, forward) in
  `_sharding/transfer.py`.  The bullets above describe the original cylinders-direct
  design (kept for rationale/history).

### 0c. Streaming = uniform pixel-batching

- The streaming unit is a fixed-size batch of **full cylinders** (a pixel batch),
  uniformly for both directions and both modes — the existing
  `transfer_pixel_batch_size` pattern, generalized.  `B_p` is the memory knob
  (bounds `B_p × num_slices` per device).
- The layer-2 driver is a single uniform pixel-batch loop with the geometry
  kernel injected; no mode-specific band axis.
- **Superseded — see P2's verdict (KEEP BAND).**  The side-by-side measurement
  chose slice-banding over pixel-batching (pixel's peak floored at ~1.7–2.7× band
  for only ~11–16% less time), so the as-built streaming unit is the **slice-band**,
  not a pixel batch.  `_slice_band_length` sizes it; a per-band compute cap
  (`_BACK_PROJECT_MAX_BAND_WORK`) also makes a single device stream.  The
  pixel-batching text above is retained for design rationale only.

### 0d. Vocabulary — user-facing vs internal

- **User-facing word is "devices."**  `configure_devices(devices=None)` is the
  control surface; "shard / mesh / placement" stay internal.  Long term, one
  concept: `configure_devices` subsumes `use_gpu`.
- **Internal** keeps `mesh`, `shard_devices`, `_sharding/`, `move_shard`,
  `recon_placement` / `sino_placement`, `move_cylinders_to_sino` /
  `sum_cylinders_to_recon`.

---

## Phase sequence

Build everything **next to the existing code on ParallelBeam first** (the proven
F1 pattern: parallel path → measure → promote winner → **delete loser**; do not
leave two paths lingering), then port to the other geometries.

### P1 — Placement foundation  ✅ DONE (committed)
- [x] `Placement` (`mbirjax/_sharding/placement.py`): device list + sharded axis
  + 1-D mesh; `shard_ranges()` / `shard_structure()` / `is_trivial`.
- [x] `move_cylinders_to_sino` / `sum_cylinders_to_recon` (cylinders-direct,
  `move_shard`-based, `N×N` / `1×1`, same-device no-op).  Adjoint identity tested.
- [x] Model builds `recon_placement` / `sino_placement` via `_set_placements()`
  (additive; coexists with `main_device`/`sinogram_device` + `mesh`, which the
  `TODO(P6)` retires).
- [x] Tests: `test_sharding_placement.py` (primitives + adjoint);
  `test_sharding_hooks.py::TestModelPlacements` (model construction + consistency).
- Deferred (intentionally): the broad `device_put → move` migration — it happens
  organically as each path is rewritten (P2/P3/P4), not as upfront churn.

### P2 — Back projection on placements (re-opened Phase D)  ✅ DONE (CPU + GPU)
- [x] `_sparse_back_project_pixel`: pixel-batched, uses `sum_cylinders_to_recon`;
  first consumer of `recon_placement` + the movement primitives.  Runs **alongside**
  the slice-banded `_sparse_back_project_sharded` via the temporary
  `_back_project_path` switch ('band'|'pixel'); shared setup in
  `_sharded_back_project_setup`.  `B_p` knob = `back_project_pixel_batch` /
  `_BACK_PROJECT_MAX_PIXEL_WORK`.
- [x] Tests: `TestPixelBackProject` (trivial bit-exact; matches band + single-device;
  Hessian; B_p sweep; match-input sharded-in).
- [x] Harness: `sparse_back_project_scaling.py` measures both paths side by side
  (+ topology/`dev2dev_safe`/per-GPU-throttle capture, `DEVICE_COUNTS`,
  `PIXEL_BATCH_SWEEP`).
- **Findings so far (clean H100×3 GPU run, 1008³):** time comparable (pixel edges
  ahead and scales slightly better at scale — 3.12× vs 2.75× on 3 dev); band uses
  **~2–2.7× less memory** (structural).  Tradeoff: pixel = simpler + scales
  as-well-or-better; band = memory-lean.
- **Decision rule:** the `B_p` sweep decides — if a smaller `B_p` brings pixel's
  memory to band's level without losing its time, **adopt pixel** (retire the band
  code: `_sparse_back_project_sharded`, `_slice_band_length`,
  `_balanced_slice_bounds`, `back_project_slice_band`; rename
  `_sparse_back_project_pixel` → `_sparse_back_project_sharded`; drop the switch);
  else keep band.  **VERDICT (2026-06-03): KEEP BAND.**  The B_p sweep showed
  pixel's peak memory plateaus at ~1.7–2.7× band (untunable — a floor below the
  per-batch transient) for only ~11–16% less time; memory/max-recon-per-GPU wins.
  The pixel path, its `_back_project_path` switch, and `back_project_pixel_batch`
  knob were removed; band remains `_sparse_back_project_sharded`.
  `_sharded_back_project_setup` (the factored shared setup) is kept.
- Anchors (met): trivial bit-exact vs prerelease; n>1 float-noise match.

### P3 — Forward projection on placements (Phase C)  ✅ DONE (CPU; standalone GPU forward-scaling sweep optional)
- [x] **Slice-banded all-gather** forward projection on ParallelBeam — the adjoint
  of the band back projector (chosen over the pixel-batched `move_cylinders_to_sino`
  for memory symmetry with back projection, same reasons band beat pixel in P2).
  New `broadcast_band_to_views` (`_sharding/transfer.py`), the adjoint of
  `sum_band_to_owner`; `_sparse_forward_project_sharded` + per-band helpers; the
  parallel-beam forward kernel sizes its output rows from the input slices.
- [x] **match-input** per the F2 contract (mirror of `back_project`).
- [x] Correctness gate: forward/back **adjoint round-trip** `<Ax,y>==<x,Aᵀy>` green
  at 1/2/4/8 devices; trivial bit-exact; prerelease forward baseline match (1.1e-5).
  Committed (`7461abb`).
- The superseded full-cylinder primitives `move_cylinders_to_sino` /
  `sum_cylinders_to_recon` were **removed** (the banded pair supersedes them; see the
  "as built" note under 0b/0c).
- The geometry-neutral `(g0, L)` slice-band interface was **designed** here (note
  below) but not yet built for parallel beam, which keeps its bit-exact external
  row-crop; the `(g0, L)` kernel lands when the geometries are ported.
- **Pending (optional):** standalone GPU forward-scaling sweep
  (`sparse_forward_project_scaling.py`); forward is already exercised on GPU inside
  the VCD scaling run.

### Design note — geometry-neutral slice-band projector interface (decided 2026-06-04)

Today the sharded back path streams the recon's slice axis in **bands** but
positions each band by **row-slicing the sinogram** (`data[:, g0:g1, :]`), which
only works for parallel beam (detector row `r` ↔ slice `r`).  The agreed
generalization:

- **Interface (B2): `(band_start g0, band_length L)` in *global slice-index*
  units**, passed to the per-view projector; the projector reads the full
  (resident) view and emits exactly slices `[g0:g0+L)`.  `g0` is a **dynamic
  (traced)** scalar; `L` is **static**.
  - *Why integer slice index, not a physical offset / `recon_slice_offset` (B1):*
    a per-geometry survey shows the z-shift is a *different* quantity in each
    geometry, so `recon_slice_offset` is **not** a uniform currency:

    | geometry | slice→row map | z-shift it uses | internal slice-batching |
    |---|---|---|---|
    | `parallel_beam` | positional, row = slice | *none* | *none* |
    | `cone_beam` | z-based vertical fan | `recon_slice_offset` (+ per-view helical) | `entries_per_cylinder_batch` |
    | `multiaxis_parallel` | z-based vertical fan | `recon_slice_offset` | `entries_per_cylinder_batch` |
    | `translation_model` | z-based vertical fan | `translation_vector[2]` (per view) | `entries_per_cylinder_batch` |

    The integer band *is* uniform.  Each geometry converts internally: parallel →
    rows `[g0:g0+L)` via `dynamic_slice` on the resident view; the z-based three →
    compute z for global indices `k = g0 + k_local` using their existing shift
    term.  `recon_slice_offset` stays each geometry's private parameter; **nothing
    is added to parallel or translation.**  This also keeps the door open for a
    future *row-sharded* parallel beam (a device just owns band
    `(g0=row_start, L=row_count)`, no reduce) — adding `recon_slice_offset` to
    parallel beam would have fought that.
- **Recompilation discipline:** `g0` **dynamic**, `L` **static** → ≤2 compiles
  (the balanced split yields ≤2 band lengths).  The per-band position must **not**
  ride through `recon_slice_offset`/`recon_shape`, which are closure/`static`
  constants of the projector — varying them per band rebuilds the projector
  (~`n_dev²` times).
  - *Rejected:* conveying `L` by passing an empty output cylinder — JAX keys the
    jit cache on input *shapes* exactly as on static-arg values, so it saves no
    compile and just adds a throwaway allocation.  Use a static `num_output_slices`
    arg.
  - *Assembly / in-place:* keep `concatenate` for the **multi-device threaded
    reduce-scatter** (an in-place `dynamic_update_slice` accumulator only goes
    in-place under `jit` + `donate_argnums`; in our Python band loop it copies
    per band and serializes the band-to-band streaming overlap).  In-place
    donation **is** the right assembly for the *jittable* single-device streaming
    (P6) and the forward accumulation (P3) — file it there.
- **Slice-batching subsumption (P6):** the z-based geometries' internal
  `entries_per_cylinder_batch` chunking *is* banding one layer down.  Once the
  kernel takes `(g0, L)`, the outer band loop subsumes it and that machinery is
  deleted from cone/multiaxis/translation — but this only fully lands when the
  **single-device path also drives band-looping** (the trivial-sharding
  unification), since today the internal chunking is what bounds single-device
  memory.
- **Adjoint test:** add the same `(g0, L)` to the forward projectors; "forward a
  band, back-project it, compare" at arbitrary `(g0, L)` becomes a strong
  per-geometry gate.

**Implementation note — why this is P3, not a quick parallel-beam change now.**
Inspecting the call stack: the per-view kernel is vmapped by the *shared* generic
layer (`projectors.py` `sparse_back_project_view_batch`, the
`jax.vmap(back_project_one_view_to_pixel_batch, ...)`), so adding a band argument
the kernel uses *internally* changes the generic projector API **and every
geometry's kernel signature at once** — there is no clean parallel-beam-only
version.  For parallel beam specifically the only *localized* way to band is the
**current external row-crop**, which is already bit-exact.  So: leave the external
row-crop as parallel beam's band mechanism for now, and build the `(g0, L)`
interface once in P3 across forward + back with the adjoint round-trip as the
gate.

**Anchor rule (added 2026-06-11, from the P5 Step 4 padding design).**  Physical
coordinates come from PROBLEM shapes + GLOBAL indices; input lengths size loops and
allocations only, never coordinates:
`z(k_global) = Δ_slice · (k_global − (S_real − 1)/2) + shift`, with
`S_real = projector_params.recon_shape[2]` (params — always REAL) and
`k_global = g0 + k_local` (`shift` stays each geometry's private term, per the table
above).  Today the z-based kernels derive that center from the INPUT cylinder length
(`voxel_cylinder.shape[0]`) — a latent identity (the full cylinder is always passed, so
length == recon_shape[2]) that the `(g0, L)` port AND slice padding both break, the same
way: a sub-band or padded length shifts every *real* coordinate by half the length
difference times Δ.  Sourcing the anchor from params is **bit-exact today** (identical
value on full cylinders) ⇒ a zero-risk mechanical edit at the port.  Sites:
`cone_beam.py` 390→402, 434, 997; `translation_model.py` 315→360;
`multiaxis_parallel.py` 207→225 (+ its validity clip at 254).  **Corollary:** padded or
device-count-dependent shapes never enter `projector_params` or any jit closure/static —
for correctness (anchor shift) and for the recompile discipline above.  **P6 choice to
record, not decide:** cone's validity clip (`cone_beam.py:448`, multiaxis 254) could clip
at `S_real` in GLOBAL terms, making padded slices exactly inert in-kernel for free — vs
keeping kernel validity band-local and masking outside; interacts with cone's
inert-padding vs enlarge-the-volume semantics (§P5 Step 4 notes).

**Forward banding is ACCUMULATION, not concatenation (noted 2026-06-11; design lands at
P6).**  Banding is clean on an operator's OUTPUT side and needs a SUM on its INPUT side.
Back projection bands its *output* — each slice band is computed independently from the
full resident view, so concatenation is valid for every geometry.  Forward projection
bands its *input* — and outside parallel beam a slice band does NOT project to a fixed
row band of a view (cone: pixel-dependent, overlapping row ranges), so view-owners must
ACCUMULATE per-band full-view partials (`view += forward(band_k)` — the in-place donated
accumulator already filed above is exactly this) instead of concatenating row-bands.
The current `_forward_project_all_bands` row-concat (tomography_model ~1341; "each
detector row has a single producer") is the parallel-beam slice↔row identity at work —
keep concat as parallel beam's assembly (smaller transients, no add pass) and make the
band assembly per-geometry at the port (concat-by-row-range | accumulate).  Adjointness
holds either way (forward = Σ_b P_b ⊣ back = stack of P_bᵀ), so the round-trip gate is
unaffected.  Padding interaction: under accumulation, zero padded slices contribute zero
automatically — cone needs NO row padding and gets forward inertness free; the Stage-1
view mask is applied POST-assembly per view-owner precisely so it survives this
concat→accumulate change.

### P4 — VCD on placements (Phase E)  ✅ DONE — multi-GPU validated (1–4 GPU); sharded-memory leak fixed & validated for all paths (const, non-const weights, positivity)
- [x] Entry/exit placements for sinogram (view) / recon (slice) / weights (view) via
  `to_sino`/`to_recon` in `vcd_recon` (replace the `device_put(..., main/sinogram_device)`
  gather hazards).
- [x] Halo exchange via `_extract_halos` in the new `_qggmrf_prior_sharded` — the
  recon-domain analogue of the sharded projectors (per-slice-owner local prior +
  `assemble_sharded`; halo-aware `qggmrf_*` ported from research, default `None` =
  reflected BC = bit-exact).
- [x] Default init: `direct_recon(sharded sinogram)` → slice-sharded init (match-input),
  `forward_project(init)` → view-sharded error sino; stays sharded into the loop.
- [x] No accidental gather/re-shard inside the loop body: recon-domain gathers use a
  mesh-replicated `recon_indices`; the cross-mesh line-search `alpha` is reduced to a
  host float (the two multi-device bugs the tests caught).  Audited by a test that
  `vcd_recon`'s pre-exit-gather return stays slice-sharded across all devices.
- Gate: full recon trivial **bit-exact**; 2/4/8-dev **NRMSE ~6e-7** vs single-device;
  prior trivial bit-exact + 2/4/8-dev float-match.  `tests/sharding/test_vcd_sharded.py`.
- [x] **GPU multi-device scaling validated, 1–4 GPU** (H100, `vcd_recon_gpu.yaml`, 10
  iters): 1008³ **1d→4d = 4.49× — super-linear** (bandwidth-bound op + smaller per-device
  working set; 2→4d = 2.16×); memory shards cleanly (1d 55.5 → 4d 12.7 GB/dev);
  correctness vs prerelease **8.79e-7**.  504³ is the size-floor (4d 1.92×, too small to
  amortize cross-NUMA).  The memory fix ENABLED the 1008³/1d measurement (was OOM).
  Prior-opt A (stage halos once/pass) + the on-device `alpha` replication (zero
  per-subset host syncs) remain in place.
- **Sharded-memory leak — DIAGNOSED & FIXED (the single-device / 1-device-mesh
  no-regression gate).**  Symptom: the sharded path's peak grew ~3.3× the no-mesh peak
  at 504³ (24.6 vs 7.4 GB) and OOM'd at 1008³, while the **default no-mesh path is
  fine** (equal-or-better than prerelease).
  - *Root cause:* **jax keeps sharded (`NamedSharding`) arrays in internal reference
    cycles** (an `ArrayImpl`'s `__dict__` holds its sharding/buffers and back-refs), so
    they are freed only by the **cyclic GC**, not by refcount.  The VCD subset loop
    updated the view-sharded **error sinogram out of place** every subset
    (`error_sinogram = error_sinogram - alpha*delta_sinogram`), allocating a fresh full
    sinogram each time; the stale ones (and the transient `delta_sinogram`) piled up —
    **one full view-sharded sinogram per subset** — until GC, so peak scaled with the
    number of subset-updates (subsets × passes).  Diagnosis was empirical: an
    instrumented memjump trace showed the `P('devices',None,None)` (view-sharded sino)
    count climbing 1/subset, `live_end` only dropping to 512 MB *after* an explicit
    `gc.collect()`, and `gc.get_referrers` naming the holding `ArrayImpl.__dict__`.
    The **recon side never leaked** because `update_recon` already updates `flat_recon`
    **in place via buffer donation** — that asymmetry was the tell.
  - *Fix (mesh-guarded; single-device path untouched):* new module-level
    `update_error_sinogram` does `error_sinogram - alpha*delta_sinogram` under
    `@partial(jax.jit, donate_argnames='error_sinogram')` so XLA updates the error
    sinogram **in place** (mirrors `update_recon`); the sharded branch of
    `vcd_subset_updater` releases the constant-weights alias (so the buffer is
    donatable), calls it, then **explicitly `.delete()`s the transient `delta_sinogram`**
    (also cycle-held) after a one-line `block_until_ready` for async safety.  Net:
    **peak flat at the `ns=1` floor (504³/5-iter: 25.8 → 6.9 GB), no per-subset growth,
    time unchanged (~47 s), no donation warning.**  Also lifts per-device memory at
    `n_dev>1` (raises max-recon-per-GPU), not just the degenerate 1-device mesh.
  - *Validated:* local correctness GREEN — `tests/sharding/test_vcd_sharded.py` 8/8
    (incl. trivial **bit-exact**; the alpha scale is kept eager so the donated update is a
    pure subtract — folding it in would emit an FMA and break bit-exactness), single-device
    `test_vcd` regression, single-device-vs-prerelease **5.3e-8 unchanged**.  **GPU 1-device
    (H100):** 504³/5-iter mesh peak **25.8 → 6.9 GB** (= no-mesh 7.8), 1008³/5-iter mesh
    **OOM → 56 GB, completes** (no-mesh 53) — **1-GPU no-regression gate MET**; `live_end`
    flat (no accumulation).  Minor: 1008³ mesh ~+6% over no-mesh (per-band/assemble
    transients; benign, shards away at n_dev>1).
  - *Consolidated cleanup (done):* the per-subset transient frees are now ONE cleanup
    section at the end of `vcd_subset_updater` (mesh only): after a single
    `block_until_ready` on the returned state, delete the scaled `delta_sinogram` and —
    for **non-constant weights** — the `weighted_error_sinogram` product.  Correctness
    coverage added (`test_vcd_sharded.py`, +4: non-const & positivity × trivial-bit-exact /
    2-4-8-dev NRMSE); full `tests/sharding/` + `tests/test_vcd.py` **77 passed**.
  - *Non-constant weights & positivity — VALIDATED (bounded).*  GPU (504³/5-iter/1-device-mesh):
    **non-const peak 7.9 GB**, **positivity peak 7.3 GB** — both ≈ const 6.9 GB, flat, no leak.
    The `weighted_error_sinogram.delete()` bounds non-const; positivity needs **no** extra
    delete (its `delta_sinogram` recompute and stranded original are forward-projection
    (`assemble_sharded`) outputs that free on refcount — the cycle is only in *eager-op*
    outputs).  (An earlier 33.4 GB "non-const leak" was a **stale GPU build** running the
    pre-fix binary — a fresh `pip install -e .` resolved it; ruler-before-code.  The
    temporary `_memprobe` localizer added to chase it has been removed.)
  - *Refactor (done):* `TomographyModel.is_sharded` property is the single source of
    truth for the sharded code path (replaced 21 `self.mesh is None/not None` checks; the
    body changes in one place at the mesh→placement migration, and this is what retires
    at unification).
- **Also pending:** hybrid timing for the `qggmrf_..._transfer` variant (deferred
  Q3); prox-map prior under sharding (untouched).

### P5 — Device-config UX (can land in parallel once P1 fixes what gets configured)

**STATUS (2026-06-09): mostly DONE; only divisibility *padding* (Step 4) + the device-aware geometry
scaling remain.**  Method renamed `set_devices_and_batch_sizes` → `set_devices` (it no longer sets
batch sizes).  **(Step 4 design DECIDED 2026-06-11 — see §"P5 Step 4 — divisibility padding" below.)**

- [x] `configure_devices(devices=None|int|list)` user-facing (None→auto pick-N, int→count,
  list→indices/devices); resolves then delegates to `configure_sharding` (the user-selected/pinned
  flag is `_sharding_configured`).
- [x] **Automatic selection in `set_devices`:** N = largest count dividing both sharded axes
  (`_auto_device_count`, gcd of the axes).  **No per-device work floor** (RESOLVED 2026-06-09 from
  the GPU sweeps — over-sharding a small problem is only a mild overhead).  Re-validate on a geometry
  change: a user-selected config that no longer fits **warns** (the hard error is at the shard op —
  see Divisibility); auto re-picks a compatible N on every recompile.
- [x] `use_gpu='sharded'` internal state; `'automatic'` **shards by default** on a multi-GPU box
  (GPU-validated).  GPU-only by default; CPU auto-sharding is the opt-in `_auto_shard_cpu` (see the
  CPU-cluster adjacent task).
- **Divisibility:** [x] warn-with-fix-instructions, **order-independent** — `configure_sharding` and a
  user-selected geometry change only **warn**; the hard, clear error is raised at the shard chokepoint
  `_shard_on_axis` (covers every entry point), so a config fixed up before the recon works regardless
  of call order.  [ ] **(Step 4, remaining)** padding to *use* a non-dividing N — **design DECIDED
  2026-06-11**, staged plan in §"P5 Step 4 — divisibility padding" below (the `weights=0`-mask and
  "problem-owned shape" ideas recorded here earlier are superseded there: the mask moved to the
  sharded forward output, and the principle is refined to "padding must be exactly inert").
- [x] **Always-on "devices in use + why" report** at recon time (`_device_report`:
  `N x PLATFORM [(sharded)]` + a "why N" note when auto left GPUs idle).
- [ ] `auto_set_recon_geometry` / `scale_recon_shape` made device-aware — **deferred to P6/geometry**
  (mainly a cone-beam lever; parallel-beam sharded axes are sinogram-side).

### P5 Step 4 — divisibility padding (design DECIDED 2026-06-11; staged implementation)

**Goal:** auto uses ALL available devices; a non-dividing axis is padded, and the padding is
**exactly inert** — results independent of device count and pad amount.  This refines the earlier
"pad to a shape the problem owns" principle: *pad however the devices need, but make the padding
exactly inert* (inertness is the reproducibility that principle was protecting).

**Core decisions (each worked through against the code, 2026-06-10/11 sessions):**
- **Masked forward output, NOT `weights=0`.**  The invariant: *padded views/rows/slices of every
  array are identically zero, always* — zero-fill at entry + a validity mask multiplied into the
  sharded forward-projection output (ONE site, applied post-assembly per view-owner; skipped when
  nothing is padded).  Then every downstream consumer — weighted or unweighted — is automatically
  clean, and the donated-FMA error-sino update preserves the invariant.  `weights=0` was REJECTED:
  it would force materializing a full weights array (the const-weights path keeps `weights` as the
  scalar `1`, tomography_model ~2317 — a 32 GB-scale regression) and still miss the *unweighted*
  reductions (`get_forward_model_loss`'s `mean` ~2785; the const-weights initial alpha ~2360).
- **Contract: explicit `output_sharded=False` kwarg; match-input RETIRED (amends O2/F2).**
  Default = a plain array in the problem's REAL shape (gather + crop); `True` = the internal device
  form (sharded, possibly padded).  Inputs stay auto-detected (plain → pad+shard at entry; prepared
  → used as-is).  One rule, every user-facing method: *padded shapes never escape unless explicitly
  requested.*  (`True` on an unsharded model just returns the trivial-placement form — no error.)
- **Params answer "what is the problem?"; placements answer "what is on the devices?".**
  `get_params('sinogram_shape')`/`recon_shape` ALWAYS return real shapes — save/load, projector
  geometry (`projector_params` is baked from get_params), FBP `π/num_views`, and metric
  normalizations stay correct for free.  The **placements own the padded global shape, each shard's
  global index range (its `g0`), and the real counts**; sharded plumbing asks the placement, never
  get_params.  **Every padding mask is the one predicate `k_global < real_count`** evaluated over a
  shard's global range — geometry-independent, correct by construction.
- **`prepare_sino_for_devices(sinogram, weights=None)`** — PUBLIC model method (padding is also
  applied automatically at entry for plain inputs).  Per-shard host→device streaming: `device_put`
  each device's host slice; the last shard transfers its partial slice into a zero-initialized
  device buffer, so **zero-padding is free and no padded host copy ever exists** (only transient =
  one shard on one device).  Two-axis-aware from day one (views in Stage 1, rows in Stage 2, same
  API).  Also the natural future hook for streaming from disk (per-shard memmap reads).  A stale
  prepared array after a device-config change is caught by the `_shard_on_axis` hard error (message
  to recognize the case: "re-run prepare_sino_for_devices").

**Stage 0 — the contract change (no padding yet; lands alone, suite-gated):**
`output_sharded` on `forward_project` (~985) / `back_project` (~1031) / `direct_recon` (~2081) /
`fbp_recon` / `fbp_filter` / `direct_filter` / `recon` (~2210) / `prox_map`; reroute internal
callers to `True` (vcd_recon's `direct_recon` init ~2336 + `forward_project` ~2355; fbp_recon's
internal filter call, parallel_beam ~545; `vcls.py:167` gets the explicit kwarg); update the
match-input tests; amend O2/F2 here and in §Decisions.

**Stage 1 — views:**
- Pad metadata derived at layout time (`set_devices`/`_apply_mesh`, re-derived every recompile):
  padded view count, padded `view_params_array` (replicate the last entry — values irrelevant,
  output masked; shape-compatible with the settable-view-params task), per-device view mask.  All
  inactive/None when views divide — the common case carries zero new work.
- Pad-aware `_shard_sinogram` entry; `_shard_on_axis`'s hard error stays as the backstop.
- The forward view-mask (post-assembly per view-owner in `_sparse_forward_project_sharded`
  ~1248–1306 — placed AFTER assembly so it survives the P6 concat→accumulate change).
- Real-count corrections: `π/num_views` (parallel_beam ~455), `get_forward_model_loss` mean +
  `total_loss`'s `sinogram.size` (~2453); the Hessian's const-weights `ones_like` transient
  (~2392) gets the view mask.
- Auto: `_auto_device_count` ignores views (largest N ≤ available dividing `num_slices`); the
  views divisibility WARN retires; `_device_report` notes "views padded V→V_pad".
- **Shape-source audit:** classify every `sinogram_shape`/`num_views` read as (a) problem-shape —
  params, keep (kernel geometry, FBP scale, metrics); (b) device-shape — switch to the placement
  (sharded dispatch ~1228, `assemble_sharded` global shape ~1304–1306, the weights/entry shape
  checks ~1793/~2549); (c) input-length-used-as-coordinate-anchor — the P6 kernel sites (see the
  anchor rule in the (g0, L) design note).  A get_params shape in sharded plumbing is a code smell.
- Tests: the zero-invariant through a full recon; prime-`num_views` recon + fbp_recon vs
  single-device at the 1e-4 gate; adjoint round-trip with padding; entry-streaming shard layout +
  zero tail (sino and weights); `_auto_device_count` unit test; fm_rmse vs an unpadded reference.

**Stage 2 — slices (ParallelBeam: slices + det rows pad TOGETHER — the kernel-shape tie; equality
is enforced at parallel_beam ~124, and the slice↔row map is index-identity, so padding both ends
by the same count is exact — no center arithmetic on that axis):**

*AMENDED 2026-06-11b (kernel proposal reviewed with Greg; designed cone-ready, not parallel-only):*
- **qGGMRF kernel = interface delta-mask** (the careful piece, approved).  In
  `qggmrf_grad_and_hessian_per_cylinder`, `delta[j]` (length L+1) is the difference across the
  interface between global slices (g0+j−1, g0+j); reflected BC is *already implemented as*
  `delta = 0` at an edge interface, so a multiplicative mask **is** reflected BC relocated to an
  arbitrary interface: `interface_mask[j] = (g0 + j < S_real)` (valid iff the interface's RIGHT
  slice is real — the one predicate again).  Real-slice outputs are bit-exact vs the unpadded
  cylinder (same values, same op order); handles mid-shard, shard-aligned, and fully-padded-shard
  cases with no special cases; composes with the halos (which stay).  New optional
  `interface_mask=None` arg threaded through `qggmrf_gradient_and_hessian_at_indices`
  (vmap in_axes None — one (L+1,) mask shared per shard); masks built from
  `padded_shard_ranges()` and CACHED per recompile (per the staged-halos lesson, never a
  per-subset device_put).  When padded, every shard gets a mask (all-ones for fully-real shards,
  uniform traces); unpadded → None everywhere (common case zero-cost).
- **Prior Hessian at padded slices stays POSITIVE** (deliberate refinement of the earlier
  "zero prior Hessian + where-guard" wording): masked deltas contribute `b_tilde(0)` to the
  Hessian and 0 to the gradient, so the VCD denominator at a padded slice is `0 + positive` and
  `delta = −0/positive = −0.0` **exactly, by construction** — no 0/0 is ever manufactured, hence
  **no `jnp.where` guard, no division mask, no update_recon mask, no init-zeroing site** (Greg:
  `where` is slow on large arrays; here nothing is needed at all).
- **Inertness is a projector POSTCONDITION** (Greg's reframe): `_mask_padded_slices` on the
  sharded **back-projector output**, post-assembly per slice-owner — the exact mirror of Stage 1's
  `_mask_padded_views` on the forward output.  Each sharded projector guarantees its own output is
  inert in the padded region; `forward_grad`, `compute_hessian_diagonal`, and the direct/fbp init
  are all back-projections, so **the VCD loop needs no edits**.  Under parallel beam the
  slice↔row identity makes this defense (padded rows/slices are structurally zero); under cone
  (P6) it is LOAD-BEARING — back projection bleeds real rows into padded slices, and this one
  site is what zeroes them.  No forward-input mask: recon padded slices stay exactly zero by
  induction (entry zero-fill + masked back projections + kernel-masked prior ⇒ updates add
  exactly ±0.0, no drift mechanism); enforced by an exact-zero invariant test, not a hot-path op.
- Padded rows zero-data / zero-weight from the entry streaming (two-axis `prepare_sino_for_devices`;
  rows pad to match S_pad for kernel-shape compatibility, NOT for divisibility — the row axis is
  unsharded).  Real slices' footprint bleed into padded rows is influence-free (w=0), and the
  measured data never had those rows — the real-slice subproblem is EXACTLY the unpadded one
  (survives `delta_voxel != delta_pixel`).  `_sino_ones_device_form` predicate extends to rows;
  real-count normalizations extend from V_real·R·C to V_real·R_real·C.
- **Entry shape tightening (Stage-1 carry, folded in):** entry requires input shape == params real
  shape OR == the placement's padded device shape, with an error naming
  `prepare_sino_for_devices` — closes the silent-wrong-results corner where a stale prepared
  array happens to divide a new device count.
- **Module move:** `_extract_halos` / `_stage_halos` / `_qggmrf_prior_sharded` move to
  `qggmrf.py` as module functions with EXPLICIT arguments (placement, shard axis, staged halos,
  interface masks — no model state); thin model wrappers remain.  Import direction is clean
  (qggmrf ← _sharding; no cycle).
- Auto ignores slices too → auto = all available devices, EXCEPT skip any N whose last shard
  would be entirely padded (fall to the next smaller N); the "why N" report note retires for
  ParallelBeam; report adds "slices padded S→S_pad".  (The broader choose-N-vs-communication
  policy discussion belongs to P6.)
- Tests: kernel unit (masked kernel on padded cylinder vs unmasked on truncated real cylinder —
  real slices bit-exact on CPU, padded grad exactly 0, padded Hessian finite-positive);
  N-independence (prime num_slices recon at 2/3/4 dev vs single device, 1e-4); exact-zero
  invariant (padded recon slices == 0.0 after a full VCD recon); no-NaN check; padded-recon ==
  unpadded-recon (the reflected-BC check).

**GPU items (Greg, cluster):** partial-shard assembly over the d2d path; perf sanity that padding
+ mask are in the noise; host-RSS check that `prepare_sino_for_devices` makes no host copy.

**Notes:** `compute_hessian_diagonal`'s legacy `output_device` kwarg is reconciled with
`output_sharded` at P6, not now; ~~`prox_map` stays main_device-pinned~~ **prox_map is
placement-correct as of 2026-06-11b** (prox_input routes through the recon entry placement and the
prox branch uses the replicated recon_indices — the prox prior is pointwise, so no halos; found by
a GPU failure under default auto-sharding); cone/translation/multiaxis pick all of this up
at the P6 port — cone pads slices WITHOUT row padding (its rows aren't tied to slices, and under
forward-accumulation zero padded slices are inert for free) and chooses there between
inert-padding and enlarge-the-volume semantics for the padded slices (the Stage-2 design above is
already cone-ready: the back-projector output mask is the load-bearing site under cone).

### P6 — Port + retire
- Port the sinogram transpose pattern from `back_project_one_view_to_pixel_batch` 
  and `forward_project_pixel_batch_to_one_view` in parallel beam to cone / 
  translation / multiaxis geometries.
- Port the placement/movement path to cone / translation / multiaxis geometries
  (each implements the `(g0, L)` slice-band projector interface from the P3 design
  note).
- Retire `main_device` / `sinogram_device` entirely (the trivial-sharding
  unification); remove the old single-device and slice-band code paths.
- With the single-device path unified onto band-looping, **delete the per-geometry
  `entries_per_cylinder_batch` slice-batching** (subsumed by the outer band loop)
  and switch the jittable single-device / forward assembly to an in-place
  `dynamic_update_slice` accumulator with `donate_argnums` (see the P3 design note).
- **(2026-06-11, from the P5 Step 4 design)** The geometry port also picks up: the
  **anchor rule** (kernel centers from problem shapes + global indices, not input
  lengths — bit-exact mechanical fix; sites listed in the design-note amendment) and
  **per-geometry forward-band assembly** (parallel = row-concat, z-based = accumulate;
  see "Forward banding is ACCUMULATION" in the design note); cone decides inert-padding
  vs enlarge-the-volume semantics for padded slices, and whether its validity clip goes
  global-`S_real` (in-kernel inertness) or stays band-local; reconcile
  `compute_hessian_diagonal`'s legacy `output_device` with `output_sharded`.
- **(2026-06-11)** Delete `initialize_recon`'s early `device_put` block (the
  `_committed_elsewhere` guard): it survives only for the unported geometries' reliance
  on pre-committed `main_device`/`sinogram_device` arrays.  After the port, inputs stay
  on the HOST through validation/regularization (which read them host-side anyway) and
  are committed exactly once, at the entry placement (`_shard_sinogram`/`_shard_recon`);
  `prepare_sino_for_devices` is the explicit early-placement opt-in.  One commitment
  point, no double placement.

**STATUS (2026-06-08): step 4 IMPLEMENTED as "2-Keep" — CPU-green, GPU re-validation pending.**
Change 1 (note 2: fold `alpha` into the donated FMA `update_error_sinogram`, drop the
`scaled_delta` transient + its `.delete()`; const-weights cleanup section now empty/skipped) and
Change 2 (ParallelBeam auto-defaults the *homogeneous* single-device case to a trivial 1-device
mesh via the new `_supports_sharding()` hook + `_sharding_configured` flag) both landed.  The
**heterogeneous recon-CPU/sino-GPU 'sinograms' mode stays on the legacy single-device branch**
(deliberate 2-Keep scope; a CPU+GPU pair is not one mesh), so `_transfer` is untouched for now.
The flip exposed + fixed a partial-view gap (`view_indices` routes to single-device on a trivial
mesh, raises only on a multi-device mesh).  No-mesh-ParallelBeam tests retired/updated.
**GPU re-validation DONE** (Greg): Change 1 + Change 2 within noise; 8-GPU 1024³/1800³/2048³
scaling at least as predicted.  **note 1 is informational** (read the baseline at ~1e-4, no code
change).  **2-Drop DECIDED — drop hybrid** (see §Decisions "Hybrid (2-Drop)" below; execution is
a P6 task).  See `sharding_status.md` HANDOFF (2026-06-08).

**Unification decision (the "step 4" trivial-placement question) — RESOLVED: Option B,
one always-on placement path.**  A trivial 1-device placement resolves to a 1-device
`NamedSharding` (not a `SingleDeviceSharding`), so the single-device case is just the
degenerate sharded case and the dual `is_sharded` branches collapse into one path.
Rationale: "no literal regression from prerelease" is the same over-strict invariant as
"bit-exact" — we accept ~1 ULP (iteratively amplified to ≤1e-4) differences, not byte
identity.  The per-subset `block_until_ready` + `.delete()` cleanup then runs single-
device too, but it only stalls *host dispatch*, which is in the noise on production-size
recons where projection compute dominates; and making the transient cleanup mandatory
everywhere means one memory strategy, with no single-device-only leak able to slip past.
Carry these into the implementation:
  1. **Swap the guard, don't drop it.**  Losing the "verbatim prerelease body" safety net
     means single-device correctness rides on the donation + eager-scale path, so replace
     the bit-exact-vs-prerelease check with a **tight `allclose`** single-device-vs-
     prerelease guard (repurpose `vcd_single_device_baseline.py` to compare at ~1e-4, not
     exact).
  2. **Unlocked simplification.**  The "keep alpha out of the donated jit to avoid an FMA"
     decision existed *only* to preserve trivial-mesh bit-exactness.  Once ~1 ULP is
     accepted, fold `error_sinogram ← error_sinogram − alpha·delta` into a **single
     donated FMA jit**, deleting the separate eager `scaled_delta` transient *and* its
     `.delete()` — the unified update gets simpler than either current path.
  3. **Heterogeneous recon-CPU / sino-GPU is the one behavior change.**  Option B routes
     the prior through `_qggmrf_prior_sharded` (two trivial placements) instead of the
     `qggmrf_..._transfer` fast path, whose *timing* is an existing deferred open question
     — measure before deleting `_transfer`.  **And: if this heterogeneous case becomes
     unwieldy** (it forces multiple code paths or other special cases) **we drop it.**  It
     is only a stop-gap for doing larger recons on a *single* GPU; the real target is large
     recons on *multiple* GPUs, and that path must not be complicated to keep the stop-gap
     alive.

**Retirement marker convention.**  Code, tests, or docs that exist *only* while the
legacy single-device path coexists with the sharded/placement path — and should be
removed once everything runs on placements — are tagged with the fixed, greppable
phrase **`RETIRE-AFTER-SHARDING`** (general stem `RETIRE-AFTER` for any milestone-
gated retirement).  Before doing the P6 retirement, `grep -rn "RETIRE-AFTER-SHARDING"`
to find every site.  Currently tagged:
  - the 7 `tests/sharding/` *trivial-mesh-vs-single-device* comparison tests
    (`test_trivial_*bit_exact*` in back/forward projection, fbp_recon, and VCD).
    These were exact-equality checks; **relaxed to a tight `allclose`** (single-shot
    `rtol/atol=1e-5`; iterative VCD recon `1e-4`, matching the GPU-proven multi-device
    sweep sibling) because the banded sharded path reorders non-associative FP sums,
    so on **GPU** the 1-device mesh differs from the monolithic single-device kernel
    by ~1 ULP (CPU compiles both identically and stays exact).  Once the legacy path
    is gone there is a single path and nothing to compare, so they retire.
  - the `is_sharded` property and its branch sites (already noted at its definition).

---

## Migration state & branch-retirement map (2026-06-08)

**Three device representations still coexist** (mid-migration); the placements are the
intended survivor:

| Representation | Role | Fate |
|---|---|---|
| `main_device` / `sinogram_device` | scalar single-device pins (pre-sharding) | retire at P6 |
| `mesh` / `shard_devices` | device set + the `is_sharded` gate | retire at P6 (folds into placements) |
| `recon_placement` / `sino_placement` | each owns its mesh + sharded axis | **survivor** — single source of device layout |

After 2-Keep, **ParallelBeam-homogeneous** auto-populates `mesh` + placements (trivial 1-device
mesh) and runs the placement path; everything else still uses the legacy representations.

**Branch-retirement map** — the live legacy paths and the *gate* that retires each (they differ):

| Live branch / artifact | Kept alive by | Retires when |
|---|---|---|
| `is_sharded` **else**-branches (VCD, projector routers) | unported geometries **+** ParallelBeam-heterogeneous | geometries ported (P6) **and** hybrid dropped |
| `RETIRE-AFTER-SHARDING` tests | the legacy single-device path | that path is gone (P6) |
| multi-device `view_indices` `NotImplementedError` | the `view_indices` mechanism | **settable view params lands** (§Adjacent tasks) |
| `self.mesh` + `NamedSharding(self.mesh, …)` sites | mesh-as-gate representation | placements become sole layout (P6) |
| `_transfer`, `'sinograms'` mode, auto-default `else`-branch | hybrid existing | **hybrid drop** (decided → P6) |

**Footgun to carry into P6: `is_sharded` and `n_devices > 1` have DECOUPLED.**  Pre-2-Keep they
were synonyms; now `is_sharded` is ~always True for ParallelBeam, so every `is_sharded` site must
be re-read for *which question it asks* — "do I have a placement?" (almost always yes) vs "do I
have ≥2 physical devices?" (`len(shard_devices) > 1`).  The `view_indices` fix was the first place
this bit (it tests `len(shard_devices) > 1`, not `is_sharded`); it won't be the last.

**Sequencing.**  Settable-view-params (§Adjacent) and the geometry port both *feed* the deletions,
so the deletions (legacy branches, `self.mesh`, `RETIRE-AFTER` tests, hybrid) come **last** in P6.
**P5** (device-config UX + device-count-aware `set_devices_and_batch_sizes` + divisibility) is
largely **parallelizable** alongside.

---

## Decisions & open questions

- **O1 (heterogeneous placement) — reframed/resolved.**  recon-CPU / sino-GPU via
  two *separate* trivial placements; the reverse (sino-CPU / recon-GPU) is **out
  of scope**.
- **O2 (return contract) — resolved (F2): match-input — AMENDED 2026-06-11 (P5 Step 4):
  match-input is RETIRED in favor of an explicit `output_sharded=False` kwarg on
  user-facing methods.**  Default `False` = a plain array in the problem's REAL shape
  (gather + crop); `True` = the internal device form (sharded, possibly padded).  Inputs
  stay auto-detected.  Why: padding makes sharded sino-domain shapes non-problem shapes
  (`V_pad`), so under match-input the *output shape* would depend on the *input
  placement*; an explicit argument beats inference.  Internal callers (vcd_recon's init
  chain, fbp_recon's filter, PnP chains) pass `True` and stay on-device.  Internal
  methods remain sharded-only, unchanged.
- **Hybrid (2-Drop) — RESOLVED (2026-06-08): DROP hybrid** (recon-CPU/sino-GPU
  'sinograms' mode; supersedes O1's "keep via two trivial placements").  Analytic
  envelope `scaling_tests/archive/analytic_hybrid_vs_full_envelope.py`: per-device peak
  ≈ 7 recon-volumes + 8 sino-volumes (≈15 total, matches the 504³ 6.9 GB anchor); hybrid
  offloads only the recon side → **+19% in N**.  A 2nd GPU gives **+26%** (2^⅓), and
  slice-subset stitching (`split_sino_recon` top/bottom for cone; overlapping slice
  subsets for parallel beam) extends ~unboundedly on one GPU and composes with sharding —
  so the alternatives all meet/beat +19% without the legacy path.  Bonus: hybrid is the
  only `main_device != sinogram_device` config, so dropping it removes `_transfer`, the
  `'sinograms'` branch in `set_devices_and_batch_sizes`, and the auto-default `else`-branch,
  and lets one `self.mesh` represent every ParallelBeam config.  Tradeoff: hybrid was exact,
  stitching is approximate at the seams (good-enough for single-GPU).  **Execution = P6.**
- **O4 (divisibility):** warn-with-instructions + `prepare_sino_for_devices` +
  pick-N; pad/crop only on the clean slice/row axis.  **Non-divisible shard axes
  (design, 2026-06-08):** *Views* — pad to a multiple of N and zero the padded views; the
  pad mask is just `weights = 0` (the weighted error sinogram and line-search sums drop
  them, and back projection is a reduction over views so zero-weight views are inert).  The
  same per-shard view masking gives **sharded view-selection** (keep the full view-sharded
  layout, zero the excluded views — no gather, no divisibility break), useful for
  view-iterating algorithms like `vcls` on multi-GPU (each device works one local view,
  masks the rest); it does NOT save compute, so single-device subsetting stays for the
  compute-saving case.  **Superseded for `vcls` by settable view parameters** (a small-view
  model + `set_view_parameters`; see §Adjacent tasks) — masking is only a fallback if a
  *full*-view model must select.  *Slices* — harder (the prior couples neighboring slices, and for
  parallel/cone the slice axis is tied to detector rows).  Principle: **pad to a shape the
  *problem* owns (reproducible), never to a multiple of the *device count* (N-dependent
  result).**  Options: pick N from divisors of `num_slices`; or problem-level pad + a slice
  mask the prior respects (reflected BC at the last real slice, padded slices inert).
  **SUPERSEDED 2026-06-11 by §P5 Step 4:** the views mask moved from `weights=0` to the
  *sharded forward output* (one site; `weights=0` would materialize a full weights array and
  still miss the unweighted reductions), slices resolved as forced-zero + a qGGMRF boundary
  mask + zero-weight padded rows, and the principle is refined to **"padding must be exactly
  inert"** — once inert, padding to a multiple of N is harmless because the result is provably
  N-independent (inertness is what "problem-owned shape" was protecting).
- **Band-vs-pixel criterion (P2) — RESOLVED (2026-06-03): KEEP BAND.**  The
  side-by-side showed pixel's peak memory floored at ~1.7–2.7× band for only
  ~11–16% less time, so band stayed and the pixel path was removed.
- **Band length / budget-driven sizing — RESOLVED (2026-06-09): KEEP the
  `slices_per_dev/n_dev²` default; do NOT build budget-driven band sizing.**  A
  multi-device band sweep (H100×4; `sparse_back_project_band_sweep.py`, band ∈
  {auto, slices_per_dev//{1,2,4}}, n_dev=1/2/4, 512³/1024³) showed **time is flat
  across band** (bigger bands if anything slightly slower) while **memory strongly
  favors the smaller auto bands** (1024³/n=4: full-shard band = 2.08× auto's peak
  for identical time).  Multi-device scaling is band-independent and near-linear
  (1024³ auto 2.10×/3.73× at n=1/2/4).  So small bands are a free memory win and
  the hypothesis that bigger bands would relieve the multi-device time wall is
  refuted ⇒ the earlier "make band sizing memory-budget-driven" idea is dropped,
  and any P5 memory estimate is for device-COUNT selection only.  (A CPU hint that
  bigger bands were faster at 256³/n=2 did not replicate on GPU.)
- **Open:** exact `B_p` default / floor (set conservatively, refine by sweep);
  `configure_devices` precedence vs `use_gpu`; how `prepare_sino_for_devices`
  returns the crop spec to undo padding on the recon.

---

## What survives unchanged
- F2 **match-input** contract on `direct_recon` / `fbp_recon` / `back_project`.
- The `direct_recon` scaling driver + baseline (re-run against new internals).
- F1 FBP filter (`apply_row_filter`) — already view-sharded, no cylinder motion.
- Cross-cutting principles, hardware facts, and the completed-work record in
  `sharding_implementation_plan.md`.

---

## Adjacent tasks (not gating the P-phases)

### Settable view parameters — retire `view_indices` (standalone, adjacent to P6)

**Decision (2026-06-08): DO IT (option "b"), standalone for now.**  Replace the
`view_indices` mechanism with **settable view parameters**.  The only nontrivial user of
`view_indices` is `vcls` (per-view back projection, `view_indices=[i]` to slice the frozen
angle table) — a hack: the natural op "project at angle θ" is expressed as "index my baked
angles at i."

**Mechanism.**  Today `create_projectors` (`projectors.py:57`) **closes over**
`view_params_array`, baking it into the jitted projectors, so changing it forces a recompile;
`view_indices` is a runtime arg that indexes the baked array.  The *kernels* already take
`single_view_params` as an argument, so only the whole-array closure capture bakes it.  Lift
`view_params_array` from a **closed-over constant to a runtime argument** of the jitted
projectors: shape is static, only values change → **no recompile across parameter changes**.
Expose via a dedicated **`set_view_parameters()`** (geometry-general via `view_params_name`;
assert `num_views` unchanged — a *count* change is a geometry change, route through
`set_params`).  NOT `set_params`, whose `recompile_flag` path is exactly what we avoid.

**Why (beyond de-hacking `vcls`).**
- Retires the multi-device sharded `view_indices` `NotImplementedError` branch (a view *subset*
  breaks the equal view-shard; with settable params there is no subset).  So **P6 can delete
  that branch** instead of building masked subsetting.
- Enables a clean **multi-GPU `vcls`**: a `num_views = n_dev` model, view-sharded one-per-device,
  `set_view_parameters` to the `n` angles each step — no masking, no wasted compute.
- Generalizes to **motion correction / variable trajectory / streaming acquisition** (on the
  radar; not the near-term driver).

**Risks.**  Confirm no *other* angle-derived quantity is baked per geometry (`geometry_params`
— cone may carry per-view source/detector geometry; FBP weights).  Dynamic-angle constant-fold
perf cost is O(num_views) vs O(N⁴) projection → negligible (measure before keeping any
baked/dynamic dual mode).  Threads the array through every projector entry point (incl. sharded
paths) across all geometries — localized to `projectors.py` + the model API, but not a one-liner.

**Sequencing.**  Standalone; land before/independent of P6; supersedes the O4 `weights=0`
view-masking idea for `vcls`.

**Intended `set_view_parameters` docstring (capture the per-call form here for implementation):**
```text
Replace the view-dependent parameters (angles for parallel/cone beam, translation vectors for
translation, etc.) used by the projectors, WITHOUT rebuilding them.

The view parameters are a runtime input to the jitted projectors (not a baked constant), so
this is a cheap value update: it does NOT recompile, provided the number of views is unchanged
(a change in view count is a geometry change -- use set_params for that). Use it to vary the
acquisition geometry on the fly -- e.g. vcls iterating one view at a time, or motion correction
perturbing per-view angles between iterations.

Args:
    view_params: array shaped like the model's current view-parameter array (first axis =
        num_views); only the values may differ, not the shape.

Per-call alternative (not yet implemented): the same flexibility can be exposed by accepting
view_params as an optional argument to forward_project / back_project / the sparse_* projectors,
so a caller can vary the parameters per call without mutating model state. That form is more
functional (no "is my state current?" question) but changes every projector signature; the
setter is the minimal-churn entry point, and the two can coexist (the setter just updates the
stored array the projectors read).
```

### Seed test RNGs — solidify the suite (standalone)

Several tests use **unseeded `np.random`** with tight default `allclose`, so they flake when a
value lands near zero (confirmed: `tests/test_qggmrf.py::test_loss_and_gradient` passes 5/5 in
isolation but intermittently fails in the full suite — it builds no model, so it is unrelated to
the sharding work).  Sweep the test tree for unseeded RNG and **seed deterministically**
(`np.random.default_rng(<fixed>)`, as the sharding suite already does), and/or add a small `atol`
where the comparison is near zero, so the gate is reproducible.  Low priority, high tidiness.

### CPU-cluster auto-sharding — investigate performance + virtual-vs-real-CPU policy (standalone)

**UPDATE 2026-06-11b: `self._auto_shard_cpu` defaults to `True`** — automatic selection now shards
across CPU devices exactly like GPUs (platform-uniform auto policy; a platform-dependent policy is
how "sharded + X" gaps stayed invisible to the CPU suite — the prox bug was exactly that).  The
library's `_device_setup` cap (2 virtual CPU devices) is the conservative perf knob; measured
(M3 Max, 2 devices, 8-iter VCD): 1.30× at 256³ (and better at larger sizes), 0.83× at 128³, 0.64×
at 64³ — a real win where time matters, a few absolute seconds where it does not, and LOWER peak
RSS sharded at small sizes.  Suite cost: the legacy test suite (8 virtual devices from
tests/conftest.py) went 154 s → 290 s (~1.9×, tiny overhead-bound problems) — accepted for the
coverage.  Set the flag `False` to opt out (`test_auto_shards_cpu_by_default` covers default and
opt-out).

Remaining (the original second half):

**To investigate:** measure sharded-recon performance on a real CPU cluster (true cores / sockets,
not virtual devices) — does multi-CPU-device sharding actually speed up the VCD recon (the embarrassingly
parallel `fbp_filter` already scaled near-linearly across virtual CPUs; projectors/VCD scaled less
well), and where is the crossover?  Then **adapt auto-selection to differentiate virtual CPUs from a
real CPU cluster** (e.g. NUMA / socket / true-core topology vs an `XLA_FLAGS`-forced virtual count)
so auto can shard CPU *only when it pays* — promoting `_auto_shard_cpu` from an experimental flag to a
proper, possibly default-on, policy with a clean API (a method or a `use_gpu='automatic'`-style
setting) rather than a raw attribute + `set_devices()` call.

### Sphinx docs: multi-GPU page + alignment review (planned 2026-06-11; implement after P6)

**Vocabulary rule (Greg):** use "sharding" SPARINGLY and introduce it clearly when used — many users
do not know (or want to know) the term.  Speak in user terms: *multiple GPUs increase memory
capacity and reduce reconstruction time*; reserve the internals vocabulary for the
behind-the-scenes section and the developer docs.

**Timing:** write the new page AFTER the P6 geometry port — today it would need a
"parallel-beam-only" caveat in every section, which P6 deletes (writing it twice is waste).  The
user-facing contract it documents (`output_sharded`, `configure_devices`,
`prepare_sino_for_devices`, `device_summary`, invisible padding) is already settled, so nothing
else churns.  The two ACTIVELY-WRONG entries were fixed immediately (2026-06-11, did not wait):
`usr_parameters.rst` `use_gpu` (documented the removed `'sinograms'` value; now
automatic/full/none + request-vs-outcome note + `configure_devices` pointer) and the
`demos_and_faqs.rst` "larger reconstructions" FAQ (described the removed hybrid; now describes
automatic multi-GPU + keeps the subset/stitching guidance + drops the stale "we will investigate
multi-GPU" promise).

**New page `usr_multi_gpu.rst` — "Multi-GPU Reconstruction"** (User Guide toctree).  Outline:
1. *Zero-effort path:* multi-GPU machines divide the reconstruction automatically; nothing in the
   script changes; the recon log / `model.device_summary` reports what was chosen
   (`'4 x GPU (sharded)'`, padding and why-GPUs-idle notes).
2. *Controlling devices:* `configure_devices(None | n | list)` (explicit, pinned) vs the
   `use_gpu` request parameter; one sentence on the experimental CPU opt-in.
3. *Workflow / efficiency tips:* `num_slices` divisibility decides how many GPUs automatic
   selection uses (the view count never constrains it); choose slice counts divisible by the GPU
   count when possible; `prepare_sino_for_devices` to pay the host->device transfer once across
   repeated reconstructions; `output_sharded=True` for on-device chaining (with the device-form
   padded-shape caveat).
4. *Performance expectations:* capacity FIRST (per-GPU memory ~1/N), speed second; near-linear
   time scaling at large sizes; a FIXED problem with more GPUs eventually degrades
   (communication/orchestration overhead stops amortizing — measured: small-size inversions,
   the 4-device crossover between ~504³ and 1008³); frame as rules of thumb, hardware-dependent.
5. *Behind the scenes (high level, where "sharding" is introduced):* views distributed across
   GPUs for the data, slices for the volume; zero-padding to equal shares, kept exactly inert
   (results independent of GPU count); banded communication; single-process (multi-node out of
   scope).

**Alignment review (same pass):** `advanced_features.rst` gains a "Use multiple GPUs" bullet
linking the page; `index.rst`/`overview.rst` key-features line mentions multi-GPU;
`usr_tomography_model.rst` gains a "Device configuration" section (`automethod` entries for
`configure_devices`, `prepare_sino_for_devices`; `autoproperty` for `device_summary`) — the
docstrings are already current, so autodoc content is free; skim `install.rst` /
`dev_maintenance.rst` (looked fine in the 2026-06-11 survey).  Optional follow-on: a small
multi-GPU demo script.  If P6 renames anything user-facing, this task is the reminder to sweep
the page.
