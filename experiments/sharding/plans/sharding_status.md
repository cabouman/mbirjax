# Sharding status (beta branch `greg/parallel_sharding`)

*Short living status. Forward plan: `sharding_implementation_plan_v2.md`.  Completed-work record +
principles: `sharding_implementation_plan.md`.*

**New-session reading guide** (these docs have grown — read selectively):
1. `.claude/claude_prompt.md` — collaboration style + workflow (read).
2. **This file, the TOP handoff only** (current phase + next step) — read.  The
   older handoffs and the "Where we are" history below are **skim/reference**.
3. **`sharding_implementation_plan_v2.md`** — the **current forward plan** (placement architecture
   + phases P1–P6; Phase D re-opened).  Read §0 Design summary and the current
   phase.
4. `sharding_implementation_plan.md` — **completed-work record + principles**
   (Phases 0/A/B/F1/D/F2, cross-cutting principles, hardware facts, O1–O4).  Read
   §Cross-cutting principles; skim the rest for history.
5. `.claude/lessons.md` — **skim** (jax/GPU playbook; consult when a problem rhymes
   with a past one).
6. `.claude/back_projection_overview.md` — read only if touching projector internals.

---

## HANDOFF (2026-06-08) — step 4 unification LANDED (CPU): Change 1 (FMA fold) + Change 2 (2-Keep: ParallelBeam default trivial 1-device mesh); GPU re-validation + note(1) baseline tolerance are NEXT

▶ **CURRENT FOCUS (next session): GPU re-validation of the step-4 unification, then note(1)
baseline tolerance + decide 2-Drop.**  Step 4 (Option B) was implemented in two reviewed
changes this session and is **CPU-green** (full suite); the remaining work is on the cluster.

### What landed (CPU-validated; staged, Greg commits from PyCharm)
- **Change 1 — the unlocked simplification (v2 §P6 note 2).**  Folded `alpha` into the
  buffer-donating `update_error_sinogram(error_sinogram, alpha, delta_sinogram)` → a single
  **fused multiply-add** `e − α·d`.  Dropped the eager `scaled_delta` transient and its
  `.delete()`; for **constant weights the end-of-subset cleanup section is now empty and
  skipped** (delta_sinogram is an `assemble_sharded` output → frees on refcount), non-const
  keeps only `weighted_error_sinogram.delete()`.  The FMA differs from the old eager pre-scale
  by ~1 ULP — fine now that bit-exactness is not required (the in-suite trivial-mesh tests are
  already relaxed to `allclose`).  Files: `tomography_model.py` (`update_error_sinogram` + the
  sharded subtract block + cleanup section).
- **Change 2 — 2-Keep: ParallelBeam defaults to a trivial 1-device mesh.**  New geometry hook
  `TomographyModel._supports_sharding()` (base False; **ParallelBeamModel overrides True**).
  At the end of `set_devices_and_batch_sizes`, when sharding was not explicitly configured and
  the geometry supports it, the **homogeneous** single-device case (`main_device ==
  sinogram_device`, i.e. 'full' GPU / 'none' CPU) auto-defaults to a trivial 1-device `Mesh`
  → `is_sharded` True → one always-on placement path.  New `self._sharding_configured` flag
  (set in `configure_sharding`) prevents `set_devices_and_batch_sizes` (re-run on every
  `set_params` recompile) from clobbering an explicit multi-device mesh; the block also
  re-evaluates each call so a mode flip homogeneous↔heterogeneous sets/clears the auto mesh.
  - **Heterogeneous recon-CPU/sino-GPU 'sinograms' mode stays on the legacy single-device
    branch** (a CPU+GPU pair is not one mesh).  So `_transfer` is untouched and note(3)
    (measure `_transfer`, then 2-Drop) is **deferred** — this is the deliberate 2-Keep scope.
  - Cone/translation/multiaxis inherit base `_supports_sharding()=False` → stay `mesh=None`
    (legacy path) until P6.  `MultiAxisParallelModel` extends `TomographyModel` directly (not
    ParallelBeam), so it is NOT auto-meshed — verified.
- **Capability gap the flip exposed + fixed: partial-view projection.**  `view_indices` (a
  subset of views) breaks the equal view-shard, so `sparse_forward_project` /
  `sparse_back_project` route `view_indices is not None` to the **single-device** impl on a
  trivial 1-device mesh, and raise `NotImplementedError` only on a real **multi-device** mesh.
  (Only `vcls.py` and `test_view_batching` pass `view_indices`; both run single-device.)
- **Tests adapted to the approved default change** (`tests/sharding/`): the no-mesh-ParallelBeam
  tests assert the old default.  Per decision: `TestNoMeshNoOp` (2 tests) **tagged
  RETIRE-AFTER-SHARDING + `@unittest.skip`** (the no-mesh branch is now non-ParallelBeam-only,
  retires at P6); `test_single_device_trivial_placements` **updated** (mesh is now a trivial
  1-device mesh, not None — placement-triviality assertions unchanged and still pass);
  `test_fbp.py::test_filter_runs_and_preserves_shape` **updated** (plain input now yields a
  1-device NamedSharding; shape still preserved).
- **Verification (CPU):** `tests/sharding/` + `tests/test_vcd.py` **76 passed, 2 skipped**;
  `test_projectors` / `test_fbp_fdk` / `test_prox` / `test_denoiser` / `test_hsnt` /
  `test_preprocessing` / `test_utilities` green.  (Pre-existing flake: `test_qggmrf::
  test_loss_and_gradient` uses unseeded `np.random` + tight default `allclose` on near-zero
  gradient entries — passes 5/5 isolated, never builds a model, unrelated to this change.
  Worth seeding — flagged as a side task.)

### NEXT (cluster — Greg)
1. **GPU re-validation of the unification.**  (a) Change 1: re-confirm the sharded-VCD memory
   is still flat (504³/5-iter/1-device-mesh const/non-const/positivity ≈ 6.9/7.9/7.3 GB) — the
   FMA changes the sharded arithmetic, so re-measure peak.  (b) Change 2: the **normal
   single-GPU ParallelBeam user now runs the 1-device-mesh path by default** — confirm
   no-regression vs the pre-flip no-mesh path (time + peak) at 504³ and 1008³.  **Fresh
   `pip install -e .` on the cluster first** (stale editable build burned a whole diagnosis
   once).
2. **note(1) — baseline tolerance.**  With 2-Keep, `vcd_single_device_baseline.py` MESH=False
   on **beta ParallelBeam now auto-meshes (1-device)** — it is no longer the verbatim
   prerelease body, so beta-vs-prerelease is no longer bit-exact.  Swap that comparison to a
   **tight `allclose` (~1e-4)**, not exact (the script captures; the comparison/tolerance lives
   wherever the two runs are diffed — wire the 1e-4 there).
3. **Then decide 2-Drop** (drop heterogeneous 'sinograms' for ParallelBeam): needs the
   `_transfer`-vs-band-streaming GPU timing (note 3) — does the 1-device-mesh band-streaming
   path match/beat the heterogeneous mode for big-recon-on-one-GPU?  If yes, drop heterogeneous
   + `_transfer` and ParallelBeam collapses to the single placement path.
4. **P6 proper** retires the `is_sharded` else-branches once cone/translation/multiaxis are
   ported (the branches are still live for them + for ParallelBeam-heterogeneous under 2-Keep).

---

## HANDOFF (2026-06-07b) — Option B decided for step 4; GPU bit-exact tests relaxed; scaling-tests tooling overhaul (plots + archive + harness dedup)

▶ **CURRENT FOCUS (next session): step 4 — mesh/no-mesh → placement unification, now decided as Option B.**
This session was test/tooling cleanup + the step-4 *decision*; the unification CODE is the next real task.
**Option B (resolved, v2 plan §P6):** a trivial 1-device placement resolves to a 1-device
`NamedSharding` → one always-on placement path; the `is_sharded` guards retire; the transient
cleanup section STAYS (reference cycles are inherent).  Accept ~1 ULP / iterated ≤1e-4 over a
literal "no prerelease regression" (treated like "bit-exact" — not what we actually want).  Three
notes to carry: (1) replace the bit-exact-vs-prerelease guard with a **tight `allclose`** (repurpose
`scaling_tests/vcd_single_device_baseline.py` at ~1e-4); (2) **fold `alpha` into the donated FMA** and
drop the `scaled_delta` transient + its `.delete()` (the FMA-avoidance existed only for bit-exactness);
(3) the heterogeneous recon-CPU/sino-GPU path goes through `_qggmrf_prior_sharded` — **measure
`_transfer` timing before deleting it, and DROP the heterogeneous case if it forces extra code paths**
(target is multi-GPU, not single-GPU stop-gaps).  Broad hot-path refactor → scope + propose first;
suite (`tests/sharding/` + `tests/test_vcd.py`) is the gate.

### What landed this session (all committed except where noted)
- **`fbp_filter` shard-on-entry fix** (`parallel_beam.py`): the internal sharded `fbp_filter` now
  shards a plain sinogram at entry (mirrors `fbp_recon`); a plain input previously only had a shard
  on device 0 → per-device fan-out `KeyError`.  Ported test into `tests/sharding/test_fbp.py`; the
  4 stale top-level `tests/test_sharding_*.py` duplicates were deleted (they collided on a
  once-per-process warning).
- **Bit-exact → tight `allclose`** for the 7 trivial-mesh-vs-single-device tests (`tests/sharding/`):
  they can't be byte-exact on GPU (banded sharded path reorders non-associative FP sums; CPU stays
  exact).  Single-shot `1e-5`, iterative VCD recon `1e-4` (matched to the GPU-proven multi-device
  siblings).  All tagged **`RETIRE-AFTER-SHARDING`** (grep stem `RETIRE-AFTER`); convention documented
  in v2 plan §P6.  They retire when the legacy single-device path is gone.
- **Scaling-tests tooling overhaul** (`experiments/.../scaling_tests/`, NOT library code):
  - size-sweep plots now **minutes / GB**, dynamic 4-decade time axis, ideal lines anchored
    bottom-left, and a **configurable `time_ideal` slope** (`voxels` for fbp, `voxels·views` for
    projectors/VCD) stored per-driver in the YAML; `replot_from_yaml.py` honors it.
  - **`scaling_tests/archive/`** holds 8 resolved one-off diagnostics (moved verbatim; each with its
    conclusion + pointer in `archive/_file_index.md`); the main `_file_index.md` was rewritten
    purpose-grouped.
  - **Phase-2 harness dedup**: the 5 scaling drivers now share one harness in `scaling_common.py`
    (`run_measure_loop`, `build_worker_env`, `build_setup_result`/`print_setup_banner`,
    `OOM_MARKERS`/`is_oom`, `beta_root`) — each driver is just config + op shims + a `build_and_time`
    callback (−646/+326 lines).  Standardized across all 5: throttle sampling, traceback-based OOM
    classification, topology/dev2dev snapshot.  Validated on CPU (fbp/vcd/back run end-to-end;
    forward/direct import-clean); post-vs-pre-refactor YAMLs compared (correctness identical; only
    timing noise + the new fields differ).
  - Still own local `_OOM_MARKERS`/`_beta_root` (out of Phase-2 scope, different structure):
    `sparse_back_project_single_device_sweep.py`, `vcd_single_device_baseline.py`.

---

## HANDOFF (2026-06-07) — P4 DONE (sharded-memory leak fixed & validated 1–4 GPU); `is_sharded` refactor; NEXT = mesh/no-mesh unification (step 4)

▶ **CURRENT FOCUS (next session): mesh/no-mesh → placement unification** (v2 plan P6 / "step 4").
P4 is complete: the sharded VCD path is correct and memory-bounded on 1–4 GPUs.  The next
placement-architecture step is to fold the dual `self.is_sharded` / no-mesh code paths into one
always-on placement path: `update_error_sinogram` becomes the single error-sino update (like
`update_recon`), the `is_sharded` guards retire, and the **transient-free cleanup section stays**
(it does NOT retire — sharded-array reference cycles are inherent; see lessons).  One design call
to make: does a trivial 1-device placement resolve to `SingleDeviceSharding` (keeps single-device
cycle-free, no `.delete()` there) or a 1-device `NamedSharding` (uniform, but then single-device
also pays the cleanup)?  After unification: **P5** (device-config UX, `configure_devices`) and
**P6** (port placements to cone/translation/multiaxis).  Deferred: hybrid `qggmrf_..._transfer`
timing; prox-map prior under sharding.

### What landed & was validated this session
- **Sharded-memory leak — DIAGNOSED, FIXED, VALIDATED (all paths).**  Root cause: jax keeps
  sharded (`NamedSharding`) arrays in **internal reference cycles**, so per-subset *out-of-place*
  updates of the view-sharded error sinogram leaked one full sinogram/subset until GC (peak grew
  with subsets×passes; 504³/5-iter 1-device-mesh 25.8 GB, OOM at 1008³).  Fix (mesh-guarded;
  single-device untouched): `update_error_sinogram` does the subtract under
  `@partial(jax.jit, donate_argnames='error_sinogram')` → **in-place** (alpha scaled eagerly to
  keep trivial-mesh bit-exact); a **single end-of-subset cleanup section** `.delete()`s the
  eager-op transients (scaled delta; + `weighted_error_sinogram` for non-const weights) after one
  `block_until_ready` on the returned state.  Forward-projection (`assemble_sharded`) outputs free
  on refcount, so **positivity needs no extra delete**.  Validated GPU (504³/5-iter/1-device-mesh):
  const **6.9 GB**, non-const **7.9 GB**, positivity **7.3 GB** (all ≈ baseline, flat); 1008³
  1-device completes (was OOM).
- **1–4 GPU scaling validated** (`vcd_recon_gpu.yaml`): 1008³ **1d→4d = 4.49× (super-linear)**,
  memory shards 1/n_dev (55.5→12.7 GB/dev), correctness vs prerelease **8.79e-7**.
- **`is_sharded` property** added — single source of truth replacing 21 `self.mesh is None/not None`
  checks (body changes once at the placement migration; retires at unification).
- **Size-sweep time-ideal curve fixed**: ∝ voxels·views (projection cost, N⁴), not voxels (N³);
  memory ideal stays ∝ voxels.  `replot_from_yaml.py` re-renders plots from a saved results YAML.
- Tests: full `tests/sharding/` + `tests/test_vcd.py` **77 passed** (incl. +4 non-const/positivity:
  trivial bit-exact + 2/4/8-dev NRMSE).

### Ruler-before-code lesson from this session
A reported **33.4 GB "non-const leak" was a STALE GPU build** running the pre-fix binary; a fresh
`pip install -e .` made it bounded (7.9 GB).  We chased it for a while with a temporary `_memprobe`
localizer (now removed) before catching the stale-build cause.  When GPU memory/behavior
contradicts the local tests, **verify the build first.**

### Commit state
The code fix (donation + cleanup + `is_sharded` + tests + curve fix + harness toggles) is committed
& pushed.  Uncommitted doc updates (Greg commits from PyCharm): `sharding_implementation_plan_v2.md`
(P4 → DONE), this status handoff, and the `.claude/lessons.md` entry.

---

## HANDOFF (2026-06-05) — P4 VCD on placements DONE (CPU, ParallelBeam); GPU scaling + hybrid timing are NEXT

▶ **CURRENT FOCUS (next session): single-device memory / no-regression vs prerelease.**
P4 core (sharded VCD + A halos-once + alpha host-sync fix) is DONE & validated; GPU scaling
is healthy (1008³ 2d→4d **2.20×** — the 4-device cap was SIZE, resolved at scale).  The open
item Greg flagged: the slice-**band** is really a per-device *memory* knob but is sized only by
device count (`~num_slices/n_dev²`), so at `n_dev=1` it's **full** (no streaming).  Beta's
1-device-MESH path OOM'd at 1008³ where **prerelease single-device fits 1008³ fine** → must not
regress the 1-GPU capability.  Plan (steps): **(1)** prerelease single-device time+mem baseline —
**script ready:** `vcd_single_device_baseline.py` (no sharding, 1 run/size, OOM-safe, output
named by branch); Greg runs on prerelease AND beta.  **(2)** verify beta *no-mesh* path matches
prerelease (normal single-GPU users; my VCD edits are all mesh-guarded so it *should* be the
verbatim prerelease body — verify at 1008³).  **(3)** diagnose the 1-device-MESH heaviness (full
band? no within-band pixel-batching like the single-device body?) via the `back_project_slice_band`
knob + `sparse_back_project_single_device_sweep.py`.  **(4)** make band sizing **memory-budget-driven**
— `band = min(reduce-scatter-optimal(n_dev), cap_from_byte_budget)` — so it streams on 1 device and
uses finer bands for huge recons on N devices (this is the v2 plan's deferred P6 "single-device
streaming / trivial-sharding unification", pulled forward).  Steps 3–4 need code review/approval.
B (parallel prior) is now low-priority (4d cap was size, not the prior).

**Sharded VCD reconstruction works end-to-end on placements (ParallelBeam).**  The
loop now composes the validated sharded forward/back projectors with one new piece
— the **qGGMRF prior on a slice-sharded recon** — and runs entirely on the
placements with no accidental gather/re-shard.  Staged (not committed — Greg commits
from PyCharm).

### What landed
- **Halo-aware qGGMRF** (`qggmrf.py`): `qggmrf_gradient_and_hessian_at_indices` and
  `qggmrf_grad_and_hessian_per_cylinder` gained `left_halo`/`right_halo` (per-cylinder
  boundary values for the inter-slice term).  Default `None` = mirror the local
  boundary slice (reflected BC) → **bit-exact** with the old single-device prior
  (the old `0`-concatenation special case is the `left_val=v[0]`/`right_val=v[-1]`
  case of the new `left_delta`/`right_delta` formulation).  Ported from the research
  branch; allocation-free.  The single-device `..._transfer` variant updated to pass
  reflected-BC neighbors (kept bit-exact).
- **`_qggmrf_prior_sharded`** (`tomography_model.py`, next to `_extract_halos`): the
  recon-domain analogue of the sharded projectors.  Calls `_extract_halos` (one host
  read of `2*(n_dev-1)` boundary slices), runs the halo-aware prior **locally on each
  slice-owner's shard**, and reassembles into a slice-sharded `(N, num_slices)` via
  `recon_placement.shard_structure(2)` + `assemble_sharded` — no in-kernel cross-device
  comm, consistent with the host-mediated movement architecture (no XLA auto-collective).
  Trivial 1-device mesh → 1 shard, reflected BC → bit-exact.
- **VCD made placement-correct** (`vcd_recon` + `create_vcd_subset_updater`):
  - `to_sino`/`to_recon` helpers route every entry placement (sinogram/weights →
    view-sharded; init/recon/flat_recon → slice-sharded) and **replace the
    `device_put(..., main_device/sinogram_device)` calls that would silently gather a
    sharded array**.  `direct_recon(sharded sino)` → slice-sharded init (match-input);
    `forward_project(init)` → view-sharded error sino; `compute_hessian_diagonal` →
    slice-sharded `fm_hessian`.
  - Subset updater: prior dispatches to `_qggmrf_prior_sharded` when `mesh is not None`.
    Recon-domain gathers/scatters (`fm_hessian[idx]`, `flat_recon[idx]`, `update_recon`)
    use a **mesh-replicated `recon_indices`** (PartitionSpec()) so the index array
    matches the slice-sharded operand's devices (the first multi-device bug the tests
    caught: indices committed to device 0 vs array on `[0..n]`).
  - The line-search `alpha`: the forward-model scalars (sino mesh) and prior scalars
    (recon mesh) live on **distinct meshes**; combining device scalars across meshes is
    illegal.  Fix: in the sharded path reduce the four scalars + `alpha` to **host
    floats** (`float(...)`) and do the alpha arithmetic on the host — a python-float
    `alpha` scales any sharding cleanly (the second multi-device bug the tests caught).
    *Perf note for the GPU run:* this is a few host syncs per subset (the original
    already gathered `forward_lin_quad` to main_device + `float(alpha)`), but watch it.
- **Tests** (`tests/sharding/test_vcd.py`, 7 tests, all green): halo boundary
  self-consistency (mesh-free) + no-halo==legacy-reflected-BC; sharded-prior trivial
  **bit-exact** and 2/4/8-dev float-match (+ slice-sharded output check); full recon
  trivial **bit-exact** and 2/4/8-dev match (measured **NRMSE ~6e-7** vs single-device,
  far inside the 1e-4 gate); and an audit test that `vcd_recon`'s return (pre-exit-gather)
  stays slice-sharded across all devices (no accidental gather in the loop).

### Verification (CPU)
Full `tests/sharding/` **70 passed**.  Single-device regressions green: `test_qggmrf`,
`test_vcd`, `test_denoiser`, `test_prox` (the qGGMRF signature change is backward-compatible).

### CPU scaling (measured 2026-06-05) — VCD sharding works; sub-256³ "slowdowns" were a ruler error
The first shard-vs-noshard demo ran at **64³** and showed VCD ~0.35× at 4 CPU devices,
which I wrongly called "expected overhead."  It was a **too-small ruler**: per-subset
sharded ops have a problem-SIZE floor, and even the *bare* back projector is only
0.44× at 4 dev at 64³ on CPU (it doesn't beat 1× until ~256³).  Measured VCD scaling
(`vcd_recon`, 4 iters, 4 virtual CPU devices):

  | size | 1d | 2d | 4d | 8d |
  |---|---|---|---|---|
  | 64³  | 1.00× | — | ~0.35× | — |  (overhead-bound)
  | 128³ | 1.00× | — | 0.86× | — |
  | 256³ | 1.00× | **1.63×** | **1.86×** | 1.73× |
  | 384³ | 1.00× | 1.57× | 1.75× | 1.41× |

So **2-dev is already a clear win, 4-dev is the peak, 8-dev regresses** — the
bandwidth-bound pattern of the bare back projector on shared-memory virtual CPU
devices (it plateaus/declines past 4 too); 384³ scales slightly worse than 256³
because larger working sets saturate the one memory bus sooner (cf. back-proj going
backwards at 400³).  256³/4-dev (~1.7–1.9×, run-to-run) matches the bare projectors'
CPU ceiling (back ~1.9×, forward ~1.6×), so **VCD scales as well as its building
blocks — no VCD-specific pathology**; ~4 devices is the CPU sweet spot.  The full
user-facing `recon()` demo at 256³ (8 iters, 4 dev) gives **1.56×** (a bit below the
pure-loop figure — it also pays the serial exit gather + partition gen) with
**identical NRMSE 0.0752** shard vs no-shard.  Attribution (256³/4-dev) showed the **sharded prior** is the one component
that regresses at fine granularity (0.47× on 1024-pixel subsets vs 1.45× on
16384-pixel ones) — its host-halo extraction (~1.35 ms) + per-shard dispatch don't
amortize; ~8% of per-subset cost, so a minor drag and a clear future optimization.
The demo default is now 256³ with size guidance.  (Tools: `vcd_recon_scaling.py` for
rigorous curves; the throwaway `/tmp/vcd_diag.py`/`vcd_sweep.py`/`vcd_attrib.py` ablations
produced the numbers above.)
**Remaining before P5/P6:** GPU scaling on the cluster (the crossover/sweet-spot is at
larger sizes there, with real per-device memory — and the d2d path only runs there) +
the per-subset scalar-host-sync watch.  CPU measurement is now done (above).

### `vcd_recon_scaling.py` CUDA_ERROR_NOT_PERMITTED — RESOLVED (the messages are benign)
The `CUDA_ERROR_NOT_PERMITTED` lines turned out to be **benign `W` warnings** from XLA's
VMM allocator (`cuda_vmm_allocator.cc`) probing FABRIC+POSIX_FD memory-handle types on the
first multi-GPU allocation; those handle types aren't permitted in this job's environment,
so XLA **retries with simpler handles and succeeds**.  Proof: a standalone multi-GPU
`jnp.sum` over a NamedSharding array emits the same warnings AND returns the correct value;
and the full run exits 0, writes the YAML, prints "Done", correct vs baseline.  Silence
with `TF_CPP_MIN_LOG_LEVEL=2` (cosmetic).  *(Two earlier hypotheses were wrong — the
orchestrator JAX-import and a blocked P2P collective — the data refuted both; classic
ruler-before-code.)*  Separately we DID fix a real latent bug found en route:
`vcd_recon_capture_baseline.py` now imports mbirjax LAZILY so the scaling orchestrator
stays JAX-free (it was pulling jax in via the top-level constants import) — worth keeping
(no orchestrator GPU preallocation), but it was NOT the cause of the warnings.

### GPU scaling (H100×4, NVLink mesh, dev2dev_safe=true, no throttle) — size-floor, mirrors CPU
`results/vcd_recon_gpu.yaml` (10 iters; correctness vs baseline 8.8e-7):

  | size | 1d | 2d | 4d | mem 1d→4d |
  |---|---|---|---|---|
  | 256³ | 5865 ms | 10450 ms (0.56×) | 21180 ms (0.28×) | 4770→355 MB |
  | 512³ | 43274 ms | 26301 ms (**1.65×**) | 26690 ms (1.62×) | 40685→2038 MB |

- **ATTRIBUTION (full op sweep, same allocation, 2026-06-05): the under-scaling is
  VCD-SPECIFIC per-subset host overhead, NOT a size floor or environment.**  At 4 GPUs every
  *bare* op scales — 256³: back 2.51×, direct 2.78×, forward 4.00×; 512³: fbp 3.10×, back
  3.33×, direct 3.65×, forward 4.75× — while **vcd_recon alone** is 0.28× (256³) / 1.62×
  (512³).  The projectors handle 256³ fine, so my earlier "256³ below the GPU crossover"
  framing was WRONG.  Mechanism: VCD calls the projectors/prior PER SUBSET
  (`recon_pixels/num_subsets` pixels), so each sharded op does `num_subsets`× less work than a
  full projection while the per-call host overhead (`_extract_halos` read, 5 `float()` alpha
  syncs/subset, per-shard dispatch) is fixed → overhead-bound.  CPU corroborates the overhead
  is host round-trips: at 256³/4d VCD scales like the other ops on CPU (2.10×, cheap shared-mem
  copies) but collapses on GPU (0.28×, expensive host syncs/PCIe).  ⇒ A/B/alpha target exactly
  this; success metric = move VCD GPU 512³/4d from 1.62× toward the projectors' ~3.3×.
- **Memory: the per-device peak is REAL (not a preallocation artifact); only cross-size
  extrapolation is unsafe.**  CORRECTED (Greg, 2026-06-05): `peak_bytes_in_use` is the peak
  *live* working set and is essentially **preallocation-invariant** — rerunning gave the same
  numbers (1d 512³ ≈ 41.6 GB either way).  Under a generous budget XLA does no remat, so this is
  the natural full-speed working set, a real figure — so my earlier "preallocation over-reports
  the 1d number" was WRONG, and the ~10× 1d→Nd drop is genuine (band-streaming + sharding really
  cut live memory).  What you must NOT do is extrapolate absolute MB across sizes.  And
  **`PREALLOCATE=false` does NOT reveal the capacity floor** (it gives ~the same peak).  To find
  the true floor / OOM threshold you must ARTIFICIALLY RESTRICT the budget: keep PREALLOCATE=true
  and LOWER `MEM_FRACTION` (hard pool cap, e.g. 0.25) so XLA rematerializes to fit — a size that
  then OOMs is the honest max-recon-per-GPU.  `vcd_recon_scaling.py` now exposes `MEM_FRACTION`
  for exactly this; the `sparse_back_project_single_device_sweep.py` continue-past-OOM pattern
  is the model.  (The `MEM_PREALLOCATE` knob is kept but flipping it does nothing to the peak.)
- **2→4 stalls at 512³** (1.65×→1.62×) while a single big projector got 3.29×@4d at 1008³ on
  the earlier clean run.  Likely VCD's **per-subset host overhead** (many small sharded calls
  ×  `_extract_halos` host reads + 5 `float()` alpha syncs/subset), amplified at 4 dev where
  the host round-trips cross the NUMA0/NUMA1 boundary (2 dev are both on NUMA0).  ⇒ the
  prior-opt **A + B + alpha fix below are now justified GPU levers**, not cosmetic.
- **Ablation (1) DONE:** the bare-projector sweep on this allocation settled it — projectors
  scale 2.5–4.75×@4d at 256³/512³, VCD alone under-scales ⇒ VCD per-subset overhead, not
  environment.  **So implement A + B + alpha next** (they directly cut the per-subset host
  round-trips), then re-measure GPU vs the 1.62×→~3.3× target.  [DONE — A+alpha landed; the
  1008³ run answered the scaling question below.]  Capacity floor, if wanted: a LOW `MEM_FRACTION`
  pass (hard cap) to find the OOM threshold — NOT `PREALLOCATE=false` (which leaves the peak ~unchanged).

### Prior-optimization plan (AGREED; deferred — decide scope after the GPU run)
The sharded qGGMRF prior regresses at fine granularity (0.47×; ~8% of per-subset CPU
cost, but host round-trips hurt the GPU more — let the GPU numbers set priority).  Agreed
approach (options A–E with pros/cons were worked through this session; A+B chosen, C/D held,
E rejected as an algorithmic change):
- **A — extract halos once per partition pass, not per subset.  ✅ DONE (CPU-validated;
  GPU perf re-measure pending).**  New `_stage_halos` (extract + pre-place each halo on its
  shard's device, once); `vcd_partition_iterator` stages once per pass and threads the staged
  halos through `vcd_subset_updater` → `_qggmrf_prior_sharded(..., staged_halos=...)`.  A
  temporary `self._vcd_halo_per_subset` switch restores per-subset extraction for A/B.
  **Replicated-pixel caveat QUANTIFIED:** the per-subset (exact) path still matches single
  device to **6e-7** (machinery sound); the halo-once approximation (replicated pixels only)
  is **NRMSE ~6e-5–3e-4, max ~1.5–2.4e-3** on small noise problems — ~1000× below the
  recon-vs-phantom error (~0.07), i.e. negligible.  Tests: exact-path sweep gated at 1e-4;
  new `test_halo_once_per_pass_approximation_is_small` bounds it <2e-3.  Full sharding suite
  75 green.  **GPU RESULT (H100×4, 10 iters; old run saved under `results/vcd/v0_per_subset_halo/`):**
  A **un-stalls the 2→4 step at 512³** — v0 1.65×→1.62× (flat) becomes A 1.76×→**1.92×**
  (4d time 26690→22534 ms, −16%); 256³ still inverted (0.28×→0.33×, below the crossover).
  Confirms the per-subset halo host-read was part of the cap.  Still 1.92× vs the projectors'
  ~3.3×, so more per-subset host overhead remains (→ B + alpha).
- **B — run the prior's shards concurrently via `run_per_device`** (the projectors' thread
  pool; the prior currently loops serially).  **NEXT** (justified: A closed only part of the
  gap).  Good, **if** we modularize the thread management so the subset-updater loop stays
  readable.
- **alpha host-sync fix.  ✅ DONE (CPU-validated; GPU perf re-measure pending).**  The line
  search did 5 `float()` device→host syncs/subset (4 scalars + alpha) to combine forward
  scalars (sino mesh) with prior scalars (recon mesh).  Replaced with on-device replication:
  new `_replicate_scalar(x, placement)` (`device_put` to a fully-replicated `NamedSharding`,
  a cheap same-device/NVLink scalar reshard, NOT host); forward scalars are replicated onto
  the recon mesh, `alpha` stays a **device scalar**, and is replicated onto the sino mesh to
  scale the view-sharded delta.  `get_forward_lin_quad` is called with `output_device=None`
  when sharded so it doesn't pre-commit to device 0.  Sharded path now has **zero per-subset
  host syncs** (verified by grep: the remaining `float(alpha)`/`device_put(main_device)` are
  all in single-device-only branches).  Single-device path unchanged (alpha stays a jax
  scalar exactly as before).  Tests: sharded VCD suite 8 green (trivial bit-exact, exact-path
  1e-4, halo-once bound).  **GPU RESULT (512³; v1=A under `results/vcd/v1_per_partition_halo/`,
  A+alpha under `v2_in_place_alpha/`):** 2d 1.76×→**1.97×** (24659→21196 ms, −14%) and 1d
  faster too (the 1-device *mesh* path also had the syncs; 256³ 1d 6152→5140 ms, −16%), but
  **4d ~flat** (1.92×→1.90×, 22534→22046 ms).  Cumulative vs v0: 512³ 2d 1.65×→1.97×.

### The 4-device cap is cross-NUMA × VCD's many-small-calls — likely structural, not host-sync
After A+alpha, **4d ≈ 2d in absolute time at 512³** (22046 vs 21196 ms) — going 2→4 no longer
pays.  The box is **GPU0/1 on NUMA0, GPU2/3 on NUMA1**; at 2 dev everything is on NUMA0, at 4
it spans both.  A single big projector hit 3.3×@4d because its one large sharded call amortizes
the cross-NUMA movement; VCD makes **many small per-subset calls** (per subset: sharded
back+forward+prior, each with its own band/move_shard loop + assemble + the alpha scalar
replications), each paying cross-NUMA latency that does NOT amortize over the tiny per-subset
compute.  So per-op host-sync removal (A, alpha) helps 1d/2d but can't move the 4d cap.  **Net:
~2× at 2 GPUs (512³) is the shippable win; on GPU sharding's primary value is capacity anyway.**
- **RESULT (504³ & 1008³, d=1,2,4): the 4-device cap was SIZE, not structural.**  504³: 2d
  1.96×, 4d 1.86× (4d<2d, same flat 2→4 as 512³).  **1008³: 2d→4d = 2.20×** (382959→173894 ms) —
  near-ideal doubling.  So once per-subset work is large enough to amortize the cross-NUMA cost,
  4 devices pays; the 4-device crossover sits between ~504³ and 1008³.  ⇒ **VCD multi-GPU scales
  fine at the sizes where you'd actually shard on GPU (capacity-driven); no structural redesign
  needed.**  Memory at 1008³ shards cleanly (2d 26278 → 4d 12721 MB).  The "many small subsets"
  ideas (in lessons) are filed for the future, not needed now.
  - *1008³ 1d FAILED* with `setting an array element with a sequence` (`oom=False`).  Almost
    certainly real OOM: 1d 512³ already needs ~41.6 GB live, so 1008³ (8×) far exceeds 80 GB; the
    numpy-looking error is the OOM surfacing at a host boundary, and the harness had swallowed the
    traceback.  Harness now records `traceback.format_exc()` + classifies OOM from the full stack;
    re-run 1008³/1d (or a low `MEM_FRACTION`) to confirm the stack shows RESOURCE_EXHAUSTED.
- **B — parallel per-shard prior** (`run_per_device`): clear CPU win; GPU benefit doubtful now
  that the 4d cap is shown to be size (resolved at 1008³).  Lower priority — do for CPU/B-completeness,
  or skip.  Decide with Greg.
- **Add a qGGMRF scaling test with a prerelease baseline** to round out the suite
  (mirrors the projector/recon baselines).

### NEXT
- **GPU (Greg, cluster):** VCD scaling on H100.  **Harnesses now written** (CPU-smoke-tested,
  in `experiments/sharding/scaling_tests/`): `vcd_recon_scaling.py` (isolated-subprocess
  device+size sweep, times internal `vcd_recon` on pre-sharded sino, correctness vs
  prerelease baseline) + `vcd_recon_capture_baseline.py` (run once from a prerelease
  checkout — **CPU baseline already captured**, beta-vs-prerelease single-device
  max_abs_diff **5.3e-8**; re-capture on the cluster as baselines/ is gitignored) +
  `vcd_shard_vs_noshard.py` (full-recon head-to-head, time/mem/NRMSE; shard/no-shard
  NRMSE identical 0.1826 — correctness; default now 256³ where CPU sharding pays).
  This is the only place the d2d path + real scaling run.  Pre-flight `nvidia-smi dmon`.
  **Watch the per-subset host syncs** (the `float(alpha)` coercions) — if they dominate
  at scale, replicate alpha onto both meshes instead of bouncing through the host.
- **Hybrid timing (deferred Q3):** measure whether the `qggmrf_..._transfer`
  small-subset variant still earns its keep under the recon-CPU/sino-GPU placement; if
  not, retire it.
- **P5/P6** per `sharding_implementation_plan_v2.md` (device-config UX; geometry port —
  cone/translation/multiaxis prior+projector, which also needs their horizontal-fan
  channel-major layout, Track A).  Prox-map prior under sharding is untouched (the
  `prox_input` branch); revisit when needed.

---

## HANDOFF (2026-06-04 #2) — P3 forward DONE (CPU); power-of-2 channel-aliasing fix (parallel beam done, GPU-gate PASSED, port to 3 geometries is NEXT); tests reorganized

**Two things landed this session: P3 (sharded forward projection) and a DETOUR —
a power-of-2 cache-aliasing bug in the projector kernels, found via forward CPU
scaling.  Both are committed; the layout fix has an open GPU gate (running).**

### P3 — forward projection on placements: DONE (CPU-green; GPU scaling pending)
The all-gather **adjoint** of the band back projector, **slice-banded (Option B)**
— chosen over the v2 plan's pixel-batch (`move_cylinders_to_sino`) for memory
symmetry with back projection (same reasons band beat pixel in P2).
- New `broadcast_band_to_views` (`_sharding/transfer.py`) — adjoint of
  `sum_band_to_owner`; `_sparse_forward_project_sharded` + `_forward_project_all_bands`
  + `_forward_project_band_to_local_views` (`tomography_model.py`); `forward_project`
  match-input (mirror of `back_project`); parallel-beam **forward kernel sizes
  output rows from input slices** (adjoint of the back kernel's row-slicing).
- **Removed** the superseded full-cylinder primitives `move_cylinders_to_sino` /
  `sum_cylinders_to_recon` (the banded `sum_band_to_owner`/`broadcast_band_to_views`
  pair supersedes them).
- Gate: **forward/back adjoint round-trip** `<Ax,y>==<x,Aᵀy>` green at 1/2/4/8
  devices; trivial n=1 bit-exact; match-input tested.  Commit `7461abb`.
- **Prerelease baseline captured + wired** (this session): `sparse_forward_project_capture_baseline.py`
  + `baselines/sparse_forward_project.{npy,yaml}` (captured CPU prerelease via a temp
  `prerelease` worktree); the forward harness's setup now checks the adjoint identity
  **and** `forward(x_cyl)` vs the prerelease reference (beta matches prerelease to
  **1.1e-5** — also validates the channel-major layout fix against prerelease).
  **TODO — capture the same forward baseline for cone / translation / multiaxis**
  (generalize `sparse_forward_project_capture_baseline.py` beyond `ParallelBeamModel`,
  or add per-geometry capture); pairs naturally with the geometry port in Track A.
- **Pending:** GPU forward scaling (`sparse_forward_project_scaling.py`) on the
  cluster — the only place the d2d path runs (re-run will also exercise the baseline
  check cross-platform).

### Tests reorganized
The 8 sharding test files moved to **`tests/sharding/`** (dropped the redundant
`test_sharding_` prefix → `test_back_projection.py`, etc.).  `preferred_devices`
now lives in `tests/sharding/conftest.py`; the parent `tests/conftest.py` keeps the
load-bearing pre-JAX XLA device flag.  Run all via `pytest tests/sharding/`.

### THE DETOUR — power-of-2 `num_det_channels` cache aliasing (commit `47b787a`)
**Symptom:** forward at 256³ on CPU ran **~6× slower** than 252³ (and the forward
scaling harness showed fake 15–33× superlinear + 400³ faster than 256³).
**Diagnosis (single-variable ablation):** the *sole* cause is **`num_det_channels`
being a power of 2**.  The kernels access `sinogram_view[:, n]` by **column**
(stride = `num_det_channels`); a power-of-2 stride aliases the CPU cache
(set-conflict misses), severity ∝ slices-per-call.  Views/rows are irrelevant; the
in-plane recon access is NOT a separate cause (confirmed at 2048 channels).
**Fix:** **channel-major layout** in the parallel-beam forward + back kernels —
build/read the view as `(channels, slices)` so the per-pixel scatter/gather is
**contiguous (stride 1)**, transpose at the kernel boundary (output layout and all
callers unchanged).  **Bit-compatible** (projectors 4+26, sharding 63 green).
**CPU payoff:** 256³ forward **27291 → 1348 ms (~20×)**; **~3.5× faster even at
non-pow2** (252³ 4474 → 1260); **flat** across the channel sweep (256≈252, 512≈504).
Micro-bench of the isolated scatter: transposed 14.5× faster at 256ch / 4× at
252ch, bit-exact, XLA honors the layout.
**Tooling:** `projector_sino_size_scan.py` — single-device forward/back timing per
sinogram axis (views/rows/channels alone + uniform; pow2 vs non-pow2); CPU/GPU;
A/B vs the old kernel via `git stash`.

**GPU GATE: PASSED (2026-06-04, H100, new kernel).**  `projector_sino_size_scan_gpu.yaml`
is **flat across power-of-2 channels** (1024≈1008: 1229≈1238 ms fwd; 2048≈2016:
5318≈5262 ms) — no aliasing in the new layout, times scale smoothly with
num_pixels; views/rows power-of-2-insensitive.  No regression (1024³ single-device
fwd 35.0 s new vs 45.3 s old forward-scaling n=1, different paths but not slower).
**Confirms the old GPU kernel WAS aliasing at power-of-2 channels** — that is what
faked the **5.17× forward superlinear at 1024³** (n=1's full band thrashed; sharded
small bands escaped).  ⇒ re-running `sparse_forward_project_scaling.py` with the new
kernel should show *normal* ~3–4× scaling.  **Layout fix is good on CPU and GPU.**

**Suspect prior numbers:** early **CPU back-projection** scaling used 256³ (power of
2) → aliased → suspect (re-run at non-pow2).  GPU back used 1008³ (non-pow2) → clean.

### FORWARD PLAN — two tracks

**(A) Finish the layout-fix detour — GPU gate PASSED, so the PORT is the next task:**
Apply the same channel-major transpose (build/read the **horizontal-fan**
`sinogram_view` as `(channels, rows)`; contiguous `.at[n, :]` scatter / `[n, :]`
gather; transpose at the kernel boundary, output layout unchanged) to the
horizontal-fan kernels of the other three geometries — the only kernels with the
strided `sinogram_view[:, n]` channel access (the vertical fan handles slices and
is untouched):
  - **cone_beam.py**: `forward_horizontal_fan_pixel_batch_to_one_view` (the
    `.at[:, n].add`, ~L356) + `back_horizontal_fan_one_view_to_pixel_batch`
    (`sinogram_view[:, n].T`, ~L529).
  - **translation_model.py**: same pair (`forward_horizontal_fan...` ~L283;
    `back_horizontal_fan...` ~L455).
  - **multiaxis_parallel.py**: `forward_horizontal_fan...` (~L325, note the
    `weight*scale*valid` factor) + `back_horizontal_fan...` (~L385, the
    `cols = sinogram_view[:, n].T`).
  Verify per geometry: `test_projectors` (all geometries) + `test_fbp_fdk` (cone)
  stay bit-exact; `projector_sino_size_scan.py` flat across the channel sweep.
  **Parallel beam is the verified template** (commit `47b787a`).  Mechanical but
  numerically sensitive across 3 files → best done with fresh context.
If a GPU regression ever appears on a geometry → platform-conditional layout.

**(B) Resume the main sharding thread (after the detour):**
- P3 wrap: GPU forward scaling confirmation.
- **P4 — VCD on placements** (NEXT after P3): entry/exit placements for
  sino(view)/recon(slice)/weights(view); halo exchange via `_extract_halos`;
  default init = already-sharded `direct_recon`; no accidental gather/re-shard in
  the loop body.
- P5 — device-config UX (`configure_devices`).
- P6 — port placement path to other geometries; retire `main_device`/`sinogram_device`;
  the `(g0,L)` slice-band projector interface + slice-batching→band unification +
  in-place donation (see the **P3/P6 design note** in `sharding_implementation_plan_v2.md`).

---

## HANDOFF (2026-06-04) — back-projection refactor landed & GPU-confirmed (H100×1–4); `(g0,L)` interface designed for P3

**What changed (staged, not committed — Greg commits from PyCharm).**  A
readability/modularity refactor of the P2 band back-projection, behavior-preserving:
- `_sparse_back_project_sharded` is now a four-beat narrative (setup → size bands →
  `_back_project_all_bands` → assemble).  Extracted `_back_project_all_bands`
  (slice-owner × band double loop) and `_back_project_local_views_to_band` (the
  parallel per-view-owner partial projection).
- New primitive `mjs.sum_band_to_owner` (`_sharding/transfer.py`) — the
  cross-device reduce (move_shard each partial + sum), pulled out of the old inline
  loop; foreshadows the deferred P3 consolidation with `sum_cylinders_to_recon`.
- `_slice_band_length` is now a `@staticmethod(…, fixed_band=None)` with the
  1/n_dev² band-sizing rationale written out.
- **Terminology** adopted throughout: *view-owner / slice-owner / view-shard /
  slice-shard* (and *band* = sub-range of a slice-shard).
- **Light placement alignment:** `devices` from `self.sino_placement.devices`;
  final assembly via `self.recon_placement.shard_structure(2)` (replacing the
  inline `NamedSharding(self.mesh, …)`).
- Bundled (separate concern): the one-line `use_ror_mask` attribute→`get_params`
  fix in `back_project` ([tomography_model.py:790]) left over from the
  anisotropic_voxels2 merge (the test/script spots were already in `56ed0bf`).

**Verification.**  CPU green: back projection + `fbp_recon` 18; full sharding suite
58; projectors + FBP regression.  Trivial n=1 **bit-exact** vs single-device; 2/4/8
float-noise match; Hessian, band-sweep, single-device-streaming, view-subset all
pass.  (`conftest.py` device count set to 8 so the sweep exercises >2 devices; note
the CPU default stays 2 because back projection degrades past it — bandwidth-bound,
not a defect.)

**GPU CONFIRMED (H100×1–4, one NUMA-0 socket, clean — no throttle, all 1980 MHz;
cross-platform correctness 3.05e-5, 0 above 1e-4).**  Times match the pre-refactor
P2 baseline within ~1% at 1/2/3 dev (behavior-preserving); 4-dev is new (no prior
1008³/4-dev baseline).  1008³:

  | n_dev | time (ms) | speedup | mem (MB) | mem_frac |
  |---|---|---|---|---|
  | 1 | 20193 | 1.00 | 11565 | 1.00 |
  | 2 | 11036 | 1.83 | 6996 | 0.61 |
  | 3 | 7363 | 2.74 | 4852¹ | 0.42 |
  | 4 | 6132 | 3.29 | 3460 | 0.30 |

4-dev speedup is also 3.32× @504³, 2.51× @252³ (small size overhead-bound).  The
3.29× (vs the old 3.91× at 1024³) is a **size** effect — 1008 = 2⁴·7·9 tiles worse
in XLA than 1024 = 2¹⁰ — not a regression (the 1/2/3-dev times are unchanged).

¹ **Harness "ruler" gotcha (suspect-the-ruler).** `worker_measure` runs **one
subprocess per *size***, looping device counts *descending* and reading the
cumulative `peak_bytes_in_use` with `gc.collect()` between — but XLA's allocator
fragments, so a config measured *after* a different-device-count config in the same
process can over-report peak.  The 4-dev run's n=3 read **5697 MB** (inflated ~17%
by the preceding n=4); the **clean value is 4852 MB** (from the `[1,2,3]`-only run,
where n=3 was measured first).  n=1/n=2 match across runs because their preceding
config is the same.  **Rule: trust each size's *first-measured* (largest-device-
count) memory; later same-process configs can be fragmentation-inflated.**  A fully
clean fix = one subprocess per `(size, n_dev)` — filed under the (separate) harness
cleanup, not chased here.  Result yamls (gitignored): top-level = 4-dev run;
`results/back_projection/v7_post_refactor_3_gpus/` = the clean `[1,2,3]` run.

**End-to-end `direct_recon` (Phase F2) re-confirmed post-refactor (H100, `[1,2,4]`).**
The full FBP pipeline (shard → `fbp_filter` F1 → `back_project` D → gather) matches
the F2 baseline (`results/direct_recon/v0/`): correctness **identical** (both
`max_abs_diff = 3.95e-7`, 0 above 1e-4 — bit-for-bit), times within ~1%, memory
within ~3%.  1024³: n=1 19594 ms / n=2 9354 ms (2.09×) / n=4 5265 ms (3.72×); the
pipeline scales better than raw back projection because the compute-bound filter
lifts the combined number.  **So the original Phase D (sharded back projection,
reduce-scatter slice-band) and Phase F2 (`direct_recon` match-input pipeline) are
COMPLETE and GPU-validated on the refactored code.**

**Designed, not built — `(g0,L)` geometry-neutral slice-band projector interface.**
Recorded as the **P3 design note** in `sharding_implementation_plan_v2.md` (with a P6
pointer): adopt **B2** — a band parameterized by `(band_start g0 dynamic, length L
static)` in global slice-index units, each geometry converting internally (per-geometry
survey shows `recon_slice_offset` is *not* a uniform currency: cone/multiaxis use it,
translation uses `translation_vector[2]`, parallel is positional).  Decisions captured:
dynamic-g0/static-L recompilation discipline; empty-buffer-for-L rejected;
in-place `dynamic_update_slice` deferred to the *jittable* single-device/forward paths
(P6/P3) only; `entries_per_cylinder_batch` slice-batching is internal banding to be
**subsumed by the band loop** in P6; forward/back adjoint round-trip is the gate.
**Implementation finding:** there is **no clean parallel-beam-only** version — the
shared `vmap` in `projectors.py` forces all four geometry kernel signatures at once —
so PB keeps the bit-exact **external row-crop** for now and the `(g0,L)` kernel is built
once in **P3** (forward+back together).

**Next:** (1) ✅ GPU correctness + scaling confirmed (H100×1–4, table above);
(2) the separate harness vestigial-plumbing cleanup (`--path`/`--pixel-batch`,
`_print_path_comparison`) — and consider the per-`(size, n_dev)` subprocess fix for
the memory-fragmentation gotcha (footnote ¹); (3) **P3 — forward projection on
placements**, building the `(g0,L)` slice-band interface per the design note.

---

## HANDOFF (2026-06-03) — P1 done; P2 DECIDED: keep band (pixel retired); finalizing → P3 next

**Where we are.**  The placement architecture is underway:
- **P1 (placement foundation) — DONE, committed.**  `mbirjax/_sharding/placement.py`
  (`Placement`; `move_cylinders_to_sino` / `sum_cylinders_to_recon` adjoint pair,
  `move_shard`-based, `N×N`/`1×1` no mode branch).  Model builds
  `recon_placement` / `sino_placement` via `_set_placements()` (additive; coexists
  with `main_device`/`sinogram_device` + `mesh`, which a `TODO(P6)` retires).
  Tests: `test_sharding_placement.py`, `test_sharding_hooks.py::TestModelPlacements`.
- **P2 (back projection on placements) — pixel path built + committed; deciding band vs pixel.**
  `_sparse_back_project_pixel` (pixel-batched, uses `sum_cylinders_to_recon`) runs
  **alongside** the slice-banded `_sparse_back_project_sharded`, selected by the
  temporary `_back_project_path` switch ('band'|'pixel').  Shared setup factored
  into `_sharded_back_project_setup`.  Tests in
  `test_sharding_back_projection.py::TestPixelBackProject` (bit-exact, matches
  band, Hessian, B_p sweep, match-input).  The `B_p` knob is
  `back_project_pixel_batch` / `_BACK_PROJECT_MAX_PIXEL_WORK`.

**Band-vs-pixel determination (GPU, H100×3 on one NUMA socket, clean — no throttle).**
Trustworthy GPU numbers (results/ is gitignored, so recorded here), 1008³:

  | n_dev | band time / mem | pixel time / mem | pixel vs band |
  |---|---|---|---|
  | 1 | 20322 ms / 11592 MB | 20424 ms / 23347 MB | t=1.01 m=2.01 |
  | 2 | 11075 ms / 6928 MB  | 9356 ms / 18618 MB  | t=0.84 m=2.69 |
  | 3 | 7396 ms / 4770 MB   | 6555 ms / 12436 MB  | t=0.89 m=2.61 |

Robust reads: **time is comparable** (pixel edges ahead and scales slightly
better at the largest size — 3.12× vs 2.75× on 3 dev; band faster at 504³);
**band uses ~2–2.7× less memory** (structural).  So the tradeoff is **pixel =
simpler + scales as-well-or-better; band = much more memory-efficient.**

**The decider is the `B_p` sweep** (`PIXEL_BATCH_SWEEP = [None, 50_000, 25_000,
12_500]`, re-run): can a smaller `B_p` bring pixel's memory to band's level
without losing its time?  If yes → adopt pixel (retire band code, rename
`_sparse_back_project_pixel` → `_sparse_back_project_sharded`, drop the switch);
if it costs too much time → keep band.  **VERDICT (2026-06-03): KEEP BAND.**  The
B_p sweep showed pixel's peak memory plateaus at **~1.7–2.7× band** (a floor B_p
can't push below — it's the accumulated output + concatenate, not the per-batch
transient), for only ~11–16% less time.  Memory / max-recon-per-GPU is the
priority, so band stays as the sole sharded path; the **pixel path was removed**
(model + tests), and the harness is set band-only.

**Uncommitted at this handoff:** the scaling-harness hardening (3 files:
`scaling_common.py`, `sparse_back_project_scaling.py`, `scaling_tests/_file_index.md`)
— topology + `dev2dev_safe` capture, per-GPU clock/temp throttle flagging,
`DEVICE_COUNTS` override, `PIXEL_BATCH_SWEEP`.  CPU-validated; commit before handoff.

**GPU-allocation gotcha (cost us an afternoon — see lessons.md).**  A single hot
GPU throttling under sustained 1024³ load (345 MHz @ 86 °C while neighbors ran
1980 MHz @ 40 °C) gated the 4-device case and looked like a code regression.
Pre-flight: glance `nvidia-smi dmon -s pct` — the warmest-at-idle card is the one
that throttles.  `DEVICE_COUNTS=[1,2,3]` keeps to the first three (cooler, and on
this node all NUMA-0) GPUs, dodging both the bad 4th card and the NUMA split.

**Next:** finish the `B_p` sweep → finalize P2 per the verdict → **P3 (forward
projection on placements, built on `move_cylinders_to_sino`, with the forward/back
adjoint round-trip test).**  P2 code finalized: pixel path + switch + knob + pixel
tests removed; `_sharded_back_project_setup` kept (band uses it); band tests +
suite green (39 sharding + back-projection).  Remaining: commit the finalization;
optional cleanup of the harness's now-vestigial path/B_p-sweep plumbing
(`--path`/`--pixel-batch`, `_print_path_comparison`).

---

## HANDOFF (2026-06-02) — direction change: placement architecture; Phase D re-opened

**F1 / D / F2 are done and GPU-validated.**  While designing the device-config UX
we converged on a cleaner target and decided to **re-open back projection** to
build on it rather than retrofit.  Two ideas (full design in `sharding_implementation_plan_v2.md`):

- **Placements, not device scalars.**  `main_device`/`sinogram_device` →
  `recon_placement`/`sino_placement` (each a `Sharding`).  Every mode is a
  placement pair; single-device = trivial placement → **sharding always on, one
  code path**.  Scope: homogeneous multi-GPU + recon-CPU/sino-GPU (big recon);
  **not** sino-CPU/recon-GPU.
- **One movement interface.**  `move_cylinders_to_sino` / `sum_cylinders_to_recon`
  (adjoint pair, cylinders-direct, `move_shard`-based, `N×N`/`1×1` no mode
  branch), with **uniform pixel-batched** streaming (`B_p` knob) — likely
  replacing Phase D's slice-banding pending a side-by-side measurement.

**Plan of attack:** build it **next to the existing code on ParallelBeam**
(F1-style: parallel path → measure → promote → delete loser), then port.  Phases:
P1 placement foundation (incl. early `device_put → move` migration) → P2 back
projection on placements (re-opened D; compare vs slice-banding, retire band code
if competitive) → P3 forward (C; adjoint test) → P4 VCD (E) → P5 device-config UX
(`configure_devices`) → P6 port geometries + retire `main_device`/`sinogram_device`.

**Unchanged:** F2's match-input contract + the `direct_recon` scaling driver
(public-API layer; re-run against new internals).

**Next: P1 — placement foundation.**

---

## HANDOFF (2026-06-01) — Phase F2 (`direct_recon`) COMPLETE, CPU-validated

**Phase F2 is done.**  The first **usable end-to-end sharded FBP pipeline** —
`direct_recon`/`fbp_recon` = shard sinogram → `fbp_filter` (F1) → `back_project`
(D) → recon — runs with **zero intermediate host transfer**.  F2 was a **contract
refactor**, not new collectives.  Landed in two rounds: round 1 = boundary
cleanup (user-facing always-gather); **round 2 = match-input** (see principle #5;
O2 RESOLVED).

**Contract (match-input):**
- **User-facing = output matches input** (`direct_recon`, `fbp_recon`,
  `back_project`): plain in → plain out (shard at entry, gather at exit); sharded
  in → sharded out (no gather).  The check is `isinstance(x.sharding,
  NamedSharding)` on the primary input, read before the entry shard.  *Why
  match-input, not always-gather:* these are dual-use — `back_project` is called
  from `fbp_recon`, and `direct_recon` is `vcd_recon`'s default init — so a
  sharded caller (e.g. Phase E's sharded VCD computing a sharded FBP init) stays
  on-device with no round-trip.
- **Internal = pure sharded-contract** (`fbp_filter` + alias `direct_filter`,
  `sparse_*`): sharded in → sharded out, **no transition code**.  Filtering is
  not exposed as a standalone gathered op.
- **Round-2 mechanism:** `back_project` gained `_assemble_recon_volume_sharded`
  (`tomography_model.py`) — a per-device scatter of the slice-sharded cylinder
  into a slice-sharded `(rows, cols, slices)` volume (`run_per_device` /
  `assemble_sharded`); the scatter is identical across slices, so it is
  embarrassingly parallel on the slice axis.  `fbp_recon` shards once, runs the
  pipeline, and gathers at exit only if the original input was plain.
- **Confined to `parallel_beam.py` + `tomography_model.py`** + tests; internal
  `sparse_*` untouched.

**Validated (CPU):** `tests/test_sharding_fbp_recon.py` + `test_sharding_back_projection.py`
+ `test_sharding_fbp.py`: no-mesh plain-in/out; `direct_recon`==`fbp_recon`;
trivial 1-dev **bit-exact**; 2/4/8-dev plain-in float-noise match (plain out);
**match-input sharded-in** → slice-sharded recon (back_project, fbp_recon,
direct_recon) matching single-device to float noise; **no-intermediate-gather**
invariant.  Full suite green: `test_sharding_*` + `test_fbp_fdk` +
`test_projectors` (**50** + 12 subtests).

**Deferred (not blocking):** GPU stress/scaling harness — driver `direct_recon_scaling.py`
ready (CPU-validated: 256³ 2.56× @ 8 dev; SIZES["gpu"] = 256/512/1024³), baseline
`direct_recon.{npy,yaml}` captured from prerelease (CPU diff ~8e-8, NOT bit-exact
— F1 rewrote the filter).  Run on H100/L40S; watch the combined transient
(filter FFT + back-projection partial).

**Next: Step 7 = Phase C (forward projection, all-gather)** — design
`forward_project` **match-input from the start** (mirror `back_project`'s round-2
impl) + the forward/back adjoint test.

---

## HANDOFF (2026-06-01) — Phase D (back projection) COMPLETE, GPU-validated

**Phase D is done.**  Sharded back projection (reduce-scatter) with slice-band
**streaming**: correct (CPU bit-exact at n=1, float noise at n>1; cross-platform
GPU 3e-5), memory-efficient (peak/shard **11× → 3.2×** at 1024³/4-dev; single GPU
1024³ **28 → ~12 GB** — streams by default), near-ideal GPU scaling (3.92× on 4),
time-clean.  Default band is device+compute-bounded (`_slice_band_length`),
overridable via `back_project_slice_band`.  One reused thread pool spans the
per-band fan-outs.  Characterized as **memory-bandwidth-bound** (GPU is the
scaling target; CPU caps ~1.5× by the shared bus, not a defect).  Remaining
memory frontier = the sino+recon floor (host↔device streaming — deferred; an
in-place-assembly attempt to beat it was measured-worse and reverted).
**Next: Step 6 = Phase F2 (`direct_recon`)** — shard → fbp_filter (done) →
back_project (done) → gather; first usable end-to-end sharded FBP pipeline.
Details below.

---

## (earlier) HANDOFF (2026-05-31) — Phase D core complete

**Sharded back projection works end-to-end on CPU (8 virtual devices); GPU run
is the open item for Greg.**  Back projection is a **reduce-scatter**: the
sinogram is view-sharded, so each device back-projects *its* views onto the full
voxel cylinders (Phase 1), then for each slice-owner the per-device partials are
summed over that owner's slice band (Phase 2, naive all-to-all via `move_shard`).
The result is slice-sharded; `back_project` gathers at exit.

**Code (`mbirjax/tomography_model.py`):**
- `sparse_back_project` now dispatches: `mesh is None` → extracted
  `_sparse_back_project_single_device` (unchanged prerelease body);
  `mesh` set → new `_sparse_back_project_sharded`.
- Parallel beam reuses the **existing jitted projector unchanged** — because
  det-row r ↔ slice r, "a device's partial over its views" is just the normal
  back projection of its view-shard; no kernel edit.
- `back_project` shards the sinogram at entry, gathers at exit.
- Loops over the mesh — **not hardcoded to 2 devices**.

**Validated (CPU, `tests/test_sharding_back_projection.py` 7/7):** trivial
n_dev=1 **bit-exact** vs prerelease; n_dev=2/4/8 match single-device to ~1e-7
rel (float noise); internal method returns slice-sharded (no gather), public
gathers; coeff_power=2 (Hessian diagonal) matches; view-subset rejected.
Full sharding suite + `test_projectors` + `test_fbp_fdk`: all green.

**Key design point (settled with Greg):** each device holds *complete views*, so
it can back-project **any** output slice band from local data — geometry-neutral.
Partitioning the *output* cylinder ⇒ each voxel computed once (no redundant
compute, parallel or cone).  The 1/n transient-memory win comes from *streaming*
slice bands (compute band k+1 while band k transfers) + the F1 overlapping-tail
for sub-tiling (SET semantics only) — **deferred**; current cut materializes the
full per-device partial (same order as single-device working set).

**Next:**
- **GPU validation (Greg):** run `tests/test_sharding_back_projection.py` on 1–4
  GPUs; watch H100 (d2d-safe) vs L40S (host-bounce) and the partial buffer.
- Then **Step 6 = Phase F2** (`direct_recon`): filter (F1) → back_project (D) →
  gather; first usable end-to-end FBP pipeline + stress test.

**Deferred/notes:** transient-buffer budgeting → scaling tests; streaming +
overlap-tail sub-tiling is the next memory layer; `compute_hessian_diagonal`
returns slice-sharded in sharded mode (Phase-E contract decision);
`_sparse_back_project_sharded` defensively shards plain input (un-ported callers).

### CPU scaling finding (2026-06-01) — memory-bandwidth-bound, not comms

First CPU sweep (`sparse_back_project_scaling.py`) and a phase-split ablation
(`sparse_back_project_phase_ablation.py`) explain the CPU curves:

- **Speedup caps ~1.4–1.5× and regresses at 8 devices** (e.g. 256³: 1d/2d/4d/8d
  = 1.0/1.34/1.44/1.39×).  Absolute times are good (beta 1-device ~528 ms beats
  the research branch's ~1022 ms; beta 8-dev ~384 ms ≈ research 8-dev ~328 ms).
- **The cap is Phase 1 (per-device compute), not the reduce-scatter.**  Ablation
  at 256³: phase1 caps at 1.54× (flat 4→8), phase2 (reduce-scatter) is ≤9% of
  total but grows with n_dev and is what tips 8 devices into regression.
- **Mechanism = memory bandwidth, NOT core contention.**  Back projection is a
  gather/scatter accumulation (low arithmetic intensity) → bandwidth-bound; the
  8 virtual CPU devices add cores but share one memory bus.  The clean control:
  the SAME harness scales `fbp_filter` ~7× on 8 CPU devices because it is
  compute-bound (FFT) — so virtual-device sharding *does* add real parallelism;
  back projection just can't use it.  (Earlier "cores already saturated" guess
  was wrong — the fbp 7× disproves it.)
- **Beta's view-sharding compounds it:** each device writes a FULL cylinder
  (`pixels × num_slices`, n× the output traffic), vs the research slice-sharded
  scheme that wrote only 1/n and got ~3.1×.  So memory-boundness caps everyone
  (research 3.1×); beta's extra write traffic + reduce-scatter caps it further.
- **Implications:** reduce-scatter optimizations (tree/pipeline) and the deferred
  streaming/sub-tiling won't move the CPU curve (phase2 too small; sub-tiling is
  memory-only).  Do NOT chase CPU scaling — the times are good and the limit is a
  shared-bus property of virtual CPU devices.
- **Sharpened GPU prediction:** a bandwidth-bound op is exactly what benefits from
  real per-device hardware — each GPU brings its own HBM (~2 TB/s), so sharding
  adds bandwidth, not just cores.  Back projection should scale *well* on GPU even
  though it caps on CPU; the full-cylinder write (~52 MB) is microseconds at HBM
  rates.  If GPU *also* caps ~1.5×, the limiter is something portable (reduce-
  scatter / a serialization), not the shared bus — a sharp test either way.

### Slice-band streaming landed (2026-06-01) — memory 11× → ~3.2×, free on GPU

Back projection now **streams the slice axis in bands** (no full-cylinder partial
is ever held): each device row-slices its sinogram to a band, projects, and the
band is reduce-scattered to its owner.  Needed a 1-line kernel fix
(`back_project_one_view_to_pixel_batch` sizes its output from the input rows, not
`projector_params`) so a row-sliced view yields just those slices.  Committed.

- **GPU memory (H100, the goal):** at 1024³/4-device peak/shard fell **11× →
  3.2×** (11461 → 3435 MB, 0.30×), and per-device memory now drops *faster* than
  1/n.  **Time is unchanged at real sizes** (512³/1024³: 0.99–1.02×) — the CPU
  many-small-band penalty does NOT transfer to GPU (launch throughput hides it).
  Only 256³ shows a time hit (tiny per-band work).  So streaming is a strict win
  on GPU: ~3× less memory at the same speed.
- **Band-length sweep (Part C):** the knee is ~`slices_per_dev/n_dev` (band 64 →
  3.35× at 1024³/4d); below it strongly diminishing (band 8 → 2.67× for 8× the
  bands).  Even band=256 (one band/owner, no sub-tiling) is 7.25×, so the *rewrite
  itself* (sequential, row-sliced, no simultaneous full partials) bought 11→7.25×
  and sub-tiling took it to 3.35×.
- **Default band is now budget-based** (`_slice_band_length`): bound the owner's
  reduce gather to ~one owner-shard (`slices_per_dev/n_dev`, the knee), with a
  per-band-work floor (`_BACK_PROJECT_MIN_BAND_WORK = 4M` elements) so small
  recons don't over-split into tiny dispatches (the 256³ penalty).  Override via
  `back_project_slice_band` (sweepable like `ROW_FILTER_BATCH`).
- **Batch tuning is irrelevant** (confirmed again on GPU: peak flat across all
  view/pixel batch configs).  The attribution script's batch sweep ("Part B") was
  **removed**; it now runs Part A (attribution) + Part C (band sweep) only.
- **Single-device streaming is now the default (2026-06-01).**  `_slice_band_length`
  gained a compute-band cap (`_BACK_PROJECT_MAX_BAND_WORK = 100M` elements): the
  reduce-gather bound is vacuous at n_dev=1, so the cap is what makes one device
  sub-tile.  H100 single-device sweep at 1024³: peak **28 GB → 10.5 GB (0.37×)**
  at band 128, heading toward the ~7.3 GB sino+recon floor, and **time was FLAT
  across all bands** — so it's a free ~3× memory win that stretches the max recon
  per GPU (~1.5× larger linear dim).  Multi-device band choice is unchanged (its
  reduce bound is smaller than the compute cap); small recons stay full.
  Correctness covered by `test_single_device_streaming_matches`.
  - **Caveat / next:** the benefit only applies on the *sharded* path — a plain
    single-GPU `back_project` (no `configure_sharding`) still uses the old
    non-streaming path.  To stream one GPU today: `model.configure_sharding(
    jax.devices('gpu'))`.  Making it fully automatic = auto-configure a trivial
    1-device mesh by default (the deferred "trivial-sharding unification"); a
    separate, bigger change that retires the old single-device path.
  - **Remaining frontier:** the ~7.3 GB floor (full sino + recon held on-device).
    Beating it needs host↔device streaming of sino/recon (bigger change).

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

**CPU and GPU are both first-class targets.**  We develop on CPU for convenience, 
but design with both in mind and then verify and tune for performance on both
(although memory scaling on CPU is primarily for catching red flags, less for tuning).
CPU sharding is a *real* performance path — the view-sharded fbp_filter measured
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

**Re-sequenced 2026-06-02 onto the placement architecture (`sharding_implementation_plan_v2.md`).**
Done work below (original Step numbering); forward work uses the P1–P6 numbering
from `sharding_implementation_plan_v2.md`.  Phase D is re-opened on the new movement interface.

Done (original sequence):
- [x] Step 1 — Phase 0: scaffolding migration
- [x] Step 2 — Phase A: primitives (transfer, threading, mesh/trivial-sharding)
- [x] Step 3 — Phase B: sharding hooks (new view/slice axes)
- [x] Step 4 — Phase F1: FBP filter (view-sharded, zero comms) — DONE; kernel is
      `tomography_utils.apply_row_filter`, 2× memory floor, B=1024, H100-validated
- [x] Step 5 — Phase D: back projection (reduce-scatter) — DONE, GPU-validated
      (slice-band streaming).  **Being re-opened in P2** on the placement/movement
      interface; current code is the baseline to compare against.
- [x] Step 6 — Phase F2: direct_recon (match-input pipeline) — DONE, GPU-validated;
      baseline captured.  Public-API layer; survives the re-open.

Forward (placement architecture — see `sharding_implementation_plan_v2.md`):
- [x] P1 — Placement foundation (Placement + move_cylinders_to_sino /
      sum_cylinders_to_recon; model builds recon/sino placements) — DONE, committed
- [x] P2 — Back projection on placements — DONE: kept band (pixel retired; B_p
      sweep showed pixel's ~2× memory gap is untunable).  Finalization committed?
      — see top handoff.
- [ ] P3 — Forward projection on placements (C; adjoint test) ← NEXT
- [ ] P4 — VCD on placements (E)
- [ ] P5 — Device-config UX (`configure_devices`, auto-select, divisibility warnings)
- [ ] P6 — Port geometries + retire `main_device`/`sinogram_device`

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
