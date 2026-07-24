# Engineering lessons (mbirjax sharding)

Operative rules from the sharding effort, organized by development question.  Each entry: the rule,
the mechanism, the tell, and a code pointer as the worked example.  Full discovery narratives (the F1
and Phase D case studies, per-entry history) live in this file's git history (pre-2026-07-03
consolidation).  The general measurement principles also live in global `~/.claude/CLAUDE.md`; the
short jax/perf tips in `claude_prompt.md`.

## Contents

1. [General principles](#1-general-principles)
2. [Float correctness gates](#2-float-correctness-gates)
3. [Writing sharded / jitted code](#3-writing-sharded--jitted-code)
4. [The 2^31 / dtype boundary](#4-the-231--dtype-boundary)
5. [Measuring honestly (the ruler)](#5-measuring-honestly-the-ruler)
6. [Performance expectations](#6-performance-expectations)
7. [Out-of-pool GPU allocations](#7-out-of-pool-gpu-allocations)
8. [Known-benign warnings](#8-known-benign-warnings)
9. [Tooling / harness](#9-tooling--harness)

## 1. General principles

- **Separate the ruler from the measured.**  When a number looks wrong, suspect the
  harness/measurement before the code — a large fraction of "bugs" live in how something is measured.
- **Trust authoritative introspection over end-to-end numbers.**  `compiled.memory_analysis()`,
  compiled HLO, `memory_stats()`, `nvidia-smi dmon` attribute a problem before you optimize it.
- **Decide with cheap, single-variable ablations.**  Vary exactly one thing; design the smallest
  experiment that discriminates between competing hypotheses.
- **Name the assumption a fix rests on, then verify it** — rather than assuming.
- **Sweep parameters; don't guess defaults.**  Let data pick the knee/threshold.

## 2. Float correctness gates

- **Exact equality is NEVER the right gate for COMPUTED floats** (project rule, Greg).  Two separate
  executables differ ~1 ULP even for identical programs (GPU autotuning); one executable run twice
  differs on GPU (scatter-add atomics reorder sums); and CPU is deterministic only WITHIN a process —
  the reduce-scatter summation order varies process-to-process, so "bit-exact on CPU" tests flake.
- **Gate sharded-vs-single comparisons on a scale-invariant relative max:**
  `max|out − ref| / max|ref| ≤ tol` (`tests/sharding/conftest.rel_max_err` /
  `assert_sharded_allclose`).  NOT a fixed `atol` (scale-dependent: passes small-magnitude operators,
  false-fails large ones for the same relative noise) and NOT per-element `rtol` (near-zero entries,
  whose noise comes from large cancelling terms, get near-zero thresholds).
- **Calibration:** same-executable GPU run-to-run noise on the projectors reaches ~8e-6 RELATIVE
  (~70 ULP, atomics), so anything touching the projectors gates at 1e-5 single-shot / 1e-4 iterated;
  pure elementwise kernels (qGGMRF cylinder) are safe at 1e-6.  The noise SOURCE is the forward
  projector's `.at[n].add` scatter (atomic float adds; arrival order varies with GPU scheduling);
  plain reductions are deterministic.  Diagnostic discriminator: rerun once with
  `XLA_FLAGS=--xla_gpu_deterministic_ops=true` — a diff that vanishes is atomics noise, not a bug
  (verified: forward_project run-vs-run 5.7e-6 → exactly 0 under the flag; too slow to leave on).
  Full-pipeline comparisons (e.g. two MAR recon_plastic_metal runs) accumulate this per-op noise and
  pass it through DISCRETE selectors (the BH constraint argmin), so their cross-run spread is
  context-dependent (measured 6e-7..1.1e-4) — gate such comparisons at 1e-3, and test CONTRACTS
  (types/shapes/sharding) with strict asserts instead of tight value comparisons.
- **Exact equality remains correct for exactly two things:** (1) DATA-MOVEMENT identities
  (shard/gather/assemble round trips, halo extraction, stored-parameter echo — bytes in = bytes out;
  a tolerance would mask corruption); (2) CONSTRUCTED-ZERO invariants (padded entries == 0.0 — an
  allclose would hide a leak into the padding).
- **Seed any global-RNG dependence before a cross-config comparison.**  VCD pixel partitions draw
  from global `np.random`; unseeded, a sharded-vs-n=1 fingerprint showed ~1e-4 reldiff (100× the
  float floor) purely from different partitions.  `np.random.seed(k)` before each call isolates the
  dimension under test; otherwise you are comparing noise.

## 3. Writing sharded / jitted code

- **Jit per-worker compute.**  Eager op-by-op dispatch silently kills multi-device scaling (a lost
  `@jax.jit` turned ~6.5× into ~2×) and materializes every intermediate at peak memory.
- **Fuse recon-loop statistics into ONE jitted function.**  Each separate eager op on a sharded array
  carries its own full-size temps AND its own collective-buffer allocation; one jitted stats function
  = one executable, one collective set, no temps (`TomographyModel._vcd_iteration_stats`).
- **Bound working memory by a fixed batch/slab, not by geometry.**  A work area proportional to
  detector size OOMs at large geometry; a `B`-bounded batch is geometry- and device-count-independent
  (`apply_row_filter`'s B; the histogram's `_HISTOGRAM_SLAB_ELEMENTS`).
- **Close over HOST (NumPy) constants for device-agnostic kernels** — they auto-promote to each
  batch's device.  But **`@jax.jit` defeats the host-preserving `xp = jnp if isinstance(x, jax.Array)
  else np` pattern**: jit traces the input and always returns a device array, so the branch is inert
  and the volume ships to one device.  Host-preserving functions must run eagerly (de-jit non-hot
  ops) or branch BEFORE the jit boundary.
- **Avoid hidden full-shard copies in batched kernels.**  An input zero-pad `concatenate` and a
  body/tail output `concatenate` each silently re-add a full shard.  Two valid fixes, chosen by cost
  structure: for a cheap idempotent per-row op, an overlapping-window `lax.scan` writing in place via
  `dynamic_update_slice`; for an expensive band (a projector call), balanced equal bands with zero
  recompute — an overlapping tail there nearly doubles a band's work.
- **Never slice or stride a SHARDED axis.**  A non-shard-aligned slice comes back `P()` — fully
  REPLICATED, a complete copy on every device (bit `split_sino_recon`'s `init_recon` half-slice; a
  strided subsample would do the same).  Gather to host at the boundary (`_gather_recon`) or reshard
  explicitly; stride only unsharded axes.
- **GSPMD does not partition scatter.**  `jnp.histogram` AND a global `.at[idx].add` both lower with
  all-gathers of the IMAGE-SIZED index/update arrays onto every device (47 GiB alloc on an ~18 GiB
  sharded recon).  For sum-decomposable reductions with small outputs, enforce the decomposition:
  per-device LOCAL blocks (`image.addressable_shards`, deduped by `shard.index` — a replicated array
  yields one identical shard per device), partials combined on the host
  (`segmentation._sharded_histogram`).  `shard_map` also achieved 0 all-gathers in HLO but its SPMD
  partitioner has produced pathological lowerings here (3–5× slower fbp filter; see
  `plans/sharding/parallel_performance/fbp_parallel_options.md`) — prefer the per-device
  dispatch pattern (dispatch all work before reading any result and the devices overlap without
  threads).
- **The device form (padded arrays) is the INTERNAL contract; crop at user boundaries.**  Internal
  sharded methods stay device-form (a non-dividing real count can't shard); user-facing methods
  gather + crop (`_gather_recon`/`_gather_sinogram`).  Corollaries:
  - Per-slice/per-view operations must anchor on the REAL count, not the padded shape (a bottom
    margin at `[-b:]` zeroes padding instead of real slices; a per-slice weight built at the real
    count crashes on the padded form — and a padded slice with coverage 0 turns `0 * inf` into NaN,
    poisoning the inert padding).
  - Padded entries must stay EXACTLY zero (the inertness spec); statistical reductions exclude them
    via `Placement.real_mask(ndim)` (a tiny broadcastable host mask, None when unpadded).
  - `jnp.ravel_multi_index` needs `mode='clip'` under jit (default `'raise'` demands concrete values).
  - Latent device-form assumptions surface only at NON-dividing device counts — run tests at a count
    that does not divide the axis; and build test arrays from the SAME model under test (a wrong-sized
    real array can coincide with another model's padded form and be silently accepted).
- **Sharded arrays leak via reference cycles — donate in-place state, `.delete()` eager transients.**
  `NamedSharding` arrays sit in internal ref cycles and free only on cyclic GC (which ignores device
  pressure), so out-of-place per-subset updates accumulate one full array per op (tell: peak grows
  with iteration count while end-state `bytes_in_use` is small).  Donate the persistent state
  (`donate_argnames`); `.delete()` one-shot eager transients after a single `block_until_ready`
  (jit/`assemble_sharded` outputs free on refcount and need no delete).  **Before deleting an array
  derived from a caller's, prove you allocated its buffers**: resharding can return a no-copy alias —
  the ownership test is device-set disjointness (`set(x.devices()).isdisjoint(placement.devices)`),
  not object identity.  Release aliases before donating or donation silently copies.
- **Never create a pytree-typed class inside a function that feeds a jit static arg.**  A
  `namedtuple(...)` call mints a new class per call; jit keys its static cache on the treedef (which
  includes the class), so equal-valued params from two instances retrace (measured 16× first-call
  cost).  Define/cache the class at module level (`ParameterHandler.make_geometry_params`).
- **cuSolver-family calls (`jnp.linalg.solve/svd/eig/...`) on small systems → host `np.linalg`.**
  Their workspaces allocate outside XLA's pool (see §7) and the async failure surfaces only at first
  read.  (`jnp.linalg.norm` is pure XLA — fine, but jit it; see §7.)
- **Keep per-call wrappers free of EAGER array ops.**  One eager gather/slice of a device array
  costs ~1 ms of HOST time (jax's eager dispatch path), invisible to device profiles and to
  micro-benches that hit a different argument path.  (Phase D's VCD +35%: ONE eager
  view-params gather per projector call, 547×/recon — the micro-bench used the empty-default
  `owned_view_indices` and measured flat.)  Restrict view/pixel subsets INSIDE the jitted
  program (traced gather); wrappers on hot paths carry an explicit no-eager-ops contract
  (`projectors.py` create_projectors).  Attribution playbook when a loop is HOST-bound
  (device trace shows device-time ≪ wall, e.g. VCD at 200³ is ~95% host): cProfile the warm
  run and diff old-vs-new by cumtime/ncalls — kernel benches and dispatch-count probes
  cannot see it.
- **`lax.map(batch_size=…)` is unsafe for large batches (jax#27591)** — use `vmap` for parallelism
  and scan without `batch_size`.
- **A heterogeneous (CPU+GPU) mesh is fragile** — the hybrid mode is two separate single-device
  meshes (one per placement), never one mixed mesh.

## 4. The 2^31 / dtype boundary

A full-size sinogram/recon (~4.6–4.8e9 elements) crosses int32's 2^31 ≈ 2.1e9; with jax x64 disabled
several mechanisms fail there, mostly SILENTLY.  Treat any element COUNT or FLAT INDEX at this scale
as suspect — and note small phantoms can never reproduce these (size-dependent, not path-dependent).

1. **`jnp/lax.argmin` over a >2^31-element array WRAPS with no warning** (index labels are int32
   regardless of axis length; a min planted at 2.3e9 returned −1,994,967,296 = off by exactly 2^32).
   A scalar read on a >2^31 flat axis separately requests int64 indices, truncated with the
   "int64 ... truncated to int32" UserWarning — the warning is smoke from the read; the fire is the
   argmin.  Fix: never form flat indices on full-size arrays — stage the argmin per axis and carry
   `(view, row, col)` tuples with basic per-axis indexing (`mar._argmin_3d`).
2. **A Python int > 2^31 as a traced operand raises OverflowError at the jit boundary** (weak int →
   int32).  Counts enter traced arithmetic as FLOATS (`float(n)`; same ~1e-7 rounding `jnp.mean` had
   internally); per-run int counts passed to jitted functions are STATIC args, float()ed in the body.
   RECURRED 2026-07-10 (`mar.py`'s padding-aware mean divided by `num_real_pixels` = 3.7e9 on a
   student's full-size run): new count-dividing code must carry the idiom, and the flat-index greps
   (item 1's `argsort`/`searchsorted`/...) do NOT catch this face — grep `/ num_`-style count
   divisions too.  Regression tests need no big arrays: the overflow is in the scalar's VALUE, so a
   tiny array divided by `2**31 + k` pins it (`test_mar.TestCorrectPlasticSinogramBigCounts`).
3. **`np.prod` of a shape accumulates in the platform default int** — int64 on Linux/macOS, int32 on
   Windows/numpy<2 (silent wrap).  Use `math.prod` for element counts.
4. **Integer counting: int32 wraps above 2^31, and f32 scatter-adds of unit counts SATURATE at 2^24**
   (ulp > 1 → +1 rounds to +0; silent undercount).  Count in slabs < 2^31 (int32 exact within a slab)
   and accumulate in int64 on the host (`_sharded_histogram`).
5. **A float64 scalar promotes the whole array** (`f32_array * np.pi` → f64, doubling memory).  Type
   constants to the array dtype or fold them into a small f32 operand.
6. **`np.histogram`'s bin edges depend on the DTYPE of `range`** (f32 scalars → f32-computed linspace;
   python floats → f64-computed then cast; a few ULP apart).  Pass python floats everywhere for
   cross-path edge consistency.

## 5. Measuring honestly (the ruler)

- **`peak_bytes_in_use` is a process-cumulative high-water mark of LIVE bytes.**  It never resets, so
  honest per-config memory needs a fresh subprocess per config, with a JAX-free orchestrator.  It is
  preallocation-INVARIANT (it tracks in-use tensors, not the pool), so `PREALLOCATE=false` does NOT
  reveal the capacity floor — to find the true OOM threshold, keep preallocation and LOWER
  `XLA_PYTHON_CLIENT_MEM_FRACTION` so XLA rematerializes to fit.  For out-of-pool starvation the
  right ruler is `pool_bytes`/`peak_pool_bytes` (the retained reservation), not `bytes_in_use` (§7).
- **Timing hygiene:** free the previous result inside a timing loop (holding it over-reports a full
  shard and causes allocation thrash); feed already-on-device inputs so you measure the op, not a
  host transfer (a fresh-host-input lambda turned a 2.25× kernel gap into a bogus 1.45×); label
  every number first-call (trace+compile) vs warm; per-call timing in one process is the clean
  instrument; don't keep two full models alive per size (swap masquerades as a cliff).
- **An OOM can surface as an unrelated-looking error** — classify from the FULL traceback
  (`traceback.format_exc()`), and never let a harness truncate `str(e)`.
- **A throttling GPU masquerades as a code regression.**  Signature: size-dependent (sustained load
  only) AND device-count-specific (only when the bad card joins).  Discriminator: low clock AND high
  temp (345 MHz alone is the H100 idle clock); the warmest-at-idle card is the culprit — a 10 s
  `nvidia-smi dmon` is a cheap preflight.  The scaling harness self-records topo/UUIDs/clock+temp and
  flags `[THROTTLED]` rows.
- **An op- and platform-specific slowdown with sibling ops flat is a TOOLCHAIN regression — bisect
  the jax version, not your diff.**  (jaxlib 0.10.2 regressed the forward GEMM path 3–9× while back/
  filter were byte-identical; 0.10.1 is the pin.)  Playbook: pin the code, downgrade jax one release
  at a time; `tooling/scaling_tests/measure_one_cell.py` reproduces one cell in ~30 s and prints
  `toolchain_info()`; the `toolchain` field in each regression YAML makes the next drift a one-line
  diff.  Re-baseline deliberately on any jax bump.
- **A driver-level win on a shape-dependent kernel effect must be re-validated END-TO-END
  before shipping — micro benchmarks don't compose.**  Twice in the batching episode
  (2026-07-04) a clean, reproducible 10% driver-level band win (balanced batching; then the
  pixel-width tuning it was re-attributed to) coexisted with a CONSISTENT full-recon LOSS at
  1024³: the real path samples many (pixel-count, band-length) shapes whose width optima
  point in different directions, plus per-shape compile costs the warm micro-benchmark never
  pays.  Tell: the isolated probe and the full-path A/B disagree in SIGN.  Corollary from the
  same episode: XLA lowers scan-over-reshaped-input and scan-of-dynamic-slice to the SAME GPU
  program (byte-identical temps/outputs) — don't hand-optimize between forms XLA canonicalizes;
  full record in `plans/projector_batching/batching_refactor_design.md`.
- **When GPU behavior contradicts local tests, verify the BUILD first.**  Editable installs can serve
  stale compiled state (a "33 GB leak" was a stale binary); and a modern `pip install -e` registers a
  `sys.meta_path` finder that beats `PYTHONPATH` — to select code under test, install it into a
  dedicated env (`mbirjax_metrics/tooling/regression/lib_env.sh`), never point `PYTHONPATH` at it.
- **A bench that constructs a `jax.jit` inside the measured call measures HOST TRACING, not the
  operation.**  A fresh `jax.jit(...)` object retraces on every invocation (the persistent cache
  skips only XLA compilation, not tracing) — in the E4 composed-back preview this inflated a ~1 ms
  weight builder to 1,828 ms (61% of the "composed" time) and earlier masqueraded as an "H100
  precompute oddity" (host tracing is why a Mac M3 looked FASTER than the H100 node).  Tells: a
  warm cost far above the arithmetic/traffic floor; near-identical work at wildly different costs
  (a module-level jit at 6 ms next to a per-call jit at 1,828 ms); cold ≈ warm.  Rule: hoist jits
  to module level (or cache the jitted callable) before timing anything; suspect any "expensive
  precompute" measured through a locally-constructed jit.  Full record:
  `plans/projector_kernels/gpu_headroom_findings.md` (the composed-preview sections).
- **A kernel spike's speedup is NOT the driver's; the two driver killers are host syncs and
  data-dependent launch shapes.**  E4 increment 2: a kernel that spiked 2.13× gated 0.68× in the
  library because the driver (a) pulled a device array to host per view chunk (`np.asarray` — a
  pipeline stall, strictly worse than an eager dispatch) and (b) sized a pallas grid from DATA, so
  every distinct VCD subset changed a cache key → Triton recompile inside the loop (invisible to
  a warm same-input bench; the tell was the JOB wall, 18 vs 6 min).  Rule: derive kernel/launch
  shapes from array SHAPES only (static bounds, padded slots made no-ops), keep the whole per-call
  chain in one cached jit, and gate the LIBRARY path at production shapes — the spike harness's
  glue is not the driver's glue.  Fixed form gated 2.57×; same file, sections above.

## 6. Performance expectations

- **Projection time ∝ N⁴ (voxels × views); memory ∝ N³.**  Ideal curves for size sweeps must use
  these, not a common exponent.
- **On GPU, sharding is primarily a CAPACITY tool; speed is the bonus above the crossover.**  Read
  whether memory shards (it does) before judging time.  Back projection is NON-monotonic in devices
  (the band kernel's GPU cost): n≥3 before sharded back wins on time — but VCD stays monotonic
  because the forward masks it.
- **Platform-divergent kernels are real; platform-gated selection is legitimate.**  The band back
  kernel is ~8× faster than the pixel kernel on CPU (fusion-barrier cache cliff) and 2.25× slower on
  GPU — hence the GPU-only n=1 short-circuit to the pixel kernel.
- **CPU sharding is a real speedup for compute-bound ops (~5–6× on the filter — users run on CPU),
  but bandwidth-bound ops (back projection) cap ~1.5× on virtual CPU devices** (shared memory bus)
  while scaling near-ideal on real per-device GPU HBM.  ~4 devices is the CPU sweet spot; 8 regresses.
- **VCD has a problem-SIZE floor — never judge sharding below ~256³.**  It calls the projectors on
  `pixels/num_subsets` per update, so per-op work is num_subsets× smaller than a bare projection; toy
  sizes show "slowdowns" that are the size, not a defect.  Its per-subset host scalar syncs (the
  line-search alpha) are the GPU-specific limiter; the sharded qGGMRF prior regresses at fine
  granularity (small share of cost — optimize only if the prior ever dominates).

## 7. Out-of-pool GPU allocations

With a near-total BFC reservation, everything allocating OUTSIDE XLA's pool shares the small
remainder — and the pool RETAINS its high-water mark for the life of the process, so late
out-of-pool allocations starve even on an "idle" GPU.  Two sub-classes:

1. **cuSolver-family workspaces** fail (`gpusolverDnCreate ... cuSolver internal error`) even with
   the pool mostly free; the ASYNC dispatch surfaces the error only when the result is first read.
   Policy: small systems solve on the host (§3).
2. **NCCL/collective buffers** (`cuda_vmm allocator ... RESOURCE_EXHAUSTED`): any eager op on a
   SHARDED array that reduces cross-device allocates collective buffers outside the pool, one set per
   separate executable — per-iteration eager stats multiply the exposure.  Policy: one jitted stats
   function (§3).

Diagnostics: check `pool_bytes` (retained reservation), not `bytes_in_use`; the confirming ablation
is lowering the mem fraction.  Confirmed on the cluster: `bytes_in_use` 0.64 GB with `pool_bytes`
83.3 GB at the failure point; 0.94 cleared it.  FIXED: `tomography_model.py` uses
`os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.94')` — conservative headroom,
overridable per-run via the environment (it was a hard-set '0.98' the env var could not override).

## 8. Known-benign warnings

- **`cuda_vmm_allocator ... FABRIC+POSIX_FD ... CUDA_ERROR_NOT_PERMITTED ... will retry with simpler
  handle types`** (W-level): the VMM allocator probing advanced handle types and falling back —
  benign, verified correct results.  Silence with `TF_CPP_MIN_LOG_LEVEL=2`, which mbirjax sets at
  import via `setdefault` — effective only if mbirjax (or the env var) precedes jax's initialization,
  which is why some harness subprocesses still show it.  Distinguish W (warning) from E/F (real)
  FIRST — this one was twice mis-chased.
- **`cpu_aot_loader ... machine features don't match ... could lead to SIGILL`** (E-level, on cache
  load): a stale persistent-compilation-cache entry (mbirjax enables `~/.mbirjax/jax_cache`) from a
  different toolchain/node — or LLVM tuning attributes (`prefer-no-gather`) spuriously compared
  against CPUID features.  A cache rebuild cleared it; results guarded by the metrics correctness
  gates either way.
- **`Some donated buffers were not usable`**: benign (surplus donations are still freed) — avoid the
  noise by donating only the in-place state and `.delete()`ing transients.

## 9. Tooling / harness

- **Gitignored `results/` does not survive a handoff** — numbers that drive decisions get written
  into committed prose (status/plan/here), or they evaporate with the session.
- **uPlot's log-axis auto-tick generator can freeze on tight non-power-of-10 bounds** — pass explicit
  splits (`logTicks` in the shared `linePlot` wrapper); see the `dashboard-verify-gotchas` memory for
  the full diagnosis pattern (rAF-throttle, probe synchronously).
