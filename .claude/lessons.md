# Engineering lessons (mbirjax sharding)

Hard-won lessons from the sharding effort — the F1 sharded FBP filter, the Phase D
back-projection reduce-scatter, the placement architecture, sharded-VCD memory + buffer
donation, and the platform-divergent back kernel — kept as a reference playbook.  The short
jax/perf tips live in `claude_prompt.md` and the general measurement principles in global
memory; this is the detailed version, with the worked examples.

## General principles (apply anywhere)

- **Separate the ruler from the measured.**  When a number looks wrong, suspect
  the harness/measurement before the code.  In F1, most "memory problems" were
  measurement artifacts, not the kernel.
- **Trust authoritative introspection over end-to-end numbers.**  XLA's
  `compiled.memory_analysis()` (temp = 94 MB) and the compiled HLO proved the
  kernel was already at the 2× floor — one number redirected the whole hunt from
  the kernel to the harness.
- **Decide with cheap, single-variable ablations.**  Vary one thing, hold the
  rest fixed.  Examples here: a jit-on/off microbench isolated the scaling cause;
  preallocate true-vs-false on one config verified memory honesty; the
  divisibility "smoking gun" (1624² ÷ 64 but not ÷ 256) attributed a memory step
  to padding.
- **Name the assumption a fix rests on, then verify it.**  "The scan updates its
  carry in place — confirm on GPU" held; "the body/tail output concat might not
  fuse" came true.  Stating the risk caught the dead end fast.
- **Sweep parameters; don't guess defaults.**  B=1024 and the c-dependence came
  from the B-sweep, not intuition.

## jax / GPU specifics

- **Jit per-worker compute.**  The per-device kernel had lost `@jax.jit` → eager,
  op-by-op dispatch → ~2× instead of ~6.5× scaling.  Every threaded worker's
  compute must compile.
- **Bound work-area memory by a fixed batch, not geometry.**  per_view's FFT
  batch = `view_batch_size × n_rows` grew with the detector and OOM'd at 1624³;
  the row-batched kernel made memory depend on `B` alone (ROW_FILTER_BATCH),
  independent of geometry and device count.
- **`peak_bytes_in_use` is a process-cumulative high-water mark.**  It never
  resets within a process, so honest per-config memory needs a fresh subprocess
  per config — and the orchestrator must touch no JAX (or it holds device memory
  while a worker measures).  `preallocate=true` reports the *same* peak (it tracks
  in-use tensors, not the pool) and avoids the per-call `cudaMalloc` growth that
  inflates timing.
- **Free the previous result in a timing loop.**  Holding the prior output while
  the next allocates over-reports memory by a full shard and causes allocation
  thrash (which also inflated timing).
- **Watch for hidden memory use.**  `np.pi` is float64;
  `f32_array * f64_scalar` (out of place) promotes the whole array to f64 →
  doubles memory (the f64 OOM).  Fold the scalar into a small operand (the f32
  filter); convolution's linearity makes it free.
- **Avoid hidden full-shard copies in batched kernels.**  Both an input zero-pad
  `concatenate` and a body/tail output `concatenate` each silently re-add a full
  shard.  A `lax.scan` over overlapping windows that writes in place via
  `dynamic_update_slice` (clamp the last window in-bounds; the overlap is
  idempotent for a per-row op) avoids both — hitting the input+output floor.
- **`lax.map(batch_size=…)` is unsafe for large batches (jax#27591).**  Supply the
  parallelism with `vmap` and scan with no `batch_size` to stay immune.
- **CPU sharding is a REAL speedup — take it seriously (users run on CPU).**
  Measured ~5.4–6.5× at 8 virtual CPU devices (device sweep).  The filter is
  embarrassingly parallel across views, and `run_per_device` runs one thread per
  virtual CPU device with the GIL released during XLA execution, so N independent
  shard-streams genuinely spread across cores — extracting parallelism a single
  CPU device's intra-op threading does not fully get for this scan/FFT workload.
  (Don't assert "CPU sharding won't help" from a model — the data says it does;
  this is the ruler-vs-measured lesson, self-inflicted.)  Caveats: on CPU the
  memory metric is whole-process RSS, not the per-device 2× floor, and RSS grows
  with device count — CPU sharding trades some memory for speed.  This is all
  SINGLE-PROCESS (the cores of one machine); spanning multiple nodes still needs
  multi-host JAX (`jax.distributed`).

- **Per-subset sharding has a problem-SIZE floor — measure VCD at realistic sizes,
  not toy ones (ruler lesson, self-inflicted).**  VCD reuses the same sharded
  projectors but calls them over a *subset* of pixels each update
  (`recon_pixels / num_subsets`), so the effective work per sharded op is far
  smaller than a full projection — VCD needs a *bigger* recon than the bare
  projectors to amortize the per-device + per-subset overhead.  A first
  shard-vs-noshard demo at 64³ showed VCD ~0.35× at 4 CPU devices and I wrongly
  called it "expected sharding overhead."  It was a too-small ruler: the bare back
  projector is itself only 0.44× at 4 dev at 64³ (it doesn't beat 1× on CPU until
  ~256³).  Measured VCD CPU scaling (4 virtual devices): 64³ ≈ 0.35×, 128³ ≈ 0.86×,
  **256³ ≈ 1.70×** — right in line with the projectors' CPU ceiling (back ~1.9×,
  forward ~1.6× at 256³).  So sub-256³ "slowdowns" are the size, not a defect; never
  conclude "sharding doesn't help here" from a toy size.  Full CPU device curve at
  256³: 2d 1.63×, **4d 1.86×**, 8d 1.73× — 2-dev already wins, **4 dev is the peak,
  8 dev regresses** (bandwidth-bound on the shared CPU memory bus, like back-proj);
  384³ is slightly worse (4d 1.75×, 8d 1.41×) as larger working sets saturate the bus
  sooner.  ~4 devices is the CPU sweet spot.
- **VCD's sharded qGGMRF prior is the one per-subset component that goes BACKWARDS
  at fine granularity** (attribution at 256³/4-dev: prior 0.47× on 1024-pixel
  subsets vs 1.45× on 16384-pixel subsets; back/forward stay ≥1.3×).  Cause: its
  host-halo extraction (`_extract_halos` ≈ 1.35 ms/call) + the per-shard Python
  dispatch + `assemble_sharded` don't amortize when the actual prior compute is ~2
  ms.  It's only ~8% of per-subset cost (the projections dominate), so it drags
  overall VCD scaling only slightly — but it's the obvious optimization if the prior
  ever dominates: avoid the per-subset host round-trip (on-device `move_shard` halo
  exchange where d2d is safe, or fuse the halo read across subsets).
- **GPU watch for VCD:** the line-search `alpha` is reduced to a host float each
  subset (5 `float()` device→host syncs/subset) to dodge the cross-mesh scalar
  problem (forward scalars on the sino mesh, prior scalars on the recon mesh).
  Cheap on CPU; potentially serializing on GPU.  If it bites, replicate `alpha`
  onto both meshes with `device_put` (a couple of scalar broadcasts) instead of
  bouncing through the host.  **Confirmed on GPU (2026-06-05) via a full op sweep — and it is VCD-SPECIFIC, not a
  size floor.**  On the SAME H100×4 allocation, at 4 dev every bare op scales (256³:
  back 2.51×, direct 2.78×, forward 4.00×; 512³: fbp 3.10×, back 3.33×, direct 3.65×,
  forward 4.75×) while vcd_recon alone is 0.28× (256³) / 1.62× (512³).  So "256³ is too
  small for GPU" is FALSE — the projectors scale there; VCD doesn't because it calls
  them per subset (`recon_pixels/num_subsets` pixels), so each op does num_subsets× less
  work against a fixed per-subset host overhead.  CPU clinches that the overhead is host
  round-trips: at 256³/4d VCD scales like the others on CPU (2.10×) but collapses on GPU
  (0.28×).  Fix = A/B/alpha (cut the per-subset host round-trips); target = move GPU
  512³/4d from 1.62× toward the projectors' ~3.3×.
- **`CUDA_ERROR_NOT_PERMITTED` from `cuda_vmm_allocator.cc` ("VMM cuMemCreate with
  FABRIC+POSIX_FD handle types failed … will retry with simpler handle types") is a
  BENIGN warning, not a failure — don't chase it.**  XLA's VMM allocator probes
  advanced memory-handle types (multi-node NVLink fabric / fd-based IPC) on the first
  multi-GPU allocation; when the job's environment forbids them it falls back to
  simpler handles and the allocation + any collective succeed (verified: a standalone
  multi-GPU `jnp.sum` emits the warnings AND returns the right value; the full run
  exits 0 with correct output).  It's a `W` line; real errors are `E`/`F` or a Python
  traceback.  Silence with `TF_CPP_MIN_LOG_LEVEL=2`.  (I twice mis-attributed it — to
  an orchestrator JAX-import and then to a blocked P2P collective — before the
  standalone ablation showed it's noise.  Distinguish warning-vs-fatal FIRST.)
- **On GPU, sharding is primarily a CAPACITY tool, not a speed tool.**  Per-device
  memory drops a lot with more devices (1 device holds full transients; multi-device
  streams in bands), so you shard to fit a bigger recon, and any speedup past the
  crossover is a bonus.  Don't read inverted *time* at a fits-on-one-GPU size as a
  defect — read whether *memory* shards (it does) and whether time scales *above* the
  crossover (it does: 512³ 1.65×@2d).
- **`peak_bytes_in_use` is the REAL live working set and is preallocation-INVARIANT;
  `PREALLOCATE=false` does NOT reveal the capacity floor — restrict the budget instead
  (ruler caution, Greg; I initially got this backwards).**  Verified: rerunning with
  preallocation off gave the SAME peak (1d 512³ ≈ 41.6 GB either way).  `peak_bytes_in_use`
  is the peak of *live* (in-use) bytes, not the reserved pool, so preallocation doesn't
  change it.  Under a *generous* budget XLA does no rematerialization, so this peak is the
  natural full-speed working set — a real number (so the ~10× 1d→Nd drop from band-streaming +
  sharding is genuine, NOT a preallocation artifact; my earlier "preallocation over-reports
  the 1d number" was wrong).  Caveats that DO hold: don't extrapolate absolute MB across sizes
  (the natural working set isn't a clean ×k in size), and to find the true **capacity floor /
  OOM threshold** you must ARTIFICIALLY RESTRICT the budget — keep `PREALLOCATE=true` and LOWER
  `XLA_PYTHON_CLIENT_MEM_FRACTION` (a hard pool cap, e.g. 0.25) so XLA rematerializes to fit;
  a size that then OOMs is the honest max-recon-per-GPU.  (`MEM_FRACTION` is ignored when
  `PREALLOCATE=false`, the other reason False is useless for this.)  See the continue-past-OOM
  sweep (`sparse_back_project_single_device_sweep.py`) and per-buffer attribution
  (`sparse_back_project_memory_attribution.py`); `vcd_recon_scaling.py` exposes `MEM_FRACTION`.
- **An OOM can surface as an unrelated-looking error — classify it from the FULL traceback,
  and don't let a harness swallow the stack.**  A 1008³/1d VCD run failed with numpy's
  "setting an array element with a sequence" and `oom=False`, because the real
  RESOURCE_EXHAUSTED was deeper in the stack and the harness only stored `str(e)[:300]`.  Fix:
  record `traceback.format_exc()` and match OOM markers against the whole traceback, not just
  the top-level message.

## Phase F1 case study (the arc)

The sharded FBP filter, start to finish — a template for "make it correct, then
fast, then honest":

1. **Scaling gap → jit.**  Beta scaled ~2× vs research's ~6.5× on CPU.  A
   jit-toggle microbench (eager vs jitted, view vs row axis) isolated the cause:
   the kernels had lost `@jax.jit`.  Restoring it closed the gap and ruled out
   the shard-axis and kernel-structure theories.
2. **Memory blowup → work area.**  On the H100, 1624³ OOM'd at 1 device.  Root
   cause: per_view's FFT work area = `view_batch_size × n_rows`, geometry-bound.
   Fix: the row-batched kernel — bound the FFT batch by `B` alone.
3. **The reshape, done right.**  First a whole-array zero-pad (a full-shard
   transient), then body/tail + concat (traded the input copy for an output
   copy — net zero, caught by the data), finally an overlapping-window scan
   writing in place (the floor).  Each step verified on the H100.
4. **f64 OOM.**  The `* (np.pi/num_views)` post-multiply promoted f32→f64 (the
   `jit_multiply` OOMs).  Folded both scalars into the f32 filter.
5. **The ruler was lying.**  The remaining "3.5× memory" and "row_batch slower
   than per_view" were harness artifacts: the timing loop held the previous
   result (over-reporting by a shard + allocation thrash).  `memory_analysis()`
   proved the kernel was at 2× — the fix was in `time_op`, not the kernel.
6. **Pick B by sweeping.**  The B-sweep gave the knee (1024): ~10× faster than a
   small batch, with the benefit past 1024 shrinking as channels grow while the
   work area grows — pointing at a future c-aware `B ≈ budget/fft_len(c)`.
7. **Consolidate.**  One kernel (`tomography_utils.apply_row_filter`); the switch
   and losing kernels removed; `view_batch_size` deprecated on user-facing
   methods.

## Phase D case study (back projection — reduce-scatter + streaming)

1. **View-sharding makes back projection a reduce-scatter, and it's
   memory-bandwidth-bound.**  Each device back-projects its views onto the full
   slice range; partials are summed to the slice owners.  On CPU it caps at
   ~1.5× (the virtual devices share one memory bus — a bandwidth limit, not a
   core limit), while the *same harness* scales `fbp_filter` ~7× (compute-bound).
   On GPU, where each device has its own HBM, it scales near-ideal (3.92× on 4).
   **Bandwidth-bound ops scale on real per-device hardware, not virtual CPU
   devices** — don't chase CPU scaling for them.
2. **Attribute the bottleneck with a phase ablation.**  Timing the per-device
   compute vs the reduce-scatter *separately* showed the cap was the compute
   (bandwidth); the reduce-scatter was ≤9%, so comms optimizations were a dead
   end.  A single-variable ablation beats guessing (cf. F1's jit-toggle).
3. **Balanced tiling, NOT the F1 overlapping tail.**  F1's overlapping-window
   tail is right when the per-unit work is a cheap, idempotent row filter.  A
   back-projection band is an *expensive projector call*, so an overlapping tail
   nearly doubles a band's compute (130 slices, B=128 → recompute 126).  Use
   balanced equal bands (zero recompute).  Same ragged-tail problem, opposite
   answer, because the cost structure flipped.
4. **The band length must be device- AND compute-aware.**  The owner's reduce
   gather is `n_dev × num_pixels × band`, so band must shrink with `n_dev` (a
   per-owner-sized band re-gathers the whole cylinder).  A separate *compute*
   working-set bound (n_dev-independent) is what makes a **single** device
   stream; plus a per-band-work floor so small recons don't over-split into tiny
   dispatches.  On GPU, time was flat across band sizes → **smaller band = free
   memory** (single-GPU 1024³: 28 → ~12 GB at no time cost).
5. **Don't port a trick without confirming the *cause* — the in-place-assembly
   miss.**  We assumed the single-device memory plateau (~1.44× the sino+recon
   floor) was the per-owner `jnp.concatenate(band_list)` doubling, and ported the
   F1 in-place donated `dynamic_update_slice`.  Measurement showed peak got
   *worse* everywhere (1024³/4-dev +41%): the concatenate was never the binding
   peak (it runs after the compute phase).  **Reverted.**  Corollaries: a clean
   CPU "no donation warning" did NOT predict the GPU memory result — measure on
   the target; confirm a plateau's cause before optimizing it.
6. **Reuse what the kernel already does.**  Parallel-beam back projection needed
   only a one-line kernel fix (size the voxel cylinder from the *input* rows, not
   `projector_params`) so a row-sliced view yields just those slices — the whole
   streaming scheme then reused the existing jitted projector unchanged.

## Placement architecture + GPU-allocation reliability (2026-06-03)

The device-config redesign and the back-projection re-open under it.

1. **One placement abstraction, two roles.**  `recon_placement` / `sino_placement`
   (each a device list + sharded axis + 1-D mesh) replace the scalar
   `main_device` / `sinogram_device`.  Every mode is a placement pair; single
   device is a trivial 1-shard placement, so sharding is "always on" and the
   `mesh is None` branching dissolves.  The hybrid (recon-CPU / sino-GPU) case is
   two *separate* single-device meshes — never one mixed mesh — which sidesteps
   the JAX heterogeneous-mesh fragility.
2. **The only thing that crosses placements is voxel cylinders.**  The sinogram
   is written locally on its view-shard and never moves.  So the whole inter-
   device interface is one adjoint pair — `move_cylinders_to_sino` (all-gather)
   / `sum_cylinders_to_recon` (reduce-scatter) — built on `move_shard`, looping
   over the placements' shards: `N×N` homogeneous, `1×1` single-device, **no
   mode branch**.  Take cylinders directly (the pixel axis is unsharded, so the
   caller slices `flat_recon[pix]`); keeps the primitives pure + unit-testable.
3. **User-facing = match input; internal = sharded-only.**  User-facing methods
   (`back_project`, `fbp_recon`, `direct_recon`, future `forward_project`) match
   their input: plain in → plain out (shard at entry, gather at exit), sharded in
   → sharded out (no gather).  Internal (`sparse_*`, `fbp_filter`/`direct_filter`)
   are sharded in → sharded out, no transition code.  Decided by one
   `isinstance(x.sharding, NamedSharding)` on the primary input, read *before* the
   entry shard.  Why match-input not always-gather: these are dual-use
   (`direct_recon` is `vcd_recon`'s init), so a sharded caller stays on-device.
4. **Pixel-batch vs slice-band back projection — fine grain is fragile.**  The
   slice-banded path fires ~n_dev² × bands small dispatches; the pixel-batched
   path (full cylinders per pixel batch, reduce-scattered) fires far fewer,
   larger ops.  On a clean H100×3 run the pixel path scaled *as-well-or-better*
   (1008³: 3.12× vs 2.75× on 3 dev) but used ~2–2.7× more memory at the default
   B_p; the band path is memory-lean but dispatch-heavy (which also made it more
   sensitive to the NUMA/throttle issues below).  **Verdict (2026-06-03): kept
   band, removed pixel.**  The B_p sweep showed pixel's memory plateaus at
   ~1.7–2.7× band — a floor B_p can't push below (it's the accumulated output +
   concatenate, not the per-batch transient) — for only ~11–16% less time.  With
   memory / max-recon-per-GPU the priority, that gap isn't worth pixel's
   simplicity/speed.  (The genuinely simpler *and* memory-lean option for parallel
   beam is the slice-sharded-sinogram / embarrassingly-parallel scheme, noted as a
   future option — not pixel.)

5. **A throttling GPU masqueraded as a code regression — separate the ruler from
   the measured, hardware edition.**  Band 4-device at 1024³ "regressed" 2.8× vs
   a prior run — but the *same Phase D commit* reproduced it, so it wasn't code.
   The tell-tale signature: **size-dependent (only the largest, sustained load)
   and device-count-specific (only when the bad card joined).**  The phase
   ablation isolated it to Phase 1 (compute), not the reduce-scatter (≤0.2%).
   Root cause via `nvidia-smi dmon`: one GPU at **345 MHz @ 86 °C** while its
   neighbors ran **1980 MHz @ 40 °C** — a thermally-throttling card; the reduce
   waits for the slowest device, so it gated the whole multi-device run.
   - **Idle temperature predicts it:** the warmest-at-idle card (61 °C vs ~30 °C)
     was the one that throttled.  A 10-second `dmon` glance is a cheap pre-flight.
   - **345 MHz alone is NOT throttling** — it's the normal H100 *idle* clock; the
     discriminator is low clock AND high temp (vs low clock + cool = idle).
   - **`DEVICE_COUNTS = [1,2,3]`** sidesteps a known-bad 4th slot (`pick_devices`
     takes the first n) and, on a 2/2-NUMA node, also keeps all three on one
     socket — so 3-on-one-socket is often the cleanest scaling a node can give.
6. **Instrument the ruler.**  The scaling harness now self-records, per run:
   `nvidia-smi topo -m` + GPU UUIDs (allocation/NUMA), `dev2dev_safe` (host-bounce
   state), and a per-GPU clock/temp sample that flags a row `[THROTTLED]` when a
   participating GPU shows the low-clock+high-temp signature.  Turns this whole
   class of afternoon-eating surprise into a one-line annotation in the result.
7. **Gitignored `results/` doesn't survive a handoff.**  Scaling numbers and the
   decisions they drive must be written into committed prose (status / plan /
   here), or they evaporate with the session.

## Sharded VCD memory: jax reference cycles + buffer donation (2026-06-07)

The 1-device-mesh / multi-GPU VCD "memory blowup" (504³/5-iter peaked 25.8 GB vs
6.9 GB single-device; OOM at 1008³) was **not** the band, not pixel-batching, not
async run-ahead — it was object lifecycle.

- **jax holds sharded (`NamedSharding`) arrays in internal reference cycles** (an
  `ArrayImpl`'s `__dict__` references its sharding/buffers with back-refs), so they are
  reclaimed only by the **cyclic GC**, not by refcounting.  Single-device
  (`SingleDeviceSharding`) arrays free on refcount-0 normally.  So any per-subset (or
  per-iteration) **out-of-place** update producing a *new* sharded array leaves the
  stale one alive until a GC runs — and Python's GC is unaware of device-memory pressure
  (it triggers on object counts), so they accumulate one full array per operation.  Peak
  grew with #subset-updates (subsets × passes); `bytes_in_use` at the end was tiny
  (gc-pending) — the tell-tale signature.
- **Fix = donate for in-place state, `.delete()` for transients.**  Update the
  persistent state in place via buffer donation —
  `@partial(jax.jit, donate_argnames='error_sinogram')` returning
  `error_sinogram - scaled_delta` — exactly as `update_recon` already did for the recon
  (that recon-vs-sino asymmetry, recon flat / sino leaking, was the diagnostic tell).
  Explicitly `.delete()` the one-shot **eager-op** transients (the `alpha*delta` scale,
  the `weights*error` product).  **Forward-projection outputs (from `assemble_sharded` /
  `make_array_from_single_device_arrays`) free on refcount** and need no delete — the
  cycle is specific to eager elementwise-op outputs.  Result: peak flat at the
  single-device floor (const 6.9, non-const 7.9, positivity 7.3 GB), time unchanged,
  per-device memory still shards 1/n_dev (1008³ 1d→4d **4.49×** super-linear —
  bandwidth-bound op + smaller per-device working set).
- **Keep the scale eager (don't fold it into the donated jit).**  Folding to
  `error - alpha*delta` lets XLA emit a fused multiply-add → last-bit difference →
  breaks the trivial-mesh bit-exactness test.  A lone subtract can't be FMA-fused.
  **UPDATE (step-4 Option B):** trivial-mesh is *not* bit-exact on GPU anyway — the
  banded sharded path reorders non-associative FP sums vs the monolithic single-device
  kernel, so a ~1 ULP difference is inherent (CPU compiles both identically, so it was
  exact only there).  We therefore **relaxed the trivial-mesh tests to a tight `allclose`**
  (1e-5 single-shot / 1e-4 iterated) and will **fold `alpha` back into the donated FMA**
  during the unification, dropping the separate `scaled_delta` transient + its `.delete()`.
  Lesson: "bit-exact across two differently-compiled kernels" is the wrong invariant on
  GPU; use a tight tolerance and reserve exact-equality for same-kernel / data-movement
  identities.
  **RE-LEARNED 2026-06-11 (settable view params), then settled as a PROJECT RULE (Greg):
  exact equality is NEVER the right gate for COMPUTED floats.**  Not across two models
  (separate executables even for identical programs/shapes; GPU autotuning differs ~1 ULP),
  and not even for one executable run twice (GPU scatter-add atomics reorder summation).
  CPU happens to compile/run deterministically, so a bit-exact test written and passed on
  CPU is exactly the kind that fails first on the GPU suite.  Gate computed floats at the
  suite's single-shot 1e-5 (a genuinely wrong/stale value misses by orders of magnitude).
  MEASURED calibration (2xH100, 2026-06-11): same-executable run-to-run GPU noise on the
  forward projector reached ~8e-6 RELATIVE (~70 ULP; scatter-add atomics reorder hundreds
  of per-detector-element contributions) — so 1e-6 is TOO TIGHT for anything touching the
  projectors; pure elementwise kernels (qGGMRF cylinder) are safe at 1e-6.  Exact equality
  remains correct for exactly two things: (1) DATA MOVEMENT
  identities (shard/gather/assemble round trips, halo extraction, stored-parameter echo —
  bytes in = bytes out, no arithmetic; a tolerance would mask corruption), and
  (2) CONSTRUCTED-ZERO invariants (padded entries == 0.0 — the "exactly inert" spec;
  allclose would hide a leak into the padding, the precise failure the invariant exists
  to catch).
- **Scale-invariant tolerances for sharded-vs-single comparisons (2026-06-19, translation T4).**
  Two refinements to the rule above, both surfaced by a translation Hessian test that "failed"
  at the suite's flat `rtol=atol=1e-5`:
  - **Even CPU is not deterministic ACROSS PROCESSES for the reduce-scatter sum.**  The earlier
    "CPU compiles both identically and stays exact" (above) holds only WITHIN one process.
    MEASURED: the sharded-vs-single back/Hessian difference on CPU is usually EXACTLY 0 but
    occasionally ~1e-7 of the peak — and IDENTICAL across device counts (n=2 == n=4) within a
    process, so it is a per-PROCESS XLA reduction-ORDER / autotuning choice, not per-call
    scatter-add reorder.  Consequence: a sharded-comparison test can FLAKE process-to-process on
    CPU, not just "fail first on the GPU suite."  So the bit-exact-on-CPU assumption is wrong for
    anything crossing a reduce-scatter / all-gather.
  - **A fixed `atol` is a scale-dependent ruler; gate on a scale-invariant relative-max.**  The
    reduce-scatter noise scales with the PEAK magnitude (~1e-7 of it), so a fixed `atol=1e-5`
    silently PASSES a small-magnitude operator (cone Hessian peak ~1.8 → noise ~2e-7 ≪ atol) and
    FALSE-FAILS / flakes a large-magnitude one (translation Hessian peak ~5e3 → noise ~5e-4 ≫
    atol) for the SAME relative noise.  `coeff_power=2` (Hessian) squares the coefficients → the
    largest peak → hit first.  This was a RULER bug, not a sharding error: cone and translation
    had the same relative noise AND the same fraction of near-zero entries; only the value SCALE
    differed.  Fix = gate on `max|out-ref| / max|ref| ≤ TOL` (the shared `tests/sharding/
    conftest.rel_max_err`).  NOT a per-element `rtol` (atol=0): that divides by each element's
    OWN value, so a near-zero entry — whose noise comes from the large terms that cancelled —
    gets a near-zero threshold and the relative diff explodes.  The shared gate is
    `conftest.assert_sharded_allclose`, used for every sharded-vs-single comparison in the
    sharding suite; exact equality / `assert_array_equal` stays for data-movement and
    constructed-zero invariants, and the 1e-6 elementwise qGGMRF kernel band tests keep their
    tight gate (no reduce-scatter, so the peak-scaled noise does not apply).
- **Donation-engagement gotchas.**  (a) Release aliases first: for constant weights
  `weighted_error_sinogram IS error_sinogram`, so `= None` it or donation silently falls
  back to a copy.  (b) Donating 2 inputs for 1 output warns "Some donated buffers were
  not usable" — benign (surplus still freed) but noisy; donate only the in-place state
  and `.delete()` the transient to avoid it.  (c) Bare `.delete()` of an array still
  feeding a pending op risks a silent race — gate it behind one `block_until_ready` on
  the returned state (every transient is upstream of it).  Consolidate all frees into one
  end-of-subset cleanup section after that single block.  (d) **Identity ≠ buffer-ownership
  when freeing a PASSED-IN array.**  Freeing the init-phase sinogram in `vcd_recon` (2026-06-25,
  ~4 GiB/shard reclaimed before the Hessian + loop) first guarded on object identity
  (`placed is not original`) — and broke 4 sweep tests with `RuntimeError: Array has been
  deleted`.  Cause: `device_put`/`move_shard` can return a **no-copy reshard** — a *different*
  `ArrayImpl` that *shares the same device buffers* as the input — when the data already lives on
  the target devices (and `_shard_on_axis` returns the input UNCHANGED on an exact-sharding match).
  So `.delete()` on the "new" object frees the caller's buffers (the caller, e.g. a `recon` sweep
  or `prepare_sino_for_devices`, then re-reads a deleted array).  Correct ownership test: we own
  fresh buffers iff `to_sino` had to TRANSFER the data — host/numpy input, or a jax array whose
  devices are **disjoint** from the placement's devices (`set(x.devices()).isdisjoint(placement.
  devices)`).  An input already resident on (any of) the placement devices may alias → do NOT
  delete.  General rule: before deleting an array derived from a caller-supplied one, prove you
  allocated its buffers (a cross-device/host transfer), not just that you hold a distinct handle.
- **Diagnostic method that cracked it.**  A per-subset/per-iteration `peak_bytes_in_use`
  "memjump" trace showed the view-sharded-sino count climbing 1/op; `gc.get_referrers`
  named the holder (`ArrayImpl.__dict__`); an explicit `gc.collect()` dropped `live_end`
  to one volume (proving gc-pending cycles).  Pitfalls: a too-short config
  (`MAX_ITERATIONS=1`) hid the per-iteration accumulation — the *mesh* peak GROWS with
  iterations (only single-device reaches peak early), suspect the ruler; and
  `gc.get_referrers` can itself pin objects, confounding a GC-frequency test.
- **STALE BUILD chimera.**  A reported 33.4 GB "non-constant-weights leak" was a stale
  GPU binary running the pre-fix code; a fresh `pip install -e .` made it bounded (7.9
  GB).  When GPU memory/behavior contradicts the local tests, **verify the build first**
  (editable installs can serve stale compiled state) — it cost a diagnosis detour.
- **Scaling model (size sweep).**  Projection **time ∝ N⁴ (voxels × views)** — each
  voxel projects to each view — while **memory ∝ N³ (voxels)** (resident sino+recon).
  Doubling linear size is ×16 time but only ×8 memory; the size-sweep TIME ideal curve
  must scale as voxels·views, the MEMORY ideal as voxels.
- **`is_sharded` over `self.mesh is not None`.**  A single `@property` (body
  `self.mesh is not None` now, placement-based later) is the one place to change at the
  mesh→placement migration and the one thing to retire once all geometries shard.  The
  transient-free **cleanup section does NOT retire at unification** — it's inherent to
  host-orchestrated sharded arrays (the reference cycle above); only the *guards* go away.
- **`namedtuple()` inside a function silently defeats jit static-arg cache sharing
  across instances.**  jax registers namedtuples as PYTREES and keys the jit
  static-argument cache on the pytree TREEDEF — which includes the namedtuple CLASS.
  `namedtuple('GeometryParams', names)` called inside `get_geometry_parameters` mints a
  NEW class every call, so two model instances with byte-identical params get DIFFERENT
  treedefs (`tree_flatten(p1).treedef != …p2`) and a shared module-level jit RE-TRACES per
  instance — *even though the params are `==` and hash-equal* (jit compares the treedef
  first and never reaches the value compare).  Symptom: `_jit_fn._cache_size()` grows by
  one per fresh same-geometry instance; the 2nd model's first call pays full trace+compile
  (measured **16× slower**: 201→13 ms once shared).  FIX: build the namedtuple CLASS once
  (cache it by field-name tuple — `ParameterHandler.make_geometry_params`) so the treedef
  is stable.  The diagnostic that cracked it: a `_replace()` copy (same nested field
  objects) HIT the cache while an `==`-but-fresh copy MISSED ⇒ the key is identity-of-type,
  not value.  General rule: **never create a pytree-typed object (namedtuple / custom
  registered node) inside a function that feeds a jit STATIC arg — define the type at
  module level.**  (Also why de-closuring alone did not share the cache until the
  namedtuple classes were hoisted.)

## Platform-divergent back-projection kernel → n=1 GPU short-circuit (2026-06-16)

Cone `back` on ONE device was +126–136% slower than `main` on GPU (propagating to cone
VCD).  Bisecting it produced a clean, surprising result: **the two cone back kernels have
OPPOSITE platform rankings, the sharded driver is free, and the right fix is
platform-gated kernel selection.**

- **The cost is the KERNEL, not the sharded driver.**  Driver-less ablation (call the
  projector functions directly, no thread pool / reduce-scatter / `assemble_sharded`): a
  band-loop tied the FULL sharded path at **1.00× time and 1.00× memory** at n=1.  So all
  the n=1 overhead is the per-view kernel: `back_project_one_view_to_band` (single vertical
  fan) vs `back_project_one_view_to_pixel_batch` (rolled `lax.map`@128 + transpose).
- **Opposite platform rankings (no kernel wins both).**  Driver-less, same batch sizes,
  same inputs, 512³: GPU — band kernel **2.25× SLOWER** than the pixel/rolled kernel; CPU —
  band kernel **~8× FASTER**.  On CPU the pixel kernel hits the documented back-vertical
  cache cliff (×62/×110 at ≥~200³): the `lax.map`+transpose UNDER the view-`vmap` is a
  fusion barrier → XLA materializes the full `(views × npix × slices)` per-view stack →
  cache-thrash.  It is WARM execution (call-1 ≈ call-N; compile ~0.4 s), and **band-size
  independent** (B128 ≈ Bmono), i.e. the `lax.map` itself, not the band length.  The band
  kernel (no `lax.map`/transpose) keeps the `vmap+sum` fused → no cliff.
- **Fix: GPU-only n=1 short-circuit** (`_sparse_back_project_sharded`): a single-GPU mesh
  routes to `_sparse_back_project_single_device` (the pixel kernel) and wraps its output as
  a 1-shard slice-sharded array (`assemble_sharded`, metadata only — validated bit-identical
  + same sharding to the band path).  Gated on `recon_placement.devices[0].platform == 'gpu'`
  (the codebase's GPU test, tomography_model ~247) AND `view_indices is None`.  CPU keeps the
  band path (cliff-avoidance).  Net = **pixel kernel on GPU, band kernel on CPU** = the
  platform-optimal choice.  NOT a clean mirror of the forward n=1 fix: forward's single-call
  was win-win (time + memory); back's is a time win that gives up only the *bonus* capacity.
- **Memory: the band kernel streams (capacity), the pixel kernel doesn't.**  At 1024³ the
  `MAX_BAND_WORK` cap makes the band path stream the slice axis → peak **12.5 GB vs 21 GB**
  for the pixel path.  But the pixel path's peak **= main's exactly** (B-back ≤ main-back at
  every size), so GPU→B matches main's single-GPU capacity (which does 1024³); it forfeits
  only the banding *bonus* headroom over main.  Watch VCD capacity via the perf-tracking tool
  rather than gate on it.
- **Ruler bug that hid all this for two sessions.**  The first A/B reported 1.45 and a
  "+56% kernel residual"; both were a measurement artifact — B was timed with a FRESH HOST
  sinogram built INSIDE the timing lambda, so it paid a host→device transfer every call.
  Feeding the already-on-device sinogram (measure the op, not the scatter) gave A/B = **2.25**
  and **B ≈ main (+1%)** — the kernel is GPU-NEUTRAL (vindicating the B2 claim), and the whole
  penalty is the band kernel.  Also: an apparent "A2 (n=2) cliff / minutes-long compile" was a
  HARNESS artifact (two full models alive per size → swap), not the code — per-call timing in
  one process (call-1 = trace+compile, call-N = warm) is the clean instrument; use it to label
  every number first-call vs warm.
- **Multi-device consequence (GPU-confirmed, `run_performance_local.py`).**  Making n=1 the fast
  pixel kernel makes the back-projection device curve NON-monotonic: n=2 (band kernel at 2.25/2 per
  device) is SLOWER than n=1, with a crossover at **n≈2.25** — so you need **≥3 GPUs before sharding
  back pays in TIME** (it always pays in MEMORY).  Confirmed: 512³ back n=1/n=2/n=4 = 648/741/355 ms.
  At the WORKLOAD level VCD stays monotonic (n=2 1.18–1.26×, n=4 ~2.0–2.1×) because the parallel
  forward masks the back crossover — so the short-circuit is a clean VCD win (faster n=1, scaling
  intact).  **Capacity:** n=1 1024³ nonc VCD = **74 GiB** (fits a ~79 GiB H100, ~5 GiB margin) ≈ main;
  the band path's ~10 GiB n=1 headroom is given up, so single-GPU is effectively capped ~1024³ and
  >1024³ uses n≥2 (DECISION (a): keep the simple short-circuit; sharding is the capacity tool, the
  memory-aware pixel-vs-band-by-fit variant was deferred).  **The band kernel's GPU cost is now the
  limiter of multi-device back scaling — the real B4.5 lever** (make the band kernel GPU-competitive,
  e.g. a rolled/pixel-like internal structure, WITHOUT reintroducing the CPU cliff).

## Exactly-inert cone slice padding (B5) + device-form footguns (2026-06-18)

Turning on cone slice padding (a non-dividing slice count → zero-padded device form) was a
small load-bearing fix plus several device-form contract subtleties.  The recurring theme:
**a per-slice operation written for the REAL slice count silently breaks when handed the
device-form (padded) array, and the failure mode depends on whether the wrong length crashes,
contaminates, or NaNs.**

- **The forward gather must crop to the real slice count.**  Cone's sharded forward GATHERS the
  full slice cylinder onto each view-owner and runs the MONOLITHIC kernel (decision C), which
  anchors its slice→detector-row geometry on `recon_shape[2]` (REAL) and ASSERTS that length.
  The gather assembled the device-form (padded) cylinder → shape assertion crash.  Fix:
  `full_cyl = full_cyl[:, :recon_placement.real_size]` before the kernel.  EXACT because the
  padded slices are zero (forced-zero invariant), so cropping them is identity on the result;
  a no-op at dividing counts.  This is the cone analogue of the masks elsewhere — the one site
  that makes the gather-forward inert.  (The BACK projector already handled padding: the banded
  kernel's global clip `k_global < S_real` + `_mask_padded_slices`.)
- **Keep internal sharded methods device-form; crop at the boundary.**  The instinct to "make
  `sparse_back_project` return the real shape" is WRONG: the VCD loop and `output_sharded` need a
  slice-shardable padded array, and a non-dividing real count (e.g. 14) cannot shard across 4
  devices.  The device form is the internal contract; the USER-facing `back_project` gathers+crops.
  Callers that want the real shape crop `[:, :num_real_slices]` themselves (the pad is inert zeros).
- **Pre-sharding tests carry a LATENT device-form assumption — exposed only at a non-dividing
  count.**  `verify_adjoint`/`verify_hessian` did `x = uniform(bp.shape)` then `reshape((-1,
  real_slices))` and `AtAx.reshape(real_shape)` — fine when `bp` is real, a crash/garble when it is
  device-form.  This is GEOMETRY-AGNOSTIC: parallel fails identically at 3 devices (40 slices → 42),
  hidden only because 40 divides the usual 2/4.  Single-variable ablation (run the SAME test at a
  device count that does NOT divide the axis) is the cheap way to surface these.  Corollary: pinning
  the test to 1 device would have *masked* a real latent bug — fix the contract, don't dodge it.
- **The back projector assumes padded views are ZERO — a hand-built device-form input can violate
  it.**  `verify_adjoint` built a random `y` over the device-form sinogram (nonzero padded-view
  tail).  Back-projecting that tail at the clamped padding angle contaminated `Aᵀy` (~3% adjoint
  gap) — NOT a crash, a quiet numerical error, and only at view-padding counts.  Production never
  hits it (entry placement zero-fills the tail).  Fix in the test: zero `y[real_views:]`.
- **`helical_fdk_z_weight`: a per-slice weight built at the real count, applied to the device form.**
  It already handled the SINOGRAM's view-padding (real `num_views`) but built the per-recon-slice
  `z_weight` at `recon_shape[2]` (real, 11 for the small helical) and multiplied the device-form
  (padded, 12) recon → broadcast crash.  Two-part fix: build `z_weight` over the recon's
  device-form slice length with the z-anchor on the REAL count (the anchor rule), AND force the
  weight to 0 on the padded slices — because an out-of-coverage padded slice has coverage 0 →
  `num_views/0 = inf`, and `0 * inf = NaN` would poison the otherwise-inert zero padding.  No-op at
  dividing counts (which is why helical passed in `test_cone_sharded`).
- **Device-form-coincidence footgun in `_pad_shard_on_axis`.**  It treats an input whose sharded-axis
  length already equals `padded_size` as ALREADY device-form (passes it through `_shard_on_axis`, no
  zero-fill).  So feeding a wrong-sized REAL array whose length happens to coincide with the device
  form (e.g. a 7-slice geometry's real recon = 8, equal to the n=2 padded form of a different
  7-slice model) is silently accepted, and its "padding" slices keep their real (nonzero) values.
  Bit me in a test (used the wrong model helper to build the recon).  Lesson: build test arrays from
  the SAME model under test (`self._make_model()`, `model.get_params('recon_shape')`), and don't
  hardcode a real slice count (helical ≠ num_det_rows) — read it from params.

## Tooling / harness

- **A modern `pip install -e` overrides `PYTHONPATH` — prepending a checkout to `PYTHONPATH` does NOT
  select it.**  Editable installs (setuptools ≥64 / PEP 660) register a `sys.meta_path` finder, which
  Python consults BEFORE the `PYTHONPATH` path-finder, so `import mbirjax` resolves to the
  editable-installed checkout regardless of `PYTHONPATH` (proven: a decoy `mbirjax` on `PYTHONPATH` was
  ignored).  This silently bit the metrics harness' `add_run.sh` — it pointed the engine at a ref's
  worktree via `PYTHONPATH` but measured whatever mbirjax was editable-installed in the active env (the
  dev checkout), mislabeled as the ref.  To select code under test you must `pip install -e <worktree>`
  into a DEDICATED env (re-points the finder) — never the user's dev env (deleting the worktree
  afterward would break its install).  Shared now via `mbirjax_metrics/tooling/regression/lib_env.sh`.
- **Seed any global-RNG partition creation before a cross-config comparison.**  `QGGMRFDenoiser.denoise`
  builds its VCD pixel partitions with `np.random` (the library's own `test_denoiser.py` calls
  `np.random.seed(0)` for this).  When the metrics harness first measured `denoise` across device counts,
  the sharded-vs-n=1 fingerprint reldiff was ~1e-4 — 100× the ~1e-7 float floor and enough to false-trip
  the gate — purely because each call drew DIFFERENT partitions.  Seeding `np.random` before each call
  (now in `run_denoise`) collapsed it to ~1e-7: it isolates the dimension under test (device count) from
  RNG variance, and also makes the fingerprint reproducible across runs/platforms (so vs-main /
  cross-platform are meaningful).  General rule: if an op's result depends on a global RNG, fix the seed
  or you are comparing noise.  (The projection `vcd_nonconst` avoids this by passing PRE-BUILT partitions.)
- **uPlot's built-in log-axis tick generator can freeze for seconds on tight, non-power-of-10 bounds —
  hand it your own splits.**  On a `distr:3` (log) axis whose scale min/max aren't round powers of ten
  (e.g. a 6%-log-padded y-min of `9.76e-5`, just under `1e-4`), uPlot's splitter seeds its increment from
  `pow(10, floor(log10(min)))`, then as it crosses the first decade boundary the increment degrades to a
  tiny NON-power value whose internal decimal-places lookup misses — so it crawls the range in millions
  of micro-steps, building a giant tick array.  Result: a ~2.5s main-thread freeze that leaves the panel
  half-drawn (`axis._splits` null) — reads exactly like "the plot disappeared."  It surfaced only for the
  GPU `parallel/back` TIME panel (a new `513³` run's timings happened to land the padded y-min in that
  spot); CPU / other ops / linear axes were fine, which made it look purely data-specific.  It WAS
  data-triggered, but the latent bug was the dashboard trusting uPlot's auto-splitter.  Tell: the X axis
  never hung because it already passes custom `xSplits` (the size ticks), which bypasses the generator —
  the Y axis didn't.  Fix: give yLog axes explicit ticks too — `logTicks(mn,mx)` = `1-9·10^k` within the
  range (O(decades), bounded) — placed in the shared `linePlot` wrapper, so all four log panels (scaling
  time+mem, history time+mem) inherit it; the X axes were already covered.  General rule: when ONE data
  shape freezes a charting lib's log plot, suspect its auto-tick / auto-range generator and feed it
  bounded ticks rather than massaging the data.  (Diagnosis aid: see [[dashboard-verify-gotchas]] — the
  rAF-throttle trap nearly hid this, since uPlot defers its first draw to a `requestAnimationFrame`.)
- **An op-specific, platform-specific slowdown with the sibling ops flat is a TOOLCHAIN regression —
  bisect the jax version, not your diff.**  2026-06-27: GPU `forward` (and `vcd`, which calls it) ran
  3–9× slower across EVERY geometry/size/device-count, while `back`/`direct_filter`/`denoise` were
  byte-identical and CPU `forward` was unchanged.  The op-specificity (forward's GEMM-heavy kernels vs
  back's custom scatter) + GPU-only pattern ruled out thermal throttling (SM/HBM clocks pinned at full
  boost in both runs) AND the measured code.  The decisive ablation: checking out the fast mbirjax commit
  **and** the matching `mbirjax_metrics` commit on the GPU STILL reproduced it ⇒ environment, not code.
  A clean reinstall had bumped `jax`/`jaxlib` 0.10.1 → 0.10.2; downgrading ONLY those (CUDA 12.9 / cuDNN
  92302 / cuBLAS 120900 unchanged) restored forward to 33.3 ms @ 200³ n=1 (from ~170 ms).  So jaxlib
  0.10.2's XLA regressed the forward GEMM path.  Playbook: (1) op- + platform-specific with siblings flat
  ⇒ suspect the compiler/runtime, not the code; (2) the cheap discriminator is to pin the code and
  downgrade jax one release at a time (`pip install 'jax[cuda12]==<prev>' 'jaxlib==<prev>'`) and
  re-measure; (3) `tooling/scaling_tests/measure_one_cell.py` reproduces ONE cell group in ~30 s for this
  bisect, and prints `toolchain_info()` so the jax/CUDA/cuDNN/cuBLAS stack is on every line; (4) the
  `toolchain` field is now in each regression YAML, so the NEXT drift is a one-line diff, not a multi-day
  hunt.  Mitigation when it recurs: pin jax/jaxlib in the regression env for a stable mbirjax baseline (+
  re-baseline deliberately on a bump), and report the upstream XLA regression.
