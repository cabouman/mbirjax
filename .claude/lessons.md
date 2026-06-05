# Engineering lessons (mbirjax sharding)

Hard-won lessons from the Phase F1 work (sharded FBP filter), kept as a reference
playbook.  The short, always-loaded versions live in `claude_prompt.md` (jax/perf
specifics) and in global memory (general principles); this is the detailed
version, with the worked example.

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
- **Preallocated `peak_bytes_in_use` over-reports the true need — don't extrapolate it
  across sizes (ruler caution, Greg).**  The scaling harnesses set
  `XLA_PYTHON_CLIENT_PREALLOCATE=true` / `MEM_FRACTION=0.9`, so the BFC allocator is
  loose: `peak_bytes_in_use` is what it chose to hold in the big pool (fragmentation-
  and size-dependent), not the minimum working set.  The 1d→Nd *ratio* at one size is
  still meaningful (structural band-streaming), but absolute MB and cross-size ×k
  extrapolation are NOT — e.g. "512³ 1d ≈ 40 GB ⇒ 1008³ won't fit one GPU" is wrong
  (1K³ has been run on a single GPU even with the older, heavier code).  For a true
  capacity claim, measure with `PREALLOCATE=false` (peak tracks actual need), or the
  OOM threshold (`sparse_back_project_single_device_sweep.py` style), or the per-buffer
  attribution (`sparse_back_project_memory_attribution.py`).

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
