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
- **Don't post-multiply a full array by a float64 scalar.**  `np.pi` is float64;
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
