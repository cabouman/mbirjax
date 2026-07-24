# Projector batching refactor — design (v2 drivers, parallel to v1)

> **DECISION RECORD (2026-07-04, supersedes the plan below): v2 retired; a surgical
> windowed-read patch to v1 tried in its place.**  Two consistent full-path runs showed v2
> ~5% slower end-to-end with identical peaks, and a ragged-shape run tripped the fp gate
> (2.27e-04 vs 1e-4) because balanced sizing REGROUPS every sum.  Balanced sizing itself
> solved a non-problem: v1's ragged tails were already zero-waste (the census's dramatic
> numbers priced pad/overlap strategies v1 never used), and the flat vmap-width knee means
> batch-size hygiene doesn't matter.  State after the pivot: `projectors.py` restarted from
> `greg/shard_profiling` with ONE change — `sum_function_in_batches` reads its full batches
> as `dynamic_slice` windows (same sizes, order, partials; no input reshape).  The old form
> is frozen in `reference_batching.py` (plans/experiments/projector_batching/) for A/B measurement; v2's history
> lives in git (`df91964`, `a550538`) and the sections below.
>
> **CORRECTION (2026-07-04, old-vs-new H100 driver A/B): the attribution behind the patch
> was WRONG.**  The windowed read alone is exactly time-neutral on GPU (new/old = 1.000 on
> all 16 cells; back/band bit-identical, rel_max 0.0) — v2's −10% band win does NOT
> reproduce, so it did not come from removing the input copy.  Best remaining explanation:
> v2's balanced PIXEL width (1977 vs 2048) sped up the access-pattern-bound band kernel —
> recorded as an open lead (a cheap pixel-batch sweep on the band driver, if band time ever
> matters).  Whether the sino-sized temp reservation is actually removed by the patch is
> being verified by a one-process GPU run of `transient_memory_probe.py`; if it is NOT
> (i.e., XLA lowers both forms identically on GPU), the patch buys nothing measurable and
> the right end state is pure shard_profiling v1 — the machinery is measured-optimal as it
> stands, and the durable outputs of this episode are the measurements and this record.
>
> **RESOLUTION (2026-07-04).**  (a) The GPU transient probe showed old and new
> BYTE-IDENTICAL in compiled temps at every vb — XLA lowers both forms to the same GPU
> program, the "copy" is just how XLA represents the scan either way, and the windowed-read
> patch was REVERTED: `projectors.py`'s batching machinery is pure shard_profiling v1,
> measured-optimal.  (b) The pixel-width lead CONFIRMED and shipped
> (`pixel_width_sweep.py`, H100 + CPU): the BAND kernel obeys two empirical width rules —
> stay 32-ALIGNED (unaligned costs up to ~25% at short band lengths on H100) and avoid
> row strides that are large POWERS OF TWO (exactly 2048: ~10% penalty at H100/L=256 and
> **1.4–1.5×** on CPU at L=26–64) — while forward and monolithic back are width-insensitive.
> Width **2016 = 32·63** satisfies both rules and was never worse than 2048 in any measured
> cell (H100: 0.892 @ L=256, 0.999 @ L=103; CPU: 0.723 @ L=64, 0.665 @ L=26).  Shipped as
> `TomographyModel.pixel_batch_size_for_vmap_back = 2016` (back/band only — their pixel
> axis concatenates, so no sums regroup and the change is fp-trivial; forward keeps 2048).
> This also re-attributes v2's band win (its widths happened to be near-optimal at both
> cells), which makes v2's ~5% full-path slowdown MORE mysterious — that remains
> unexplained (aggregate compile differences are the unfalsified suspect).  Related note
> for the future (Greg): forward projection is ~2× back's time on GPU and 3–4× on CPU — a
> forward-kernel-internals project would shift these width/batching trade-offs again.

**Drafted 2026-07-03**, from `projector_batching_characterization.md` (structure + census +
memory probe).  Scope: the **driver tier** in `mbirjax/projectors.py` only.  The host-tier
sites in `tomography_model.py` (§1 of the characterization) are explicitly out of scope for
this phase; the two ad-hoc ones are follow-on candidates if the driver results transfer.

**Ground rules** (Greg, 2026-07-03): build **parallel v2 versions alongside v1**, both
callable, compared apples-to-apples; code must be steppable.  Device-count-dependent batch
sizes are fine; the gate is fp equivalence (`assert_sharded_allclose`), never byte identity.

## 1. What changes and why

v1 mechanics (both helpers): odd remainder batch inlined FIRST, then `lax.map`/`lax.scan`
over full fixed-size batches.  Census verdict: zero FLOP waste but an extra inlined kernel on
43/86 realistic axes, and the fixed batch size is what forces the "divisor" gymnastics on the
adaptive-knob design.

v2 mechanic — **one uniform pattern for every batched axis**:

1. **Balanced batch size** (host, static): for axis length `n` and memory-derived cap
   `B_max`:  `num_b = ceil(n / B_max)`;  `B* = ceil(n / num_b)`;  `r = num_b * B* - n`
   (residual, `0 <= r < num_b`).  With the leading-partial mechanics below, wasted FLOPs are
   exactly zero and the residual only determines whether a second kernel shape exists.
2. **Windowed `lax.scan`** with `lax.dynamic_slice_in_dim` reads — **on the SUM axes only**
   *(narrowed 2026-07-04 during implementation; measurements below)*.  No input reshape,
   hence no input copy (the reshape is what materializes a full view-shard temp in v1 back
   on GPU; `apply_row_filter` precedent, lessons §3).  The **concatenate axes keep v1's
   reshape + `lax.map` + concatenate mechanics** and gain only balanced sizing: their
   batched inputs are small index/parameter arrays by construction (view params, pixel
   indices — the big arrays are closed over, not batched, on those axes), so the reshape
   copy is negligible there, and the windowed form measured **~1.6× slower on CPU** for the
   forward view axis (cone N=128, identical width 128: scan-writing-into-carry 489 ms vs
   `lax.map` 296 ms) — the map's stacked output fuses with the kernel where the carry
   `dynamic_update_slice` does not.
3. **Residual handling: v1's leading-partial mechanics with balanced sizes** *(REVISED
   2026-07-04 from the original "overlap + output-side mask" — see note below)*.  When
   `r > 0`, the first `b0 = B* − r` items run as one inlined partial call (as in v1, but
   near-full-size instead of arbitrarily small) and the map/scan covers the remaining
   `num_b − 1` full batches.  When `r == 0`:
   - **sum axes**: the scan carry initializes to zeros (via `jax.eval_shape`, which traces
     without compiling) and the scan covers ALL windows — one kernel instance in the HLO,
     fewer than v1.  Measured neutral-to-faster on CPU (back at width 128: 86.7 ms v2 vs
     97.8 v1).
   - **concatenate axes**: the first FULL batch still runs inline (exactly v1) — an
     all-batches `lax.map` with no inline peel measured **~1.5× slower on CPU** for the
     forward view axis (447 vs 296 ms at identical widths; `lax.map`'s loop body does not
     get the optimization the inlined instance gets).  The peel is load-bearing there.

   No overlap, no recompute, no masks, no linearity contract anywhere.

   *Why revised:* output-side masking is IMPOSSIBLE at the helper level on sum axes — the
   per-item stack is reduced inside the caller's function, so the helper only sees the
   already-summed contribution.  The alternatives (a weights argument threaded into every
   per-window function, or input-zeroing with a linearity contract and a NaN hazard on
   zeroed view params) buy one fewer inlined kernel shape — and Greg's H100 run measured
   driver compiles at 0.6–1.2 s, so that shape is nearly free.  Leading-partial keeps v1's
   skeleton (steppable, minimal diff) with zero numerical surface.
4. **Degenerate cases** (static Python branches): `n <= B_max` → single direct kernel call,
   no scan.  `batch_size=None` → `n` (v1 contract preserved).

This collapses the `concatenate_function_in_batches`(+`lax.map`) / `sum_function_in_batches`
pair into two thin wrappers over ONE windowed-scan core, removes the odd-batch inlined
kernel, removes `lax.map` entirely, and makes any `B_max` shape-safe — which is what the
size-adaptive knob (current_plans.md (then post_shard_plans) item 1) needs.

## 2. New code, parallel to old

In `projectors.py` (names open to bikeshedding):

- `balanced_batch(n, batch_size) -> (b_star, num_batches, residual)` — pure host helper,
  unit-tested directly.
- `map_in_balanced_batches(function, data_to_batch, batch_size)` — v2 of
  `concatenate_function_in_batches`, same signature and same output contract.
- `sum_in_balanced_batches(function_to_sum, data_to_batch, batch_size, extra_args=())` —
  v2 of `sum_function_in_batches`, same signature.
- `_sparse_forward_project_v2` / `_sparse_back_project_v2` / `_sparse_back_project_band_v2`
  — line-for-line structural mirrors of the v1 drivers with only the helper calls swapped,
  jitted at module level with the same static args.  Keeping the drivers near-diffs of v1 is
  deliberate: stepping through v1 and v2 side by side shows exactly one mechanism changing.

v1 stays untouched.  Nothing in the package calls the helpers except the drivers (verified),
so the blast radius of the new code is exactly these six functions.

## 3. Selection and comparison plumbing

- `Projectors.create_projectors` builds BOTH public entry sets and exposes them as
  `sparse_forward_project_v1/_v2` (etc.); the existing names (`sparse_forward_project`, ...)
  bind to the version selected by a plain model attribute
  `TomographyModel.projector_batching_version` (default **1** until validated; an attribute
  like `view_batch_size_for_vmap`, not a persisted param — it must never change results
  beyond fp noise, so it does not belong in the saved model).
- Because both versions hang off ONE model instance, comparisons share the identical
  `view_params_array`, placements, and inputs — apples-to-apples by construction.

## 4. Gates

**Equivalence** (`tests/sharding/conftest.assert_sharded_allclose`, rel-max):

- v1-vs-v2 on both drivers + band, cone AND parallel beam, `coeff_power` ∈ {1, 2}, at
  1e-5 (projector single-shot calibration).  No exact gates anywhere: v1 and v2 are
  different executables (~1 ULP apart even for identical programs), and the sum-axis
  reduction order legitimately changes.
- Axis-shape matrix chosen to hit every static branch:  `n < B`, `n == B`, `n mod B` ∈
  {0, 1, B−1}, `num_b` ∈ {1, 2, many}, `r` ∈ {0, 1, num_b−1} — on both axes.  Build test
  arrays from the same model under test; exercise a device count that does NOT divide the
  axis (lessons §3).
- Integration: the existing projector/VCD suites run with version 2 forced (one env/fixture
  toggle), unchanged tolerances.

**Performance / memory:**

- Re-run `transient_memory_probe.py` against the v2 drivers (same sweep): expect the slope
  (k) at or below v1 and no new constant term — the §4 caveat that buffer reuse is a
  property of the current lowering cuts both ways, so this is the regression check.
- ~~`gpu_knee_probe.py` results decide whether B\* needs a floor~~ — **RESOLVED 2026-07-04**
  (H100 run, characterization §5): no knee down to B/2 (vb=64 costs ≤1.04×, typically ≤2%),
  so `balanced_batch` ships with NO floor logic.  B\* ≥ B_max/2 by construction also clears
  the CPU vb=16 cliff.  CPU timing sanity locally remains a gate (cheap).
- ~~v2 must remove the back driver's sino-sized GPU temp~~ — **CONFIRMED 2026-07-04**
  (Greg's H100 v1-vs-v2 run): back v2 temp const = **0.00× sino** at N ∈ {256, 512} (v1:
  1.00×) — at N=512/vb=128 that is 1077 MB vs 4969 MB of compiled temp reservation (memory_analysis, not a runtime peak) — AND back v2 is ~3%
  faster across all vb.  Forward v1 ≡ v2 on GPU to 0.1 MB / timing
  noise, confirming the hybrid's concat-axis mechanics are untouched.  Slope k: back 2.01
  v2 vs 1.76 v1 (minor, dwarfed by the const win); forward identical (1.04/1.27).
  *[CORRECTION 2026-07-04: "the copy's bandwidth" attribution was falsified by the
  old-vs-new A/B — the windowed read alone is time-neutral; see the decision record at the
  top.  Which v2 mechanism produced the 0.00× const is part of the same open verification.]*

## 5. Risks / open points

- **Scan-of-dynamic_slice vs reshape+map lowering:** ~~may fuse differently~~ — CONFIRMED
  on CPU during implementation, twice: the windowed form is ~1.6× slower on the concatenate
  axes and an unpeeled all-batches `lax.map` ~1.5× slower (see §1.2–1.3; both fixed by
  keeping v1 mechanics there).  On the SUM axes the windowed scan measured neutral-to-faster
  on CPU with lower temp; GPU timing/memory of the final hybrid is what the updated
  `gpu_knee_probe.py` (now v1-vs-v2) checks on the cluster.
- **Partial-batch kernel shape:** when `r > 0` the HLO carries a second (near-full-size)
  inlined kernel, like v1's odd batch.  Measured compile cost 0.6–1.2 s/executable (H100) —
  accepted in exchange for zero recompute and no numerical surface.
- **Band driver `g0`:** stays traced; windows touch only the view/pixel axes.  Unchanged.
- **2^31:** window starts and axis lengths are per-shard/per-subset counts, orders of
  magnitude below 2^31; `dynamic_slice` indices are per-axis, not flat.  No exposure.
- **Rollout:** ~~land v2 + harness + tests with default 1~~ **LANDED 2026-07-04** (v2
  helpers + drivers in projectors.py, call-time version dispatch, `projector_batching_
  version` attribute + `MBIRJAX_PROJECTOR_BATCHING_VERSION` env override,
  `tests/sharding/test_batching_v2.py` — 56 equivalence tests over every batching branch ×
  both geometries × coeff_power; full suite green under v1 default AND with v2 forced at 4
  CPU devices; CPU memory probe: v2 slope 5.86 vs 7.21 forward, 1.12 vs 1.48 back, back
  const → ~0).  **Cluster validation COMPLETE and DEFAULT FLIPPED to 2, 2026-07-04.**  The three
  cluster gates (all run by Greg, H100):
  - `gpu_knee_probe.py` v1-vs-v2: back v2 loses the 1.00×-sino temp (const → 0.00×), ~3%
    faster; forward identical to 0.1 MB.
  - `full_path_ab.py` (cone 1024³, 1024 views, 5 iters, 2 GPUs): rel-max **4.4e-05**
    (gate 1e-4, PASS); peak/device identical 21.0 GiB both versions (the whole-path peak
    is set above the back transient at this config, so the driver win is latent there).
    Wall time showed +3.3% for v2 on a single sample — reclassified as noise by the
    driver-level probe below.
  - `driver_ab_probe.py` (N=1024, 51.4k pixels — the map helper's first multi-batch GPU
    timing — views 512/471, all three drivers): forward 1.001/0.996; back 1.001/**1.010**
    (the ragged +1% is the KNOWN vb-width curve at balanced width 118 vs 128, bounded ≤~4%
    worst-case by the vb sweep, not a mechanics regression); band **0.901/0.971** — the
    multi-GPU back driver is up to 10% FASTER, and the 44 ms gap matches the bandwidth of
    the eliminated 2 GiB sino-shard reshape copy.
  Remaining: remove v1 (drivers, helpers, dispatch, env var) after a deprecation interval —
  Greg's call on timing.

## 6. Follow-ons unlocked (not in this phase)

- **Adaptive `view_batch_size_for_vmap`** (item 1): `B_max = budget / (k · pixel_batch ·
  det_rows · 4)` with the GPU k from the knee probe; plugs into `balanced_batch` untouched.
- Host-tier cleanups (characterization §1): the cone gather-forward tail executable and the
  n=1-GPU back transfer loop (`transfer_pixel_batch_size`'s last consumer).
- The `[0,2,4,6,7]` partition-sequence knob rides the same cluster session as validation.
