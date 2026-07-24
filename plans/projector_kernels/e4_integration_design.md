# E4: integrating the Pallas projector kernels — design for review

**Drafted 2026-07-12 (discussion-first; no library code until approved).**  Scope per
the E4 agreements: the parallel-beam forward + back pair ships TOGETHER (+ the Hessian
path); cone forward hfan rides the same integration (fine-tail policy); cone back
(fused vfan) and the multi-device band tables are wave 2.  Measured basis:
`gpu_headroom_findings.md` E3/E3b sections; composed-back preview in flight
(`e4_back_composed.py`).

## 1. What ships

| kernel | measured (isolated) | policy |
|---|---|---|
| fwd hfan, parallel | 2.13× (subset, two-phase) / 1.59× (raster, hybrid) | pallas on the VCD fine tail; XLA for one-shot coarse/full-grid calls |
| fwd hfan, cone | 2.13× (subset, row-chunked) | same fine-tail policy |
| back, parallel | 16–26× (+Hessian 15–23×) | pallas everywhere on supported archs |
| adjointness | ⟨Ax,y⟩=⟨x,By⟩ exact-to-f32 | pair-level gate in the suite |

## 2. The precompute: an explicit `ProjectorPlan` (the placement detail)

The kernels consume per-(pixel-set, view) structures: BACK needs centers (V, P) i32
(already exists) + weights (V, T, P) f32 — a pure formula, NO sort; FWD additionally
needs the channel-sorted streams (weights, pixel-ids) + segment tables (a host-side
cap-and-split with data-dependent shapes).  Rather than hidden caching inside the
wrappers, the design makes the lifetime EXPLICIT:

- **`plan = projectors.plan_projections(pixel_indices, direction, coeff_power)`** — an
  eager builder (the `n_p_centers` idiom: separate jits, concrete outputs, view-chunked
  by the existing 256 MB rule; the fwd cap-and-split runs host-side on the sorted
  counts).  The plan object holds device (or host, see §4) arrays + the segment stats
  that drive variant/warp choice.
- **Wrappers take `plan=None`**: `sparse_back_project(sino, idx, plan=None, ...)` —
  when None and the policy selects pallas, a transient plan is built in-call (charged
  to that call); when the policy selects XLA, no plan is built.  No behavioral change
  for existing callers.
- **`vcd_recon` builds plans once per recon**: partitions are fixed, so after
  `gen_set_of_pixel_partitions` it builds per-subset plans lazily on first use and
  reuses them across all iterations (the amortization that makes the fine tail ~free).
  The plan cache lives in the recon call's scope — created and released with the recon,
  no global state, donation-safe.
- The no-eager-array-ops wrapper contract is preserved: plan building is one eager call
  per (subset × direction) per recon, not per projector call.
- **Weights storage (Greg 2026-07-12).**  Store-central-weight-and-derive-the-taps was
  considered and declined: recovering the ±1 taps needs the fractional coordinate
  anyway, so (n_p, W) is the minimal sufficient pair at the same footprint, reuses the
  verified formula verbatim, and avoids a rederivation that could disagree with the
  centers at rounding ties.  Better: for PARALLEL beam, W and the weight scale are
  per-VIEW constants — the per-pixel plan payload is n_p alone (4 B beyond the centers);
  cone keeps per-pixel (n_p, W, scale) (12 B).  On run_per_device: that parallelizes
  across DEVICES (the sharded path already builds per-device plans in its per-device
  drivers); within one device the builder is a single jitted call whose cost was
  dominated by a bench artifact — see §8 — and production naturally OVERLAPS chunk c+1's
  build with chunk c's kernel (the timed breakdown serialized them for attribution).

## 3. Policy: TilePolicy flags + measured guards

Two new TilePolicy fields: `fwd_pallas`, `back_pallas` (int flags like
`sort_by_channel`), set by `_select_tile_policy` under ALL of:
  - GPU platform AND a pallas-triton capability probe (one cached tiny-kernel compile
    at layout time; any failure → 0, XLA path) AND an arch allowlist seeded with
    H100 (extend per measured arch — the platform-conditional precedent).
  - FWD only: pixel-count / segment-stat guard — pallas for repeated-subset (fine-tail)
    calls; XLA for one-shot coarse + full-grid (also where pallas is weakest:
    raster-skew).  BACK: no workload guard (16–26× everywhere measured), but the same
    arch/capability gates.
- Within pallas: variant (two-phase vs hybrid) and num_warps chosen from the plan's
  segment stats (fwd) and rc table (back: rc=256 default) — constants recorded like the
  sorted-reduce guards, reproducible by the bench scripts.
- The XLA kernels remain compiled-in fallbacks at every site; a single env override
  (e.g. `MBIRJAX_DISABLE_PALLAS=1`) forces them globally (the escape hatch for a bad
  driver/toolchain day).

## 4. Driver changes

- **BACK: a new simplified path** — the kernel has no scan carry, so the per-device
  driver becomes: view-chunk loop (128/chunk keeps the L2-phase slice ~130 MB) ×
  ONE pallas call over ALL pixels, accumulating across chunks (jnp add or aliased
  chaining).  This deletes the 94-step pixel scan and the transfer-chunk concat on the
  pallas path — the composed preview measures exactly this.  The banded multi-device
  entry keeps its interface; its pallas path uses row-index tables (bands = contiguous
  tables), wave 2.
- **FWD: keep the existing scan structure, swap the kernel** — per pixel-batch the
  hybrid/two-phase call replaces `horizontal_fan_project`'s sorted reduce; the (V,B,C)
  accumulation stays as-is.  (A scan-free fwd is possible later via all-atomic
  accumulation; not needed for the fine-tail policy where batches are single.)
- Memory transients, in PER-DEVICE SINO-SHARD multiples (shard = (V/n)·R·C·4 B; all
  plans scale ∝ V_chunk·P_pixels, so the ratios below are size-independent at fixed
  detector aspect C≈R and fixed V_chunk/V; they shrink linearly with the pixel
  fraction):
  - **back plan (in-kernel weights)**: bytes = V_chunk·P·(4 + payload); payload 4 B
    parallel (n_p only) / 12 B cone.  Ratio ≈ (1+payload/4)·0.79·(C/R)·(V_chunk/V).
    At the 1024³ cell, V_chunk=V/8: full grid **0.10 shard parallel / 0.19 cone**
    (0.39/0.77 GB); a granularity-128 subset **0.6% / 1.3% shard** (26/52 MB, single
    call all views).  Transient, freed per chunk.
  - **fwd streams** (fine-tail subsets ONLY by policy — the full-grid 0.58-shard
    variant never ships): per subset ≈ **3.8% shard** (156 MB at the cell); built
    lazily per subset, freed or host-cached (device cache of all 128 subsets = 4.6
    shards — excluded); host-stream cost ~6 ms/subset if cached.
  - **padding**: +1.6% typical on the padded values/output axes (pow-2 row/band);
    pathological counts avoided by pow-2-friendly band choices (parallel — the band is
    our knob) and row-chunk tables (cone); confirmed in the A/B memory gates.

## 5. Gates (all must pass before the flag defaults on anywhere)

1. Kernel-equality tests: pallas vs XLA per geometry × direction × coeff_power ×
   variant, rel ≤ 1e-5 single-shot (the float-gate calibration), on CPU-interpret (CI)
   and GPU (nightly).
2. **Pair adjoint test**: ⟨Ax, y⟩ = ⟨x, Bᵀy⟩ with BOTH pallas kernels (the E3 bench
   version used the XLA forward; the suite version uses the shipped pair).
3. Poison-the-padding (inert-padding invariant) on the pallas paths; chunked-vs-single
   plan equality; plan-refuses-tracers (the concreteness contract, as with centers).
4. Model-level A/B on H100: fwd/back/Hessian cells + VCD guard cells (the
   CUDA-graph/dispatch caveat watches VCD at interactive sizes), memory gates with the
   §4 acks pre-declared, at 512³- and 1024³-class shapes; then the nightly across the
   tracked branches.
5. Rollout: opt-in flag → nightly soak → default-on for the arch allowlist, with
   re-baseline + release note (the campaign's established path).

## 6. Risks and their handling

- **Toolchain churn** (Triton backend is best-effort upstream; documented cross-version
  cliffs): the jax pin already exists; add the capability probe (§3) + the bench
  scripts as re-validation on any bump (the lessons §5 discipline).
- **Arch fragmentation**: allowlist + fallback; L40S/A100 cells added when measured.
- **VCD dispatch** (pallas skips CUDA graphs by default): the fine-tail policy applies
  pallas where calls are device-bound (E1: ~92–96%); interactive sizes keep XLA via the
  existing size-aware policy knobs if the guard cells flag it.
- **Maintenance and fragmentation (Greg 2026-07-12: a first-class deliverable, not an
  afterthought)**: kernels live in ONE module (`_pallas_kernels.py`); every constant
  traceable to a findings entry; and the code PR MUST include (gate 6) a readable
  dev-docs section ("Pallas projector kernels", alongside the existing "Projector
  kernel design" in `docs/source/dev_sharding_overview.rst`) covering: the
  plan/kernel/policy structure and why each exists; how the two kernel shapes work in
  words (the forward segment walk; the back register-tile with L2 row-chunk phases);
  where the fallbacks live and how selection is decided; and THE UPDATE PROTOCOL — what
  to re-run on a jax bump or new arch (capability probe, the bench scripts as the
  reproducible record, the gate suite), and how to retire the pallas path wholesale
  (one env var / one policy line) if maintenance ever outweighs the win.

## 8. Which granularities use the kernels (the break-even question, Greg 2026-07-12)

- **BACK: all granularities, including one-shot coarse.**  Its plan is ~free — the
  centers are built for the XLA path anyway, and the payload (n_p, +W/scale on cone)
  comes from the SAME jit at marginal cost.  The measured one-shot composed gain (3.5×,
  projected 8–9× post-fix) applies at every level.
- **FWD: break-even per (subset, view-chunk) ≈ build_cost / (t_xla − t_pallas).**  Per
  8k-pixel chunk the kernel saves ~0.3 ms; the bench builder cost (~145 ms) makes that
  ~500 reuses — RESOLVED (job 13486103): the builder cost was a bench artifact (fresh
  jax.jit per call → host retrace; hoisted, the back weights builder measures **~1 ms/
  chunk**, and the composed back went 3.54× → **9.07×** with the kernel at 96% of the
  time).  The fwd stream builder (sort included) gets the same hoisted-jit measurement
  in E4 — expected ~2–5 ms/chunk → break-even ≈ **6–15 reuses**: the fine tail pays
  clearly; one-shot coarse fwd stays on XLA (also its weak case, skew) unless the
  measured builder lands ≤1 ms, in which case the policy can widen.  The in-kernel-
  weights variant (§2) is retained as a MEMORY option (0.10- vs 0.19-shard plans), no
  longer performance-motivated.

## 7. Open items folded into the E4 work

- **Composed-back preview (measured, job 13484992): 3.54× one-shot (10.56 → 2.98 s),
  Hessian 3.53×, values pass.**  The simplified driver composes (scan/chunk overheads
  ~zero); the limiter is the (V, T, P) weights build (1.83 s of 2.98).  DESIGN CHANGE
  ADOPTED: the back kernel computes weights IN-KERNEL from per-pixel (n_p, W) + the
  per-view scale — the plan shrinks to 2 f32 per (view, pixel) (emitted by the same jit
  that already computes centers), projected composed ≈ 8–9× one-shot; VCD fine-tail
  calls amortize the plan regardless and see near-kernel-level gains.
- The H100 precompute slowness (145 ms vs 63 ms on M3) — profile once during
  integration; suspected sort-path detail, amortized away in VCD regardless.
- Wave 2: cone back (fused hfan+vfan register-tile — required for cone back gains, per
  the composition no-op finding), the multi-device band tables ((c) target: the n=2
  anti-scaling), and slice-index sets (parity hook — currently a quality option, so the
  tables ship with bands first).
