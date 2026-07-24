# GPU projector-kernel headroom: investigation plan

**Drafted 2026-07-12; updated same day with Greg's prioritization — (c) multi-GPU cone
time scaling first, (a) single-device VCD wall second (§3)** (drafted on
`greg/kernel_investigation`; work to proceed on a dedicated branch/worktree, see
Logistics).  This is the plan of record for `current_plans.md` §3's
second headroom pool: the GPU kernels sitting well above compute-only bounds, attributed to
memory access patterns.  It follows a five-report research pass (2026-07-12) whose full
reports are in `headroom_appendices/` — cited below as [Roofline], [Amdahl], [Pallas],
[CT-practice], [XLA-lowering].  Numbers below are from those reports; the traffic-model
numbers are PRE-E0 estimates (arithmetic + HLO reading, not yet profiled).

## 1. The recalibrated prize

Three independent brackets replace the raw "~10×" framing:

- **Traffic model** [Roofline]: at the 1024³ cell (V=1024, R=1008, C=992, ~771k ROR
  pixels; fwd = 4 bands × 8 view chunks × 95-step pixel scans; measured fwd 8.19 s /
  back 10.92 s on H100), the perfect-cache floor is ~0.25 s fwd / ~0.3–0.5 s back
  (25–35×), FLOP floor ~0.08–0.09 s (~100×).
- **Published SOTA, bandwidth-normalized** [CT-practice]: hand-CUDA codes (LEAP, tuned
  ASTRA/KernelKit, branchless DD) cluster at ~80–120 G voxel·view-updates/s per TB/s;
  mbirjax sits at ~40 (parallel fwd) / ~30 (back) / ~17 (cone fwd).  Nobody reaches the
  cache-perfect bound — everyone is access-pattern-limited (independently confirming the
  June ncu finding: 97% L1, 8% HBM).  A hand-written H100 kernel plausibly lands at
  ~2.7–3.7 s fwd / ~3.5–4 s back.
- **XLA-only endpoint** [Roofline][XLA-lowering]: sort hoisting + fusion repair bounds
  at ~2.5–5 s fwd / ~1–3 s back — roughly where a first custom kernel would land.

**Planning numbers: 2–3× kernel-level from a custom kernel; 1.5–3× from XLA-level work at
a fraction of the cost.**  Upside beyond that exists (KernelKit's register-tiled small-view
regime hits 4–5× the SOTA cluster) but is not the planning basis.

## 2. Where the time goes (to be verified by E0)

- **XLA's sorted-scatter emitter is already near-optimal** [XLA-lowering, verified against
  openxla source + local HLO at the 0.10.1 pin]: `segment_sum(indices_are_sorted=True)`
  selects `ScatterWithDistributedIndices` — per-warp register accumulation of equal-index
  runs, ~one atomic per run of ~24 (its dispatch heuristic literally recomputes our
  collision ratio).  <2× headroom inside the scatter fusion.  Phase A landed on a good path.
- **Forward cost is AROUND the scatter**: (i) the runtime CUB radix sort, re-run every
  step (~2–4.6 s of 8.19 s — extrapolated, E0 measures it); (ii) the sort is an unfusable
  kernel boundary, so the sorted (V, T·P, B) updates stream (~3.2 GB/step) round-trips
  HBM (~5.8 s at peak BW).  These two terms fully explain the measured time.
- **Back cost is materialized gather intermediates**: the (V, T·P, R) gathered/weighted
  streams + the (V, P, R) per-view stack before the view sum ≈ 15 of 18 GB/step.  Fully
  fused bounds at ~2.9 s, plausibly ~1 s with cache effects.
- **Cross-view data reuse is impossible in XLA** (confirmed: vmap lanes share nothing but
  hardware L2; no inter-lane fusion exists).  Keeping a voxel tile (fwd) or pixel-cylinder
  accumulator (back) on-chip across a view tile is expressible only in Pallas/CUDA — the
  structural remainder of the gap, and the mechanism behind ASTRA's canonical back kernel.
- Toolchain: nothing relevant landed upstream after the 0.10.1 pin (verified through
  2026-07-11); 0.10.2's cone regression is unfixed.  Stay pinned [XLA-lowering].

## 3. Where kernel wins pay (Amdahl) — and the gating measurement

| workload | kernel fraction f | E2E gain @ 5× kernels |
|---|---|---|
| full-grid one-shot ops (see below), 1024³ | ≈1.0 | ~3–5× |
| VCD 1024³ coarse granularity | ~0.78–0.91 (derived) | ~2.7× |
| VCD 512³-class | ~0.85 (derived) | ~3.1× |
| **VCD at production granularity 128, 512³/1024³** | **UNMEASURED — gates the investment** | ? |
| VCD 200³ (interactive) | ~0.05 | ~1.04× (nothing) |

"Full-grid one-shot ops" = the public `sparse_forward_project`/`sparse_back_project` on
the full pixel grid plus their internal uses: FBP/FDK `direct_recon` (1 full back), the
recon-init error sinogram (1 full fwd), `compute_hessian_diagonal` (1 full back at
coeff_power=2), MAR's (1+num_metal) forward projections.  There f≈1 and kernel wins
transfer ~fully — but they occur a few times per recon, so they dominate only for
FDK-init/MAR pipelines and direct simulation use.

At granularity 128 a subset is ~6.3k pixels — a single pixel-batch call, where per-call
fixed costs loom largest; 200³ evidence says the kernel share collapses in that regime,
but no one has measured it at 512³/1024³.  **E1 produces this number.**

Distinct beneficiaries [Amdahl]:
- **Cone is the real-data geometry** and the hardest case: forward vertical fan untouched
  by the campaign (hfan/vfan split at 1024³ unmeasured); cone back needs a whole-kernel
  restructure (per-fan substitution is a proven composition no-op).
- **Multi-GPU cone TIME scaling**: the band-kernel transpose (L1-bound) makes cone back
  anti-scale (n=2 = 0.87× of n=1), capping cone VCD at 1.26× on 2 GPUs.  Memory sharding
  already works; a custom band kernel converts added GPUs into wall-clock reduction.

**Prioritization (Greg, 2026-07-12): (c) multi-GPU cone TIME scaling is the MAIN target;
(a) large single-device VCD wall time second.**  Consequences threaded through the ladder
below: the Pallas pilot (E3) is the multi-device BAND back kernel (transpose-free,
register-tile across views), not the parallel-back kernel; E2 gains the XLA-level
transpose-avoidance restructure (the June "B4.5 lever", approach A5) as the cheap attempt
first; E0's ncu explicitly covers the band kernel at n≥2.  The parallel-back
register-tile kernel remains the second target (serves (a)).

## 4. Approach inventory

### A. XLA-level (no new dependencies; compose with everything)

- **A1 — hoist the sort out of the compiled programs.**  The permutation is a pure
  function of geometry (the already-concrete `n_p_centers`) — precompute `order` in a
  separate eager jit (the Phase D idiom, extended) and pass it concrete; in-kernel work
  collapses to gather + multiply + sorted scatter.  Expected 1.2–2× on forward; reused
  across the 4 slice-bands per call and across VCD iterations (partitions fixed per
  recon).  **Bitwise gate available**: both today's in-jit sort and the eager sort are
  stable sorts of the same concrete int32 keys → same permutation → identical values
  (fall back to the 1e-5 rel gate only if fusion context shifts).  Memory: see §5.
  Forward-only (back does not sort).
- **A2 — flatten the vmapped scatter to 1-D segment ids** (`view·C + channel`): the
  current 2-vector (view, channel) index forfeits the emitter's vectorized index reads
  (`index_vector_length == 1` precondition, verified in scatter.cc).  1.05–1.3×, trivial A/B.
- **A3 — fusion repair** (contingent on E0): if the dump shows the 3 GB intermediates
  materializing where they need not, formulation changes may be worth 1.6–3× fwd / up to
  3.8× back.
- **A4 — channel-coherent pixel ordering** (gather-side L1/L2 hit rate; hardware effect,
  not a compiler path).  Uncertain 1.0–1.5×; interacts with VCD partition semantics;
  cheap ncu A/B before any real work.
- **A5 — band-kernel transpose avoidance (the June "B4.5 lever"; serves the MAIN (c)
  target at XLA level).**  The multi-GPU limiter is the band kernel's internal transpose
  (`input_transpose_fusion`, L1/TEX-bound: 99.9% L1, 6% HBM — June ncu; the band kernel
  is unchanged by the campaign, so this attribution is current).  Restructure
  `back_project_one_view_to_band` to write pixel-like via `dynamic_update_slice` WITHOUT
  reintroducing the CPU cache cliff (GPU-gated, the established platform-conditional
  pattern).  Attempt before the Pallas band kernel; E0's n≥2 ncu re-check first.

### B. Pallas custom kernels (the structural fix)

- Pure-Python preserved (Pallas compiles via jaxlib's bundled XLA — no packaging change;
  the only custom-kernel vehicle meeting that hard constraint).  `custom_vjp` over the
  existing fwd/adjoint pair is the endorsed autodiff pattern (verified in interpret mode);
  a Pallas call inside the per-device thread-pool jits is a plain single-device XLA op —
  no GSPMD interaction.  Interpret mode = CPU-CI value correctness (race-blind; GPU float
  gates still required) [Pallas].
- **Backend dilemma at the 0.10.1 pin**: Mosaic GPU (default; SMEM scratch, TMA, warp
  specialization — everything these kernels want) is Hopper/Blackwell-only; the Triton
  backend covers all archs but has NO SMEM scratch (verified in installed source), is
  best-effort maintained, with documented 5–10× cross-version perf cliffs on exactly our
  odd-shape/computed-index patterns.  Ampere support for MGPU lands in 0.10.2 — excluded
  for the unrelated cone regression.  Arch policy: DEFERRED until after the spike (Greg,
  2026-07-12) — decide with data in hand.
- **Design patterns are precedented, just not in a DSL** [CT-practice]: forward-side =
  PyFAI's published transform (same scatter problem → precomputed CSR segments →
  workgroup segmented reduction, "random read / linear write"); back-side = ASTRA's
  canonical register-tile-across-views kernel (6 voxels × 32 angles per thread).  No
  published X-ray CT projector exists in Triton/Pallas — publishable novelty, with the
  corresponding risk premium.
- **VCD hazard**: Pallas calls are not captured in CUDA command buffers by default — a
  Pallas kernel could WORSEN dispatch-bound cases.  The TilePolicy guard must gate it to
  large sizes from day one.
- Side benefit: fused kernels eliminate the multi-GB materialized transients — a capacity
  win at exactly the sizes that OOM.

### C. Hand CUDA via jax.ffi — fallback only

Breaks the pure-Python wheel constraint (per-platform binaries, doubled release surface).
Only defensible as an optional plugin if Pallas demonstrably hits a wall raw CUDA would
clear — and the texture path's 9-bit lerp weights fail our float gates anyway, so the
marginal advantage here is small [CT-practice][Pallas].

### D. Algorithmic reorganizations — constrained by matchedness

Unmatched-pair convergence fixes exist only for gradient/Krylov outer loops; there is no
theory for unmatched operators inside coordinate descent (updates use exact columns A_j;
the Hessian uses A_j²).  Ray-driven forwards à la ASTRA are therefore inadmissible for the
VCD inner loop.  Admissible — and exactly what A1/B do — is reorganizing the computation
of the SAME trapezoid A (gather formulations of identical weights preserve matchedness by
construction; only summation order changes — a reproducibility-gate concern, not a
convergence concern; svMBIR is the in-family precedent) [CT-practice].

### E. Closed/rejected (for the record)

TF32/one-hot matmul (fp32 gates; FLOP blowup); `unique_indices=True` (no safe
application); toolchain bump (nothing relevant upstream; 0.10.2 regression unfixed);
hardware textures (9-bit lerp precision inadmissible).

## 5. A1 memory cost (quantified 2026-07-12, per Greg's request)

A1's concrete arrays are 3× the `n_p_centers` footprint (m=1: `order` only, sorted ids
derived in-kernel by a deterministic integer gather) or 6× (m=2: order + sorted ids passed
concrete), chunked by the same 256 MB-rule machinery, resident only during forward calls:

    delta_resident = m · T · P_tot · min(V/n, 128) · 4 B          (T=3 taps)
    delta / sino_shard ≈ 2.3 · m · min(1, 128·n/V)                (R ≈ C, ROR grid)

At the 1024³ cell (C=992, P_tot=771k; sino shard = (V/n)·R·C·4 B):

| devices n | V = C (α=1) | V = C/2 (α=0.5) |
|---|---|---|
| 1 | 0.29·m shard (1.18·m GB) | 0.58·m shard |
| 2 | 0.58·m shard | 1.16·m shard |
| 4 | 1.16·m shard | 2.31·m shard (cap) |
| 8 | 2.31·m shard (cap) | 2.31·m shard |

The absolute resident delta caps at ~1.2 GB (m=1) / ~2.4 GB (m=2) at C≈1000 for any n —
it is the RELATIVE cost per shrinking shard that grows with device count and falls with
view count.  Mitigations: prefer m=1; scale A1's chunk with the shard at high n, or
disable A1 there (narrow bands already disable the sorted reduce below 48 columns).  At
VCD subset sizes (~6.3k pixels) the arrays are ~10 MB — memory-negligible; the concern is
added host dispatch in the dispatch-bound regime (the +35% eager-op lesson) — candidate
policy: enable A1 only above a pixel-count threshold, decided in E2.  Caching all
(subset × view) permutations for a granularity-128 recon is ~9.5 GB — host-side only, if
ever.  Memory-gate impact lands in the same acknowledged-regression class as the Phase
B/D acks, ~3–6× their size at worst — call it out in the E2 A/B.

## 6. Experiment ladder (E0 ∥ E1 approved by Greg 2026-07-12; each rung gates the next)

**E0 — attribution repair (cluster, ~half a day, measurement only), run in parallel with E1:**
- `--xla_dump_to` HLO dump of the fwd/back programs at 1024³: does a ≥3 GB fusion output
  exist per step (the updates materialization)?  Is back's gather→multiply→tap-sum→
  view-sum one fusion?  Is the channel-major transpose hoisted out of the pixel map?  Do
  the scatter fusion launch dims match `ScatterWithDistributedIndices`?
- ncu on the CURRENT kernels (June data predates the campaign; update the
  `mbirjax_metrics/experiments/profiling` fusion-name regexes; use the
  `cudaProfilerStart/Stop` + `--profile-from-start off` bracket).  Roofline the sorted
  reduce and stacked gather: L1/LSU vs HBM.  Explicitly include the BAND back kernel at
  n≥2 (the (c) main target's limiter — confirm the transpose attribution on today's
  toolchain and name the saturated pipe, the June open item #5).
- Sort share of the fwd kernel (nsys kernel timeline: CUB sort kernels vs the rest).

**E1 — the Amdahl gate (cluster, ~1 day), in parallel with E0:**
- Device-trace a warm 2-iteration `vcd_recon` with `partition_sequence=[7]` at 512³ and
  1024³ (existing trace tooling) → device-vs-host share + kernel attribution.  THE
  deciding number.
- Cone forward hfan/vfan split at 1024³ (forward analog of `cone_back_kernel_ab.py` —
  open item #1 in the profiling findings).
- Pixel-count sweep of `sparse_forward/back_project` at 1024³
  {1.6k, 6.3k, 12.7k, 50k, full} → predicts fine-granularity VCD kernel share directly.
- (Nice-to-have: post-campaign VCD walls at 512³/1024³ via one harness run.)

**E2 — XLA-level A/Bs (bench harness, days):** A5 band transpose-avoidance A/B at n=2
(the MAIN-target cheap attempt, informed by E0's band ncu); A2 flatten (trivial — do
alongside); A1 prototype (eager sort + concrete order; bitwise gate; the §5 memory
check; a debug-mode monotone-ids assertion); A4 ncu L2-hit A/B.  A3 if E0 shows
repairable fusion.

**E3 — Pallas de-risking spike (scratch benches, NOT library code, ~1–2 weeks).**
Step zero (backend validation) CLOSED 2026-07-12 with a green light — findings doc:
Triton backend + ref-level gathers, bitwise-exact, 1.03× XLA untuned.  **First real
kernel (Greg, 2026-07-12): the HFAN-FORWARD kernel** — the CSR segment-walk gather
replacing the sorted-reduce fusion (86–88% of parallel fwd device time, 72% of cone
fwd; one kernel serves all four geometries' forwards; simplest value gates; its
load-balance machinery reuses everywhere).  Success bar: **≥1.5–2× kernel-level at
production shapes on H100 vs the XLA sorted reduce, at BOTH the raster full-grid batch
(real channel skew) and the VCD-subset batch (uniform); rel-max ≤1e-5; no
register-spill cliff.**  SECOND: the multi-device BAND back kernel (the (c) main
target — transpose-free, register-tile across views, composition bar for E4: cone back
n=2 must beat n=1), reusing the segment-walk machinery; the parallel-back register-tile
kernel after, as the (a)-track follow-on.

**E3b — the back-projection kernel design (drafted 2026-07-12, discussion-first;
parallel beam first, cone-with-vfan second).**

*The operation and its structure.*  Back's hfan is the forward's adjoint:
`out[p, :] = Σ_v Σ_t A[t,p,v] · sino[v, center[p,v]+t−r, :]` — per-pixel gathers of T
channel rows per view, weighted-summed ACROSS views.  The current XLA form (E0) is one
fusion doing the per-view work with the view sum folded in; its cost is the
transaction-bound row gathers over a 512 MB per-step working set (E2a: not
L2-capacity-bound at the vmap level — but chunkable below it, see lever 2).

*Why back is EASIER than forward was:*  (1) NO sort, NO segments, NO skew — every pixel
has exactly T taps, so the work is perfectly uniform (forward's whole v2/v3 saga was
skew handling); (2) single-write outputs by construction (each program owns its output
cells — no atomics anywhere); (3) the precompute is just the weight formula — no
permutation: concrete centers (V, P) already exist per call, plus a precomputed A
(V, T, P) f32 = 3× the centers' bytes (≈1.2 GB chunk-resident full-grid, ~5 MB at VCD
subsets), or 2× with in-kernel weight arithmetic.  coeff_power=2 (the Hessian) is the
same kernel with squared weights — a precompute flag.

*The kernel (ASTRA's register-tile, adapted):*  grid = (row-chunk, pixel); each program
owns out[p, r·RC:(r+1)·RC] in REGISTERS and loops over ALL V views × T taps,
ref-gathering RC-float row chunks from the channel-major (V, C, rows) sinogram, one
store at the end.  Two levers XLA cannot express:
  1. **Cross-view register accumulation** — the view sum never touches memory (XLA's
     fusion re-materializes per-view partials into the reduction).
  2. **Row-chunk L2 residency** — with the row-chunk as the SLOWEST grid dimension,
     every program in a chunk phase gathers from the same (V, C, RC) slice: RC=128 →
     ~65 MB ≈ L2, RC=64 → ~33 MB comfortably resident.  The transaction-bound gathers
     become L2 hits — the direct attack on the E2a-confirmed limiter, unavailable to
     the fused XLA kernel (whose gathers span the full row width).
Program count P × rows/RC (~65k at the 1024³ cell) keeps occupancy high; the V×T-long
program body is the latency-hiding question (num_warps sweep; fallback if single-program
-all-views stalls: split the view axis across the grid with a small second-phase
reduction — the two-phase machinery from forward v3 reuses directly).

*Slice-set generality from day one:*  the row-chunk dimension indexes through a small
row-index table, so contiguous bands (the multi-device band path — the (c) target) and
strided parity sets are the same kernel with different tables.  For parallel beam,
rows ≡ slices, so THIS kernel IS the band kernel; cone back adds the vertical-fan
fusion as its own design step (E0 showed cone's hfan hides behind the vfan, so cone
back needs the fused treatment to pay).

*Bench plan (mirrors the forward spike):*  baseline = the library back kernel
(stacked gather, coeff_power ∈ {1,2}) vmapped + view-summed, at the raster and subset
batches of the 1024³ cell; success bar ≥1.5–2× at both; value gates rel ≤1e-5 PLUS an
explicit adjoint check against the Pallas FORWARD kernel (⟨A x, y⟩ = ⟨x, Aᵀ y⟩ on
random pairs — the pair ships together per the E4 agreement).  Sweeps: RC ∈
{64, 128, 256}, num_warps ∈ {1, 2, 4}, view-loop-in-program vs view-split-grid.

**E4 — composition (only if E3 clears its bar):** TilePolicy-gated integration of the
pilot band kernel on the multi-device back path (cone first), model-level n=1/2/4 A/B —
the scaling curve IS the deliverable for (c) — plus VCD guard cells (the
CUDA-graph/dispatch caveat), memory gates, fallback-path check on a non-Hopper arch.  The
campaign lesson applies in full force: measure compositions, not pieces (cone back is the
geometry where per-fan substitutions were a proven no-op — the band kernel restructure is
a whole-kernel change precisely for that reason).  The parallel-back kernel and the cone
forward (fused vfan+hfan design) are the second wave, justified by E4 + E1's cone split.

**Decision gates:** after E0/E1 — if the traffic model is wrong (fusion already tight),
re-rank §4.  Note the MAIN (c) track is E1-independent: the band-kernel work is justified
by the n=1/2/4 scaling curve and gated by E0's band ncu, not by the VCD device share.  E1
gates the SECONDARY (a) track: if production-granularity VCD kernel share is low, (a)
shrinks to A1/A2 + the full-grid one-shot workloads.  After E2 — if A5 alone restores
n=2 scaling, the Pallas band kernel's bar rises accordingly; if A1+A2 reach ≥1.5× fwd
end-to-end, the bar for a Pallas FORWARD kernel rises.  After E3 — bar not met → stop at
the XLA level and document.

## 7. Safety/verification requirements (Greg 2026-07-12: required before going all in)

- **A1 rounding-class check**: A1 reintroduces an argsort-then-regather SHAPE.  The eager
  concrete-output pattern is the same defense Phase D established, and all quantities are
  deterministic integer ops on concrete inputs — but run the T15-style repro suite against
  the A1 formulation before trusting it, plus a debug-mode assertion that the concrete
  ids are monotone (a chunk-alignment bug would otherwise silently mis-reduce under
  `indices_are_sorted=True`).
- **Value gates**: A1/A2 target BITWISE equality (stable sort, same keys); Pallas kernels
  change summation order → the standard scale-invariant rel-max gates (1e-5 single-shot /
  1e-4 iterated), never exact equality; adjointness preserved by construction (same A) —
  add a fwd/back adjoint dot-product test to the kernel-equality suite.
- **Memory gates**: §5's deltas go through the acknowledged-regression path with the
  numbers stated up front.
- **Guard rails**: production-shape probes FIRST (992/1008, and the translation
  wide-detector shapes stay excluded — the collision-cliff lesson); per-arch policy
  gating with the XLA path as the always-correct fallback (the established
  platform-conditional pattern); no eager array ops in wrappers (the +35% lesson).

## 8. Logistics

- **Branch/worktree** (Greg's proposal, agreed): new branch, suggested `greg/gpu_headroom`,
  in a SEPARATE worktree locally and on gautschi — isolates kernel-spike churn from the
  flash-remediation implementation in the main checkouts.  Branch off
  `greg/kernel_investigation` (the work builds on the TilePolicy/campaign code); after the
  kernel PR merges to main, re-anchor with the squash-merge ancestry recipe
  (`git merge -s ours origin/main` before merging).  On gautschi: dedicated staging dir
  (pattern of `~/flash_p2b`), and select code-under-test by installing into a dedicated
  env — never `PYTHONPATH` (lessons §5); mind the NFS scp-staleness wait before srun.
- E0/E1 are measurement-only: scripts live in `plans/experiments/projector_kernels/`
  (layout rule), runnable from the new worktree; no library changes until E2 approval.
- Workflow: discussion-first for all library code changes; per-step real-data/production
  validation as in the flash-remediation program.

## Appendices (in `headroom_appendices/`)

- `appendix_roofline_traffic_model.md` — the verified-shape traffic model and bounds.
- `appendix_amdahl_accounting.md` — VCD loop structure, workload table, missing
  measurements and cheapest instruments.
- `appendix_pallas_assessment.md` — Pallas/Mosaic-GPU/Triton capabilities at the 0.10.1
  pin, composition, portability, risks (with sources).
- `appendix_ct_kernel_practice.md` — projector taxonomy across ASTRA/TIGRE/LEAP/CTorch/
  svMBIR, normalized throughputs, matched-vs-unmatched literature, CSR/register-tile
  prior art (with sources).
- `appendix_xla_lowering.md` — how XLA:GPU lowers our scatter/sort/gather patterns
  (verified against openxla source + local HLO), flag inventory, ranked no-custom-kernel
  levers (with sources).
