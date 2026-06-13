# P6 increment B — cone two-stage banded projector: design note (2026-06-13, DRAFT for review)

*Implementation plan for increment B of the cone port.  High-level design + the
measurements that justify it: `p6_projector_rework_proposal.md` (read §8a-design).
Increment A — channel-major horizontal fans — is committed (CPU win, GPU-neutral,
build-verified).  This note is the concrete structure + the STAGED checkpoints, each
with correctness / memory / timing gates (Greg's request).*

**PROGRESS (2026-06-13):**
- **B1 DONE (staged, CPU-green):** banded cone kernels + `tests/geometries/test_cone_banded.py`
  (circular + helical: back band-decomposition, full-band==monolithic, adjoint-at-(g0,L),
  Hessian).  Review fixes applied (RETIRE markers, plain comments, renames, helical).
- **Forward structure DECIDED = C** (per-pixel-batch all-gather + monolithic; harness
  `cone_forward_structure_compare.py`): B (banded/streamed) is 5–14× slower on CPU
  (dispatch-bound) for ~13–23% memory; C ≈ current (no regression).  So the cone sharded
  **forward does NOT band**; **back stays banded** (reduce-scatter).
- **OPEN before B2:** remove the now-unneeded forward banded kernel
  (`forward_project_band_to_one_view` / `forward_vertical_fan_band_*`) + adjust
  test_cone_banded (its forward tests / the adjoint, which uses forward_band) — Greg to
  confirm; back_band correctness is already gated by the back band-decomposition.
- **NEXT:** B2 (single-device + sharded driver: back on the banded kernel; forward =
  gather + monolithic; delete `entries_per_cylinder_batch`), gated by memory/timing vs
  the §8a baseline.  Then B3 (de-closuring), B4 (sharded cone + GPU validation), B5
  (inert padding), then C/D/E.

---

## 0. Goal and non-goals

**Goal.**  Give cone the banded projector structure the multi-GPU sharded path needs
(slice-banded recon ⇄ view-sharded sinogram via a reduce-scatter), with NO
single-device regression and bounded per-device memory, then turn on
`_supports_sharding()` for cone.

**The value is CAPACITY, not single-GPU speed** (§8a-design): on GPU the cone
projectors already scale ~N⁴ with no cliff and fit 1024³ single-device for the
projectors; sharding is what lets *VCD* exceed one GPU at 1024³+ (the measured wall).
So B is judged on correctness + memory-shards-1/n_dev + no-single-device-regression,
NOT on a single-GPU kernel speedup.

**Non-goals (deferred):** translation/multiaxis ports (increment D); the retirement
cascade (increment E); any row-window or fused/sino-accumulator structure (measured
out — proposal §2/§4).

---

## 1. The two-stage decomposition (what changes in the kernels)

Today cone's per-view kernels are monolithic two-stage compositions:
- back: `back_project_one_view_to_pixel_batch` = horizontal_fan → vertical_fan.
- forward: `forward_project_pixel_batch_to_one_view` = vertical_fan → horizontal_fan.

B splits them so the driver can compute the **horizontal fan ONCE per view** and
**band the VERTICAL fan by global slice `(g0, L)`** (Option B; horizontal dominates
on GPU and recomputing it per band would ~n_dev²× the dominant stage).  The four
sub-kernels already exist as static methods — B exposes them as the driver's seams
and adds the `(g0, L)` band argument to the vertical fans:

| stage | existing method | B change |
|---|---|---|
| back horizontal | `back_horizontal_fan_one_view_to_pixel_batch` | none (channel-major, increment A) — computed once per view → det cylinder `(pixels, num_det_rows)` |
| back vertical | `back_vertical_fan_one_view_to_pixel_batch` → `..._to_one_pixel` | **add `(g0, L)`**: produce global slices `[g0, g0+L)` only; `compute_vertical_data_single_pixel`'s `slice_indices = g0 + arange(L)` (the seam already takes slice indices — cone_beam.py:617) |
| forward vertical | `forward_vertical_fan_pixel_batch_to_one_view` → `..._one_pixel...` | **add `(g0, L)`**: input is a band of L slices; emit the full `(pixels, num_det_rows)` contribution for those slices |
| forward horizontal | `forward_horizontal_fan_pixel_batch_to_one_view` | none (channel-major) — computed once per view on the accumulated det cylinder |

**New per-view banded kernels** (the driver-facing interface, added ALONGSIDE the
monolithic ones; F1 "build next to, delete loser"):
- `back_project_one_view_to_band(sinogram_view, pixel_indices, single_view_params, projector_params, g0, num_band_slices, coeff_power=1)` → `(num_pixels, L)`.
  Body: `det_cyl = back_horizontal_fan(view, ...)` (full rows, once if the driver
  hoists it — see §3); `band = back_vertical_fan_band(det_cyl, g0, L, ...)`.
- `forward_project_band_to_one_view(band_values, pixel_indices, single_view_params, projector_params, g0)` → `(num_det_rows, num_det_channels)` contribution.
  Body: `det_cyl_contrib = forward_vertical_fan_band(band_values, g0, ...)`;
  `view_contrib = forward_horizontal_fan(det_cyl_contrib, ...)`.

`g0` is **traced** (dynamic scalar); `L`=`num_band_slices` is **static** (≤2 values
from `_balanced_slice_bounds` → ≤2 programs).

---

## 2. Anchor rule + global validity clip (the correctness-critical math)

**Anchor (proposal §3):** physical coordinates from PROBLEM shapes + GLOBAL indices,
never the band length.  In the vertical fans, the slice→z map must use
`S_real = projector_params.recon_shape[2]` and `k_global = g0 + k_local`:
`z(k_global) = Δ_slice·(k_global − (S_real−1)/2) + (recon_slice_offset − helical_z_shift)`.
- Back: `compute_vertical_data_single_pixel` already derives z from `recon_shape`
  (params) — it just needs `slice_indices = g0 + arange(L)` instead of the chunk
  indices.  **Bit-exact today** on full cylinders (g0=0, L=S_real).
- Forward: `forward_vertical_fan_one_pixel_to_one_view` currently derives
  `num_slices = voxel_cylinder.shape[0]` (the INPUT length) and centers on it
  (cone_beam.py:390,402,434) — this BREAKS under banding (a length-L band would
  shift z).  **Fix: source `S_real` from params and `k_global = g0 + arange(L)`.**

**Global validity clip (proposal §5 — exactly-inert padding):** the vertical fans'
slice-validity becomes a GLOBAL test `k_global < S_real` (and `≥ 0`):
- Back-vertical: zero output slices with `g0 + local ≥ S_real` (padded slices, inert).
- Forward-vertical: input band slices with `g0 + local ≥ S_real` contribute zero
  (already zero by the forced-zero invariant; the clip is defense).
This makes padded slices exactly inert in-kernel, so pad-to-multiple-of-n is
provably N-independent (the P5 invariant), and `_mask_padded_slices` / the
forward-mask stay as the one-site postconditions.

---

## 3. Driver structure (horizontal-once + pixel-batch + band)

### 3a. Single-device (increment B2)
The single-device projector must drive band-looping too (so `entries_per_cylinder_batch`
can be deleted and the kernel path is unified).  The memory knob is the EXISTING
internal `pixel_batch_size` (projectors.py); the band is the new slice knob.

Back (pseudocode; one view-batch, inside the existing pixel-batch loop):
```
for pixel_batch P:                      # bounds the pixels×rows transient (existing knob)
    det_cyl = back_horizontal_fan(views, P)          # (P, num_det_rows)  -- ONCE
    for band (g0, L) in balanced_slice_bounds(S_real):
        out[P, g0:g0+L] = back_vertical_fan_band(det_cyl, g0, L)   # cheap
# sum over views happens as today (vmap over views + sum)
```
Forward (adjoint):
```
for pixel_batch P:
    det_cyl = zeros(P, num_det_rows)
    for band (g0, L):
        det_cyl += forward_vertical_fan_band(voxels[P, g0:g0+L], g0)   # accumulate
    view += forward_horizontal_fan(det_cyl, P)        # ONCE
```
Horizontal computed once per pixel-batch (not per band).  `det_cyl` transient =
`pixel_batch × num_det_rows` — same floor as parallel.  `entries_per_cylinder_batch`
chunking deleted (the band loop subsumes it).

### 3b. Sharded (increment B4)
Recon slice-sharded, sino view-sharded; back is a reduce-scatter.  Restructure
`_back_project_all_bands` so each VIEW-OWNER computes its horizontal det cylinder
ONCE and reuses it across all bands (else horizontal is recomputed n_dev²×):
```
for pixel_batch P:                                  # bounds per-device pixels×rows
    for view-owner d:  cyl[d] = back_horizontal_fan(view_shard[d], P)   # ONCE, on d
    for slice-owner t:
        for band (g0, L) in t's slice-shard:
            partials = [back_vertical_fan_band(cyl[d], g0, L) on each d]  # (P, L)
            band_t   = reduce-scatter(partials) onto t                    # sum over views
        owned_p[t] = concat_bands
    accumulate owned_p into owned (over pixel-batches, on the pixel axis)
owned = _mask_padded_slices(owned)                  # postcondition (proposal §5)
```
Forward is the all-gather adjoint: per pixel-batch, broadcast each slice-band to all
view-owners, `forward_vertical_fan_band` → accumulate into each view-owner's det
cylinder, then `forward_horizontal_fan` once → view-shard; `_mask_padded_views`
postcondition.  The transitional driver branch routes "geometry has banded kernels?"
→ this path, else the existing parallel row-crop path (tagged RETIRE-AFTER).

**Loop-ordering note (the one real subtlety):** pixel-batch must be the OUTER loop
so the held `cyl[d]` is `P×rows` not `pixels×rows`.  This differs from today's driver
(bands outer, full pixels to the projector).  This is the bulk of B4's code and its
main risk — gated by the memory check below.

---

## 4. De-closuring (increment B3)

Lift the projector drivers from per-instance closures (projectors.py
`sparse_back_project_fcn` etc.) to MODULE-LEVEL functions taking explicit args
(`projector_params` static — already a hashable namedtuple; the kernel pair static;
view-params traced).  Shares the jit cache across model INSTANCES (sweeps, the vcls
sibling stop re-tracing) and is the template the settable-view-params lift already
established.  **Preserve ONLY the two top-level signatures** (`sparse_forward_project`,
`sparse_back_project`); everything else is a simplification target.  Tail batches pad
to one signature (proposal §7).  Can land before or after B4; recommended AFTER B4
(so the band structure is settled first), but the gate is just allclose + a
trace-sharing check.

---

## 5. Staging with per-stage gates (correctness / memory / timing)

Each stage lands CPU-green for review.  Hard gate = CORRECTNESS (allclose; project
rule — never exact for computed floats).  MEMORY/TIMING are REPORTED vs the recorded
§8a baselines (machine-dependent → guard-rails, not hard fails), via the scaling
harness; GPU runs are Greg's.

| stage | what | CORRECTNESS gate | MEMORY gate | TIMING gate |
|---|---|---|---|---|
| **B1** | banded kernels added (cone `*_to_band`), anchor rule, global clip; NOT wired | NEW unit tests: band-decomposition (full == Σ/concat of bands at arbitrary (g0,L), 1e-5) + adjoint-at-(g0,L) (`<A_band x,y>=<x,A_bandᵀy>`, rel 1e-5/1e-6 per the projector-noise rule); g0=0,L=S_real == legacy kernel (1e-5) | — (pure kernel) | — |
| **B2** | single-device cone on banded kernel; horizontal-once + band; delete `entries_per_cylinder_batch` | cone test_projectors / test_fbp_fdk / test_vcd (existing convergence gates) green; allclose vs pre-B output 1e-5 | `cone_baseline_scaling.py` (CPU 64–256³, GPU 256³–1024³): peak ≤ recorded §8a baseline (report ratio; ≤1.1× ok) | same harness: time ≤ ~1.1× baseline (report; the §8a numbers are the ruler) |
| **B3** | de-closuring (module-level banded drivers) | allclose vs B2 (1e-5); full suite green | — | trace-sharing: program-cache HIT across two fresh model instances (first-call cost drops) |
| **B4** | sharded cone (reduce-scatter on banded vertical); `_supports_sharding()=True` | NEW `tests/sharding/` cone tests: sharded vs single-device 1e-4 iterated (const + non-const weights), adjoint identity; trivial-1-device-mesh vs single-device 1e-5 | `cone_baseline_scaling.py` multi-device sweep (n_dev 1/2/4, GPU): per-device peak ~1/n_dev | multi-device speedup ≥ projectors' measured ceiling; VCD 1024³ that OOM'd single-device now FITS sharded (the capacity win) |
| **B5** | exactly-inert padding for cone (global clip already in B1; wire masks + non-dividing n) | NEW: padded-slice exact-zero invariant (constructed-zero → EXACT equality); padded recon vs unpadded 1e-4; auto-pads a prime slice count | per-device peak unchanged by padding (inert) | padding cost in the noise (report) |

Parallel beam is untouched through B1–B5 (its row-crop path stays); increment C
converts it and deletes the monolithic cone kernel + the transitional branch.

### B1 — LANDED (CPU-green, 2026-06-13)
Added to `cone_beam.py` ALONGSIDE the monolithic kernels (not yet wired):
`forward_vertical_fan_one_pixel_to_one_view_band` / `..._band_pixel_batch`,
`back_vertical_fan_one_view_to_one_pixel_band` / `..._band_pixel_batch`, and the
driver-facing `back_project_one_view_to_band` (g0 traced, num_band_slices static) /
`forward_project_band_to_one_view`.  Anchor fix applied to the forward vertical fan
(z from params `S_real` + global `k = g0 + arange(L)`, gather via local index);
back vertical already anchored on params (just takes global band indices).  Global
validity clip (`k_global < S_real`) in both — a no-op for g0+L ≤ S_real, the inert-
padding hook for B5.  Forward banded drops the `entries_per_cylinder_batch` det-row
chunking (produces all rows directly).
Tests: `tests/geometries/test_cone_banded.py` (6 tests × {circular, helical} subtests,
all pass) — back/forward band-decomposition, full-band == monolithic,
adjoint-at-(g0,L), Hessian (coeff_power=2) decomposition.  Monolithic cone suite
(test_projectors / test_fbp_fdk cone, 12) unaffected.

Review fixes (Greg, 2026-06-13): helical coverage added (exercises the anchor's
z-shift term); the monolithic-comparison tests marked RETIRE (retire when the
monolithic kernels are deleted), the adjoint test kept as a permanent self-contained
gate; plan notation (§/increment refs) removed from cone_beam.py in favor of plain
descriptions, with the only RETIRE marker on the genuinely-transient "coexists with
monolithic" note; the one-pixel banded fns renamed to `*_band_one_pixel` ("band" =
the slice band, not the output).

**OPEN (raised by Greg's review) — the forward banded vertical over-allocates.**
A band of L input slices reaches only a limited detector-row window, but the current
`forward_vertical_fan_band_one_pixel` evaluates ALL `num_det_rows` per band (most
zero) and the caller sums full columns — O(num_det_rows) work per band, i.e.
num_bands× the monolithic vertical work, plus a full-column intermediate per band.
This is the FORWARD-vertical analogue of the row window we measured out for BACK-
horizontal — but here it is genuinely needed, because back's vertical OUTPUT is the
small thing (L slices) while forward's vertical OUTPUT is the big thing (det rows)
and the band fills only ~W of them.  It is asymmetric: back has no over-allocation
(output is exactly L); forward does.  **Resolved at B4** (it depends on how slice
bands are delivered to each device), with these options to measure:
  (i) **Windowed forward vertical** — evaluate only the band's row window
      `[r0, r0+W)` (W static from worst-case magnification; r0 PER-PIXEL traced,
      since magnification/`t` vary per pixel) and scatter-add into the full column
      (which the horizontal-once stage needs anyway).  Saves the per-band vertical
      compute + intermediate; the per-pixel r0 placement is the added complexity.
  (ii) **All-gather full cylinder + monolithic forward** — don't band the forward at
      all; gather the whole slice cylinder onto each view-owner (pixel-batched to
      bound memory) and run the monolithic forward.  No window, no banded-forward
      kernel; departs from the current band-streaming (`broadcast_band_to_views`).
For B1 the banded-forward kernel is a CORRECTNESS REFERENCE (it validates back_band's
adjoint + the decomposition); its full-column body is annotated (TODO) as such.

**HARNESS + CPU MEASUREMENT (2026-06-13): `cone_forward_structure_compare.py`.**
Direct, empirical B-vs-C(-vs-mono) comparison at the per-view-owner level (recon
slice-sharded across n devices; one owner gathers (C) or streams (B); isolated
subprocess per variant for clean peak).  B implemented as option (i) with a PER-BAND
shared window (r0 per band, static W from worst-case magnification) + det-cylinder
accumulator + horizontal once; all three structures JITTED (apples-to-apples).
- **Correctness: mono == C == B** at every size/n_dev (CPU) — validates the windowed
  vertical fan + the gather/stream structures.
- **Memory (CPU RSS, n=2/4): B beats C at scale** — B/C peak **0.75–0.76× at 256³**
  (B never holds the full cylinder), ~1.0 at tiny sizes (RSS baseline).  Confirms B's
  capacity advantage.
- **Time (CPU): B is 4.9–13.7× slower than C** — even after hoisting the per-pixel
  geometry out of the band loop (which helped: 256³ 7.4→4.8× at n=2).  The remaining
  cost is per-band DISPATCH: B makes `num_bands` jitted calls per pixel-batch, which
  STREAMING REQUIRES (one band at a time from the remote slice-owner).  Fusing the
  band loop into one jit would remove the dispatch but force holding all bands at
  once — killing B's only advantage (the streamed memory).  So B is structurally
  dispatch-bound on CPU and cannot be rescued without giving up streaming.

**DECISION (measured, 2026-06-13): adopt C; B is out.**  CPU is a target, and B's
~5–14× CPU time penalty for a ~13–23% transient-memory edge is a bad trade (Greg's
"can't absorb ~10× for 25%").  C's memory is bounded by pixel-batch (gather only
`(P_b, S)` per pixel-batch), so the sharding CAPACITY goal is preserved by the recon's
sharded STORAGE — the forward just gathers transiently.  **Consequences (simplifying):**
- The cone sharded **forward = per-pixel-batch all-gather of the slice cylinder +
  the existing MONOLITHIC forward**.  No forward banding, no windowed vertical fan,
  no accumulator.
- The B1 **forward banded kernel is no longer needed for production**
  (`forward_project_band_to_one_view` / `forward_vertical_fan_band_*`).  Keep it only
  if it still earns its place as a test helper for the back_band adjoint; otherwise
  remove it (back_band correctness is already gated by the back band-decomposition).
- **Back projection still bands** (the reduce-scatter — back's vertical OUTPUT is the
  small thing, no over-allocation); the B1 back banded kernel stays.
- The earlier window-vs-no-window / fused-accumulator debate is fully resolved by
  this: forward doesn't band at all.

GPU run (Greg) is now optional CONFIRMATION (does C's pixel-batched gather stay
bounded at 1024³? is B's edge ever big enough to reconsider a second, GPU-only path?
— unlikely given the single-path + CPU-target requirements), not a decision input.

NEXT: B2 (single-device + sharded driver: back on the banded kernel; forward = gather
+ monolithic; delete entries_per_cylinder_batch; memory/timing vs the §8a baseline).

---

## 6. Tests to BUILD (Greg's "tests at multiple points")

**Correctness (pytest — fast, CPU, in the suite):**
- `tests/geometries/test_projectors.py`: extend with **band-decomposition** and
  **adjoint-at-(g0,L)** for cone (B1).  These are the strong per-geometry gates and
  they run on every suite invocation.
- existing cone recon/fbp/vcd convergence gates: re-run each stage (regression net).
- `tests/sharding/`: cone sharded-vs-single-device + trivial-mesh + padded
  exact-zero (B4/B5), mirroring the ParallelBeam sharding tests.

**Memory + timing (scaling harness — isolated subprocess; CPU local + GPU cluster):**
- `cone_baseline_scaling.py` is the ruler; re-run at B2 (single-device) and B4
  (multi-device sweep).  Add a small **`--compare <baseline.yaml>` report mode** (or
  a `replot`-style helper) that prints time/peak DELTAS vs the recorded §8a baseline
  per (op,size), flagging >1.1× memory or >1.3× time as ⚠ (reported, not asserted —
  timing is machine-dependent; the hard gate stays correctness).
- `cone_fan_split_microbench.py` / `cone_channel_major_ablation.py` remain available
  for localized re-measurement if a stage moves the cost structure unexpectedly.

**Why memory/timing are reported, not hard-gated:** the project rule is exact
equality is never the gate for computed floats; likewise absolute time/memory are
machine- and occupancy-dependent (the swap-contamination and GPU-variance episodes).
So correctness is the hard gate; memory/timing are tracked against the recorded
baseline with guard-rails and human judgment ("comparable or better, not literal" —
Greg).

---

## 7. Risks / open questions for review

1. **B4 loop reorder (pixel-batch outer) is the main risk** — it changes the sharded
   driver's structure and the per-device transient bound.  Mitigation: the memory
   gate (per-device peak ~1/n_dev) catches a wrong bound; build B2's single-device
   horizontal-once first so the ordering is proven before adding the reduce-scatter.
2. **Curved detector:** the v→m (row) map is linear even when curved (curvature is in
   u/channels), so banding the vertical fan should be curvature-agnostic — verify
   against `geometry_xyz_to_uv_mag`/`detector_uv_to_mn` in B1's tests.
3. **Helical z-shift** enters z per-view (traced via `single_view_params`) — confirm
   the banded vertical handles it (it reads `helical_z_shift` from `single_view_params`
   today; unchanged).
4. **Hessian diagonal** (`coeff_power=2`) goes through the back banded kernel — the
   `coeff_power` arg threads through unchanged; B1's adjoint test should also cover
   `coeff_power=2` (it is a back-projection).
5. **B3 ordering:** de-closure after B4 (recommended) vs before — does settling the
   band structure first ease the module-level lift, or does lifting first ease B4?
   Lean: B4 first (band structure is the harder unknown).
6. **Delete-the-loser timing:** the monolithic cone kernel + `entries_per_cylinder_batch`
   are deleted at B2 (single-device) — confirm nothing else (FDK? `compute_horizontal_data`
   callers?) depends on them before deleting.
