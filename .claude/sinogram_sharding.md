# Discussion of sinogram sharding strategies

## Overview

Current (06/13/2026) code uses view-sharding for sinograms.
Since most big recons have many views, this has the advantage of providing
good allocation across many GPUs.  However, it has the disadvantage of 
breaking locality, particularly for parallel beam, in which sharding
by detector rows makes projections embarrassingly parallel.  

The following is a discussion of these alternatives.  Here, C (or Option C) 
refers to distributing all the slices of a pixel batch to the view-owners
and doing projection on full voxel cylinders.  

## Prompt
Let's pause for a moment and ask some exploratory questions:  no code yet, no definitive plans, just a curiosity-first exploration.  First, do you think that cone-beam would parallelize more efficiently if the recon were sharded along rows or columns instead of slices?  Part of the problem seems to be that one slice-band projects to significantly different parts of a view. If I understand correctly, C is in some sense mimicking this alternate sharding through the transfer.  Second, in back projection, the horizontal projection takes longer than the vertical projection.  But the cone horizontal projection is essentially the same as the parallel beam projection, which is horizontal only.  Are these two horizontal projections comparable in efficiency?

## Response
A: Love this framing — let me think it through geometrically, because both questions turn out to point at the same underlying structure.

### First, the axis map

Recon is `(rows i, cols j, slices k)`; sinogram is `(views, det_rows m, det_channels n)`. The two fans couple different axes:
- **Vertical fan**: recon **slices k (z)** ↔ **det rows m**. A localized, magnification-spread relationship (z → a *window* of rows).
- **Horizontal fan**: recon **(i, j) / (x, y) plane** ↔ **det channels n**, *via the ray integral* — each channel sums a ray across the whole x–y plane.

The key asymmetry: the slice↔row coupling is **local** (a band of slices touches a bounded window of rows), but the (x,y)↔channel coupling is **global** (every channel's ray crosses the entire x–y extent of the object).

### Q1 — would row/column sharding be better? I think it's *worse*, and it clarifies why slice is right

If you shard the recon by **row or column** (the x–y plane), then every detector channel's ray crosses *every* x–y shard, so each channel value is a sum of partials over **all** shards — a full all-reduce per output element. That's heavier communication than slice-sharding, where the only cross-shard coupling is the **local vertical spread** (a boundary halo). So the slice axis is actually the *right* recon shard axis for cone — its coupling to the output is localized, not global.

But here's what I think your intuition is really pointing at: **cone factors as "parallel-beam (horizontal, per-slice-independent 2-D radon) + a vertical z-coupling."** In parallel beam the slices are *independent* (slice k → row k, no z-coupling), which is why slice-sharding is embarrassingly parallel there. Cone is the same per-slice 2-D radon **plus** the vertical fan's z-spread — and that spread is *local* (a halo), exactly like the qGGMRF prior's slice halos.

So the inefficiency you're sensing isn't slice-sharding — it's that the current scheme shards the **sinogram by view** and therefore has to either all-gather the recon to each view-owner (forward) or reduce over view-owners (back). **C's gather is a symptom of that**: it reconstitutes the full z-extent because the view-sharding doesn't align with the geometry. An alternative — **recon sharded by slice, sinogram sharded by det-row, aligned, with a vertical halo** — would do the sum-over-views *locally* (each row-owner holds all views for its rows) and exchange only boundary halos. That's potentially much lighter communication, and it's "parallel-beam sharding + a halo," which the codebase already knows how to do.

I don't know the history, so genuine questions: **why was view-sharding chosen over a slice/row-aligned halo scheme?** Is it the VCD structure (view-subset partitions, the error sinogram) that wants views co-resident, or was it just the path of least resistance from the F1 view-sharded FBP filter? Does the halo idea collide with how VCD updates subsets? You'll know whether this is a dead end or worth a deeper look.

### Q2 — cone-horizontal vs parallel-beam back: they *should* be the same op

You're right that they're structurally identical. Parallel-beam back projection **is** the horizontal/channel radon (slice = row, so it's purely the 2-D channel back-projection producing `(pixels, rows)`). Cone's `back_horizontal_fan` produces the same `(pixels, det_rows)` from `(rows, channels)` by the same channel-psf gather/accumulate. The only structural difference is that cone's `compute_horizontal_data` carries the per-pixel magnification (channel mapping scales with `pixel_mag`) where parallel's `compute_proj_data` doesn't — but that's per-pixel scalar work, not the `(pixels × rows × psf)` bulk. So I'd *expect* them comparable; if they're not, it's an implementation gap (the magnification arithmetic, a curved-detector branch, layout) worth closing.

This is cheaply measurable — time cone's `back_horizontal_fan` against parallel-beam's full `back_project_one_view_to_pixel_batch` at matched sizes (the fan-split harness already isolates the cone half). My prediction: **cone-horizontal ≈ parallel-back, and cone-back ≈ parallel-back + the vertical fan.** If that holds, the GPU back-projection "horizontal bottleneck" isn't a cone inefficiency at all — it's the inherent cost of the channel radon, which parallel beam pays too; cone just adds the vertical z-fan on top. Which again says: **cone ≈ parallel + the vertical z-coupling.**

### The through-line

Both questions converge on: *treat cone as parallel-beam plus a local vertical (z) coupling.* That reframing suggests the most efficient sharding isn't a new recon axis but a **halo-aligned slice/row decomposition** that keeps projection local and exchanges only the z-spread at boundaries — and it predicts the horizontal fans are already as efficient as parallel beam's (measurable). Whether the halo scheme is worth pursuing hinges entirely on the VCD/view-sharding history you have and I don't — so I'm curious what made view-sharding the choice, and whether a halo scheme was ever on the table.

## Prompt
I advocated for detector row-sharding, at least for parallel beam, but was outvoted.  At the time, it wasn't clear to me that row-sharding would be better for anything except parallel beam, so I gave in, with the plan to revisit row-sharding for parallel beam after we implemented view-sharding.  One of our goals is code simplicity; we'd rather not have different sharding strategies for different geometries.  Even now, it's not completely clear to me that row-sharding will be better for translation or multi-axis parallel, but it's possible it wouldn't be worse, and maybe that's good enough, particularly since conebeam is really the primary target.  

I think the question of view vs row sharding is essentially independent of vcd.  The sinogram really enters only through projections, so how the sharding is done should make no difference to the rest of the code.  

One question with row-sharding is the fact that the halos are not the same as the are for qggmrf: some pixel bands would overlap very little with neighboring row bands, while others would overlap a lot.  How would that be handled?

## Response

That history is clarifying — and your row-sharding instinct for parallel beam looks even stronger once cone is the primary target. Let me work through the variable-halo question, because I think it actually resolves into something clean and unifying rather than messy.

### Why the halo varies (confirming your observation)

A recon slice at axial coordinate `z` back-projects from detector rows around `m(z) ≈ (pixel_mag·z)/Δ_row + center`. The *spread* of that — how many rows one slice touches — comes from the per-pixel magnification range: `Δm ≈ z·(mag_max − mag_min)/Δ_row`. So the spread is **proportional to |z|**:
- Slices near iso (`z ≈ 0`): spread ≈ just the psf — a tiny halo.
- Slices at the axial extremes (`|z|` large): spread ≈ `|z|·Δmag/Δ_row`, which can be 10–30% of the detector height for a typical cone.

That's exactly your "some bands overlap little, some a lot": the **central** device needs almost no halo; the **end** devices need a wide one. (And for parallel beam, `mag_max = mag_min`, so the spread collapses to zero — slice = row exactly, no halo at all. That's the perfect-locality case you were after.)

### How to handle it: a per-device, geometry-computed *footprint* (not a fixed halo)

The qGGMRF halo is a fixed `±k` neighbor count because the prior coupling is uniform. The projection coupling isn't — so don't use a fixed halo; instead, **each device computes the contiguous det-row range its own slices project to/from**, from the geometry (its slice z-range × the volume-wide magnification bounds × the view params), and gathers exactly that range. Concretely, device `d` owns rows `[m_d0, m_d1)` but its footprint is `[m_lo_d, m_hi_d]` ⊇ its own rows, extending into neighbors by however much its slices actually spread. The non-uniformity is then *just* non-uniform footprint widths — the central device's footprint barely exceeds its own rows, the end device's reaches further — and nothing special-cases it.

Key properties that make this attractive:
- **It's static.** Geometry is fixed at config time, slices map monotonically to rows, so each footprint is a precomputed contiguous range. No data-dependent or runtime halo logic — just a config-time computation of `[m_lo_d, m_hi_d]` per device and the resulting gather/scatter pattern.
- **It's one mechanism across geometries.** The *strategy* is "gather your slices' row-footprint"; only the *footprint computation* is geometry-specific (which it has to be — the geometry defines the projection). Parallel → footprint = own rows → **zero transfer**. Cone → own rows + variable halo. Translation/multiaxis → own rows + whatever their geometry gives. That's the single strategy you want, with parallel getting its row-sharding locality for free as the degenerate case.
- **It strictly dominates C.** The footprint is always a *subset* of the full cylinder's rows, so footprint-gather is never more communication than C's full gather, and it's *zero* for parallel and *small* for central cone bands. Worst case (wide cone, axial extreme) it degrades gracefully toward C — never worse. So as a replacement for C it's all upside.

Forward is the adjoint: each device projects its slices, producing rows in `[m_lo_d, m_hi_d]`; the halo rows (which belong to neighbors) are **scatter-added** to the owning device. So back = halo-gather + local back-project; forward = local project + halo scatter-add. The horizontal fan stays fully local (each device has the complete x–y plane for its slices → all channels), and — your VCD point — the **view-sum is local** (a row-owner holds all views for its rows), so there's no view-reduce at all. Communication is *only* the variable vertical halo.

### The honest caveats

- **Helical** is the real stress case: a slice is seen by views spread over a z-travel, so its footprint spans roughly the helical pitch — a wide, possibly device-spanning halo. Still ≤ full gather, but the locality advantage shrinks. Worth checking what the pitch does to the footprint before assuming row-sharding helps helical.
- **Wide cones / large axial extent** push the end-device footprints toward "most of the detector," degrading toward C. For the primary target (circular cone, modest cone angle) the footprints should stay modest, but the break-even is a measurement, not a derivation.
- **Translation / multiaxis**: the footprint just gets computed from their geometry; likely wider than cone (per-view translation / elevation move things) but still ≤ full gather, so "not worse than C, sometimes much better" — which by your bar is good enough, especially with cone as the target.

### What I find compelling here

This reframes the whole thing: **C is the footprint-gather with a trivial (all-rows) footprint.** Replacing C's "gather everything" with "gather your geometric footprint" gives you (a) one strategy for all geometries, (b) parallel-beam's row-sharding win as the zero-halo degenerate case — finally getting the thing you advocated for — and (c) cone communication that scales with the actual vertical spread instead of the full cylinder. The thing to measure before believing it: the **actual footprint widths** at representative cone angles and at the axial extremes (and the helical pitch case), to confirm the halos stay well below "most of the detector." That's a cheap geometry computation — no projection needed, just `m_lo_d/m_hi_d` per device over the slice ranges and mag bounds — which I could work out for the standard test geometries if you want to see whether the halos are actually small before we take this any further.

One thing I'm now curious about, given you were outvoted: was the objection to row-sharding specifically the *variable halo* (this question), or something else — load balancing across the non-uniform footprints, or the det-row count being a coarser sharding axis than views?

## Prompt
This is very helpful.  

Re the variable halos:  jax requires shards to be equal along the shard axis.  Does a variable halo violate this?  It seems like each row-owner would have a different length row-footprint, which seems counter to how jax operates naturally.  

Re the objection to row-sharding:  some of it is stubbornness, but some of it is the fact that sometimes you want to reconstruct just a few central slices, so there's limited room for multiple GPUs.

## Response

Both excellent — the first has a clean answer, the second is the real strategic crux.

### Q1 — does the variable halo violate jax's equal-shard rule?

No, because the variable thing isn't a shard of a jax array — it's a **per-device local working buffer**. There are two different kinds of array here, and the equal-shard rule only touches one:

- **The persistent sharded arrays** — recon (slice-sharded) and sino (row-sharded) — *do* have to be equal along the shard axis. They stay equal, padded to a multiple of `n_dev` exactly as P5 already does for views/slices. Each device's *output* (its recon slice-shard for back, its sino row-shard for forward) is equal-size, so `assemble_sharded` wraps them into a valid `NamedSharding` array as today.
- **The footprint/halo** is a *transient local array* each device builds by gathering (`move_shard` a neighbor's row-range, concat with its own). It's never a partition of a single sharded array, so it can be any size per device. This is exactly how the codebase already operates: `run_per_device` runs a worker per device that produces a local array, and **C's gathered cylinder is already a per-device local of size "all rows."** The footprint scheme just makes that local buffer *smaller and variable* (own rows + the geometric halo) instead of "all rows."

So C is the degenerate "uniform all-rows footprint"; the footprint scheme is a tighter, variable one — and both live entirely in the per-device-local layer, below jax's equal-shard requirement.

The one place the variation actually bites is **jit shapes**: a different footprint length per device means a different compiled program per device — `n_dev` traces (fixed and cached for static geometry, but it's the tracing-cost concern from §7). The escape, if that matters, is to **pad every footprint up to the max** (the axial-extreme device's width): uniform shape → one program, at the cost of the central devices gathering/computing a bit more than they need. So it's a tight-variable-vs-padded-uniform choice, not a jax obstruction.

### Q2 — the thin-central-slices objection is the real one, and it's robust

This is the genuinely strong argument for view-sharding, and it's not stubbornness. **The view axis is essentially always large** (hundreds–thousands of views), so view-sharding gives you parallelism *regardless of recon shape* — including a 4-slice central recon on 8 GPUs. Slice/row-sharding parallelizes along the slice (and aligned row) axis, which **is exactly the axis a thin recon makes small** — 4 slices can't use 8 GPUs. So:

- **View-sharding**: scales to arbitrary GPU count independent of recon thinness; the robust all-regime choice. Cost: cone's forward needs the gather (C), which we measured ≈ no regression.
- **Row/slice-sharding (footprint)**: communication scales with the local vertical spread (great for large cone recons; zero for parallel beam — your original win), but parallelism is capped by the slice/row count → thin recons can't use many GPUs.

They're genuinely complementary regimes, and they're *mutually exclusive* per problem: row-sharding's whole advantage is doing the view-sum **locally**, which is precisely giving up view-parallelism. You can't have both without a 2-D (slice × view) mesh — which is the opposite of the simplicity goal.

So I think the honest read is: **view-sharding is the more robust single strategy** (works for thin and tall, scales to any GPU count), and the current scheme already *works* for cone (C forward ≈ no regression, banded reduce-scatter back). Row-sharding's appeal — communication locality for large cones, and finally giving parallel beam its zero-halo locality — is real but regime-specific, and it carries the thin-recon limitation plus the variable-halo machinery.

That suggests a pragmatic split that doesn't force the strategic decision now:
- **Finish cone on the existing view-sharding** (C forward + banded back) — low incremental cost, no-regression, unblocks the port and the team.
- **Keep row-sharding as a documented future exploration** — its strongest case is parallel beam (the thing you originally advocated, where it's *pure* locality with zero halo and no cone-angle caveat) and large circular-cone capacity runs. If the footprint widths turn out small at representative geometries (the cheap geometry computation I offered), it could eventually become a unifying *replacement* — but only if the thin-recon regime is handled (e.g., a config-time axis choice), which is a bigger design question.

A reframing that might interest you: the thin-recon objection is fundamentally "**you sometimes need to parallelize an axis other than the one you're sharding for capacity.**" View-sharding picks the always-abundant axis; row-sharding picks the capacity axis. The only way to truly have both is a 2-D mesh, and the question is whether any real workload needs *both* many GPUs *and* few slices *and* large views simultaneously — if thin recons are also small/fast, maybe the few-GPU limitation there just doesn't hurt, and row-sharding's large-recon wins dominate where it matters.

Where I land for *now*, curiosity aside: the measured "C ≈ current" plus view-sharding's robustness says we can finish the cone port on the current scheme without regret, and row-sharding is worth a real evaluation later — starting with parallel beam (its best case, and your original proposal) and the cheap footprint-width computation — rather than blocking cone on it. Does that match your read, or do you think row-sharding's cone-communication win is large enough that it's worth settling the architecture before finishing the port?


