# Multiaxis elevation projector — corrections and evidence

Human-readable evidence for two corrections to the **vertical (elevation) fan** of
`MultiAxisParallelModel`. Run the script, open the four figures, read this.

```bash
# on a GPU node with the editable mbirjax env
python validate_elevation_fix.py     # prints tables, writes figures/*.png
```

## The two corrections (independent of each other)

**Elevation pathlength correction (a normalization bug).** A parallel-beam projection is a line
integral (OPL): the total projected mass of a fixed object must not depend on the viewing angle.
The old vertical fan deposited each voxel with amplitude `1.0`, so a voxel's projected mass fell
off as `cos(el)` (≈23 % low at 40°, 50 % low at 60°). Fix: a **mass-conserving amplitude**
`scaling = (Δ_slice/Δ_row)/W_p_r`, which equals `1/cos(el)` in the simple case but is computed
from the floored footprint so it never blows up — stable even at 90°. This is the dominant,
must-have fix.

**Elevation footprint correction (a separability approximation).** The vertical pass is the
`(t,z) → v` projection, i.e. 2-D parallel beam at angle `el`. Its footprint should be the same
**max-of-edges** amplitude-method footprint `ParallelBeamModel` already uses. The old code kept
only the foreshortened z-edge `cos(el)`; the corrected one also picks up the in-plane `sin(el)`
edge. Because `max` selects the z-edge below ~45°, this **changes nothing below 45° and only
helps above it**. It does not affect mass — it is a blur/shape refinement, a small safe bonus.

Why both went unnoticed: under **inverse crime** (simulate and reconstruct with the same
projector) both errors cancel, so they only appear against independent or real data.

## The temporary A/B flags

```python
model.set_elevation_correction(correct_elevation_pathlength=True,   # mass-conserving amplitude
                               broaden_elevation_footprint=True)    # max-of-edges footprint
```

The three modes compared throughout, named for what they apply:

| mode | pathlength | footprint |
|------|:----------:|:---------:|
| **uncorrected** | off | off |
| **pathlength** | on | off |
| **pathlength + footprint** | on | on |

These flags are scaffolding for review; they will be removed before merge (the corrected path
becomes unconditional). The durable checks live in `tests/`.

## The crime-free reference (how "truth" is made, and why to trust it)

The crime-free tests need data the model under test did **not** produce — otherwise the comparison
is an inverse crime. We make it by **oversampling**: the same object is placed on a grid `F=4`×
finer (each voxel block-replicated, so the physical object is unchanged) and projected with the
projector at that fine resolution onto the *same* detector. The separable projector's only
approximation is *per voxel* (the intra-voxel shear `~ Δ_voxel·sin el`), so at `F`× finer
resolution that error shrinks `~1/F` and the fine projection converges to the true continuous line
integral. The coarse model being tested never produced this data, so it is not an inverse crime.

Three reasons to trust it — all demonstrated with numbers by `check_reference_trust()` in the
script:

1. **It matches a projector-free truth.** A uniform sphere's projection is known in closed form,
   `2·sqrt(R² − ρ²)`, independent of any projector. With the sphere re-voxelized at each
   resolution, the `pathlength` projection converges to that analytic value down to the
   detector-pixel discretization floor (~0.017 NRMSE at el=40°, the same floor the corrected modes
   show in fig2), while `uncorrected` stays ~0.24 off — so that residual floor is *discretization*,
   and the projector once mass-correct reproduces the real line integral. That is why oversampling
   can be trusted for the structured phantom, where no closed form exists. (This is the answer to
   "aren't you validating a projector with a finer copy of itself?")
2. **It is mode-independent.** At fine resolution the footprint is sub-pixel and stops mattering:
   the reference built with `pathlength` vs `pathlength + footprint` agree to ~0 NRMSE, so it does
   not secretly favor any mode under test.
3. **It is converged.** Successive `F` (2 → 4 → 8) agree, so `F=4` is already in the converged
   regime, not a coincidence of one resolution.

## The figures (`figures/`)

- **fig1_mass_conservation.png** — total projected mass of a fixed sphere vs elevation. The
  *uncorrected* mode rides the `cos(el)` curve (wrong); the *pathlength* mode sits flat on the
  true mass. The normalization bug in one picture.
- **fig2_projection_accuracy.png** — forward-projection NRMSE vs elevation against *exact* truth
  (analytic sphere) and an *oversampled* reference. *uncorrected* diverges with elevation;
  *pathlength* flattens it to the discretization floor; *pathlength + footprint* matches it below
  45° and is slightly better above.
- **fig3_projection_images.png** — the actual detector image at 60°: *uncorrected* is globally
  too dim (the mass loss you can *see*); the corrected modes match the oversampled truth;
  difference maps below.
- **fig4_recon_crime_vs_nocrime.png** — reconstruction NRMSE, low (±20°) and high (±55°)
  elevation, `init_recon=0`. The **inverse-crime** bars are flat across modes — proof that
  inverse crime hides the bug. The **crime-free** bars (independent reference data) show
  *uncorrected* worst, the corrected modes best, the footprint term adding a little at high
  elevation.

## What the evidence shows

- The **pathlength correction is decisive on real data**, catastrophically so at high elevation:
  in the crime-free recon at ±55°, NRMSE drops from `0.33` (uncorrected) to `0.05` (≈6.5× better).
- The **footprint correction is a safe ~1 % bonus that only acts above 45°** (identical to the
  pathlength mode below it). Worth keeping behind the existing >45° warning; not load-bearing for
  typical low-elevation geometries.
- The FBP `direct_recon` filter is a separate item (already order-invariant on this branch);
  validated last.
