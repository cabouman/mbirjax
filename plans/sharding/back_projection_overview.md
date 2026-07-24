A high-level view of TomographyModel and projections:  

Sinograms are stored as (views, rows, channels)

Recons are stored in two ways: 
1. recon: (rows, columns, slices)  (for ParallelBeam, slices = detector rows, i.e. recon slice s is back-projected from sinogram detector row s; this 1-to-1 alignment is not true for other geometries such as cone beam, where magnification maps a slice to a geometry-dependent range of rows)
2. flat_recon: (rows x columns, slices)

A 'pixel' is a point in (row, column) space, either in the 2D form or the flattened form.  A 'voxel cylinder' is the 1D set of voxels at a single (row, column) point, hence has length slices. The internal projection code operates on a batch of voxel cylinders, identified as a batch of pixel indices pointing into the flat_recon.  

Tomography model has a user-facing `back_project()`, which converts a recon to a flat recon, selects a region of interest, then calls `sparse_back_project()`, still within TomographyModel, which then further calls jitted code in projectors.py, which manages most of the distribution and collection of voxel cylinders and sinograms to reduce the problem to the geometry-specific per-view kernel `back_project_one_view_to_pixel_batch()` (ParallelBeam's in parallel_beam.py; cone's in cone_beam.py).  

There is some top-level batching of views and pixels in `sparse_back_project()`.  Each batch is processed by `self.projector_functions.sparse_back_project()`, which -- after the B3 de-closuring -- is a thin reference to the module-level `_sparse_back_project()` in projectors.py (formerly the per-instance closure `sparse_back_project_fcn()`).  `_sparse_back_project()` is jitted and uses `lax.scan` (`sum_function_in_batches()`) to sum the back projection over batches of views, then `lax.map` (`concatenate_function_in_batches()`) to concatenate over batches of pixels, ending in a `vmap` on the per-view kernel.  

Since we're sharding recon by slice and sino by view, a single back projection will mean that each device will back project all of its views onto all of the slices in the voxel cylinders in this batch.  Then the relevant slices from each voxel cylinder will have to be summed across the view-holding devices (the back projection sums over views) and scattered to the slice-owning device (a reduce-scatter).

Note (ParallelBeam-specific): because detector row r maps only to slice r with no cross-row mixing (the kernel's offset loop runs only over channels), a device can produce just the slices a given destination needs by first slicing its sinogram views to those rows -- so the transient cylinder buffer can be held at one destination's slice range rather than the full slice span.  Cone beam cannot restrict this cheaply (a slice maps to a data-dependent band of rows), so it computes the full cylinder once and splits it.  

Cone now has two back kernels (P6): a rolled-pixel single-device kernel (`back_project_one_view_to_pixel_batch`) and a band kernel (`back_project_one_view_to_band`) for the multi-device reduce-scatter.  They have opposite platform rankings, so the sharded driver selects by platform (GPU n=1 -> pixel; CPU and multi-device -> band).  See `sharding_implementation_plan_v3.md` §4 and `lessons.md`.

This system supports a CPU, a single GPU (sino and recon together), and multiple GPUs (sino view-sharded, recon slice-sharded).  An earlier hybrid mode -- recon on a CPU with the sino on a GPU -- was dropped (2026-06-08).

Some of this batching machinery (the nested scan/map/vmap) may be simplified in a future refactor (tracked in `sharding_implementation_plan.md` §"Future project: simplify the sparse-projector batching machinery").
