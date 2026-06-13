A high-level view of TomographyModel and projections:  

Sinograms are stored as (views, rows, channels)

Recons are stored in two ways: 
1. recon: (rows, columns, slices)  (for ParallelBeam, slices = detector rows, i.e. recon slice s is back-projected from sinogram detector row s; this 1-to-1 alignment is not true for other geometries such as cone beam, where magnification maps a slice to a geometry-dependent range of rows)
2. flat_recon: (rows x columns, slices)

A 'pixel' is a point in (row, column) space, either in the 2D form or the flattened form.  A 'voxel cylinder' is the 1D set of voxels at a single (row, column) point, hence has length slices. The internal projection code operates on a batch of voxel cylinders, identified as a batch of pixel indices pointing into the flat_recon.  

Tomography model has a user-facing `back_project()`, which converts a recon to a flat recon, selects a region of interest, then calls `sparse_back_project()`, still within TomographyModel, which then further calls jitted code in projectors.py, which manages most of the distribution and collection of voxel cylinders and sinograms to reduce the problem to `back_project_one_view_to_pixel_batch()`, which lives in ParallelBeam (parallel_beam.py).  

For the prerelease version of back projection, there is some top level batching of views and pixels in `sparse_back_project()`.  One batch is then processed by `self.projector_functions.sparse_back_project()` is an alias to `sparse_back_project_fcn()` in projectors.py.  

Within projectors.py:  
`sparse_back_project_fcn()` is jitted and uses lax.scan through `sum_function_in_batches()` to sum the back projection over batches of views returned from `sparse_back_project_view_batch()`.  

`sparse_back_project_view_batch()` uses lax.map through `concatenate_function_in_batches()` to concatenate batches of pixels returned from `back_project_pixel_batch()`, which uses a vmap on `back_project_one_view_to_pixel_batch()`, which is a geometry-specific function.  We're interested in the version in ParallelBeam.  

Since we're sharding recon by slice and sino by view, a single back projection will mean that each device will back project all of its views onto all of the slices in the voxel cylinders in this batch.  Then the relevant slices from each voxel cylinder will have to be summed across the view-holding devices (the back projection sums over views) and scattered to the slice-owning device (a reduce-scatter).

Note (ParallelBeam-specific): because detector row r maps only to slice r with no cross-row mixing (the kernel's offset loop runs only over channels), a device can produce just the slices a given destination needs by first slicing its sinogram views to those rows -- so the transient cylinder buffer can be held at one destination's slice range rather than the full slice span.  Cone beam cannot restrict this cheaply (a slice maps to a data-dependent band of rows), so it will instead compute the full cylinder once and split it.  

This complicated system was designed for use with either a CPU, a single GPU containing both the sino and the recon, and a CPU containing the recon with a GPU containing the sino and voxel cylinder batches transferred as needed.

In the context of this problem, we may be able to simplify some of this, starting from `sparse_back_project()` in TomographyModel through `back_project_one_view_to_pixel_batch()`.    



