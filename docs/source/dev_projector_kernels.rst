.. _ProjectorKernels:

=================
Projector kernels
=================

The two projectors are the heart of the reconstruction: **forward projection**
turns a volume of voxels into a sinogram (what the detector would have seen), and
**back projection** turns a sinogram into a volume (spreading each measurement back
along its ray).  Every iteration of the reconstruction calls both, so their speed
sets the speed of the whole solver.

This page describes how those kernels are built and where they live in the code.  It
picks up *below* the multi-device level: :doc:`dev_sharding_overview` explains how a
reconstruction is split across devices into bands, and this page explains the kernel
that then runs on each band or shard.  There are two layers.  The **XLA kernels** are
written in ordinary JAX and run everywhere -- CPU, any GPU, every geometry; they are
the baseline and the permanent fallback.  On top of them, for the geometries and GPUs
where it has been measured to help, sits an optional layer of **Pallas** custom
kernels that squeeze out several more times the throughput.  Both layers come from the
2026-07 kernel campaign; the full design notes and every measured number are in the
repo under ``plans/projector_kernels/``.


The XLA kernels
---------------

**One tile policy.**  A projection has several tunable knobs -- how many views to
batch at once, how wide a slice band to stream, which of two reduction algorithms to
use, and so on.  Rather than scatter these through the code, they are collected into a
single ``TilePolicy`` object (``model.tiles``), chosen in one place
(``TomographyModel._select_tile_policy``) and re-chosen whenever the device layout
changes.  Every consumer reads the policy at call time, so a re-layout takes effect on
the next projection.  A geometry overrides only the knobs it has actually measured, and
an experiment can override one from the outside with
``model.tiles = model.tiles._replace(...)``.

**Forward scatters, back gathers.**  The two directions are not mirror images.  Back
projection reads each detector value and adds it into the voxels along the ray -- a
*gather*, with no collisions.  Forward projection does the opposite: many voxels land
on the same detector channel, so the kernel's inner loop is a *scatter-add* into
shared locations, and on a GPU those colliding atomic adds were essentially the entire
forward-kernel cost.  To avoid them, the GPU forward kernel first **sorts the
contributions by channel** and then reduces each run of equal channels as a segment
(``projectors.channel_scatter_reduce``), behind the ``sort_by_channel`` flag.  Parallel,
cone, and multiaxis turn it on; **translation deliberately does not** -- at its real
detector shapes the sorted form measured 4.5--6.5x *slower*, because too few voxels
collide per channel for the sort to pay for itself.  Three guard constants in
``projectors.py`` (minimum columns, maximum PSF radius, minimum collision ratio) encode
where the crossover sits, and CPU keeps the simple scatter loop throughout.  The lesson
those guards carry is worth stating plainly: a kernel-choice policy has to be validated
at production shapes, not just at the benchmark cell where it was born.

**Geometry decides whether a slice can stream.**  For parallel beam a detector row
looks straight across at exactly one recon slice, so back projection can crop the
sinogram to a band of rows and produce just those slices, and forward projection can do
the reverse -- this is what lets the multi-device code stream one slice band at a time.
For the diverging (vertical-fan) geometries -- cone, and its relatives -- a single
recon slice spreads across a *range* of detector rows, so that trick is unavailable: a
view-owner must read the full detector rows and, on the forward side, hold the whole
voxel cylinder at once.  This is why cone forward projection always works on whole
cylinders and cone back projection always reads full rows, under both the XLA and the
Pallas paths (the multi-device reduce-scatter still bands the *output* slice axis, but
each band is computed from the full rows).  It is also why the kernels themselves differ
by geometry, described below.

**Stacked back gather (parallel only).**  The parallel-beam GPU back kernel gathers all
of a voxel's PSF taps in one shot (``back_stacked_gather``).  Every geometry's back
kernel honors the flag, but only parallel's policy turns it on: for the vertical-fan
geometries the gather is already hidden behind the heavier band work, so stacking it
changes nothing (measured as a no-op on three geometries).

**Shared machinery, and the rounding defense.**  The trapezoid-tap arithmetic that both
fans share lives in one place (``horizontal_fan_project`` / ``horizontal_fan_back`` /
``vertical_fan_band_gather`` in ``projectors.py``); a geometry file contributes only its
own coordinate math and weight scale.  One subtle point: the integer detector channel
each voxel maps to is computed in a small separate step and passed into the projector as
a concrete input, so the compiled projector never rounds internally.  That is a
deliberate defense against a known XLA rounding hazard (see
``plans/bugs_and_artifacts/`` in the repo).  Relatedly, the projector wrappers keep a
**no-eager-array-ops** discipline: a single stray eager array operation per call
measured +35% on whole-reconstruction time, because at interactive sizes the solver is
limited by how fast the host can dispatch work, not by the GPU.

**Where the remaining time goes.**  After all of the above, the two dominant kernels --
the forward sorted reduction and the back gather -- still run about 10x above their
compute-only bounds.  Profiling attributes the gap to memory *access patterns* rather
than to raw bandwidth: the kernels move data efficiently in bulk but revisit it in an
order the hardware caches poorly.  Closing that gap is what a hand-written custom kernel
can do, and is the reason for the second layer.


The Pallas custom kernels
-------------------------

The custom kernels are written in **Pallas**, a JAX facility for writing GPU kernels in
plain Python that the already-installed JAX compiles through its Triton backend -- there
is no separate CUDA build step, and the code ships in the package like any other module
(``mbirjax/_pallas_kernels.py``).  Every custom kernel is value-equal to the XLA kernel
it replaces up to the order in which floating-point terms are summed, and the XLA kernel
stays compiled in at every call site as the fallback, so the custom path can be turned
off at any time with no loss of function.

**What runs on which kernel.**  The custom path was built one increment at a time, each
adding a kernel behind a ``TilePolicy`` flag.  There are four dispatch points:

* *Back projection on a single GPU* (``back_pallas``) -- taken by the ``n = 1``
  short-circuit for a one-GPU reconstruction.  Parallel beam uses a register-tile row
  kernel; cone uses a fused vertical-fan kernel.
* *Back projection across several GPUs* (``back_pallas_band``) -- each device's share of
  the views is back-projected onto a slice band through the same kernels, with the band
  orchestration otherwise unchanged.
* *Forward projection on a single GPU* (``fwd_pallas``, parallel beam) -- serves every
  problem size; a 70-cell sweep found the custom kernel faster at every point, so there
  is deliberately no size threshold.
* *Forward projection across several GPUs* (``fwd_pallas_band``, parallel beam) -- each
  device forward-projects its slice band through the same kernel.

Only parallel beam and cone enable any of these, and only on the GPUs where they were
measured; cone has a custom back kernel but still uses the XLA forward.  Every other
geometry, and all CPU work, stays on the XLA kernels.

**Gradient and Hessian can take different paths.**  Back projection is called with the
weights unchanged for the gradient and *squared* for the Hessian diagonal, and a
geometry may send one through the custom kernel and keep the other on XLA.  The class
attribute ``_PALLAS_BACK_COEFF_POWERS`` lists which it serves.  Parallel beam serves
both: its weights are reproduced exactly.  Cone serves the gradient only -- its fused
kernel computes the detector row from a compact in-kernel formula that carries a tiny
(~2e-5) error in the *squared* weights, and because the solver divides the gradient by
the Hessian, that error is amplified at low-Hessian edge voxels (a test measured 8.5e-3
divergence in the reconstruction when the Hessian went through the kernel).  So cone
takes the once-per-reconstruction Hessian on the exact XLA path and only the
per-iteration gradient through the kernel.  The base class serves neither and raises if
a flag is ever set without a matching kernel, so a misconfiguration fails loudly rather
than silently mis-projecting.

**Turning it on, and seeing what ran.**  A geometry enables a flag only on the GPU and
only where it measured a win, and every flag is additionally gated by a runtime check,
``_pallas_kernels.is_available()``, that has to pass three tests: an environment
kill-switch (``MBIRJAX_DISABLE_PALLAS=1``) must be unset, the GPU must be on a small
allowlist of models the kernels were tuned for, and a tiny probe kernel must actually
compile.  If any fails, the model silently uses XLA.  Because the fallback is silent,
two hooks report what a run will really use, and you should check one before trusting a
timing comparison: ``model.get_compute_config(print_results=True)`` prints the full tile
policy and, when the custom path is off, *why*; and the ``Reconstruction devices:`` log
line gains a ``(pallas: ...)`` tag naming the active kernels (for example
``(pallas: band-back+band-fwd)`` for a multi-GPU parallel-beam run).


How the kernels work
--------------------

**The parallel-beam back kernel: a register tile.**  Back projection sums, for each
voxel and detector row, a handful of weighted sinogram values across every view.  The
XLA version does this a view at a time and its cost is the scattered row reads.  The
custom kernel instead launches one tiny GPU program per (row-chunk, voxel) that keeps
its output in registers and loops over *all* views and taps internally, so the running
sum never leaves the chip.  The programs are laid out so that the ones running at the
same time all read the same small slice of the sinogram, which then stays resident in
the L2 cache.  Because every voxel has exactly the same number of taps, the work is
perfectly uniform -- no sorting, no atomics, each output written once.

**The cone back kernel: a fused vertical fan.**  Cone back projection has an extra
factor: each voxel spreads over a small block of detector rows as well as channels (a
3x3 tap block rather than a row of taps).  The custom kernel folds both fans into one
program, which is possible because of a geometry fact -- the detector row a voxel maps
to is an exact linear function of the slice index, so the kernel can march up the slices
of its register tile and recompute the row cheaply as it goes, without any per-slice
precomputed table.  As noted above it reads the full detector rows (a cone slice draws
from a range of them), and at ``n = 1`` it covers the whole cylinder in one launch.

**The forward kernel: sort, then store in two phases.**  The forward kernel handles the
scatter-add collisions described earlier by sorting the contributions by channel and
walking them in fixed-size segments, writing the result in two passes (a plain store,
then an atomic accumulate) so that one very busy channel cannot stall an entire launch.
Its launch shape is fixed by the array shapes, never by the data, so it compiles once
and is reused across every iteration.

In all three, the weights come from exactly the same geometry code the XLA kernels use,
so forward and back stay a matched pair and the two paths agree to floating-point
reordering noise -- which is what the correctness tests check
(``tests/test_pallas_kernels.py``, run in Pallas *interpret* mode on the CPU CI and
compiled on the GPU).


The rules a Pallas kernel must follow
-------------------------------------

These are hard contracts, not style preferences: each was measured as a large slowdown
when broken.

* **Shapes come from arrays, never from data.**  Every launch and block size is derived
  from array shapes.  A shape that depends on the *values* changes the compiler's cache
  key on each reconstruction subset and triggers a recompile in the middle of the solve.
* **No host round-trips in the hot path.**  Nothing pulls a device array back to the
  host (no ``np.asarray`` / ``device_get``) inside a per-call path; a single such sync
  per view chunk stalls the pipeline and turns the forward kernel's win into a loss.
  View bookkeeping is done in NumPy from shapes alone, never fetched from the device.
* **Build each kernel once.**  The Pallas call objects and their JIT wrappers are cached
  per shape (``functools.cache``); constructing one per call re-traces and re-compiles
  every time.
* **Gather at the reference, not after a load.**  In-kernel gathers index the sinogram
  reference directly (``ref[idx, :]``); the alternative -- load a block, then gather from
  it -- does not compile on the pinned JAX version.

One backend note: on Hopper GPUs a bare Pallas call selects the wrong backend for these
kernels, so the Triton backend is requested explicitly through ``compiler_params`` --
that is backend *selection*, not performance tuning.


Updating or retiring a kernel
-----------------------------

The tuned constants (the register-tile and segment sizes, the slice-chunk size, and the
GPU allowlist) all come from the benchmark scripts under
``plans/experiments/projector_kernels/``.  On a new GPU model or a JAX upgrade, rerun
those to revalidate the constants and the speedups, then extend the allowlist only with
measurements in hand; the probe compile in ``is_available()`` catches hard toolchain
breaks on its own.  To retire a kernel, set the environment kill-switch or delete its
one policy line -- every call site keeps the XLA kernel it was always compiled with, so
nothing else has to change.
