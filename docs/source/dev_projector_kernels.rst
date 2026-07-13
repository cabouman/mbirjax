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
attribute ``_PALLAS_BACK_COEFF_POWERS`` lists which it serves.  Both parallel beam and
cone serve both powers.  Their accuracy contracts differ: parallel reproduces the XLA
weights exactly, while cone's fused kernel computes the detector row from a compact
in-kernel formula whose floating-point rounding differs slightly from the XLA chain's
-- invisible in the gradient (the trapezoid taps are a partition of unity, so the
difference cancels) and a small, bounded effect (~2e-5, against a 1e-4 contract) in
the squared Hessian weights.  The base class serves neither and raises if a flag is
ever set without a matching kernel, so a misconfiguration fails loudly rather than
silently mis-projecting.

A calibration note for anyone comparing reconstructions across kernel paths: after a
few solver iterations, *any* implementation that sums in a different order -- a custom
kernel, or even the XLA path with a different view batch -- diverges pointwise at the
weakly-constrained voxels near the edge of the field of view, where the solution is
ill-conditioned and float-level differences are amplified about a thousandfold.  A
max-norm comparison over the full support therefore only ever passes for bitwise
identical code.  Kernel-path equivalence is instead verified occasionally (not in the
nightly runs) at the reconstruction level: both paths are run on real data against
deep reference reconstructions, and their convergence curves must agree within a
noise band calibrated by an XLA-vs-XLA control (the record lives with the campaign
notes under ``plans/projector_kernels/``).

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


Supporting a new GPU model
--------------------------

The custom kernels are enabled only on the GPU models named in ``_ARCH_ALLOWLIST``
(currently H100 alone), and that restriction is deliberate.  The constants baked into the
kernels -- ``ROW_CHUNK`` and ``NUM_WARPS`` for the parallel back kernel,
``FWD_SEGMENT_CAP`` and ``FWD_NUM_WARPS`` for the forward kernel, ``CONE_LC`` and
``CONE_NUM_WARPS`` for the cone kernel -- are *measurements on an H100*, not universal
choices, and so are the speedups themselves.  The ``is_available()`` probe that gates each
flag only confirms that a kernel *compiles*; it says nothing about whether it is *faster*.
An unmeasured GPU therefore keeps the XLA path by design, and enabling the custom kernels
on a new model is a small validation project rather than a one-line edit.  (A JAX or
toolchain upgrade is the same kind of change for the same reason -- the constants and
speedups are measured against a particular compiler as well as a particular GPU -- so it
calls for the same re-validation.)

**Find the device string first.**  The allowlist matches by substring against the GPU's
``device_kind``.  ``model.get_compute_config(print_results=True)`` prints that string in
its ``pallas_status`` line -- it names the device it saw and the allowlist it failed -- so
run it on the target machine to learn exactly what to add.

**Then walk the validation ladder**, in order, correctness before speed and kernels before
the assembled solver.  Each rung has a script under
``plans/experiments/projector_kernels/``:

* **Correctness.**  Run ``tests/test_pallas_kernels.py`` *compiled on the new GPU*.
  Everywhere else the tests run in Pallas interpret mode, which emulates the kernel on the
  host and never touches the real GPU backend, so this is the only step that exercises the
  actual compiled kernel -- temporarily add the device to the allowlist (or monkeypatch it)
  so the tests take the Pallas path.
* **Kernel constants.**  Re-run the tuning sweeps and adopt what they find best on the new
  hardware: ``e3_back_pallas_v1.py`` (the row-chunk and warp count for the parallel back
  kernel), ``e5_cone_fused_back.py`` (the slice-chunk and warp count for the cone kernel;
  it also runs a day-0 lowering probe that aborts with a clear message if the backend
  cannot compile the kernel's gathers -- a new-architecture instance of the Hopper
  backend-selection trap noted above), and ``fwd_guard_sweep.py`` (the forward kernel
  across problem sizes).  The forward sweep matters especially: on H100 the custom forward
  was faster at every size, which is why the dispatch carries no size threshold -- but that
  "no crossover" result is an H100 measurement, and a different architecture is exactly
  where a size threshold could reappear, so re-run the sweep rather than assume the
  guard-free dispatch still holds.
* **Composed gates.**  Run the model-level A/B harnesses -- ``w2_inc3_ab.py`` and
  ``w2_inc4_ab.py`` (the parallel back and forward bands) and ``w2_inc5_ab.py`` (the cone
  back kernel) -- which compare a full projection or short reconstruction flag-on against
  the kill-switch (``MBIRJAX_DISABLE_PALLAS=1``) and report wall time, value agreement, and
  per-GPU peak memory.  These catch the integration costs a bare kernel bench cannot.
* **Convergence, occasionally.**  For a change that reaches the reconstruction itself (the
  cone Hessian in particular), run ``w2_inc5_convergence.py``.  As the calibration note
  above explains, a max-norm comparison across kernel paths is unpassable after a few
  iterations, so this gate instead asks whether both paths *converge equally well* on real
  data, within a noise band set by an XLA-vs-XLA control.

**Only after all of that, extend the allowlist**, and record the constants and speedups in
``plans/projector_kernels/gpu_headroom_findings.md`` next to the H100 numbers, so the next
person can see what was measured where.  One caution: if the new GPU turns out to want
*different* constants than the H100, do not split the difference and guess a compromise.
Per-architecture constant tables are a real design decision for the kernel maintainers --
flag it and let them choose, rather than inventing a mechanism on the spot.

**When a kernel loses on the new hardware, leave it off.**  If a sweep shows a kernel no
faster than XLA on the new model, or slower, do not add it to the allowlist: the silent XLA
fallback is the correct end state, not a failure to fix.  This is the lesson the
translation geometry already taught, where the "faster" sorted reduction measured 4.5--6.5x
*slower* at real detector shapes -- a kernel choice is valid only where it was measured, and
declining to enable one is a legitimate outcome of the validation, not a gap in it.


Retiring a kernel
-----------------

To turn the custom path off, set the environment kill-switch
(``MBIRJAX_DISABLE_PALLAS=1``), which disables every custom kernel at once; to retire a
single kernel, delete the one policy line that sets its flag.  Either way every call site
keeps the XLA kernel it was always compiled with, so nothing else has to change.
