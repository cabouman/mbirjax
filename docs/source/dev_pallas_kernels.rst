.. _PallasKernels:

========================
Pallas projector kernels
========================

The 2026-07 GPU-headroom campaign added a custom-kernel path for the projectors,
written in **Pallas** (pure Python, compiled by the JAX already shipped -- no CUDA
build step; the Triton backend).  It exists because, after the XLA kernel campaign,
the two dominant projector kernels still sat roughly 10x above their compute-only
bounds -- limited by memory *access patterns* rather than HBM bandwidth, a gap only a
custom kernel can close (the substrate it builds on -- the tile policy, the sorted
channel reduction, the forward/back asymmetry -- is described under
:doc:`dev_sharding_overview`).

This page is the developer-oriented architecture of that path: what the kernels are,
where they are dispatched from, and how to update or retire them.  The design
rationale and every measured constant live in the repo under
``plans/projector_kernels/`` (``e4_integration_design.md`` is the integration design;
``gpu_headroom_findings.md`` is the measured record; the ``e3_*``/``e4_*``/``e5_*``
scripts under ``plans/experiments/projector_kernels/`` reproduce every number).  All
kernels and their drivers live in one module, ``mbirjax/_pallas_kernels.py``.


What lives here, and where it dispatches
----------------------------------------

The path grew one increment at a time; each adds a kernel behind a ``TilePolicy`` flag
and a dispatch hook, with the XLA path kept compiled-in as the fallback.  The current
map, by operation:

* **Back projection, single device** (``back_pallas``).  The GPU ``n = 1``
  short-circuit in ``_sparse_back_project_sharded`` routes here (a single-GPU recon;
  multi-device recons take the banded path below).  The driver is geometry-specific,
  reached through the hook ``_pallas_back_project_single_device``: parallel beam uses
  the **register-tile row kernel**; cone uses the **fused vertical-fan kernel**.
* **Back projection, multi-device band** (``back_pallas_band``).  At ``n >= 2`` each
  view-owner's per-owner slice-band back call routes through the same kernels, via the
  ``_back_project_view_shard_to_band`` override.  For parallel beam detector row ``r``
  maps to slice ``r``, so the band is a detector-row crop and the single-device kernel
  *is* the band kernel; for cone a slice draws from a *range* of rows, so the fused
  kernel runs on the full-row view with an explicit slice band.  The driver's
  per-owner mode takes the owner's GLOBAL view indices and never re-places data, so the
  banded reduce-scatter orchestration is unchanged.
* **Forward projection, single device** (``fwd_pallas``, parallel beam).  Serves ALL
  pixel counts -- there is deliberately no pixel-count guard: a 70-cell value-gated
  sweep (``fwd_guard_sweep.md``) measured Pallas faster at every point with no
  crossover, because past L2 the kernel streams at near the HBM traffic bound while XLA
  pays the same traffic plus its sort/scatter constant factor.  Dispatched from
  ``projectors.sparse_forward_project_public``.
* **Forward projection, multi-device band** (``fwd_pallas_band``, parallel beam).  At
  ``n >= 2`` each view-owner forward-projects its broadcast slice band through the same
  forward kernel, via the ``_forward_project_band_to_view_shard`` override (the adjoint
  of the back band seam).  The band-broadcast orchestration is unchanged.

Only ``ParallelBeamModel`` and ``ConeBeamModel`` enable any of these, and only on GPU;
every other geometry, and CPU, keeps the XLA path.


The gradient / Hessian split (``_PALLAS_BACK_COEFF_POWERS``)
------------------------------------------------------------

Back projection is called with ``coeff_power = 1`` for the gradient and
``coeff_power = 2`` for the Hessian diagonal (which back-projects the *squared*
weights).  A geometry may serve one power through Pallas and keep the other on XLA,
so the class attribute ``_PALLAS_BACK_COEFF_POWERS`` lists which powers its Pallas
back drivers serve; the dispatch takes the Pallas path only when the flag is set AND
``coeff_power`` is in that tuple.

* **Parallel beam: ``(1, 2)``** -- its trapezoid weights are exact-class (the kernel
  reproduces the XLA weight formula), so both the gradient and the once-per-recon
  Hessian go through Pallas.
* **Cone: ``(1,)`` -- gradient only.**  The fused kernel computes the detector-row
  center ``m`` from an in-kernel affine expression, which carries a ~2e-5 error in the
  *squared* (Hessian) weights.  VCD divides the gradient by the Hessian, and at
  low-Hessian edge voxels that division amplifies the small Hessian error -- the
  increment-5 trajectory gate measured 8.5e-3 recon divergence with the Hessian taken
  through the kernel.  So cone keeps the Hessian on the (exact-center) XLA path; the
  gradient, which is not divided by anything that small, is safe.
* **Base ``TomographyModel``: ``()``** -- no Pallas back path.  The base
  ``_pallas_back_project_single_device`` raises ``NotImplementedError``, so a policy
  that sets the flag without defining a driver fails loudly rather than silently
  misprojecting.


Policy, availability, and fallback
----------------------------------

**Policy.**  Every flag above is a field of the single ``TilePolicy`` (``model.tiles``),
set in the geometry's ``_select_tile_policy`` and read late-bound at each call, so a
``configure_devices()`` re-layout takes effect on the next projection.  A geometry
enables a flag only where it has measured a win (GPU only; the ``n == 1`` flags and the
``_band`` flags are mutually exclusive by device count).

**Availability.**  Every flag is gated by ``_pallas_kernels.is_available()``, which
combines three guards: an environment kill-switch (``MBIRJAX_DISABLE_PALLAS=1``), a
device-kind allowlist (H100 at present -- extend only with measurements), and a probe
compile of a tiny kernel.  An incompatible toolchain therefore falls back to XLA
silently, once per process.

**Fallback.**  The XLA kernels remain compiled-in at every call site.  They serve CPU,
non-allowlisted GPUs, every geometry that does not enable a flag, and any
``coeff_power`` a geometry does not route through Pallas.  Retiring a kernel is
removing its one policy line; every call site keeps working.


Introspection
-------------

Because the fallback is silent, two hooks report which kernels a run will actually use
-- check one before trusting any cross-machine timing comparison:

* ``model.get_compute_config(print_results=True)`` prints the full tile policy and,
  when Pallas is unavailable, WHY (which of the three availability guards failed).
* The ``Reconstruction devices:`` log line appends a ``(pallas: ...)`` token naming the
  active paths -- ``back``, ``fwd``, ``band-back``, ``band-fwd`` -- e.g.
  ``(pallas: band-back+band-fwd)`` for a multi-device parallel-beam recon.


How the back kernels work
-------------------------

**Register-tile + L2-phase (parallel beam).**  Back projection is, per pixel ``p`` and
detector row ``r``, a sum over views ``v`` and psf taps ``t`` of
``A[v,t,p] * sino[v, center[v,p]+t, r]``.  The XLA path evaluates this per view and
sums, and its cost is transaction-bound row gathers.  The kernel instead runs one small
GPU program per ``(row-chunk, pixel)`` that holds its output row-chunk in registers and
loops over ALL views and taps, so the view sum never touches memory (the "register
tile" -- the piece the whole-array XLA model cannot express).  The row-chunk grid
dimension varies slowest, so concurrent programs gather from the same
mostly-L2-resident sinogram slice (the "L2 phase").  Work is perfectly uniform -- every
pixel has exactly ``psf_width`` taps -- so there is no sort, no atomics, and each output
cell is written once.

**Fused vertical fan (cone).**  A cone back tap is a 3x3 product: the horizontal fan's
3 channel taps times the vertical fan's 3 row taps.  The kernel extends the
register-tile design to compute both fans in-program, which is possible because of a
geometry fact -- the detector row center ``m`` is *affine* in the slice index
(``m(v,p,l) = m0(v,p) + slope(v,p) * l`` exactly, flat and curved detectors alike), and
the row-weight scale is slice-independent.  So the vertical fan needs no per-slice
precompute, only three scalars per (view, pixel); the program holds its output
slice-chunk in registers, loops the view chunk, and accumulates the 3x3 tap products
factored as row-weight x channel-scalar multiplies.

Both kernels rebuild their trapezoid weights from the same geometry chain the XLA
kernels use (``compute_hfan_data``), so the operator -- and hence forward/back
adjointness -- is identical; only the float summation ORDER differs, so results agree
to reordering noise and are gated at the standard relative tolerance.  The integer
channel centers are the existing concrete-centers arrays, so the XLA rounding-bug
contract carries over.


How the forward kernel works
----------------------------

The forward horizontal fan is a scatter-add with *duplicated* channel indices (roughly
``psf_width * pixels / channels`` pixels collide per channel), so the colliding atomic
adds are essentially the whole XLA forward-kernel cost.  The Pallas forward kernel sorts
the taps by channel and walks them in fixed-size segments, storing in **two phases** (a
first store phase then an atomic-add phase) so a hot channel cannot stall a whole
launch.  Its launch and segment shapes derive from array shapes only -- never from data
-- so it does not recompile inside the recon loop.  It serves every pixel count with no
guard (see the dispatch map above).


Constraints that must hold
--------------------------

These are contracts, not style preferences -- each was measured as a large regression
when violated:

* **Data-independent launch shapes.**  Every launch / grid / block shape derives from
  ARRAY SHAPES only.  A data-dependent shape changes a Triton cache key per VCD subset
  and triggers a recompile *inside* the recon loop.
* **No per-call host<->device sync.**  No ``np.asarray`` / ``device_get`` of a device
  array in any per-call path: one sync per view chunk stalls the pipeline and flips the
  forward kernel's win into a loss.  (View bookkeeping stays in NumPy, computed from
  shapes, never pulled from the device.)
* **Construct kernels once.**  ``pl.pallas_call`` objects and their jitted wrappers are
  built once per static shape (``functools.cache``); per-call construction re-lowers and
  re-traces on every call.
* **Ref-level gathers only.**  In-kernel gather of a block-LOADED array does not lower
  on either Pallas backend at the pinned JAX version; gathers are ref-level indexing
  (``ref[idx, :]``).  Whole-array block specs are pointer refs on the Triton backend,
  not materialized copies (interpret mode does materialize them -- acceptable for tests
  only).

A further backend note: a bare ``pallas_call`` selects the Mosaic backend on Hopper,
which the cone kernel's copies do not satisfy; the Triton backend is selected
explicitly through ``compiler_params`` (backend *selection*, not tuning).


Updating or retiring the path
-----------------------------

The tuned constants (``ROW_CHUNK``, ``NUM_WARPS``, ``FWD_SEGMENT_CAP``, ``CONE_LC``,
and the ``_ARCH_ALLOWLIST``) all come from the bench scripts named above.  On a new GPU
architecture or a JAX upgrade, rerun those scripts to revalidate the constants and the
speedups, then extend ``_ARCH_ALLOWLIST`` only with measurements in hand; the probe
compile in ``is_available()`` catches hard toolchain breaks automatically.  Correctness
gates in ``tests/test_pallas_kernels.py`` (relative-tolerance equality, chunking
consistency, the per-owner band equality, and the adjoint identity) run in Pallas
interpret mode on CPU CI and compiled on GPU.  To retire a kernel entirely, set the
environment kill-switch or remove its one policy line -- every call site falls back to
the XLA kernel it was always compiled with.
