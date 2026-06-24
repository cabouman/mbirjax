.. _usr_multi_gpu:

Multi-GPU Reconstruction
========================

MBIRJAX can spread a single reconstruction across multiple GPUs (or, on a machine with no
GPU, across CPU devices).  This **increases the available memory**, so you can reconstruct
larger volumes than fit on one GPU, and it **typically reduces reconstruction time**.  It
works for every geometry (parallel-beam, cone-beam, translation, and multi-axis parallel)
and for the QGGMRF denoiser (:doc:`usr_denoising`).

The zero-effort path
--------------------

On a machine with more than one GPU, MBIRJAX divides the reconstruction across them
**automatically** -- your script does not change.  The reconstruction log (and
``model.device_summary``) report what was chosen, for example ``4 x GPU (sharded)``.  When
only one GPU is available the reconstruction simply runs on that single device.  With no GPU
at all, MBIRJAX runs on the CPU instead -- across several CPU devices when a cluster exposes
them, and in every case using the machine's cores -- so CPU reconstructions still benefit from
parallelism.

Using a subset of the devices
-----------------------------

You need this only if you want MBIRJAX to use a subset of the available GPUs -- for example, to
leave one or more GPUs free for a deep neural network or another job on the same node.  By
default MBIRJAX already uses every GPU it finds (the zero-effort path above), so most users can
skip this section.

JAX assigns an index to each device it can see.  ``jax.devices()`` returns that list for the
default backend (the devices that do the work) -- the GPUs when any are present, otherwise the
CPU(s); ``jax.devices('gpu')`` and ``jax.devices('cpu')`` return a specific backend.  Use this
list to decide which devices to hand to :meth:`~mbirjax.TomographyModel.configure_devices`::

    import jax
    jax.devices()                     # a list of entries such as CudaDevice(id=0), indexed 0, 1, 2, ...

    model.configure_devices(2)        # use the first 2 devices
    model.configure_devices((0, 2))   # use devices at these indices into jax.devices()
    model.configure_devices(jax.devices('gpu')[:2])   # use exactly these device objects
    model.configure_devices(1)        # force a single device

    model.configure_devices()         # reset: go back to using all available devices

A configuration set this way is *pinned* -- MBIRJAX keeps it across later parameter changes
rather than reverting to the automatic all-device choice.

To run on the CPU even when one or more GPUs are present, set ``use_gpu='none'`` (see
:ref:`param-use_gpu`).

Tips for efficiency
-------------------

* **Reuse a prepared sinogram.**  If you reconstruct the same large sinogram several times,
  :meth:`~mbirjax.TomographyModel.prepare_sino_for_devices` distributes it across the devices once, so
  the host-to-device transfer is not repeated on every reconstruction.
* **Keep results on-device.**  Passing ``output_sharded=True`` to ``recon``, ``fbp_recon``, or
  ``fdk_recon`` returns the reconstruction already distributed across the devices (no gather
  back to the host), so it can feed another on-device step directly.
* **More devices is not always faster.**  For a small problem, spreading the work across more
  devices can actually *increase* reconstruction time: the per-device communication and
  coordination overhead can outweigh the reduced compute per device.  Reach for more devices
  when you need the extra memory or when the problem is large.

What to expect from more GPUs
-----------------------------

* **Memory first.**  Peak memory per GPU scales roughly as 1/N with N GPUs, so the primary
  benefit is the ability to reconstruct larger volumes.
* **Speed second.**  Reconstruction time scales nearly linearly with N at large problem
  sizes.  For a fixed (especially small) problem, adding GPUs eventually stops helping once
  the communication and coordination overhead no longer amortizes.  These are
  hardware-dependent rules of thumb, not guarantees.

Behind the scenes
-----------------

Internally MBIRJAX uses *sharding*: the sinogram is split across the devices by view and the
reconstruction volume by slice, and the two are combined with a small amount of banded
communication between devices.  When a count does not divide evenly, the data is zero-padded
to equal shares and the padding is kept exactly inert, so the result is independent of the
number of devices.  A reconstruction runs within a single process; multi-node execution is
out of scope.
