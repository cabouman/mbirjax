.. _MACE4DDocs:

=================
4D Reconstruction
=================

MBIRJAX reconstructs a time sequence of volumes from a single continuous scan of a moving
object, using multi-agent consensus equilibrium :cite:`venkatakrishnan2013plug`
:cite:`sreehari2016plug`.

+++++++++++
MACE4DModel
+++++++++++

The scan is divided into overlapping angular windows, one per time frame.
``frames_per_rotation`` sets how many frames make up a full rotation, and
``frame_overlap_factor`` sets how many frames share any given view, so each frame spans
``frame_overlap_factor * (360 / frames_per_rotation)`` degrees.  Wider frames give each one
more views and better SNR at the cost of temporal resolution.  The number of frames follows
from the scan length rather than being set directly, so check ``mace.nt`` and
``mace.view_slices`` before committing to a long run.

Each iteration runs one :meth:`~mbirjax.TomographyModel.prox_map` per time frame as the
forward agent, together with three batched qGGMRF denoisers acting on the XY-t, YZ-t and XZ-t
hyperplanes as prior agents, and reconciles them by consensus equilibrium.  Gating imprints a
modulation on the time axis whose period is ``frames_per_rotation``; a DCT-I filter removes it
inside every agent, under the ``dejitter`` parameter.

The agents are independent within an iteration, so they are distributed across devices one
task at a time rather than by sharding a single array, which is what
:meth:`~mbirjax.MACE4DModel.set_device_pool` below configures.

.. note::

   ``weights=None`` means unit weights, following :meth:`~mbirjax.TomographyModel.recon`.  For
   transmission data the validated choice is ``transmission_root``, which must be passed
   explicitly::

       weights = mj.gen_weights(sinogram, weight_type='transmission_root')

Constructor
-----------

.. autoclass:: mbirjax.MACE4DModel
   :show-inheritance:

Reconstruction
--------------

.. automethod:: mbirjax.MACE4DModel.recon

Device Pool
-----------

.. automethod:: mbirjax.MACE4DModel.set_device_pool

Time Frames
-----------

The frame decomposition is also available on its own, for building a 4D forward model or
inspecting the split before a reconstruction.

.. autofunction:: mbirjax.utilities.construct_time_frame_models
.. autofunction:: mbirjax.utilities.construct_time_frames
