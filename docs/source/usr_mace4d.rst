.. _MACE4DDocs:

=================
4D Reconstruction
=================

MBIRJAX reconstructs a time sequence of volumes from a single continuous scan of a moving
object, using multi-agent consensus equilibrium.

+++++++++++
MACE4DModel
+++++++++++

The scan is divided into overlapping angular windows, one per time frame.
``frames_per_rotation`` sets how many frames make up a full rotation, and
``frame_overlap_factor`` sets how many frames share any given view, so each frame spans
``frame_overlap_factor * (360 / frames_per_rotation)`` degrees.  Wider frames give each one
more views and better SNR at the cost of temporal resolution.

The reconstruction is performed using the MACE algorithm of :cite:`mace4d`.
Each MACE iteration runs one :meth:`~mbirjax.TomographyModel.prox_map` per time frame and
three qGGMRF denoisers.  The prox maps fit the measured data.  The denoisers regularize
the XY-t, YZ-t and XZ-t hyperplanes of the 4D volume.  The MACE update combines these
outputs into a single consensus reconstruction.

The MACE algorithm also incorporates a dejittering algorithm that removes oscillations
produced when consecutive time frames are reconstructed from different angular windows.
The window pattern repeats once per rotation, so the oscillation has a period of
``frames_per_rotation`` frames.  A DCT filter removes this component of the temporal
spectrum within the MACE loop.  The ``dejitter`` parameter controls the filter.

The computations within a MACE iteration are independent, so they run as separate tasks
distributed across the available devices.  :meth:`~mbirjax.MACE4DModel.set_device_pool`
selects the devices.

Constructor
-----------

.. autoclass:: mbirjax.MACE4DModel
   :show-inheritance:

Parameters
----------

.. automethod:: mbirjax.MACE4DModel.set_params

Reconstruction
--------------

.. automethod:: mbirjax.MACE4DModel.recon

Device Pool
-----------

.. automethod:: mbirjax.MACE4DModel.set_device_pool
