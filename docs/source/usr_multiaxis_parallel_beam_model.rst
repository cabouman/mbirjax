.. _MultiAxisParallelBeamModelDocs:

=========================
Multi-Axis Parallel Model
=========================

The ``MultiAxisParallelModel`` class implements a geometry and reconstruction model for parallel beam but with the ability to rotate in azimuth and tilt in elevation.
Therefore, parallel beam laminography is a special case of this geometry when there is a constant tilt for all views.

This class inherits all behaviors and attributes of the :ref:`TomographyModelDocs`.
It also implements multi-axis parallel-beam direct reconstruction methods such as ``direct_recon`` and ``fbp_recon``.

For multi-axis parallel beam geometry, the default detector channel spacing is ``delta_det_channel`` is 1 ALU,
and the voxels are 3D cubes with spacing ``delta_voxel = delta_det_channel``.
However, these parameters can be changed by the user using the ``TomographyModel.set_params()`` method.

See the API docs for the :class:`~mbirjax.TomographyModel` class for details on a wide range
of functions that can be implemented using the ``MultiAxisParallelModel``.

Constructor
-----------

.. autoclass:: mbirjax.MultiAxisParallelModel
   :show-inheritance:

Alternative Reconstruction
--------------------------

.. automethod:: mbirjax.MultiAxisParallelModel.fbp_recon
