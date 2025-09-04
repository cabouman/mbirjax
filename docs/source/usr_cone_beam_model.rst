.. _ConeBeamModelDocs:

===============
Cone Beam Model
===============

The ``ConeBeamModel`` class implements a geometry and reconstruction model for cone beam computed tomography.
This class inherits all behaviors and attributes of the :ref:`TomographyModelDocs`.
It also implements some cone-beam specific functions such as FDK (Feldkamp-Davis-Kress) reconstruction.

For cone-beam geometry, the default detector channel spacing is ``delta_det_channel`` is 1 ALU,
and the voxels are 3D cubes with spacing ``delta_voxel``.

The default voxel spacing is set to ``delta_voxel = delta_det_channel / magnification`` where ``magnification = source_detector_dist / source_iso_dist``.
This implies that as the magnification increases, the default voxel spacing decreases.
However, these parameters can be changed by the user using the ``TomographyModel.set_params()`` method.

See the API docs for the :class:`~mbirjax.TomographyModel` class for details on a wide range
of functions that can be implemented using the ``ConeBeamModel``.

Constructor
-----------

.. autoclass:: mbirjax.ConeBeamModel
   :show-inheritance:

FDK Reconstruction
------------------

.. automethod:: mbirjax.ConeBeamModel.fdk_recon
.. automethod:: mbirjax.ConeBeamModel.recon_split_sino
