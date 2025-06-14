.. _QGGMRFDenoiserDocs:

===============
QGGMRF Denoiser
===============

The ``QGGMRFDenoiser`` class implements a 3D volume denoiser using a Bayesian formulation.
The loss function in this formulation is the QGGMRF function, which promotes
nearby voxels to have similar values while preserving edges.  Using :math:`Q(v)` to denote the loss function, the
denoiser is

        .. math::

            F(x) = \arg\min_v \left\{ Q(v) + \frac{1}{2}\|v - x\|^{2} \right\}.

This denoiser automatically estimates the noise level in the image.  The amount of denoising can be adjusted using
parameters sharpness (default=1.0) and/or snr_db (default=30).
This class inherits the behaviors and attributes of the :ref:`TomographyModelDocs`:  the proximal map above is
implemented through forward and backprojections that are the identity map.

Constructor
-----------

.. autoclass:: mbirjax.QGGMRFDenoiser
   :show-inheritance:

Denoise
-------

.. automethod:: mbirjax.QGGMRFDenoiser.denoise



