.. _DenoisingDocs:

=========
Denoising
=========

MBIRJAX includes a Bayesian MAP denoising using the qGGMRF prior and a 3D median filter.

++++++++++++++
QGGMRFDenoiser
++++++++++++++

The ``QGGMRFDenoiser`` class implements a 3D volume denoiser for additive white Gaussian denoising.
More specifically, it computes the MAP assuming additive white Gaussian noise and a qGGMRF prior distribution.
Using :math:`H(x)` to denote the denoising function, the denoiser is

        .. math::

            H(x) = \arg\min_v \left\{ \frac{1}{2 \sigma_{noise}^2}\|x - v\|^{2} + h(v) \right\}.

The denoiser will automatically estimate the noise level in the image, or the can directly set the value of
noise standard deviation through the parameter `sigma_noise`.  Larger values of `sigma_noise` lead to smoother images.
Alternatively, the amount of denoising can be adjusted using parameter `sharpness` (default=0.0).
This class inherits many of the behaviors and attributes of the :ref:`TomographyModelDocs`.

Constructor
-----------

.. autoclass:: mbirjax.QGGMRFDenoiser
   :show-inheritance:

Denoise
-------

.. automethod:: mbirjax.QGGMRFDenoiser.denoise


+++++++++++++
Median Filter
+++++++++++++

MBIRJAX also includes a 3x3x3 median filter, which can be used as a simple denoiser.

.. autofunction:: mbirjax.median_filter3d


