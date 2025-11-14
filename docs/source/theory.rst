======
Theory
======


Model-Based Iterative Reconstruction
------------------------------------


The following describes how Model-Based Iterative Reconstruction (MBIR) works, and the low-level parameters used to control it.
However, while these low level MBIR parameters can be accessed, we strongly recommend that you control image quality using the meta-parameter,
``sharpnesss``.
The default value of ``sharpness`` is 0. Larger values will increase sharpness, and small values will decrease it.

MBIR reconstruction works by solving the following optimization problem

.. math::

    {\hat x} = \arg \min_x \left\{ f(x) + h(x) \right\}

where :math:`f(x)` is the forward model term and :math:`h(x)` is the prior model term.
The Multi-Granular Vectorized Coordinate Descent (VCD) algorithm is then used to efficiently perform this optimization.


**Forward Model:**

*Note:* More details
about the forward model for specific geometries is available by downloading the `associated
zip file <https://www.datadepot.rcac.purdue.edu/bouman/data/tomography_geometry.zip>`_.

The forward model term has the form,

.. math::

    f(x) = \frac{1}{2 \sigma_y^2} \Vert y - Ax \Vert_\Lambda^2

where :math:`y` is the sinogram data,
where :math:`x` is the unknown image to be reconstructed,
:math:`A` is the linear projection operator for the specified imaging geometry,
:math:`\Lambda` is the diagonal matrix of sinogram weights, :math:`\Vert y \Vert_\Lambda^2 = y^T \Lambda y`, and
:math:`\sigma_y` is a parameter controlling the assumed standard deviation of the measurement noise.

These quantities correspond to the following python variables:

* :math:`y` corresponds to ``sino``
* :math:`\sigma_y` corresponds to ``sigma_y``
* :math:`\Lambda` corresponds to ``weights``

The weights can either be set automatically using the ``weight_type`` input, or they can be explicitly set to an array of precomputed weights.
For many new users, it is easier to use one of the automatic weight settings shown below.

* weight_type="unweighted": :math:`\Lambda = 1 + 0*y`  (array of ones of same size as sinogram)
* weight_type="transmission": :math:`\Lambda = e^{-y}`
* weight_type="transmission_root": :math:`\Lambda = e^{-y/2}`
* weight_type="emission": :math:`\Lambda = 1/(y + 0.1)`

Option "unweighted" provides unweighted reconstruction; Option "transmission" is the correct weighting for transmission CT with constant dosage; Option "transmission_root" is commonly used with transmission CT data to improve image homogeneity; Option "emission" is appropriate for emission CT data.

**Prior Model:**
MBIRJAX allows the prior model to be set either as a qGGMRF or a proximal map prior.
The qGGRMF prior is the default method recommended for new users.
Alternatively, the proximal map prior is an advanced feature required for the implementation of the Plug-and-Play algorithm. The Plug-and-Play algorithm allows the modular use of a wide variety of advanced prior models including priors implemented with machine learning methods such as deep neural networks.

The qGGMRF prior model has the form

.. math::

    h(x) = \sum_{ \{s,r\} \in {\cal P}} b_{s,r} \rho ( x_s - x_r) \ ,

where

.. math::

    \rho ( \Delta ) = \frac{|\Delta |^p }{ p \sigma_x^p } \left( \frac{\left| \frac{\Delta }{ T \sigma_x } \right|^{q-p}}{1 + \left| \frac{\Delta }{ T \sigma_x } \right|^{q-p}} \right)

where :math:`{\cal P}` represents a 8-point 2D neighborhood of pixel pairs in the :math:`(x,y)` plane and a 2-point neighborhood along the slice axis;
:math:`\sigma_x` is the primary regularization parameter;
:math:`b_{s,r}` controls the neighborhood weighting;
:math:`p<q=2.0` are shape parameters;
and :math:`T` is a threshold parameter.

These quantities correspond to the following python variables:

* :math:`\sigma_x` corresponds to ``sigma_x``
* :math:`p` corresponds to ``p``
* :math:`q` corresponds to ``q``
* :math:`T` corresponds to ``T``


**Proximal Map Prior:**
The proximal map prior is provided as a option for advanced users who would like to use plug-and-play methods.
If ``prox_image`` is supplied, then the proximal map prior model is used, and the qGGMRF parameters are ignored.
In this case, the reconstruction solves the optimization problem:

.. math::

    {\hat x} = \arg \min_x \left\{ f(x) + \frac{1}{2\sigma_p^2} \Vert x -v \Vert^2 \right\}

where the quantities correspond to the following python variables:

* :math:`v` corresponds to ``prox_image``
* :math:`\sigma_p` corresponds to ``sigma_prox``


Vectorized Coordinate Descent (VCD)
-----------------------------------

At its core, MBIRJAX is based on multi-granular VCD (MG-VCD) optimization as described in :cite:`2024CV4SciencePoster`.
For the user, this is all "under the hood", but it is critically important because it results in fast robust convergence that can be efficiently implemented in JAX and on modern GPU architectures.
Moreover, MG-VCD does not require the selection of a geometry specific pre-conditioner, as would be typically required with gradient-based methods, so it enables MBIRJAX's unique support for multiple geometries.

