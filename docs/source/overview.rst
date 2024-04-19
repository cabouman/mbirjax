========
Overview
========

**MBIRJAX** is a Python package for Model Based Iterative Reconstruction (MBIR) of images from tomographic data.

MBIRJAX uses an object oriented Python interface, so it is modular and easy to use.
It also support automatic selection of MBIR parameters, so users can control image sharpness through the use of intuitive meta-parameters.
In other words, you first reconstruction is likely to look good with little tuning.
Finally, it has hooks to support Plug-and-Play prior models that can dramatically improve image quality :cite:`venkatakrishnan2013plug` :cite:`sreehari2016plug`.

MBIRJAX is fast, portable, and will run on CPUs or GPUs because it is based on the JAX programming language.
At its heart it uses an algorithm called Vectorized Coordinate Descent to solve the associated optimization problem in a fast and robust way.
All computations are on-the-fly, so there is no need to manage the storage of system matrices.

**Geometry**

Right now MBIRJAX supports *parallel-beam* imaging geometry shown below, but more geometries are on the way.

.. list-table::

    * - .. figure:: figs/geom-parallel.png
           :align: center
           :width: 50%

           Parallel-beam geometry


**Conversion from Arbitrary Length Units (ALU)**

In order to simplify usage, reconstructions are done using arbitrary length units (ALU).
In this system, 1 ALU can correspond to any convenient measure of distance chosen by the user.
So for example, it is often convenient to take 1 ALU to be the distance between pixels, which by default is also taken to be the distance between detector channels.

*Transmission CT Example:* For this example, assume that the physical spacing between detector channels is 5 mm.
In order to simplify our calculations, we also use the default detector channel spacing and voxel spacing of ``delta_channel=1.0`` and ``delta_xy=1.0``.
In other words, we have adopted the convention that the voxel spacing is 1 ALU = 5 mm, where 1 ALU is now our newly adopted measure of distance.

Using this convention, the 3D reconstruction array, ``image``, will be in units of :math:`\mbox{ALU}^{-1}`.
However, the image can be converted back to more conventional units of :math:`\mbox{mm}^{-1}` using the following equation:

.. math::

    \mbox{image in mm$^{-1}$} = \frac{ \mbox{image in ALU$^{-1}$} }{ 5 \mbox{mm} / \mbox{ALU}}


*Emission CT Example:* Once again, we assume that the channel spacing in the detector is 5 mm, and we again adopt the default reconstruction parameters of ``delta_channel=1.0`` and ``delta_xy=1.0``. So we have that 1 ALU = 5 mm. 

Using this convention, the 3D array, ``image``, will be in units of photons/AU. However, the image can be again converted to units of photons/mm using the following equation:

.. math::

    \mbox{image in photons/mm} = \frac{ \mbox{image in photons/ALU} }{ 5 \mbox{mm} / \mbox{ALU}}

