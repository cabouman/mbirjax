.. mbirjax documentation master file, created by
   sphinx-quickstart on Fri Jun 25 14:24:26 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


**MBIRJAX** is a Python package for Model Based Iterative Reconstruction (MBIR) of images from tomographic data.
MBIRJAX uses an object oriented Python interface, so it is modular and easy to use.
It also support automatic selection of MBIR parameters, so users can control image sharpness through the use of intuitive meta-parameters. In other words, you first reconstruction is likely to look good with little tuning. Finally, it has hooks to support Plug-and-Play prior models that can dramatically improve image quality [] [].
MBIRJAX is fast, portable, and will run on CPUs or GPUs because it is based on the JAX programming language.
At its heart it uses an algorithm called Vectorized Coordinate Descent to solve the associated optimization problem in a fast and robust way. All computations are on-the-fly, so there is no need to manage the storage of system matrices.




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. toctree::
   :hidden:
   :maxdepth: 4
   :caption: Background

   overview
   examples
   theory
   credits

.. toctree::
   :hidden:
   :maxdepth: 4
   :caption: User Guide

   install
   api

.. toctree::
   :hidden:
   :maxdepth: 4
   :caption: Developer Guide

   pytest
