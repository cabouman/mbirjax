.. mbirjax documentation master file, created by
   sphinx-quickstart on Fri Jun 25 14:24:26 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MBIRJAX: High-performance tomographic reconstruction
====================================================

**MBIRJAX** is a Python package for Model Based Iterative Reconstruction (MBIR) of images from tomographic data.

**Key features:**


* Vectorized Coordinate Descent algorithm for fast, robust convergence :cite:`2024CV4SciencePoster`.
* Automatic parameter selection, with fine-tuning using intuitive meta-parameters.
* Support for Plug-and-Play prior models that can dramatically improve image quality :cite:`venkatakrishnan2013plug` :cite:`sreehari2016plug`.
* Modular, extensible, easy-to-use, object-oriented Python interface.
* Fast, portable, seamless use on CPUs or GPUs through the use of JAX_.



.. grid:: 3
   :margin: 0
   :padding: 0
   :gutter: 0

   .. grid-item-card:: Simple API
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      Reconstructions can be performed in just a few lines of python code.

   .. grid-item-card:: Fast, robust convergence
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      Vectorized Coordinate Descent produces high-quality images quickly.

   .. grid-item-card:: Portability
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      Based on JAX_, MBIRJAX can easily run on CPU or GPU.


.. grid:: 3

    .. grid-item-card:: :material-regular:`rocket_launch;2em` Getting Started
      :class-card: getting-started
      :columns: 12 6 6 4
      :link: overview
      :link-type: doc

    .. grid-item-card:: :material-regular:`library_books;2em` User Guides
      :class-card: user-guides
      :columns: 12 6 6 4
      :link: install
      :link-type: doc

    .. grid-item-card:: :material-regular:`laptop_chromebook;2em` Developer Docs
      :class-card: developer-docs
      :columns: 12 6 6 4
      :link: dev_api
      :link-type: doc


.. toctree::
   :hidden:
   :maxdepth: 4
   :caption: Background

   overview
   quick_start
   advanced_features
   theory
   credits

.. toctree::
   :hidden:
   :maxdepth: 4
   :caption: User Guide

   install
   usr_api

.. toctree::
   :hidden:
   :maxdepth: 4
   :caption: Developer Guide

   dev_api
   dev_maintenance


.. _JAX: https://jax.readthedocs.io/