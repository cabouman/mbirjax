==================
Plotting utilities
==================

MBIRJAX includes some utilities to view 3D data slice-by-slice and to plot the relationship between granularity
and loss function.

Here is an example showing views of a modified Shepp-Logan phantom, with changing intensity window and displayed slice.

.. image:: https://www.math.purdue.edu/~buzzard/images/slice_viewer_demo.gif
   :alt: An animated image of the slice viewer.

.. automodule:: mbirjax.plot_utils
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

   .. rubric:: **Functions:**

   .. autosummary::
      plot_utils.slice_viewer
      plot_utils.plot_granularity_and_loss
