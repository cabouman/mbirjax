import numpy as np
import matplotlib.pyplot as plt


def imshow_with_options(array, title='', vmin=None, vmax=None, cmap='viridis', show=False):
    """ Display an array as an image along with optional title and intensity window.
    Args:
        array: The array to display
        title: The title for the plot
        vmin: Minimum of the intensity window - same as vmin in imshow
        vmax: Maximum of the intensity window - same as vmax in imshow
        cmap: The color map as in imshow - same as cmap in imshow
        show: If true, then plt.show() is called to display immediately, otherwise call
              fig.show() on the object returned from this function to show the plot.

    Returns:
        The pyplot figure object
    """
    fig = plt.figure(layout="constrained")
    plt.imshow(array, vmin=vmin, vmax=vmax, cmap=cmap, interpolation='none')
    plt.colorbar()
    plt.title(title)
    if show:
        plt.show()
    return fig

