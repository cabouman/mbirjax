import numpy as np
import os
import matplotlib.pyplot as plt
import mbirjax

param_dict = {    # Set parameters
    'num_views' : 128,
    'num_det_rows' : 40,
    'num_det_channels' : 256,
    'start_angle' : 0,
    'end_angle' : np.pi,
    'granularity': [1, 8, 48, 128, 256, 256, 256, 256, 256, ],
    'partition_sequence' : [0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4,],
    'max_iterations' : 10
}


def display_images_for_abstract(image1, image2, image3, labels, fig_title=None, vmin=None, vmax=None,
                                cmap='gray', show_colorbar=False):
    # Set global font size
    plt.rcParams.update({'font.size': 15})  # Adjust font size here

    if vmin is None:
        vmin = 0.0
    if vmax is None:
        vmax = 1.0

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
    fig.suptitle(fig_title)

    a0 = ax[0].imshow(image1, vmin=vmin, vmax=vmax, cmap=cmap)
    #plt.colorbar(a0, ax=ax[0])
    ax[0].set_title(labels[0])

    a0 = ax[1].imshow(image2, vmin=vmin, vmax=vmax, cmap=cmap)
    #plt.colorbar(a0, ax=ax[1])
    ax[1].set_title(labels[1])

    a2 = ax[2].imshow(image3, vmin=vmin, vmax=vmax, cmap=cmap)
    #plt.colorbar(a2, ax=ax[2])
    ax[2].set_title(labels[2])

    if show_colorbar:
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes((0.81, 0.16, 0.02, 0.68))
        fig.colorbar(a2, cax=cbar_ax)

    plt.show()
    figure_folder_name = mbirjax.make_figure_folder()
    os.makedirs(figure_folder_name, exist_ok=True)
    fig.savefig(os.path.join(figure_folder_name, fig_title + '.png'), bbox_inches='tight')

