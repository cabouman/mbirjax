import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys

def display_results(filename):

    # Load the existing data
    data = np.load(filename, allow_pickle=True)
    mem_values = data['mem_values']
    time_values = data['time_values']
    eval_type_index = data['eval_type_index']
    pixel_batch_size = data['pixel_batch_size']
    max_percent_used_gb = data['max_percent_used_gb']
    max_avail_gb = data['max_avail_gb']
    num_views = data['num_views']
    num_channels = data['num_channels']
    num_det_rows = data['num_det_rows']
    num_indices = data['num_indices']

    # Set up the info for plotting
    outer_labels = ['# views', '# det channels']
    outer_values = [num_views, num_channels]
    inner_labels = ['# det rows', '# indices']
    inner_values = [num_det_rows, num_indices]

    vmin_mem = -2
    vmax_mem = 6
    vmin_time = -12
    vmax_time = 5

    plt.ion()

    eval_types = ['Forward_projection', 'Backward_projection']
    super_title_mem = 'Peak log2(GB) as a function of #views, #channels, #rows, and #indices'
    super_title_mem += ', pixel_batch_size={}\n'.format(pixel_batch_size)
    super_title_mem += eval_types[eval_type_index]
    super_title_mem += ':  Avail GB = {:.1f}, Max percent mem used = {:.2f}'.format(max_avail_gb, max_percent_used_gb)
    super_title_time = 'log2(seconds elapsed) as a function of #views, #channels, #rows, and #indices\n'
    super_title_time += eval_types[eval_type_index]

    create_tiled_heatmap(np.log2(mem_values), outer_labels, outer_values, inner_labels, inner_values,
                         super_title=super_title_mem, vmin=vmin_mem, vmax=vmax_mem)
    create_tiled_heatmap(np.log2(time_values), outer_labels, outer_values, inner_labels, inner_values,
                         super_title=super_title_time, vmin=vmin_time, vmax=vmax_time)


def create_tiled_heatmap(plot_values, outer_labels, outer_values, inner_labels, inner_values,
                         super_title=None, vmin=None, vmax=None):
    """
    Creates a nested grid to display the 4D data in plot_values, assuming a shape on the order of 3x3x3x3, where
    each axis of plot_values is assumed to correspond to an independent variable as described in outer_labels and
    inner_labels.  The first 2 axes of plot values determine the location within an outer grid, and the next 2 axes
    determine the location within each inner grid.  For each [i,j], the values in plot_values[i, j] are plotted using
    a colormap along with the text of the value in each grid square.

    plot_values is assumed to be obtained by evaluating
    a function f at points determined by inner_values and outer_values as follows:
    plot_values[i, j, k, l] = f[outer_values[0][i], outer_values[1][j], inner_values[0][k], inner_values[1][l]]

    Args:
        plot_values (4D numpy array):  Values to display
        outer_labels (length 2 string): The labels describing axes 0 and 1 of plot_values (e.g., #views, #channels)
        outer_values (length 2 list):  outer_values[0] (or [1]) contains the values of the independent variable for axis 0 (or 1)
        inner_labels (length 2 string): The labels describing axes 2 and 3 of plot_values (e.g., #slices, #indices)
        inner_values (length 2 list): inner_values[0] (or [1]) contains the values of the independent variable for axis 2 (or 3)
        sup_title (string, optional): Overall title for the display
        vmin (float, optional): lower clipping value for colorbar display
        vmax (float, optional): upper clipping value for colorbar display

    Returns:
        Nothing

    # Example:
    plot_values = 15*np.random.rand(3, 3, 3, 3)
    outer_labels = ['# views', '# det channels']
    outer_values = [[128, 256, 512], [128, 256, 512]]
    inner_labels = ['# det rows', '# indices']
    inner_values = [[128, 256, 512], [129, 256, 512]]
    super_title = 'Peak log2(GB) as a function of views, channels, rows, and #indices'
    vmin = 0.0
    vmax = 10.0
    create_tiled_heatmap(np.log2(plot_values), outer_labels, outer_values, inner_labels, inner_values,
                         super_title=super_title, vmin=vmin, vmax=vmax)
    """
    if vmin is None:
        vmin = np.amin(plot_values)
    if vmax is None:
        vmax = np.amax(plot_values)
    # Create a grid for the subplots with some space between them
    nrows = plot_values.shape[0] + 1
    ncols = plot_values.shape[1] + 1
    width_ratios = [1, ] + (ncols - 1) * [3, ]
    height_ratios = (nrows - 1) * [3, ] + [1, ]

    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols, width_ratios=width_ratios, height_ratios=height_ratios,
                           hspace=0.0, wspace=0.5)

    # Convert all displayed items to strings
    inner_labels = [str(label) for label in inner_labels]
    row_names = [str(row_name) for row_name in inner_values[0]]
    col_names = [str(col_name) for col_name in inner_values[1]]
    inner_values = [row_names, col_names]

    outer_labels = [str(sup_label) for sup_label in outer_labels]
    sup_row_names = [str(row_name) for row_name in outer_values[0]]
    sup_col_names = [str(col_name) for col_name in outer_values[1]]
    outer_values= [sup_row_names, sup_col_names]

    # Set up the overall figure
    fig, axs = plt.subplots(figsize=(5*(ncols-1), 5*(nrows-1)))
    cax = None
    fig.get_axes()[0].axis('off')
    # Create a subplot for each 2D array in 'plot_values'
    for i in range(nrows):
        for j in range(ncols):
            ax = fig.add_subplot(gs[i, j])

            # Display the row outer labels
            if j == 0:
                ax.axis('off')
                if i < nrows-1:
                    ax.text(0.5, 0.5, outer_labels[0] + '\n' + outer_values[0][i], horizontalalignment='center', verticalalignment='center',
                            transform=ax.transAxes)
            # Display the column outer labels
            elif i == nrows-1:
                ax.axis('off')
                ax.text(0.5, 0.5, outer_labels[1] + '\n' + outer_values[1][j-1], horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes)
            else:
                cur_values = plot_values[i, j-1]  # Offset by one since the leftmost subplot was used for a label
                # Create the heatmap
                cax = ax.matshow(cur_values, cmap='viridis', vmin=vmin, vmax=vmax)

                # Set the inner tick labels for rows and columns
                ax.set_yticks(np.arange(len(inner_values[0])), inner_values[0])
                ax.set_xticks(np.arange(len(inner_values[1])), inner_values[1])
                ax.xaxis.set_ticks_position('bottom')

                # Set the inner axes labels
                ax.set_ylabel(inner_labels[0])
                ax.set_xlabel(inner_labels[1])

                # Display the plot_values in the squares
                for m in range(cur_values.shape[0]):
                    for n in range(cur_values.shape[1]):
                        ax.text(n, m, f'{cur_values[m, n]:.1f}', ha='center', va='center', color='w')

    # Add a colorbar on the right and the super title
    fig.colorbar(cax, ax=fig.get_axes())
    if super_title is not None:
        plt.suptitle(super_title)

    plt.show()


if __name__ == '__main__':
    # filename, nv, nc, nr
    filename = sys.argv[1]

    display_results(filename)
    plt.pause(5)
    input('Press return to exit')
