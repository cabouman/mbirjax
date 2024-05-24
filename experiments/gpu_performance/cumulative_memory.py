import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import jax.numpy as jnp
import time
import mbirjax


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
    super_title = 'Peak log(GB) as a function of views, channels, rows, and #indices'
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


if __name__ == "__main__":
    """
    This is a script to evaluate the time and memory use of the parallel beam projector
    """
    eval_types = ['Forward_projection', 'Backward_projection_0', 'Backward_projection_1', 'Backward_projection_2']
    eval_type_index = 0
    pixel_batch_size = None

    # Set up the independent variable values
    num_views = [1024, 2048]  # [256, 512, 768]
    num_channels = [512, 1024]  # [256, 512, 768]
    num_det_rows = [256]
    num_indices = [256*256]  #[256*256, 512*512, 768*768]

    # Set up the info for plotting
    outer_labels = ['# views', '# det channels']
    outer_values = [num_views, num_channels]
    inner_labels = ['# det rows', '# indices']
    inner_values = [num_det_rows, num_indices]

    vmin = -3.5
    vmax = 1.5

    if mbirjax.get_memory_stats() is None:
        raise EnvironmentError('This script is for gpu only.')

    m1 = mbirjax.get_memory_stats()
    max_avail_gb = m1[0]['bytes_limit'] / (1024 ** 3)

    # Make room for the data
    mem_values = np.zeros((len(num_views), len(num_channels), len(num_det_rows), len(num_indices)))
    time_values = np.zeros_like(mem_values)

    # Set up for projections
    start_angle = 0.0
    end_angle = jnp.pi
    sinogram = None
    bp = None

    for i, nv in enumerate(num_views):
        for j, nc in enumerate(num_channels):
            for k, nr in enumerate(num_det_rows):
                for l, ni in enumerate(num_indices):
                    angles = jnp.linspace(start_angle, jnp.pi, nv, endpoint=False)

                    # Set up parallel beam model
                    sinogram_shape = (nv, nr, nc)
                    parallel_model = mbirjax.ParallelBeamModel(sinogram_shape, angles)
                    parallel_model.set_params(pixel_batch_size=pixel_batch_size)

                    # Generate phantom for forward projection
                    recon_shape = parallel_model.get_params('recon_shape')
                    phantom = mbirjax.gen_cube_phantom(recon_shape)

                    # Get a subset of the given size
                    indices = np.arange(ni, dtype=int)
                    voxel_values = phantom.reshape((-1,) + recon_shape[2:])[indices]

                    if eval_type_index == 0:

                        print('Initial forward projection for memory: nv={}, nc={}, nr={}, ni={}'.format(nv, nc, nr, ni))
                        try:
                            sinogram = parallel_model.forward_project(voxel_values, indices)
                            m1 = mbirjax.get_memory_stats()
                            peak_mem_gb = m1[0]['peak_bytes_in_use'] / (1024 ** 3)
                        except:
                            print('Out of memory')
                            peak_mem_gb = 1000 * max_avail_gb
                        mem_values[i, j, k, l] = peak_mem_gb
                        print('Peak GB used = {}'.format(peak_mem_gb))

                        print('Forward projection for speed')
                        t0 = time.time()
                        try:
                            sinogram = parallel_model.forward_project(voxel_values, indices)
                        except:
                            print('Out of memory on pass 2')
                        t1 = time.time()
                        time_diff_secs = t1 - t0
                        time_values[i, j, k, l] = time_diff_secs
                        print('Elapsed time = {}'.format(time_diff_secs))

                    else:
                        print('Initial back projection for memory: nv={}, nc={}, nr={}, ni={}'.format(nv, nc, nr, ni))
                        try:
                            sinogram = parallel_model.forward_project(voxel_values, indices)
                        except:
                            print('Out of memory in forward projector')
                            sinogram = np.ones((nv, nr, nc))
                        try:
                            bp = parallel_model.back_project(sinogram, indices)
                            m1 = mbirjax.get_memory_stats()
                            peak_mem_gb = m1[0]['peak_bytes_in_use'] / (1024 ** 3)
                        except:
                            print('Out of memory')
                            peak_mem_gb = 1000 * max_avail_gb

                        mem_values[i, j, k, l] = peak_mem_gb
                        print('Peak GB used = {}'.format(peak_mem_gb))

                        print('Back projection for speed')
                        t0 = time.time()
                        try:
                            bp = parallel_model.sparse_back_project(sinogram, indices)
                        except:
                            print('Out of memory on pass 2')
                        t1 = time.time()
                        time_diff_secs = t1 - t0
                        time_values[i, j, k, l] = time_diff_secs
                        print('Elapsed time = {}'.format(time_diff_secs))


    m1 = mbirjax.get_memory_stats()
    peak_mem_gb = m1[0]['peak_bytes_in_use'] / (1024 ** 3)
    max_percent_used_gb = 100 * peak_mem_gb / max_avail_gb
    print('Max percentage GB used = {}%'.format(max_percent_used_gb))

    mem_title = 'GB_secs used for ' + eval_types[eval_type_index]
    np.savez(mem_title, mem_values=mem_values, time_values=time_values, eval_type_index=np.array(eval_type_index),
             max_percent_used_gb=np.array(max_percent_used_gb),
             num_views=np.array(num_views), num_channels=np.array(num_channels),
             num_det_rows=np.array(num_det_rows), num_indices=np.array(num_indices))
    plt.ion()

    super_title_mem = 'Peak log2(GB) as a function of #views, #channels, #rows, and #indices\n'
    super_title_mem += eval_types[eval_type_index]
    super_title_time = 'log2(seconds elapsed) as a function of #views, #channels, #rows, and #indices\n'
    super_title_time += eval_types[eval_type_index]

    create_tiled_heatmap(np.log2(mem_values), outer_labels, outer_values, inner_labels, inner_values,
                         super_title=super_title_mem, vmin=vmin, vmax=vmax)
    create_tiled_heatmap(np.log2(time_values), outer_labels, outer_values, inner_labels, inner_values,
                         super_title=super_title_time, vmin=vmin, vmax=vmax)

    plt.figure()
    if eval_type_index == 0:
        plt.imshow(sinogram[:, 0, :])
    else:
        back_projection = np.zeros((nc * nc, nr))
        back_projection[indices] = bp
        plt.imshow(back_projection[:, 0].reshape((nc, nc)))

    plt.colorbar()
    plt.show()
    plt.pause(1)
    input('Hit return to exit')
