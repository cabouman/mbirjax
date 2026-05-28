import os
import mbirjax as mj
import numpy as np
from bh_curve_fit_utils import plot_bh_correction_curve


if __name__ == '__main__':

    # Path or URL to projection data in npy format with tgz wrapper.
    # The tgz file should contain these three files:
    #   training_linear_projection.npy
    #   training_beam_hardened_projection.npy
    #   test_linear_projection.npy
    #dataset_url = 'https://www.datadepot.rcac.purdue.edu/bouman/data/ORNL/fit_bh_curve_demo_data.tgz'
    dataset_url = '/depot/bouman/data/ORNL/fit_bh_curve_demo_data.tgz'

    # Path to directory for storage of data.
    download_dir = './demo_data/'

    #####################
    # Set BH parameters
    #####################
    N_bh_params = 5

    ###############
    # Download Data
    ###############
    # Download and extract data.
    dataset_dir = mj.download_and_extract(dataset_url, download_dir)

    # Load training pair into workspace.
    print('Loading training linear projection')
    training_linear_projection = np.load(os.path.join(dataset_dir, 'training_linear_projection.npy'))
    print('Shape of training linear projection: {}'.format(training_linear_projection.shape))

    print('Loading training beam-hardened projection')
    training_beam_hardened_projection = np.load(
        os.path.join(dataset_dir, 'training_beam_hardened_projection.npy'))
    print('Shape of training beam-hardened projection: {}'.format(training_beam_hardened_projection.shape))

    # Load test projection for inference.
    print('Loading test linear projection')
    test_linear_projection = np.load(os.path.join(dataset_dir, 'test_linear_projection.npy'))
    print('Shape of test linear projection: {}'.format(test_linear_projection.shape))

    #####################
    # Fit BH curve
    #####################
    linear_projection = training_linear_projection.ravel()
    target_projection = training_beam_hardened_projection.ravel()

    best_bh_params = mj.fit_beam_hardening_curve(linear_projection, target_projection, N_bh_params)
    print('Best BH parameters: {}'.format(best_bh_params))

    #####################
    # Apply BH curve
    #####################
    fitted_nonlinear_projection = mj.apply_fitted_beam_hardening_curve(
        test_linear_projection,
        best_bh_params,
    )
    print('Shape of fitted nonlinear projection: {}'.format(fitted_nonlinear_projection.shape))

    #####################
    # Plot BH curve
    #####################
    length = 60
    x_path_length_plot = np.linspace(0, length, 200)
    plot_bh_correction_curve(x_path_length_plot, best_bh_params)