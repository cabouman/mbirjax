import types
import numpy as np
import yaml
import warnings

import jax
import jax.numpy as jnp
from jax import lax

import mbirjax
import mbirjax._utils as utils


class TomographyModel:
    """
    Represents a general model for tomographic reconstruction using MBIRJAX.

    Forward Model Parameters:
        delta_det_channel (float): Detector channel spacing in ALU.
        det_channel_offset (float): Distance in ALU from center of detector to the source-detector line along a row.
        sigma_y (float): Sinogram additive noise standard deviation.

    Reconstruction Model Parameters:
        init_recon (jax array): 3D volume being reconstructed with shape (num_recon_slices, num_recon_rows, num_recon_cols).
        prox_recon (jax array): 3D proximal map input with the same shape as reconstruction.
        num_recon_rows (int): number of reconstruction rows.
        num_recon_cols (int): number of reconstruction columns.
        delta_pixel_recon (float): Reconstruction pixel spacing in ALU.
        sigma_x (float): qGGMRF prior model regularization parameter.
        sigma_p (float): Proximal map regularization parameter.
        p (float): qGGMRF shape parameter in the range (1,2].
        q (float): qGGMRF shape parameter in range [p,2].
        T (float): qGGMRF threshold parameter, >0 and typically 1.0.
        num_neighbors (int): Number of neighbors in the qGGMRF neighborhood. Possible values are {26,18,6}.

    Reconstruction Parameters:
        auto_regularize_flag (bool): Automatically adjust regularization using sharpness and snr_db
        prior_model_type (str): Can be one of "qggmrf" or "prox_map".
        positivity_flag (bool): True => enforces positivity constraint,
        initialization (str): Options include "zero", "constant", or "recon" for starting from a provided reconstruction.
        snr_db (float): Assumed signal-to-noise ratio of the input data, expressed in decibels (dB).
        sharpness (float): Controls the reconstructed reconstruction sharpness.
                           sharpness=0.0 is neutral; sharpness>0 increases sharpness; sharpness<0 reduces sharpness.
        max_resolutions (int): Specifies the maximum number of grid resolutions for multi-resolution processing.
        max_equits (float): Defines the maximum number of equivalent iterations (equits) used in reconstruction.
        verbose (int): Controls verbosity. 0 => quiet; 1 => basic information; 2 => full debugging information.
    """

    def __init__(self, angles, sinogram_shape, **kwargs):
        """
        Create a TomographyModel object using the given angles and sinogram shape, along with other keyword arguments.
        Args:
            angles (array_like): A 1D array of angles in radians.  The number of angles must match the number of views.
            sinogram (jax array): 3D jax array containing sinogram with shape (num_views, num_det_rows, num_det_channels).
            **kwargs:
        """
        self.params = utils.get_default_params()
        self.forward_project, self.back_project = None, None  # These are callable functions compiled in set_params
        self.set_params(angles=angles, sinogram_shape=sinogram_shape)
        self.auto_set_recon_size(sinogram_shape)  # Determine auto image size before processing user parameters
        self.set_params(**kwargs)

    def compile_projectors(self):
        warnings.warn('Projectors not implemented yet')

    def forward_project(self, voxel_values, voxel_indices):
        """
        Forward project the given voxel values to a sinogram.
        In the ParallelBeam case, the indices are into a flattened 2D array of shape (recon_rows, recon_cols), and the
        projection is done using all voxels with those indices across all the slices.
        Args:
            voxel_values (jax array): 2D jax array of size (len(voxel_indices), num_recon_slices) containing values of
                    voxels to be projected.  voxel_values[i, :] contains the values for all the slices at this location
            voxel_indices (int array): 1D integer index array of voxels to be projected
        Returns:
            A jax array of shape (len(voxel_indices), num_slices)
        """
        return self.forward_project(voxel_values, voxel_indices)

    def back_project(self, sinogram, indices):
        """
        Back project the given sinogram to the voxels given by the indices.
        In the ParallelBeam case, the indices are into a flattened 2D array of shape (recon_rows, recon_cols), and the
        projection is done onto all voxels with those indices across all the slices.
        Args:
            sinogram (jax array): 3D jax array containing sinogram with shape (num_views, num_det_rows, num_det_channels).
            indices (int array): (N_indices, num_recon_slices) integer index array of voxels to be computed in a flattened recon.
        Returns:
            A jax array of shape (len(voxel_indices), num_slices)
        """
        return self.back_project(sinogram, indices)

    def compute_hessian_diagonal(self, weights, angles, sinogram_shape=None):
        return self.compute_hessian_diagonal(weights, angles, sinogram_shape=None)

    def auto_set_regularization_params(self, sinogram, weights=1):
        """
        Sets the value of the parameters self.sigma_y, self.sigma_x, and self.sigma_p used for use in MBIR reconstruction.
        Args:
            sinogram (jax array): 3D jax array containing sinogram with shape (num_views, num_det_rows, num_det_channels).
            weights (scalar or 3D jax array): scalar value or 3D weights array with the same shape as sinogram.
        """
        if self.params.auto_regularize_flag == True:
            self.auto_set_sigma_y(sinogram, weights)
            self.auto_set_sigma_x(sinogram)
            self.auto_set_sigma_p(sinogram)

    def auto_set_sigma_y(self, sinogram, weights=1):
        """
        Sets the value of the parameter self.sigma_y used for use in MBIR reconstruction.
        Args:
            sinogram (jax array): 3D jax array containing sinogram with shape (num_views, num_det_rows, num_det_channels).
            weights (scalar or 3D jax array): scalar value or 3D weights array with the same shape as sinogram.
        """

        # Get parameters
        snr_db = self.params.snr_db
        delta_pixel_recon = self.params.delta_pixel_recon
        delta_det_channel = self.params.delta_det_channel
        if hasattr(self.params, 'magnification'):
            magnification = self.params.magnification
        else:
            magnification = 1.0

        # Compute indicator function for sinogram support
        sino_indicator = self.sino_indicator(sinogram)

        # Compute RMS value of sinogram excluding empty space
        signal_rms = np.average(weights * sinogram ** 2, None, sino_indicator) ** 0.5

        # Convert snr to relative noise standard deviation
        rel_noise_std = 10 ** (-snr_db / 20)
        # compute the default_pixel_pitch = the detector pixel pitch in the recon plane given the magnification
        default_pixel_pitch = delta_det_channel / magnification

        # Compute the recon pixel pitch relative to the default.
        pixel_pitch_relative_to_default = delta_pixel_recon / default_pixel_pitch

        # Compute sigma_y and scale by relative pixel pitch
        sigma_y = rel_noise_std * signal_rms * (pixel_pitch_relative_to_default ** 0.5)
        self.set_params(sigma_y=sigma_y, auto_regularize_flag=True)

    def auto_set_sigma_x(self, sinogram):
        """
        Compute the automatic value of ``sigma_x`` for use in MBIR reconstruction with qGGMRF prior.
        Args:
            sinogram (jax array): 3D jax array containing sinogram with shape (num_views, num_det_rows, num_det_channels).
        """
        sigma_x = 0.2 * self.estimate_recon_std(sinogram)
        self.set_params(sigma_x=sigma_x)

    def auto_set_sigma_p(self, sinogram):
        """
        Compute the automatic value of ``sigma_p`` for use in MBIR reconstruction with proximal map prior.
        Args:
            sinogram (jax array): 3D jax array containing sinogram with shape (num_views, num_det_rows, num_det_channels).
        """
        sigma_p = 0.2 * self.estimate_recon_std(sinogram)
        self.set_params(sigma_p=sigma_p)

    def auto_set_recon_size(self, sinogram_shape, magnification=1.0):
        """Compute the default recon size using the internal parameters delta_channel and delta_pixel plus
          the number of channels from the sinogram"""
        delta_det_row, delta_det_channel = self.get_params(['delta_det_row', 'delta_det_channel'])
        delta_pixel_recon = self.get_params('delta_pixel_recon')
        num_det_rows, num_det_channels = sinogram_shape[1:3]
        num_recon_rows = int(np.ceil(num_det_channels * delta_det_channel / (delta_pixel_recon * magnification)))
        num_recon_cols = num_recon_rows
        num_recon_slices = int(np.round(num_det_rows * ((delta_det_row / delta_pixel_recon) / magnification)))

        self.set_params(num_recon_rows=num_recon_rows, num_recon_cols=num_recon_cols, num_recon_slices=num_recon_slices)

    def estimate_recon_std(self, sinogram):
        """
        Estimate the standard deviation of the reconstruction from the sinogram.  This is used to scale sigma_p and
        sigma_x in MBIR reconstruction.
        Args:
            sinogram (jax array): 3D jax array containing sinogram with shape (num_views, num_det_rows, num_det_channels).
        """
        # Get parameters
        delta_det_channel = self.params.delta_det_channel
        sharpness = self.params.sharpness
        if hasattr(self.params, 'magnification') :
            magnification = self.params.magnification
        else :
            magnification = 1.0

        num_det_channels = sinogram.shape[-1]

        # Compute indicator function for sinogram support
        sino_indicator = self.sino_indicator(sinogram)

        # Compute a typical recon value by dividing average sinogram value by a typical projection path length
        typical_img_value = np.average(sinogram, weights=sino_indicator) / (
                num_det_channels * delta_det_channel / magnification)

        # Compute sigma_x as a fraction of the typical recon value
        sigma_prior = (2 ** sharpness) * typical_img_value
        return sigma_prior

    def print_params(self):
        print("----")
        for key, val in vars(self.params).items():
            print("{} = {}".format(key, val))
        print("----")

    def save_params(self, fname='param_dict.npy', binaries=False):
        """Save parameter dict to numpy/pickle file/yaml file"""
        output_params = self.params.copy()
        if binaries is False:
            # Wipe the binaries before saving.
            output_params['weights'] = None
            output_params['prox_recon'] = None
            output_params['init_proj'] = None
            if not np.isscalar(output_params['init_recon']):
                output_params['init_recon'] = None
        # Determine file type
        if fname[-4:] == '.npy':
            np.save(fname, output_params)
        elif fname[-4:] == '.yml' or fname[-5:] == '.yaml':
            # Work through all the parameters by group, with a heading for each group
            with open(fname, 'w') as file:
                for heading, dic in zip(ParallelBeamModel.headings, ParallelBeamModel.dicts):
                    file.write('# ' + heading + '\n')
                    for key in dic.keys():
                        val = self.params[key]
                        file.write(key + ': ' + str(val) + '\n')
        else:
            raise ValueError('Invalid file type for saving parameters: ' + fname)

    def load_params(self, fname):
        """Load parameter dict from numpy/pickle file/yaml file, and merge into instance params"""
        # Determine file type
        if fname[-4:] == '.npy':
            read_dict = np.load(fname, allow_pickle=True).item()
            self.params = utils.get_default_params()
            self.set_params(**read_dict)
        elif fname[-4:] == '.yml' or fname[-5:] == '.yaml':
            with open(fname) as file:
                try:
                    params = yaml.safe_load(file)
                    self.params = utils.get_default_params()
                    self.set_params(**params)
                except yaml.YAMLError as exc:
                    print(exc)

    def set_params(self, **kwargs):
        """
        Updates instance parameters using keyword arguments and checks for changes to critical geometry parameters.
        If a parameter key from kwargs does not exist within the instance's parameters, a NameError is raised.
        After setting parameters, it checks if key geometry-related parameters have changed and, if so, recompiles the projectors.
        Args:
            **kwargs: Arbitrary keyword arguments where keys are parameter names and values are the new parameter values.
        Raises:
            NameError: If any key provided in kwargs is not an attribute of `self.params`.
        Side effects:
            May recompile projectors if geometry parameters such as angles or sinogram shape are changed.
        """
        # Get initial geometry parameters
        initial_params = self.get_geometry_parameters()
        recompile = False
        regularization_parameter_change = False
        meta_parameter_change = False

        # Set all the given parameters
        for key, val in kwargs.items():
            if key in vars(self.params).keys():
                setattr(self.params, key, val)
            elif key == 'angles':
                self.angles = kwargs[key]
                recompile = True
            elif key == 'sinogram_shape':
                self.sinogram_shape = kwargs[key]
                recompile = True
            elif key in ["sigma_y", "sigma_x", "sigma_p"]:
                regularization_parameter_change = True
            elif key == ["sharpness", "snr_db"]:
                meta_parameter_change = True
            else:
                raise NameError('"{}" not a recognized argument'.format(key))

        # Check for compatibility between angles and sinogram views
        if recompile:
            if len(self.sinogram_shape) != 3:
                error_message = "sinogram_shape must be (views, rows, channels). \n"
                error_message += "Got {} for sinogram shape.".format(self.sinogram_shape)
                raise ValueError(error_message)
            if len(self.angles) != self.sinogram_shape[0]:
                error_message = "Number of angles must equal the number of views. \n"
                error_message += "Got {} for number of angles and {} for number of views.".format(len(self.angles), self.sinogram_shape[0])
                raise ValueError(error_message)

        # Handle case if any regularization parameter changed
        if regularization_parameter_change:
            self.params.auto_regularize_flag = False
            warnings.warn('You are directly setting regularization parameters, sigma_x, sigma_y or sigma_p. '
                          'This is an advanced feature that will disable auto-regularization.')

        # Handle case if any meta regularization parameter changed
        if meta_parameter_change:
            if self.params.auto_meta_regularization_flag is False:
                self.params.auto_regularize_flag = True
                warnings.warn('You have re-enabled auto-regularization. It was previously disabled')

        # Get final geometry parameters
        new_params = self.get_geometry_parameters()

        # Compare the two outputs
        if recompile or initial_params != new_params:
            self.compile_projectors()

    def get_geometry_parameters(self):
        return None

    def get_params(self, parameter_names):
        """
        Get the values of the listed parameter names.
        Raises an exception if a parameter name is not defined in parameters.
        Args:
            parameter_names: String or list of strings

        Returns:
            Single value or list of values
        """
        if isinstance(parameter_names, str):
            if parameter_names in vars(self.params).keys():
                value = getattr(self.params, parameter_names)
            else:
                raise NameError('"{}" not a recognized argument'.format(parameter_names))
            return value
        values = []
        for name in parameter_names:
            if name in vars(self.params).keys():
                values.append(getattr(self.params, name))
            else:
                raise NameError('"{}" not a recognized argument'.format(name))
        return values


    def get_voxels_at_indices(self, recon, indices):
        """
        Get voxels at the specified indices.
        Args:
            recon:
            indices:
        Returns:
        """
        # Get number of rows in detector
        shape = self.sinogram_shape

        # To Do: Bug: This doesn't work.
        # Either we need to implement self.params.sinogram_shape or we need to add this functionality to self.get_params().
        #shape = self.get_params('sinogram_shape')

        # Set number of detector rows
        num_det_rows = shape[1]

        # Retrieve values of recon at the indices locations
        voxel_values = recon.reshape((-1, num_det_rows))[indices]

        return voxel_values


    @staticmethod
    def get_cos_sin_angles(angles):
        """
        Take the sin and cosine of an array of num_view angles and return as a num_view x 1 jax array.
        Args:
            angles: array of angles

        Returns:
            num_view x 1 jax array containing cos and sin of the angles
        """
        cos_angles = jnp.cos(angles).flatten()
        sin_angles = jnp.sin(angles).flatten()
        return jnp.stack([cos_angles, sin_angles], axis=0)

    @staticmethod
    def sino_indicator(sinogram):
        """
        Compute a binary function that indicates the region of sinogram support.
        Args:
            sinogram (jax array): 3D jax array containing sinogram with shape (num_views, num_det_rows, num_det_channels).
        Returns:
            (jax array): Weights used in mbircone reconstruction, with the same array shape as ``sinogram``.
        """
        percent_noise_floor = 5.0
        # Form indicator by thresholding sinogram
        indicator = jnp.int8(sinogram > (0.01 * percent_noise_floor) * jnp.mean(jnp.fabs(sinogram)))
        return indicator

    @staticmethod
    def calc_weights(sinogram, weight_type):
        """
        Compute the weights used in MBIR reconstruction.
        Args:
            sinogram (jax array): 3D jax array containing sinogram with shape (num_views, num_det_rows, num_det_channels).
            weight_type (string): Type of noise model used for data
                    - weight_type = 'unweighted' => return numpy.ones(sinogram.shape).
                    - weight_type = 'transmission' => return numpy.exp(-sinogram).
                    - weight_type = 'transmission_root' => return numpy.exp(-sinogram/2).
                    - weight_type = 'emission' => return 1/(numpy.absolute(sinogram) + 0.1).
        Returns:
            (jax array): Weights used in mbircone reconstruction, with the same array shape as ``sinogram``.
        Raises:
            Exception: Raised if ``weight_type`` is not one of the above options.
        """
        if weight_type == 'unweighted':
            weights = jnp.ones(sinogram.shape)
        elif weight_type == 'transmission':
            weights = jnp.exp(-sinogram)
        elif weight_type == 'transmission_root':
            weights = jnp.exp(-sinogram / 2)
        elif weight_type == 'emission':
            weights = 1.0 / (jnp.absolute(sinogram) + 0.1)
        else:
            raise Exception("calc_weights: undefined weight_type {}".format(weight_type))

        return weights


    def recon( self, sinogram, weights = 1.0 ):
        """
        Perform MBIR reconstruction using the Multi-Granular Vector Coordinate Descent algorithm.
        This function takes care of generating its own partitions and partition sequence.
        Args:
            sinogram (jax array): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels).
            weights (scalar or jax array): scalar or 3D positive weights with same shape as error_sinogram.
        Returns:
            [recon, fm_rmse]: reconstruction and array of loss for each iteration.
        """
        # Run auto regularization. If self.params.auto_regularize_flag is False, then this will have no effect
        self.auto_set_regularization_params(sinogram, weights=weights)

        # Generate set of voxel partitions
        partitions = self.gen_set_of_voxel_partitions()

        # Generate sequence of partitions to use
        partition_sequence = self.gen_partition_sequence()

        # Compute reconstruction
        recon, fm_rmse = self.vcd_recon(sinogram, partitions, partition_sequence, weights=weights)

        return recon, fm_rmse


    def vcd_recon( self, sinogram, partitions, partition_sequence, weights = 1.0 ):
        """
        Perform MBIR reconstruction using the Multi-Granular Vector Coordinate Descent algorithm
        for a given set of partitions and a prescribed partition sequence.
        Args:
            sinogram (jax array): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels).
            partitions (tuple): A collection of K partitions, with each partition being an (N_indices) integer index array of voxels to be updated in a flattened recon.
            partition_sequence (jax array): A sequence of integers that specify which partition should be used at each iteration.
            weights (scalar or jax array): scalar or 3D positive weights with same shape as error_sinogram.
        Returns:
            [recon, fm_rmse]: reconstruction and array of loss for each iteration.
        """
        # Get required parameters
        num_iters = partition_sequence.size
        num_recon_rows = self.params.num_recon_rows
        num_recon_cols = self.params.num_recon_cols
        num_recon_slices = self.params.num_recon_slices
        angles = self.angles

        # Initialize VCD error sinogram, recon, and hessian
        error_sinogram = sinogram
        recon = jnp.zeros((num_recon_rows, num_recon_cols, num_recon_slices))
        hessian = self.compute_hessian_diagonal(weights=weights, angles=angles)

        # Initialize forward model normalized RMSE error array
        fm_rmse = np.zeros(num_iters)

        for i in range(num_iters):
            error_sinogram, recon = self.vcd_partition_iteration(error_sinogram, recon, partitions[partition_sequence[i]], hessian, weights=weights)
            fm_rmse[i] = self.forward_model_loss(error_sinogram)
            if self.params.verbose>=1:
                print(f'VCD iteration={i}; Loss={fm_rmse[i]}')

        return recon, fm_rmse


    def vcd_partition_iteration( self, error_sinogram, recon, partition, fm_hessian, weights = 1.0 ):
        """
        Calculate an iteration of the VCD algorithm for each subset of the partition

        Each iteration of the algorithm should return a better reconstructed recon. The error_sinogram should always be:
                error_sinogram = measured_sinogram - forward_proj(recon)
        where measured_sinogram is the measured sinogram and recon is the current reconstruction.
        Args:
            error_sinogram (jax array): 3D error sinogram with shape (num_views, num_det_rows, num_det_channels).
            partition (int array): (K, N_indices) an integer index arrays that partitions
                the voxels into K arrays, each of which indexes into a flattened recon.
            recon (jax array): 3D array reconstruction with shape (num_recon_rows, num_recon_cols, num_recon_slices).
            fm_hessian (jax array): Array with same shape as recon containing diagonal of hessian for forward model loss.
            weights (scalar or jax array): scalar or 3D positive weights with same shape as error_sinogram.
        Returns:
            [error_sinogram, recon]: Both have the same shape as above, but are updated to reduce overall loss function.
        """
        for subset in np.random.permutation(partition.shape[0]):
            error_sinogram, recon = self.vcd_subset_iteration(error_sinogram, recon, partition[subset], fm_hessian,
                                                              weights=weights)

        return error_sinogram, recon


    def vcd_subset_iteration(self, error_sinogram, recon, indices, fm_hessian, weights = 1.0):
        """
        Calculate an iteration of the VCD algorithm on a single subset of the partition

        Each iteration of the algorithm should return a better reconstructed recon.
        The combination of (error_sinogram, recon) form a overcomplete state that make computation efficient.
        However, it is important that at each application the state should meet the constraint that:
                error_sinogram = measured_sinogram - forward_proj(recon)
        where measured_sinogram forward_proj() is whatever forward projection is being used in reconstruction.

        Args:
            error_sinogram (jax array): 3D error sinogram with shape (num_views, num_det_rows, num_det_channels).
            indices (int array): (N_indices) integer index array of voxels to be updated in a flattened recon.
            recon (jax array): 3D array reconstruction with shape (num_recon_rows, num_recon_cols, num_recon_slices).
            fm_hessian (jax array): Array with same shape as recon containing diagonal of hessian for forward model loss.
            weights (scalar or jax array): scalar or 3D positive weights with same shape as error_sinogram.
        Returns:
            [error_sinogram, recon]: Both have the same shape as above, but are updated to reduce overall loss function.
        """
        # Recover recon shape parameters and make sure that recon has a 3D shape
        num_recon_rows = self.params.num_recon_rows
        num_recon_cols = self.params.num_recon_cols
        num_recon_slices = self.params.num_recon_slices
        recon = recon.reshape((num_recon_rows, num_recon_cols, num_recon_slices))

        # flatten the recon and hessian if they are not already flat
        fm_hessian = fm_hessian.reshape((-1, num_recon_slices))

        # Compute the forward model gradient and hessian at each pixel in the index set.
        # Assumes Loss(delta) = 1/(2 sigma_y^2) || error_sinogram - A delta ||_weights^2
        constant = 1.0/(self.params.sigma_y ** 2.0)

        fm_gradient = -constant*self.back_project(error_sinogram*weights, indices)
        fm_sparse_hessian = constant*fm_hessian[indices]

        # Compute the prior model gradient and hessian at each pixel in the index set.
        sigma_x, p, q, T = self.get_params(['sigma_x', 'p', 'q', 'T'])
        pm_gradient, pm_hessian = pm_gradient_and_hessian_at_indices(recon, indices, sigma_x, p, q, T)

        # Compute update vector update direction in recon domain
        delta_recon_at_indices = (- fm_gradient - pm_gradient) / (fm_sparse_hessian + pm_hessian)

        # Compute update direction in sinogram domain
        delta_sinogram = self.forward_project(delta_recon_at_indices, indices)

        # Compute "optimal" update step
        # This is really only optimal for the forward model component.
        # We can compute the truly optimal update, but it's complicated so maybe this is good enough
        alpha = jnp.sum(error_sinogram * delta_sinogram * weights) / jnp.sum(delta_sinogram * delta_sinogram * weights)

        # Flatten recon; perform sparse updates at index locations; reshape as 3D array
        recon = recon.reshape((-1, num_recon_slices))
        recon = recon.at[indices].add(alpha * delta_recon_at_indices)
        recon = recon.reshape((num_recon_rows, num_recon_cols, num_recon_slices))

        # Update sinogram
        error_sinogram = error_sinogram - alpha * delta_sinogram

        return error_sinogram, recon

    def forward_model_loss(self, error_sinogram, weights=1.0, normalize = True):
        """
        Calculate the loss function for the forward model from the error_sinogram and weights.
        The error sinogram should be error_sinogram = measured_sinogram - forward_proj(recon)
        Args:
            error_sinogram (jax array): 3D error sinogram with shape (num_views, num_det_rows, num_det_channels).
            weights (jax array): 3D weights array with same shape as sinogram
        Returns:
            [loss].
        """
        if normalize:
            avg_weight =jnp.average(weights)
            loss = jnp.sqrt((1.0 / (self.params.sigma_y ** 2)) * jnp.mean((error_sinogram * error_sinogram) * (weights/avg_weight)))
        else:
            loss = (1.0 / (2 * self.params.sigma_y ** 2)) * jnp.sum((error_sinogram * error_sinogram) * weights)
        return loss

    def gen_set_of_voxel_partitions(self):
        """
        Generates a collection of voxel partitions for an array of specified partition sizes.
        This function creates a tuple of randomly generated 2D voxel partitions.
        Args:
            num_recon_rows (int): Number of rows in the reconstruction grid.
            num_recon_cols (int): Number of columns in the reconstruction grid.
            granularity (list of integers): 1D array of integers where each integer specifies the number of subsets in a partition.
        Returns:
            tuple: A tuple of 2D arrays each representing a partition of voxels into the specified number of subsets.
        """
        # Convert granularity to an np array
        granularity = np.array(self.params.granularity)

        partitions = ()
        for size in granularity :
            partition = mbirjax.gen_voxel_partition(self.params.num_recon_rows, self.params.num_recon_cols, size)
            partitions += (partition,)

        return partitions

    def gen_full_indices( self ):
        """
        Generates a full array of voxels in the region of reconstruction.
        This is useful for computing forward projections.
        """
        # Convert granularity to an np array
        partition = mbirjax.gen_voxel_partition(self.params.num_recon_rows, self.params.num_recon_cols, num_subsets=1)
        full_indices = partition[0]

        return full_indices

    def gen_partition_sequence(self) :
        # Get sequence from params and convert it to a np array
        partition_sequence = np.array(self.params.partition_sequence)

        # Tile sequence so it at least iterations long
        num_iterations = self.params.num_iterations
        extended_partition_sequence = np.tile(partition_sequence, (num_iterations // partition_sequence.size + 1))[0 :num_iterations]
        return extended_partition_sequence

    def gen_3d_shepp_logan_phantom(self):
        """
        Generates a 3D Shepp-Logan phantom.
        Returns:
            ndarray: A 3D numpy array of shape specified by TomographyModel class parameters.
        """
        phantom = mbirjax.generate_3d_shepp_logan(self.params.num_recon_rows, self.params.num_recon_cols, self.params.num_recon_slices)
        return phantom

    def reshape_recon(self, recon):
        """
        Reshape recon into its 3D form
        """
        return recon.reshape(self.params.num_recon_rows, self.params.num_recon_cols, self.params.num_recon_slices)


def get_rho(delta, b, sigma_x, p, q, T):
    """
    Computes the sum of the neighboring qGGMRF prior potential functions rho for a given delta.
    Args:
        delta (float or np.array): (batch_size, P) array of pixel differences between center pixel and each of P neighboring pixels.
        b (float or np.array): (1,N) array of neighbor pixel weights that usually sums to 1.0.
    Returns:
        float or np.array: (batch_size,) array of locally summed potential function rho values for the given pixel.
    """

    # Smallest single precision float
    eps_float32 = np.finfo(np.float32).eps
    delta = abs(delta) + sigma_x * eps_float32

    # Compute terms of complex expression
    first_term = ((delta / sigma_x) ** p) / p
    second_term = (delta / (T * sigma_x)) ** (q - p)
    third_term = second_term / (1 + second_term)

    result = np.sum(b * (first_term * third_term), axis=-1)  # Broadcast b over batch_size dimension
    return result


@jax.jit
def get_btilde(delta_prime, b, sigma_x, p, q, T):
    """
    Compute the quadratic surrogate coefficients btilde from page 117 of FCI for the qGGMRF prior model.
    Args:
        delta_prime (float or np.array): (batch_size, P) array of pixel differences between center and each of P neighboring pixels.
        b (float or np.array): (1,N) array of neighbor pixel weights that usually sums to 1.0.
    Returns:
        float or np.array: (batch_size, P) array of surrogate coefficients btilde.
    """

    # Smallest single precision float
    eps_float32 = np.finfo(np.float32).eps
    delta_prime = abs(delta_prime) + sigma_x * eps_float32

    # first_term is the product of the first three terms reorganized for numerical stability when q=0.
    first_term = (delta_prime ** (q-2.0))/(2.0*(sigma_x ** p)*(T*sigma_x ** (q-p)))

    # third_term is the third term in formula.
    second_term = (delta_prime/(T*sigma_x)) ** (q-p)
    third_term = ((q/p)+second_term)/((1+second_term) ** 2.0)

    result = b*(first_term*third_term)  # Broadcast b over batch_size dimension
    return result


@jax.jit
def pm_gradient_and_hessian(delta_prime, b, sigma_x, p, q, T):
    """
    Computes the first and second derivatives of the surrogate function at a pixel for the qGGMRF prior model.
    Calculations taken from Figure 8.5 (page 119) of FCI for the qGGMRF prior model.
    Args:
        delta_prime (float or np.array): (batch_size, N) array of pixel differences between center and each of N neighboring pixels.
        b (float or np.array): (1,N) array of neighbor pixel weights that usually sums to 1.0.
    Returns:
        float or np.array: (batch_size,) array of first derivatives of the surrogate function at pixel.
        float or np.array: (batch_size,) array of second derivatives of the surrogate function at pixel.
    """
    # Compute the btilde values required for quadratic surrogate
    btilde = get_btilde(delta_prime, b, sigma_x, p, q, T)

    # Compute first derivative
    pm_first_derivative = jnp.sum(2 * btilde * delta_prime, axis=-1)

    # Compute second derivative
    pm_second_derivative = jnp.sum(2 * btilde, axis=-1)

    return pm_first_derivative, pm_second_derivative


@jax.jit
def pm_gradient_and_hessian_at_indices(recon, indices, sigma_x, p, q, T):
    """
    Calculate the gradient and hessian at each index location in recon using the qGGMRF prior.
    Args:
        recon (jax array): 3D recon with shape (num_recon_rows, num_recon_cols, num_recon_slices)
        indices (int array): (N_indices, num_recon_slices) integer index array of voxels to be updated in a flattened recon.
    Returns:
        [first_derivative, second_derivative]: two jpn arrays each of shape (N_indices, num_recon_slices)
    """
    # Set neighborhood weights
    b = jnp.array([1, 1, 1, 1]).reshape(1, -1)/4.0

    num_rows, num_cols, num_slices = recon.shape
    row_index, col_index = jnp.unravel_index(indices, (num_rows, num_cols))
    xs = recon[row_index, col_index]
    offsets = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    xr = [recon[row_index + offset[0], col_index + offset[1]] for offset in offsets]
    x_neighbors_prime_new = jnp.stack(xr, axis=-1)
    delta_prime = xs[:, :, jnp.newaxis] - x_neighbors_prime_new
    delta_prime_up_down = jnp.diff(xs, axis=-1)  # TODO:  qggmrf in slice direction

    delta_prime = delta_prime.reshape((-1, b.shape[-1]))
    first_derivative, second_derivative = pm_gradient_and_hessian(delta_prime, b, sigma_x, p, q, T)
    first_derivative = first_derivative.reshape(-1, num_slices)
    second_derivative = second_derivative.reshape(-1, num_slices)

    return first_derivative, second_derivative


class ParallelBeamModel(TomographyModel):
    """
    A class to do forward and back projection for parallel beam geometry.
    """

    def __init__(self, angles, sinogram_shape, **kwargs):
        super().__init__(angles, sinogram_shape, **kwargs)

    def get_geometry_parameters(self):
        """
        Get a list of the primary geometry parameters.
        Returns:
            List of delta_det_channel, det_channel_offset, delta_pixel_recon, num_recon_rows, num_recon_cols
        """
        geometry_params = self.get_params(['delta_det_channel', 'det_channel_offset', 'delta_pixel_recon'])
        num_recon_rows, num_recon_cols = self.get_params(['num_recon_rows', 'num_recon_cols'])

        # TODO:  Need to include detector rows/recon slices
        return geometry_params + [num_recon_rows, num_recon_cols]

    def compile_projectors(self):
        """
        Compute the forward and back projectors for this set of angles
        Args:

        Returns:
            A callable of the form back_projector(sinogram, indices), where sinogram is an array with shape
            (views, rows, channels) and indices is a 1D array of integer indices into the flattened reconstruction
            volume.
            The output of a call to back_projector is the backprojection of the sinogram onto the corresponding voxels.

        Note:
            The returned function will be jit compiled each time it is called with a new shape of input.  If it is
            called multiple times with the same shape of input, then the cached version will be used, which will
            give reduced execution time relative to the initial call.
        """
        geometry_params = self.get_geometry_parameters()
        cos_sin_angles = self.get_cos_sin_angles(self.angles)

        # def bp_to_voxel(sinogram, cur_index):
        #     return self.backproject_to_voxel(sinogram, cur_index, cos_sin_angles, geometry_params)
        #
        # # There is significant speed up from each of
        # #   (1) Using jit on the vmap and
        # #   (2) Defining bp_to_voxel to include the externally defined angles and geometry parameters.
        # # Inputs are sinogram, indices
        # back_project_fcn = jax.jit(jax.vmap(bp_to_voxel, in_axes=(None, 0)))
        @jax.jit
        def back_project_fcn(sinogram, indices):
            return ParallelBeamModel.back_project_to_voxels_scan(sinogram, indices, cos_sin_angles, geometry_params)

        @jax.jit
        def forward_project_fcn(voxel_values, voxel_indices):
            forward_vmap = jax.vmap(self.forward_project_voxels_one_view, in_axes=(None, None, 1, None, None))
            sinogram = forward_vmap(voxel_values, voxel_indices, cos_sin_angles, geometry_params, self.sinogram_shape)
            return sinogram

        # See TomographyModel.forward_project() TomographyModel.back_project() for documentation
        self.forward_project, self.back_project = forward_project_fcn, back_project_fcn

    @staticmethod
    def back_project_to_voxels_scan(sinogram, voxel_indices, cos_sin_angles, geometry_params, coeff_power=1):
        """
        Use jax.lax.scan to backproject one view at a time and accumulate the results in the specified voxels.
        Args:
            sinogram:
            voxel_indices:
            cos_sin_angles:
            geometry_params:
            coeff_power:

        Returns:

        """
        # jax.lax.scan applies a function to each entry indexed by the leading dimension of its input, the incorporates
        # the output of that function into an accumulator.  Here we apply accumulate and project to one sinogram
        # view and the cos and sin of the corresponding angle.
        initial_bp = jnp.zeros((voxel_indices.shape[0], sinogram.shape[1]))
        extra_args = voxel_indices, geometry_params, coeff_power
        initial_carry = [extra_args, initial_bp]
        sino_angles = (sinogram, cos_sin_angles.T)
        # Use lax.scan to process each (slice, angle) pair of 'sino_angles'
        final_carry, _ = lax.scan(ParallelBeamModel.accumulate_and_project, initial_carry, sino_angles)

        return final_carry[1]

    @staticmethod
    def accumulate_and_project(carry, sino_angle_pair):
        """

        Args:
            carry:
            sino_angle_pair:

        Returns:

        """
        extra_args, accumulated = carry
        sinogram_view, cos_sin_angle = sino_angle_pair
        voxel_indices, geometry_params, coeff_power = extra_args
        bp_view = ParallelBeamModel.back_project_one_view_to_voxels(sinogram_view, voxel_indices, cos_sin_angle, geometry_params, coeff_power)
        new_accumulated = accumulated + bp_view
        return [extra_args, new_accumulated], None

    @staticmethod
    @jax.jit
    def back_project_one_view_to_voxels(sinogram_view, voxel_indices, cos_sin_angle, geometry_params, coeff_power=1):
        """

        Args:
            sinogram_view:
            voxel_indices:
            cos_sin_angle:
            geometry_params:
            coeff_power:

        Returns:

        """
        bp_vmap = jax.vmap(ParallelBeamModel.back_project_one_view_to_voxel, in_axes=(None, 0, None, None, None))
        bp = bp_vmap(sinogram_view, voxel_indices, cos_sin_angle, geometry_params, coeff_power)
        return bp

    @staticmethod
    def back_project_one_view_to_voxel(sinogram_view, voxel_index, cos_sin_angle, geometry_params, coeff_power=1):
        """
        Calculate the backprojection value at a specified recon voxel given a sinogram view and various parameters.
        This code uses the distance driven projector.
        Args:
            sinogram_view: [jax array] one view of the sinogram to be back projected
            voxel_index: the integer index into flattened recon - need to apply unravel_index(voxel_index, recon_shape) to get i, j, k
            cos_sin_angle:
            geometry_params:
            coeff_power: [int] backproject using the coefficients of (A_ij ** coeff_power).
                Normally 1, but should be 2 when computing theta 2.
        Returns:
            The value of the voxel at the input index obtained by backprojecting the input sinogram.
        """

        # Get the geometry parameters and the system matrix and channel indices
        num_det_rows, num_det_channels = sinogram_view.shape
        num_slices = num_det_rows

        Aji, channel_index = ParallelBeamModel.compute_Aji_channel_index(voxel_index, cos_sin_angle, geometry_params,
                                                                         (1,) + sinogram_view.shape)

        # Extract out the relevant entries from the sinogram
        sinogram_array = sinogram_view[:, channel_index.T.flatten()]

        # Compute dot product
        return jnp.sum(sinogram_array * (Aji ** coeff_power), axis=1)

    def compute_hessian_diagonal(self, weights, angles, sinogram_shape=None):
        """
        Computes the diagonal of the Hessian matrix, which is computed by doing a backprojection of the weight
        matrix except using the square of the coefficients in the backprojection to a given voxel.
        One of weights or sinogram_shape must be not None. If weights is not None, it must be an array with the same
        shape as the sinogram to be backprojected.  If weights is None, then a weights matrix will computed as an
        array of ones of size sinogram_shape.
        Args:
            weights (None or array): The weights with shape (views,)
            angles (1D array): The angles with shape (views,).
            sinogram_shape (None or tuple):

        Returns:
            An array that is the same size as the reconstruction.
        """
        if weights is None:
            weights = jnp.ones(sinogram_shape)
        geometry_params = self.get_geometry_parameters()
        cos_sin_angles = self.get_cos_sin_angles(angles)

        num_recon_rows, num_recon_cols = geometry_params[-2:]
        num_recon_slices = weights.shape[1]
        max_index = num_recon_rows * num_recon_cols
        indices = jnp.arange(max_index)

        # def bp_to_voxel(cur_index):
        #     return self.backproject_to_voxel(weights, cur_index, cos_sin_angles, geometry_params, coeff_power=2)
        #
        # hessian_function = jax.jit(jax.vmap(bp_to_voxel, in_axes=0))
        # hessian_diagonal = hessian_function(indices)

        hessian_diagonal = ParallelBeamModel.back_project_to_voxels_scan(weights, indices, cos_sin_angles, geometry_params, coeff_power=2)

        return hessian_diagonal.reshape((num_recon_rows, num_recon_cols, num_recon_slices))

    @staticmethod
    def forward_project_voxels_one_view(voxel_values, voxel_indices, cos_sin_angle, geometry_params, sinogram_shape):
        """
        Forward project a set of voxels determined by indices into the flattened array of size num_rows x num_cols.
        Args:
            voxel_values (jax array):  2D array of shape (num_indices, num_slices) of voxel values, where
                voxel_values[i, j] is the value of the voxel in slice j at the location determined by voxel_indices[i].
            voxel_indices (jax array of int):  1D vector of indices into flattened array of size num_rows x num_cols.
            cos_sin_angle (jax array):  2D array of cosines and sines from get_cos_sin_angles()
            geometry_params (list): Geometry parameters from get_geometry_params()
            sinogram_shape (tuple): Sinogram shape

        Returns:
            jax array of shape sinogram_shape
        """
        if voxel_values.ndim != 2:
            raise ValueError('voxel_values must have shape (num_indices, num_slices)')
        num_slices = voxel_values.shape[1]

        # Get the geometry parameters and the system matrix and channel indices
        num_views, num_det_rows, num_det_channels = sinogram_shape
        Aji, channel_index = ParallelBeamModel.compute_Aji_channel_index(voxel_indices, cos_sin_angle, geometry_params, sinogram_shape)

        # Add axes to be able to broadcast while multiplying.
        # sinogram_values has shape num_indices x (2P+1) x num_slices
        sinogram_values = (Aji[:, :, None] * voxel_values[:, None, :])

        # Now sum over indices into the locations specified by channel_index.
        # Directly using index_add for indexed updates
        sinogram_view = jnp.zeros((num_slices, num_det_channels))  # num_det_rows x num_det_channels

        # Apply the vectorized update function with a vmap over slices
        # sinogram_view is num_slices x num_det_channels, sinogram_values is num_indices x (2P+1) x num_slices
        sinogram_values = sinogram_values.transpose((2, 0, 1)).reshape((num_slices, -1))
        sinogram_view = sinogram_view.at[:, channel_index.flatten()].add(sinogram_values)
        # sinogram_view = ParallelBeamModel.update_slices_vmap()(sinogram_view, sinogram_values, channel_index)
        return sinogram_view

    @staticmethod
    @jax.jit
    def backproject_to_voxel(sinogram, voxel_index, cos_sin_angles, geometry_params, coeff_power=1):
        """
        Calculate the backprojection value at a specified recon voxel given the sinogram and various parameters.
        This code uses the distance driven projector.
        Args:
            sinogram: [jax array] the sinogram to be back projected
            voxel_index: the integer index into flattened recon - need to apply unravel_index(voxel_index, recon_shape) to get i, j, k
            cos_sin_angles:
            geometry_params:
            coeff_power: [int] backproject using the coefficients of (A_ij ** coeff_power).
                Normally 1, but should be 2 when computing theta 2.
        Returns:
            The value of the voxel at the input index obtained by backprojecting the input sinogram.
        """

        # Get the geometry parameters and the system matrix and channel indices
        num_views, num_det_rows, num_det_channels = sinogram.shape
        num_slices = num_det_rows

        Aji, channel_index = ParallelBeamModel.compute_Aji_channel_index(voxel_index, cos_sin_angles, geometry_params,
                                                                         sinogram.shape)

        # Extract out the relevant entries from the sinogram
        view_index = jnp.arange(num_views)
        view_index = jnp.concatenate((view_index, view_index, view_index))  # Should be 3*num_views
        sinogram_array = sinogram[view_index, :, channel_index.T.flatten()]

        # Compute dot product
        return jnp.sum(sinogram_array * (Aji.T.reshape((-1, 1)) ** coeff_power), axis=0)

    @staticmethod
    @jax.jit
    def compute_Aji_channel_index(voxel_indices, cos_sin_angles, geometry_params, sinogram_shape):

        # TODO:  P should be included in function signature with a partial on the jit
        warnings.warn('Using hard-coded detectors per side.  These should be set dynamically based on the geometry.')
        P = 1  # This is the assumed number of channels per side

        # Get all the geometry parameters
        delta_det_channel, det_channel_offset, delta_pixel_recon, num_recon_rows, num_recon_cols = geometry_params

        num_views, num_det_rows, num_det_channels = sinogram_shape

        # Convert the index into (i,j,k) coordinates corresponding to the indices into the 3D voxel array
        recon_shape = (num_recon_rows, num_recon_cols)
        row_index, col_index = jnp.unravel_index(voxel_indices, recon_shape)

        # Compute the x,y position of the voxel relative to the center of rotation
        # Assumes: rows index top to bottom; slice is viewed from the top; rotation of object is clockwise
        y_pos = -delta_pixel_recon * (row_index - ((num_recon_rows - 1.0) / 2.0))  # length = num_indices
        x_pos = delta_pixel_recon * (col_index - ((num_recon_cols - 1.0) / 2.0))

        # Compute projection of the scalar center-of-rotation onto the detector in ALUs
        channel_center = (delta_det_channel * (num_det_channels - 1.0) / 2.0) + det_channel_offset

        # Precompute cosine and sine of view angle
        # angles = angles.reshape((-1, 1))  # Reshape to be a column vector of size num_views x 1
        cosine = cos_sin_angles[0:1].T  # jnp.cos(angles)    # length = num_views
        sine = cos_sin_angles[1:2].T  # jnp.sin(angles)      # length = num_views

        # Rotate coordinates of pixel
        x_pos_rot = cosine * x_pos + sine * y_pos  # length = num_indices
        # y_pos_rot = -sine*x_pos + cosine*y_pos

        # Calculate cos alpha = cos ( smallest angle between source-voxel line and voxel edge )
        cos_alpha = jnp.maximum(jnp.abs(cosine), jnp.abs(sine))  # length = num_indices

        # Calculate W = length of projection of flattened voxel on detector
        W = delta_pixel_recon * cos_alpha  # length = num_indices

        # Compute the location on the detector in ALU of the projected center of the voxel
        x_pos_on_detector = x_pos_rot + channel_center  # length = num_indices

        # Compute a jnp array with 2P+1 entries that are the channel indices of the relevant channels
        # Hardwired for P=1, i.e., 3 pixel window
        channel_index = jnp.round(x_pos_on_detector / delta_det_channel).astype(int)  # length = num_indices
        channel_index = channel_index.reshape((-1, 1))
        # Compute channel indices for 3 adjacent channels at each view angle
        # Should be num_indices x 3
        channel_index = jnp.concatenate([channel_index - 1, channel_index, channel_index + 1], axis=-1)

        # Compute the distance of each channel from the projected center of the voxel
        delta = jnp.abs(
            channel_index * delta_det_channel - x_pos_on_detector.reshape((-1, 1)))  # Should be num_indices x 3

        # Calculate L = length of intersection between detector element and projection of flattened voxel
        tmp1 = (W + delta_det_channel) / 2.0  # length = num_indices
        tmp2 = (W - delta_det_channel) / 2.0  # length = num_indices
        Lv = jnp.maximum(tmp1 - jnp.maximum(jnp.abs(tmp2), delta), 0)  # Should be num_indices x 3

        # Compute the values of Aij
        Aji = (delta_pixel_recon / cos_alpha) * (Lv / delta_det_channel)  # Should be num_indices x 3
        Aji = Aji * (channel_index >= 0) * (channel_index < num_det_channels)
        return Aji, channel_index


def get_transpose(linear_map, input_shape):
    """
    Use jax to determine the transpose of a linear map.
    Args:
        linear_map:  [function] The linear function to be transposed
        input_shape: [ndarray] The shape of the input to the function

    Returns:
        transpose: A function to evaluate the transpose of the given map.  The input to transpose
        must be a jax or ndarray with the same shape as the output of the original linear_map.
        transpose(input) returns a 1 element tuple containing an array holding the result, so the final output
        must be obtained using transpose(input)[0]
    """
    # print('Defining transpose map')
    # t0 = time.time()
    input_info = types.SimpleNamespace(shape=input_shape, dtype=np.dtype(np.float32))
    transpose = jax.linear_transpose(linear_map, input_info)
    # print('Done: ' + str(time.time() - t0))
    return transpose
