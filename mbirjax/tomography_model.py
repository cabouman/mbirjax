import types
import numpy as np
import yaml
import warnings
import gc
import jax
import jax.numpy as jnp
import mbirjax
import mbirjax._utils as utils


class TomographyModel:
    """
    Represents a general model for tomographic reconstruction using MBIRJAX. This class encapsulates the parameters and
    methods for the forward and back projection processes required in tomographic imaging.

    Note that this class is a template for specific subclasses.  TomographyModel by itself does not implement
    projectors or recon.  Use self.print_params() to print the parameters of the model after initialization.

    Args:
        sinogram_shape (tuple): The shape of the sinogram array expected (num_views, num_det_rows, num_det_channels).
        **kwargs: Arbitrary keyword arguments for setting model parameters dynamically.

    Sets up the reconstruction size and parameters.
    """

    def __init__(self, sinogram_shape, **kwargs):

        self.params = utils.get_default_params()
        self.add_new_params(**kwargs)
        self._sparse_forward_project, self._sparse_back_project = None, None  # These are callable functions compiled in set_params
        self.set_params(sinogram_shape=sinogram_shape, **kwargs)
        self.auto_set_recon_size(sinogram_shape)  # Determine auto image size before processing user parameters

    def compile_projectors(self):
        """Placeholder for compiling projector methods."""
        warnings.warn('Projectors not implemented yet')

    def forward_project(self, recon):
        """
        Perform a full forward projection at all voxels in the field-of-view.
        Args:
            recon (jnp array): The 3D reconstruction array.
        Returns:
            jnp array: The resulting 3D sinogram after projection.
        """
        full_indices = self.gen_full_indices()
        voxel_values = self.get_voxels_at_indices(recon, full_indices)
        sinogram = self.sparse_forward_project(voxel_values, full_indices)

        return sinogram

    def back_project(self, sinogram):
        """
        Perform a full back projection at all voxels in the field-of-view.
        Args:
            sinogram (jnp array): 3D jax array containing sinogram.
        Returns:
            jnp array: The resulting 3D sinogram after projection.
        """
        full_indices = self.gen_full_indices()
        recon_at_indices = self.sparse_back_project(sinogram, full_indices)

        # Get shape of recon
        num_recon_rows, num_recon_cols, num_recon_slices = self.get_params(['num_recon_rows','num_recon_cols','num_recon_slices'])

        # Allocate recon of correct shape
        recon = jnp.zeros((num_recon_rows * num_recon_cols, num_recon_slices))

        # Add back projections to allocated recon
        recon = recon.at[full_indices].add(recon_at_indices)

        return recon

    def sparse_forward_project(self, voxel_values, indices, view_batch_size=None):
        """
        Forward project the given voxel values to a sinogram.
        The indices are into a flattened 2D array of shape (recon_rows, recon_cols), and the projection is done using
        all voxels with those indices across all the slices.

        Args:
            voxel_values (jax.numpy.DeviceArray): 2D array of voxel values to project, size (len(voxel_indices), num_recon_slices).
            indices (numpy.ndarray): Array of indices specifying which voxels to project.
            view_batch_size (int):

        Returns:
            jnp array: The resulting 3D sinogram after projection.
        """
        sinogram = self._sparse_forward_project(voxel_values, indices, view_batch_size=view_batch_size).block_until_ready()
        gc.collect()
        return sinogram

    def sparse_back_project(self, sinogram, indices, voxel_batch_size=None):
        """
        Back project the given sinogram to the voxels given by the indices.
        The indices are into a flattened 2D array of shape (recon_rows, recon_cols), and the projection is done using
        all voxels with those indices across all the slices.

        Args:
            sinogram (jnp array): 3D jax array containing sinogram.
            indices (jnp array): Array of indices specifying which voxels to back project.
            voxel_batch_size (int):

        Returns:
            A jax array of shape (len(indices), num_slices)
        """
        recon = self._sparse_back_project(sinogram, indices, voxel_batch_size=voxel_batch_size).block_until_ready()
        gc.collect()
        return recon

    def compute_hessian_diagonal(self, weights, voxel_batch_size=None):
        """
        Computes the diagonal elements of the Hessian matrix for given weights and angles.

        Args:
            weights (jnp array): Sinogram Weights for the Hessian computation.
            voxel_batch_size:

        Returns:
            jnp array: Diagonal of the Hessian matrix with same shape as recon.
        """
        hessian = self.compute_hessian_diagonal(weights, voxel_batch_size=voxel_batch_size).block_until_ready()
        gc.collect()
        return hessian

    def auto_set_regularization_params(self, sinogram, weights=1):
        """
        Automatically sets the regularization parameters (self.sigma_y, self.sigma_x, and self.sigma_p) used in MBIR reconstruction based on the provided sinogram and optional weights.

        Args:
            sinogram (jnp.array): 3D jax array containing the sinogram with shape (num_views, num_det_rows, num_det_channels).
            weights (scalar or jnp.array, optional): Scalar value or 3D weights array with the same shape as the sinogram. Defaults to 1.

        The method adjusts the regularization parameters only if `auto_regularize_flag` is set to True within the model's parameters.
        """
        if self.get_params('auto_regularize_flag'):
            self.auto_set_sigma_y(sinogram, weights)
            self.auto_set_sigma_x(sinogram)
            self.auto_set_sigma_p(sinogram)

    def auto_set_sigma_y(self, sinogram, weights=1):
        """
        Sets the value of the parameter sigma_y used for use in MBIR reconstruction.

        Args:
            sinogram (jax array): 3D jax array containing sinogram with shape (num_views, num_det_rows, num_det_channels).
            weights (scalar or 3D jax array): scalar value or 3D weights array with the same shape as sinogram.
        """

        # Get parameters
        snr_db, magnification = self.get_params(['snr_db', 'magnification'])
        delta_pixel_recon, delta_det_channel = self.get_params(['delta_pixel_recon', 'delta_det_channel'])

        # Compute indicator function for sinogram support
        sino_indicator = self._get_sino_indicator(sinogram)

        # Compute RMS value of sinogram excluding empty space
        signal_rms = np.average(weights * sinogram**2, None, sino_indicator)**0.5

        # Convert snr to relative noise standard deviation
        rel_noise_std = 10**(-snr_db / 20)
        # compute the default_pixel_pitch = the detector pixel pitch in the recon plane given the magnification
        default_pixel_pitch = delta_det_channel / magnification

        # Compute the recon pixel pitch relative to the default.
        pixel_pitch_relative_to_default = delta_pixel_recon / default_pixel_pitch

        # Compute sigma_y and scale by relative pixel pitch
        sigma_y = rel_noise_std * signal_rms * (pixel_pitch_relative_to_default**0.5)
        self.set_params(no_warning=True, sigma_y=sigma_y, auto_regularize_flag=True)

    def auto_set_sigma_x(self, sinogram):
        """
        Compute the automatic value of ``sigma_x`` for use in MBIR reconstruction with qGGMRF prior.
        Args:
            sinogram (jax array): 3D jax array containing sinogram with shape (num_views, num_det_rows, num_det_channels).
        """
        sigma_x = 0.2 * self._get_estimate_of_recon_std(sinogram)
        self.set_params(no_warning=True, sigma_x=sigma_x, auto_regularize_flag=True)

    def auto_set_sigma_p(self, sinogram):
        """
        Compute the automatic value of ``sigma_p`` for use in MBIR reconstruction with proximal map prior.
        Args:
            sinogram (jax array): 3D jax array containing sinogram with shape (num_views, num_det_rows, num_det_channels).
        """
        sigma_p = 0.2 * self._get_estimate_of_recon_std(sinogram)
        self.set_params(no_warning=True, sigma_p=sigma_p, auto_regularize_flag=True)

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

    def print_params(self):
        print("----")
        for key, entry in vars(self.params).items():
            param_val = entry.get('val')
            recompile_flag = entry.get('recompile_flag')
            print("{} = {}, recompile_flag = {}".format(key, param_val, recompile_flag))
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

    def set_params(self, no_warning=False, **kwargs):
        """
        Updates parameters using keyword arguments.
        After setting parameters, it checks if key geometry-related parameters have changed and, if so, recompiles the projectors.

        Args:
            no_warning (bool, optional, default=False): This is used internally to allow for some initial parameter setting.
            **kwargs: Arbitrary keyword arguments where keys are parameter names and values are the new parameter values.

        Raises:
            NameError: If any key provided in kwargs is not a recognized parameter.
        """
        # Get initial geometry parameters
        initial_params = self.get_geometry_parameters()
        recompile = False
        regularization_parameter_change = False
        meta_parameter_change = False

        # Set all the given parameters
        for key, val in kwargs.items():
            if key in vars(self.params).keys():
                recompile_flag = getattr(self.params, key)['recompile_flag']
                new_entry = {'val': val, 'recompile_flag': recompile_flag}
                setattr(self.params, key, new_entry)
            else:
                raise NameError('"{}" not a recognized argument'.format(key))

            # Handle special cases
            if recompile_flag:
                recompile = True
            elif key in ["sigma_y", "sigma_x", "sigma_p"]:
                regularization_parameter_change = True
            elif key in ["sharpness", "snr_db"]:
                meta_parameter_change = True

        # Check for valid parameters
        self.verify_valid_params()

        # Handle case if any regularization parameter changed
        if regularization_parameter_change:
            self.set_params(auto_regularize_flag=False)
            if not no_warning:
                warnings.warn('You are directly setting regularization parameters, sigma_x, sigma_y or sigma_p. '
                              'This is an advanced feature that will disable auto-regularization.')

        # Handle case if any meta regularization parameter changed
        if meta_parameter_change:
            if self.get_params('auto_regularize_flag') is False:
                self.set_params(auto_regularize_flag=True)
                if not no_warning:
                    warnings.warn('You have re-enabled auto-regularization by setting sharpness or snr_db. '
                                  'It was previously disabled')

        # Get final geometry parameters
        new_params = self.get_geometry_parameters()

        # Compare the two outputs
        if recompile or initial_params != new_params:
            self.compile_projectors()

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
                value = getattr(self.params, parameter_names)['val']
            else:
                raise NameError('"{}" not a recognized argument'.format(parameter_names))
            return value
        values = []
        for name in parameter_names:
            if name in vars(self.params).keys():
                values.append(getattr(self.params, name)['val'])
            else:
                raise NameError('"{}" not a recognized argument'.format(name))
        return values

    def add_new_params(self, **kwargs):
        """
        Add parameters using keyword arguments.

        Args:
            **kwargs: Arbitrary keyword arguments where keys are parameter names and values are the new parameter values.

        """
        # Set all the given parameters
        for key, val in kwargs.items():
            if key in vars(self.params).keys():
                raise NameError('"{}" is an existing parameter - use set_params to set the value'.format(key))
            else:
                new_entry = {'val': val, 'recompile_flag': True}
                setattr(self.params, key, new_entry)

    def verify_valid_params(self):
        """
        Verify any conditions that must be satisfied among parameters for correct projections.
        """

        sinogram_shape = self.get_params('sinogram_shape')
        if len(sinogram_shape) != 3:
            error_message = "sinogram_shape must be (views, rows, channels). \n"
            error_message += "Got {} for sinogram shape.".format(sinogram_shape)
            raise ValueError(error_message)

    def get_geometry_parameters(self):
        return None

    def get_voxels_at_indices(self, recon, indices):
        """
        Retrieves voxel values from a reconstruction array at specified indices.

        Args:
            recon (jnp array): The 3D reconstruction array.
            indices (jnp array): Indices for which voxel values are required.

        Returns:
            numpy.ndarray or jax.numpy.DeviceArray: Array of voxel values at the specified indices.
        """
        # Get number of rows in detector
        shape = self.get_params('sinogram_shape')

        # Set number of detector rows
        num_det_rows = shape[1]

        # Retrieve values of recon at the indices locations
        voxel_values = recon.reshape((-1, num_det_rows))[indices]

        return voxel_values

    def get_forward_model_loss(self, error_sinogram, weights=1.0, normalize=True):
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
            avg_weight = jnp.average(weights)
            loss = jnp.sqrt((1.0 / (self.get_params('sigma_y')**2)) * jnp.mean(
                (error_sinogram * error_sinogram) * (weights / avg_weight)))
        else:
            loss = (1.0 / (2 * self.get_params('sigma_y')**2)) * jnp.sum((error_sinogram * error_sinogram) * weights)
        return loss

    @staticmethod
    def _get_cos_sin_angles(angles):
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
    def _get_sino_indicator(sinogram):
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

    def _get_estimate_of_recon_std(self, sinogram):
        """
        Estimate the standard deviation of the reconstruction from the sinogram.  This is used to scale sigma_p and
        sigma_x in MBIR reconstruction.

        Args:
            sinogram (jax array): 3D jax array containing sinogram with shape (num_views, num_det_rows, num_det_channels).
        """
        # Get parameters
        delta_det_channel = self.get_params('delta_det_channel')
        sharpness = self.get_params('sharpness')
        magnification = self.get_params('magnification')
        num_det_channels = sinogram.shape[-1]

        # Compute indicator function for sinogram support
        sino_indicator = self._get_sino_indicator(sinogram)

        # Compute a typical recon value by dividing average sinogram value by a typical projection path length
        typical_img_value = np.average(sinogram, weights=sino_indicator) / (
                num_det_channels * delta_det_channel / magnification)

        # Compute sigma_x as a fraction of the typical recon value
        sigma_prior = (2**sharpness) * typical_img_value
        return sigma_prior

    def recon(self, sinogram, weights=1.0):
        """
        Perform MBIR reconstruction using the Multi-Granular Vector Coordinate Descent algorithm.
        This function takes care of generating its own partitions and partition sequence.

        Args:
            sinogram (jax array): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels).
            weights (scalar or jax array): scalar or 3D positive weights with same shape as error_sinogram.

        Returns:
            [recon, fm_rmse]: reconstruction and array of loss for each iteration.
        """
        # Run auto regularization. If auto_regularize_flag is False, then this will have no effect
        self.auto_set_regularization_params(sinogram, weights=weights)

        # Generate set of voxel partitions
        partitions = self.gen_set_of_voxel_partitions()

        # Generate sequence of partitions to use
        partition_sequence = self.gen_partition_sequence()

        # Compute reconstruction
        recon, fm_rmse = self.vcd_recon(sinogram, partitions, partition_sequence, weights=weights)

        return recon, fm_rmse

    def vcd_recon(self, sinogram, partitions, partition_sequence, weights=1.0):
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
        num_recon_rows, num_recon_cols, num_recon_slices = \
            self.get_params(['num_recon_rows', 'num_recon_cols', 'num_recon_slices'])
        angles = self.get_params('angles')

        # Initialize VCD error sinogram, recon, and hessian
        error_sinogram = sinogram
        recon = jnp.zeros((num_recon_rows, num_recon_cols, num_recon_slices))
        hessian = self.compute_hessian_diagonal(weights=weights)

        # Initialize forward model normalized RMSE error array
        fm_rmse = np.zeros(num_iters)

        for i in range(num_iters):
            error_sinogram, recon = self.vcd_partition_iteration(error_sinogram, recon,
                                                                 partitions[partition_sequence[i]], hessian,
                                                                 weights=weights)
            fm_rmse[i] = self.get_forward_model_loss(error_sinogram)
            if self.get_params('verbose') >= 1:
                print(f'VCD iteration={i}; Loss={fm_rmse[i]}')

        return recon, fm_rmse

    def vcd_partition_iteration(self, error_sinogram, recon, partition, fm_hessian, weights=1.0):
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

    def vcd_subset_iteration(self, error_sinogram, recon, indices, fm_hessian, weights=1.0):
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
        # Get positivity flag
        positivity_flag = self.get_params('positivity_flag')

        # Recover recon shape parameters and make sure that recon has a 3D shape
        num_recon_rows, num_recon_cols, num_recon_slices = \
            self.get_params(['num_recon_rows', 'num_recon_cols', 'num_recon_slices'])
        recon = recon.reshape((num_recon_rows, num_recon_cols, num_recon_slices))

        # flatten the recon and hessian if they are not already flat
        fm_hessian = fm_hessian.reshape((-1, num_recon_slices))

        # Compute the forward model gradient and hessian at each pixel in the index set.
        # Assumes Loss(delta) = 1/(2 sigma_y^2) || error_sinogram - A delta ||_weights^2
        constant = 1.0 / (self.get_params('sigma_y')**2.0)

        fm_gradient = -constant * self._sparse_back_project(error_sinogram * weights, indices)
        fm_sparse_hessian = constant * fm_hessian[indices]

        # Compute the prior model gradient and hessian at each pixel in the index set.
        sigma_x, p, q, T, b = self.get_params(['sigma_x', 'p', 'q', 'T', 'b'])
        pm_gradient, pm_hessian = pm_gradient_and_hessian_at_indices(recon, indices, sigma_x, p, q, T, b)

        # Compute update vector update direction in recon domain
        delta_recon_at_indices = (- fm_gradient - pm_gradient) / (fm_sparse_hessian + pm_hessian)

        # Compute update direction in sinogram domain
        delta_sinogram = self._sparse_forward_project(delta_recon_at_indices, indices)

        # Compute "optimal" update step
        # This is really only optimal for the forward model component.
        # We can compute the truly optimal update, but it's complicated so maybe this is good enough
        alpha = jnp.sum(error_sinogram * delta_sinogram * weights) / jnp.sum(delta_sinogram * delta_sinogram * weights)

        # Flatten recon for next steps
        recon = recon.reshape((-1, num_recon_slices))

        # Enforce positivity constraint if desired
        # Greg, this may result in excess compilation. Not sure.
        if positivity_flag is True:
            # Get recon at indices
            recon_at_indices = recon[indices]

            # Clip updates to ensure non-negativity
            delta_recon_at_indices = jnp.maximum(-recon_at_indices * (1.0 / alpha), delta_recon_at_indices)

            # Recompute sinogram projection
            delta_sinogram = self._sparse_forward_project(delta_recon_at_indices, indices)

        # Perform sparse updates at index locations, and reshape as 3D array
        recon = recon.at[indices].add(alpha * delta_recon_at_indices)
        recon = recon.reshape((num_recon_rows, num_recon_cols, num_recon_slices))

        # Update sinogram
        error_sinogram = error_sinogram - alpha * delta_sinogram

        return error_sinogram, recon

    @staticmethod
    def gen_weights(sinogram, weight_type):
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
            raise Exception("gen_weights: undefined weight_type {}".format(weight_type))

        return weights

    def gen_set_of_voxel_partitions(self):
        """
        Generates a collection of voxel partitions for an array of specified partition sizes.
        This function creates a tuple of randomly generated 2D voxel partitions.
        Returns:
            tuple: A tuple of 2D arrays each representing a partition of voxels into the specified number of subsets.
        """
        # Convert granularity to an np array
        granularity = np.array(self.get_params('granularity'))
        num_recon_rows, num_recon_cols = self.get_params(['num_recon_rows', 'num_recon_cols'])
        partitions = ()
        for size in granularity:
            partition = mbirjax.gen_voxel_partition(num_recon_rows, num_recon_cols, size)
            partitions += (partition,)

        return partitions

    def gen_full_indices(self):
        """
        Generates a full array of voxels in the region of reconstruction.
        This is useful for computing forward projections.
        """
        # Convert granularity to an np array
        num_recon_rows, num_recon_cols = self.get_params(['num_recon_rows', 'num_recon_cols'])
        partition = mbirjax.gen_voxel_partition(num_recon_rows, num_recon_cols, num_subsets=1)
        full_indices = partition[0]

        return full_indices

    def gen_partition_sequence(self):
        # Get sequence from params and convert it to a np array
        partition_sequence = np.array(self.get_params('partition_sequence'))

        # Tile sequence so it at least iterations long
        num_iterations = self.get_params('num_iterations')
        extended_partition_sequence = np.tile(partition_sequence, (num_iterations // partition_sequence.size + 1))[
                                      0:num_iterations]
        return extended_partition_sequence

    def gen_3d_sl_phantom(self):
        """
        Generates a 3D Shepp-Logan phantom.
        Returns:
            ndarray: A 3D numpy array of shape specified by TomographyModel class parameters.
        """
        num_recon_rows, num_recon_cols, num_recon_slices = \
            self.get_params(['num_recon_rows', 'num_recon_cols', 'num_recon_slices'])
        phantom = mbirjax.generate_3d_shepp_logan(num_recon_rows, num_recon_cols, num_recon_slices)
        return phantom

    def reshape_recon(self, recon):
        """
        Reshape recon into its 3D form
        """
        num_recon_rows, num_recon_cols, num_recon_slices = \
            self.get_params(['num_recon_rows', 'num_recon_cols', 'num_recon_slices'])
        return recon.reshape(num_recon_rows, num_recon_cols, num_recon_slices)


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
    btilde = _get_btilde(delta_prime, b, sigma_x, p, q, T)

    # Compute first derivative
    pm_first_derivative = jnp.sum(2 * btilde * delta_prime, axis=-1)

    # Compute second derivative
    pm_second_derivative = jnp.sum(2 * btilde, axis=-1)

    return pm_first_derivative, pm_second_derivative


@jax.jit
def pm_gradient_and_hessian_at_indices(recon, indices, sigma_x, p, q, T, b):
    """
    Calculate the gradient and hessian at each index location in a reconstructed image using the qGGMRF prior.

    Args:
        recon (jax.array): 3D reconstructed image array with shape (num_recon_rows, num_recon_cols, num_recon_slices).
        indices (int array): Array of shape (N_indices, num_recon_slices) representing the indices of voxels in a flattened array to be updated.
        sigma_x (float): Standard deviation parameter of the qGGMRF prior.
        p (float): Norm parameter p in the qGGMRF prior.
        q (float): Norm parameter q in the qGGMRF prior.
        T (float): Scaling parameter in the qGGMRF prior.
        b (list of 6 float): list of 6 qGGMRF prior neightborhood weights.

    Returns:
        tuple: Contains two arrays (first_derivative, second_derivative) each of shape (N_indices, num_recon_slices)
               representing the gradient and Hessian values at specified indices.
    """
    # Initialize the neighborhood weights for averaging surrounding pixel values.
    # Order is (I think) [row+1, row-1, col+1, col-1, slice+1, slice-1]
    b = jnp.array(b).reshape(1, -1)
    b /= jnp.sum(b)

    # Extract the shape of the reconstruction array.
    num_rows, num_cols, num_slices = recon.shape

    # Convert flat indices to 2D indices for row and column access.
    row_index, col_index = jnp.unravel_index(indices, shape=(num_rows, num_cols))

    # Access the central voxels' values at the given indices. Shape of xs is (num indices)x(num slices)
    xs = recon[row_index, col_index]

    # Define relative positions for accessing neighborhood voxels.
    offsets = [[1, 0], [-1, 0], [0, 1], [0, -1]]

    # Create a list derived from 4 neighbors, each entry in list is a 1D vector containing the recon values at that location over all slices.
    xr = [recon[row_index + offset[0], col_index + offset[1]] for offset in offsets]

    # Shift slices up with zero padding
    xs_up = jnp.roll(xs, shift=-1, axis=1).at[:, -1].set(0)

    # Shift slices down with zero padding
    xs_down = jnp.roll(xs, shift=1, axis=1).at[:, 0].set(0)

    # Append the up-down differences to xr
    xr = xr + [xs_up, xs_down]

    # Convert to a jnp array with shape (num index elements)x(num slices)x(6 neighbors).
    x_neighbors_prime_new = jnp.stack(xr, axis=-1)

    # Compute differences between the central voxels and their neighbors. (BTW, xs is broadcast.)
    delta_prime = xs[:, :, jnp.newaxis] - x_neighbors_prime_new

    # Reshape delta_prime for processing with the qGGMRF prior.
    delta_prime = delta_prime.reshape((-1, b.shape[-1]))

    # Compute the first and second derivatives using the qGGMRF model.
    first_derivative, second_derivative = pm_gradient_and_hessian(delta_prime, b, sigma_x, p, q, T)

    # Reshape outputs to match the number of indices and slices.
    first_derivative = first_derivative.reshape(-1, num_slices)
    second_derivative = second_derivative.reshape(-1, num_slices)

    return first_derivative, second_derivative


def _get_rho(delta, b, sigma_x, p, q, T):
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
    first_term = ((delta / sigma_x)**p) / p
    second_term = (delta / (T * sigma_x))**(q - p)
    third_term = second_term / (1 + second_term)

    result = np.sum(b * (first_term * third_term), axis=-1)  # Broadcast b over batch_size dimension
    return result


@jax.jit
def _get_btilde(delta_prime, b, sigma_x, p, q, T):
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
    first_term = (delta_prime**(q - 2.0)) / (2.0 * (sigma_x**p) * (T * sigma_x**(q - p)))

    # third_term is the third term in formula.
    second_term = (delta_prime / (T * sigma_x))**(q - p)
    third_term = ((q / p) + second_term) / ((1 + second_term)**2.0)

    result = b * (first_term * third_term)  # Broadcast b over batch_size dimension
    return result


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
