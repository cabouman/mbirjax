import types
import numpy as np
import warnings
import gc
import jax
import jax.numpy as jnp
import mbirjax
from mbirjax import ParameterHandler
from functools import partial
from collections import namedtuple


class TomographyModel(ParameterHandler):
    """
    Represents a general model for tomographic reconstruction using MBIRJAX. This class encapsulates the parameters and
    methods for the forward and back projection processes required in tomographic imaging.

    Note that this class is a template for specific subclasses.  TomographyModel by itself does not implement
    projectors or recon.  Use self.print_params() to print the parameters of the model after initialization.

    Args:
        sinogram_shape (tuple): The shape of the sinogram array expected (num_views, num_det_rows, num_det_channels).
        recon_shape (tuple): The shape of the reconstruction array (num_rows, num_cols, num_slices).
        **kwargs (dict): Arbitrary keyword arguments for setting model parameters dynamically.
            See the full list of parameters and their descriptions at :ref:`detailed-parameter-docs`.

    Sets up the reconstruction size and parameters.
    """

    def __init__(self, sinogram_shape, recon_shape=None, **kwargs):

        super().__init__()
        self.sparse_forward_project, self.sparse_back_project = None, None  # These are callable functions compiled in set_params
        self.compute_hessian_diagonal = None
        self.set_params(no_compile=True, no_warning=True, sinogram_shape=sinogram_shape, recon_shape=recon_shape, **kwargs)
        delta_voxel = self.get_params('delta_voxel')
        if delta_voxel is None:
            magnification = self.get_magnification()
            delta_det_channel = self.get_params('delta_det_channel')
            delta_voxel = delta_det_channel / magnification
            self.set_params(no_compile=True, no_warning=True, delta_voxel=delta_voxel)
        if recon_shape is None:
            self.auto_set_recon_size(sinogram_shape, no_compile=True, no_warning=True)

        self.set_params(geometry_type=str(type(self)))
        self.verify_valid_params()
        self.create_projectors()

    @classmethod
    def from_file(cls, filename):
        """
        Construct a TomographyModel (or a subclass) from parameters saved using to_file()

        Args:
            filename (str): Name of the file containing parameters to load.

        Returns:
            ConeBeamModel with the specified parameters.
        """
        # Load the parameters and convert to use the ConeBeamModel keywords.
        params = ParameterHandler.load_param_dict(filename, values_only=True)
        return cls(**params)

    def to_file(self, filename):
        """
        Save parameters to yaml file.

        Args:
            filename (str): Path to file to store the parameter dictionary.  Must end in .yml or .yaml

        Returns:
            Nothing but creates or overwrites the specified file.
        """
        self.save_params(filename)

    def create_projectors(self):
        """
        Creates an instance of the Projectors class and set the local instance variables needed for forward
        and back projection and compute_hessian_diagonal.  This method requires that the current geometry has
        implementations of :meth:`forward_project_pixel_batch_to_one_view` and :meth:`back_project_one_view_to_pixel_batch`

        Returns:
            Nothing, but creates jit-compiled functions.
        """
        projector_functions = mbirjax.Projectors(self)
        self.sparse_forward_project = projector_functions.sparse_forward_project
        self.sparse_back_project = projector_functions.sparse_back_project
        self.compute_hessian_diagonal = projector_functions.compute_hessian_diagonal

    @staticmethod
    def forward_project_pixel_batch_to_one_view(voxel_values, pixel_indices, view_params, projector_params):
        """
        Forward project a set of voxels determined by indices into the flattened array of size num_rows x num_cols.

        Note:
            This method must be overridden for a specific geometry.

        Args:
            voxel_values (jax array):  2D array of shape (num_indices, num_slices) of voxel values, where
                voxel_values[i, j] is the value of the voxel in slice j at the location determined by indices[i].
            pixel_indices (jax array of int):  1D vector of indices into flattened array of size num_rows x num_cols.
            view_params (jax array):  A 1D array of view-specific parameters (such as angle) for the current view.
            projector_params (namedtuple):  Tuple containing (sinogram_shape, recon_shape, get_geometry_params())

        Returns:
            jax array of shape (num_det_rows, num_det_channels)
        """
        warnings.warn('Forward projector not implemented for TomographyModel.')
        return None

    @staticmethod
    def back_project_one_view_to_pixel_batch(sinogram_view, pixel_indices, single_view_params, projector_params, coeff_power=1):
        """
        Calculate the backprojection value at a specified recon voxel cylinder given a sinogram view and parameters.

        Note:
            This method must be overridden for a specific geometry.

        Args:
            sinogram_view (jax array): one view of the sinogram to be back projected
            pixel_indices (jax array of int):  1D vector of indices into flattened array of size num_rows x num_cols.
            single_view_params (jax array): A 1D array of view-specific parameters (such as angle) for the current view.
            projector_params (namedtuple):  Tuple containing (sinogram_shape, recon_shape, get_geometry_params())
            coeff_power (int): backproject using the coefficients of (A_ij ** coeff_power).
                Normally 1, but should be 2 for compute_hessian_diagonal.

        Returns:
            The value of the voxel for all slices at the input index (i.e., a voxel cylinder) obtained by backprojecting
            the input sinogram view.
        """
        warnings.warn('Back projector not implemented for TomographyModel.')
        return None

    def forward_project(self, recon):
        """
        Perform a full forward projection at all voxels in the field-of-view.

        Note:
            This method should generally not be used directly for iterative reconstruction.  For iterative
            reconstruction, use :meth:`recon`.

        Args:
            recon (jnp array): The 3D reconstruction array.
        Returns:
            jnp array: The resulting 3D sinogram after projection.
        """
        recon_shape = self.get_params('recon_shape')
        full_indices = mbirjax.gen_full_indices(recon_shape)
        voxel_values = self.get_voxels_at_indices(recon, full_indices)
        sinogram = self.sparse_forward_project(voxel_values, full_indices)

        return sinogram

    def back_project(self, sinogram):
        """
        Perform a full back projection at all voxels in the field-of-view.

        Note:
            This method should generally not be used directly for iterative reconstruction.  For iterative
            reconstruction, use :meth:`recon`.

        Args:
            sinogram (jnp array): 3D jax array containing sinogram.
        Returns:
            jnp array: The resulting 3D sinogram after projection.
        """
        recon_shape = self.get_params('recon_shape')
        full_indices = mbirjax.gen_full_indices(recon_shape)
        recon_cylinder = self.sparse_back_project(sinogram, full_indices)
        row_index, col_index = jnp.unravel_index(full_indices, recon_shape[:2])
        recon = jnp.zeros(recon_shape)
        recon = recon.at[row_index, col_index].set(recon_cylinder)
        return recon

    def sparse_forward_project(self, voxel_values, indices, view_indices=()):
        """
        Forward project the given voxel values to a sinogram.
        The indices are into a flattened 2D array of shape (recon_rows, recon_cols), and the projection is done using
        all voxels with those indices across all the slices.

        Args:
            voxel_values (jax.numpy.DeviceArray): 2D array of voxel values to project, size (len(pixel_indices), num_recon_slices).
            indices (numpy.ndarray): Array of indices specifying which voxels to project.
            view_indices (ndarray or jax array, optional): 1D array of indices into the view parameters array.
                If None, then all views are used.

        Returns:
            jnp array: The resulting 3D sinogram after projection.
        """
        sinogram = self.sparse_forward_project(voxel_values, indices, view_indices=view_indices).block_until_ready()
        gc.collect()
        return sinogram

    def sparse_back_project(self, sinogram, indices, view_indices=()):
        """
        Back project the given sinogram to the voxels given by the indices.
        The indices are into a flattened 2D array of shape (recon_rows, recon_cols), and the projection is done using
        all voxels with those indices across all the slices.

        Args:
            sinogram (jnp array): 3D jax array containing sinogram.
            indices (jnp array): Array of indices specifying which voxels to back project.
            view_indices (ndarray or jax array, optional): 1D array of indices into the view parameters array.
                If None, then all views are used.

        Returns:
            A jax array of shape (len(indices), num_slices)
        """
        recon_at_indices = self.sparse_back_project(sinogram, indices, view_indices=view_indices).block_until_ready()
        gc.collect()
        return recon_at_indices

    def compute_hessian_diagonal(self, weights=None, view_indices=()):
        """
        Computes the diagonal elements of the Hessian matrix for given weights.

        Args:
            weights (jax array, optional): 3D positive weights with same shape as sinogram.  Defaults to all 1s.
            view_indices (ndarray or jax array, optional): 1D array of indices into the view parameters array.
                If None, then all views are used.

        Returns:
            jnp array: Diagonal of the Hessian matrix with same shape as recon.
        """
        hessian = self.compute_hessian_diagonal(weights, view_indices=view_indices).block_until_ready()
        gc.collect()
        return hessian

    def set_params(self, no_warning=False, no_compile=False, **kwargs):
        """
        Updates parameters using keyword arguments.
        After setting parameters, it checks if key geometry-related parameters have changed and, if so, recompiles the projectors.

        Args:
            no_warning (bool, optional, default=False): This is used internally to allow for some initial parameter setting.
            no_compile (bool, optional, default=False): Prevent (re)compiling the projectors.  Used for initialization.
            **kwargs: Arbitrary keyword arguments where keys are parameter names and values are the new parameter values.

        Raises:
            NameError: If any key provided in kwargs is not a recognized parameter.
        """
        recompile_flag = super().set_params(no_warning=no_warning, no_compile=no_compile, **kwargs)
        if recompile_flag:
            self.create_projectors()

    def auto_set_regularization_params(self, sinogram, weights=None):
        """
        Automatically sets the regularization parameters (self.sigma_y, self.sigma_x, and self.sigma_p) used in MBIR reconstruction based on the provided sinogram and optional weights.

        Args:
            sinogram (jnp.array): 3D jax array containing the sinogram with shape (num_views, num_det_rows, num_det_channels).
            weights (jnp.array, optional): 3D weights array with the same shape as the sinogram. Defaults to all 1s.

        Returns:
            namedtuple containing the parameters sigma_y, sigma_x, sigma_p

        The method adjusts the regularization parameters only if `auto_regularize_flag` is set to True within the model's parameters.
        """
        if self.get_params('auto_regularize_flag'):
            self.auto_set_sigma_y(sinogram, weights)
            self.auto_set_sigma_x(sinogram)
            self.auto_set_sigma_p(sinogram)

        regularization_param_names = ['sigma_y', 'sigma_x', 'sigma_p']
        RegularizationParams = namedtuple('RegularizationParams', regularization_param_names)
        regularization_param_values = [float(val) for val in self.get_params(regularization_param_names)]
        regularization_params = RegularizationParams(*tuple(regularization_param_values))

        return regularization_params

    def auto_set_sigma_y(self, sinogram, weights=None):
        """
        Sets the value of the parameter sigma_y used for use in MBIR reconstruction.

        Args:
            sinogram (jax array): 3D jax array containing sinogram with shape (num_views, num_det_rows, num_det_channels).
            weights (jax array, optional): 3D positive weights with same shape as sinogram.  Defaults to all 1s.
        """

        # Get parameters
        snr_db = self.get_params('snr_db')
        magnification = self.get_magnification()
        delta_voxel, delta_det_channel = self.get_params(['delta_voxel', 'delta_det_channel'])

        # Compute indicator function for sinogram support
        sino_indicator = self._get_sino_indicator(sinogram)

        # Compute RMS value of sinogram excluding empty space
        if weights is None:  # For this function, we don't need a full sinogram of 1s for default weights
            weights = 1
        signal_rms = jnp.average(weights * sinogram ** 2, None, sino_indicator) ** 0.5

        # Convert snr to relative noise standard deviation
        rel_noise_std = 10 ** (-snr_db / 20)
        # compute the default_pixel_pitch = the detector pixel pitch in the recon plane given the magnification
        default_pixel_pitch = delta_det_channel / magnification

        # Compute the recon pixel pitch relative to the default.
        pixel_pitch_relative_to_default = delta_voxel / default_pixel_pitch

        # Compute sigma_y and scale by relative pixel pitch
        sigma_y = rel_noise_std * signal_rms * (pixel_pitch_relative_to_default ** 0.5)
        self.set_params(no_warning=True, sigma_y=sigma_y, auto_regularize_flag=True)

    def auto_set_sigma_x(self, sinogram):
        """
        Compute the automatic value of ``sigma_x`` for use in MBIR reconstruction with qGGMRF prior.

        Args:
            sinogram (jax array): 3D jax array containing sinogram with shape (num_views, num_det_rows, num_det_channels).
        """
        # Get parameters
        sharpness = self.get_params('sharpness')
        recon_std = self._get_estimate_of_recon_std(sinogram)

        # Compute sigma_x as a fraction of the typical recon value
        # 0.2 is an empirically determined constant
        sigma_x = 0.2 * (2 ** sharpness) * recon_std
        self.set_params(no_warning=True, sigma_x=sigma_x, auto_regularize_flag=True)

    def auto_set_sigma_p(self, sinogram):
        """
        Compute the automatic value of ``sigma_p`` for use in MBIR reconstruction with proximal map prior.

        Args:
            sinogram (jax array): 3D jax array containing sinogram with shape (num_views, num_det_rows, num_det_channels).
        """
        # Get parameters
        sharpness = self.get_params('sharpness')
        recon_std = self._get_estimate_of_recon_std(sinogram)

        # Compute sigma_x as a fraction of the typical recon value
        # 0.2 is an empirically determined constant
        sigma_p = 0.2 * (2 ** sharpness) * recon_std
        self.set_params(no_warning=True, sigma_p=sigma_p, auto_regularize_flag=True)

    def auto_set_recon_size(self, sinogram_shape, no_compile=True, no_warning=False):
        """Compute the default recon size using the internal parameters delta_channel and delta_pixel plus
          the number of channels from the sinogram"""
        raise NotImplementedError('auto_set_recon_size must be implemented by each specific geometry model.')

    def get_voxels_at_indices(self, recon, indices):
        """
        Retrieves voxel values from a reconstruction array at specified indices.
        The indices are into a flattened 2D array of shape (recon_rows, recon_cols), and the values are retrieved
        using all voxels with those indices across all the slices.

        Args:
            recon (jnp array): The 3D reconstruction array.
            indices (numpy.ndarray): Array of indices specifying which voxels to project.

        Returns:
            numpy.ndarray or jax.numpy.DeviceArray: Array of voxel values at the specified indices.
        """
        recon_shape = self.get_params('recon_shape')

        # Flatten the recon along the first two dimensions, then retrieve values of recon at the indices locations
        voxel_values = recon.reshape((-1,) + recon_shape[2:])[indices]

        return voxel_values

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
        delta_voxel = self.get_params('delta_voxel')
        recon_shape = self.get_params('recon_shape')
        magnification = self.get_magnification()
        num_det_channels = sinogram.shape[-1]

        # Compute the typical magnitude of a sinogram value
        sino_indicator = self._get_sino_indicator(sinogram)
        typical_sinogram_value = jnp.average(jnp.abs(sinogram), weights=sino_indicator)

        # TODO: Can we replace this with some type of approximate operator norm of A? That would make it universal.
        # Compute a typical projection path length based on the soft minimum of the recon width and height
        typical_path_length_space = (2*recon_shape[0] * recon_shape[1])/(recon_shape[0] + recon_shape[1])*delta_voxel

        # Compute a typical projection path length based on the detector column width
        typical_path_length_sino = num_det_channels * delta_det_channel / magnification

        # Compute a typical projection path as the minimum of the two estimates
        typical_path_length = jnp.minimum(typical_path_length_space, typical_path_length_sino)

        # Compute a typical recon value by dividing average sinogram value by a typical projection path length
        recon_std = typical_sinogram_value / typical_path_length

        return recon_std

    def recon(self, sinogram, weights=None, num_iterations=15, first_iteration=0, init_recon=None):
        """
        Perform MBIR reconstruction using the Multi-Granular Vector Coordinate Descent algorithm.
        This function takes care of generating its own partitions and partition sequence.
        TO restart a recon using the same partition sequence, set first_iteration to be the number of iterations
        completed so far, and set init_recon to be the output of the previous recon.  This will continue using
        the same partition sequence from where the previous recon left off.

        Args:
            sinogram (jax array): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels).
            weights (jax array, optional): 3D positive weights with same shape as error_sinogram.  Defaults to all 1s.
            num_iterations (int, optional): number of iterations of the VCD algorithm to perform.
            first_iteration (int, optional): Set this to be the number of iterations previously completed when restarting a recon using init_recon.
            init_recon (jax array, optional): Optional reconstruction to be used for initialization.

        Returns:
            [recon, recon_params]: reconstruction and a named tuple containing the recon parameters.
            recon_params (namedtuple): num_iterations, granularity, partition_sequence, fm_rmse, regularization_params
        """
        # Run auto regularization. If auto_regularize_flag is False, then this will have no effect
        regularization_params = self.auto_set_regularization_params(sinogram, weights=weights)

        # Generate set of voxel partitions
        recon_shape, granularity = self.get_params(['recon_shape', 'granularity'])
        partitions = mbirjax.gen_set_of_pixel_partitions(recon_shape, granularity)

        # Generate sequence of partitions to use
        partition_sequence = self.get_params('partition_sequence')
        partition_sequence = mbirjax.gen_partition_sequence(partition_sequence, num_iterations=num_iterations)
        partition_sequence = partition_sequence[first_iteration:]

        # Compute reconstruction
        recon, fm_rmse = self.vcd_recon(sinogram, partitions, partition_sequence, weights=weights,
                                        init_recon=init_recon)

        # Return num_iterations, granularity, partition_sequence, fm_rmse values, regularization_params
        recon_param_names = ['num_iterations', 'granularity', 'partition_sequence', 'fm_rmse',
                             'regularization_params']
        ReconParams = namedtuple('ReconParams', recon_param_names)
        partition_sequence = [int(val) for val in partition_sequence]
        fm_rmse = [float(val) for val in fm_rmse]
        recon_param_values = [num_iterations, granularity, partition_sequence, fm_rmse, regularization_params._asdict()]
        recon_params = ReconParams(*tuple(recon_param_values))

        return recon, recon_params

    def vcd_recon(self, sinogram, partitions, partition_sequence, weights=None, init_recon=None, prox_input=None):
        """
        Perform MBIR reconstruction using the Multi-Granular Vector Coordinate Descent algorithm
        for a given set of partitions and a prescribed partition sequence.

        Args:
            sinogram (jax array): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels).
            partitions (tuple): A collection of K partitions, with each partition being an (N_indices) integer index array of voxels to be updated in a flattened recon.
            partition_sequence (jax array): A sequence of integers that specify which partition should be used at each iteration.
            weights (jax array, optional): 3D positive weights with same shape as error_sinogram.  Defaults to all 1s.
            init_recon (jax array, optional): Initial reconstruction to use in reconstruction.
            prox_input (jax array, optional): Reconstruction to be used as input to a proximal map.

        Returns:
            [recon, fm_rmse]: 3D reconstruction and array of loss for each iteration.
        """
        # Get required parameters
        num_iters = partition_sequence.size
        recon_shape = self.get_params('recon_shape')

        if weights is None:
            weights = jnp.ones_like(sinogram)

        if init_recon is None:
            # Initialize VCD recon, and error sinogram
            recon = jnp.zeros(recon_shape)
            error_sinogram = sinogram
        else:
            # Make sure that init_recon has the correct shape and type
            if init_recon.shape != recon_shape:
                error_message = "init_recon does not have the correct shape. \n"
                error_message += "Expected {}, but got shape {} for init_recon shape.".format(recon_shape,
                                                                                              init_recon.shape)
                raise ValueError(error_message)

            # Initialize VCD recon, and error sinogram
            recon = jnp.array(init_recon)
            error_sinogram = sinogram - self.forward_project(recon)

        # Test to make sure the prox_input input is correct
        if prox_input is not None:
            # Make sure that prox_input has the correct size
            if prox_input.shape != recon.shape:
                error_message = "prox_input does not have the correct size. \n"
                error_message += "Expected {}, but got shape {} for prox_input shape.".format(recon.shape,
                                                                                              prox_input.shape)
                raise ValueError(error_message)

            # If used, make sure that prox_input has the correct a 3D shape and type
            prox_input = jnp.array(prox_input.reshape(recon.shape))

        # Initialize the diagonal of the hessian of the forward model
        num_recon_slices = recon_shape[2]
        fm_hessian = self.compute_hessian_diagonal(weights=weights).reshape((-1, num_recon_slices))

        # Initialize the emtpy recon
        flat_recon = recon.reshape((-1, num_recon_slices))

        # Create the finer grained recon update operators
        vcd_subset_iterator = self.create_vcd_subset_iterator(fm_hessian, weights=weights, prox_input=prox_input)
        vcd_partition_iterator = TomographyModel.create_vcd_partition_iterator(vcd_subset_iterator)

        verbose, sigma_y = self.get_params(['verbose', 'sigma_y'])

        if verbose >= 1:
            print('Starting VCD iterations')
            if verbose >= 2:
                mbirjax.get_memory_stats()
                print('--------')

        # Do the iterations
        fm_rmse = np.zeros(num_iters)
        for i in range(num_iters):
            partition = partitions[partition_sequence[i]]
            subset_indices = np.random.permutation(partition.shape[0])
            error_sinogram, flat_recon = vcd_partition_iterator([error_sinogram, flat_recon, partition], subset_indices)
            fm_rmse[i] = self.get_forward_model_loss(error_sinogram, sigma_y, weights)
            if verbose >= 1:
                print(f'VCD iteration={i}; Loss={fm_rmse[i]}')
                es_rmse = jnp.linalg.norm(error_sinogram) / jnp.sqrt(error_sinogram.size)
                print(f'Error sinogram RMSE = {es_rmse}')
                if verbose >= 2:
                    mbirjax.get_memory_stats()
                    print('--------')

        return self.reshape_recon(flat_recon), fm_rmse

    @staticmethod
    def create_vcd_partition_iterator(vcd_subset_iterator):
        """
        Create a jit-compiled function to update all the pixels in the recon and error sinogram by applying
        the supplied vcd_subset_iterator to each subset in a partition.

        Args:
            vcd_subset_iterator (callable):  The function returned by create_vcd_subset_iterator.

        Returns:
            (callable) vcd_partition_iterator(error_sinogram, flat_recon, partition, subset_indices
        """

        def vcd_partition_iterator(sinogram_recon_partition, subset_indices):
            """
            Calculate a full iteration of the VCD algorithm by scanning over the subsets of the partition.
            Each iteration of the algorithm should return a better reconstructed recon.
            The error_sinogram should always be:  error_sinogram = measured_sinogram - forward_proj(recon)
            where measured_sinogram is the measured sinogram and recon is the current reconstruction.

            Args:
                sinogram_recon_partition (list): 3 element tuple containing

                    * error_sinogram (jax array): 3D error sinogram with shape (num_views, num_det_rows, num_det_channels).
                    * flat_recon (jax array): 2D array reconstruction with shape (num_recon_rows x num_recon_cols, num_recon_slices).
                    * partition (jax array): 2D array where partition[subset_index] gives a 1D array of pixel indices.

                subset_indices (jax array): An array of indices into the partition - this gives the order in which the subsets are updated.

            Returns:
                [error_sinogram, flat_recon]: Both have the same shape as above, but are updated to reduce overall loss function.
            """

            # Scan over the subsets of the partition, using the subset_indices to order them.
            sinogram_recon_partition, _ = jax.lax.scan(vcd_subset_iterator, sinogram_recon_partition, subset_indices)

            error_sinogram, flat_recon, _ = sinogram_recon_partition
            return error_sinogram, flat_recon

        return jax.jit(vcd_partition_iterator)

    def create_vcd_subset_iterator(self, fm_hessian, weights=None, prox_input=None):
        """
        Create a jit-compiled function to update a subset of pixels in the recon and error sinogram.

        Args:
            fm_hessian (jax array): Array with same shape as recon containing diagonal of hessian for forward model loss.
            weights (jax array, optional): 3D positive weights with same shape as sinogram.  Defaults to all 1s.
            prox_input (jax array): optional input for proximal map with same shape as reconstruction.

        Returns:
            (callable) vcd_subset_iterator(error_sinogram, flat_recon, pixel_indices) that updates the recon.
        """

        positivity_flag = self.get_params('positivity_flag')
        fm_constant = 1.0 / (self.get_params('sigma_y') ** 2.0)
        sigma_x, p, q, T, b = self.get_params(['sigma_x', 'p', 'q', 'T', 'b'])
        sigma_p = self.get_params('sigma_p')
        pixel_batch_size = self.get_params('pixel_batch_size')
        recon_shape = self.get_params('recon_shape')
        sparse_back_project = self.sparse_back_project
        sparse_forward_project = self.sparse_forward_project

        def vcd_subset_iterator(sinogram_recon_partition, subset_index):
            """
            Calculate an iteration of the VCD algorithm on a single subset of the partition
            Each iteration of the algorithm should return a better reconstructed recon.
            The combination of (error_sinogram, recon) forms an overcomplete state that makes computation efficient.
            However, it is important that at each application the state should meet the constraint that:
            error_sinogram = measured_sinogram - forward_proj(recon)
            where measured_sinogram forward_proj() is whatever forward projection is being used in reconstruction.

            Args:
                sinogram_recon_partition (list): 3 element tuple containing

                    * error_sinogram (jax array): 3D error sinogram with shape (num_views, num_det_rows, num_det_channels).
                    * flat_recon (jax array): 2D array reconstruction with shape (num_recon_rows x num_recon_cols, num_recon_slices).
                    * partition (jax array): 2D array where partition[subset_index] gives a 1D array of pixel indices.

                subset_index (int): integer index of the subset within the partition.

            Returns:
                [error_sinogram, flat_recon]: Both have the same shape as above, but are updated to reduce overall loss function.
            """
            error_sinogram, flat_recon, partition = sinogram_recon_partition
            pixel_indices = partition[subset_index]

            def delta_recon_batch(index_batch):
                # Compute the forward model gradient and hessian at each pixel in the index set.
                # Assumes Loss(delta) = 1/(2 sigma_y^2) || error_sinogram - A delta ||_weights^2
                fm_gradient = -fm_constant * sparse_back_project(error_sinogram * weights, index_batch)
                fm_sparse_hessian = fm_constant * fm_hessian[index_batch]

                # Compute the prior model gradient and hessian (i.e., second derivative) terms
                if prox_input is None:
                    # This is for the qGGMRF prior - compute the prior model gradient and hessian at each pixel in the index set.
                    pm_gradient, pm_hessian = pm_qggmrf_gradient_and_hessian_at_indices(flat_recon, recon_shape, index_batch, sigma_x, p, q, T, b)
                else:
                    # This is for the proximal map prior - compute the prior model gradient at each pixel in the index set.
                    pm_hessian = sigma_p ** 2
                    pm_gradient = pm_prox_gradient_at_indices(flat_recon, prox_input, index_batch, sigma_p)

                # Compute update vector update direction in recon domain
                delta_recon_at_indices_batch = (- fm_gradient - pm_gradient) / (fm_sparse_hessian + pm_hessian)
                return delta_recon_at_indices_batch

            # Apply the function on a single batch of pixels if the batch is small enough
            num_pixels = len(pixel_indices)
            max_index_subsets = 10
            cur_pixel_batch_size = max(pixel_batch_size, flat_recon.shape[0] // (max_index_subsets - 1))
            if cur_pixel_batch_size >= num_pixels:
                delta_recon_at_indices = delta_recon_batch(pixel_indices)

            # Otherwise batch the pixels and map over the batches.
            else:
                num_batches = num_pixels // cur_pixel_batch_size
                length_of_batches = num_batches * cur_pixel_batch_size
                pixel_indices_batched = jnp.reshape(pixel_indices[:length_of_batches], (num_batches, cur_pixel_batch_size))

                batched_voxel_values = jax.lax.map(delta_recon_batch, pixel_indices_batched)
                delta_recon_at_indices = batched_voxel_values.reshape((length_of_batches,) + batched_voxel_values.shape[2:])

                # Add in any leftover pixels
                num_remaining = num_pixels - num_batches * cur_pixel_batch_size
                if num_remaining > 0:
                    end_batch_indices = pixel_indices[-num_remaining:]
                    end_batch_voxel_values = delta_recon_batch(end_batch_indices)
                    delta_recon_at_indices = jnp.concatenate((delta_recon_at_indices, end_batch_voxel_values), axis=0)

            # Compute update direction in sinogram domain
            delta_sinogram = sparse_forward_project(delta_recon_at_indices, pixel_indices)

            # Compute "optimal" update step
            # This is really only optimal for the forward model component.
            # We can compute the truly optimal update, but it's complicated so maybe this is good enough
            alpha = jnp.sum(error_sinogram * delta_sinogram * weights) / (
                        jnp.sum(delta_sinogram * delta_sinogram * weights) + jnp.finfo(np.float32).eps)
            alpha = jnp.clip(alpha, jnp.finfo(np.float32).eps, 1)
            # TODO: test for alpha<0 and terminate.

            # Enforce positivity constraint if desired
            # Greg, this may result in excess compilation. Not sure.
            if positivity_flag is True:
                # Get recon at index_batch
                recon_at_indices = flat_recon[pixel_indices]

                # Clip updates to ensure non-negativity
                pos_constant = 1.0 / (alpha + jnp.finfo(np.float32).eps)
                delta_recon_at_indices = jnp.maximum(-pos_constant * recon_at_indices, delta_recon_at_indices)

                # Recompute sinogram projection
                delta_sinogram = sparse_forward_project(delta_recon_at_indices, pixel_indices)

            # Perform sparse updates at index locations
            flat_recon = flat_recon.at[pixel_indices].add(alpha * delta_recon_at_indices)

            # Update sinogram
            error_sinogram = error_sinogram - alpha * delta_sinogram

            return [error_sinogram, flat_recon, partition], None

        return jax.jit(vcd_subset_iterator)

    @staticmethod
    def get_forward_model_loss(error_sinogram, sigma_y, weights=None, normalize=True):
        """
        Calculate the loss function for the forward model from the error_sinogram and weights.
        The error sinogram should be error_sinogram = measured_sinogram - forward_proj(recon)

        Args:
            error_sinogram (jax array): 3D error sinogram with shape (num_views, num_det_rows, num_det_channels).
            sigma_y (float): Estimate obtained from auto_set_sigma_y or get_params('sigma_y')
            weights (jax array, optional): 3D positive weights with same shape as sinogram.  Defaults to all 1s.
            normalize (bool, optional, default=True):  If true, then

        Returns:
            float loss.
        """
        if normalize:
            avg_weight = 1 if weights is None else jnp.average(weights)
            loss = jnp.sqrt((1.0 / (sigma_y ** 2)) * jnp.mean(
                (error_sinogram * error_sinogram) * (weights / avg_weight)))
        else:
            loss = (1.0 / (2 * sigma_y ** 2)) * jnp.sum((error_sinogram * error_sinogram) * weights)
        return loss

    def prox_map(self, prox_input, sinogram, weights=None, num_iterations=3, init_recon=None):
        """
        Proximal Map function for use in Plug-and-Play applications.
        This function is similar to recon, but it essentially uses a prior with a mean of prox_input and a standard deviation of sigma_p.

        Args:
            prox_input (jax array): proximal map input with same shape as reconstruction.
            sinogram (jax array): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels).
            weights (jax array, optional): 3D positive weights with same shape as sinogram.  Defaults to all 1s.
            num_iterations (int, optional): number of iterations of the VCD algorithm to perform.
            init_recon (jax array, optional): optional reconstruction to be used for initialization.

        Returns:
            [recon, fm_rmse]: reconstruction and array of loss for each iteration.
        """
        # TODO:  Refactor to operate on a subset of pixels and to use previous state
        # Generate set of voxel partitions
        recon_shape, granularity = self.get_params(['recon_shape', 'granularity'])
        partitions = mbirjax.gen_set_of_pixel_partitions(recon_shape, granularity)

        # Generate sequence of partitions to use
        partition_sequence = self.get_params('partition_sequence')
        partition_sequence = mbirjax.gen_partition_sequence(partition_sequence, num_iterations=num_iterations)

        # Compute reconstruction
        recon, fm_rmse = self.vcd_recon(sinogram, partitions, partition_sequence, weights=weights,
                                        init_recon=init_recon, prox_input=prox_input)

        return recon, fm_rmse

    @staticmethod
    def gen_weights(sinogram, weight_type):
        """
        Compute the optional weights used in MBIR reconstruction.

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

    def gen_modified_3d_sl_phantom(self):
        """
        Generates a simplified, low-dynamic range version of the 3D Shepp-Logan phantom.

        Returns:
            ndarray: A 3D numpy array of shape specified by TomographyModel class parameters.
        """
        recon_shape = self.get_params('recon_shape')
        phantom = mbirjax.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
        return phantom

    def reshape_recon(self, recon):
        """
        Reshape recon into its 3D form.

        Args:
            recon (ndarray or jax array): A 3D array of shape specified by (num_recon_rows, num_recon_cols, num_recon_slices)
        """
        recon_shape = self.get_params('recon_shape')
        return recon.reshape(recon_shape)

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


@partial(jax.jit, static_argnames='recon_shape')
def pm_qggmrf_gradient_and_hessian_at_indices(voxel_values, recon_shape, pixel_indices, sigma_x, p, q, T, b):
    """
    Calculate the gradient and hessian at each index location in a reconstructed image using the qGGMRF prior.

    Args:
        voxel_values (jax.array): 2D reconstructed image array with shape (num_recon_rows x num_recon_cols, num_recon_slices).
        recon_shape (tuple of ints): shape of the original recon:  (num_recon_rows, num_recon_cols, num_recon_slices).
        pixel_indices (int array): Array of shape (N_indices, num_recon_slices) representing the indices of voxels in a flattened array to be updated.
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
    num_rows, num_cols, num_slices = recon_shape[:3]

    # Convert flat indices to 2D indices for row and column access.
    row_index, col_index = jnp.unravel_index(pixel_indices, shape=(num_rows, num_cols))

    # Access the central voxels' values at the given pixel_indices. Shape of xs is (num indices)x(num slices)
    xs = voxel_values[pixel_indices]

    # Define relative positions for accessing neighborhood voxels.
    offsets = [[1, 0], [-1, 0], [0, 1], [0, -1]]

    # Create a list derived from 4 neighbors, each entry in list is a 1D vector containing the recon values at that
    # location over all slices.
    xr = []
    for offset in offsets:
        new_indices = jnp.ravel_multi_index([row_index + offset[0], col_index + offset[1]], dims=(num_rows, num_cols), mode='clip')
        xr.append(voxel_values[new_indices])

    # Shift slices up with reflection
    xs_up = jnp.roll(xs, shift=-1, axis=1).at[:, -1].set(xs[:, -1])

    # Shift slices down with reflection
    xs_down = jnp.roll(xs, shift=1, axis=1).at[:, 0].set(xs[:, 0])

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


@jax.jit
def pm_prox_gradient_at_indices(recon, prox_input, pixel_indices, sigma_p):
    """
    Calculate the gradient and hessian at each index location in a reconstructed image using the qGGMRF prior.

    Args:
        recon (jax.array): 2D reconstructed image array with shape (num_recon_rows x num_recon_cols, num_recon_slices).
        prox_input (jax.array): 2D reconstructed image array with shape (num_recon_rows x num_recon_cols, num_recon_slices).
        pixel_indices (int array): Array of shape (N_indices, num_recon_slices) representing the indices of voxels in a flattened array to be updated.
        sigma_p (float): Standard deviation parameter of the proximal map.

    Returns:
        first_derivative of shape (N_indices, num_recon_slices) representing the gradient of the prox term at specified indices.
    """

    # Compute the prior model gradient at all voxels
    cur_diff = recon[pixel_indices] - prox_input[pixel_indices]
    pm_gradient = (1.0 / (sigma_p ** 2.0)) * cur_diff

    # Shape of pm_gradient is (num indices)x(num slices)
    return pm_gradient


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
    first_term = ((delta / sigma_x) ** p) / p
    second_term = (delta / (T * sigma_x)) ** (q - p)
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
    first_term = (delta_prime ** (q - 2.0)) / (2.0 * (sigma_x ** p) * (T * sigma_x ** (q - p)))

    # third_term is the third term in formula.
    second_term = (delta_prime / (T * sigma_x)) ** (q - p)
    third_term = ((q / p) + second_term) / ((1 + second_term) ** 2.0)

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
