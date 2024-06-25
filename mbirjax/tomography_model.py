import types
import numpy as np
import warnings
import gc
import jax
import jax.numpy as jnp
import mbirjax
from mbirjax import ParameterHandler
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

    def __init__(self, sinogram_shape, **kwargs):

        super().__init__()
        self.sparse_forward_project, self.sparse_back_project = None, None  # These are callable functions compiled in set_params
        self.compute_hessian_diagonal = None
        self.set_params(no_compile=True, no_warning=True, sinogram_shape=sinogram_shape, **kwargs)
        delta_voxel = self.get_params('delta_voxel')
        if delta_voxel is None:
            magnification = self.get_magnification()
            delta_det_channel = self.get_params('delta_det_channel')
            delta_voxel = delta_det_channel / magnification
            self.set_params(no_compile=True, no_warning=True, delta_voxel=delta_voxel)

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
    def back_project_one_view_to_pixel_batch(sinogram_view, pixel_indices, single_view_params, projector_params,
                                             coeff_power=1):
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
        Automatically sets the regularization parameters (self.sigma_y, self.sigma_x, and self.sigma_prox) used in MBIR reconstruction based on the provided sinogram and optional weights.

        Args:
            sinogram (jnp.array): 3D jax array containing the sinogram with shape (num_views, num_det_rows, num_det_channels).
            weights (jnp.array, optional): 3D weights array with the same shape as the sinogram. Defaults to all 1s.

        Returns:
            namedtuple containing the parameters sigma_y, sigma_x, sigma_prox

        The method adjusts the regularization parameters only if `auto_regularize_flag` is set to True within the model's parameters.
        """
        if self.get_params('auto_regularize_flag'):
            self.auto_set_sigma_y(sinogram, weights)
            self.auto_set_sigma_x(sinogram)
            self.auto_set_sigma_prox(sinogram)

        regularization_param_names = ['sigma_y', 'sigma_x', 'sigma_prox']
        RegularizationParams = namedtuple('RegularizationParams', regularization_param_names)
        regularization_param_values = [float(val) for val in self.get_params(
            regularization_param_names)]  # These should be floats, but the user may have set them to jnp.float
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
        signal_rms = float(jnp.average(weights * sinogram ** 2, None, sino_indicator) ** 0.5)

        # Convert snr to relative noise standard deviation
        rel_noise_std = 10 ** (-snr_db / 20)
        # compute the default_pixel_pitch = the detector pixel pitch in the recon plane given the magnification
        default_pixel_pitch = delta_det_channel / magnification

        # Compute the recon pixel pitch relative to the default.
        pixel_pitch_relative_to_default = delta_voxel / default_pixel_pitch

        # Compute sigma_y and scale by relative pixel pitch
        sigma_y = np.float32(rel_noise_std * signal_rms * (pixel_pitch_relative_to_default ** 0.5))
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
        sigma_x = np.float32(0.2 * (2 ** sharpness) * recon_std)
        self.set_params(no_warning=True, sigma_x=sigma_x, auto_regularize_flag=True)

    def auto_set_sigma_prox(self, sinogram):
        """
        Compute the automatic value of ``sigma_prox`` for use in MBIR reconstruction with proximal map prior.

        Args:
            sinogram (jax array): 3D jax array containing sinogram with shape (num_views, num_det_rows, num_det_channels).
        """
        # Get parameters
        sharpness = self.get_params('sharpness')
        recon_std = self._get_estimate_of_recon_std(sinogram)

        # Compute sigma_x as a fraction of the typical recon value
        # 0.2 is an empirically determined constant
        sigma_prox = np.float32(0.2 * (2 ** sharpness) * recon_std)
        self.set_params(no_warning=True, sigma_prox=sigma_prox, auto_regularize_flag=True)

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
        Estimate the standard deviation of the reconstruction from the sinogram.  This is used to scale sigma_prox and
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
        typical_path_length_space = (2 * recon_shape[0] * recon_shape[1]) / (
                recon_shape[0] + recon_shape[1]) * delta_voxel

        # Compute a typical projection path length based on the detector column width
        typical_path_length_sino = num_det_channels * delta_det_channel / magnification

        # Compute a typical projection path as the minimum of the two estimates
        typical_path_length = jnp.minimum(typical_path_length_space, typical_path_length_sino)

        # Compute a typical recon value by dividing average sinogram value by a typical projection path length
        recon_std = typical_sinogram_value / typical_path_length

        return recon_std

    def recon(self, sinogram, weights=None, num_iterations=15, first_iteration=0, init_recon=None,
              compute_prior_loss=False):
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
            first_iteration (int, optional): Set this to be the number of iterations previously completed when
            restarting a recon using init_recon.
            init_recon (jax array, optional): Optional reconstruction to be used for initialization.
            compute_prior_loss (bool, optional):  Set true to calculate and return the prior model loss.  This will
            lead to slower reconstructions and is meant only for small recons.

        Returns:
            [recon, recon_params]: reconstruction and a named tuple containing the recon parameters.
            recon_params (namedtuple): num_iterations, granularity, partition_sequence, fm_rmse, prior_loss, regularization_params
        """
        # Run auto regularization. If auto_regularize_flag is False, then this will have no effect
        if compute_prior_loss:
            msg = 'Computing the prior loss on every iteration uses significant memory and computing power.\n'
            msg += 'Set compute_prior_loss=False for most applications aside from debugging and demos.'
            warnings.warn(msg)

        regularization_params = self.auto_set_regularization_params(sinogram, weights=weights)

        # Generate set of voxel partitions
        recon_shape, granularity = self.get_params(['recon_shape', 'granularity'])
        partitions = mbirjax.gen_set_of_pixel_partitions(recon_shape, granularity)

        # Generate sequence of partitions to use
        partition_sequence = self.get_params('partition_sequence')
        partition_sequence = mbirjax.gen_partition_sequence(partition_sequence, num_iterations=num_iterations)
        partition_sequence = partition_sequence[first_iteration:]

        # Compute reconstruction
        recon, loss_vectors = self.vcd_recon(sinogram, partitions, partition_sequence, weights=weights,
                                             init_recon=init_recon, compute_prior_loss=compute_prior_loss,
                                             first_iteration=first_iteration)

        # Return num_iterations, granularity, partition_sequence, fm_rmse values, regularization_params
        recon_param_names = ['num_iterations', 'granularity', 'partition_sequence', 'fm_rmse', 'prior_loss',
                             'regularization_params', 'nrms_recon_change']
        ReconParams = namedtuple('ReconParams', recon_param_names)
        partition_sequence = [int(val) for val in partition_sequence]
        fm_rmse = [float(val) for val in loss_vectors[0]]
        if compute_prior_loss:
            prior_loss = [float(val) for val in loss_vectors[1]]
        else:
            prior_loss = [0]
        nrms_recon_change = [float(val) for val in loss_vectors[2]]
        recon_param_values = [num_iterations, granularity, partition_sequence, fm_rmse, prior_loss,
                              regularization_params._asdict(), nrms_recon_change]
        recon_params = ReconParams(*tuple(recon_param_values))

        return recon, recon_params

    def vcd_recon(self, sinogram, partitions, partition_sequence, weights=None, init_recon=None, prox_input=None,
                  compute_prior_loss=False, first_iteration=0):
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
            compute_prior_loss (bool, optional):  Set true to calculate and return the prior model loss.
            first_iteration (int, optional): Set this to be the number of iterations previously completed when
            restarting a recon using init_recon.

        Returns:
            (recon, recon_stats): tuple of 3D reconstruction and a tuple containing arrays of per-iteration stats.
            recon_stats = (fm_rmse, pm_rmse, nrms_update), where fm is forward model, pm is prior model, and
            nrms_update is ||recon(i+1) - recon(i)||_2 / ||recon(i+1)||_2.
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
        pm_loss = np.zeros(num_iters)
        nrms_update = np.zeros(num_iters)
        for i in range(num_iters):
            # Get the current partition (set of subsets) and shuffle the subsets
            partition = partitions[partition_sequence[i]]
            subset_indices = np.random.permutation(partition.shape[0])

            # Do an iteration
            vcd_data = [error_sinogram, flat_recon, partition, nrms_update[i]]
            error_sinogram, flat_recon, norm_square_update = vcd_partition_iterator(vcd_data, subset_indices)

            # Compute the stats and display as desired
            fm_rmse[i] = self.get_forward_model_loss(error_sinogram, sigma_y, weights)
            nrms_update[i] = norm_square_update / jnp.sum(flat_recon * flat_recon)
            es_rmse = jnp.linalg.norm(error_sinogram) / jnp.sqrt(error_sinogram.size)

            if verbose >= 1:
                iter_output = 'After iteration {}: Pct change={:.3f}, Forward loss={:.3f}'.format(i + first_iteration, 100*nrms_update[i], fm_rmse[i])
                if compute_prior_loss:
                    b, sigma_x, p, q, T = self.get_params(['b', 'sigma_x', 'p', 'q', 'T'])
                    b = tuple(b)
                    qggmrf_params = (b, sigma_x, p, q, T)
                    pm_loss[i] = mbirjax.qggmrf_loss(flat_recon.reshape(recon.shape), qggmrf_params)
                    pm_loss[i] /= flat_recon.size
                    # Each loss is scaled by the number of elements, but the optimization uses unscaled values.
                    # To provide an accurate, yet properly scaled total loss, first remove the scaling and add,
                    # then scale by the average number of elements between the two.
                    total_loss = ((fm_rmse[i] * sinogram.size + pm_loss[i] * flat_recon.size) /
                                  (0.5 * (sinogram.size + flat_recon.size)))
                    iter_output += ', Prior loss={:.3f}, Weighted total loss={:.3f}'.format(pm_loss[i], total_loss)

                print(iter_output)
                print(f'Error sino RMSE={es_rmse:.3f}')
                if verbose >= 2:
                    mbirjax.get_memory_stats()
                    print('--------')

        return self.reshape_recon(flat_recon), (fm_rmse, pm_loss, nrms_update)

    @staticmethod
    def create_vcd_partition_iterator(vcd_subset_iterator):
        """
        Create a jit-compiled function to update all the pixels in the recon and error sinogram by applying
        the supplied vcd_subset_iterator to each subset in a partition.

        Args:
            vcd_subset_iterator (callable):  The function returned by create_vcd_subset_iterator.

        Returns:
            (callable) vcd_partition_iterator(sinogram_recon_partition, subset_indices)
        """

        def vcd_partition_iterator(sinogram_recon_partition, subset_indices):
            """
            Calculate a full iteration of the VCD algorithm by scanning over the subsets of the partition.
            Each iteration of the algorithm should return a better reconstructed recon.
            The error_sinogram should always be:  error_sinogram = measured_sinogram - forward_proj(recon)
            where measured_sinogram is the measured sinogram and recon is the current reconstruction.

            Args:
                sinogram_recon_partition (list or tuple): 4 element tuple containing

                    * error_sinogram (jax array): 3D error sinogram with shape (num_views, num_det_rows, num_det_channels).
                    * flat_recon (jax array): 2D array reconstruction with shape (num_recon_rows x num_recon_cols, num_recon_slices).
                    * partition (jax array): 2D array where partition[subset_index] gives a 1D array of pixel indices.
                    * Squared sum of changes to the recon during this iteration but before this subset.

                subset_indices (jax array): An array of indices into the partition - this gives the order in which the subsets are updated.

            Returns:
                (error_sinogram, flat_recon, norm_square_update): The first two have the same shape as above, but
                are updated to reduce overall loss function.  The norm_square_update includes the changes from this subset.
            """

            # Scan over the subsets of the partition, using the subset_indices to order them.
            sinogram_recon_partition, _ = jax.lax.scan(vcd_subset_iterator, sinogram_recon_partition, subset_indices)

            error_sinogram, flat_recon, _, norm_square_update = sinogram_recon_partition
            return error_sinogram, flat_recon, norm_square_update

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
        b, sigma_x, p, q, T = self.get_params(['b', 'sigma_x', 'p', 'q', 'T'])
        b = np.array(b)
        b = b / np.sum(b)
        b = tuple(b)
        qggmrf_params = tuple((b, sigma_x, p, q, T))
        sigma_prox = self.get_params('sigma_prox')
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
                sinogram_recon_partition (list or tuple): 4 element tuple containing

                    * error_sinogram (jax array): 3D error sinogram with shape (num_views, num_det_rows, num_det_channels).
                    * flat_recon (jax array): 2D array reconstruction with shape (num_recon_rows x num_recon_cols, num_recon_slices).
                    * partition (jax array): 2D array where partition[subset_index] gives a 1D array of pixel indices.
                    * Squared sum of changes to the recon during this iteration but before this subset.

                subset_index (int): integer index of the subset within the partition.

            Returns:
                [error_sinogram, flat_recon, partition, norm_square_update]: The first two have the same shape as above, but
                are updated to reduce overall loss function.  The norm_square_update includes the changes from this subset.
            """
            error_sinogram, flat_recon, partition, norm_square_update = sinogram_recon_partition
            pixel_indices = partition[subset_index]

            def delta_recon_batch(index_batch):
                # Compute the forward model gradient and hessian at each pixel in the index set.
                # Assumes Loss(delta) = 1/(2 sigma_y^2) || error_sinogram - A delta ||_weights^2
                forward_grad_batch = -fm_constant * sparse_back_project(error_sinogram * weights, index_batch)
                forward_hess_batch = fm_constant * fm_hessian[index_batch]

                # Compute the prior model gradient and hessian (i.e., second derivative) terms
                if prox_input is None:
                    # qGGMRF prior - compute the qggmrf gradient and hessian at each pixel in the index set.
                    prior_grad_batch, prior_hess_batch = (
                        mbirjax.qggmrf_gradient_and_hessian_at_indices(flat_recon, recon_shape, index_batch, qggmrf_params))
                else:
                    # Proximal map prior - compute the prior model gradient at each pixel in the index set.
                    prior_hess_batch = sigma_prox ** 2
                    prior_grad_batch = mbirjax.prox_gradient_at_indices(flat_recon, prox_input, index_batch, sigma_prox)

                # Compute update vector update direction in recon domain
                delta_recon_at_indices_batch = - ((forward_grad_batch + prior_grad_batch) /
                                                  (forward_hess_batch + prior_hess_batch))

                # Compute delta^T \nabla Q(x_hat; x'=x_hat) for use in finding alpha
                prior_grad_delta = jnp.sum(prior_grad_batch * delta_recon_at_indices_batch)
                prior_grad_delta = prior_grad_delta.reshape((1, 1))
                # Estimated upper bound for hessian
                prior_over_relaxation_factor = 2
                prior_hess_max_delta = ((1 / prior_over_relaxation_factor) *
                                        jnp.sum(prior_hess_batch * delta_recon_at_indices_batch ** 2))
                prior_hess_max_delta = prior_hess_max_delta.reshape((1, 1))
                return delta_recon_at_indices_batch, prior_grad_delta, prior_hess_max_delta

            # Get the update direction
            batch_update_at_indices = mbirjax.concatenate_function_in_batches(delta_recon_batch, pixel_indices,
                                                                              pixel_batch_size)
            delta_recon_at_indices = batch_update_at_indices[0]

            # Then sum over the batched outputs to get delta^T \nabla Q(x_hat; x'=x_hat) for use in finding alpha
            prior_linear = jnp.sum(batch_update_at_indices[1])
            prior_quadratic_approx = jnp.sum(batch_update_at_indices[2])

            # Compute update direction in sinogram domain
            delta_sinogram = sparse_forward_project(delta_recon_at_indices, pixel_indices)
            forward_linear = jnp.sum(fm_constant * error_sinogram * delta_sinogram * weights)
            forward_quadratic = jnp.sum(fm_constant * delta_sinogram * delta_sinogram * weights)

            # Compute optimal update step
            alpha_numerator = forward_linear - prior_linear
            alpha_denominator = forward_quadratic + prior_quadratic_approx + jnp.finfo(jnp.float32).eps
            alpha = alpha_numerator / alpha_denominator
            max_alpha = 1.5
            alpha = jnp.clip(alpha, jnp.finfo(jnp.float32).eps, max_alpha)  # a_max=alpha_clip_value

            # # Debug/demo code to determine the quadratic part of the prior exactly, but expensively.
            # x_prime = flat_recon.reshape(recon_shape)
            # delta = jnp.zeros_like(flat_recon)
            # delta = delta.at[pixel_indices].set(delta_recon_at_indices)
            # delta = delta.reshape(recon_shape)
            # _, grad_at_delta = mbirjax.compute_surrogate_and_grad(delta, x_prime, qggmrf_params)
            # grad_at_delta = grad_at_delta.reshape(flat_recon.shape)[pixel_indices]
            # prior_quadratic = jnp.sum(delta_recon_at_indices * grad_at_delta)
            # alpha_denominator_exact = forward_quadratic + prior_quadratic
            # alpha_exact = alpha_numerator / alpha_denominator_exact
            # jax.debug.print('---')
            # jax.debug.print('ae:{alpha_exact}, \ta:{alpha}', alpha_exact=alpha_exact, alpha=alpha)
            # jax.debug.print('fl:{forward_linear}, \tfq:{forward_quadratic}',
            #                 forward_linear=forward_linear, forward_quadratic=forward_quadratic)
            # jax.debug.print('pl:{prior_linear}, \tpq:{prior_quadratic}, \tpqa:{pqa}',
            #                 prior_linear=prior_linear, prior_quadratic=prior_quadratic, pqa=prior_quadratic_approx)
            # alpha = alpha_exact
            # # End debug/demo code

            # Enforce positivity constraint if desired
            # Greg, this may result in excess compilation. Not sure.
            if positivity_flag is True:
                # Get recon at index_batch
                recon_at_indices = flat_recon[pixel_indices]

                # Clip updates to ensure non-negativity
                pos_constant = 1.0 / (alpha + jnp.finfo(jnp.float32).eps)
                delta_recon_at_indices = jnp.maximum(-pos_constant * recon_at_indices, delta_recon_at_indices)

                # Recompute sinogram projection
                delta_sinogram = sparse_forward_project(delta_recon_at_indices, pixel_indices)

            # Perform sparse updates at index locations
            update = alpha * delta_recon_at_indices
            flat_recon = flat_recon.at[pixel_indices].add(update)

            # Update sinogram and loss
            error_sinogram = error_sinogram - alpha * delta_sinogram
            norm_square_update += jnp.sum(update * update)

            return [error_sinogram, flat_recon, partition, norm_square_update], None

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
        This function is similar to recon, but it essentially uses a prior with a mean of prox_input and a standard deviation of sigma_prox.

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
        recon, loss_vectors = self.vcd_recon(sinogram, partitions, partition_sequence, weights=weights,
                                             init_recon=init_recon, prox_input=prox_input)

        return recon, loss_vectors

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
    input_info = types.SimpleNamespace(shape=input_shape, dtype=jnp.dtype(jnp.float32))
    transpose = jax.linear_transpose(linear_map, input_info)
    # print('Done: ' + str(time.time() - t0))
    return transpose
