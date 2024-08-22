import types
import numpy as np
import warnings
import time
import os
num_cpus = 3 * os.cpu_count() // 4
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count={}'.format(num_cpus)
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

        self.main_device, self.worker, self.pixels_per_batch, self.views_per_batch = None, None, None, None
        self.cpus = jax.devices('cpu')
        self.projector_functions = None

        self.set_devices_and_batch_sizes()
        self.create_projectors()

    def set_devices_and_batch_sizes(self):

        # Get the cpu and any gpus
        # If no gpu, then use the cpu and return
        cpus = jax.devices('cpu')
        gb = 1024 ** 3
        use_gpu = self.get_params('use_gpu')
        try:
            gpus = jax.devices('gpu')
            gpu_memory_stats = gpus[0].memory_stats()
            gpu_memory = float(gpu_memory_stats['bytes_limit']) - float(gpu_memory_stats['bytes_in_use'])
            gpu_memory /= gb
        except RuntimeError:
            if use_gpu not in ['automatic', 'none']:
                raise RuntimeError("'use_gpu' is set to {} but no gpu is available.  Reset to 'automatic' or 'none'.".format(use_gpu))
            gpus = []
            gpu_memory = 0

        # Estimate the memory available and required for this problem
        cpu_memory_stats = mbirjax.get_memory_stats(print_results=False)[-1]
        cpu_memory = float(cpu_memory_stats['bytes_limit']) - float(cpu_memory_stats['bytes_in_use'])
        cpu_memory /= gb

        sinogram_shape = self.get_params('sinogram_shape')
        recon_shape = self.get_params('recon_shape')

        sino_reps_for_vcd = 6
        recon_reps_for_vcd = 8
        reps_for_vcd_update = 3

        zero = jnp.zeros(1)
        bits_per_byte = 8
        mem_per_entry = float(str(zero.dtype)[5:]) / bits_per_byte / gb  # Parse floatXX to get the number of bits
        memory_per_sinogram = mem_per_entry * np.prod(sinogram_shape)
        memory_per_recon = mem_per_entry * np.prod(recon_shape)

        total_memory_required = memory_per_sinogram * sino_reps_for_vcd + memory_per_recon * recon_reps_for_vcd
        subset_update_memory_required = (memory_per_sinogram + memory_per_recon) * reps_for_vcd_update
        if isinstance(self, mbirjax.ConeBeamModel):
            reps_per_projection = 2  # This is determined empirically and should probably be determined by each subclass rather than set here.
        elif isinstance(self, mbirjax.ParallelBeamModel):
            reps_per_projection = 2
        else:
            raise ValueError('Unknown reps_per_projection for {}.'.format(self.get_params('geometry_type')))

        # Set the default batch sizes, then adjust as needed and update memory requirements
        self.views_per_batch = 256
        self.pixels_per_batch = 2048
        num_slices = max(sinogram_shape[1], recon_shape[2])
        projection_memory_per_view = self.pixels_per_batch * reps_per_projection * num_slices * mem_per_entry

        subset_memory_excess = gpu_memory - subset_update_memory_required
        subset_views_per_batch = subset_memory_excess // projection_memory_per_view
        subset_views_per_batch = int(np.clip(subset_views_per_batch, 2, self.views_per_batch))

        total_memory_excess = cpu_memory - total_memory_required
        total_views_per_batch = total_memory_excess // projection_memory_per_view
        total_views_per_batch = int(np.clip(total_views_per_batch, 2, self.views_per_batch))

        subset_update_memory_required += subset_views_per_batch * projection_memory_per_view
        total_memory_required += total_views_per_batch * projection_memory_per_view

        if (gpu_memory > total_memory_required and use_gpu == 'automatic') or use_gpu == 'full':
            main_device = gpus[0]
            worker = gpus[0]
        elif (gpu_memory > subset_update_memory_required and use_gpu == 'automatic') or use_gpu == 'worker':
            main_device = cpus[0]
            worker = gpus[0]
            if use_gpu != 'worker':
                warnings.warn("GPU memory likely smaller than full problem size. Estimated required = {}GB.  "
                              "Available = {}GB.".format(subset_update_memory_required, gpu_memory))
                warnings.warn("Using CPU for main memory and  "
                              "GPU for update steps only, which will increase recon time.")
                warnings.warn("  Use set_params(use_gpu='full') to try the full problem on GPU.")
        elif cpu_memory > total_memory_required:
            main_device = cpus[0]
            worker = cpus[0]
            gpu_memory_required = 0
            if len(gpus) > 0 and use_gpu != 'none':
                warnings.warn('Insufficient GPU memory for this problem. Estimated required = {}GB.  '
                              'Available = {}GB.'.format(subset_update_memory_required, gpu_memory))
                warnings.warn('Trying on CPU, but this may be slow.')
                warnings.warn("  Use set_params(use_gpu='worker') to try projections on the GPU.")
        else:
            message = 'Problem is likely too large for available memory.'
            message += '\nEstimated memory required = {:.3f}GB.  Available:  CPU = {:.3f}GB, GPU = {:.3f}GB.'.format(
                total_memory_required, cpu_memory, gpu_memory)
            warnings.warn(message)

        if gpu_memory > 0:
            print('Available GPU memory = {:.3f}GB.'.format(gpu_memory))
        print('Estimated memory required = {:.3f}GB full, {:.3f}GB update'.format(total_memory_required,
                                                                                  subset_update_memory_required))

        print('Using {} for main memory, {} as worker.'.format(main_device, worker))
        print('views_per_batch = {}; pixels_per_batch = {}'.format(self.views_per_batch, self.pixels_per_batch))

        self.main_device = main_device
        self.worker = worker

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
        raise ValueError('from_file is not implemented for base TomographyModel')

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
        self.projector_functions = mbirjax.Projectors(self)

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

    def sparse_forward_project(self, voxel_values, indices, output_device=None):
        """
        Forward project the given voxel values to a sinogram.
        The indices are into a flattened 2D array of shape (recon_rows, recon_cols), and the projection is done using
        all voxels with those indices across all the slices.

        Args:
            voxel_values (jax.numpy.DeviceArray): 2D array of voxel values to project, size (len(pixel_indices), num_recon_slices).
            indices (jax array): Array of indices specifying which voxels to project.
            output_device (jax device): Device on which to put the output

        Returns:
            jnp array: The resulting 3D sinogram after projection.
        """
        # Get the current devices and move the data to the worker
        max_views = 256
        sinogram_shape = self.get_params('sinogram_shape')
        view_indices = jnp.arange(sinogram_shape[0])
        num_batches = jnp.ceil(sinogram_shape[0] / max_views).astype(int)
        view_indices_batched = jnp.array_split(view_indices, num_batches)
        voxel_values, indices = jax.device_put([voxel_values, indices], self.worker)

        sinogram = []
        for view_indices_batch in view_indices_batched:
            sinogram_views = self.projector_functions.sparse_forward_project(voxel_values, indices, view_indices=view_indices_batch)
            # Put the data on the appropriate device
            sinogram.append(jax.device_put(sinogram_views, output_device))

        sinogram = jnp.concatenate(sinogram)
        return sinogram

    def sparse_back_project(self, sinogram, indices, coeff_power=1, output_device=None):
        """
        Back project the given sinogram to the voxels given by the indices.
        The indices are into a flattened 2D array of shape (recon_rows, recon_cols), and the projection is done using
        all voxels with those indices across all the slices.

        Args:
            sinogram (jnp array): 3D jax array containing sinogram.
            indices (jnp array): Array of indices specifying which voxels to back project.
            output_device (jax device): Device on which to put the output

        Returns:
            A jax array of shape (len(indices), num_slices)
        """
        max_views = 256
        num_views = sinogram.shape[0]
        view_indices = jnp.arange(num_views)
        num_batches = jnp.ceil(sinogram.shape[0] / max_views).astype(int)
        view_indices_batched = jnp.array_split(view_indices, num_batches)
        recon_shape = self.get_params('recon_shape')
        with jax.default_device(output_device):
            recon_at_indices = jnp.zeros((len(indices), recon_shape[2]))
            recon_at_indices = jax.device_put(recon_at_indices, output_device)
        pixel_indices = jax.device_put(indices, self.worker)

        view_batch_inds = [index_set[0] for index_set in view_indices_batched] + [num_views]
        for j in range(len(view_batch_inds) - 1):
            cur_views = sinogram[view_batch_inds[j]:view_batch_inds[j + 1]]
            # Get the current devices and move the data to the worker
            cur_views = jax.device_put(cur_views, self.worker)

            cur_recon_at_indices = self.projector_functions.sparse_back_project(cur_views, pixel_indices,
                                                                                view_indices=view_indices_batched[j],
                                                                                coeff_power=coeff_power)

            # Put the data on the appropriate device
            recon_at_indices += jax.device_put(cur_recon_at_indices, output_device)

        return recon_at_indices

    def compute_hessian_diagonal(self, weights=None, output_device=None):
        """
        Computes the diagonal elements of the Hessian matrix for given weights.

        Args:
            weights (jax array, optional): 3D positive weights with same shape as sinogram.  Defaults to all 1s.
            output_device (jax device): Device on which to put the output

        Returns:
            jnp array: Diagonal of the Hessian matrix with same shape as recon.
        """
        """
        Computes the diagonal of the Hessian matrix, which is computed by doing a backprojection of the weight
        matrix except using the square of the coefficients in the backprojection to a given voxel.
        One of weights or sinogram_shape must be not None. If weights is not None, it must be an array with the same
        shape as the sinogram to be backprojected.  If weights is None, then a weights matrix will be computed as an
        array of ones of size sinogram_shape.

        Args:
           weights (ndarray or jax array or None, optional): 3D array of shape
                (cur_num_views, num_det_rows, num_det_cols), where cur_num_views is recon_shape[0]
                if view_indices is () and len(view_indices) otherwise, in which case the views in weights should
                match those indicated by view_indices.  Defaults to all 1s.
           view_indices (ndarray or jax array, optional): 1D array of indices into the view parameters array.
                If None, then all views are used.

        Returns:
            An array that is the same size as the reconstruction.
        """
        sinogram_shape, recon_shape = self.get_params(['sinogram_shape', 'recon_shape'])
        num_views = sinogram_shape[0]
        if weights is None:
            with jax.default_device(self.main_device):
                weights = jnp.ones((num_views,) + sinogram_shape[1:])
        elif weights.shape != (num_views,) + sinogram_shape[1:]:
            error_message = 'Weights must be constant or an array compatible with sinogram'
            error_message += '\nGot weights.shape = {}, but sinogram.shape = {}'.format(weights.shape, sinogram_shape)
            raise ValueError(error_message)

        num_recon_rows, num_recon_cols, num_recon_slices = recon_shape[:3]
        max_index = num_recon_rows * num_recon_cols
        indices = jnp.arange(max_index)

        hessian_diagonal = self.sparse_back_project(weights, indices, coeff_power=2, output_device=output_device)

        return hessian_diagonal.reshape((num_recon_rows, num_recon_cols, num_recon_slices))

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
            self.set_devices_and_batch_sizes()
            self.create_projectors()

    def verify_valid_params(self):
        """
        Check that all parameters are compatible for a reconstruction.

        Note:
            Raises ValueError for invalid parameters.
        """
        super().verify_valid_params()
        use_gpu = self.get_params('use_gpu')

        if use_gpu not in ['automatic', 'full', 'worker', 'none']:
            error_message = "use_gpu must be one of \n"
            error_message += " 'automatic' (code will try to determine problem size and use gpu appropriately),\n'"
            error_message += " 'full' (use gpu for all calculations),\n"
            error_message += " 'worker' (use gpu for projections only),\n"
            error_message += " 'none' (do not use gpu at all)."
            raise ValueError(error_message)

    def auto_set_regularization_params(self, sinogram, weights=None):
        """
        Automatically sets the regularization parameters (self.sigma_y, self.sigma_x, and self.sigma_prox) used in MBIR reconstruction based on the provided sinogram and optional weights.

        Args:
            sinogram (ndarray): 3D jax array containing the sinogram with shape (num_views, num_det_rows, num_det_channels).
            weights (ndarray, optional): 3D weights array with the same shape as the sinogram. Defaults to all 1s.

        Returns:
            namedtuple containing the parameters sigma_y, sigma_x, sigma_prox

        Notes:
            The method adjusts the regularization parameters only if `auto_regularize_flag` is set to True within the model's parameters.
            Also, the inputs may be jax arrays, but they are cast to numpy arrays before calculation to avoid
            duplicating large sinograms on the GPU.
        """
        if self.get_params('auto_regularize_flag'):
            # Make sure sinogram and weights are on the cpu to avoid duplication of large sinos on the GPU.
            sinogram = np.array(sinogram)
            if weights is None:
                weights = 1
            # Compute indicator function for sinogram support
            sino_indicator = self._get_sino_indicator(sinogram)
            self.auto_set_sigma_y(sinogram, sino_indicator, weights)

            recon_std = self._get_estimate_of_recon_std(sinogram, sino_indicator)
            self.auto_set_sigma_x(recon_std)
            self.auto_set_sigma_prox(recon_std)

        regularization_param_names = ['sigma_y', 'sigma_x', 'sigma_prox']
        RegularizationParams = namedtuple('RegularizationParams', regularization_param_names)
        regularization_param_values = [float(val) for val in self.get_params(
            regularization_param_names)]  # These should be floats, but the user may have set them to jnp.float
        regularization_params = RegularizationParams(*tuple(regularization_param_values))

        return regularization_params

    def auto_set_sigma_y(self, sinogram, sino_indicator, weights=1):
        """
        Sets the value of the parameter sigma_y used for use in MBIR reconstruction.

        Args:
            sinogram (jax array or ndarray): 3D jax array containing sinogram with shape (num_views, num_det_rows, num_det_channels).
            sino_indicator (jax array or ndarray): a binary mask that indicates the region of sinogram support; same shape as sinogram.
            weights (jax array, optional): 3D positive weights with same shape as sinogram.  Defaults to all 1s.
        """

        # Get parameters
        snr_db = self.get_params('snr_db')
        magnification = self.get_magnification()
        delta_voxel, delta_det_channel = self.get_params(['delta_voxel', 'delta_det_channel'])

        # Compute RMS value of sinogram excluding empty space
        signal_rms = float(np.average(weights * sinogram ** 2, None, sino_indicator) ** 0.5)

        # Convert snr to relative noise standard deviation
        rel_noise_std = 10 ** (-snr_db / 20)
        # compute the default_pixel_pitch = the detector pixel pitch in the recon plane given the magnification
        default_pixel_pitch = delta_det_channel / magnification

        # Compute the recon pixel pitch relative to the default.
        pixel_pitch_relative_to_default = delta_voxel / default_pixel_pitch

        # Compute sigma_y and scale by relative pixel pitch
        sigma_y = np.float32(rel_noise_std * signal_rms * (pixel_pitch_relative_to_default ** 0.5))
        self.set_params(no_warning=True, sigma_y=sigma_y, auto_regularize_flag=True)

    def auto_set_sigma_x(self, recon_std):
        """
        Compute the automatic value of ``sigma_x`` for use in MBIR reconstruction with qGGMRF prior.

        Args:
            recon_std (float): Estimated standard deviation of the reconstruction from _get_estimate_of_recon_std.
        """
        # Get parameters
        sharpness = self.get_params('sharpness')

        # Compute sigma_x as a fraction of the typical recon value
        # 0.2 is an empirically determined constant
        sigma_x = np.float32(0.2 * (2 ** sharpness) * recon_std)
        self.set_params(no_warning=True, sigma_x=sigma_x, auto_regularize_flag=True)

    def auto_set_sigma_prox(self, recon_std):
        """
        Compute the automatic value of ``sigma_prox`` for use in MBIR reconstruction with proximal map prior.

        Args:
            recon_std (float): Estimated standard deviation of the reconstruction from _get_estimate_of_recon_std.
        """
        # Get parameters
        sharpness = self.get_params('sharpness')

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
            recon (ndarray): The 3D reconstruction array.
            indices (ndarray): Array of indices specifying which voxels to project.

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
        Compute a binary mask that indicates the region of sinogram support.

        Args:
            sinogram (ndarray): 3D jax array containing sinogram with shape (num_views, num_det_rows, num_det_channels).

        Returns:
            (ndarray): Weights used in mbircone reconstruction, with the same array shape as ``sinogram``.
        """
        percent_noise_floor = 5.0
        # Form indicator by thresholding sinogram
        indicator = np.int8(sinogram > (0.01 * percent_noise_floor) * np.mean(np.fabs(sinogram)))
        return indicator

    def _get_estimate_of_recon_std(self, sinogram, sino_indicator):
        """
        Estimate the standard deviation of the reconstruction from the sinogram.  This is used to scale sigma_prox and
        sigma_x in MBIR reconstruction.

        Args:
            sinogram (ndarray): 3D jax array containing sinogram with shape (num_views, num_det_rows, num_det_channels).
            sino_indicator (ndarray): a binary mask that indicates the region of sinogram support; same shape as sinogram.
        """
        # Get parameters
        delta_det_channel = self.get_params('delta_det_channel')
        delta_voxel = self.get_params('delta_voxel')
        recon_shape = self.get_params('recon_shape')
        magnification = self.get_magnification()
        num_det_channels = sinogram.shape[-1]

        # Compute the typical magnitude of a sinogram value
        typical_sinogram_value = np.average(np.abs(sinogram), weights=sino_indicator)

        # TODO: Can we replace this with some type of approximate operator norm of A? That would make it universal.
        # Compute a typical projection path length based on the soft minimum of the recon width and height
        typical_path_length_space = (2 * recon_shape[0] * recon_shape[1]) / (
                recon_shape[0] + recon_shape[1]) * delta_voxel

        # Compute a typical projection path length based on the detector column width
        typical_path_length_sino = num_det_channels * delta_det_channel / magnification

        # Compute a typical projection path as the minimum of the two estimates
        typical_path_length = np.minimum(typical_path_length_space, typical_path_length_sino)

        # Compute a typical recon value by dividing average sinogram value by a typical projection path length
        recon_std = typical_sinogram_value / typical_path_length

        return recon_std

    def recon(self, sinogram, weights=None, num_iterations=13, first_iteration=0, init_recon=None,
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
        # Check that sinogram and weights are not taking up GPU space
        if isinstance(sinogram, type(jnp.zeros(1))) and list(sinogram.devices())[0] != self.main_device:
            raise ValueError(
                'With limited GPU memory, sinogram should be either a numpy array or a jax array on the cpu.')
        if weights is not None and isinstance(weights, type(jnp.zeros(1))) and list(weights.devices())[0] != self.main_device:
            raise ValueError(
                'With limited GPU memory, weights should be either a numpy array or a jax array on the cpu.')
        if init_recon is not None and isinstance(init_recon, type(jnp.zeros(1))) and list(init_recon.devices())[0] != self.main_device:
            raise ValueError(
                'With limited GPU memory, init_recon should be either a numpy array or a jax array on the cpu.')

        # Run auto regularization. If auto_regularize_flag is False, then this will have no effect
        if compute_prior_loss:
            msg = 'Computing the prior loss on every iteration uses significant memory and computing power.\n'
            msg += 'Set compute_prior_loss=False for most applications aside from debugging and demos.'
            warnings.warn(msg)

        regularization_params = self.auto_set_regularization_params(sinogram, weights=weights)

        # Generate set of voxel partitions
        recon_shape, granularity = self.get_params(['recon_shape', 'granularity'])
        partitions = mbirjax.gen_set_of_pixel_partitions(recon_shape, granularity, output_device=self.main_device)
        partitions = [jax.device_put(partition, self.main_device) for partition in partitions]

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
                             'regularization_params', 'nrms_recon_change', 'alpha_values']
        ReconParams = namedtuple('ReconParams', recon_param_names)
        partition_sequence = [int(val) for val in partition_sequence]
        fm_rmse = [float(val) for val in loss_vectors[0]]
        if compute_prior_loss:
            prior_loss = [float(val) for val in loss_vectors[1]]
        else:
            prior_loss = [0]
        nrms_recon_change = [float(val) for val in loss_vectors[2]]
        alpha_values = [float(val) for val in loss_vectors[3]]
        recon_param_values = [num_iterations, granularity, partition_sequence, fm_rmse, prior_loss,
                              regularization_params._asdict(), nrms_recon_change, alpha_values]
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

        Note:
            To maximize GPU memory, each of sinogram, weights, init_recon, and prox_input should be on the CPU for large recons.
        """
        # Ensure that everything has the right shape and is on the main device
        if weights is None:
            weights = 1
            constant_weights = True
        else:
            weights = jax.device_put(weights, self.main_device)
            constant_weights = False

        recon_shape = self.get_params('recon_shape')
        num_recon_slices = recon_shape[2]

        if init_recon is None:
            # Initialize VCD recon, and error sinogram
            with jax.default_device(self.main_device):
                recon = jnp.zeros(recon_shape)
            error_sinogram = sinogram
        else:
            # Make sure that init_recon has the correct shape and type
            if init_recon.shape != recon_shape:
                error_message = "init_recon does not have the correct shape. \n"
                error_message += "Expected {}, but got shape {} for init_recon shape.".format(recon_shape,
                                                                                              init_recon.shape)
                raise ValueError(error_message)

            # Initialize VCD recon and error sinogram using the init_reco
            recon = jax.device_put(init_recon, self.worker)
            error_sinogram = self.forward_project(recon)
            error_sinogram = jax.device_put(error_sinogram, self.main_device)
            error_sinogram = sinogram - error_sinogram

        recon = jax.device_put(recon, self.main_device)  # Even if recon was created with main_device as the default, it wasn't committed there.
        error_sinogram = jax.device_put(error_sinogram, self.main_device)

        # Test to make sure the prox_input input is correct
        if prox_input is not None:
            # Make sure that prox_input has the correct size
            if prox_input.shape != recon.shape:
                error_message = "prox_input does not have the correct size. \n"
                error_message += "Expected {}, but got shape {} for prox_input shape.".format(recon.shape,
                                                                                              prox_input.shape)
                raise ValueError(error_message)

            with jax.default_device(self.main_device):
                prox_input = jnp.array(prox_input.reshape((-1, num_recon_slices)))
            prox_input = jax.device_put(prox_input, self.main_device)

        # Get required parameters
        verbose, sigma_y = self.get_params(['verbose', 'sigma_y'])

        # Initialize the diagonal of the hessian of the forward model
        if constant_weights:
            weights = jnp.ones_like(sinogram)

        if verbose >= 1:
            print('Computing Hessian diagonal')
        fm_hessian = self.compute_hessian_diagonal(weights=weights, output_device=self.main_device)
        fm_hessian = fm_hessian.reshape((-1, num_recon_slices))
        if constant_weights:
            weights = 1
        else:
            weights = jax.device_put(weights, self.main_device)

        # Initialize the emtpy recon
        flat_recon = recon.reshape((-1, num_recon_slices))
        flat_recon = jax.device_put(flat_recon, self.main_device)

        # Create the finer grained recon update operators
        vcd_subset_updater = self.create_vcd_subset_updater(fm_hessian, weights=weights, prox_input=prox_input)

        if verbose >= 1:
            print('Starting VCD iterations')
            if verbose >= 2:
                mbirjax.get_memory_stats()
                print('--------')

        # Do the iterations
        num_iters = partition_sequence.size
        fm_rmse = np.zeros(num_iters)
        pm_loss = np.zeros(num_iters)
        nrms_update = np.zeros(num_iters)
        alpha_values = np.zeros(num_iters)
        for i in range(num_iters):
            # Get the current partition (set of subsets) and shuffle the subsets
            partition = partitions[partition_sequence[i]]

            # Do an iteration
            flat_recon, error_sinogram, norm_squared_update, alpha = self.vcd_partition_iterator(vcd_subset_updater,
                                                                                                 flat_recon,
                                                                                                 error_sinogram,
                                                                                                 partition)

            # Compute the stats and display as desired
            fm_rmse[i] = self.get_forward_model_loss(error_sinogram, sigma_y, weights)
            nrms_update[i] = norm_squared_update / jnp.sum(flat_recon * flat_recon)
            es_rmse = jnp.linalg.norm(error_sinogram) / jnp.sqrt(error_sinogram.size)
            alpha_values[i] = alpha

            if verbose >= 1:
                iter_output = 'After iteration {}: Pct change={:.4f}, Forward loss={:.4f}'.format(i + first_iteration,
                                                                                                  100 * nrms_update[i],
                                                                                                  fm_rmse[i])
                if compute_prior_loss:
                    qggmrf_nbr_wts, sigma_x, p, q, T = self.get_params(['qggmrf_nbr_wts', 'sigma_x', 'p', 'q', 'T'])
                    b = mbirjax.get_b_from_nbr_wts(qggmrf_nbr_wts)
                    qggmrf_params = (b, sigma_x, p, q, T)
                    pm_loss[i] = mbirjax.qggmrf_loss(flat_recon.reshape(recon.shape), qggmrf_params)
                    pm_loss[i] /= flat_recon.size
                    # Each loss is scaled by the number of elements, but the optimization uses unscaled values.
                    # To provide an accurate, yet properly scaled total loss, first remove the scaling and add,
                    # then scale by the average number of elements between the two.
                    total_loss = ((fm_rmse[i] * sinogram.size + pm_loss[i] * flat_recon.size) /
                                  (0.5 * (sinogram.size + flat_recon.size)))
                    iter_output += ', Prior loss={:.4f}, Weighted total loss={:.4f}'.format(pm_loss[i], total_loss)

                print(iter_output)
                print(f'Relative step size (alpha)={alpha:.2f}, Error sino RMSE={es_rmse:.4f}')
                if verbose >= 2:
                    mbirjax.get_memory_stats()
                    print('--------')

        return self.reshape_recon(flat_recon), (fm_rmse, pm_loss, nrms_update, alpha_values)

    @staticmethod
    def vcd_partition_iterator(vcd_subset_updater, flat_recon, error_sinogram, partition):
        """
        Calculate a full iteration of the VCD algorithm by scanning over the subsets of the partition.
        Each iteration of the algorithm should return a better reconstructed recon.
        The error_sinogram should always be:  error_sinogram = measured_sinogram - forward_proj(recon)
        where measured_sinogram is the measured sinogram and recon is the current reconstruction.

        Args:
            vcd_subset_updater (callable): Function to iterate over each subset in the partition.
            flat_recon (jax array): 2D array reconstruction with shape (num_recon_rows x num_recon_cols, num_recon_slices).
            error_sinogram (jax array): 3D error sinogram with shape (num_views, num_det_rows, num_det_channels).
            partition (jax array): 2D array where partition[subset_index] gives a 1D array of pixel indices.

        Returns:
            (flat_recon, error_sinogram, norm_squared_for_partition, alpha): The first two have the same shape as above, but
            are updated to reduce overall loss function.
            The norm_squared_for_partition includes the changes from all subsets of this partition.
            alpha is the relative step size in the gradient descent step, averaged over the subsets
            in the partition.
        """

        # Loop over the subsets of the partition, using random subset_indices to order them.
        norm_squared_for_partition = 0
        alpha_sum = 0
        subset_indices = np.random.permutation(partition.shape[0])

        times = np.zeros(13)
        # np.set_printoptions(precision=1, floatmode='fixed', suppress=True)

        for index in subset_indices:
            subset = partition[index]
            flat_recon, error_sinogram, norm_squared_for_subset, alpha_for_subset, times = vcd_subset_updater(flat_recon,
                                                                                                       error_sinogram,
                                                                                                       subset, times)
            norm_squared_for_partition += norm_squared_for_subset
            alpha_sum += alpha_for_subset
        # print('Times = ')
        # print(times)
        # print('Pct time = ')
        # print(100 * times / np.sum(times))

        return flat_recon, error_sinogram, norm_squared_for_partition, alpha_sum / partition.shape[0]

    def create_vcd_subset_updater(self, fm_hessian, weights, prox_input=None):
        """
        Create a jit-compiled function to update a subset of pixels in the recon and error sinogram.

        Args:
            fm_hessian (jax array): Array with same shape as recon containing diagonal of hessian for forward model loss.
            weights (jax array, optional): 3D positive weights with same shape as sinogram.  Defaults to all 1s.
            prox_input (jax array): optional input for proximal map with same shape as reconstruction.

        Returns:
            (callable) vcd_subset_updater(error_sinogram, flat_recon, pixel_indices) that updates the recon.
        """

        positivity_flag = self.get_params('positivity_flag')
        fm_constant = 1.0 / (self.get_params('sigma_y') ** 2.0)
        qggmrf_nbr_wts, sigma_x, p, q, T = self.get_params(['qggmrf_nbr_wts', 'sigma_x', 'p', 'q', 'T'])
        b = mbirjax.get_b_from_nbr_wts(qggmrf_nbr_wts)
        qggmrf_params = tuple((b, sigma_x, p, q, T))
        sigma_prox = self.get_params('sigma_prox')
        recon_shape = self.get_params('recon_shape')
        sparse_back_project = self.sparse_back_project
        sparse_forward_project = self.sparse_forward_project
        try:
            const_weights = False
            sinogram_shape = self.get_params('sinogram_shape')
            if weights.shape != sinogram_shape:
                raise ValueError('weights must be a constant or have the same shape as sinogram.')
        except AttributeError:
            eps = 1e-5
            if np.abs(weights - 1) > eps:
                raise ValueError('Constant weights must have value 1.')
            const_weights = True

        def vcd_subset_updater(flat_recon, error_sinogram, pixel_indices, times):
            """
            Calculate an iteration of the VCD algorithm on a single subset of the partition
            Each iteration of the algorithm should return a better reconstructed recon.
            The combination of (error_sinogram, recon) forms an overcomplete state that makes computation efficient.
            However, it is important that at each application the state should meet the constraint that:
            error_sinogram = measured_sinogram - forward_proj(recon)
            where measured_sinogram forward_proj() is whatever forward projection is being used in reconstruction.

            Args:
                flat_recon (jax array): 2D array reconstruction with shape (num_recon_rows x num_recon_cols, num_recon_slices).
                error_sinogram (jax array): 3D error sinogram with shape (num_views, num_det_rows, num_det_channels).
                pixel_indices (jax array): 1D array of pixel indices.

            Returns:
                flat_recon, error_sinogram, norm_squared_for_subset, alpha_for_subset:
                The first two have the same shape as above, but are updated to reduce the overall loss function.
                norm_squared is for the change to the recon from this one subset.
                alpha is the relative step size for this subset.
            """

            # Compute the forward model gradient and hessian at each pixel in the index set.
            # Assumes Loss(delta) = 1/(2 sigma_y^2) || error_sinogram - A delta ||_weights^2
            # time_index = 0
            # time_start = time.time()
            if not const_weights:
                weighted_error_sinogram = weights * error_sinogram  # Note that fm_constant will be included below
            else:
                weighted_error_sinogram = error_sinogram
            # weighted_error_sinogram = weighted_error_sinogram.block_until_ready()
            # times[time_index] += time.time() - time_start
            # time_index += 1

            # Transfer to worker for later use
            # time_start = time.time()
            pixel_indices_worker = jax.device_put(pixel_indices, self.worker)

            # Back project to get the gradient
            forward_grad = - fm_constant * sparse_back_project(weighted_error_sinogram, pixel_indices_worker,
                                                               output_device=self.main_device)
            # forward_grad = forward_grad.block_until_ready()
            # times[time_index] += time.time() - time_start
            # time_index += 1

            # Get the forward hessian for this subset
            # time_start = time.time()
            forward_hess = fm_constant * fm_hessian[pixel_indices]
            # forward_hess = forward_hess.block_until_ready()
            # times[time_index] += time.time() - time_start
            # time_index += 1

            # Compute the prior model gradient and hessian (i.e., second derivative) terms
            # time_start = time.time()
            if prox_input is None:
                # qGGMRF prior - compute the qggmrf gradient and hessian at each pixel in the index set.
                with jax.default_device(self.main_device):
                    prior_grad, prior_hess = (
                        mbirjax.qggmrf_gradient_and_hessian_at_indices(flat_recon, recon_shape, pixel_indices,
                                                                       qggmrf_params))
            else:
                # Proximal map prior - compute the prior model gradient at each pixel in the index set.
                prior_hess = sigma_prox ** 2
                prior_grad = mbirjax.prox_gradient_at_indices(flat_recon, prox_input, pixel_indices, sigma_prox)
            # prior_grad = prior_grad.block_until_ready()
            # times[time_index] += time.time() - time_start
            # time_index += 1

            # Compute update vector update direction in recon domain
            # time_start = time.time()
            delta_recon_at_indices = - ((forward_grad + prior_grad) / (forward_hess + prior_hess))
            # delta_recon_at_indices = delta_recon_at_indices.block_until_ready()
            # times[time_index] += time.time() - time_start
            # time_index += 1

            # Compute delta^T \nabla Q(x_hat; x'=x_hat) for use in finding alpha
            # time_start = time.time()
            prior_linear = jnp.sum(prior_grad * delta_recon_at_indices)
            # prior_linear = prior_linear.block_until_ready()
            # times[time_index] += time.time() - time_start
            # time_index += 1

            # Estimated upper bound for hessian
            # time_start = time.time()
            prior_overrelaxation_factor = 2
            prior_quadratic_approx = ((1 / prior_overrelaxation_factor) *
                                      jnp.sum(prior_hess * delta_recon_at_indices ** 2))
            # prior_quadratic_approx = prior_quadratic_approx.block_until_ready()
            # times[time_index] += time.time() - time_start
            # time_index += 1

            # Compute update direction in sinogram domain
            # time_start = time.time()
            delta_sinogram = sparse_forward_project(delta_recon_at_indices, pixel_indices_worker,
                                                    output_device=self.worker)
            # delta_sinogram = delta_sinogram.block_until_ready()
            # times[time_index] += time.time() - time_start
            # time_index += 1

            # time_start = time.time()
            forward_linear, forward_quadratic = self.get_forward_lin_quad(weighted_error_sinogram, delta_sinogram,
                                                                          weights, fm_constant, const_weights)

            # Compute optimal update step
            alpha_numerator = forward_linear - prior_linear
            alpha_denominator = forward_quadratic + prior_quadratic_approx + jnp.finfo(jnp.float32).eps
            alpha = alpha_numerator / alpha_denominator
            max_alpha = 1.5
            alpha = jnp.clip(alpha, jnp.finfo(jnp.float32).eps, max_alpha)  # a_max=alpha_clip_value
            # alpha = alpha.block_until_ready()
            # times[time_index] += time.time() - time_start
            # time_index += 1

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
                delta_recon_at_indices = jax.device_put(delta_recon_at_indices, self.main_device)
                delta_recon_at_indices = jnp.maximum(-pos_constant * recon_at_indices, delta_recon_at_indices)

                # Recompute sinogram projection
                delta_sinogram = sparse_forward_project(delta_recon_at_indices, pixel_indices, output_device=self.worker)

            # time_start = time.time()
            # Perform sparse updates at index locations
            delta_recon_at_indices = jax.device_put(delta_recon_at_indices, self.main_device)
            delta_recon_at_indices = alpha * delta_recon_at_indices
            # delta_recon_at_indices = delta_recon_at_indices.block_until_ready()
            # times[time_index] += time.time() - time_start
            # time_index += 1

            # time_start = time.time()
            flat_recon = update_recon(flat_recon, pixel_indices, delta_recon_at_indices)
            # flat_recon = flat_recon.block_until_ready()
            # times[time_index] += time.time() - time_start
            # time_index += 1

            # Update sinogram and loss
            # time_start = time.time()
            delta_sinogram = float(alpha) * jax.device_put(delta_sinogram, self.main_device)
            error_sinogram = error_sinogram - delta_sinogram
            # error_sinogram = error_sinogram.block_until_ready()
            # times[time_index] += time.time() - time_start
            # time_index += 1

            # time_start = time.time()
            norm_squared_for_subset = jnp.sum(delta_recon_at_indices * delta_recon_at_indices)
            alpha_for_subset = alpha
            # norm_squared_for_subset = norm_squared_for_subset.block_until_ready()
            # times[time_index] += time.time() - time_start
            # time_index += 1

            return flat_recon, error_sinogram, norm_squared_for_subset, alpha_for_subset, times

        return vcd_subset_updater

    def get_forward_lin_quad(self, weighted_error_sinogram, delta_sinogram, weights, fm_constant, const_weights):
        """
        Compute
            forward_linear = fm_constant * jnp.sum(weighted_error_sinogram * delta_sinogram)
            forward_quadratic = fm_constant * jnp.sum(delta_sinogram * delta_sinogram * weights)
        with batching to the worker if needed, which is feasible since the data transfer is mostly from
        the main deice to the worker, with only 2 floats sent back with each batch.

        Args:
            weighted_error_sinogram (jax array):
            delta_sinogram (jax array):
            weights (jax array or constant):
            fm_constant (constant):
            const_weights (bool): True if the weights are constant 1

        Returns:
            tuple:
            forward_linear, forward_quadratic
        """
        num_views = weighted_error_sinogram.shape[0]
        views_per_batch = self.views_per_batch

        # If this can be done without data transfer, then do it.
        if self.worker == self.main_device:
            forward_linear = fm_constant * jnp.sum(weighted_error_sinogram * delta_sinogram)
            forward_quadratic = fm_constant * jnp.sum(delta_sinogram * delta_sinogram * weights)
            return forward_linear, forward_quadratic

        # Otherwise batch the sinogram by view, send to the worker and calculate linear and quadratic terms
        # First apply the batch projector directly to an initial batch to get the initial output
        views_per_batch = num_views if views_per_batch is None else views_per_batch
        num_remaining = num_views % views_per_batch

        # If the input is a multiple of batch_size, then we'll do a full batch, otherwise just the excess.
        initial_batch_size = views_per_batch if num_remaining == 0 else num_remaining
        # Make the weights into a 1D vector along views if it's a constant

        def linear_quadratic(start_ind, stop_ind, previous_linear=0, previous_quadratic=0):
            """
            Send a batch to the worker and compute forward linear and quadratic for that batch.
            """
            worker_wes = jax.device_put(weighted_error_sinogram[start_ind:stop_ind], self.worker)
            worker_ds = jax.device_put(delta_sinogram[start_ind:stop_ind], self.worker)
            if not const_weights:
                worker_wts = jax.device_put(weights[start_ind:stop_ind], self.worker)
            else:
                worker_wts = weights

            # previous_linear += fm_constant * jnp.sum(worker_wes * worker_ds)
            # previous_quadratic += fm_constant * jnp.sum(worker_ds * worker_ds * worker_wts)

            previous_linear += sum_product(worker_wes, worker_ds)
            worker_ds = jax.vmap(jnp.multiply)(worker_ds, worker_ds)
            if not const_weights:
                quadratic_entries = sum_product(worker_ds, worker_wts)
            else:
                quadratic_entries = jnp.sum(worker_ds)
            previous_quadratic += quadratic_entries

            del worker_wes, worker_ds, worker_wts
            return previous_linear, previous_quadratic

        forward_linear, forward_quadratic = linear_quadratic(0, initial_batch_size)

        # Then deal with the batches if there are any
        if views_per_batch < num_views:
            num_batches = (num_views - initial_batch_size) // views_per_batch
            for j in jnp.arange(num_batches):
                start_ind_j = initial_batch_size + j * views_per_batch
                stop_ind_j = start_ind_j + views_per_batch
                forward_linear, forward_quadratic = linear_quadratic(start_ind_j, stop_ind_j,
                                                                     forward_linear, forward_quadratic)

        forward_linear = jax.device_put(forward_linear, self.main_device)
        forward_quadratic = jax.device_put(forward_quadratic, self.main_device)
        return fm_constant * forward_linear, fm_constant * forward_quadratic

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

    def gen_weights_mar(self, sinogram, init_recon=None, metal_threshold=None, beta=1.0, gamma=3.0):
        """
        Generates the weights used for reducing metal artifacts in MBIR reconstruction.

        This function computes sinogram weights that help to reduce metal artifacts.
        More specifically, it computes weights with the form:

            weights = exp( -(sinogram/beta) * ( 1 + gamma * delta(metal) )

        delta(metal) denotes a binary mask indicating the sino entries that contain projections of metal.
        Providing ``init_recon`` yields better metal artifact reduction.
        If not provided, the metal segmentation is generated directly from the sinogram.

        Args:
            sinogram (jax array): 3D jax array containing sinogram with shape (num_views, num_det_rows, num_det_channels).
            init_recon (jax array, optional): An initial reconstruction used to identify metal voxels. If not provided, Otsu's method is used to directly segment sinogram into metal regions.
            metal_threshold (float, optional): Values in ``init_recon`` above ``metal_threshold`` are classified as metal. If not provided, Otsu's method is used to segment ``init_recon``.
            beta (float, optional): Scalar value in range :math:`>0`.
                A larger ``beta`` improves the noise uniformity, but too large a value may increase the overall noise level.
            gamma (float, optional): Scalar value in range :math:`>=0`.
                A larger ``gamma`` reduces the weight of sinogram entries with metal, but too large a value may reduce image quality inside the metal regions.

        Returns:
            (jax array): Weights used in mbircone reconstruction, with the same array shape as ``sinogram``
        """
        return mbirjax.gen_weights_mar(self, sinogram, init_recon=init_recon, metal_threshold=metal_threshold,
                                       beta=beta, gamma=gamma)

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

    def scale_recon_shape(self, row_scale=1.0, col_scale=1.0, slice_scale=1.0):
        """
        Scale the recon shape by the given factors.  This can be used before starting a reconstruction to improve the
        reconstruction when part of the object projects outside the detector.

        Args:
            row_scale (float): Scale for the recon rows.
            col_scale (float): Scale for the recon columns.
            slice_scale (float): Scale for the recon slices.
        """
        num_rows, num_cols, num_slices = self.get_params('recon_shape')
        num_rows = int(num_rows * row_scale)
        num_cols = int(num_cols * col_scale)
        num_slices = int(num_slices * slice_scale)
        self.set_params(recon_shape=(num_rows, num_cols, num_slices))


from functools import partial


@partial(jax.jit, donate_argnames='cur_flat_recon')
def update_recon(cur_flat_recon, cur_indices, cur_delta):
    cur_flat_recon = cur_flat_recon.at[cur_indices].add(cur_delta)
    return cur_flat_recon


@jax.jit
def sum_product(array0, array1):
    prod = jax.vmap(jnp.multiply)(array0, array1)
    sum_of_prod = jax.vmap(jnp.sum)(prod)
    sum_of_prod = jnp.sum(sum_of_prod)
    return sum_of_prod


def get_transpose(linear_map, input_shape):
    """
    Use jax to determine the transpose of a linear map.

    Args:
        linear_map:  [function] The linear function to be transposed
        input_shape: [ndarray] The shape of the input to the function

    Returns:
        transpose: A function to evaluate the transpose of the given map.  The input to transpose
        must be a jax or ndarray with the same shape as the output of the original linear_map.
        transpose(input) returns an array of shape input_shape.
    """
    # print('Defining transpose map')
    # t0 = time.time()
    input_info = types.SimpleNamespace(shape=input_shape, dtype=jnp.dtype(jnp.float32))
    transpose_list = jax.linear_transpose(linear_map, input_info)

    def transpose(input):
        return transpose_list(input)[0]

    return transpose
