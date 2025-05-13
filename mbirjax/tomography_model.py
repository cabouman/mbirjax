import types
import numpy as np
import warnings
import time  # Used for debugging/performance tuning
import os

from jaxlib.xla_extension import XlaRuntimeError

num_cpus = 3 * os.cpu_count() // 4
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count={}'.format(num_cpus)
import jax
import jax.numpy as jnp
import mbirjax
from mbirjax import ParameterHandler
from collections import namedtuple
import subprocess
import re
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
# Set the GPU memory fraction for JAX
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.98'


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

        self.main_device, self.sinogram_device, self.worker = None, None, None
        self.cpus = jax.devices('cpu')
        self.projector_functions = None

        # The following may be adjusted based on memory in set_devices_and_batch_sizes()
        self.view_batch_size_for_vmap = 512
        self.pixel_batch_size_for_vmap = 2048
        self.transfer_pixel_batch_size = 100 * self.pixel_batch_size_for_vmap
        self.gpu_memory = 0
        self.cpu_memory = 0
        self.mem_required_for_gpu = 0
        self.mem_required_for_cpu = 0
        self.use_gpu = 'none'  # This is set in set_devices_and_batch_sizes based on memory and get_params('use_gpu')
        self.set_devices_and_batch_sizes()
        self.create_projectors()

    def set_devices_and_batch_sizes(self):
        """
        Determine how much memory is required for each of projections, all the sinograms needed for vcd, and all
        the recons needed for vcd, then determine whether to use the GPU for projections only, or for all sinograms,
        or for the entire reconstruction, or nothing.

        This determination can be overridden by using ct_model.set_params(use_gpu=string), where string is one of
        'automatic', 'full', 'sinograms', 'projections', 'none'

        Returns:
            Nothing, but instance variables are set to appropriate values.
        """

        # Get the cpu and any gpus
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
                warnings.warn("'use_gpu' is set to {} but no gpu is available. Proceeding on cpu. "
                              "Use 'set_params(use_gpu='automatic') to avoid this warning.".format(use_gpu))
            gpus = []
            gpu_memory = 0
        self.gpu_memory = gpu_memory

        # Estimate the CPU memory available
        cpu_memory = 0
        try:
            # On SLURM at Purdue, we can parse the job info to determine the allocated memory
            status = subprocess.run(['scontrol', 'show', 'job', os.environ['SLURM_JOB_ID']], check=True, text=True,
                                    capture_output=True)
            # status.stdout is an output string with multiple lines, one of which looks like this:
            #   ReqTRES=cpu=42,mem=386400M,node=1,billing=1,gres/gpu=1
            # Use a regular expression to capture the digits and one letter between 'mem=' and ',node'
            pattern = r"mem=(\d+)([A-Za-z]),node"
            match = re.search(pattern, status.stdout)
            if match:
                number = int(match.group(1))  # Capture the digits
                letter = match.group(2)  # Capture the letter
                # Convert the indicated memory to GB
                scales = ['K', 'M', 'G', 'T']
                scale_factor = scales.index(letter) - scales.index('G')
                cpu_memory = number * (1024 ** scale_factor)

        except Exception:  # If anything goes wrong, we'll just continue without detailed CPU memory info.
            pass

        if cpu_memory == 0:
            cpu_memory_stats = mbirjax.get_memory_stats(print_results=False)[-1]
            cpu_memory = float(cpu_memory_stats['bytes_limit']) - float(cpu_memory_stats['bytes_in_use'])
            cpu_memory /= gb
        self.cpu_memory = cpu_memory

        # Get basic parameters
        sinogram_shape = self.get_params('sinogram_shape')
        num_views, num_det_rows, num_det_channels = sinogram_shape
        recon_shape = self.get_params('recon_shape')
        num_slices = recon_shape[2]

        zero = jnp.zeros(1)
        bits_per_byte = 8
        mem_per_entry = float(str(zero.dtype)[5:]) / bits_per_byte / gb  # Parse floatXX to get the number of bits
        mem_per_cylinder = num_slices * mem_per_entry

        # Make an empirical estimate of memory used per projection (on H100 as of 2025):
        # vmap works in parallel over a batch of views, so we have view_batch_size_for_vmap * mem_per_view,
        # with a rough floor on the number of channels, all multiplied by a constant factor from the implementation of
        # the projectors.
        mem_per_view_with_floor = mem_per_entry * num_det_rows * max(num_det_channels, 512)
        cone_beam_projection_factor = 16  # This says cone beam but is very similar for parallel beam
        mem_per_projection =  cone_beam_projection_factor * self.view_batch_size_for_vmap * mem_per_view_with_floor
        mem_per_voxel_batch = mem_per_cylinder * self.transfer_pixel_batch_size

        # Make an estimate of the memory needed to do all sinogram processing and projections on gpu
        # To do all sinos on the GPU, we use the greater of the following two since sino_reps_for_vcd includes
        # copies of the sino that are not used during projection:
        #       sino_reps_for_vcd * mem_per_sinogram + mem_per_voxel_batch
        #       mem_for_minimal_vcd_sinos + mem_per_projection + mem_per_voxel_batch
        sino_reps_for_vcd = 6  # error sinogram, weights, weighted error sinogram, delta sinogram, 2 intermediate copies
        sino_reps_minimal = 3  # error sinogram, weights, weighted error sinogram
        mem_per_sinogram = mem_per_entry * num_views * num_det_rows * num_det_channels

        # The memory when all sinograms are on GPU appears to have a floor on detector rows.
        mem_per_sinogram_with_floor = mem_per_entry * num_views * max(num_det_rows, 100) * num_det_channels
        mem_for_vcd_sinos_gpu = sino_reps_for_vcd * mem_per_sinogram_with_floor
        mem_for_minimal_vcd_sinos_gpu = sino_reps_minimal * mem_per_sinogram_with_floor

        mem_for_all_sinos_on_gpu = max(mem_for_vcd_sinos_gpu, mem_for_minimal_vcd_sinos_gpu + mem_per_projection) + mem_per_voxel_batch

        # Reducing vmap batch size can reduce memory, but reducing below 128 = 512 / 4 increases time substantially.
        # Estimate the memory required in the minimal case.
        mem_for_minimal_sinos_on_gpu = max(mem_for_vcd_sinos_gpu, mem_for_minimal_vcd_sinos_gpu + mem_per_projection / 4) + mem_per_voxel_batch

        mem_per_recon = mem_per_entry * np.prod(recon_shape)
        recon_reps_for_vcd = 6
        mem_for_all_vcd = recon_reps_for_vcd * mem_per_recon + mem_for_all_sinos_on_gpu - mem_per_voxel_batch

        frac_gpu_mem_to_use = 0.9
        gpu_memory_to_use = frac_gpu_mem_to_use * gpu_memory

        # 'full':  Everything on GPU
        if use_gpu == 'full' or (mem_for_all_vcd < gpu_memory_to_use and use_gpu not in ['none', 'projections', 'sinograms']):
            self.main_device, self.sinogram_device, self.worker = gpus[0], gpus[0], gpus[0]
            self.use_gpu = 'full'
            mem_required_for_gpu = mem_for_all_vcd
            mem_required_for_cpu = 2 * mem_per_recon + 2 * mem_per_sinogram  # recon plus sino and weights

        # 'sinograms': All sinos and projections on GPU.  Adjust projection vmap batch size if needed.
        elif use_gpu == 'sinograms' or (mem_for_minimal_sinos_on_gpu < gpu_memory_to_use and use_gpu not in ['none', 'projections']):
            self.main_device, self.sinogram_device, self.worker = cpus[0], gpus[0], gpus[0]
            self.use_gpu = 'sinograms'
            mem_avail_for_projection = gpu_memory_to_use - mem_per_voxel_batch - mem_for_minimal_vcd_sinos_gpu
            projection_scale = min(1, mem_avail_for_projection / mem_per_projection)
            max_view_batch_size = int(self.view_batch_size_for_vmap * projection_scale)
            num_batches = np.ceil(num_views / max_view_batch_size).astype(int)
            self.view_batch_size_for_vmap = np.ceil(num_views / num_batches).astype(int)

            # Recalculate the memory per projection with the new batch size
            mem_per_projection = cone_beam_projection_factor * self.view_batch_size_for_vmap * mem_per_view_with_floor

            mem_required_for_gpu = max(mem_for_vcd_sinos_gpu,
                                       mem_for_minimal_vcd_sinos_gpu + mem_per_projection) + mem_per_voxel_batch
            mem_required_for_cpu = recon_reps_for_vcd * mem_per_recon + 2 * mem_per_sinogram  # All recons plus sino and weights

        # 'projections': Only projections on GPU.  Adjust projection vmap batch size if needed.
        elif use_gpu == 'projections' or (mem_per_projection / 16 < gpu_memory_to_use and use_gpu not in ['none']):
            self.main_device, self.sinogram_device, self.worker = cpus[0], cpus[0], gpus[0]
            self.use_gpu = 'projections'
            mem_avail_for_projection = gpu_memory_to_use - mem_per_voxel_batch
            projection_scale = min(1, mem_avail_for_projection / mem_per_projection)
            max_view_batch_size = int(self.view_batch_size_for_vmap * projection_scale)
            num_batches = np.ceil(num_views / max_view_batch_size).astype(int)
            self.view_batch_size_for_vmap = np.ceil(num_views / num_batches).astype(int)

            # Recalculate the memory per projection with the new batch size
            mem_per_projection = cone_beam_projection_factor * self.view_batch_size_for_vmap * mem_per_view_with_floor

            mem_required_for_gpu = mem_per_projection
            mem_required_for_cpu = recon_reps_for_vcd * mem_per_recon + sino_reps_for_vcd * mem_per_sinogram

        # 'none': All on CPU
        else:
            if gpu_memory > 0:
                warnings.warn('MBIRJAX is installed with cuda, but there is not enough GPU memory to use cuda. This may lead to a fatal error.')

            self.main_device, self.sinogram_device, self.worker = cpus[0], cpus[0], cpus[0]
            self.use_gpu = 'none'

            mem_required_for_gpu = 0
            mem_required_for_cpu = mem_for_all_vcd

        if cpu_memory < mem_required_for_cpu:
            warnings.warn('CPU memory may be insufficient for this problem.  This may lead to a fatal error.')

        self.mem_required_for_gpu = mem_required_for_gpu
        self.mem_required_for_cpu = mem_required_for_cpu

        verbose = self.get_params('verbose')
        if verbose >= 2:
            print('mem per recon = {}'.format(mem_per_recon))
            print('mem per sino = {}'.format(mem_per_sinogram))
            print('mem per projection = {}'.format(mem_per_projection))
            print('mem for vcd sinograms = {}'.format(mem_for_vcd_sinos_gpu))
            print('mem for all sinos on gpu = {}'.format(mem_for_all_sinos_on_gpu))
            print('mem for all vcd = {}'.format(mem_for_all_vcd))
            print('view_batch_size_for_vmap = {}'.format(self.view_batch_size_for_vmap))

        return

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
        output_device = self.main_device
        sinogram = self.sparse_forward_project(voxel_values, full_indices, output_device=output_device)

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
        output_device = self.main_device
        recon_cylinder = self.sparse_back_project(sinogram, full_indices, output_device=output_device)
        row_index, col_index = jnp.unravel_index(full_indices, recon_shape[:2])
        recon = jnp.zeros(recon_shape, device=output_device)
        recon = recon.at[row_index, col_index].set(recon_cylinder)
        return recon

    def sparse_forward_project(self, voxel_values, pixel_indices, view_indices=None, output_device=None):
        """
        Forward project the given voxel values to a sinogram.
        The indices are into a flattened 2D array of shape (recon_rows, recon_cols), and the projection is done using
        all voxels with those indices across all the slices.

        Args:
            voxel_values (jax.numpy.DeviceArray): 2D array of voxel values to project, size (len(pixel_indices), num_recon_slices).
            pixel_indices (jax array): Array of indices specifying which voxels to project.
            view_indices (jax array): Array of indices of views to project
            output_device (jax device): Device on which to put the output

        Returns:
            jnp array: The resulting 3D sinogram after projection.
        """
        # Batch the views and pixels for possible transfer to the gpu
        transfer_view_batch_size = self.view_batch_size_for_vmap
        transfer_pixel_batch_size = self.transfer_pixel_batch_size
        sinogram_shape = self.get_params('sinogram_shape')
        if view_indices is None:
            view_indices = jnp.arange(sinogram_shape[0])
        num_view_batches = jnp.ceil(sinogram_shape[0] / transfer_view_batch_size).astype(int)
        view_indices_batched = jnp.array_split(view_indices, num_view_batches)
        sinogram_shape = self.get_params('sinogram_shape')

        num_pixels = len(pixel_indices)
        pixel_batch_boundaries = np.arange(start=0, stop=num_pixels, step=transfer_pixel_batch_size)
        pixel_batch_boundaries = np.append(pixel_batch_boundaries, num_pixels)

        sinogram = []
        for view_indices_batch in view_indices_batched:
            sinogram_views = jnp.zeros((len(view_indices_batch), *sinogram_shape[1:]), device=self.worker)
            # Loop over pixel batches
            for k, pixel_index_start in enumerate(pixel_batch_boundaries[:-1]):
                # Send a batch of pixels to worker
                pixel_index_end = pixel_batch_boundaries[k + 1]
                voxel_batch, pixel_index_batch = jax.device_put([voxel_values[pixel_index_start:pixel_index_end],
                                                                 pixel_indices[pixel_index_start:pixel_index_end]],
                                                                self.worker)
                sinogram_views = sinogram_views.block_until_ready()
                sinogram_views = sinogram_views + self.projector_functions.sparse_forward_project(voxel_batch, pixel_index_batch, view_indices=view_indices_batch)

            # Include these views in the sinogram
            sinogram.append(jax.device_put(sinogram_views, output_device))

        sinogram = jnp.concatenate(sinogram)
        return sinogram

    def sparse_back_project(self, sinogram, pixel_indices, view_indices=None, coeff_power=1, output_device=None):
        """
        Back project the given sinogram to the voxels given by the indices.
        The indices are into a flattened 2D array of shape (recon_rows, recon_cols), and the projection is done using
        all voxels with those indices across all the slices.

        Args:
            sinogram (jnp array): 3D jax array containing sinogram.
            pixel_indices (jnp array): Array of indices specifying which voxels to back project.
            view_indices (jax array): Array of indices of views to project
            coeff_power (int, optional): Normally 1, but set to 2 for Hessian diagonal
            output_device (jax device, optional): Device on which to put the output

        Returns:
            A jax array of shape (len(indices), num_slices)
        """
        # Batch the views and pixels for possible transfer to the gpu
        transfer_view_batch_size = self.view_batch_size_for_vmap
        transfer_pixel_batch_size = self.transfer_pixel_batch_size
        num_views = sinogram.shape[0]
        if view_indices is None:
            view_indices = jnp.arange(num_views)
        num_view_batches = jnp.ceil(sinogram.shape[0] / transfer_view_batch_size).astype(int)
        view_indices_batched = jnp.array_split(view_indices, num_view_batches)
        view_batch_boundaries = [view_indices_batched[j][0] for j in range(len(view_indices_batched))]
        view_batch_boundaries = np.append(np.array(view_batch_boundaries), num_views)

        pixel_indices = jax.device_put(pixel_indices, self.worker)
        num_pixel_batches = jnp.ceil(pixel_indices.shape[0] / transfer_pixel_batch_size).astype(int)
        pixel_indices_batched = jnp.array_split(pixel_indices, num_pixel_batches)

        recon_shape = self.get_params('recon_shape')
        num_pixels = len(pixel_indices)
        num_slices = recon_shape[2]

        # Get the final recon as a jax array
        recon_at_indices = jnp.zeros((num_pixels, num_slices), device=output_device)
        for j, view_index_start in enumerate(view_batch_boundaries[:-1]):
            view_index_end = view_batch_boundaries[j + 1]
            view_batch = sinogram[view_index_start:view_index_end]
            view_batch = jax.device_put(view_batch, self.worker)

            # Loop over pixel batches
            voxel_batch_list = []
            for pixel_index_batch in pixel_indices_batched:
                # Back project a batch
                voxel_batch = self.projector_functions.sparse_back_project(view_batch, pixel_index_batch,
                                                                           view_indices=view_indices_batched[j],
                                                                           coeff_power=coeff_power)
                voxel_batch = voxel_batch.block_until_ready()
                voxel_batch_list.append(jax.device_put(voxel_batch, output_device))

            recon_at_indices = recon_at_indices + jnp.concatenate(voxel_batch_list, axis=0)

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

        if use_gpu not in ['automatic', 'full', 'sinograms', 'projections', 'none']:
            error_message = "use_gpu must be one of \n"
            error_message += " 'automatic' (code will try to determine problem size and use gpu appropriately),\n'"
            error_message += " 'full' (use gpu for all calculations),\n"
            error_message += " 'sinograms' (use gpu for projections and all copies of sinogram needed for vcd),\n"
            error_message += " 'projections' (use gpu for projections only),\n"
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
            max_views_to_use = np.minimum(20, sinogram.shape[0])
            step_size = sinogram.shape[0] // max_views_to_use

            small_sinogram = np.array(sinogram[::step_size])
            if weights is None:
                small_weights = 1
            else:
                small_weights = np.array(weights[::step_size])
            # Compute indicator function for sinogram support
            sino_indicator = self._get_sino_indicator(small_sinogram)
            self.auto_set_sigma_y(small_sinogram, sino_indicator, small_weights)

            recon_std = self._get_estimate_of_recon_std(small_sinogram, sino_indicator)
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

    def direct_recon(self, sinogram, filter_name=None, view_batch_size=100):
        """
        Do a direct (non-iterative) reconstruction, typically using a form of filtered backprojection.  The
        implementation details are geometry specific, and direct_recon may not be available for all geometries.

        Args:
            sinogram (ndarray or jax array): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels).
            filter_name (string or None, optional): The name of the filter to use, defaults to None, in which case the geometry specific method chooses a default, typically 'ramp'.
            view_batch_size (int, optional): An integer specifying the size of a view batch to limit memory use.  Defaults to 100.

        Returns:
            recon (jax array): The reconstructed volume after direct reconstruction.
        """
        warnings.warn('direct_recon not implemented for TomographyModel.')
        recon_shape = self.get_params('recon_shape')
        return jnp.zeros(recon_shape, device=self.main_device)

    def recon(self, sinogram, weights=None, init_recon=None, max_iterations=15, stop_threshold_change_pct=0.2, first_iteration=0,
              compute_prior_loss=False, num_iterations=None):
        """
        Perform MBIR reconstruction using the Multi-Granular Vector Coordinate Descent algorithm.
        This function takes care of generating its own partitions and partition sequence.
        TO restart a recon using the same partition sequence, set first_iteration to be the number of iterations
        completed so far, and set init_recon to be the output of the previous recon.  This will continue using
        the same partition sequence from where the previous recon left off.

        Args:
            sinogram (ndarray or jax array): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels).
            weights (ndarray or jax array, optional): 3D positive weights with same shape as error_sinogram.  Defaults to None, in which case the weights are implicitly all 1.
            init_recon (jax array or None or 0, optional): Initial reconstruction to use in reconstruction. If None, then direct_recon is called with default arguments.  Defaults to None.
            max_iterations (int, optional): maximum number of iterations of the VCD algorithm to perform.
            stop_threshold_change_pct (float, optional): Stop reconstruction when 100 * ||delta_recon||_1 / ||recon||_1 change from one iteration to the next is below stop_threshold_change_pct.  Defaults to 0.2.  Set this to 0 to guarantee exactly max_iterations.
            first_iteration (int, optional): Set this to be the number of iterations previously completed when restarting a recon using init_recon.  This defines the first index in the partition sequence.  Defaults to 0.
            compute_prior_loss (bool, optional):  Set true to calculate and return the prior model loss.  This will lead to slower reconstructions and is meant only for small recons.
            num_iterations (int, optional): This option is deprecated and will be used to set max_iterations if this is not None.  Defaults to None.

        Returns:
            [recon, recon_params]: reconstruction and a named tuple containing the recon parameters.
            recon_params (namedtuple): max_iterations, granularity, partition_sequence, fm_rmse, prior_loss, regularization_params
        """
        if num_iterations is not None:
            warnings.warn('num_iterations has been deprecated and will be removed in a future release.\nIn the current run, the value of num_iterations will be used to set max_iterations.')
            max_iterations = num_iterations

        if self.get_params('verbose') >= 1:
            print('GPU used for: {}'.format(self.use_gpu))
            print('Estimated GPU memory required = {:.3f} GB, available = {:.3f} GB'.format(self.mem_required_for_gpu, self.gpu_memory))
            print('Estimated CPU memory required = {:.3f} GB, available = {:.3f} GB'.format(self.mem_required_for_cpu, self.cpu_memory))

        try:

            # Check that sinogram and weights are not taking up GPU space
            if isinstance(sinogram, type(jnp.zeros(1))) and list(sinogram.devices())[0] != self.sinogram_device:
                sinogram = jax.device_put(sinogram, self.sinogram_device)
            if weights is not None and isinstance(weights, type(jnp.zeros(1))) and list(weights.devices())[0] != self.sinogram_device:
                weights = jax.device_put(weights, self.sinogram_device)
            if init_recon is not None and isinstance(init_recon, type(jnp.zeros(1))) and list(init_recon.devices())[0] != self.main_device:
                init_recon = jax.device_put(init_recon, self.main_device)

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
            partition_sequence = mbirjax.gen_partition_sequence(partition_sequence, max_iterations=max_iterations)
            partition_sequence = partition_sequence[first_iteration:]

            # Compute reconstruction
            recon, loss_vectors = self.vcd_recon(sinogram, partitions, partition_sequence, weights=weights,
                                                 init_recon=init_recon, compute_prior_loss=compute_prior_loss,
                                                 first_iteration=first_iteration,
                                                 stop_threshold_change_pct=stop_threshold_change_pct)

            # Return num_iterations, granularity, partition_sequence, fm_rmse values, regularization_params
            recon_param_names = ['num_iterations', 'granularity', 'partition_sequence', 'fm_rmse', 'prior_loss',
                                 'regularization_params', 'stop_threshold_change_pct', 'alpha_values']
            ReconParams = namedtuple('ReconParams', recon_param_names)
            partition_sequence = [int(val) for val in partition_sequence]
            fm_rmse = [float(val) for val in loss_vectors[0]]
            if compute_prior_loss:
                prior_loss = [float(val) for val in loss_vectors[1]]
            else:
                prior_loss = [0]
            stop_threshold_change_pct = [100 * float(val) for val in loss_vectors[2]]
            alpha_values = [float(val) for val in loss_vectors[3]]
            num_iterations = len(fm_rmse)
            recon_param_values = [num_iterations, granularity, partition_sequence, fm_rmse, prior_loss,
                                  regularization_params._asdict(), stop_threshold_change_pct, alpha_values]
            recon_params = ReconParams(*tuple(recon_param_values))

        except MemoryError as e:
            print('Insufficient CPU memory')
            raise e
        except XlaRuntimeError as e:
            print(e)
            if self.gpu_memory > 0:
                if self.mem_required_for_gpu / self.gpu_memory < self.mem_required_for_cpu / self.cpu_memory:
                    print('Insufficient memory for jax (likely insufficient CPU memory)')
                else:
                    print('Insufficient memory for jax (likely insufficient GPU memory)')
                    if self.use_gpu == 'full':
                        print(">>> You may try using ct_model.set_params(use_gpu='sinograms') before calling recon")
                    elif self.use_gpu == 'sinograms':
                        print(">>> You may try using ct_model.set_params(use_gpu='projections') before calling recon")
            else:
                print('Insufficient memory for jax (insufficient CPU memory)')

            raise e

        return recon, recon_params

    def vcd_recon(self, sinogram, partitions, partition_sequence, stop_threshold_change_pct, weights=None,
                  init_recon=None, prox_input=None, compute_prior_loss=False, first_iteration=0):
        """
        Perform MBIR reconstruction using the Multi-Granular Vector Coordinate Descent algorithm
        for a given set of partitions and a prescribed partition sequence.

        Args:
            sinogram (jax array): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels).
            partitions (tuple or list): A collection of K partitions, with each partition being an (N_indices) integer index array of voxels to be updated in a flattened recon.
            partition_sequence (jax array): A sequence of integers that specify which partition should be used at each iteration.
            stop_threshold_change_pct (float): Stop reconstruction when NMAE percent change from one iteration to the next is below stop_threshold_change_pct.
            weights (jax array, optional): 3D positive weights with same shape as error_sinogram.  Defaults to all 1s.
            init_recon (jax array or None or 0, optional): Initial reconstruction to use in reconstruction. If None, then direct_recon is called with default arguments.  Defaults to None.
            prox_input (jax array, optional): Reconstruction to be used as input to a proximal map.
            compute_prior_loss (bool, optional):  Set true to calculate and return the prior model loss.
            first_iteration (int, optional): Set this to be the number of iterations previously completed when restarting a recon using init_recon.
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
            weights = jax.device_put(weights, self.sinogram_device)
            constant_weights = False

        recon_shape = self.get_params('recon_shape')
        num_recon_slices = recon_shape[2]

        if init_recon is None:
            # Initialize VCD recon, and error sinogram
            print('Starting direct recon for initial reconstruction')
            with jax.default_device(self.sinogram_device):
                init_recon = self.direct_recon(sinogram)  # init_recon is output to self.main device because of the default output device in self.back_project
        elif isinstance(init_recon, int) and init_recon == 0:
            init_recon = jnp.zeros(recon_shape, device=self.main_device)

        # Make sure that init_recon has the correct shape and type
        if init_recon.shape != recon_shape:
            error_message = "init_recon does not have the correct shape. \n"
            error_message += "Expected {}, but got shape {} for init_recon shape.".format(recon_shape,
                                                                                          init_recon.shape)
            raise ValueError(error_message)

        # Initialize VCD recon and error sinogram using the init_recon
        # We find the optimal alpha to minimize (1/2)||y - alpha Ax||_weights^2, where y is the sinogram and x is init_recon
        print('Initializing error sinogram')
        error_sinogram = self.forward_project(init_recon)
        if not constant_weights:
            weighted_error_sinogram = weights * error_sinogram  # Note that fm_constant will be included below
        else:
            weighted_error_sinogram = error_sinogram
        wtd_err_sino_norm = jnp.sum(weighted_error_sinogram * error_sinogram)
        if wtd_err_sino_norm > 0:
            alpha = jnp.sum(weighted_error_sinogram * sinogram) / wtd_err_sino_norm
        else:
            alpha = 1

        error_sinogram = sinogram - alpha * error_sinogram
        init_recon = alpha * init_recon

        recon = init_recon
        recon = jax.device_put(recon, self.main_device)  # Even if recon was created with main_device as the default, it wasn't committed there.
        error_sinogram = jax.device_put(error_sinogram, self.sinogram_device)

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
            weights = jax.device_put(weights, self.sinogram_device)

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
        max_iters = partition_sequence.size
        fm_rmse = np.zeros(max_iters)
        pm_loss = np.zeros(max_iters)
        nmae_update = np.zeros(max_iters)
        alpha_values = np.zeros(max_iters)
        num_iters = 0
        for i in range(max_iters):
            # Get the current partition (set of subsets) and shuffle the subsets
            partition = partitions[partition_sequence[i]]

            # Do an iteration
            flat_recon, error_sinogram, ell1_for_partition, alpha = self.vcd_partition_iterator(vcd_subset_updater,
                                                                                                 flat_recon,
                                                                                                 error_sinogram,
                                                                                                 partition)

            # Compute the stats and display as desired
            fm_rmse[i] = self.get_forward_model_loss(error_sinogram, sigma_y, weights)
            nmae_update[i] = ell1_for_partition / jnp.sum(jnp.abs(flat_recon))
            es_rmse = jnp.linalg.norm(error_sinogram) / jnp.sqrt(float(error_sinogram.size))
            alpha_values[i] = alpha

            if verbose >= 1:
                iter_output = '\nAfter iteration {} of a max of {}: Pct change={:.4f}, Forward loss={:.4f}'.format(i + first_iteration, max_iters + first_iteration,
                                                                                                  100 * nmae_update[i],
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
                print('Number subsets = {}'.format(partition.shape[0]))
                if verbose >= 2:
                    mbirjax.get_memory_stats()
                    print('--------')
            num_iters += 1
            if nmae_update[i] < stop_threshold_change_pct / 100:
                print('Change threshold stopping condition reached')
                break

        return self.reshape_recon(flat_recon), (fm_rmse[0:num_iters], pm_loss[0:num_iters], nmae_update[0:num_iters],
                                                alpha_values[0:num_iters])

    def vcd_partition_iterator(self, vcd_subset_updater, flat_recon, error_sinogram, partition):
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
            (flat_recon, error_sinogram, ell1_for_partition, alpha): The first two have the same shape as above, but
            are updated to reduce overall loss function.
            The ell1_for_partition includes the changes from all subsets of this partition.
            alpha is the relative step size in the gradient descent step, averaged over the subsets
            in the partition.
        """

        # Loop over the subsets of the partition, using random subset_indices to order them.
        ell1_for_partition = 0
        alpha_sum = 0
        subset_indices = np.random.permutation(partition.shape[0])

        times = np.zeros(13)
        # np.set_printoptions(precision=1, floatmode='fixed', suppress=True)
        partition_worker = jax.device_put(partition, self.worker)
        for index in subset_indices:
            subset = partition[index]
            subset_worker = partition_worker[index]
            flat_recon, error_sinogram, ell1_for_subset, alpha_for_subset, times, time_names = vcd_subset_updater(flat_recon,
                                                                                                       error_sinogram,
                                                                                                       subset, subset_worker, times)
            ell1_for_partition += ell1_for_subset
            alpha_sum += alpha_for_subset
        # # Debug code to go with timing info in vcd_subset_updater
        # max_len = max(len(s) for s in time_names)
        # formatted_names = [f"{s:<{max_len}}," for s in time_names]
        # formatted_names = " ".join(formatted_names)
        #
        # pct_times = 100 * times / np.sum(times)
        # formatted_times = ['{:.2f}'.format(pct_times[j]) for j in range(len(pct_times))]
        # formatted_times = [f"{s:<{max_len}}," for s in formatted_times]
        # formatted_times = " ".join(formatted_times)
        # print('Pct time = ')
        # print(formatted_names)
        # print(formatted_times)
        # # End debug code

        return flat_recon, error_sinogram, ell1_for_partition, alpha_sum / partition.shape[0]

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

        def vcd_subset_updater(flat_recon, error_sinogram, pixel_indices, pixel_indices_worker, times):
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
                pixel_indices_worker (jax array): Same as pixel_indices, but copied onto the worker device.
                times (ndarray): 1D array of elapsed times for debugging/performance tuning.

            Returns:
                flat_recon, error_sinogram, ell1_for_subset, alpha_for_subset:
                The first two have the same shape as above, but are updated to reduce the overall loss function.
                ell1_for_subset is for the change to the recon from this one subset.
                alpha is the relative step size for this subset.
            """

            # Compute the forward model gradient and hessian at each pixel in the index set.
            # Assumes Loss(delta) = 1/(2 sigma_y^2) || error_sinogram - A delta ||_weights^2
            # All the time assignments and block_until_ready() are for debugging/performance tracking only.
            # The cryptic labels in the comments match the printed timing labels when these are activated.
            # time_index = 0
            time_names = []

            # Compute the prior model gradient and hessian (i.e., second derivative) terms
            # time_names.append('qggmrf')
            # time_start = time.time()
            if prox_input is None:

                # qGGMRF prior - compute the qggmrf gradient and hessian at each pixel in the index set.
                with jax.default_device(self.main_device):
                    if self.worker != self.main_device and len(pixel_indices) < self.transfer_pixel_batch_size:
                        prior_grad, prior_hess = (
                            mbirjax.qggmrf_gradient_and_hessian_at_indices_transfer(flat_recon, recon_shape, pixel_indices,
                                                                           qggmrf_params, self.main_device, self.worker))
                    else:
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

            # time_names.append('wterrsin')
            # time_start = time.time()
            if not const_weights:
                weighted_error_sinogram = weights * error_sinogram  # Note that fm_constant will be included below
            else:
                weighted_error_sinogram = error_sinogram
            # weighted_error_sinogram = weighted_error_sinogram.block_until_ready()
            # times[time_index] += time.time() - time_start
            # time_index += 1

            # Transfer to worker for later use
            # time_names.append('bproj')
            # time_start = time.time()

            # Back project to get the gradient
            forward_grad = - fm_constant * sparse_back_project(weighted_error_sinogram, pixel_indices_worker,
                                                               output_device=self.main_device)
            # forward_grad = forward_grad.block_until_ready()
            # times[time_index] += time.time() - time_start
            # time_index += 1

            # Get the forward hessian for this subset
            # time_names.append('forhess')
            # time_start = time.time()
            forward_hess = fm_constant * fm_hessian[pixel_indices]
            # forward_hess = forward_hess.block_until_ready()
            # times[time_index] += time.time() - time_start
            # time_index += 1

            # Compute update vector update direction in recon domain
            # time_names.append('deltrec')
            # time_start = time.time()
            delta_recon_at_indices = - ((forward_grad + prior_grad) / (forward_hess + prior_hess))
            # delta_recon_at_indices = delta_recon_at_indices.block_until_ready()
            # times[time_index] += time.time() - time_start
            # time_index += 1

            # Compute delta^T \nabla Q(x_hat; x'=x_hat) for use in finding alpha
            # time_start = time.time()
            # time_names.append('priorlin')
            prior_linear = jnp.sum(prior_grad * delta_recon_at_indices)
            # prior_linear = prior_linear.block_until_ready()
            # times[time_index] += time.time() - time_start
            # time_index += 1

            # Estimated upper bound for hessian
            # time_names.append('pquad')
            # time_start = time.time()
            prior_overrelaxation_factor = 2
            prior_quadratic_approx = ((1 / prior_overrelaxation_factor) *
                                      jnp.sum(prior_hess * delta_recon_at_indices ** 2))
            # prior_quadratic_approx = prior_quadratic_approx.block_until_ready()
            # times[time_index] += time.time() - time_start
            # time_index += 1

            # Compute update direction in sinogram domain
            # time_names.append('fproj')
            # time_start = time.time()
            delta_sinogram = sparse_forward_project(delta_recon_at_indices, pixel_indices_worker,
                                                    output_device=self.sinogram_device)
            # delta_sinogram = delta_sinogram.block_until_ready()
            # times[time_index] += time.time() - time_start
            # time_index += 1

            # time_names.append('forlqu')
            # time_start = time.time()
            forward_linear, forward_quadratic = self.get_forward_lin_quad(weighted_error_sinogram, delta_sinogram,
                                                                          weights, fm_constant, const_weights,
                                                                          output_device=self.main_device)

            # Compute optimal update step
            alpha_numerator = forward_linear - prior_linear
            alpha_denominator = forward_quadratic + prior_quadratic_approx + jnp.finfo(jnp.float32).eps
            alpha = alpha_numerator / alpha_denominator
            max_alpha = 1.5
            alpha = jnp.clip(alpha, jnp.finfo(jnp.float32).eps, max_alpha)
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
                delta_sinogram = sparse_forward_project(delta_recon_at_indices, pixel_indices, output_device=self.sinogram_device)

            # time_names.append('scaledr')
            # time_start = time.time()
            # Perform sparse updates at index locations
            delta_recon_at_indices = jax.device_put(delta_recon_at_indices, self.main_device)
            delta_recon_at_indices = alpha * delta_recon_at_indices
            # delta_recon_at_indices = delta_recon_at_indices.block_until_ready()
            # times[time_index] += time.time() - time_start
            # time_index += 1

            # time_names.append('flatrec')
            # time_start = time.time()
            flat_recon = update_recon(flat_recon, pixel_indices, delta_recon_at_indices)
            # flat_recon = flat_recon.block_until_ready()
            # times[time_index] += time.time() - time_start
            # time_index += 1

            # Update sinogram and loss
            # time_names.append('deltsin')
            # time_start = time.time()
            delta_sinogram = float(alpha) * delta_sinogram
            error_sinogram = error_sinogram - delta_sinogram
            # error_sinogram = error_sinogram.block_until_ready()
            # times[time_index] += time.time() - time_start
            # time_index += 1

            # time_names.append('stats')
            # time_start = time.time()
            ell1_for_subset = jnp.sum(jnp.abs(delta_recon_at_indices))
            alpha_for_subset = alpha
            # norm_squared_for_subset = norm_squared_for_subset.block_until_ready()
            # times[time_index] += time.time() - time_start
            # time_index += 1

            return flat_recon, error_sinogram, ell1_for_subset, alpha_for_subset, times, time_names

        return vcd_subset_updater

    def get_forward_lin_quad(self, weighted_error_sinogram, delta_sinogram, weights, fm_constant, const_weights,
                             output_device=None):
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
            output_device (jax device): device on which the output will be placed

        Returns:
            tuple:
            forward_linear, forward_quadratic
        """
        forward_linear = fm_constant * jnp.sum(weighted_error_sinogram * delta_sinogram)
        forward_quadratic = fm_constant * jnp.sum(delta_sinogram * delta_sinogram * weights)

        # The code below does batching of the sinogram on the GPU, but in a comparison on a sinogram
        # of size 1800x512x512, it was noticeably faster to do the computation on the CPU.

        # If this can be done without data transfer, then do it.
        # if True:  # self.worker == self.sinogram_device:
        #     forward_linear = fm_constant * jnp.sum(weighted_error_sinogram * delta_sinogram)
        #     forward_quadratic = fm_constant * jnp.sum(delta_sinogram * delta_sinogram * weights)
        #
        # Otherwise batch the sinogram by view, send to the worker and calculate linear and quadratic terms
        # First apply the batch projector directly to an initial batch to get the initial output
        # else:
        #     num_views = weighted_error_sinogram.shape[0]
        #     views_per_batch = self.view_batch_size_for_vmap
        #     views_per_batch = num_views if views_per_batch is None else views_per_batch
        #     num_remaining = num_views % views_per_batch
        #
        #     # If the input is a multiple of batch_size, then we'll do a full batch, otherwise just the excess.
        #     initial_batch_size = views_per_batch if num_remaining == 0 else num_remaining
        #     # Make the weights into a 1D vector along views if it's a constant
        #
        #     def linear_quadratic(start_ind, stop_ind, previous_linear=0, previous_quadratic=0):
        #         """
        #         Send a batch to the worker and compute forward linear and quadratic for that batch.
        #         """
        #         worker_wes = jax.device_put(weighted_error_sinogram[start_ind:stop_ind], self.worker)
        #         worker_ds = jax.device_put(delta_sinogram[start_ind:stop_ind], self.worker)
        #         if not const_weights:
        #             worker_wts = jax.device_put(weights[start_ind:stop_ind], self.worker)
        #         else:
        #             worker_wts = weights
        #
        #         # previous_linear += fm_constant * jnp.sum(worker_wes * worker_ds)
        #         # previous_quadratic += fm_constant * jnp.sum(worker_ds * worker_ds * worker_wts)
        #
        #         previous_linear += sum_product(worker_wes, worker_ds)
        #         worker_ds = jax.vmap(jnp.multiply)(worker_ds, worker_ds)
        #         if not const_weights:
        #             quadratic_entries = sum_product(worker_ds, worker_wts)
        #         else:
        #             quadratic_entries = jnp.sum(worker_ds)
        #         previous_quadratic += quadratic_entries
        #
        #         del worker_wes, worker_ds, worker_wts
        #         return previous_linear, previous_quadratic
        #
        #     forward_linear, forward_quadratic = linear_quadratic(0, initial_batch_size)
        #
        #     # Then deal with the batches if there are any
        #     if views_per_batch < num_views:
        #         num_batches = (num_views - initial_batch_size) // views_per_batch
        #         for j in jnp.arange(num_batches):
        #             start_ind_j = initial_batch_size + j * views_per_batch
        #             stop_ind_j = start_ind_j + views_per_batch
        #             forward_linear, forward_quadratic = linear_quadratic(start_ind_j, stop_ind_j,
        #                                                                  forward_linear, forward_quadratic)
        #
        #     forward_linear = fm_constant * forward_linear
        #     forward_quadratic = fm_constant * forward_quadratic

        forward_linear = jax.device_put(forward_linear, output_device)
        forward_quadratic = jax.device_put(forward_quadratic, output_device)
        return forward_linear, forward_quadratic

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

    def prox_map(self, prox_input, sinogram, weights=None, init_recon=None, stop_threshold_change_pct=0.2, max_iterations=3, first_iteration=0):
        """
        Proximal Map function for use in Plug-and-Play applications.
        This function is similar to recon, but it essentially uses a prior with a mean of prox_input and a standard deviation of sigma_prox.

        Args:
            prox_input (jax array): proximal map input with same shape as reconstruction.
            sinogram (jax array): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels).
            weights (jax array, optional): 3D positive weights with same shape as sinogram.  Defaults to None, in which case the weights are implicitly all 1s.
            init_recon (jax array, optional): optional reconstruction to be used for initialization.  Defaults to None, in which case the initial recon is determined by vcd_recon.
            stop_threshold_change_pct (float, optional): Stop reconstruction when NMAE percent change from one iteration to the next is below stop_threshold_change_pct.  Defaults to 0.2.
            max_iterations (int, optional): maximum number of iterations of the VCD algorithm to perform.
            first_iteration (int, optional): Set this to be the number of iterations previously completed when restarting a recon using init_recon.  This defines the first index in the partition sequence.  Defaults to 0.

        Returns:
            [recon, fm_rmse]: reconstruction and array of loss for each iteration.
        """
        # TODO:  Refactor to operate on a subset of pixels and to use previous state
        # Generate set of voxel partitions
        recon_shape, granularity = self.get_params(['recon_shape', 'granularity'])
        partitions = mbirjax.gen_set_of_pixel_partitions(recon_shape, granularity)

        # Generate sequence of partitions to use
        partition_sequence = self.get_params('partition_sequence')
        partition_sequence = mbirjax.gen_partition_sequence(partition_sequence, max_iterations=max_iterations)
        partition_sequence = partition_sequence[first_iteration:]

        # Compute reconstruction
        recon, loss_vectors = self.vcd_recon(sinogram, partitions, partition_sequence, stop_threshold_change_pct,
                                             weights=weights, init_recon=init_recon, prox_input=prox_input,
                                             first_iteration=first_iteration)

        return recon, loss_vectors

    def gen_weights_mar(self, sinogram, init_recon=None, metal_threshold=None, beta=1.0, gamma=3.0):
        """
        Generates the weights used for reducing metal artifacts in MBIR reconstruction.

        This function computes sinogram weights that help to reduce metal artifacts.
        More specifically, it computes weights with the form:

            weights = exp( -(sinogram/beta) * ( 1 + gamma * delta(metal) ) )

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

    def gen_weights(self, sinogram, weight_type):
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
        weight_list = []
        num_views = sinogram.shape[0]
        batch_size = self.view_batch_size_for_vmap
        for i in range(0, num_views, batch_size):
            sino_batch = jax.device_put(sinogram[i:min(i + batch_size, num_views)], self.worker)

            if weight_type == 'unweighted':
                weights = jnp.ones(sino_batch.shape)
            elif weight_type == 'transmission':
                weights = jnp.exp(-sino_batch)
            elif weight_type == 'transmission_root':
                weights = jnp.exp(-sino_batch / 2)
            elif weight_type == 'emission':
                weights = 1.0 / (jnp.absolute(sino_batch) + 0.1)
            else:
                raise Exception("gen_weights: undefined weight_type {}".format(weight_type))
            weight_list.append(jax.device_put(weights, self.sinogram_device))

        weights = jnp.concatenate(weight_list, axis=0)
        return weights

    def gen_modified_3d_sl_phantom(self):
        """
        Generates a simplified, low-dynamic range version of the 3D Shepp-Logan phantom.

        Returns:
            ndarray: A 3D numpy array of shape specified by TomographyModel class parameters.
        """
        recon_shape = self.get_params('recon_shape')
        phantom = mbirjax.generate_3d_shepp_logan_low_dynamic_range(recon_shape, device=self.main_device)
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

    def transpose(input_array):
        return transpose_list(input_array)[0]

    return transpose
