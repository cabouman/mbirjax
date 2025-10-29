import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np

import mbirjax as mj

class DummyClass:

    def __init__(self, sinogram_shape, recon_shape, len_gpus, **kwargs):

        self.main_device, self.sinogram_device, self.worker = None, None, None
        self.replicated_device = None
        self.cpus = jax.devices('cpu')
        self.projector_functions = None
        self.prox_data = None

        self.sinogram_shape = sinogram_shape
        self.recon_shape = recon_shape
        self.len_gpus = len_gpus

# The following may be adjusted based on memory in set_devices_and_batch_sizes()
        self.view_batch_size_for_vmap = 512
        self.pixel_batch_size_for_vmap = 2048
        self.transfer_pixel_batch_size = 100 * self.pixel_batch_size_for_vmap
        self.gpu_memory = 0
        self.cpu_memory = 0
        self.mem_required_for_gpu = 0
        self.mem_required_for_cpu = 0
        self.use_gpu = 'none'  # This is set in set_devices_and_batch_sizes based on memory and get_params('use_gpu')

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
        use_gpu = 'automatic'
        try:
            gpus = jax.devices('gpu')
            gpu_memory_stats = gpus[0].memory_stats()
            gpu_memory = float(gpu_memory_stats['bytes_limit']) - float(gpu_memory_stats['bytes_in_use'])
            gpu_memory /= gb
        except RuntimeError:
            gpus = []
            gpu_memory = 0
        self.gpu_memory = gpu_memory

        # Estimate the CPU memory available
        cpu_memory = 0
        if cpu_memory == 0:
            cpu_memory_stats = mj.get_memory_stats(print_results=False)[-1]
            cpu_memory = float(cpu_memory_stats['bytes_limit']) - float(cpu_memory_stats['bytes_in_use'])
            cpu_memory /= gb
        self.cpu_memory = cpu_memory

        # Get basic parameters
        sinogram_shape = self.sinogram_shape
        num_views, num_det_rows, num_det_channels = sinogram_shape
        recon_shape = self.recon_shape
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
        recon_reps_for_vcd = 5
        mem_for_all_vcd = recon_reps_for_vcd * mem_per_recon + mem_for_all_sinos_on_gpu - mem_per_voxel_batch

        frac_gpu_mem_to_use = 0.9
        gpu_memory_to_use = frac_gpu_mem_to_use * gpu_memory

        # 'automatic' and more than one GPU: Everything will be done with sharding
        if True or use_gpu == 'automatic' and len(gpus) > 1:

            num_gpus = self.len_gpus
            excess_views = num_views % num_gpus
            if excess_views != 0:
                raise ValueError(f"Sharding has been invoked because use_gpu='automatic' and multiple GPUs are detected."
                                    "The number of views must be an exact multiple of the number of GPUs."
                                    f"Currently there are {num_views} views and {num_gpus} detected GPUs, so there are {excess_views} excess views."
                                    "To disable sharding, use ct_model.set_params(use_gpu='sinograms').")

            self.use_gpu = 'sharding'

            # create devices and named shardings
            devices = np.array(gpus).reshape((-1, 1))
            mesh = Mesh(devices, ('views', 'rows'))

            self.main_device = cpus[0]
            self.sinogram_device = NamedSharding(mesh, P('views'))
            self.replicated_device = NamedSharding(mesh, P())
            self.worker = gpus[0]

            # sharding requires a single view batch
            self.view_batch_size_for_vmap = num_views

            # Recalculate the memory per projection with the new batch size
            mem_per_projection = cone_beam_projection_factor * self.view_batch_size_for_vmap * mem_per_view_with_floor
            mem_per_projection_total = mem_per_projection

            mem_sino_per_gpu = (mem_for_vcd_sinos_gpu + mem_per_projection_total) / num_gpus
            mem_budget_for_voxel = gpu_memory_to_use - mem_sino_per_gpu

            # the results of the math were different from what has empirically been seen
            mem_per_cylinder_conservative_factor = 600
            conservative_mem_per_cylinder = mem_per_cylinder * mem_per_cylinder_conservative_factor

            if mem_budget_for_voxel < conservative_mem_per_cylinder:
                raise ValueError('Insufficient GPU memory per shard to fit a voxel batch; reduce reconstruction size or GPU usage.')

            pixel_batch_size = int(np.floor(mem_budget_for_voxel / conservative_mem_per_cylinder))

            self.pixel_batch_size_for_vmap = pixel_batch_size
            self.transfer_pixel_batch_size = self.pixel_batch_size_for_vmap

            # Recalculate the memory per voxel batch with the new batch size
            mem_per_voxel_batch = mem_per_cylinder * self.transfer_pixel_batch_size

            mem_required_for_gpu = mem_sino_per_gpu + mem_per_voxel_batch
            mem_required_for_cpu = recon_reps_for_vcd * mem_per_recon + 2 * mem_per_sinogram  # All recons plus sino and weights

            self.mem_required_for_gpu = mem_required_for_gpu
            self.mem_required_for_cpu = mem_required_for_cpu


if __name__ == "__main__":
    dummy = DummyClass((1800, 1800, 1800), (1800, 1800, 1800), 8)
    dummy.set_devices_and_batch_sizes()

    print(dummy.mem_required_for_cpu)
    print(dummy.mem_required_for_gpu)
    print(dummy.pixel_batch_size_for_vmap)