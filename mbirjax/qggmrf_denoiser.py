import jax
import jax.numpy as jnp
from collections import namedtuple
from mbirjax import TomographyModel


class QGGMRFDenoiser(TomographyModel):
    """
    The QGGMRFDenoiser uses the MBIRJAX recon framework to implement a qggmrf proximal map denoiser.
    The primary interface is through :meth:`denoise`.
    """

    def __init__(self, sinogram_shape):

        view_params_name = ''
        super().__init__(sinogram_shape, view_params_name=view_params_name)
        self.use_ror_mask = False

    def denoise(self, image, use_ror_mask=False, init_image=None, max_iterations=15, stop_threshold_change_pct=0.2,
                 first_iteration=0, compute_prior_loss=False, logfile_path='./logs/recon.log', print_logs=True):
        """
        Use the VCD algorithm with the QGGMRF loss to denoise a 3D image (volume).

        The noise level is estimated from the image, and the denoising strength can be adjusted using parameters
        sharpness (default=1.0) and/or snr_db (default=30).

        Args:
            image (numpy or jax array):  The 3D volume to be denoised.
            use_ror_mask (bool, optional): Set true to restrict denoising to an inscribed circle in the image.  Defaults to False.
            init_image (numpy or jax array, optional):  An initial image for the minimization.  Defaults to image.
            max_iterations (int, optional): maximum number of iterations of the VCD algorithm to perform.
            stop_threshold_change_pct (float, optional): Stop reconstruction when 100 * ||delta_recon||_1 / ||recon||_1 change from one iteration to the next is below stop_threshold_change_pct.  Defaults to 0.2.  Set this to 0 to guarantee exactly max_iterations.
            first_iteration (int, optional): Set this to be the number of iterations previously completed when restarting a recon using init_recon.  This defines the first index in the partition sequence.  Defaults to 0.
            compute_prior_loss (bool, optional):  Set true to calculate and return the prior model loss.  This will lead to slower reconstructions and is meant only for small recons.
            logfile_path (str, optional): Path to the output log file.  Defaults to './logs/recon.log'.
            print_logs (bool, optional): If true then print logs to console.  Defaults to True.

        Returns:
            tuple: (denoised_image, denoiser_dict)
                - denoised_image (jax array): A denoised image of the same shape as image
                - denoiser_dict (dict): A dict obtained from :meth:`get_recon_dict` with entries
                    * 'recon_params'
                    * 'notes'
                    * 'recon_logs'
                    * 'model_params'

        Example:
            >>> denoiser = mj.QGGMRFDenoiser(noisy_image.shape)
            >>> denoiser.set_params(sharpness=1.1, snr_db=33)
            >>> denoised_image, denoised_dict = denoiser.denoise(noisy_image)
            >>> mj.slice_viewer(noisy_image, denoised_image, data_dicts=[None, denoised_dict], title='Noisy and denoised images')

        See Also
        --------
        TomographyModel : The base class from which this class inherits.
        """
        self.use_ror_mask = use_ror_mask
        return self.recon(image, init_recon=init_image, max_iterations=max_iterations,
                          stop_threshold_change_pct=stop_threshold_change_pct, first_iteration=first_iteration,
                          compute_prior_loss=compute_prior_loss, logfile_path=logfile_path, print_logs=print_logs)

    def get_magnification(self):
        """
        Return 1 to satisfy the TomographyModel interface.

        Returns:
            (float): magnification
        """
        magnification = 1.0
        return magnification

    def verify_valid_params(self):
        """
        Check that all parameters are compatible for a reconstruction.

        Note:
            Raises ValueError for invalid parameters.
        """
        super().verify_valid_params()
        sinogram_shape = self.get_params('sinogram_shape')

        recon_shape = self.get_params('recon_shape')
        if recon_shape != sinogram_shape:
            error_message = "recon_shape and sinogram_shape must be the same. \n"
            error_message += "Got {} for recon_shape and {} for sinogram_shape".format(recon_shape, sinogram_shape)
            raise ValueError(error_message)

    def get_geometry_parameters(self):
        """
        Return a minimal set of parameters to satisfy the TomographyModel interface.

        Returns:
            namedtuple of required geometry parameters.
        """
        # First get the parameters managed by ParameterHandler
        geometry_param_names = ['delta_det_channel', 'det_channel_offset', 'delta_voxel']
        geometry_param_values = self.get_params(geometry_param_names)

        # Then create a namedtuple to access parameters by name in a way that can be jit-compiled.
        GeometryParams = namedtuple('GeometryParams', geometry_param_names)
        geometry_params = GeometryParams(*tuple(geometry_param_values))

        return geometry_params

    def create_projectors(self):
        """
        This method does nothing for QGGMRFDenoiser.
        """
        self.projector_functions = None

    def auto_set_recon_size(self, sinogram_shape, no_compile=True, no_warning=False):
        """Compute the default recon shape to equal the sinogram shape"""

        self.set_params(no_compile=no_compile, no_warning=no_warning, recon_shape=sinogram_shape)

    def forward_project(self, recon):
        """
        Perform a full forward projection at all voxels in the field-of-view, which in this case is the identity.

        Args:
            recon (jnp array): The 3D reconstruction array.

        Returns:
            jnp array: The resulting 3D sinogram after projection.
        """
        sinogram = recon  # Since jax arrays are immutable, we shouldn't need to do recon.copy()
        return sinogram

    def back_project(self, sinogram):
        """
        Perform a full back projection at all voxels in the field-of-view, which in this case is the identity.

        Args:
            sinogram (jnp array): 3D jax array containing sinogram.

        Returns:
            jnp array: The resulting 3D sinogram after projection.
        """
        recon = sinogram  # Since jax arrays are immutable, we shouldn't need to do sinogram.copy()
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
        recon_shape = self.get_params('recon_shape')
        row_index, col_index = jnp.unravel_index(pixel_indices, recon_shape[:2])
        sinogram = jnp.zeros(recon_shape, device=output_device)
        sinogram = sinogram.at[row_index, col_index].set(voxel_values)
        return sinogram

    def sparse_back_project(self, sinogram, pixel_indices, view_indices=None, coeff_power=1, output_device=None):
        """
        Back project the given sinogram to the voxels given by the indices.  The sinogram should be the full sinogram
        associated with all of the angles used to define the ct model, even if a set of view_indices is provided.
        The indices are into a flattened 2D array of shape (recon_rows, recon_cols), and the projection is done using
        all voxels with those indices across all the slices.  If view_indices is a jax array of ints, then they should
        be the indices into the sinogram that is passed in here.

        Args:
            sinogram (jnp array): 3D jax array containing the full sinogram.
            pixel_indices (jnp array): Array of indices specifying which voxels to back project.
            view_indices (jax array): Array of indices of views to project.  These are indices into the first axis of sinogram.
            coeff_power (int, optional): Normally 1, but set to 2 for Hessian diagonal
            output_device (jax device, optional): Device on which to put the output

        Returns:
            A jax array of shape (len(indices), num_slices)
        """
        row_index, col_index = jnp.unravel_index(pixel_indices, sinogram.shape[:2])
        recon = jax.device_put(sinogram[row_index, col_index], output_device)  # Since jax arrays are immutable, we shouldn't need to do sinogram.copy()
        return recon

    def direct_recon(self, sinogram, filter_name="ramp", view_batch_size=1):
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
        recon = sinogram  # Since jax arrays are immutable, we shouldn't need to do sinogram.copy()
        return recon

    def direct_filter(self, sinogram, filter_name="ramp", view_batch_size=1):
        sinogram = sinogram  # Since jax arrays are immutable, we shouldn't need to do sinogram.copy()
        return sinogram
