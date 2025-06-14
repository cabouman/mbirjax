import jax
import jax.numpy as jnp
from collections import namedtuple
from mbirjax import TomographyModel


class QGGMRFDenoiser(TomographyModel):
    """
    The QGGMRFDenoiser is meant for internal use to implement a qggmrf proximal map using the
    MBIRJAX recon framework.
    """

    def __init__(self, sinogram_shape):

        view_params_name = ''
        super().__init__(sinogram_shape, view_params_name=view_params_name)

    def denoise(self, volume, sigma=None, init_image=None, max_iterations=15, stop_threshold_change_pct=0.2,
                 first_iteration=0, compute_prior_loss=False, logfile_path='./logs/recon.log', print_logs=True):
        if sigma is not None:
            self.set_params(sigma_x=sigma)

        return self.recon(volume, init_recon=init_image, max_iterations=max_iterations,
                          stop_threshold_change_pct=stop_threshold_change_pct, first_iteration=first_iteration,
                          compute_prior_loss=compute_prior_loss, logfile_path=logfile_path, print_logs=print_logs)

    def get_magnification(self):
        """
        Compute the scale factor from a voxel at iso (at the origin on the center of rotation) to
        its projection on the detector.  For parallel beam, this is 1, but it may be parameter-dependent
        for other geometries.

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
        Function to get a list of the primary geometry parameters for identity projection.

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
        Perform a full forward projection at all voxels in the field-of-view.

        Note:
            This method should generally not be used directly for iterative reconstruction.  For iterative
            reconstruction, use :meth:`recon`.

        Args:
            recon (jnp array): The 3D reconstruction array.

        Returns:
            jnp array: The resulting 3D sinogram after projection.
        """
        sinogram = recon  # Since jax arrays are immutable, we shouldn't need to do recon.copy()
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
