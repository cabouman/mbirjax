import jax.numpy as jnp
from mbirjax import TomographyModel


class TemplateModel(TomographyModel):
    """
    This is a template for a class designed to handle a particular geometry that extends the :ref:`TomographyModelDocs`.
    This class inherits all methods and properties from the :ref:`TomographyModelDocs` and may override some
    to suit the specific geometry.

    Args:
        sinogram_shape (tuple):
            Shape of the sinogram as a tuple in the form `(views, rows, channels)`, where 'views' is the number of
            different projection angles, 'rows' correspond to the number of detector rows, and 'channels' index columns of
            the detector that are assumed to be aligned with the rotation axis.
        param1, param2 (scalars):
            These are view-independent scalar parameters that are required for the geometry and are not already included in the parent class.
        view_dependent_vec1, view_dependent_vec2 (jnp.ndarray):
            These are view-dependent parameter vectors each with length = number of views.
        **kwargs (dict):
            Additional keyword arguments that are passed to the :ref:`TomographyModelDocs` constructor. These can
            include settings and configurations specific to the tomography model such as noise models or image dimensions.
            Refer to :ref:`TomographyModelDocs` documentation for a detailed list of possible parameters.
    """

    def __init__(self, sinogram_shape, param1, param2, view_dependent_vec1, view_dependent_vec2, **kwargs):
        # Convert the view-dependent vectors to an array
        view_dependent_vecs = [vec.flatten() for vec in [view_dependent_vec1, view_dependent_vec2]]
        try:
            view_params_array = jnp.stack(view_dependent_vecs, axis=1)
        except ValueError as e:
            raise ValueError("Incompatible view dependent vector lengths:  all view-dependent vectors must have the "
                             "same length.")

        super().__init__(sinogram_shape, param1=param1, param2=param2, view_params_array=view_params_array, **kwargs)

    def verify_valid_params(self):
        """
        Check that all parameters are compatible for a reconstruction.
        Extend to include any geometry-specific conditions.
        """
        super().verify_valid_params()
        sinogram_shape, view_params_array = self.get_params(['sinogram_shape', 'view_params_array'])

        if view_params_array.shape[0] != sinogram_shape[0]:
            error_message = "Number view dependent parameter vectors must equal the number of views. \n"
            error_message += "Got {} for length of view-dependent parameters and "
            error_message += "{} for number of views.".format(view_params_array.shape[0], sinogram_shape[0])
            raise ValueError(error_message)

    def get_geometry_parameters(self):
        """
        Required function to get a list of the view independent geometry parameters required for projection.

        Returns:
            List of any parameters required for back_project_one_view_to_pixel or forward_project_pixels_to_one_view.
            This does not need to include view dependent parameters, or sinogram_shape or recon_shape, which
            are passed in automatically to projector_params.
        """
        geometry_params = self.get_params(['delta_det_channel', 'det_channel_offset', 'delta_pixel_recon'])

        return geometry_params

    @staticmethod
    def back_project_one_view_to_pixel(sinogram_view, pixel_index, single_view_params, projector_params, coeff_power=1):
        """
        Calculate the backprojection value at a specified recon voxel cylinder given a sinogram view and various parameters.
        This code uses the distance driven projector.

        NOTE: This function must be able to be jit-compiled.

        Args:
            sinogram_view (2D jax array): one view of the sinogram to be back projected
            pixel_index (int):  index into flattened array of size num_rows x num_cols.
            single_view_params: These are the view dependent parameters for the view being back projected.
            projector_params (1D jax array): tuple of (sinogram_shape, recon_shape, get_geometry_params()).
            coeff_power (int): backproject using the coefficients of (A_ij ** coeff_power).
                Normally 1, but should be 2 when computing theta 2.

        Returns:
            The voxel values for all slices at the input index (i.e., a voxel cylinder) obtained by backprojecting
            the input sinogram view.
        """
        # The number of slices will need to come from geometry_params
        num_slices = 1

        # Computes the voxel values in all slices corresponding to pixel_index
        voxel_values_cylinder = jnp.zeros(num_slices)
        return voxel_values_cylinder

    @staticmethod
    def forward_project_pixels_to_one_view(voxel_values, pixel_indices, single_view_params, projector_params):
        """
        Forward project a set of voxels determined by indices into a single view.

        NOTE: This function must be able to be jit-compiled.

        Args:
            voxel_values (jax array):  2D array of shape (num_indices, num_slices) of voxel values, where
                voxel_values[i, j] is the value of the voxel in slice j at the location determined by indices[i].
            pixel_indices (jax array of int):  1D vector of indices into flattened array of size num_rows x num_cols.
            single_view_params: These are the view dependent parameters for this view.
            projector_params (1D jax array): tuple of (sinogram_shape, recon_shape, get_geometry_params()).

        Returns:
            jax array of shape (num_det_rows, num_det_channels)
        """

        # Returns a single view of the sinogram
        sinogram_shape = projector_params[0]
        sinogram_view = jnp.zeros(sinogram_shape[1:])
        return sinogram_view
