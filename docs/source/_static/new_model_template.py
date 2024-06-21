import jax
import jax.numpy as jnp
from functools import partial
from collections import namedtuple
from mbirjax import TomographyModel, ParameterHandler


class TemplateModel(TomographyModel):
    """
    This is a template for a class designed to handle a particular geometry that extends the :ref:`TomographyModelDocs`.
    This class inherits all methods and properties from the :ref:`TomographyModelDocs` and may override some
    to suit the specific geometry.

    Parameters not included in the constructor can be set using the set_params method of :ref:`TomographyModelDocs`.
    Refer to :ref:`TomographyModelDocs` documentation for a detailed list of possible parameters.

    Items to change for a particular geometry are highlighted with TODO.
    It is not recommended to use **kwargs in the constructor since doing so complicates checking for valid parameter
    names in set_params.

    Args:
        sinogram_shape (tuple):
            Shape of the sinogram as a tuple in the form `(views, rows, channels)`, where 'views' is the number of
            different projection angles, 'rows' correspond to the number of detector rows, and 'channels' index columns of
            the detector that are assumed to be aligned with the rotation axis.
        param1, param2 (scalars):
            These are view-independent scalar parameters that are required for the geometry and are not already included in the parent class.
        view_dependent_vec1, view_dependent_vec2 (jnp.ndarray):
            These are view-dependent parameter vectors each with length = number of views.
    """

    # TODO: Adjust the signature as needed for a particular geometry and update the docstring to match.
    # Don't include any additional unspecified keyword arguments in the form of **kwargs - use only the parameters that
    # are required for the geometry.  Any changes to existing parameters should be done by the user with set_params,
    # which checks for invalid parameter names.  There is no check for invalid parameter names here because the
    # geometry may need to define new parameters.
    def __init__(self, sinogram_shape, param1, param2, view_dependent_vec1, view_dependent_vec2):
        # Convert the view-dependent vectors to an array
        view_dependent_vecs = [vec.flatten() for vec in [view_dependent_vec1, view_dependent_vec2]]
        try:
            view_params_array = jnp.stack(view_dependent_vecs, axis=1)
        except ValueError as e:
            raise ValueError("Incompatible view dependent vector lengths:  all view-dependent vectors must have the "
                             "same length.")

        super().__init__(sinogram_shape, param1=param1, param2=param2, view_params_array=view_params_array)

    @classmethod
    def from_file(cls, filename):
        """
        Construct a ConeBeamModel from parameters saved using save_params()

        Args:
            filename (str): Name of the file containing parameters to load.

        Returns:
            ConeBeamModel with the specified parameters.
        """
        # Load the parameters and convert view-dependent parameters to use the geometry-specific keywords.
        params = ParameterHandler.load_param_dict(filename, values_only=True)
        view_params_array = params['view_params_array']

        # TODO: Adjust these to match the signature of __init__
        view_dependent_vec1 = view_params_array[:, 0]
        view_dependent_vec2 = view_params_array[:, 1]
        del params['view_params_array']
        return cls(view_dependent_vec1=view_dependent_vec1, view_dependent_vec2=view_dependent_vec2, **params)

    def get_magnification(self):
        """
        Compute the scale factor from a voxel at iso (at the origin on the center of rotation) to
        its projection on the detector.  For parallel beam, this is 1, but it may be parameter-dependent
        for other geometries.

        Returns:
            (float): magnification
        """
        # TODO: Adjust as needed for the geometry.
        magnification = 1.0
        return magnification

    def verify_valid_params(self):
        """
        Check that all parameters are compatible for a reconstruction.
        """
        super().verify_valid_params()
        sinogram_shape, view_params_array = self.get_params(['sinogram_shape', 'view_params_array'])

        # TODO: Modify as needed for the geometry.
        if view_params_array.shape[0] != sinogram_shape[0]:
            error_message = "Number view dependent parameter vectors must equal the number of views. \n"
            error_message += "Got {} for length of view-dependent parameters and "
            error_message += "{} for number of views.".format(view_params_array.shape[0], sinogram_shape[0])
            raise ValueError(error_message)

    def get_geometry_parameters(self):
        """
        Required function to get the view independent geometry parameters required for projection.

        Returns:
            namedtuple of any parameters required for back_project_one_view_to_pixel_batch or forward_project_pixel_batch_to_one_view.
            This does not need to include view dependent parameters, or sinogram_shape or recon_shape, which
            are passed in automatically to projector_params.
        """
        # TODO: Include additional names as needed for the projectors.
        # First get the parameters managed by ParameterHandler
        geometry_param_names = ['delta_det_channel', 'det_channel_offset', 'delta_voxel']
        geometry_param_values = self.get_params(geometry_param_names)

        # Then get additional parameters that are calculated separately, such as psf_radius and magnification.
        # geometry_param_names += ['psf_radius']
        # geometry_param_values.append(self.get_psf_radius())
        # geometry_param_names += ['magnification']
        # geometry_param_values.append(self.get_magnifiction())

        # Then create a namedtuple to access parameters by name in a way that can be jit-compiled.
        GeometryParams = namedtuple('GeometryParams', geometry_param_names)
        geometry_params = GeometryParams(*tuple(geometry_param_values))

        return geometry_params

    def auto_set_recon_size(self, sinogram_shape, no_compile=True, no_warning=False):
        """Compute the default recon size using the internal parameters delta_channel and delta_pixel plus
          the number of channels from the sinogram"""
        # TODO: provide code to implement this
        raise NotImplementedError('auto_set_recon_size must be implemented by each specific geometry model.')

    @staticmethod
    @partial(jax.jit, static_argnames='projector_params')
    def forward_project_pixel_batch_to_one_view(voxel_values, pixel_indices, single_view_params, projector_params):
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
        # Get all the geometry parameters - we use gp since geometry parameters is a named tuple and we'll access
        # elements using, for example, gp.delta_det_channel, so a longer name would be clumsy.
        gp = projector_params.geometry_params  # This is the namedtuple from get_geometry_parameters
        num_views, num_det_rows, num_det_channels = projector_params.sinogram_shape
        num_recon_rows, num_recon_columns, num_recon_slices = projector_params.recon_shape

        # Returns a single view of the sinogram
        sinogram_view = jnp.zeros((num_det_rows, num_det_channels))

        # TODO:  Provide code to implement forward projection

        return sinogram_view

    @staticmethod
    @partial(jax.jit, static_argnames='projector_params')
    def back_project_one_view_to_pixel_batch(sinogram_view, pixel_indices, single_view_params, projector_params, coeff_power=1):
        """
        Calculate the backprojection value at a specified recon voxel cylinders given a sinogram view and various parameters.

        NOTE: This function must be able to be jit-compiled.

        Args:
            sinogram_view (2D jax array): one view of the sinogram to be back projected
            pixel_indices (1D jax array of int):  indices into flattened array of size num_rows x num_cols.
            single_view_params: These are the view dependent parameters for the view being back projected.
            projector_params (1D jax array): tuple of (sinogram_shape, recon_shape, get_geometry_params()).
            coeff_power (int): backproject using the coefficients of (A_ij ** coeff_power).
                Normally 1, but should be 2 when computing theta 2.

        Returns:
            The voxel values for all slices at the input index (i.e., a voxel cylinder) obtained by backprojecting
            the input sinogram view.
        """
        # Get all the geometry parameters - we use gp since geometry parameters is a named tuple and we'll access
        # elements using, for example, gp.delta_det_channel, so a longer name would be clumsy.
        gp = projector_params.geometry_params  # This is the namedtuple from get_geometry_parameters
        num_views, num_det_rows, num_det_channels = projector_params.sinogram_shape
        num_recon_rows, num_recon_columns, num_recon_slices = projector_params.recon_shape

        # Computes the voxel values in all slices corresponding to pixel_index
        num_pixels = pixel_indices.shape[0]
        voxel_values_cylinder = jnp.zeros((num_pixels, num_recon_slices))

        # TODO:  Provide code to implement forward projection

        return voxel_values_cylinder
