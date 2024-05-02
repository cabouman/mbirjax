import warnings

import jax
from jax import numpy as jnp, lax

from mbirjax import TomographyModel


class GeometryTemplateModel(TomographyModel):
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
            These are view independent scalar parameters that are required for the geometery and are not already included in the parent class.
        view_dependent_vec1, view_dependent_vec2 (jnp.ndarray):
            These are view dependent parameter vectors each with length = number of views.
        **kwargs (dict):
            Additional keyword arguments that are passed to the :ref:`TomographyModelDocs` constructor. These can
            include settings and configurations specific to the tomography model such as noise models or image dimensions.
            Refer to :ref:`TomographyModelDocs` documentation for a detailed list of possible parameters.
    """

    def __init__(self, sinogram_shape, param1, param2, view_dependent_vec1, view_dependent_vec2, **kwargs):
        view_params_array = jnp.stack([view_dependent_vec1, view_dependent_vec2], axis=-1)

        super().__init__(sinogram_shape, param1=param1, param2=param2, view_params_array=view_params_array, **kwargs)

    def verify_valid_params(self):
        """
        Check that all parameters are compatible for a reconstruction.
        """
        super().verify_valid_params()
        sinogram_shape, view_params_array = self.get_params(['sinogram_shape', 'view_params_array'])

        if view_params_array.shape[0] != sinogram_shape[0]:
            error_message = "Number view dependent parameter vectors must equal the number of views. \n"
            error_message += "Got {} for number of angles and {} for number of views.".format(len(angles),
                                                                                              sinogram_shape[0])
            raise ValueError(error_message)

    def get_geometry_parameters(self):
        """
        TODO: This needs to be changed so that it returns the view_params_array
        Required function to get a list of the primary geometry parameters for projection.

        Returns:
            List of delta_det_channel, det_channel_offset, delta_pixel_recon,
            num_recon_rows, num_recon_cols, num_recon_slices
        """
        geometry_params = self.get_params(['delta_det_channel', 'det_channel_offset', 'delta_pixel_recon',
                                           'num_recon_rows', 'num_recon_cols', 'num_recon_slices'])

        return geometry_params


    @staticmethod
    def back_project_one_view_to_voxel(sinogram_view, voxel_index, single_view_params, geometry_params, coeff_power=1):
        """
        Calculate the backprojection value at a specified recon voxels given a sinogram view and various parameters.
        This code uses the distance driven projector.

        Args:
            sinogram_view (jax array): one view of the sinogram to be back projected
            voxel_index: the integer index into flattened recon - need to apply unravel_index(voxel_index, recon_shape) to get i, j, k
            single_view_params: These are the view dependent parameters for this view
            geometry_params:
            coeff_power: [int] backproject using the coefficients of (A_ij ** coeff_power).
                Normally 1, but should be 2 when computing theta 2.

        Returns:
            The value of the voxel at the input index obtained by backprojecting the input sinogram.
        """
        # The number of slices will need to come from geometry_params
        num_slices = 1

        # Computes the voxel values in all slices corresponding to voxel_index
        voxel_values_cylinder = jnp.zeros(num_slices)
        return voxel_values_cylinder


    @staticmethod
    def forward_project_voxels_one_view(voxel_values, voxel_indices, single_view_params, geometry_params, sinogram_shape):
        """
        Forward project a set of voxels determined by indices into the flattened array of size num_rows x num_cols.

        Args:
            voxel_values (jax array):  2D array of shape (num_indices, num_slices) of voxel values, where
                voxel_values[i, j] is the value of the voxel in slice j at the location determined by indices[i].
            voxel_indices (jax array of int):  1D vector of indices into flattened array of size num_rows x num_cols.
            angle (float):  angle for this view
            geometry_params (list): Geometry parameters from get_geometry_params()
            sinogram_shape (tuple): Sinogram shape (num_views, num_det_rows, num_det_channels)

        Returns:
            jax array of shape (num_det_rows, num_det_channels)
        """

        # Returns a single view of the sinogram
        sinogram_view = jnp.zeros(sinogram_shape[1:])
        return sinogram_view
