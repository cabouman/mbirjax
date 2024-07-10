import jax
import jax.numpy as jnp
import scipy.ndimage as snd
import mbirjax
from functools import partial
from mbirjax import TomographyModel, ParameterHandler


class Blur(TomographyModel):
    """
    A class designed for handling forward and backward projections using a blur.
    This class is used mainly for development and demo.  It shows how to use
    jax.vjp to implement a back projector given a forward projector.

    This class inherits all methods and properties from the :ref:`TomographyModelDocs` and may override some
    to suit parallel beam geometrical requirements. See the documentation of the parent class for standard methods
    like setting parameters and performing projections and reconstructions.

    Parameters not included in the constructor can be set using the set_params method of :ref:`TomographyModelDocs`.
    Refer to :ref:`TomographyModelDocs` documentation for a detailed list of possible parameters.

    Args:
        recon_shape (tuple):
            Shape of the recon as a tuple in the form `(rows, columns, slices)`.
        sigma_psf (float):
            The 2D spatial standard deviation of the Gaussian blur to be applied..

    See Also
    --------
    TomographyModel : The base class from which this class inherits.
    """

    def __init__(self, recon_shape, sigma_psf):
        view_params_array = jnp.zeros((1, 1))
        sinogram_shape = recon_shape
        super().__init__(sinogram_shape, view_params_array=view_params_array, sigma_psf=sigma_psf)
        self.set_params(recon_shape=recon_shape)

    @classmethod
    def from_file(cls, filename):
        """
        Construct a model from parameters saved using save_params()

        Args:
            filename (str): Name of the file containing parameters to load.

        Returns:
            Model with the specified parameters.
        """
        # Load the parameters and convert to use the ParallelBeamModel keywords.
        required_param_names = ['recon_shape', 'sigma_psf']
        required_params, params = ParameterHandler.load_param_dict(filename, required_param_names, values_only=True)

        new_model = cls(**required_params)
        new_model.set_params(**params)
        return new_model

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
        pass

    def auto_set_recon_size(self, sinogram_shape, no_compile=True, no_warning=False):

        recon_shape = sinogram_shape
        self.set_params(no_compile=no_compile, no_warning=no_warning, recon_shape=recon_shape)

    def create_projectors(self):
        """
        Creates an instance of the Projectors class and set the local instance variables needed for forward
        and back projection and compute_hessian_diagonal.  This method requires that the current geometry has
        implementations of :meth:`forward_project_pixel_batch_to_one_view` and :meth:`back_project_one_view_to_pixel_batch`

        Returns:
            Nothing, but creates jit-compiled functions.
        """
        self.sparse_forward_project = self.sparse_forward
        self.sparse_back_project = self.sparse_back
        self.compute_hessian_diagonal = self.hessian_diagonal

    def sparse_forward(self, voxel_values, indices, view_indices=()):
        """
        Forward project the given voxel cylinders.
        The indices are into a flattened 2D array of shape (recon_rows, recon_cols), and the projection is done using
        all voxels with those indices across all the slices.

        Args:
            voxel_values (ndarray or jax array): 2D array of voxel values to project, size (len(pixel_indices), num_recon_slices).
            indices (ndarray or jax array): Array of indices specifying which voxels to project.
            view_indices (ndarray or jax array, optional): Unused

        Returns:
            jnp array: The resulting 3D image after applying the blur.
        """
        blurred_image_shape, sigma = self.get_params(['sinogram_shape', 'sigma_psf'])
        blurred_image = jnp.zeros(blurred_image_shape).reshape((-1, blurred_image_shape[2]))
        blurred_image = blurred_image.at[indices].set(voxel_values)
        blurred_image = blurred_image.reshape(blurred_image_shape)
        if sigma > 0:
            blurred_image = gaussian_filter(blurred_image, sigma, axes=(0, 1))

        return blurred_image

    def sparse_back(self, blurred_image, indices, coeff_power=1, view_indices=()):
        """
        Back project the image to the voxels given by the indices.
        The indices are into a flattened 2D array of shape (recon_rows, recon_cols), and the projection is done using
        all voxels with those indices across all the slices.

        Args:
            blurred_image (jnp array): 3D jax array containing the image.
            indices (jnp array): Array of indices specifying which voxels to back project.
            coeff_power (int): backproject using the coefficients of (A_ij ** coeff_power).
                Normally 1, but should be 2 for compute_hessian_diagonal.
            view_indices (ndarray or jax array, optional): Unused

        Returns:
            A jax array of shape (len(indices), num_slices)
        """
        sigma = self.get_params('sigma_psf')
        # TODO:  This assumes a symmetric kernel.  Should change to allow a more general kernel.
        if sigma > 0:
            bp_image = gaussian_filter(blurred_image, sigma, axes=(0, 1), coeff_power=coeff_power)
        else:
            bp_image = blurred_image
        bp_image = bp_image.reshape((-1, blurred_image.shape[2]))
        recon_at_indices = bp_image[indices]
        return recon_at_indices

    def sparse_back_vjp(self, blurred_image, indices, coeff_power=1, view_indices=()):
        """
        Back project the image to the voxels given by the indices by using autodifferentiation on sparse_forward.
        The indices are into a flattened 2D array of shape (recon_rows, recon_cols), and the projection is done using
        all voxels with those indices across all the slices.

        Args:
            blurred_image (jnp array): 3D jax array containing the image.
            indices (jnp array): Array of indices specifying which voxels to back project.
            coeff_power (int): backproject using the coefficients of (A_ij ** coeff_power).
                Normally 1, but should be 2 for compute_hessian_diagonal.
            view_indices (ndarray or jax array, optional): Unused

        Returns:
            A jax array of shape (len(indices), num_slices)
        """
        sigma = self.get_params('sigma_psf')
        input_shape = blurred_image.shape
        x = jnp.ones((len(indices), blurred_image.shape[2]))
        # TODO:  Implement a direct Hessian diagonal.  The problem is how to differentiate efficiently on just one input and output at a time.
        if coeff_power == 2:
            return self.sparse_back(blurred_image, indices, coeff_power=2)

        # Set up the forward model to be able to use vjp = vector Jacobian product = v^T A = (A^T v)^T
        def local_forward(voxel_values):
            return self.sparse_forward(voxel_values, indices)
        if sigma > 0:
            vjp_fun = jax.vjp(local_forward, x)[1]
            bp_image = vjp_fun(blurred_image)[0]
            if coeff_power == 2:
                bp_image = bp_image.reshape(input_shape)
                bp_image = vjp_fun(bp_image)[0]
        else:
            bp_image = blurred_image
        return bp_image

    def hessian_diagonal(self, weights=None):
        """
        Computes the diagonal elements of the Hessian matrix for given weights.

        Args:
            weights (jax array, optional): 3D positive weights with same shape as sinogram.  Defaults to all 1s.
            view_indices (ndarray or jax array, optional): 1D array of indices into the view parameters array.
                If None, then all views are used.

        Returns:
            jnp array: Diagonal of the Hessian matrix with same shape as recon.
        """
        projected_shape, recon_shape = self.get_params(['sinogram_shape', 'recon_shape'])
        if weights is None:
            weights = jnp.ones(projected_shape)
        elif weights.shape != projected_shape:
            error_message = 'Weights must be constant or an array compatible with sinogram'
            error_message += '\nGot weights.shape = {}, but sinogram.shape = {}'.format(weights.shape, projected_shape)
            raise ValueError(error_message)

        num_recon_rows, num_recon_cols, num_recon_slices = recon_shape[:3]
        max_index = num_recon_rows * num_recon_cols
        indices = jnp.arange(max_index)
        hessian = self.sparse_back(weights, indices, coeff_power=2)
        return hessian


# From scipy:
# Copyright (C) 2003-2005 Peter J. Verveer
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. The name of the author may not be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from scipy.ndimage import _ni_support


@partial(jax.jit, static_argnames=['sigma', 'order', 'axes', 'truncate', 'radius', 'coeff_power'])
def gaussian_filter(input_image, sigma, order=0, axes=None,
                    truncate=4.0, *, radius=None, coeff_power=1):
    """Multidimensional Gaussian filter.

    Parameters
    ----------
    %(input)s
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes.
    order : int or sequence of ints, optional
        The order of the filter along each axis is given as a sequence
        of integers, or as a single number. An order of 0 corresponds
        to convolution with a Gaussian kernel. A positive order
        corresponds to convolution with that derivative of a Gaussian.
    truncate : float, optional
        Truncate the filter at this many standard deviations.
        Default is 4.0.
    radius : None or int or sequence of ints, optional
        Radius of the Gaussian kernel. The radius are given for each axis
        as a sequence, or as a single number, in which case it is equal
        for all axes. If specified, the size of the kernel along each axis
        will be ``2*radius + 1``, and `truncate` is ignored.
        Default is None.

    Returns
    -------
    gaussian_filter : ndarray
        Returned array of same shape as `input`.

    Notes
    -----
    The multidimensional filter is implemented as a sequence of
    1-D convolution filters. The intermediate arrays are
    stored in the same data type as the output. Therefore, for output
    types with a limited precision, the results may be imprecise
    because intermediate results may be stored with insufficient
    precision.

    The Gaussian kernel will have size ``2*radius + 1`` along each axis
    where ``radius = round(truncate * sigma)``.

    Examples
    --------
    >>> from scipy.ndimage import gaussian_filter
    >>> import numpy as np
    >>> a = np.arange(50, step=2).reshape((5,5))
    >>> a
    array([[ 0,  2,  4,  6,  8],
           [10, 12, 14, 16, 18],
           [20, 22, 24, 26, 28],
           [30, 32, 34, 36, 38],
           [40, 42, 44, 46, 48]])
    >>> gaussian_filter(a, sigma=1)
    array([[ 4,  6,  8,  9, 11],
           [10, 12, 14, 15, 17],
           [20, 22, 24, 25, 27],
           [29, 31, 33, 34, 36],
           [35, 37, 39, 40, 42]])

    >>> from scipy import datasets
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ax1 = fig.add_subplot(121)  # left side
    >>> ax2 = fig.add_subplot(122)  # right side
    >>> ascent = datasets.ascent()
    >>> result = gaussian_filter(ascent, sigma=5)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result)
    >>> plt.show()
    """
    if axes is None:
        axes = list(range(input_image.ndim))
    orders = _ni_support._normalize_sequence(order, len(axes))
    sigmas = _ni_support._normalize_sequence(sigma, len(axes))
    radiuses = _ni_support._normalize_sequence(radius, len(axes))

    axes = [(axes[ii], sigmas[ii], orders[ii], radiuses[ii])
            for ii in range(len(axes)) if sigmas[ii] > 1e-15]

    if len(axes) > 0:
        for axis, sigma, order, radius in axes:
            sd = float(sigma)
            # make the radius of the filter equal to truncate standard deviations
            lw = int(truncate * sd + 0.5)
            if radius is not None:
                lw = radius
            if not isinstance(lw, int) or lw < 0:
                raise ValueError('Radius must be a nonnegative integer.')

            weights = _gaussian_kernel1d(sigma, order, lw) ** coeff_power
            input_image = jnp.moveaxis(input_image, axis, 0)
            cur_shape = input_image.shape
            input_image = input_image.reshape((cur_shape[0], -1))
            input_image = jax.vmap(convolve_reflect_1d, in_axes=(1, None), out_axes=1)(input_image, weights)
            input_image = input_image.reshape(cur_shape)
            input_image = jnp.moveaxis(input_image, 0, axis)
    else:
        input_image = jnp.copy(input_image)
    return input_image


def convolve_reflect_1d(input1, weights):
    """
    Perform 1D convolution with reflected boundary conditions to match scipy.ndimage.convolve1d
    """
    lw = (len(weights) - 1) // 2
    # Convolve with full overlap between inputs and weights
    c1 = jnp.convolve(input1, weights)
    # To get reflected convolution, flip the two ends outside the original region and add them to the full convolution
    # so that each end point is added twice, then subtract the contributions that were doubled
    c1 = c1.at[lw:2 * lw].set(c1[lw:2 * lw] + jnp.flip(c1[0:lw]))
    c1 = c1.at[-2 * lw:-lw].set(c1[-2 * lw:-lw] + jnp.flip(c1[-lw:]))
    c1 = c1.at[lw:-lw].get()
    return c1


def _gaussian_kernel1d(sigma, order, radius):
    """
    Computes a 1-D Gaussian convolution kernel.
    """
    if order < 0:
        raise ValueError('order must be non-negative')
    exponent_range = jnp.arange(order + 1)
    sigma2 = sigma * sigma
    x = jnp.arange(-radius, radius+1)
    phi_x = jnp.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    if order == 0:
        return phi_x
    else:
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        q = jnp.zeros(order + 1)
        q[0] = 1
        D = jnp.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = jnp.diag(jnp.ones(order)/-sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x
