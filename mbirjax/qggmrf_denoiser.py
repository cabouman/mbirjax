from collections import namedtuple
import io
import datetime
from typing import Literal, Union, overload, Any
import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
import mbirjax as mj
from mbirjax import TomographyModel

QGGMRFDenoiserParamNames = mj.ParamNames | Literal['sigma_noise']


class QGGMRFDenoiser(TomographyModel):
    """
    The QGGMRFDenoiser uses the MBIRJAX recon framework to implement a qggmrf proximal map denoiser.
    The primary interface is through :meth:`denoise`.
    """

    def __init__(self, sinogram_shape):

        view_params_name = 'None'
        super().__init__(sinogram_shape, view_params_name=view_params_name, sigma_noise=None)
        self.use_ror_mask = False
        self.set_params(sharpness=0)  # The default sharpness level is 0 for the denoiser.

        self.set_params(granularity=[16], partition_sequence=[0])  # For qggmrf denoising, we can fix a partition
        try:
            jax.devices('gpu')
            self.set_params(use_gpu='full')  # If the denoising problem doesn't fit on the gpu, we should divide it up.
        except RuntimeError:
            self.set_params(use_gpu='none')

    @overload
    def get_params(self, parameter_names: Union[QGGMRFDenoiserParamNames, list[QGGMRFDenoiserParamNames]]) -> Any: ...

    def get_params(self, parameter_names) -> Any:
        return super().get_params(parameter_names)

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

        image_shape = self.get_params('recon_shape')
        if image_shape != sinogram_shape:
            error_message = "image_shape and sinogram_shape must be the same. \n"
            error_message += "Got {} for image_shape and {} for sinogram_shape".format(image_shape, sinogram_shape)
            raise ValueError(error_message)

    def get_geometry_parameters(self):
        raise NotImplementedError('get_geometry_parameters is not implemented for QGGMRFDenoiser.')

    def create_projectors(self):
        pass

    def auto_set_recon_size(self, sinogram_shape, no_compile=True, no_warning=False):
        """Compute the default recon shape to equal the sinogram shape"""
        self.set_params(no_compile=no_compile, no_warning=no_warning, recon_shape=sinogram_shape)

    def auto_set_sigma_y(self, sinogram, sino_indicator, weights=1):
        sigma_y = self.get_params('sigma_noise')
        self.set_params(no_warning=True, sigma_y=sigma_y, auto_regularize_flag=True)

    def estimate_image_noise_std(self, image):
        """
        Estimate the standard deviation of the reconstruction from the noisy image.

        Args:
            image (jax array or ndarray): 3D array containing noisy image with shape (num_views, num_det_rows, num_det_channels).
        """
        support_indicator = self._get_sino_indicator(image, sigma_noise=0.0)
        sigma_noise = self._get_estimate_of_recon_std(image, support_indicator)
        support_indicator = self._get_sino_indicator(image, sigma_noise=sigma_noise)
        sigma_noise = self._get_estimate_of_recon_std(image, support_indicator)
        return sigma_noise

    def _get_estimate_of_recon_std(self, noisy_image, support_indicator):
        """
        Estimate the standard deviation of the reconstruction from the noisy image.  This is used to scale sigma_prox and
        sigma_x in MBIR reconstruction.

        Args:
            noisy_image (ndarray): 3D jax array containing noisy image with shape (num_views, num_det_rows, num_det_channels).
            support_indicator (ndarray): a binary mask that indicates the region of image support; same shape as noisy_image.
        """

        # # Compute the typical magnitude of a noisy image value
        # typical_image_value = np.average(np.abs(noisy_image), weights=support_indicator)
        # return typical_image_value

        inds = np.where(support_indicator)
        inds = [np.clip(inds[j], 1, None) for j in range(len(inds))]
        vals = np.stack([noisy_image[inds[0], inds[1], inds[2]],
                        noisy_image[inds[0]-1, inds[1], inds[2]],
                        noisy_image[inds[0], inds[1]-1, inds[2]],
                        noisy_image[inds[0], inds[1], inds[2]-1]], axis=0)
        std = np.std(vals, axis=0)
        recon_std = np.mean(std)

        return recon_std  # typical_image_value

    def _get_sino_indicator(self, noisy_image, sigma_noise=None):
        """
        Compute a binary mask that indicates the region of noisy_image support.

        Args:
            noisy_image (jax array or ndarray): 3D array containing noisy_image with shape (num_views, num_det_rows, num_det_channels).
            sigma_noise (float, optional): Estimated noise standard deviation in the image.  If None, then this is estimated from the image.

        Returns:
            (ndarray): Weights used in mbircone reconstruction, with the same array shape as ``noisy_image``.
        """
        if sigma_noise is None:
            sigma_noise = self.get_params('sigma_noise')

        percent_noise_floor = 5.0
        # Form indicator by thresholding noisy_image
        threshold = (0.01 * percent_noise_floor) * np.mean(np.fabs(noisy_image)) + sigma_noise
        threshold = min(threshold, np.amax(noisy_image))
        indicator = np.int8(noisy_image >= threshold)
        return indicator

    def recon(self, *args, **kwargs):
        raise NotImplementedError('recon is not implemented for QGGMRFDenoiser.  Use `denoise` instead.')

    def denoise(self, image, sigma_noise=None, use_ror_mask=False, init_image=None, max_iterations=15,
                stop_threshold_change_pct=0.2, first_iteration=0, logfile_path='./logs/recon.log', print_logs=True):
        """
        Use the VCD algorithm with the QGGMRF loss to denoise a 3D image (volume).

        With default settings, and with X a clean image and W equal to AWGN of standard deviation sigma_noise,
        the result of :meth:`denoise` applied to X+W is roughly the denoised image X' that minimizes
        NRMSE(X, X') over all choices of sigma_noise.

        The amount of denoising can be changed by changing sigma_noise.  If sigma_noise is None, then sigma_noise
        is estimated from a set of samples from the image.

        Denoising strength can also be adjusted using parameters sharpness (default=0.0) and/or snr_db (default=30).

        Args:
            image (numpy or jax array):  The 3D volume to be denoised.
            sigma_noise (float, optional):  The estimated noise variance in the noisy image.  If None, then the noise level is estimated from the image.
            use_ror_mask (bool, optional): Set true to restrict denoising to an inscribed circle in the image.  Defaults to False.
            init_image (numpy or jax array, optional):  An initial image for the minimization.  Defaults to image.
            max_iterations (int, optional): maximum number of iterations of the VCD algorithm to perform.
            stop_threshold_change_pct (float, optional): Stop reconstruction when 100 * ||delta_recon||_1 / ||recon||_1 change from one iteration to the next is below stop_threshold_change_pct.  Defaults to 0.2.  Set this to 0 to guarantee exactly max_iterations.
            first_iteration (int, optional): Set this to be the number of iterations previously completed when restarting a recon using init_recon.  This defines the first index in the partition sequence.  Defaults to 0.
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
            >>> denoiser.set_params(sharpness=0.5)  # Increase sharpness a little over the default of 0.0
            >>> denoised_image, denoised_dict = denoiser.denoise(noisy_image)  # Estimate the noise level from the image
            >>> mj.slice_viewer(noisy_image, denoised_image, data_dicts=[None, denoised_dict], title='Noisy and denoised images')

        See Also
        --------
        TomographyModel : The base class from which this class inherits.
        """
        self.use_ror_mask = use_ror_mask
        if sigma_noise is None:
            sigma_noise = self.estimate_image_noise_std(image)
        self.set_params(sigma_noise=sigma_noise)
        if first_iteration == 0 or self.logger is None:
            self.setup_logger(logfile_path=logfile_path, print_logs=print_logs)

        self.logger.info('Initializing QGGMRFDenoiser')
        regularization_params = self.auto_set_regularization_params(image)

        # Generate set of voxel partitions
        image_shape, granularity = self.get_params(['recon_shape', 'granularity'])
        partition_sequence = self.get_params('partition_sequence')
        partition_index = partition_sequence[0]
        partitions = mj.gen_set_of_pixel_partitions(image_shape, [granularity[partition_index]], use_ror_mask=self.use_ror_mask)

        # Generate sequence of partitions to use
        partition = partitions[0]

        if init_image is None:
            init_image = image.copy()

        flat_error_image = (image - init_image).reshape((-1, image.shape[-1]))
        flat_error_image = jax.device_put(flat_error_image, self.worker)
        image_out = init_image

        verbose, sigma_y = self.get_params(['verbose', 'sigma_y'])

        # Initialize the output image
        flat_image = image_out.reshape((-1, image_shape[2]))
        flat_image = jax.device_put(flat_image, self.worker)

        # Create the finer grained image update operators

        fm_constant = 1.0 / (self.get_params('sigma_y') ** 2.0)
        qggmrf_nbr_wts, sigma_x, p, q, T = self.get_params(['qggmrf_nbr_wts', 'sigma_x', 'p', 'q', 'T'])
        b = mj.get_b_from_nbr_wts(qggmrf_nbr_wts)
        qggmrf_params = tuple((b, sigma_x, p, q, T))
        image_shape = self.get_params('recon_shape')

        @jax.jit  # JIT the whole sweep
        def denoise_over_partition(local_flat_image, local_flat_error_image):
            """Run vcd_subset_denoiser over every subset in `partition`
            using lax.fori_loop to keep the loop on-device and JIT-compatible."""
            # Analog of vcd_partition_iterator

            # Bundle *all* mutable state into one carry object
            def body_fn(i, carry):
                body_flat_image, body_flat_error_image, ell1_accum, alpha_accum = carry

                subset = partition[i]  # pick i-th subset

                body_flat_image, body_flat_error_image, ell1_subset, alpha_subset = (
                    vcd_subset_denoiser(body_flat_image, body_flat_error_image,
                                        subset, fm_constant, qggmrf_params, image_shape))

                # update running totals
                ell1_accum = ell1_accum + ell1_subset
                alpha_accum = alpha_accum + alpha_subset

                return (body_flat_image, body_flat_error_image,
                        ell1_accum, alpha_accum)

            # initial carry
            init_carry = (local_flat_image, local_flat_error_image,
                          0.0, 0.0)

            # run 0 … N-1
            local_flat_image, local_flat_error_image, local_ell1, local_alpha = (
                lax.fori_loop(0, partition.shape[0], body_fn, init_carry))
            final_carry = (local_flat_image, local_flat_error_image, local_ell1, local_alpha / partition.shape[0])
            return final_carry  # unpack outside if you like

        # pre-allocate history arrays (static length = max_iters)
        max_iters = max_iterations
        fm_rmse_init = jnp.zeros(max_iters)
        nmae_update_init = jnp.zeros(max_iters)
        alpha_values_init = jnp.zeros(max_iters)

        stop_thresh = stop_threshold_change_pct / 100.0  # scalar threshold

        @jax.jit
        def run_denoising_loop(local_flat_image, local_flat_error_image):
            """
            Runs the outer optimisation loop with lax.while_loop.
            Returns:
                (local_flat_image, local_flat_error_image,
                 fm_rmse_hist, nmae_hist, alpha_hist, num_iters)
            """

            # -------- carry = all mutable state --------
            # i                 : iteration counter
            # local_flat_image        : current image
            # local_flat_error_image  : current residual
            # fm_rmse_hist      : history array (filled progressively)
            # nmae_hist         : "
            # alpha_hist        : "
            # nmae_curr         : nmae from *most recent* iteration
            carry0 = (0, local_flat_image, local_flat_error_image, fm_rmse_init,
                      nmae_update_init, alpha_values_init, jnp.inf  # nmae_curr (start high so loop begins)
                      )

            # -------- termination condition --------
            def cond_fn(carry):
                i, *_, nmae_curr = carry
                not_enough_iters = i < max_iters
                change_big_enough = nmae_curr >= stop_thresh
                return jnp.logical_and(not_enough_iters, change_big_enough)

            # -------- body: one outer iteration --------
            def body_fn(carry):
                (i, flat_img, flat_err_img, fm_hist, nmae_body, alpha_body, _) = carry

                # inner loop (already JAX-ified with fori_loop previously)
                flat_img, flat_err_img, ell1_for_part, alpha_val = \
                    denoise_over_partition(flat_img, flat_err_img)

                # --- compute stats ---
                nmae = ell1_for_part / jnp.sum(jnp.abs(flat_img))
                fm = self.get_forward_model_loss(flat_err_img, sigma_y)

                # --- write into the history arrays ---
                fm_hist = fm_hist.at[i].set(fm)
                nmae_body = nmae_body.at[i].set(nmae)
                alpha_body = alpha_body.at[i].set(alpha_val)

                # bump iteration counter
                return i + 1, flat_img, flat_err_img, fm_hist, nmae_body, alpha_body, nmae  # new nmae for cond_fn

            # run the while-loop
            final_carry = lax.while_loop(cond_fn, body_fn, carry0)

            # unpack results
            (num_iters_loop, local_flat_image, local_flat_error_image,
             fm_rmse_hist, nmae_hist, alpha_hist, _) = final_carry

            return (local_flat_image, local_flat_error_image,
                    fm_rmse_hist, nmae_hist, alpha_hist, num_iters_loop)

        self.logger.info('Starting VCD iterations')
        if verbose >= 2:
            output = io.StringIO()
            mj.get_memory_stats(file=output)
            self.logger.debug(output.getvalue())
            self.logger.debug('--------')

        # Do the iterations
        (flat_image, flat_error_image, fm_rmse,  # full length = max_iters (unused slots stay 0)
         nmae_update, alpha_values, num_iters) = run_denoising_loop(flat_image, flat_error_image)

        recon_params = (fm_rmse[0:num_iters], nmae_update[0:num_iters], alpha_values[0:num_iters])

        fm_rmse = [float(val) for val in recon_params[0]]
        stop_threshold_change_pct = [100 * float(val) for val in recon_params[1]]
        alpha_values = [float(val) for val in recon_params[2]]
        num_iterations = len(fm_rmse)

        prior_loss = None
        recon_param_values = [num_iterations, granularity, partition_sequence, fm_rmse, prior_loss,
                              regularization_params, stop_threshold_change_pct, alpha_values]
        recon_params = mj.ReconParams(*tuple(recon_param_values))._asdict()

        notes = 'Reconstruction completed: {}\n\n'.format(datetime.datetime.now())
        denoiser_dict = self.get_recon_dict(recon_params, notes=notes)
        denoised_image = self.reshape_recon(flat_image)

        return denoised_image, denoiser_dict

def vcd_subset_denoiser(flat_image, flat_error_image, pixel_indices,
                        fm_constant, qggmrf_params, image_shape):
    # This is the analog of vcd_subset_updater
    # qGGMRF prior - compute the qggmrf gradient and hessian at each pixel in the index set.
    prior_grad, prior_hess = (
        mj.qggmrf_gradient_and_hessian_at_indices(flat_image, image_shape, pixel_indices,
                                                  qggmrf_params))

    # Back project to get the gradient - the forward Hessian is all 1s for the qggmrf proximal map
    cur_error_image = flat_error_image[pixel_indices]
    forward_grad = - fm_constant * cur_error_image
    forward_hess = 1

    # Compute update vector update direction in recon domain
    delta_recon_at_indices = - ((forward_grad + prior_grad) / (forward_hess + prior_hess))

    # Compute delta^T \nabla Q(x_hat; x'=x_hat) for use in finding alpha
    prior_linear = jnp.sum(prior_grad * delta_recon_at_indices)

    # Estimated upper bound for hessian
    prior_overrelaxation_factor = 1.1
    prior_quadratic_approx = ((1 / prior_overrelaxation_factor) *
                              jnp.sum(prior_hess * delta_recon_at_indices ** 2))

    # Compute update direction in sinogram domain
    delta_sinogram = delta_recon_at_indices
    forward_linear = fm_constant * jnp.tensordot(cur_error_image, delta_sinogram, axes=2)
    forward_quadratic = fm_constant * jnp.tensordot(delta_sinogram, delta_sinogram, axes=2)

    # Compute optimal update step
    alpha_numerator = forward_linear - prior_linear
    alpha_denominator = forward_quadratic + prior_quadratic_approx + jnp.finfo(jnp.float32).eps
    alpha = alpha_numerator / alpha_denominator
    max_alpha = 1.5
    alpha = jnp.clip(alpha, jnp.finfo(jnp.float32).eps, max_alpha)

    delta_recon_at_indices = alpha * delta_recon_at_indices
    flat_image = flat_image.at[pixel_indices].add(delta_recon_at_indices)

    # Update sinogram and loss
    cur_error_image = cur_error_image - alpha * delta_sinogram
    flat_error_image = flat_error_image.at[pixel_indices].set(cur_error_image)
    ell1_for_subset = jnp.sum(jnp.abs(delta_recon_at_indices))
    alpha_for_subset = alpha

    return flat_image, flat_error_image, ell1_for_subset, alpha_for_subset
