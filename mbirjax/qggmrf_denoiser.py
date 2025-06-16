from collections import namedtuple
import io
import datetime
import numpy as np
import jax
import jax.numpy as jnp
import mbirjax as mj
from mbirjax import TomographyModel


class QGGMRFDenoiser(TomographyModel):
    """
    The QGGMRFDenoiser uses the MBIRJAX recon framework to implement a qggmrf proximal map denoiser.
    The primary interface is through :meth:`denoise`.
    """

    def __init__(self, sinogram_shape):

        view_params_name = 'None'
        super().__init__(sinogram_shape, view_params_name=view_params_name)
        self.use_ror_mask = False
        self.sigma_noise = None
        try:
            jax.devices('gpu')
            self.set_params(use_gpu='full')
        except RuntimeError:
            self.set_params(use_gpu='none')

    def denoise(self, image, sigma_noise, use_ror_mask=False, init_image=None, max_iterations=15, stop_threshold_change_pct=0.2,
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
        self.sigma_noise = sigma_noise
        if first_iteration == 0 or self.logger is None:
            self.setup_logger(logfile_path=logfile_path, print_logs=print_logs)
        regularization_params = self.auto_set_regularization_params(image)

        # Generate set of voxel partitions
        image_shape, granularity = self.get_params(['recon_shape', 'granularity'])
        partitions = mj.gen_set_of_pixel_partitions(image_shape, granularity, output_device=self.main_device,
                                                    use_ror_mask=self.use_ror_mask)
        partitions = [jax.device_put(partition, self.main_device) for partition in partitions]

        # Generate sequence of partitions to use
        partition_sequence = self.get_params('partition_sequence')
        partition_sequence = mj.gen_partition_sequence(partition_sequence, max_iterations=max_iterations)
        partition_sequence = partition_sequence[first_iteration:]

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
        vcd_subset_denoiser = self.create_vcd_subset_denoiser()

        import jax.lax as lax

        @jax.jit  # JIT the whole sweep
        def denoise_over_partition(flat_image, flat_error_image, partition, times):
            """Run vcd_subset_denoiser over every subset in `partition`
            using lax.fori_loop to keep the loop on-device and JIT-compatible."""

            # Bundle *all* mutable state into one carry object
            def body_fn(i, carry):
                (flat_image, flat_error_image,
                 ell1_accum, alpha_accum, times) = carry

                subset = partition[i]  # pick i-th subset

                (flat_image, flat_error_image,
                 ell1_subset, alpha_subset,
                 times, _time_names) = vcd_subset_denoiser(
                    flat_image,
                    flat_error_image,
                    subset, subset, times)

                # update running totals
                ell1_accum = ell1_accum + ell1_subset
                alpha_accum = alpha_accum + alpha_subset

                return (flat_image, flat_error_image,
                        ell1_accum, alpha_accum, times)

            # initial carry
            init_carry = (flat_image, flat_error_image,
                          0.0, 0.0, times)

            # run 0 … N-1
            final_carry = lax.fori_loop(0, partition.shape[0], body_fn, init_carry)
            return final_carry  # unpack outside if you like

        self.logger.info('Starting VCD iterations')
        if verbose >= 2:
            output = io.StringIO()
            mj.get_memory_stats(file=output)
            self.logger.debug(output.getvalue())
            self.logger.debug('--------')

        # Do the iterations
        max_iters = partition_sequence.size
        fm_rmse = np.zeros(max_iters)
        pm_loss = np.zeros(max_iters)
        nmae_update = np.zeros(max_iters)
        alpha_values = np.zeros(max_iters)
        num_iters = 0
        for i in range(max_iters):
            partition = partitions[partition_sequence[i]]
            times = []
            flat_image, flat_error_image, ell1_for_partition, alpha, times = \
                denoise_over_partition(flat_image, flat_error_image, partition, times)
            # # Test local loop
            #
            # ell1_for_partition = 0
            # alpha = 0
            #
            # for subset in partition:
            #     flat_image, flat_error_image, ell1_for_subset, alpha_for_subset, times, time_names = vcd_subset_denoiser(
            #         flat_image,
            #         flat_error_image,
            #         subset, subset, times)
            #
            #     ell1_for_partition += ell1_for_subset
            #     alpha += alpha_for_subset
            # # End test local loop
            #
            # # Do an iteration
            # flat_image, flat_error_image, ell1_for_partition, alpha = self.vcd_partition_iterator(
            #     vcd_subset_denoiser,
            #     flat_image,
            #     flat_error_image,
            #     partition)

            # Compute the stats and display as desired
            fm_rmse[i] = self.get_forward_model_loss(flat_error_image, sigma_y)
            nmae_update[i] = ell1_for_partition / jnp.sum(jnp.abs(flat_image))
            es_rmse = jnp.linalg.norm(flat_error_image) / jnp.sqrt(float(flat_error_image.size))
            alpha_values[i] = alpha

            if verbose >= 1:
                iter_output = '\nAfter iteration {} of a max of {}: Pct change={:.4f}, Forward loss={:.4f}'.format(
                    i + first_iteration, max_iters + first_iteration,
                    100 * nmae_update[i],
                    fm_rmse[i])
                if compute_prior_loss:
                    qggmrf_nbr_wts, sigma_x, p, q, T = self.get_params(['qggmrf_nbr_wts', 'sigma_x', 'p', 'q', 'T'])
                    b = mj.get_b_from_nbr_wts(qggmrf_nbr_wts)
                    qggmrf_params = (b, sigma_x, p, q, T)
                    pm_loss[i] = mj.qggmrf_loss(flat_image.reshape(image.shape), qggmrf_params)
                    pm_loss[i] /= flat_image.size
                    # Each loss is scaled by the number of elements, but the optimization uses unscaled values.
                    # To provide an accurate, yet properly scaled total loss, first remove the scaling and add,
                    # then scale by the average number of elements between the two.
                    total_loss = ((fm_rmse[i] * image.size + pm_loss[i] * flat_image.size) /
                                  (0.5 * (image.size + flat_image.size)))
                    iter_output += ', Prior loss={:.4f}, Weighted total loss={:.4f}'.format(pm_loss[i], total_loss)

                self.logger.info(iter_output)
                self.logger.info(f'Relative step size (alpha)={alpha:.2f}, Error sino RMSE={es_rmse:.4f}')
                self.logger.info('Number subsets = {}'.format(partition.shape[0]))
                if verbose >= 2:
                    output = io.StringIO()
                    mj.get_memory_stats(file=output)
                    self.logger.debug(output.getvalue())
                    self.logger.debug('--------')
            num_iters += 1
            if nmae_update[i] < stop_threshold_change_pct / 100:
                self.logger.warning('Change threshold stopping condition reached')
                break

        recon_params = (fm_rmse[0:num_iters], pm_loss[0:num_iters], nmae_update[0:num_iters],
                                                alpha_values[0:num_iters])
        notes = 'Reconstruction completed: {}\n\n'.format(datetime.datetime.now())

        denoiser_dict = self.get_recon_dict(recon_params, notes=notes)
        denoised_image = self.reshape_recon(flat_image)

        return denoised_image, denoiser_dict

    def create_vcd_subset_denoiser(self):
        """
        Create a jit-compiled function to apply the qggmrf proximal map to a subset of pixels.

        Returns:
            (callable) vcd_subset_denoiser(error_image, flat_image, pixel_indices) that updates the image.
        """

        fm_constant = 1.0 / (self.get_params('sigma_y') ** 2.0)
        qggmrf_nbr_wts, sigma_x, p, q, T = self.get_params(['qggmrf_nbr_wts', 'sigma_x', 'p', 'q', 'T'])
        b = mj.get_b_from_nbr_wts(qggmrf_nbr_wts)
        qggmrf_params = tuple((b, sigma_x, p, q, T))
        image_shape = self.get_params('recon_shape')

        def vcd_subset_denoiser(flat_image, flat_error_image, pixel_indices, pixel_indices_worker, times):

            # qGGMRF prior - compute the qggmrf gradient and hessian at each pixel in the index set.
            with jax.default_device(self.main_device):
                if self.worker != self.main_device and len(pixel_indices) < self.transfer_pixel_batch_size:
                    prior_grad, prior_hess = (
                        mj.qggmrf_gradient_and_hessian_at_indices_transfer(flat_image, image_shape, pixel_indices,
                                                                           qggmrf_params, self.main_device,
                                                                           self.worker))
                else:
                    prior_grad, prior_hess = (
                        mj.qggmrf_gradient_and_hessian_at_indices(flat_image, image_shape, pixel_indices,
                                                                  qggmrf_params))

            # Back project to get the gradient - the forward Hessian is all 1s for the qggmrf proximal map
            cur_error_image = flat_error_image[pixel_indices_worker]
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
            flat_error_image = flat_error_image.at[pixel_indices_worker].set(cur_error_image)
            ell1_for_subset = jnp.sum(jnp.abs(delta_recon_at_indices))
            alpha_for_subset = alpha

            time_names = []
            return flat_image, flat_error_image, ell1_for_subset, alpha_for_subset, times, time_names

        return jax.jit(vcd_subset_denoiser)

    def get_magnification(self):
        """
        Return 1 to satisfy the TomographyModel interface.

        Returns:
            (float): magnification
        """
        magnification = 1.0
        return magnification

    def auto_set_sigma_y(self, sinogram, sino_indicator, weights=1):
        sigma_y = self.sigma_noise
        self.set_params(no_warning=True, sigma_y=sigma_y, auto_regularize_flag=True)

    # def auto_set_sigma_x(self, recon_std):
    #     """
    #     Compute the automatic value of ``sigma_x`` for use in MBIR reconstruction with qGGMRF prior.
    #
    #     Args:
    #         recon_std (float): Estimated standard deviation of the reconstruction from _get_estimate_of_recon_std.
    #     """
    #     # Get parameters
    #     sharpness = self.get_params('sharpness')
    #
    #     # Compute sigma_x as a fraction of the typical recon value
    #     # 0.2 is an empirically determined constant
    #     sigma_x = np.float32(0.04 * (2 ** sharpness) * recon_std)
    #     self.set_params(no_warning=True, sigma_x=sigma_x, auto_regularize_flag=True)

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

    def _get_sino_indicator(self, noisy_image):
        """
        Compute a binary mask that indicates the region of noisy_image support.

        Args:
            noisy_image (ndarray): 3D jax array containing noisy_image with shape (num_views, num_det_rows, num_det_channels).

        Returns:
            (ndarray): Weights used in mbircone reconstruction, with the same array shape as ``noisy_image``.
        """
        sigma_noise = self.sigma_noise

        percent_noise_floor = 5.0
        # Form indicator by thresholding noisy_image
        threshold = (0.01 * percent_noise_floor) * np.mean(np.fabs(noisy_image)) + sigma_noise
        threshold = min(threshold, np.amax(noisy_image))
        indicator = np.int8(noisy_image >= threshold)
        return indicator
