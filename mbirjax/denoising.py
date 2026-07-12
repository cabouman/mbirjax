import io
import datetime
import warnings
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

    The forward model is the identity (the residual image plays the role of the error sinogram), so
    the image slice-shards on the recon mesh exactly like a reconstruction.  :meth:`denoise` has two
    paths: on a single device it runs the whole sweep in one JIT (the fast path, no qGGMRF halos --
    a single shard uses the reflected boundary condition); across multiple devices it slice-shards
    the image and runs a Python loop that stages the qGGMRF halos once per pass (host-side, so it
    cannot live in a JIT), mirroring ``TomographyModel.vcd_recon``.
    """

    def __init__(self, image_shape):

        view_params_name = 'None'
        if len(image_shape) != 3:
            raise ValueError('image_shape must be 3-dimensional. Got image_shape={}. To denoise a 2D image, use shape (1, m, n).'.format(image_shape))
        super().__init__(image_shape, view_params_name=view_params_name, sigma_noise=None)
        self.set_params(use_ror_mask=False)
        self.set_params(sharpness=0)  # The default sharpness level is 0 for the denoiser.

        self.set_params(granularity=[16], partition_sequence=[0])  # For qggmrf denoising, we can fix a partition
        # Device selection: the automatic default (construction-time layout) already uses the
        # GPU when one is available and the CPU otherwise; use_gpu is deprecated.

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

    def auto_set_recon_geometry(self, no_compile=False, no_warning=False):
        """Compute the default recon shape to equal the sinogram shape"""

        sinogram_shape = self.get_params('sinogram_shape')
        self.set_params(no_compile=no_compile, no_warning=no_warning, recon_shape=sinogram_shape)

    def auto_set_sigma_y(self, sinogram, sino_indicator, weights=1):
        sigma_y = self.get_params('sigma_noise')
        self.set_params(no_warning=True, sigma_y=sigma_y, auto_regularize_flag=True)

    def _check_lateral_truncation(self, sino_indicator):
        """No-op override: the denoiser's 'sinogram' is an ordinary image, and image content
        reaching the frame edge is normal -- not the lateral FoV truncation the base check
        warns about (see TomographyModel._check_lateral_truncation)."""
        return

    def estimate_image_noise_std(self, image):
        """
        Estimate the standard deviation of the reconstruction from the noisy image.

        Args:
            image (jax array or ndarray): 3D array containing noisy image with shape (num_views, num_det_rows, num_det_channels).
        """

        num_pts_to_use = np.minimum(5_000_000, image.size)
        stride = round((image.size / num_pts_to_use) ** (1 / 3))
        small_image = image[::stride, ::stride, ::stride]

        support_indicator = self._get_sino_indicator(small_image, sigma_noise=0.0)
        sigma_noise = self._get_estimate_of_recon_std(small_image, support_indicator)
        support_indicator = self._get_sino_indicator(small_image, sigma_noise=sigma_noise)
        sigma_noise = self._get_estimate_of_recon_std(small_image, support_indicator)
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

        vals = np.stack([noisy_image[inds[0], inds[1], inds[2]],
                        noisy_image[inds[0]-1, inds[1], inds[2]],
                        noisy_image[inds[0], inds[1]-1, inds[2]],
                        noisy_image[inds[0], inds[1], inds[2]-1]], axis=0)
        std = np.std(vals, axis=0)
        recon_std = np.mean(std)

        return recon_std  # typical_image_value

    def _get_sino_indicator(self, noisy_image, sigma_noise=None, verbose=1):
        """
        Compute a binary mask that indicates the region of noisy_image support.

        Args:
            noisy_image (jax array or ndarray): 3D array containing noisy_image with shape (num_views, num_det_rows, num_det_channels).
            sigma_noise (float, optional): Estimated noise standard deviation in the image.  If None, then this is estimated from the image.
            verbose (int, optional): Unused in the body; present because TomographyModel
                calls this method with verbose= (signature compatibility).

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
                stop_threshold_change_pct=0.2, first_iteration=0, logfile_path='~/.mbirjax/logs/recon.log',
                print_logs=True, output_sharded=False):
        """
        Compute the MAP denoiser assuming AWGN and the 3D qGGMRF prior.

        With default settings, and with X a clean image and W equal to AWGN of standard deviation sigma_noise,
        the result of :meth:`denoise` applied to X+W is the MAP estimate of the denoised image using the
        qGGMRF prior function.

        The amount of denoising can be changed by changing sigma_noise.  If sigma_noise is None, then sigma_noise
        is estimated from a set of samples from the image.

        Denoising strength can also be adjusted using the parameter `sharpness` (default=0.0).

        Args:
            image (numpy or jax array):  The 3D volume to be denoised.
            sigma_noise (float, optional):  The estimated noise variance in the noisy image.  If None, then the noise level is estimated from the image.
            use_ror_mask: Option to restrict denoising to a masked region in the image.
                Defaults to False. Use False for no mask, True for an ellipse inscribed in
                the reconstruction volume, or a custom binary 2D array with shape
                ``recon_shape[:2]``.
            init_image (numpy or jax array, optional):  An initial image for the minimization.  Defaults to image.
            max_iterations (int, optional): maximum number of iterations of the VCD algorithm to perform.
            stop_threshold_change_pct (float, optional): Stop reconstruction when 100 * ||delta_recon||_1 / ||recon||_1 change from one iteration to the next is below stop_threshold_change_pct.  Defaults to 0.2.  Set this to 0 to guarantee exactly max_iterations.
            first_iteration (int, optional): Set this to be the number of iterations previously completed when restarting a recon using init_recon.  This defines the first index in the partition sequence.  Defaults to 0.
            logfile_path (str, optional): Path to the output log file.  Defaults to '~/.mbirjax/logs/recon.log'.
            print_logs (bool, optional): If true then print logs to console.  Defaults to True.
            output_sharded (bool, optional): If False (default), return a numpy array in the
                problem's real shape.  If True, return the internal device form (slice-sharded on a
                multi-device denoiser, with a possibly padded slice axis; on a single device the
                same array either way).  Defaults to False.

        Returns:
            tuple: (denoised_image, denoiser_dict)
                - denoised_image (numpy or jax array): A denoised image of the same shape as image
                - denoiser_dict (dict): A dict obtained from :meth:`~mbirjax.TomographyModel.get_recon_dict` with entries
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
        self.set_params(use_ror_mask=use_ror_mask)
        if sigma_noise is None:
            sigma_noise = self.estimate_image_noise_std(image)
        self.set_params(sigma_noise=sigma_noise)
        self._log_run_header(first_iteration, logfile_path, print_logs)
        self.logger.info('Initializing QGGMRFDenoiser')
        # Disable warning about background estimation
        verbose = self.get_params('verbose')
        self.set_params(verbose=0)
        regularization_params = self.auto_set_regularization_params(image)
        self.set_params(verbose=verbose)

        # Generate set of voxel partitions
        image_shape, granularity = self.get_params(['recon_shape', 'granularity'])
        partition_sequence = self.get_params('partition_sequence')
        partition_index = partition_sequence[0]
        use_ror_mask = self.get_params('use_ror_mask')
        partitions = mj.gen_set_of_pixel_partitions(image_shape, [granularity[partition_index]], use_ror_mask=use_ror_mask)

        # Generate sequence of partitions to use
        partition = partitions[0]

        if init_image is None:
            init_image = image.copy()

        # Recon-domain flat layout (num_pixels, num_slices), sharded on the last axis (slices)
        # immediately: distributing image/init_image across devices first makes the residual
        # subtraction per-shard, so no full-sized array is materialized on one device (a single
        # device is the trivial 1-shard case -- identical values to a plain device_put there).
        image_shape = self.get_params('recon_shape')
        flat_image = self._shard_recon(init_image.reshape((-1, image_shape[2])))
        flat_error_image = self._shard_recon(image.reshape((-1, image.shape[-1]))) - flat_image

        verbose = self.get_params('verbose')
        fm_constant = 1.0 / (self.get_params('sigma_y') ** 2.0)
        qggmrf_nbr_wts, sigma_x, p, q, T = self.get_params(['qggmrf_nbr_wts', 'sigma_x', 'p', 'q', 'T'])
        b = mj.get_b_from_nbr_wts(qggmrf_nbr_wts)
        qggmrf_params = tuple((b, sigma_x, p, q, T))

        max_iters = max_iterations
        stop_thresh = stop_threshold_change_pct / 100.0  # scalar threshold

        self.logger.info('Starting VCD iterations')
        if verbose >= 2:
            output = io.StringIO()
            mj.get_memory_stats(file=output)
            self.logger.debug(output.getvalue())
            self.logger.debug('--------')

        # Two paths (see class docstring): a single device runs the whole sweep on-device in one
        # JIT (the fast path when the image fits on one device -- a single shard needs no qGGMRF
        # halos, so the reflected BC applies as-is); multiple devices slice-shard the flat arrays
        # and run a Python loop that stages the qGGMRF halos once per pass (extract_halos is
        # host-side, so it cannot live in a JIT), mirroring vcd_recon.  Both have the same
        # signature, so dispatch by picking the function.
        denoise_fcn = (self._denoise_single_device if len(self.recon_placement.devices) == 1
                       else self._denoise_sharded)
        flat_image, nmae_update, alpha_values, num_iters = denoise_fcn(
            flat_image, flat_error_image, partition, fm_constant, qggmrf_params, image_shape,
            max_iters, stop_thresh, first_iteration)

        fm_rmse = None
        recon_params = (fm_rmse, nmae_update[0:num_iters], alpha_values[0:num_iters])
        stop_threshold_change_pct = [100 * float(val) for val in recon_params[1]]
        alpha_values = [float(val) for val in recon_params[2]]

        prior_loss = None
        recon_param_values = [int(num_iters), granularity, partition_sequence, fm_rmse, prior_loss,
                              regularization_params, stop_threshold_change_pct, alpha_values]
        recon_params = mj.ReconParams(*tuple(recon_param_values))._asdict()

        notes = 'Reconstruction completed: {}\n\n'.format(datetime.datetime.now())
        denoiser_dict = self.get_recon_dict(recon_params, notes=notes)
        if output_sharded:
            # Keep the internal device form (slice-sharded; the slice axis may be padded).
            denoised_image = flat_image.reshape(self._recon_device_shape())
        else:
            # Default: gather to a numpy real-shape array (a no-op on one device; crops any padded
            # slices on multiple devices), then restore the 3-D image shape.
            denoised_image = self.reshape_recon(self._gather_recon(flat_image))

        return denoised_image, denoiser_dict

    def _log_denoise_progress(self, cur_iter, cur_nmae, first_iteration, max_iters):
        """Log one denoising-iteration progress line.  Called directly in the sharded path and via
        ``jax.debug.callback`` from the single-device JIT, so it accepts numpy/jax scalars."""
        self.logger.info('After iteration {} of a max of {}: Pct change={:.4f}'.format(
            int(cur_iter) + first_iteration, max_iters, 100 * float(cur_nmae)))

    def _denoise_single_device(self, flat_image, flat_error_image, partition, fm_constant,
                               qggmrf_params, image_shape, max_iters, stop_thresh, first_iteration):
        """Run the whole denoising sweep on one device in a single JIT (the fast path when the
        image fits on one device; a single shard needs no qGGMRF halos -- reflected BC applies).

        ``flat_image``/``flat_error_image`` arrive already on the (single) recon device -- a 1-shard
        NamedSharding from ``_shard_recon``, which the whole-sweep JIT and the single-device prior
        handle identically to a plain device_put.  Returns (flat_image, nmae_history,
        alpha_history, num_iters), the histories padded to ``max_iters`` (the exit slices them).
        """
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
        nmae_update_init = jnp.zeros(max_iters)
        alpha_values_init = jnp.zeros(max_iters)

        @jax.jit
        def run_denoising_loop(local_flat_image, local_flat_error_image):
            """
            Runs the outer optimisation loop with lax.while_loop.
            Returns:
                (local_flat_image, local_flat_error_image,
                nmae_hist, alpha_hist, num_iters)
            """

            # -------- carry = all mutable state --------
            # i                 : iteration counter
            # local_flat_image        : current image
            # local_flat_error_image  : current residual
            # nmae_hist         : history array (filled progressively)
            # alpha_hist        : "
            # nmae_curr         : nmae from *most recent* iteration
            carry0 = (0, local_flat_image, local_flat_error_image,
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
                (i, flat_img, flat_err_img, nmae_body, alpha_body, _) = carry

                # inner loop (already JAX-ified with fori_loop previously)
                flat_img, flat_err_img, ell1_for_part, alpha_val = \
                    denoise_over_partition(flat_img, flat_err_img)

                # --- compute stats ---
                nmae = ell1_for_part / jnp.sum(jnp.abs(flat_img))

                # --- write into the history arrays ---
                nmae_body = nmae_body.at[i].set(nmae)
                alpha_body = alpha_body.at[i].set(alpha_val)

                # Insert call back to print progress
                print_rate = 5
                _ = lax.cond(
                    (i % print_rate) == 0,
                    lambda c: (
                        # call host_callback then return unchanged carry
                        jax.debug.callback(self._log_denoise_progress, i, nmae_body[i],
                                           first_iteration, max_iters)
                    ),
                    lambda c: c,
                    None
                )

                # bump iteration counter
                return i + 1, flat_img, flat_err_img, nmae_body, alpha_body, nmae  # new nmae for cond_fn

            # run the while-loop
            final_carry = lax.while_loop(cond_fn, body_fn, carry0)

            # unpack results
            (num_iters_loop, local_flat_image, local_flat_error_image,
             nmae_hist, alpha_hist, _) = final_carry

            return local_flat_image, local_flat_error_image, nmae_hist, alpha_hist, num_iters_loop

        flat_image, _flat_error_image, nmae_update, alpha_values, num_iters = (
            run_denoising_loop(flat_image, flat_error_image))
        return flat_image, nmae_update, alpha_values, num_iters

    def _denoise_sharded(self, flat_image, flat_error_image, partition, fm_constant,
                         qggmrf_params, image_shape, max_iters, stop_thresh, first_iteration):
        """Run the denoising sweep across devices on the (already slice-sharded) flat arrays.

        Mirrors vcd_recon's sharded path: a Python outer loop stages the qGGMRF halos once per
        pass (host-side), and an eager per-subset updater computes the halo-aware prior and the
        identity-forward line search.  Because the forward model is the identity there is only the
        recon mesh -- every reduction (prior and line-search scalars) lands there, so ``alpha``
        needs no cross-mesh reconciliation.  Returns (flat_image, nmae_history, alpha_history,
        num_iters), the histories at actual length (the exit slices to num_iters).
        """
        eps = jnp.finfo(jnp.float32).eps
        max_alpha = 1.5

        def sharded_subset(cur_image, cur_error_full, subset, staged_halos):
            """One VCD update over a subset of pixels: combine the qGGMRF prior with the identity
            forward model, take the optimal step, and update the image and residual in place."""
            # Replicate the subset indices across the recon mesh so the pixel-axis gather/scatter
            # is local to each slice-shard (a single device is the trivial 1-shard case).
            recon_indices = jax.device_put(
                subset, jax.sharding.NamedSharding(self.recon_placement.mesh,
                                                   jax.sharding.PartitionSpec()))
            # qGGMRF prior gradient/Hessian at this subset's pixels, with halo exchange for the
            # inter-slice term; pass the plain subset, like vcd_recon.
            prior_grad, prior_hess = self._qggmrf_prior_sharded(
                cur_image, subset, qggmrf_params, staged_halos=staged_halos)
            # Forward-model gradient/Hessian.  The forward model is the identity, so the gradient is
            # just -fm_constant * residual and the Hessian is all 1s (no projection).
            cur_error = cur_error_full[recon_indices]
            forward_grad = - fm_constant * cur_error
            forward_hess = 1
            # Newton update direction in the recon domain (combined forward + prior).
            delta = - ((forward_grad + prior_grad) / (forward_hess + prior_hess))
            # Prior terms of the line search: delta^T grad and an upper bound on delta^T H delta.
            prior_linear = jnp.sum(prior_grad * delta)
            prior_quadratic_approx = jnp.sum(prior_hess * delta ** 2)
            # Forward terms of the line search.  Identity forward model: the "sinogram" delta is
            # delta itself.  All four scalars reduce over the recon mesh, so alpha is computed there
            # directly (no cross-mesh reconciliation).
            forward_linear = fm_constant * jnp.tensordot(cur_error, delta, axes=2)
            forward_quadratic = fm_constant * jnp.tensordot(delta, delta, axes=2)
            # Optimal step size for this subset, clamped to (eps, max_alpha) for stability.
            alpha = (forward_linear - prior_linear) / (forward_quadratic + prior_quadratic_approx + eps)
            alpha = jnp.clip(alpha, eps, max_alpha)
            # Apply the step with update_recon (a donated, in-place scatter-add, so XLA reuses the
            # slice-sharded buffers): add the step to the image, and add its negative to the
            # residual (for the subset's unique indices, set(cur_error - alpha*delta) is exactly
            # adding -alpha*delta).
            scaled_delta = alpha * delta
            cur_image = mj.update_recon(cur_image, recon_indices, scaled_delta)
            cur_error_full = mj.update_recon(cur_error_full, recon_indices, -scaled_delta)
            # L1 norm of this subset's update, accumulated into the NMAE stop metric.
            ell1 = jnp.sum(jnp.abs(scaled_delta))
            return cur_image, cur_error_full, ell1, alpha

        num_subsets = int(partition.shape[0])
        print_rate = 5
        nmae_hist, alpha_hist = [], []
        num_iters = 0
        # Outer optimization loop: each iteration is one pass over all subsets of the partition,
        # stopping at max_iters or once the per-iteration change (NMAE) falls below the threshold.
        for it in range(max_iters):
            # Stage the qGGMRF boundary halos ONCE for this pass (host-side) and reuse across subsets.
            staged_halos = self._stage_halos(flat_image)
            # Sweep every subset, accumulating this pass's update L1 and step sizes.
            ell1_accum, alpha_accum = 0.0, 0.0
            for s in range(num_subsets):
                flat_image, flat_error_image, ell1_sub, alpha_sub = sharded_subset(
                    flat_image, flat_error_image, partition[s], staged_halos)
                ell1_accum = ell1_accum + ell1_sub
                alpha_accum = alpha_accum + alpha_sub
            # Per-iteration stats: NMAE = ||update||_1 / ||image||_1 (the stop metric) and mean step.
            nmae = float(ell1_accum / jnp.sum(jnp.abs(flat_image)))
            nmae_hist.append(nmae)
            alpha_hist.append(float(alpha_accum / num_subsets))
            num_iters += 1
            if (it % print_rate) == 0:
                self._log_denoise_progress(it, nmae, first_iteration, max_iters)
            if nmae < stop_thresh:   # converged: change this pass is below the stop threshold
                break

        return flat_image, jnp.asarray(nmae_hist), jnp.asarray(alpha_hist), num_iters


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
    prior_overrelaxation_factor = 1.0
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


def median_filter3d(x, max_block_gb=4.0, return_min_max=False) -> Union[jnp.ndarray | tuple]:
    """
    Apply a 27‑point (3x3x3) median filter to a 3‑D JAX array using replicated
    (edge) boundary conditions.  Optionally return the min and max in each 27 point neighborhood.

    The volume is processed in d0‑blocks so that the kernel can be
    `jax.jit`‑compiled while limiting peak device memory. Each block is padded with
    a one‑voxel halo; halos duplicate the nearest edge voxel so that the result
    matches NumPy’s `"edge"` mode.

    Args:
        x (jax array or ndarray): Input array. Any numeric dtype supported by JAX is allowed.
        max_block_gb (float. optional): A rough upper bound on the amount of memory in GB to use for the filtering.  Defaults to 4.0.
        return_min_max (bool, optional): If true, the output is a tuple of median, min, max.

    Returns:

        jax.numpy.ndarray or tuple: An array (or tuple of 3 arrays) of the same shape and dtype as *x* containing the median‑filtered result.

    Notes
    -----
    * The function automatically splits the 0‑dimension into blocks so that at
      most roughly ``max_block_gb`` of temporary data are materialised.
      If the array is large and the 0 dimension is small relative to another dimension, it may be more memory efficient
      to apply jnp.swapaxes(x, 0, long_dim) before applying median_filter3d, although swapaxes will make a copy of x.
    * Within each block the filter is computed by rolling the data in all 26
      neighbour directions, stacking the 27 volumes, and taking
      ``jnp.median`` along the new axis.
    * This is a whole-volume operation and is **not** sharding-aware.  The built-in d0-blocking
      already bounds single-device memory, so a large volume can be filtered on one device.  If ``x``
      is distributed across multiple devices (e.g. a recon returned with ``output_sharded=True``),
      gather it to a single device first; otherwise JAX partitions the sharded-axis stencil with
      cross-device communication (a warning is issued in that case).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import mbirjax as mj
    >>> vol = jnp.arange(27.).reshape(3, 3, 3)
    >>> mj.median_filter3d(vol)
    Array([[[3., 3., 4.],
            [6., 6., 7.],
            [6., 7., 8.]],
           ...
           dtype=float32)
    """
    # Not sharding-aware: a multi-device input would have its 3x3x3 stencil partitioned by XLA
    # (cross-device communication, possibly slow).  Warn so the user can gather it to one device.
    shards = getattr(x, 'addressable_shards', None)
    if shards is not None and len(shards) > 1:
        warnings.warn(
            'median_filter3d received an array sharded across {} devices; it is a whole-volume '
            'operation, so JAX will partition the 3x3x3 stencil across devices (cross-device '
            'communication).  For predictable performance gather it to one device first, e.g. '
            'np.asarray(x) or jax.device_put(x, jax.devices()[0]).'.format(len(shards)))
    d0, d1, d2 = x.shape
    x_gb = x.size * 4 / (1024**3)
    num_blocks = np.ceil(27 * x_gb / max_block_gb).astype(int)
    block_size = max(d0 // num_blocks, 1)
    # 1) Pad d2 in every dim by 1 for the edge‐replicated halo
    xp = jnp.pad(x, pad_width=1, mode='edge')      # (d0+2, d1+2, d2+2)

    # 2) Pad d0 *further* up to a multiple of block_size
    n_blocks = (d0 + block_size - 1) // block_size
    padded_Z = n_blocks * block_size
    # pad only at the *end* in d0 so that we can dynamic‐slice fixed size blocks
    pad_extra = padded_Z - d0
    xp = jnp.pad(xp,
                 pad_width=((0, pad_extra), (0,0), (0,0)),
                 mode='edge')                        # (padded_Z+2, d1+2, d2+2)

    # fixed slice size for every block
    slice_sz = (block_size + 2,  d1 + 2,  d2 + 2)

    def filter_one_block(i):
        # compute the dynamic start index in the padded xp
        z0 = i * block_size
        start = (z0, 0, 0)
        # grab a fixed‐shape window [block_size+2, d1+2, d2+2]
        block = lax.dynamic_slice(xp, start, slice_sz)

        # replicate the 27‐roll → stack → median recipe on this small block
        patches = [
          jnp.roll(block, shift=(dz,dy,dx), axis=(0,1,2))
          for dz in (-1,0,1) for dy in (-1,0,1) for dx in (-1,0,1)
        ]
        stacked  = jnp.stack(patches, axis=0)      # (27, blkZ+2, d1+2, d2+2)
        filtered = jnp.median(stacked, axis=0)    # (blkZ+2, d1+2, d2+2)
        if return_min_max:
            min_filtered = jnp.min(stacked, axis=0)
            max_filtered = jnp.max(stacked, axis=0)
            filtered = jnp.stack([filtered, min_filtered, max_filtered], axis=3)

        # strip off the 1‐voxel halo in all dims → (blkZ, d1, d2)
        return filtered[1:-1, 1:-1, 1:-1]

    # 3) Map over blocks
    blocks = lax.map(filter_one_block, jnp.arange(n_blocks))  # (n_blocks, blkZ, d1, d2)

    # 4) Stitch & then crop back to original d0
    out = jnp.concatenate(blocks, axis=0)  # (padded_Z, d1, d2)
    out = out[:d0, :, :]                   # (d0, d1, d2)
    if return_min_max:
        out = (out[..., 0], out[..., 1], out[..., 2])
    return out
