"""Simple 4D MACE demo using MBIRJAX cone-beam prox_map and qGGMRF prior denoisers."""

from __future__ import annotations

import time
import jax.numpy as jnp
import numpy as np
import mbirjax as mj
import os
import mbirjax.preprocess as mjp

def _normalize_prior_weights(prior_weight):
    if isinstance(prior_weight, (list, tuple, np.ndarray)):
        prior = [float(w) for w in prior_weight]
        return [1.0 - sum(prior)] + prior
    w = float(prior_weight)
    return [1.0 - w, w / 3.0, w / 3.0, w / 3.0]


def _denoiser_wrapper(x, permute_vector, denoiser=None, sigma_noise=None):
    x_perm = np.transpose(x, permute_vector)
    if denoiser is None:
        y_perm = np.asarray(_qggmrf_hyperplane_denoise(x_perm, sigma_noise=sigma_noise))
    else:
        y_perm = np.asarray(denoiser(x_perm))
    y = np.transpose(y_perm, np.argsort(permute_vector))
    return y


def _qggmrf_hyperplane_denoise(x, sigma_noise=None, sigma_noise_floor=1e-6):
    x = np.asarray(x)
    denoiser = mj.QGGMRFDenoiser(x.shape[1:])
    y = np.empty_like(x)
    for i in range(x.shape[0]):
        if sigma_noise is None:
            sigma_use = float(denoiser.estimate_image_noise_std(x[i]))
        else:
            sigma_use = float(sigma_noise)
        if (not np.isfinite(sigma_use)) or (sigma_use <= sigma_noise_floor):
            y[i] = x[i]
        else:
            y_i, _ = denoiser.denoise(image=x[i], sigma_noise=sigma_use)
            y[i] = np.asarray(y_i)
    return y


def _run_mace_with_models(
    models,
    sino_list,
    weights_list,
    denoiser,
    prior_sigma_noise,
    beta,
    max_admm_itr=10,
    rho=0.5,
    forward_num_iterations=3,
    stop_threshold=0.02,
    init_image=None,
    sigma_p=None,
    verbose=1,
):
    nt = len(sino_list)
    if verbose:
        print(f"[MACE] Start 4D reconstruction with {nt} time bins.")

    if init_image is None:
        if verbose:
            print("[MACE] Computing initial MBIR recon for each time bin...")
        t0 = time.time()
        init_image = np.asarray(
            [np.asarray(models[t].recon(jnp.asarray(sino_list[t]),
                                        weights=jnp.asarray(weights_list[t]),
                                        max_iterations=20,
                                        stop_threshold_change_pct=stop_threshold)[0]) for t in range(nt)])
        if verbose:
            print(f"[MACE] Initialization done in {time.time() - t0:.2f} sec.")
    else:
        init_image = np.asarray(init_image)
        if verbose:
            print("[MACE] Using provided init_image.")

    W = [np.copy(init_image) for _ in range(4)]
    X = [np.copy(init_image) for _ in range(4)]

    for itr in range(max_admm_itr):
        if verbose:
            itr_t0 = time.time()
            print(f"[MACE] Iteration {itr + 1}/{max_admm_itr}")

        if verbose:
            print("[MACE]  Forward agent: cone-beam prox_map for all time bins...")
        X[0] = np.asarray(
            [np.asarray(
                models[t].prox_map(
                prox_input=jnp.asarray(W[0][t]),
                sinogram=jnp.asarray(sino_list[t]),
                sigma_prox=sigma_p,
                weights=jnp.asarray(weights_list[t]),
                init_recon=jnp.asarray(X[0][t]),
                max_iterations=forward_num_iterations,
                stop_threshold_change_pct=stop_threshold)[0]) for t in range(nt)] )

        if verbose:
            print("[MACE]  Prior agent XY-t (fixed z slabs)...")
        X[1] = _denoiser_wrapper(W[1], permute_vector=(3, 0, 1, 2), denoiser=denoiser, sigma_noise=prior_sigma_noise)
        if verbose:
            print("[MACE]  Prior agent YZ-t (fixed row slabs)...")
        X[2] = _denoiser_wrapper(W[2], permute_vector=(1, 0, 2, 3), denoiser=denoiser, sigma_noise=prior_sigma_noise)
        if verbose:
            print("[MACE]  Prior agent XZ-t (fixed col slabs)...")
        X[3] = _denoiser_wrapper(W[3], permute_vector=(2, 0, 1, 3), denoiser=denoiser, sigma_noise=prior_sigma_noise)

        if verbose:
            print("[MACE]  Consensus / ADMM update...")
        z = sum(beta[k] * (2.0 * X[k] - W[k]) for k in range(4))
        for k in range(4):
            W[k] = W[k] + 2.0 * rho * (z - X[k])

        if verbose:
            print(f"[MACE] Iteration {itr + 1} done in {time.time() - itr_t0:.2f} sec.")

    if verbose:
        print("[MACE] Reconstruction complete.")
    return sum(beta[k] * X[k] for k in range(4))


def mace4d_from_cone_beam_params(
    sino_list,
    cone_beam_params_list,
    optional_params_list,
    weight_type="transmission_root",
    prior_weight=0.5,
    max_admm_itr=10,
    rho=0.5,
    forward_num_iterations=3,
    stop_threshold=0.02,
    init_image=None,
    sigma_p=None,
    sharpness=1.0,
    denoiser=None,
    prior_sigma_noise=None,
    verbose=1,
):
    if verbose:
        print("[MACE] Building weights and per-bin cone-beam models...")

    weights_list = [np.asarray(mj.gen_weights(jnp.asarray(s), weight_type=weight_type)) for s in sino_list]

    models = []
    for cone_t, opt_t in zip(cone_beam_params_list, optional_params_list):
        ct_model = mj.ConeBeamModel(**cone_t)
        ct_model.set_params(**opt_t)

        ct_model.set_params(
            positivity_flag=True,
            sharpness=sharpness,
            verbose=verbose,
        )
        models.append(ct_model)
    if verbose:
        print(f"[MACE] Built {len(models)} cone-beam models.")

    recon_4d = _run_mace_with_models(
        models=models,
        sino_list=sino_list,
        weights_list=weights_list,
        denoiser=denoiser,
        prior_sigma_noise=prior_sigma_noise,
        beta=_normalize_prior_weights(prior_weight),
        max_admm_itr=max_admm_itr,
        rho=rho,
        forward_num_iterations=forward_num_iterations,
        stop_threshold=stop_threshold,
        init_image=init_image,
        sigma_p=sigma_p,
        verbose=verbose,
    )
    meta = {"cone_params": cone_beam_params_list, "optional_params": optional_params_list, "weight_type": weight_type}
    return recon_4d, meta

if __name__ == "__main__":

    output_path = "./output"
    os.makedirs(output_path, exist_ok=True)

    USE_SAVED_SINO = True
    USE_SAVED_INIT_IMAGE = True

    if USE_SAVED_SINO :
        sino_folder = ''
        param_folder = ''

        # ---------- load sinos ----------
        sinos = sorted(
            [f for f in os.listdir(sino_folder) if f.startswith("sino_timebin")],
            key=lambda x: int(x.split("timebin")[1].split(".")[0]))

        sino_list = [np.load(os.path.join(sino_folder, f)) for f in sinos]

        # ---------- load params ----------
        params = sorted(
            [f for f in os.listdir(param_folder) if f.startswith("params_timebin")],
            key=lambda x: int(x.split("timebin")[1].split(".")[0]))

        cone_beam_params_list = []
        optional_params_list = []

        for f in params:
            data = np.load(os.path.join(param_folder, f), allow_pickle=True)

            cone_t = data["cone_beam_params"].item()
            opt_t = data["optional_params"].item()

            cone_beam_params_list.append(cone_t)
            optional_params_list.append(opt_t)

    else:

        dataset_url = '/depot/bouman/data/Lilly/4DCT/Phantom_30s_Run1_Dec2024.tgz'
        tag = '4D_Phantom_30s_Run1_Dec2024'

        download_dir = "/home/li5273/PycharmProjects/lilly_exp/nsi/demo_data/"
        dataset_dir = mj.download_and_extract(dataset_url, download_dir)

        # Preprocessing parameters
        downsample_rate = [1, 1]
        subsample_view_factor = 1

        # Recon parameters
        sharpness = 1.0
        verbose = 1

        # 4D split parameters
        views_per_bin = 48
        stride = 24

        print("\n************** NSI dataset preprocessing **************")
        sino, cone_beam_params, optional_params = mjp.nsi.compute_sino_and_params(
            dataset_dir,
            downsample_factor=downsample_rate,
            subsample_view_factor=subsample_view_factor,
        )
        # mj.slice_viewer(sino, slice_axis=0)

        print("\n************** Split into time bins **************")
        # Select a range of the time stamps for faster execution
        start = 50
        end = 60
        time_range = slice(start, end)

        bins = mjp.truncate_sino_into_time_bins(
            sino=sino,
            cone_beam_params=cone_beam_params,
            optional_params=optional_params,
            views_per_bin=views_per_bin,
            stride=stride)[time_range]

        print(f"Total bins: {len(bins)}")

        sino_list = []
        cone_beam_params_list = []
        optional_params_list = []

        print("\n***************** Recon each bin ****************")
        for t, (sino_t, cone_t, opt_t, sl) in enumerate(bins):
            sino_list.append(sino_t)
            cone_beam_params_list.append(cone_t)
            optional_params_list.append(opt_t)


    if USE_SAVED_INIT_IMAGE:
        init_image_folder = ""

        # Get only MBIR recon files and sort by timebin index
        mbir_files = sorted(
            [f for f in os.listdir(init_image_folder) if f.startswith("mbir_timebin") and f.endswith(".npy")],
            key=lambda x: int(x.split("timebin")[1].split(".")[0])
        )

        # Load and stack
        init_image_list = [np.load(os.path.join(init_image_folder, f)) for f in mbir_files]
        init_image = np.stack(init_image_list, axis=0)
    else:
        init_image = None

    # Call MACE 4D
    recon_4d, meta = mace4d_from_cone_beam_params(
            sino_list,
            cone_beam_params_list,
            optional_params_list,
            init_image=init_image,
            weight_type="transmission_root",
            prior_weight=0.5,
            max_admm_itr=10,
            rho=0.5,
            forward_num_iterations=3,
            stop_threshold=0.02,
            sigma_p=None,
            sharpness=1.0,
            denoiser=None,
            prior_sigma_noise=None,
            verbose=1,
    )

    np.save(os.path.join(output_path, "recon_4d.npy"), recon_4d)