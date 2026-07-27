"""Transmission (Poisson) noise for synthetic sinograms, with the fixed-weights
discipline (plan: "Noise on/off variants").

Two rules enforced here:
  1. WEIGHTS come from the NOISELESS sinogram, so a noise-on/noise-off comparison
     holds the weights -- hence the forward-model Hessian and preconditioner -- fixed
     (noise-dependent weights would confound the very subset-step dynamics under
     study).  The weight type matches the real BGA pipeline ('transmission_root').
  2. The noise RNG is an independent np.random.Generator, seeded separately from the
     partition seeds and never touching the global np.random stream (which carries
     the partition/subset-order discipline).
"""

import numpy as np

import mbirjax as mj  # mbirjax must be imported before jax (sets XLA env vars)


def add_transmission_noise(sinogram_clean, i0=1.0e4, noise_seed=7):
    """Poisson transmission noise on a (log-domain) sinogram.

    Photon model: counts ~ Poisson(i0 * exp(-sino)); noisy sino = -log(counts / i0).
    Counts are floored at 1 so the log stays finite (relevant only for line
    integrals near log(i0)).

    Args:
        sinogram_clean: host array, the noiseless log-domain sinogram.
        i0: incident photon count per detector element (sets the noise level).
        noise_seed: seed for the INDEPENDENT noise generator (never the global
            np.random stream).

    Returns:
        (sinogram_noisy, weights): float32 host arrays.  weights are
        gen_weights(sinogram_CLEAN, 'transmission_root') -- identical whether or not
        the returned noisy sinogram is used (fixed-weights discipline).
    """
    sinogram_clean = np.asarray(sinogram_clean)
    rng = np.random.default_rng(noise_seed)
    counts = rng.poisson(i0 * np.exp(-sinogram_clean.astype(np.float64)))
    counts = np.maximum(counts, 1.0)
    sinogram_noisy = np.log(i0 / counts).astype(np.float32)
    weights = np.asarray(mj.gen_weights(sinogram_clean.astype(np.float32),
                                        weight_type='transmission_root'))
    return sinogram_noisy, weights
