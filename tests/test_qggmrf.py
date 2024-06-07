# test_qggmrf.py

import numpy as np
import jax
import jax.numpy as jnp
import mbirjax
import unittest


class TestQGGMRF(unittest.TestCase):
    """
    Test components of the qggmrf prior model
    """

    def setUp(self):
        """Set up before each test method."""
        pass

    def tearDown(self):
        """Clean up after each test method."""
        pass

    def test_b_tilde(self):
        # Make some random qggmrf parameters
        p = np.random.rand(1)[0] + 1
        q = p - 0.9 * np.random.rand(1)[0]
        T = 0.5 + np.random.rand(1)[0]
        sigma_x = 0.1 + np.random.rand(1)[0]
        b = (1, 1, 1, 1, 1, 1)
        delta = 0.1 + np.random.rand(10)
        qggmrf_params = (b, sigma_x, p, q, T)

        # Get the calculated value of b_tilde
        b_tilde = mbirjax.get_2_b_tilde(delta, b[0], qggmrf_params) / 2

        # Compute from scratch
        delta_scale = delta / (T * sigma_x)
        ds_q_minus_p = (delta_scale ** (q - p))
        c1 = (delta ** (p - 2)) / (2 * sigma_x ** p)
        c1 *= ds_q_minus_p * ((q / p) + ds_q_minus_p)
        b_tilde_ref = c1 / (1 + ds_q_minus_p) ** 2

        assert(jnp.allclose(b_tilde, b_tilde_ref))

    def test_gradient_and_hessian(self):
        gradient = (
            np.array([[-0.24734181, -0.06174505, -0.2730214],
                    [0.19395277, -0.12066687, 0.36728308],
                    [0.25866383, 0.1520284, -0.2846858],
                    [0.41440377, -0.37881848, 0.3841581],
                    [-0.5139092, 0.5978173, -0.41555712],
                    [-0.04249194, -0.48688126, 0.38089654],
                    [-0.23896863, 0.22469066, -0.314788],
                    [0.10269843, -0.13829419, 0.27461702],
                    [0.28494033, 0.16065526, -0.27963576]], dtype=np.float32))
        hessian = (
            np.array([[1.53219031e+03, 1.02421265e+03, 1.53115833e+03],
                    [1.02177649e+03, 5.13915710e+02, 1.02134473e+03],
                    [1.53115247e+03, 1.02255969e+03, 1.53082593e+03],
                    [1.02077545e+03, 5.12037476e+02, 1.02113623e+03],
                    [5.11005737e+02, 1.41176522e+00, 5.12126953e+02],
                    [1.02121881e+03, 5.11294250e+02, 1.02108868e+03],
                    [1.53138721e+03, 1.02211853e+03, 1.53054639e+03],
                    [1.02421106e+03, 5.18873352e+02, 1.02465173e+03],
                    [1.53083044e+03, 1.02169562e+03, 1.53087988e+03]], dtype=np.float32))

        np.random.seed(0)
        p = np.random.rand(1)[0] + 1
        q = p - 0.9 * np.random.rand(1)[0]
        T = 0.5 + np.random.rand(1)[0]
        sigma_x = 0.1 + np.random.rand(1)[0]
        b = (1, 1, 1, 1, 1, 1)
        qggmrf_params = (b, sigma_x, p, q, T)

        recon_shape = (3, 3, 3)
        flat_recon = np.random.rand(recon_shape[0] * recon_shape[1], recon_shape[2])
        flat_recon = flat_recon.reshape((recon_shape[0] * recon_shape[1], recon_shape[2]))
        pixel_indices = np.arange(flat_recon.shape[0])

        grad, hess = mbirjax.qggmrf_gradient_and_hessian_at_indices(flat_recon, recon_shape, pixel_indices, qggmrf_params)
        assert(jnp.allclose(grad, gradient))
        assert(jnp.allclose(hess, hessian))


if __name__ == '__main__':
    unittest.main()
