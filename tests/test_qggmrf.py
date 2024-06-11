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
        b_tilde_2 = mbirjax.get_2_b_tilde(delta, b[0], qggmrf_params)
        b_tilde = b_tilde_2 / 2

        # Compute b_tilde from Eq. 8.19 and Table 8.1 in FCI
        # b_tilde = b = rho'(delta) / (2 delta)
        delta_scale = abs(delta / (T * sigma_x)) + jnp.finfo(jnp.float32).eps
        ds_q_minus_p = (delta_scale ** (q - p))
        numerator = (delta ** (p - 2)) / (2 * sigma_x ** p)
        numerator *= ds_q_minus_p * ((q / p) + ds_q_minus_p)
        b_tilde_ref = numerator / (1 + ds_q_minus_p) ** 2
        b_tilde_ref *= b[0]

        assert(jnp.allclose(b_tilde, b_tilde_ref))

    def test_gradient_and_hessian(self):
        grad_ref = (
            np.array([[-0.28758478, 0.17966035, 0.06372651],
                      [0.21930355, 0.42724127, -0.33447355],
                      [-0.03735046, -0.3333393, 0.04014717],
                      [0.4248102, -0.4625442, -0.21327719],
                      [-0.54877305, 0.2317356, 0.04053728],
                      [0.19778192, 0.42639467, 0.14354284],
                      [-0.2730683, 0.41404432, -0.17601134],
                      [0.3569668, -0.5147747, 0.40523687],
                      [-0.09706858, -0.00686641, -0.2859974]], dtype=np.float32))
        hess_ref = (
            np.array([[1530.8113, 1021.26227, 1531.6426],
                      [1021.8176, 512.0654, 1021.831],
                      [1531.7743, 1021.9556, 1531.9774],
                      [1020.6803, 512.4941, 1023.63727],
                      [510.69043, 2.4552107, 513.4677],
                      [1021.80176, 511.99084, 1022.6514],
                      [1530.9652, 1020.7792, 1531.6259],
                      [1021.4198, 510.99796, 1020.9037],
                      [1531.5032, 1021.65314, 1530.8649]], dtype=np.float32))

        # The values below were obtained using the commented out code, and the results above were obtained
        # using these values on a verified working version of the code.
        # np.random.seed(0)
        p = 1.5488135039273248  # np.random.rand(1)[0] + 1
        q = 0.9051430741921472  # p - 0.9 * np.random.rand(1)[0]
        T = 1.102763376071644  # 0.5 + np.random.rand(1)[0]
        sigma_x = 0.6448831829968968  # 0.1 + np.random.rand(1)[0]
        b = (1, 1, 1, 1, 1, 1)
        qggmrf_params = (b, sigma_x, p, q, T)

        recon_shape = (3, 3, 3)
        # flat_recon = np.random.rand(recon_shape[0] * recon_shape[1], recon_shape[2])
        # flat_recon = flat_recon.reshape((recon_shape[0] * recon_shape[1], recon_shape[2]))
        flat_recon = (
            np.array([[0.4236548, 0.64589411, 0.43758721],
                      [0.891773, 0.96366276, 0.38344152],
                      [0.79172504, 0.52889492, 0.56804456],
                      [0.92559664, 0.07103606, 0.0871293],
                      [0.0202184, 0.83261985, 0.77815675],
                      [0.87001215, 0.97861834, 0.79915856],
                      [0.46147936, 0.78052918, 0.11827443],
                      [0.63992102, 0.14335329, 0.94466892],
                      [0.52184832, 0.41466194, 0.26455561]])
        )
        pixel_indices = np.arange(flat_recon.shape[0])

        grad0, hess0 = mbirjax.qggmrf_gradient_and_hessian_at_indices(flat_recon, recon_shape, pixel_indices,
                                                                      qggmrf_params)
        assert (jnp.allclose(grad0, grad_ref))
        assert (jnp.allclose(hess0, hess_ref))

    def test_cost_and_gradient(self):
        p = 2
        q = 1.2
        T = 1
        sigma_x = 1
        b = (1, 1, 1, 1, 1, 1)
        qggmrf_params = (b, sigma_x, p, q, T)

        # Get a random recon, x
        recon_shape = (3, 3, 3)
        recon0 = np.random.rand(*recon_shape)
        flat_recon0 = recon0.reshape((-1, recon_shape[2]))
        pixel_indices = np.arange(flat_recon0.shape[0])
        grad0, hess0 = mbirjax.qggmrf_gradient_and_hessian_at_indices(flat_recon0, recon_shape, pixel_indices,
                                                                      qggmrf_params)

        # Then get a perturbation
        delta = np.random.rand(*recon_shape)
        with jax.experimental.enable_x64(True):  # Finite difference requires 64 bit arithmetic
            epsilon = 1e-7
            recon1 = recon0 + epsilon * delta

            cost0 = mbirjax.qggmrf_cost(recon0, qggmrf_params)
            cost1 = mbirjax.qggmrf_cost(recon1, qggmrf_params)

            # Verify (cost(x + eps * delta) - cost(x)) / epsilon = grad(x)^T delta

            finite_diff = (cost1 - cost0) / epsilon
            taylor = jnp.sum(grad0.flatten() * delta.flatten())

        assert(jnp.allclose(finite_diff, taylor, rtol=1e-3))


if __name__ == '__main__':
    unittest.main()
