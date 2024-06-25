import numpy as np
import jax
import jax.numpy as jnp
import mbirjax
import unittest

from mbirjax import b_tilde_by_definition, compute_surrogate_and_grad, compute_qggmrf_grad_and_hessian


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

    def test_alpha_derivative(self):
        # Make some random qggmrf parameters
        p = np.random.rand(1)[0] + 1
        q = p - 0.9 * np.random.rand(1)[0]
        T = 0.5 + np.random.rand(1)[0]
        sigma_x = 0.1 + np.random.rand(1)[0]
        b = (1, 1, 1, 1, 1, 1)
        qggmrf_params = (b, sigma_x, p, q, T)

        # Get a random recon, x
        recon_shape = (3, 3, 3)
        recon0 = np.random.rand(*recon_shape)
        flat_shape = (recon_shape[0] * recon_shape[1], recon_shape[2])

        # Then get a perturbation to verify a finite difference approximation
        delta = np.random.rand(*recon_shape)
        pixel_indices = np.arange(flat_shape[0])

        with jax.experimental.enable_x64(True):  # Finite difference requires 64 bit arithmetic
            epsilon = 1e-7
            alpha = 0.6
            recon_alpha = recon0 + alpha * delta
            recon_alpha_eps = recon_alpha + epsilon * delta

            x_prime = recon0
            gradient0, _ = mbirjax.qggmrf_gradient_and_hessian_at_indices(recon0.reshape(flat_shape), recon_shape,
                                                                                 pixel_indices, qggmrf_params)
            _, gradient0 = compute_surrogate_and_grad(recon0, x_prime, qggmrf_params)
            _, gradient_delta = compute_surrogate_and_grad(delta, x_prime, qggmrf_params)
            surrogate_alpha, _ = compute_surrogate_and_grad(recon_alpha, x_prime, qggmrf_params)
            surrogate_alpha_eps, _ = compute_surrogate_and_grad(recon_alpha_eps, x_prime, qggmrf_params)

            # Verify (surrogate(x + (alpha + eps) * delta) - surrogate(x + alpha * delta)) / epsilon =
            #                 grad(x)^T delta + grad(delta)^T delta
            finite_diff = (surrogate_alpha_eps - surrogate_alpha) / epsilon   # Deriv wrt alpha of Q(x + alpha delta; x'=x))
            taylor = jnp.sum(gradient0.flatten() * delta.flatten()) + alpha * jnp.sum(gradient_delta.flatten() * delta.flatten())

            assert (jnp.allclose(finite_diff, taylor))

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

        # Compute b_tilde from Eq. 8.19 and Table 8.1 in FCI
        # b_tilde = b = rho'(delta) / (2 delta)
        b_tilde_ref = b[0] * b_tilde_by_definition(delta, sigma_x, p, q, T)

        assert (jnp.allclose(b_tilde_2 / 2, b_tilde_ref))

    def test_gradient_and_hessian(self):
        # Compare the gradient and hessian against a known baseline
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
        b = normalize_b(b)

        qggmrf_params = (b, sigma_x, p, q, T)

        recon_shape = (3, 3, 3)

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

    def test_loss_and_gradient(self):
        # Compare the loss and gradient using a finite difference approximation on loss and a reference
        # implementation of the gradient.  Also compare the hessian to a reference implementation.
        p = 2.0413
        q = 1.124
        T = 1.46
        sigma_x = 0.789
        b = (1, 1, 1, 1, 1, 1)
        b = normalize_b(b)
        qggmrf_params = (b, sigma_x, p, q, T)

        # Get a random recon, x
        recon_shape = (3, 3, 3)
        recon0 = np.random.rand(*recon_shape)
        flat_recon0 = recon0.reshape((-1, recon_shape[2]))
        pixel_indices = np.arange(flat_recon0.shape[0])
        grad0, hess0 = mbirjax.qggmrf_gradient_and_hessian_at_indices(flat_recon0, recon_shape, pixel_indices,
                                                                             qggmrf_params)

        # Then get a perturbation to verify a finite difference approximation
        delta = np.random.rand(*recon_shape)
        with jax.experimental.enable_x64(True):  # Finite difference requires 64 bit arithmetic
            epsilon = 1e-7
            recon1 = recon0 + epsilon * delta

            loss0 = mbirjax.qggmrf_loss(recon0, qggmrf_params)
            loss1 = mbirjax.qggmrf_loss(recon1, qggmrf_params)

            # Verify (loss(x + eps * delta) - loss(x)) / epsilon = grad(x)^T delta
            finite_diff = (loss1 - loss0) / epsilon
            taylor = jnp.sum(grad0.flatten() * delta.flatten())

        assert (jnp.allclose(finite_diff, taylor, rtol=1e-3))

        grad_direct, hess_direct = compute_qggmrf_grad_and_hessian(recon0, qggmrf_params)
        assert (jnp.allclose(hess_direct, hess0.reshape(recon_shape)))
        assert (jnp.allclose(grad_direct, grad0.reshape(recon_shape)))


def normalize_b(b):
    b_sum = np.sum(np.array(b))
    b = tuple([b_entry / b_sum for b_entry in b])
    return b

if __name__ == '__main__':
    unittest.main()
