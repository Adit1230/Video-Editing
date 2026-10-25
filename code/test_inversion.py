"""Analytical tests for fixed-point inversion, runnable without GPU or PyTorch."""
import unittest
import numpy as np
from inversion import fixed_point_solve


class FixedPointTests(unittest.TestCase):
    def test_solves_inverse_of_deterministic_ddim(self):
        # DDIM forward reverse-step algebra: x_t = A*x_prev + B*eps(x_t).
        alpha_previous, alpha_next = .8, .5
        a = np.sqrt(alpha_next / alpha_previous)
        b = np.sqrt(1-alpha_next) - a*np.sqrt(1-alpha_previous)
        previous = np.array([.7, -.2])
        slope, bias = .2, .1
        expected = (a*previous + b*bias) / (1-b*slope)
        def predict(x, final): return slope*x + bias
        def update(noise, anchor): return a*anchor + b*noise
        one = fixed_point_solve(previous, predict, update, 1)
        refined = fixed_point_solve(previous, predict, update, 12)
        np.testing.assert_allclose(refined, expected, rtol=1e-11, atol=1e-11)
        # Reconstruct the cleaner latent using the noise at the solved x_t.
        eps = predict(refined, True)
        x0 = (refined-np.sqrt(1-alpha_next)*eps) / np.sqrt(alpha_next)
        recovered = np.sqrt(alpha_previous)*x0 + np.sqrt(1-alpha_previous)*eps
        np.testing.assert_allclose(recovered, previous, atol=1e-11)
        self.assertLess(np.max(abs(refined-expected)), np.max(abs(one-expected)))
        np.testing.assert_array_equal(previous, [.7, -.2])

    def test_only_final_evaluation_is_marked_for_cache(self):
        calls, recorded = [], []
        def predict(x, final):
            calls.append(final)
            if final: recorded.append(x.copy())
            return .25*x
        result = fixed_point_solve(np.array([1.]), predict,
                                   lambda noise, anchor: anchor+noise, 4)
        self.assertEqual(calls, [False, False, False, True])
        self.assertEqual(len(recorded), 1)
        np.testing.assert_allclose(recorded[0], [1.328125])
        np.testing.assert_allclose(result, [1.33203125])

    def test_relaxation_and_validation(self):
        result = fixed_point_solve(np.array([1.]), lambda x, final: x,
                                   lambda noise, anchor: 2*anchor, 2, .5)
        np.testing.assert_allclose(result, [1.75])
        for iterations in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                fixed_point_solve(1., lambda x, final: x, lambda n, a: a, iterations)
        for relaxation in [0, -1, 1.1, float('nan'), float('inf'), True]:
            with self.assertRaises(ValueError):
                fixed_point_solve(1., lambda x, final: x, lambda n, a: a, 2, relaxation)


if __name__ == '__main__':
    unittest.main()
