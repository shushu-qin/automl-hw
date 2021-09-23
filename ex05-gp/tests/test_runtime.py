import time
import unittest

import matplotlib
import numpy as np

from src.main import gp_prediction_a, gp_prediction_b, phi_n_of_x

matplotlib.use('Agg')


class TestRuntime(unittest.TestCase):

    def test_runtime_comparison(self):
        """
        Test whether the runtime of the two implementation is roughly what we expect.
        """
        train_data = np.array([(np.array(x), np.sin(x) + 0.1 * np.cos(10 * x)) for x in np.linspace(0, 2 * np.pi, 64)])
        X = np.array([np.array(x) for x in np.linspace(0, 2 * np.pi, 16, endpoint=False)])

        # Warm up the processor ;) The results are not needed, but they allow for a more precise comparison of the two GP implementations.
        num_features = 2
        for i in range(100):
            gp_prediction_a(train_data, X, phi_n_of_x(num_features), 1.0, np.eye(num_features))
            gp_prediction_b(train_data, X, phi_n_of_x(num_features), 1.0, np.eye(num_features))

        tsa = []
        tsb = []
        dimensions = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

        for n in dimensions:
            print('Computing in feature dimension: ', n)
            I_n = np.eye(n)  # setup Identity of dimensionality n
            phi_n = phi_n_of_x(n)  # get basis functions
            start = time.time()
            gp_prediction_a(train_data, X, phi_n, 1.0, I_n)
            run_time = time.time() - start
            tsa.append(run_time)
            start = time.time()
            gp_prediction_b(train_data, X, phi_n, 1.0, I_n)
            run_time = time.time() - start
            tsb.append(run_time)
        self.assertTrue(
            tsb[-1] < tsa[-1], msg='GP algo A should be less efficient in higher dimensional kernel space than algo B.')
        self.assertTrue(
            tsa[0] < tsb[0], msg='GP algo B should be less efficient in low dimensional kernel space than algo A.')


if __name__ == '__main__':
    unittest.main()
