import unittest

import matplotlib
import numpy as np
from scipy.optimize import minimize

from src.main import negative_log_likelihood

matplotlib.use('Agg')


class TestGPHPO(unittest.TestCase):

    def test_hpo(self):
        """
        Test that the results of the HPO are correct
        """
        np.random.seed(666)
        train_data = np.array([(np.random.uniform(0, 2 * np.pi), np.sin(x) + 1.5 * np.random.random() * 5) for x in
                               np.linspace(0, 2 * np.pi, 9)])

        # Initial kernel parameters
        l_init, sigma_f_init, sigma_n_init = 0.3, 1.0, 0.5

        res = minimize(negative_log_likelihood(train_data), [l_init, sigma_f_init, sigma_n_init],
                       bounds=((1e-5, None), (1e-5, None), (1e-5, None)), method='L-BFGS-B')
        l_opt, sigma_f_opt, sigma_n_opt = res.x
        self.assertTrue(np.isclose(l_opt, 0.7329739536458941))
        self.assertTrue(np.isclose(sigma_f_opt, 3.477364782409653))
        self.assertTrue(np.isclose(sigma_n_opt, 0.3585607633219344))


if __name__ == '__main__':
    unittest.main()
