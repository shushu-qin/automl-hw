import unittest

import matplotlib
import numpy as np

from src.main import gp_prediction_a, gp_prediction_b, phi_n_of_x

matplotlib.use('Agg')


class TestAlgoEquality(unittest.TestCase):

    def test_prediction_equality(self):
        """
        Test to check whether the two implementations of gps produce the same output.
        """
        num_features = 2
        train_data = np.array([(np.array(x), np.sin(x) + 0.2 * np.cos(13 * x)) for x in np.linspace(0, 2 * np.pi, 64)])
        X = np.array([np.array(x) for x in np.linspace(0, 2 * np.pi, 16, endpoint=False)])
        x, y = (zip(*train_data))
        mean1, var1 = gp_prediction_a(train_data, X, phi_n_of_x(num_features), 1.0, np.eye(num_features))
        mean2, var2 = gp_prediction_b(train_data, X, phi_n_of_x(num_features), 1.0, np.eye(num_features))
        self.assertTrue(np.allclose(mean1, mean2), np.allclose(var1, var2))


if __name__ == '__main__':
    unittest.main()
