import unittest
import numpy as np
import pandas as pd
from sklearn.utils import check_X_y

from cpm.utils import check_data, matrix_to_vector_3d
from cpm.simulate_data import simulate_regression_data_scenarios

class TestUtils(unittest.TestCase):
    def setUp(self):
        # Base 2D inputs
        self.X2d, self.y1d, self.cov2d = simulate_regression_data_scenarios(
            n_samples=50, n_features=5
        )
        # Covariate variants
        self.cov_series = pd.Series(np.arange(50))
        self.cov_1d_array = np.arange(50)
        # Two features per sample
        self.cov_2d_array = np.vstack([np.arange(50), np.arange(50)]).T

    def test_matrix_to_vector_3d(self):
        n = 4
        n_samples = 10
        mat = np.random.randn(n_samples, n, n)
        vec = matrix_to_vector_3d(mat)
        expected_dim = n * (n - 1) // 2
        self.assertEqual(vec.shape, (n_samples, expected_dim))

    def test_accepts_3d_X(self):
        n = 6
        n_samples = 20
        X3d = np.random.randn(n_samples, n, n)
        X_out, y_out, cov_out = check_data(X3d, self.y1d[:n_samples], self.cov2d[:n_samples])
        expected_dim = n * (n - 1) // 2
        self.assertEqual(X_out.shape, (n_samples, expected_dim))

    def test_error_on_invalid_X_dim(self):
        X1d = np.arange(10)
        with self.assertRaises(ValueError):
            check_data(X1d, self.y1d[:10], self.cov2d[:10])

    def test_squeeze_y(self):
        y2d = self.y1d.reshape(-1, 1)
        X_out, y_out, cov_out = check_data(self.X2d, y2d, self.cov2d)
        self.assertEqual(y_out.ndim, 1)
        self.assertEqual(y_out.shape, (len(self.y1d),))

    def test_error_on_invalid_y_dim(self):
        y2d_bad = np.random.randn(10, 2)
        with self.assertRaises(ValueError):
            check_data(self.X2d[:10], y2d_bad, self.cov2d[:10])

    def test_covariates_series(self):
        X_out, y_out, cov_out = check_data(self.X2d, self.y1d, self.cov_series)
        self.assertEqual(cov_out.ndim, 2)
        self.assertEqual(cov_out.shape, (50, 1))

    def test_covariates_1d_array(self):
        X_out, y_out, cov_out = check_data(self.X2d, self.y1d, self.cov_1d_array)
        self.assertEqual(cov_out.shape, (50, 1))

    def test_covariates_2d_array(self):
        X_out, y_out, cov_out = check_data(self.X2d, self.y1d, self.cov_2d_array)
        self.assertEqual(cov_out.shape, self.cov_2d_array.shape)

    def test_accepts_dataframe_covariates(self):
        cov_df = pd.DataFrame({
            'cat': np.random.choice(['A', 'B', 'C'], size=50),
            'num': np.random.randn(50)
        })
        X_out, y_out, cov_out = check_data(self.X2d, self.y1d, cov_df)
        # 'cat' → 2 one-hot columns + 'num' = 3 features
        self.assertEqual(cov_out.shape, (50, 3))

    def test_missing_values_behavior(self):
        # NaN in X → error without impute
        X_nan = self.X2d.copy()
        X_nan[0, 0] = np.nan
        with self.assertRaises(ValueError):
            check_data(X_nan, self.y1d, self.cov2d, impute_missings=False)
        # NaN in X allowed with impute
        X_out, _, _ = check_data(X_nan, self.y1d, self.cov2d, impute_missings=True)
        self.assertTrue(np.isnan(X_out[0, 0]))
        # NaN in y → always error
        y_nan = self.y1d.copy()
        y_nan[0] = np.nan
        with self.assertRaises(ValueError):
            check_data(self.X2d, y_nan, self.cov2d, impute_missings=True)

    def test_error_on_invalid_covariate_dim(self):
        cov_3d = np.zeros((50, 2, 2))
        with self.assertRaises(ValueError):
            check_data(self.X2d, self.y1d, cov_3d)


if __name__ == '__main__':
    unittest.main()
