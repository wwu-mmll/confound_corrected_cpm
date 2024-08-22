import unittest

import numpy as np
import pandas as pd
import pingouin as pg

from scipy.stats import pearsonr, spearmanr, t

from cpm.simulate_data import simulate_data
from cpm.edge_selection import (pearson_correlation_with_pvalues, spearman_correlation_with_pvalues,
                                semi_partial_correlation_pearson, semi_partial_correlation_spearman)


class TestEdgeStatistics(unittest.TestCase):
    def setUp(self):
        super(TestEdgeStatistics, self).setUp()
        self.X, self.y, self.covariates = simulate_data(n_samples=100, n_features=45)

    def test_cpm_pearson(self):
        """Test CPM implementation of Pearson correlation with p-values"""
        cpm_r, cpm_p = pearson_correlation_with_pvalues(self.y, self.X)
        scipy_r = list()
        scipy_p = list()
        for feature in range(self.X.shape[1]):
            c, p = pearsonr(self.X[:, feature], self.y)
            scipy_r.append(c)
            scipy_p.append(p)
        scipy_r = np.array(scipy_r)
        scipy_p = np.array(scipy_p)
        np.testing.assert_almost_equal(scipy_r, cpm_r, decimal=10)
        np.testing.assert_almost_equal(scipy_p, cpm_p, decimal=10)

    def test_cpm_spearman(self):
        """Test CPM implementation of Spearman correlation with p-values"""
        cpm_r, cpm_p = spearman_correlation_with_pvalues(self.y, self.X)
        scipy_r = list()
        scipy_p = list()
        for feature in range(self.X.shape[1]):
            c, p = spearmanr(self.X[:, feature], self.y)
            scipy_r.append(c)
            scipy_p.append(p)
        scipy_r = np.array(scipy_r)
        scipy_p = np.array(scipy_p)
        np.testing.assert_almost_equal(scipy_r, cpm_r, decimal=10)
        np.testing.assert_almost_equal(scipy_p, cpm_p, decimal=10)

    def test_semi_partial_correlation_pearson(self):
        # Calculate partial correlation using the provided function
        partial_corr, p_values = semi_partial_correlation_pearson(self.y, self.X, self.covariates)

        # Calculate partial correlation using pingouin
        df = pd.DataFrame(np.column_stack([self.y, self.X, self.covariates]),
                          columns=["y"] + [f"x{i}" for i in range(self.X.shape[1])] + [f"cov{i}" for i in range(self.covariates.shape[1])])
        pcorr_pingouin = []
        pval_pingouin = []
        for i in range(self.X.shape[1]):
            result = pg.partial_corr(data=df, x="y", y=f"x{i}", covar=[f"cov{j}" for j in range(self.covariates.shape[1])], method='pearson')
            pcorr_pingouin.append(result['r'].values[0])
            pval_pingouin.append(result['p-val'].values[0])

        # Convert to numpy arrays for easier comparison
        pcorr_pingouin = np.array(pcorr_pingouin)
        pval_pingouin = np.array(pval_pingouin)

        # Assert that the partial correlation results are almost equal between the two methods
        np.testing.assert_almost_equal(partial_corr, pcorr_pingouin, decimal=10)

        # Assert that the p-values results are almost equal between the two methods
        np.testing.assert_almost_equal(p_values, pval_pingouin, decimal=10)


if __name__ == '__main__':
    unittest.main()
