import unittest

import numpy as np
import pandas as pd
import pingouin as pg

from scipy.stats import pearsonr, spearmanr

from cpm.simulate_data import simulate_regression_data
from cpm.edge_selection import (pearson_correlation_with_pvalues, spearman_correlation_with_pvalues,
                                semi_partial_correlation_pearson, semi_partial_correlation_spearman)


class TestEdgeStatistics(unittest.TestCase):
    def setUp(self):
        super(TestEdgeStatistics, self).setUp()
        self.X, self.y, self.covariates = simulate_regression_data(n_samples=100, n_features=45)

    def _test_correlation(self, method, cpm_func, scipy_func):
        """Generalized test for correlation with p-values"""
        cpm_r, cpm_p = cpm_func(self.y, self.X)
        scipy_r, scipy_p = [], []

        for feature in range(self.X.shape[1]):
            c = scipy_func(self.X[:, feature], self.y)
            scipy_r.append(c.correlation if method == 'pearson' else c.statistic)
            scipy_p.append(c.pvalue)

        np.testing.assert_almost_equal(np.array(scipy_r), cpm_r, decimal=10)
        np.testing.assert_almost_equal(np.array(scipy_p), cpm_p, decimal=10)

    def test_cpm_pearson(self):
        self._test_correlation('pearson', pearson_correlation_with_pvalues, pearsonr)

    def test_cpm_spearman(self):
        self._test_correlation('spearman', spearman_correlation_with_pvalues, spearmanr)

    def _test_semi_partial_correlation(self, method, func):
        # Calculate partial correlation using the provided function
        partial_corr, p_values = func(self.y, self.X, self.covariates)

        # Prepare DataFrame
        df = pd.DataFrame(np.column_stack([self.y, self.X, self.covariates]),
                          columns=["y"] + [f"x{i}" for i in range(self.X.shape[1])] + [f"cov{i}" for i in
                                                                                       range(self.covariates.shape[1])])
        pcorr_pingouin, pval_pingouin = [], []

        for i in range(self.X.shape[1]):
            result = pg.partial_corr(data=df, x="y", y=f"x{i}",
                                     covar=[f"cov{j}" for j in range(self.covariates.shape[1])],
                                     method=method)
            pcorr_pingouin.append(result['r'].values[0])
            pval_pingouin.append(result['p-val'].values[0])

        np.testing.assert_almost_equal(partial_corr, np.array(pcorr_pingouin), decimal=10)
        np.testing.assert_almost_equal(p_values, np.array(pval_pingouin), decimal=10)

    def test_semi_partial_correlation_pearson(self):
        self._test_semi_partial_correlation('pearson', semi_partial_correlation_pearson)

    def test_semi_partial_correlation_spearman(self):
        self._test_semi_partial_correlation('spearman', semi_partial_correlation_spearman)


if __name__ == '__main__':
    unittest.main()
