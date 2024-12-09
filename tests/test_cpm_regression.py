import unittest

import numpy as np
import pandas as pd
from numpy.random import shuffle
from pandas.core.common import random_state

from cpm.simulate_data import simulate_regression_data_2
from cpm.edge_selection import UnivariateEdgeSelection, PThreshold
from cpm.cpm_analysis import CPMRegression
from cpm.utils import check_data
from sklearn.model_selection import KFold


class TestMissingValues(unittest.TestCase):
    def setUp(self):
        super(TestMissingValues, self).setUp()
        univariate_edge_selection = UnivariateEdgeSelection(edge_statistic=['pearson'],
                                                            edge_selection=[PThreshold(threshold=[0.05],
                                                                                       correction=['fdr_by'])])
        # setup an instance of CPMRegression just to initialize the logger instance
        cpm = CPMRegression(results_directory='./tmp',
                            cv=KFold(n_splits=10, shuffle=True, random_state=42),
                            edge_selection=univariate_edge_selection,
                            add_edge_filter=True,
                            n_permutations=2)
        self.X, self.y, self.covariates = simulate_regression_data_2(n_samples=100, n_features=45)

    def test_nan_in_X(self):
        self.X[0, 0] = np.nan

        with self.assertRaises(ValueError):
            _, _, _ = check_data(self.X, self.y, self.covariates, impute_missings=False)

        _, _, _ = check_data(self.X, self.y, self.covariates, impute_missings=True)

    def test_nan_in_y(self):
        self.y[0] = np.nan

        # raise error if y contains nan and impute_missings is False
        with self.assertRaises(ValueError):
            _, _, _ = check_data(self.X, self.y, self.covariates, impute_missings=False)

        # but also raise an error if y contains nan and impute_missings is True
        # values in y should never be missing
        with self.assertRaises(ValueError):
            _, _, _ = check_data(self.X, self.y, self.covariates, impute_missings=True)

    #def test_nan_in_covariates(self):
    #    self.covariates[0, :] = np.nan

#        with self.assertRaises(ValueError):
 #           _, _, _ = check_data(self.X, self.y, self.covariates, impute_missings=False)


if __name__ == '__main__':
    unittest.main()
