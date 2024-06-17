from typing import Union

import numpy as np
import pandas as pd

from sklearn.model_selection import BaseCrossValidator, BaseShuffleSplit

from cpm.models import LinearCPMModel
from cpm.utils import score_regression, score_classification


class CPMAnalysis:
    def __init__(self,
                 results_directory: str,
                 cv: Union[BaseCrossValidator, BaseShuffleSplit],
                 stat_method: str = 'pearson',
                 stat_threshold: float = 0.5):
        self.results_directory = results_directory
        self.cv = cv
        self.stat_method = stat_method
        self.stat_threshold = stat_threshold
        self.res_pos = None
        self.res_neg = None

    def fit(self,
            X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, pd.DataFrame, np.ndarray],
            covariates: Union[pd.Series, pd.DataFrame, np.ndarray]):

        results_positive = pd.DataFrame()
        results_negative = pd.DataFrame()

        for train, test in self.cv.split(X, y):
            X_train, X_test, y_train, y_test, cov_train, cov_test = (X[train], X[test], y[train], y[test],
                                                                     covariates[train], covariates[test])

            # calculate edge statistics (e.g. Pearson correlation, Spearman correlation, partial correlation)
            r, p = self._edge_statistics(X=X_train, y=y_train, covariates=cov_train)

            # select significant edges based on specified threshold
            pos_edges, neg_edges = self._edge_selection(r, p, threshold=self.stat_threshold)

            # build linear models using positive and negative edges (training data)
            pos_model = LinearCPMModel(significant_edges=pos_edges).fit(X_train, y_train, cov_train)
            neg_model = LinearCPMModel(significant_edges=neg_edges).fit(X_train, y_train, cov_train)

            # predict on test set
            y_pred_pos_test = pos_model.predict(X_test, cov_test)
            y_pred_neg_test = neg_model.predict(X_test, cov_test)

            # score metrics (how well do the predictions fit the test data?)
            metrics_positive = score_regression(y_true=y_test, y_pred=y_pred_pos_test)
            metrics_negative = score_regression(y_true=y_test, y_pred=y_pred_neg_test)
            results_positive = pd.concat([results_positive, pd.DataFrame(metrics_positive, index=[0])], ignore_index=True)
            results_negative = pd.concat([results_negative, pd.DataFrame(metrics_negative, index=[0])], ignore_index=True)
        self.res_pos = results_positive.mean()
        self.res_neg = results_negative.mean()
        return self.res_pos, self.res_neg

    def permutation_test(self,
                         X,
                         y,
                         covariates,
                         n_perms: int = 1000,
                         random_state: int = 42,
                         ):
        np.random.seed(random_state)
        perms_pos = list()
        perms_neg = list()
        for i in range(n_perms):
            print(i)
            y_perm = np.random.permutation(y)
            pos, neg = self.fit(X, y_perm, covariates)
            perms_pos.append(pos)
            perms_neg.append(neg)
        perms_pos = pd.DataFrame(perms_pos)
        perms_neg = pd.DataFrame(perms_neg)
        p_pos = self._calculate_p_value(pd.DataFrame(self.res_pos).transpose(), perms_pos)
        p_neg = self._calculate_p_value(pd.DataFrame(self.res_neg).transpose(), perms_neg)
        return p_pos, p_neg

    def _calculate_p_value(self, true_results, perms):
        result_dict = {}

        # Iterate over each column in self.res_pos
        for column in true_results.columns:
            condition_count = 0
            if column.endswith('error'):
                # Count occurrences where the value in self.res_pos is larger than perms_pos values
                condition_count = (true_results[column].values[0] > perms[column]).sum()
            elif column.endswith('score'):
                # Count occurrences where the value in self.res_pos is smaller than perms_pos values
                condition_count = (true_results[column].values[0] < perms[column]).sum()

            # Divide the resulting sum by 1001 and add to the result dictionary
            result_dict[column] = [condition_count / (len(perms.iloc[:, 0]) + 1)]

        # Convert the result dictionary to a dataframe
        return pd.DataFrame(result_dict)

    @staticmethod
    def _edge_statistics(X: Union[pd.DataFrame, np.ndarray],
                         y: Union[pd.Series, pd.DataFrame, np.ndarray],
                         covariates: Union[pd.Series, pd.DataFrame, np.ndarray]):
        r_edges = np.random.randn(X.shape[0])
        p_edges = np.random.randn(X.shape[0])
        return r_edges, p_edges

    @staticmethod
    def _edge_selection(r: np.ndarray,
                        p: np.ndarray,
                        threshold: float):
        pos_edges = np.where((p < threshold) & (r > 0))[0]
        neg_edges = np.where((p < threshold) & (r < 0))[0]
        return pos_edges, neg_edges


