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
        return results_positive.mean(), results_negative.mean()

    def permutation_test(self,
                         n_perms: int):
        pass

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


