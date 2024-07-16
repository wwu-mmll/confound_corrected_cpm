from typing import Union

import numpy as np
import pandas as pd

from sklearn.model_selection import BaseCrossValidator, BaseShuffleSplit

from cpm.models import LinearCPMModel
from cpm.edge_selection import UnivariateEdgeSelection
from cpm.utils import score_regression, score_classification
from cpm.edge_selection import partial_correlation


class CPMAnalysis:
    def __init__(self,
                 results_directory: str,
                 cv: Union[BaseCrossValidator, BaseShuffleSplit],
                 cv_edge_selection: Union[BaseCrossValidator, BaseShuffleSplit],
                 edge_selection: UnivariateEdgeSelection,
                 estimate_model_increments: bool = True,
                 add_edge_filter: bool = True):
        self.results_directory = results_directory
        self.cv = cv
        self.inner_cv = cv_edge_selection
        self.edge_selection = edge_selection
        self.estimate_model_increments = estimate_model_increments
        self.add_edge_filter = add_edge_filter

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
            n_hps = len(self.edge_selection.param_grid)
            n_inner_folds = self.inner_cv.n_splits
            inner_cv_results = pd.DataFrame({
                'fold': list(np.arange(n_inner_folds)) * n_hps,
                'params': list(self.edge_selection.param_grid) * n_inner_folds,
                'param_id': np.repeat(np.arange(n_hps), n_inner_folds),
                'mean_absolute_error': np.empty(n_hps * n_inner_folds),
                'pearson_score': np.empty(n_hps * n_inner_folds)}).set_index(['fold', 'param_id'])

            for param_id, param in enumerate(self.edge_selection.param_grid):
                self.edge_selection.set_params(**param)
                for inner_fold, (nested_train, nested_test) in enumerate(self.inner_cv.split(X_train, y_train)):
                    X_train_nested, X_test_nested, y_train_nested, y_test_nested, cov_train_nested, cov_test_nested = (X[nested_train], X[nested_test], y[nested_train], y[nested_test],
                                                                             covariates[nested_train], covariates[nested_test])
                    pos_edges, neg_edges = self.edge_selection.fit_transform(X=X_train_nested, y=y_train_nested, covariates=cov_train_nested)
                    # build linear models using positive and negative edges (training data)
                    pos_model = LinearCPMModel(significant_edges=pos_edges).fit(X_train_nested, y_train_nested, cov_train_nested)
                    neg_model = LinearCPMModel(significant_edges=neg_edges).fit(X_train_nested, y_train_nested, cov_train_nested)

                    # predict on test set
                    y_pred_pos_test_nested = pos_model.predict(X_test_nested, cov_test_nested)
                    y_pred_neg_test_nested = neg_model.predict(X_test_nested, cov_test_nested)

                    # score metrics (how well do the predictions fit the test data?)
                    metrics_positive = score_regression(y_true=y_test_nested, y_pred=y_pred_pos_test_nested)
                    metrics_negative = score_regression(y_true=y_test_nested, y_pred=y_pred_neg_test_nested)

                    inner_cv_results.loc[(inner_fold, param_id)] = metrics_positive
                    inner_cv_results.loc[(inner_fold, param_id)] = metrics_negative

            # calculate edge statistics (e.g. Pearson correlation, Spearman correlation, partial correlation)
            #r, p = self._edge_statistics(X=X_train, y=y_train, covariates=cov_train)

            # select significant edges based on specified threshold
            #pos_edges, neg_edges = self._edge_selection(r, p, threshold=self.stat_threshold)

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
        p_edges = partial_correlation(X=X, y=y, covariates=covariates)
        r_edges = np.random.randn(X.shape[1])
        #p_edges = np.random.randn(X.shape[0])
        return r_edges, p_edges

    @staticmethod
    def _edge_selection(r: np.ndarray,
                        p: np.ndarray,
                        threshold: float):
        pos_edges = np.where((p < threshold) & (r > 0))[0]
        neg_edges = np.where((p < threshold) & (r < 0))[0]
        return pos_edges, neg_edges


