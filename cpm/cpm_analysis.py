import os
from typing import Union

import numpy as np
import pandas as pd

from sklearn.model_selection import BaseCrossValidator, BaseShuffleSplit

from cpm.models import LinearCPMModel, LinearCPMModelv2
from cpm.edge_selection import UnivariateEdgeSelection
from cpm.utils import score_regression, score_classification, score_regression_models, regression_metrics
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

        self.results_outer_cv = None
        self.results_inner_cv = list()

        os.makedirs(results_directory, exist_ok=True)

    def fit(self,
            X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, pd.DataFrame, np.ndarray],
            covariates: Union[pd.Series, pd.DataFrame, np.ndarray]):

        n_outer_folds = self.cv.n_splits
        cv_results = pd.DataFrame({
                'fold': list(np.arange(n_outer_folds)) * 3 * 3,
                'network': (['positive'] * n_outer_folds + ['negative'] * n_outer_folds + ['both'] * n_outer_folds) * 3,
                'model': ['full'] * n_outer_folds * 3 + ['covariates'] * n_outer_folds * 3 + ['connectome'] * n_outer_folds * 3,
                'params': [{}] * n_outer_folds * 3 * 3
                }).set_index(['fold', 'network', 'model'])
        cv_results.sort_index(inplace=True)

        for outer_fold, (train, test) in enumerate(self.cv.split(X, y)):
            print(f"Running fold {outer_fold}")
            fold_dir = os.path.join(self.results_directory, f'fold_{outer_fold}')
            os.makedirs(fold_dir, exist_ok=True)
            X_train, X_test, y_train, y_test, cov_train, cov_test = (X[train], X[test], y[train], y[test],
                                                                     covariates[train], covariates[test])
            n_hps = len(self.edge_selection.param_grid)
            n_inner_folds = self.inner_cv.n_splits
            inner_cv_results = pd.DataFrame({
                'fold': list(np.arange(n_inner_folds)) * n_hps * 3 * 3,
                'param_id': list(np.repeat(np.arange(n_hps), n_inner_folds)) * 3 * 3,
                'network': (['positive'] * n_hps * n_inner_folds + ['negative'] * n_hps * n_inner_folds + ['both'] * n_hps * n_inner_folds) * 3,
                'model': ['full'] * n_hps * n_inner_folds * 3 + ['covariates'] * n_hps * n_inner_folds * 3 + ['connectome'] * n_hps * n_inner_folds * 3,
                'params': list(np.repeat(list(self.edge_selection.param_grid), n_inner_folds * 3 * 3))
                }).set_index(['fold', 'param_id', 'network', 'model'])

            for param_id, param in enumerate(self.edge_selection.param_grid):
                print("   Optimizing hyperparameters using nested CV")
                self.edge_selection.set_params(**param)
                for inner_fold, (nested_train, nested_test) in enumerate(self.inner_cv.split(X_train, y_train)):
                    X_train_nested, X_test_nested, y_train_nested, y_test_nested, cov_train_nested, cov_test_nested = (X[nested_train], X[nested_test], y[nested_train], y[nested_test],
                                                                             covariates[nested_train], covariates[nested_test])
                    pos_edges, neg_edges = self.edge_selection.fit_transform(X=X_train_nested, y=y_train_nested, covariates=cov_train_nested)

                    model = LinearCPMModelv2(positive_edges=pos_edges,
                                             negative_edges=neg_edges).fit(X_train_nested, y_train_nested, cov_train_nested)
                    y_pred = model.predict(X_test_nested, cov_test_nested)
                    metrics = score_regression_models(y_true=y_test_nested, y_pred=y_pred)

                    for model_type in ['full', 'covariates', 'connectome']:
                        for network in ['positive', 'negative', 'both']:
                            inner_cv_results.loc[(inner_fold, param_id, network, model_type), metrics[model_type][network].keys()] = metrics[model_type][network]


            # find best params
            increments = inner_cv_results[regression_metrics].xs(key='full', level='model') - \
                         inner_cv_results[regression_metrics].xs(key='covariates', level='model')
            increments['params'] = inner_cv_results.xs(key='full', level='model')['params']
            increments['model'] = 'increment'
            increments = increments.set_index('model', append=True)
            inner_cv_results = pd.concat([inner_cv_results, increments])
            inner_cv_results.sort_index(inplace=True)
            inner_cv_results.to_csv(os.path.join(fold_dir, 'inner_cv_results.csv'))
            agg_results = inner_cv_results.groupby(['network', 'param_id', 'model'])[regression_metrics].agg(['mean', 'std'])
            agg_results.to_csv(os.path.join(fold_dir, 'inner_cv_results_mean_std.csv'))

            best_params_ids = agg_results['mean_absolute_error'].groupby(['network', 'model'])['mean'].idxmin()
            best_params = inner_cv_results.loc[(0, best_params_ids.loc[('both', 'full')][1], 'both', 'full'), 'params']
            self.edge_selection.set_params(**best_params)

            # build model using best hyperparameters
            pos_edges, neg_edges = self.edge_selection.fit_transform(X=X_train, y=y_train,
                                                                     covariates=cov_train)
            # build linear models using positive and negative edges (training data)
            model = LinearCPMModelv2(positive_edges=pos_edges,
                                     negative_edges=neg_edges).fit(X_train, y_train, cov_train)
            y_pred = model.predict(X_test, cov_test)
            metrics = score_regression_models(y_true=y_test, y_pred=y_pred)

            for model_type in ['full', 'covariates', 'connectome']:
                for network in ['positive', 'negative', 'both']:
                    cv_results.loc[(outer_fold, network, model_type), regression_metrics] = metrics[model_type][network]
                    cv_results.loc[(outer_fold, network, model_type), 'params'] = [best_params]

        increments = cv_results[regression_metrics].xs(key='full', level='model') - \
                     cv_results[regression_metrics].xs(key='covariates', level='model')
        increments['params'] = cv_results.xs(key='full', level='model')['params']
        increments['model'] = 'increment'
        increments = increments.set_index('model', append=True)
        cv_results = pd.concat([cv_results, increments])
        cv_results.sort_index(inplace=True)

        self.results_outer_cv = cv_results
        cv_results.to_csv(os.path.join(self.results_directory, 'cv_results.csv'))
        agg_results = cv_results.groupby(['network', 'model'])[regression_metrics].agg(['mean', 'std'])
        agg_results.to_csv(os.path.join(self.results_directory, 'cv_results_mean_std.csv'))
        return agg_results

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


