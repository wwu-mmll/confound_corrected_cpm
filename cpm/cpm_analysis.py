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
        n_hps = len(self.edge_selection.param_grid)
        n_inner_folds = self.inner_cv.n_splits
        n_features = X.shape[1]
        n_samples = X.shape[0]

        cv_results = self._initialize_outer_cv_results(n_outer_folds=n_outer_folds)
        positive_edges = self._initialize_edges(n_outer_folds=n_outer_folds, n_features=n_features)
        negative_edges = self._initialize_edges(n_outer_folds=n_outer_folds, n_features=n_features)
        predictions = self._initialize_predictions(n_samples=n_samples, y_true=y)

        for outer_fold, (train, test) in enumerate(self.cv.split(X, y)):
            print(f"Running fold {outer_fold}")
            fold_dir = os.path.join(self.results_directory, f'fold_{outer_fold}')
            os.makedirs(fold_dir, exist_ok=True)

            X_train, X_test, y_train, y_test, cov_train, cov_test = self._train_test_split(train, test, X, y, covariates)

            inner_cv_results = self._initialize_inner_cv_results(n_inner_folds=n_inner_folds,
                                                                 n_hyperparameters=n_hps,
                                                                 param_grid=self.edge_selection.param_grid)

            for param_id, param in enumerate(self.edge_selection.param_grid):
                print("   Optimizing hyperparameters using nested CV")
                self.edge_selection.set_params(**param)
                for inner_fold, (nested_train, nested_test) in enumerate(self.inner_cv.split(X_train, y_train)):
                    (X_train_nested, X_test_nested, y_train_nested,
                     y_test_nested, cov_train_nested, cov_test_nested) = self._train_test_split(nested_train, nested_test,
                                                                                                X_train, y_train, cov_train)

                    pos_edges, neg_edges = self.edge_selection.fit_transform(X=X_train_nested, y=y_train_nested, covariates=cov_train_nested)

                    model = LinearCPMModelv2(positive_edges=pos_edges,
                                             negative_edges=neg_edges).fit(X_train_nested, y_train_nested, cov_train_nested)
                    y_pred = model.predict(X_test_nested, cov_test_nested)
                    metrics = score_regression_models(y_true=y_test_nested, y_pred=y_pred)

                    for model_type in ['full', 'covariates', 'connectome']:
                        for network in ['positive', 'negative', 'both']:
                            inner_cv_results.loc[(inner_fold, param_id, network, model_type), metrics[model_type][network].keys()] = metrics[model_type][network]

            # find best params
            inner_cv_results = self._calculate_model_increments(cv_results=inner_cv_results,
                                                                metrics=regression_metrics)
            inner_cv_results.to_csv(os.path.join(fold_dir, 'inner_cv_results.csv'))

            agg_results = inner_cv_results.groupby(['network', 'param_id', 'model'])[regression_metrics].agg(['mean', 'std'])
            agg_results.to_csv(os.path.join(fold_dir, 'inner_cv_results_mean_std.csv'))

            best_params_ids = agg_results['mean_absolute_error'].groupby(['network', 'model'])['mean'].idxmin()
            best_params = inner_cv_results.loc[(0, best_params_ids.loc[('both', 'full')][1], 'both', 'full'), 'params']
            self.edge_selection.set_params(**best_params)

            # build model using best hyperparameters
            pos_edges, neg_edges = self.edge_selection.fit_transform(X=X_train, y=y_train,
                                                                     covariates=cov_train)
            positive_edges[outer_fold, pos_edges] = 1
            negative_edges[outer_fold, neg_edges] = 1

            # build linear models using positive and negative edges (training data)
            model = LinearCPMModelv2(positive_edges=pos_edges,
                                     negative_edges=neg_edges).fit(X_train, y_train, cov_train)
            y_pred = model.predict(X_test, cov_test)

            metrics = score_regression_models(y_true=y_test, y_pred=y_pred)

            for model_type in ['full', 'covariates', 'connectome']:
                for network in ['positive', 'negative', 'both']:
                    predictions.loc[test, f'y_pred_{model_type}_{network}'] = y_pred[model_type][network]
                    predictions.loc[test, 'fold_id'] = outer_fold

                    cv_results.loc[(outer_fold, network, model_type), regression_metrics] = metrics[model_type][network]
                    cv_results.loc[(outer_fold, network, model_type), 'params'] = [best_params]

        cv_results = self._calculate_model_increments(cv_results=cv_results, metrics=regression_metrics)
        cv_results.to_csv(os.path.join(self.results_directory, 'cv_results.csv'))

        self.results_outer_cv = cv_results

        agg_results = cv_results.groupby(['network', 'model'])[regression_metrics].agg(['mean', 'std'])
        agg_results.to_csv(os.path.join(self.results_directory, 'cv_results_mean_std.csv'), float_format='%.4f')

        predictions.to_csv(os.path.join(self.results_directory, 'predictions.csv'))

        np.save(os.path.join(self.results_directory, 'positive_edges.npy'), positive_edges)
        np.save(os.path.join(self.results_directory, 'negative_edges.npy'), negative_edges)

        weights_positive_edges = np.sum(positive_edges, axis=0) / positive_edges.shape[0]
        weights_negative_edges = np.sum(negative_edges, axis=0) / negative_edges.shape[0]

        overlap_positive_edges = weights_positive_edges == 1
        overlap_negative_edges = weights_negative_edges == 1
        np.save(os.path.join(self.results_directory, 'weights_positive_edges.npy'), weights_positive_edges)
        np.save(os.path.join(self.results_directory, 'weights_negative_edges.npy'), weights_negative_edges)
        np.save(os.path.join(self.results_directory, 'overlap_positive_edges.npy'), overlap_positive_edges)
        np.save(os.path.join(self.results_directory, 'overlap_negative_edges.npy'), overlap_negative_edges)

        return agg_results

    @staticmethod
    def _train_test_split(train, test, X, y, covariates):
        return X[train], X[test], y[train], y[test], covariates[train], covariates[test]

    @staticmethod
    def _initialize_outer_cv_results(n_outer_folds):
        cv_results = pd.DataFrame({
                'fold': list(np.arange(n_outer_folds)) * 3 * 3,
                'network': (['positive'] * n_outer_folds + ['negative'] * n_outer_folds + ['both'] * n_outer_folds) * 3,
                'model': ['full'] * n_outer_folds * 3 + ['covariates'] * n_outer_folds * 3 + ['connectome'] * n_outer_folds * 3,
                'params': [{}] * n_outer_folds * 3 * 3
                }).set_index(['fold', 'network', 'model'])
        cv_results.sort_index(inplace=True)
        return cv_results

    @staticmethod
    def _initialize_edges(n_outer_folds, n_features):
        return np.zeros((n_outer_folds, n_features))

    @staticmethod
    def _initialize_predictions(n_samples, y_true):
        predictions = pd.DataFrame({'index': np.arange(n_samples),
                                    'fold_id': np.zeros(n_samples),
                                    'y_true': y_true,
                                    'y_pred_full_positive': np.zeros(n_samples),
                                    'y_pred_covariates_positive': np.zeros(n_samples),
                                    'y_pred_connectome_positive': np.zeros(n_samples),
                                    'y_pred_full_negative': np.zeros(n_samples),
                                    'y_pred_covariates_negative': np.zeros(n_samples),
                                    'y_pred_connectome_negative': np.zeros(n_samples),
                                    'y_pred_full_both': np.zeros(n_samples),
                                    'y_pred_covariates_both': np.zeros(n_samples),
                                    'y_pred_connectome_both': np.zeros(n_samples)
                                    })
        return predictions

    @staticmethod
    def _initialize_inner_cv_results(n_inner_folds, n_hyperparameters, param_grid):
        n_networks = 3
        n_models = 3
        inner_cv_results = pd.DataFrame({
            'fold': list(np.arange(n_inner_folds)) * n_hyperparameters * n_networks * n_models,
            'param_id': list(np.repeat(np.arange(n_hyperparameters), n_inner_folds)) * n_networks * n_models,
            'network': (['positive'] * n_hyperparameters * n_inner_folds +
                        ['negative'] * n_hyperparameters * n_inner_folds +
                        ['both'] * n_hyperparameters * n_inner_folds) * n_models,
            'model': ['full'] * n_hyperparameters * n_inner_folds * n_networks +
                     ['covariates'] * n_hyperparameters * n_inner_folds * n_networks +
                     ['connectome'] * n_hyperparameters * n_inner_folds * n_networks,
            'params': list(np.repeat(list(param_grid), n_inner_folds * n_networks * n_models))
        }).set_index(['fold', 'param_id', 'network', 'model'])
        return inner_cv_results

    @staticmethod
    def _calculate_model_increments(cv_results, metrics):
        increments = cv_results[metrics].xs(key='full', level='model') - \
                     cv_results[metrics].xs(key='covariates', level='model')
        increments['params'] = cv_results.xs(key='full', level='model')['params']
        increments['model'] = 'increment'
        increments = increments.set_index('model', append=True)
        cv_results = pd.concat([cv_results, increments])
        cv_results.sort_index(inplace=True)
        return cv_results

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


