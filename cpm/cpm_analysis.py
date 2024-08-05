import os
import typer
import pickle

from typing import Union
from glob import glob

import numpy as np
import pandas as pd

from sklearn.model_selection import BaseCrossValidator, BaseShuffleSplit, KFold

from cpm.models import LinearCPMModel
from cpm.edge_selection import UnivariateEdgeSelection, PThreshold
from cpm.utils import (score_regression_models, regression_metrics,
                       train_test_split, vector_to_upper_triangular_matrix)
from cpm.fold import compute_inner_folds


class CPMRegression:
    def __init__(self,
                 results_directory: str,
                 cv: Union[BaseCrossValidator, BaseShuffleSplit] = KFold(n_splits=10, shuffle=True, random_state=42),
                 cv_edge_selection: Union[BaseCrossValidator, BaseShuffleSplit] = KFold(n_splits=10, shuffle=True, random_state=42),
                 edge_selection: UnivariateEdgeSelection = UnivariateEdgeSelection(edge_statistic=['pearson'],
                                                                                   edge_selection=[PThreshold(threshold=[0.05],
                                                                                        correction=[None])]),
                 add_edge_filter: bool = True,
                 n_permutations: int = 0):
        self.results_directory = results_directory
        self.cv = cv
        self.inner_cv = cv_edge_selection
        self.edge_selection = edge_selection
        self.add_edge_filter = add_edge_filter
        self.n_permutations = n_permutations

        self.results_outer_cv = None
        self.results_inner_cv = list()

    def save_configuration(self, config_filename: str):
        with open(os.path.splitext(config_filename)[0] + '.pkl', 'wb') as file:
            pickle.dump({'cv': self.cv,
                         'inner_cv': self.inner_cv,
                         'edge_selection': self.edge_selection,
                         'add_edge_filter': self.add_edge_filter,
                         'n_permutations': self.n_permutations}, file)

    def load_configuration(self, results_directory, config_filename):
        with open(config_filename, 'rb') as file:
            loaded_config = pickle.load(file)

        self.results_directory = results_directory
        self.cv = loaded_config['cv']
        self.inner_cv = loaded_config['inner_cv']
        self.edge_selection = loaded_config['edge_selection']
        self.add_edge_filter = loaded_config['add_edge_filter']
        self.n_permutations = loaded_config['n_permutations']

    def estimate(self,
                 X: Union[pd.DataFrame, np.ndarray],
                 y: Union[pd.Series, pd.DataFrame, np.ndarray],
                 covariates: Union[pd.Series, pd.DataFrame, np.ndarray]):

        np.random.seed(42)

        # estimate models
        self._estimate(X=X, y=y, covariates=covariates, perm_run=0)

        # run permutation test
        for perm_id in range(1, self.n_permutations + 1):
            self._estimate(X=X, y=y, covariates=covariates, perm_run=perm_id)

        self.calculate_permutation_results(self.results_directory)

    def _estimate(self,
                  X: Union[pd.DataFrame, np.ndarray],
                  y: Union[pd.Series, pd.DataFrame, np.ndarray],
                  covariates: Union[pd.Series, pd.DataFrame, np.ndarray],
                  perm_run: int = 0):
        np.random.seed(42)
        if perm_run > 0:
            y = np.random.permutation(y)
            current_results_directory = os.path.join(self.results_directory, 'permutation', f'{perm_run}')
        else:
            current_results_directory = self.results_directory
        os.makedirs(current_results_directory, exist_ok=True)
        self._write_info()

        n_outer_folds = self.cv.n_splits
        n_hps = len(self.edge_selection.param_grid)
        n_inner_folds = self.inner_cv.n_splits
        n_features = X.shape[1]
        n_samples = X.shape[0]

        cv_results = self._initialize_outer_cv_results(n_outer_folds=n_outer_folds)
        positive_edges = self._initialize_edges(n_outer_folds=n_outer_folds, n_features=n_features)
        negative_edges = self._initialize_edges(n_outer_folds=n_outer_folds, n_features=n_features)
        #predictions = self._initialize_predictions(n_samples=n_samples, y_true=y)
        predictions = pd.DataFrame()

        for outer_fold, (train, test) in enumerate(self.cv.split(X, y)):
            print(f"Running fold {outer_fold}")
            fold_dir = os.path.join(self.results_directory, f'fold_{outer_fold}')
            if not perm_run:
                os.makedirs(fold_dir, exist_ok=True)

            X_train, X_test, y_train, y_test, cov_train, cov_test = train_test_split(train, test, X, y, covariates)

            inner_cv_results = list()
            print("   Optimizing hyperparameters using nested CV")
            for param_id, param in enumerate(self.edge_selection.param_grid):
                print("       Parameter {}".format(param_id))

                inner_cv_results.append(compute_inner_folds(X_train, y_train, cov_train, self.inner_cv,
                                                            self.edge_selection, param, param_id))
            inner_cv_results = pd.concat(inner_cv_results)

            # model increments
            inner_cv_results = self._calculate_model_increments(cv_results=inner_cv_results,
                                                                metrics=regression_metrics)

            # aggregate over folds to calculate mean and std
            agg_results = inner_cv_results.groupby(['network', 'param_id', 'model'])[regression_metrics].agg(['mean', 'std'])

            if not perm_run:
                inner_cv_results.to_csv(os.path.join(fold_dir, 'inner_cv_results.csv'))
                agg_results.to_csv(os.path.join(fold_dir, 'inner_cv_results_mean_std.csv'))

            # find parameters that perform best
            best_params_ids = agg_results['mean_absolute_error'].groupby(['network', 'model'])['mean'].idxmin()
            best_params = inner_cv_results.loc[(0, best_params_ids.loc[('both', 'full')][1], 'both', 'full'), 'params']

            # use best parameters to estimate performance on outer fold test set
            self.edge_selection.set_params(**best_params)

            # build model using best hyperparameters
            edges = self.edge_selection.fit_transform(X=X_train, y=y_train, covariates=cov_train)
            positive_edges[outer_fold, edges['positive']] = 1
            negative_edges[outer_fold, edges['negative']] = 1

            # build linear models using positive and negative edges (training data)
            model = LinearCPMModel(edges=edges).fit(X_train, y_train, cov_train)
            y_pred = model.predict(X_test, cov_test)

            metrics = score_regression_models(y_true=y_test, y_pred=y_pred)

            for model_type in ['full', 'covariates', 'connectome', 'residuals']:
                for network in ['positive', 'negative', 'both']:
                    n_test_set = y_pred[model_type][network].shape[0]
                    preds = {}
                    preds['y_pred'] = y_pred[model_type][network]
                    preds['y_true'] = y_test
                    preds['model'] = [model_type] * n_test_set
                    preds['network'] = [network] * n_test_set
                    preds['fold'] = [outer_fold] * n_test_set
                    preds['params'] = [best_params] * n_test_set
                    predictions = pd.concat([predictions, pd.DataFrame(preds)], ignore_index=True)

                    cv_results.loc[(outer_fold, network, model_type), regression_metrics] = metrics[model_type][network]
                    cv_results.loc[(outer_fold, network, model_type), 'params'] = [best_params]

        predictions.set_index(['fold', 'network', 'model'], inplace=True)
        predictions.sort_index(inplace=True)

        cv_results = self._calculate_model_increments(cv_results=cv_results, metrics=regression_metrics)

        cv_results.to_csv(os.path.join(current_results_directory, 'cv_results.csv'))

        self.results_outer_cv = cv_results

        agg_results = cv_results.groupby(['network', 'model'])[regression_metrics].agg(['mean', 'std'])
        agg_results.to_csv(os.path.join(current_results_directory, 'cv_results_mean_std.csv'), float_format='%.4f')

        if not perm_run:
            predictions.to_csv(os.path.join(current_results_directory, 'predictions.csv'))

        for sign, edges in [('positive', positive_edges), ('negative', negative_edges)]:
            np.save(os.path.join(current_results_directory, f'{sign}_edges.npy'), vector_to_upper_triangular_matrix(edges[0]))

            weights_edges = np.sum(edges, axis=0) / edges.shape[0]
            overlap_edges = weights_edges == 1

            np.save(os.path.join(current_results_directory, f'weights_{sign}_edges.npy'), vector_to_upper_triangular_matrix(weights_edges))
            np.save(os.path.join(current_results_directory, f'overlap_{sign}_edges.npy'), vector_to_upper_triangular_matrix(overlap_edges))

        return agg_results

    def _write_info(self):
        pass

    @staticmethod
    def _initialize_outer_cv_results(n_outer_folds):
        cv_results = pd.DataFrame({
                'fold': list(np.arange(n_outer_folds)) * 4 * 3,
                'network': (['positive'] * n_outer_folds + ['negative'] * n_outer_folds + ['both'] * n_outer_folds) * 4,
                'model': ['full'] * n_outer_folds * 3 + ['covariates'] * n_outer_folds * 3 + ['connectome'] * n_outer_folds * 3 + ['residuals'] * n_outer_folds * 3,
                'params': [{}] * n_outer_folds * 4 * 3
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

    @staticmethod
    def calculate_permutation_results(results_directory):
        true_results = CPMRegression._load_cv_results(results_directory)

        perm_dir = os.path.join(results_directory, 'permutation')
        valid_perms = glob(os.path.join(perm_dir, '*'))
        perm_results = list()
        for perm_run_folder in valid_perms:
            try:
                perm_res = CPMRegression._load_cv_results(perm_run_folder)
                perm_res['permutation'] = os.path.basename(perm_run_folder)
                perm_res = perm_res.set_index('permutation', append=True)
                perm_results.append(perm_res)
            except FileNotFoundError:
                print(f'No permutation results found for {perm_run_folder}')
        concatenated_df = pd.concat(perm_results)
        p_values = CPMRegression.calculate_p_values(true_results, concatenated_df)
        p_values.to_csv(os.path.join(results_directory, 'p_values.csv'))
        return

    @staticmethod
    def _load_cv_results(folder):
        results = pd.read_csv(os.path.join(folder, 'cv_results_mean_std.csv'), header=[0, 1], index_col=[0, 1])
        results = results.loc[:, results.columns.get_level_values(1) == 'mean']
        results.columns = results.columns.droplevel(1)
        return results

    @staticmethod
    def _calculate_p_value(true_results, perms):
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
    def _calculate_group_p_value(true_group, perms_group):
        result_dict = {}

        # Iterate over each column (metric) in true_group
        for column in true_group.columns:
            condition_count = 0
            if column.endswith('error'):
                condition_count = (true_group[column].values[0] > perms_group[column]).sum()
            elif column.endswith('score'):
                condition_count = (true_group[column].values[0] < perms_group[column]).sum()

            result_dict[column] = condition_count / (len(perms_group[column]) + 1)

        return pd.Series(result_dict)

    @staticmethod
    def calculate_p_values(true_results, perms):
        # Group by 'network' and 'model'
        grouped_true = true_results.groupby(['network', 'model'])
        grouped_perms = perms.groupby(['network', 'model'])

        p_values = []

        for (name, true_group), (_, perms_group) in zip(grouped_true, grouped_perms):
            p_value_series = CPMRegression._calculate_group_p_value(true_group, perms_group)
            p_values.append(pd.DataFrame(p_value_series).T.assign(network=name[0], model=name[1]))

        # Concatenate all the p-values DataFrames into a single DataFrame
        p_values_df = pd.concat(p_values).reset_index(drop=True)
        p_values_df = p_values_df.set_index(['network', 'model'])

        return p_values_df


def main(results_directory: str = typer.Option(...,
                                               exists=True,
                                               file_okay=False,
                                               dir_okay=True,
                                               writable=True,
                                               readable=True,
                                               resolve_path=True,
                                               help="Define results folder for analysis"),
         data_directory: str = typer.Option(
             ...,
             exists=True,
             file_okay=False,
             dir_okay=True,
             writable=True,
             readable=True,
             resolve_path=True,
             help="Path to input data containing targets.csv and data.csv."),
         config_file: str = typer.Option(..., help="Absolute path to config file"),
         perm_run: int = typer.Option(default=0, help="Current permutation run.")):

    X = np.load(os.path.join(data_directory, 'X.npy'))
    y = np.load(os.path.join(data_directory, 'y.npy'))
    covariates = np.load(os.path.join(data_directory, 'covariates.npy'))

    cpm = CPMRegression(results_directory=results_directory)
    cpm.load_configuration(results_directory=results_directory, config_filename=config_file)
    cpm._estimate(X=X, y=y, covariates=covariates, perm_run=perm_run)


if __name__ == "__main__":
    typer.run(main)
