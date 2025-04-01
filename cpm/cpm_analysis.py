import os
import pickle
import logging
import typer
import shutil
from typing import Union
from glob import glob

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, BaseShuffleSplit, KFold
from sklearn.impute import SimpleImputer

from cpm.logging import setup_logging
from cpm.models import LinearCPMModel
from cpm.edge_selection import UnivariateEdgeSelection, PThreshold
from cpm.utils import (
    score_regression_models, regression_metrics,
    train_test_split, vector_to_upper_triangular_matrix, check_data
)
from cpm.fold import compute_inner_folds
from cpm.models import NetworkDict, ModelDict


class CPMRegression:
    """
    This class handles the process of performing CPM Regression with cross-validation and permutation testing.
    """
    def __init__(self,
                 results_directory: str,
                 cv: Union[BaseCrossValidator, BaseShuffleSplit] = KFold(n_splits=10, shuffle=True, random_state=42),
                 inner_cv: Union[BaseCrossValidator, BaseShuffleSplit] = None,
                 edge_selection: UnivariateEdgeSelection = UnivariateEdgeSelection(
                     edge_statistic=['pearson'],
                     edge_selection=[PThreshold(threshold=[0.05], correction=[None])]
                 ),
                 select_stable_edges: bool = True,
                 stability_threshold: float = 0.8,
                 impute_missing_values: bool = True,
                 n_permutations: int = 0,
                 atlas_labels: str = None):
        """
        Initialize the CPMRegression object.

        Parameters
        ----------
        results_directory: str
            Directory to save results.
        cv: Union[BaseCrossValidator, BaseShuffleSplit]
            Outer cross-validation strategy.
        inner_cv: Union[BaseCrossValidator, BaseShuffleSplit]
            Inner cross-validation strategy for edge selection.
        edge_selection:  UnivariateEdgeSelection
            Method for edge selection.
        impute_missing_values: bool
            Whether to impute missing values.
        n_permutations: int
            Number of permutations to run for permutation testing.
        atlas_labels: str
            CSV file containing atlas and regions labels.
        """
        self.results_directory = results_directory
        self.cv = cv
        self.inner_cv = inner_cv
        self.edge_selection = edge_selection
        self.select_stable_edges = select_stable_edges
        self.stability_threshold = stability_threshold
        self.impute_missing_values = impute_missing_values
        self.n_permutations = n_permutations
        self.atlas_labels = atlas_labels

        np.random.seed(42)
        os.makedirs(self.results_directory, exist_ok=True)
        self._copy_atlas_labels()
        self._setup_logging(os.path.join(self.results_directory, "cpm_log.txt"))
        self.logger = logging.getLogger(__name__)

        # Log important configuration details
        self._log_analysis_details()

    def _setup_logging(self, log_level: str):
        setup_logging(log_level)

    def _log_analysis_details(self):
        """
        Log important information about the analysis in a structured format.
        """
        self.logger.info("Starting CPM Regression Analysis")
        self.logger.info("="*50)
        self.logger.info(f"Results Directory:       {self.results_directory}")
        self.logger.info(f"Outer CV strategy:       {self.cv}")
        self.logger.info(f"Inner CV strategy:       {self.inner_cv}")
        self.logger.info(f"Edge selection method:   {self.edge_selection}")
        self.logger.info(f"Select stable edges:     {'Yes' if self.select_stable_edges else 'No'}")
        self.logger.info(f"Stability threshold:     {self.stability_threshold}")
        self.logger.info(f"Impute Missing Values:   {'Yes' if self.impute_missing_values else 'No'}")
        self.logger.info(f"Number of Permutations:  {self.n_permutations}")
        self.logger.info("="*50)

    def _copy_atlas_labels(self):
        # ToDo: add checks for atlas file
        if self.atlas_labels is not None:
            shutil.copy(self.atlas_labels, os.path.join(self.results_directory, 'atlas_labels.csv'))

    def estimate(self,
                 X: Union[pd.DataFrame, np.ndarray],
                 y: Union[pd.Series, pd.DataFrame, np.ndarray],
                 covariates: Union[pd.Series, pd.DataFrame, np.ndarray]):
        """
        Estimates a model using the provided data and conducts permutation testing. This method first fits the model to the actual data and subsequently performs estimation on permuted data for a specified number of permutations. Finally, it calculates permutation results.

        Parameters
        ----------
        X: Feature data used for the model. Can be a pandas DataFrame or a NumPy array.
        y: Target variable used in the estimation process. Can be a pandas Series, DataFrame, or a NumPy array.
        covariates: Additional covariate data to include in the model. Can be a pandas Series, DataFrame, or a NumPy array.

        """
        self.logger.info(f"Starting estimation with {self.n_permutations} permutations.")

        # check data and convert to numpy
        X, y, covariates = check_data(X, y, covariates, impute_missings=self.impute_missing_values)

        # check missing data
        # ToDo

        # Estimate models on actual data
        self._estimate(X=X, y=y, covariates=covariates, perm_run=0)
        self.logger.info("=" * 50)

        # Estimate models on permuted data
        for perm_id in range(1, self.n_permutations + 1):
            self._estimate(X=X, y=y, covariates=covariates, perm_run=perm_id)

        self._calculate_permutation_results()
        self.logger.info("Estimation completed.")

    def _estimate(self,
                  X: Union[pd.DataFrame, np.ndarray],
                  y: Union[pd.Series, pd.DataFrame, np.ndarray],
                  covariates: Union[pd.Series, pd.DataFrame, np.ndarray],
                  perm_run: int = 0):
        """
        Perform an estimation run (either real or permuted data).

        :param X: Features (predictors).
        :param y: Labels (target variable).
        :param covariates: Covariates to control for.
        :param perm_run: Permutation run identifier.
        """
        if perm_run > 0:
            self.logger.debug(f"Permutation run {perm_run}")
            y = np.random.permutation(y)

        current_results_directory = self._get_results_directory(perm_run=perm_run)
        cv_edges = self._initialize_edges(n_outer_folds=self.cv.get_n_splits(), n_features=X.shape[1])
        cv_results = pd.DataFrame()
        cv_predictions = pd.DataFrame()

        for outer_fold, (train, test) in enumerate(self.cv.split(X, y)):
            if not perm_run:
                self.logger.debug(f"Running fold {outer_fold + 1}/{self.cv.get_n_splits()}")

            X_train, X_test, y_train, y_test, cov_train, cov_test = train_test_split(train, test, X, y, covariates)
            if self.impute_missing_values:
                # Initialize imputers with chosen strategy (e.g., mean, median, most_frequent)
                x_imputer = SimpleImputer(strategy='mean')
                cov_imputer = SimpleImputer(strategy='mean')

                # Fit on training data and transform both training and test data
                X_train = x_imputer.fit_transform(X_train)
                X_test = x_imputer.transform(X_test)
                cov_train = cov_imputer.fit_transform(cov_train)
                cov_test = cov_imputer.transform(cov_test)

            if self.inner_cv:
                best_params, stability_edges = self._run_inner_folds(X_train, y_train, cov_train, outer_fold, perm_run)
                if not perm_run:
                    self.logger.info(f"Best hyperparameters: {best_params}")
            else:
                if len(self.edge_selection.param_grid) > 1:
                    raise RuntimeError("Multiple hyperparameter configurations but no inner cv defined. "
                                       "Please provide only one hyperparameter configuration or an inner cv.")
                if self.select_stable_edges:
                    raise RuntimeError("Stable edges can only be selected when using an inner cv.")
                best_params = self.edge_selection.param_grid[0]

            # Use best parameters to estimate performance on outer fold test set
            if self.select_stable_edges:
                edges = {'positive': np.where(stability_edges['positive'] > self.stability_threshold)[0],
                         'negative': np.where(stability_edges['negative'] > self.stability_threshold)[0]}
            else:
                self.edge_selection.set_params(**best_params)
                self.edge_selection.fit(X=X_train, y=y_train, covariates=cov_train)
                edges = self.edge_selection.return_selected_edges()
                
            cv_edges['positive'][outer_fold, edges['positive']] = 1
            cv_edges['negative'][outer_fold, edges['negative']] = 1

            # Build model and make predictions
            model = LinearCPMModel(edges=edges)
            model.fit(X_train, y_train, cov_train)
            y_pred = model.predict(X_test, cov_test)
            metrics = score_regression_models(y_true=y_test, y_pred=y_pred)

            y_pred = self._update_predictions(y_pred, y_test, best_params, outer_fold)
            cv_predictions = pd.concat([cv_predictions, y_pred], axis=0)

            metrics = self._update_metrics(metrics, best_params, outer_fold)
            cv_results = pd.concat([cv_results, metrics], axis=0)

        cv_results, agg_results = self._calculate_final_cv_results(cv_results, current_results_directory)
        self._calculate_edge_stability(cv_edges, current_results_directory)

        if not perm_run:
            self.logger.info(agg_results.round(4).to_string())
            self._save_predictions(cv_predictions, current_results_directory)

    def _run_inner_folds(self, X, y, covariates, fold, perm_run):
        """
        Run inner folds to find the best hyperparameters.

        :param X: Training features.
        :param y: Training labels.
        :param covariates: Training covariates.
        :param fold: Current fold number.
        :param perm_run: Permutation run identifier.
        :return: Best hyperparameters found in inner folds.
        """
        fold_dir = os.path.join(self.results_directory, "folds", f'{fold}')
        os.makedirs(fold_dir, exist_ok=True)

        # 1. split in folds
        # 2. for train set, calculate one sample t test
        # 3. calculate test statistics r and p
        # 4. use one set of edge selection parameters and select edges
        # 5. calculate linear model
        # 6. calculate predictions and metrics

        n_folds = self.inner_cv.get_n_splits()
        n_features = X.shape[1]
        n_params = len(self.edge_selection.param_grid)
        cv_results = pd.DataFrame()
        cv_edges = {'positive': np.zeros((n_folds, n_features, n_params)),
                    'negative': np.zeros((n_folds, n_features, n_params))}

        for fold_id, (nested_train, nested_test) in enumerate(self.inner_cv.split(X, y)):
            (X_train, X_test, y_train,
             y_test, cov_train, cov_test) = train_test_split(train=nested_train, test=nested_test,
                                                             X=X, y=y, covariates=covariates)

            # calculate test statistics for all edges (r, p)
            self.edge_selection.fit(X_train, y_train, cov_train)

            # now loop through all edge selection configurations (e.g. different p-levels)
            for param_id, param in enumerate(self.edge_selection.param_grid):
                self.edge_selection.set_params(**param)
                selected_edges = self.edge_selection.return_selected_edges()
                model = LinearCPMModel(edges=selected_edges).fit(X_train, y_train, cov_train)
                y_pred = model.predict(X_test, cov_test)
                metrics = score_regression_models(y_true=y_test, y_pred=y_pred, primary_metric_only=True)
                results_fold = self._collect_results(fold_id=fold_id, param_id=param_id, param=param, metrics=metrics)

                cv_results = pd.concat([cv_results, pd.DataFrame(results_fold)], ignore_index=True)

                cv_edges['positive'][fold_id, selected_edges['positive'], param_id] = 1
                cv_edges['negative'][fold_id, selected_edges['negative'], param_id] = 1

        cv_results.set_index(['fold', 'param_id', 'network', 'model'], inplace=True)
        cv_results.sort_index(inplace=True)

        stability_edges = {'positive': np.sum(cv_edges['positive'], axis=0) / cv_edges['positive'].shape[0],
                           'negative': np.sum(cv_edges['negative'], axis=0) / cv_edges['negative'].shape[0]}

        # no changes required from this point onwards
        # calculate model increments
        #cv_results = self._calculate_model_increments(cv_results=cv_results, metrics=regression_metrics)
        cv_results = self._calculate_model_increments(cv_results=cv_results, metrics=['spearman_score'])

        # aggregate metrics across folds so that we can find the best model later
        #agg_results = cv_results.groupby(['network', 'param_id', 'model'])[regression_metrics].agg(
        #    ['mean', 'std'])
        agg_results = cv_results.groupby(['network', 'param_id', 'model'])[['spearman_score']].agg(
            ['mean', 'std'])
        # save inner cv results to csv in case this is not a permutation run
        if not perm_run:
            cv_results.to_csv(os.path.join(fold_dir, 'inner_cv_results.csv'))
            agg_results.to_csv(os.path.join(fold_dir, 'inner_cv_results_mean_std.csv'))

        # find the best hyperparameter configuration (best edge selection)
        best_params_ids = agg_results['spearman_score'].groupby(['network', 'model'])['mean'].idxmax()
        best_params = cv_results.loc[(0, best_params_ids.loc[('both', 'connectome')][1], 'both', 'connectome'), 'params']
        stable_edges_best_param = {'positive': stability_edges['positive'][:, best_params_ids.loc[('both', 'connectome')][1]],
                        'negative': stability_edges['negative'][:, best_params_ids.loc[('both', 'connectome')][1]]}
        return best_params, stable_edges_best_param

    @staticmethod
    def _collect_results(fold_id, param_id, param, metrics):
        df = pd.DataFrame()
        for model_type in ModelDict().keys():
            for network in NetworkDict().keys():
                results_dict = metrics[model_type][network]
                results_dict['model'] = model_type
                results_dict['network'] = network
                results_dict['fold'] = fold_id
                results_dict['param_id'] = param_id
                results_dict['params'] = [param]
                df = pd.concat([df, pd.DataFrame(results_dict, index=[0])], ignore_index=True)
        return df

    def _calculate_final_cv_results(self, cv_results: pd.DataFrame, results_directory: str):
        """
        Calculate mean and standard deviation of cross-validation results and save to CSV.

        :param cv_results: DataFrame with cross-validation results.
        :param results_directory: Directory to save the results.
        :return: Updated cross-validation results DataFrame.
        """
        cv_results.set_index(['fold', 'network', 'model'], inplace=True)
        cv_results = self._calculate_model_increments(cv_results=cv_results, metrics=regression_metrics)
        agg_results = cv_results.groupby(['network', 'model'])[regression_metrics].agg(['mean', 'std'])

        # Save results to CSV
        cv_results.to_csv(os.path.join(results_directory, 'cv_results.csv'))
        agg_results.to_csv(os.path.join(results_directory, 'cv_results_mean_std.csv'), float_format='%.4f')

        return cv_results, agg_results

    def _get_results_directory(self, perm_run: int = 0):
        """
        Determine the directory to save results.

        :param perm_run: Permutation run identifier.
        :return: Results directory path.
        """
        if perm_run > 0:
            perm_directory = os.path.join(self.results_directory, 'permutation', f'{perm_run}')
            if not os.path.exists(perm_directory):
                os.makedirs(perm_directory)
            return perm_directory

        if not os.path.exists(self.results_directory):
            os.makedirs(self.results_directory)
        return self.results_directory

    def _save_predictions(self, predictions: pd.DataFrame, results_directory: str):
        """
        Save predictions to CSV.

        :param predictions: DataFrame containing predictions.
        :param results_directory: Directory to save the predictions.
        """
        predictions.to_csv(os.path.join(results_directory, 'cv_predictions.csv'))
        self.logger.info(f"Predictions saved to {results_directory}/cv_predictions.csv")

    def _calculate_permutation_results(self):
        """
        Calculate and save the permutation test results.

        :param results_directory: Directory where the results are saved.
        """
        true_results = CPMRegression._load_cv_results(self.results_directory)

        perm_dir = os.path.join(self.results_directory, 'permutation')
        valid_perms = glob(os.path.join(perm_dir, '*'))
        perm_results = list()
        stability_positive = list()
        stability_negative = list()
        for perm_run_folder in valid_perms:
            try:
                perm_res = CPMRegression._load_cv_results(perm_run_folder)
                perm_res['permutation'] = os.path.basename(perm_run_folder)
                perm_res = perm_res.set_index('permutation', append=True)
                perm_results.append(perm_res)

                # load edge stability
                stability_positive.append(np.load(os.path.join(perm_run_folder, 'stability_positive_edges.npy')))
                stability_negative.append(np.load(os.path.join(perm_run_folder, 'stability_negative_edges.npy')))

            except FileNotFoundError:
                print(f'No permutation results found for {perm_run_folder}')
        concatenated_df = pd.concat(perm_results)
        concatenated_df.to_csv(os.path.join(self.results_directory, 'permutation_results.csv'))
        p_values = CPMRegression.calculate_p_values(true_results, concatenated_df)
        p_values.to_csv(os.path.join(self.results_directory, 'p_values.csv'))

        # stability
        stability_positive = np.stack(stability_positive)
        stability_negative = np.stack(stability_negative)
        true_stability_positive = np.load(os.path.join(self.results_directory, 'stability_positive_edges.npy'))
        true_stability_negative = np.load(os.path.join(self.results_directory, 'stability_negative_edges.npy'))
        sig_stability_positive = np.sum((stability_positive >= np.expand_dims(true_stability_positive, 0)), axis=0) / (len(valid_perms) + 1)
        sig_stability_negative = np.sum((stability_negative >= np.expand_dims(true_stability_negative, 0)), axis=0) / (len(valid_perms) + 1)
        np.save(os.path.join(self.results_directory, 'sig_stability_positive_edges.npy'), sig_stability_positive)
        np.save(os.path.join(self.results_directory, 'sig_stability_negative_edges.npy'), sig_stability_negative)

        self.logger.debug("Saving significance of edge stability.")
        self.logger.info("Permutation test results")
        self.logger.info(p_values.to_string())
        return

    @staticmethod
    def _load_cv_results(folder):
        """
        Load cross-validation results from a CSV file.

        :param folder: Directory containing the results file.
        :return: DataFrame with the loaded results.
        """
        results = pd.read_csv(os.path.join(folder, 'cv_results_mean_std.csv'), header=[0, 1], index_col=[0, 1])
        results = results.loc[:, results.columns.get_level_values(1) == 'mean']
        results.columns = results.columns.droplevel(1)
        return results

    @staticmethod
    def _initialize_edges(n_outer_folds, n_features):
        """
        Initialize a dictionary to store edges for cross-validation.

        :param n_outer_folds: Number of outer folds.
        :param n_features: Number of features in the data.
        :return: Dictionary to store edges.
        """
        return {'positive': np.zeros((n_outer_folds, n_features)), 'negative': np.zeros((n_outer_folds, n_features))}

    def _update_predictions(self, y_pred, y_true, best_params, fold):
        """
        Update predictions DataFrame with new predictions and parameters.

        :param y_pred: Predicted values.
        :param y_true: True values.
        :param best_params: Best hyperparameters from inner cross-validation.
        :param fold: Current fold number.
        :return: Updated predictions DataFrame.
        """
        preds = (pd.DataFrame.from_dict(y_pred).stack().explode().reset_index().rename(
            {'level_0': 'network', 'level_1': 'model', 0: 'y_pred'}, axis=1).set_index(['network', 'model']))
        n_network_model = ModelDict.n_models() * NetworkDict.n_networks()
        preds['y_true'] = np.tile(y_true, n_network_model)
        preds['params'] = [best_params] * y_true.shape[0] * n_network_model
        preds['fold'] = [fold] * y_true.shape[0] * n_network_model
        return preds

    def _update_metrics(self, metrics, params, fold):
        """
        Update metrics DataFrame with new metrics and parameters.

        :param metrics: Dictionary with computed metrics.
        :param params: Best hyperparameters from inner cross-validation.
        :param fold: Current fold number.
        :return: Updated metrics DataFrame.
        """
        df = pd.DataFrame()
        for model in ModelDict().keys():
            d = pd.DataFrame.from_dict(metrics[model], orient='index')
            d['model'] = [model] * NetworkDict.n_networks()
            d['params'] = [params] * NetworkDict.n_networks()
            d['fold'] = [fold] * NetworkDict.n_networks()
            df = pd.concat([df, d], axis=0)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'network'}, inplace=True)
        return df

    @staticmethod
    def _calculate_model_increments(cv_results, metrics):
        """
        Calculate model increments comparing full model to a baseline.

        :param cv_results: Cross-validation results.
        :param metrics: List of metrics to calculate.
        :return: Cross-validation results with increments.
        """
        increments = cv_results[metrics].xs(key='full', level='model') - cv_results[metrics].xs(key='covariates',
                                                                                                level='model')
        increments['params'] = cv_results.xs(key='full', level='model')['params']
        increments['model'] = 'increment'
        increments = increments.set_index('model', append=True)
        cv_results = pd.concat([cv_results, increments])
        cv_results.sort_index(inplace=True)
        return cv_results

    def _calculate_edge_stability(self, cv_edges, results_directory):
        """
        Calculate and save edge stability and overlap.

        :param cv_edges: Cross-validation edges.
        :param results_directory: Directory to save the results.
        """
        for sign, edges in cv_edges.items():
            np.save(os.path.join(results_directory, f'{sign}_edges.npy'),
                    vector_to_upper_triangular_matrix(edges[0]))

            stability_edges = np.sum(edges, axis=0) / edges.shape[0]

            np.save(os.path.join(results_directory, f'stability_{sign}_edges.npy'),
                    vector_to_upper_triangular_matrix(stability_edges))

    @staticmethod
    def _calculate_group_p_value(true_group, perms_group):
        """
        Calculate p-value for a group of metrics.

        :param true_group: DataFrame with the true results.
        :param perms_group: DataFrame with the permutation results.
        :return: Series with calculated p-values.
        """
        result_dict = {}
        for column in true_group.columns:
            condition_count = 0
            if column.endswith('error'):
                condition_count = (true_group[column].values[0] > perms_group[column].astype(float)).sum()
            elif column.endswith('score'):
                condition_count = (true_group[column].values[0] < perms_group[column].astype(float)).sum()

            result_dict[column] = condition_count / (len(perms_group[column]) + 1)

        return pd.Series(result_dict)

    @staticmethod
    def calculate_p_values(true_results, perms):
        """
        Calculate p-values based on true results and permutation results.

        :param true_results: DataFrame with the true results.
        :param perms: DataFrame with the permutation results.
        :return: DataFrame with the calculated p-values.
        """
        grouped_true = true_results.groupby(['network', 'model'])
        grouped_perms = perms.groupby(['network', 'model'])

        p_values = []
        for (name, true_group), (_, perms_group) in zip(grouped_true, grouped_perms):
            p_value_series = CPMRegression._calculate_group_p_value(true_group, perms_group)
            p_values.append(pd.DataFrame(p_value_series).T.assign(network=name[0], model=name[1]))

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
