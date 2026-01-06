import os
import pandas as pd

from typing import Union

import numpy as np

from glob import glob

from cccpm.models import NetworkDict, ModelDict
from cccpm.utils import vector_to_upper_triangular_matrix
from cccpm.scoring import regression_metrics


class ResultsManager:
    """
    A class to handle the aggregation, formatting, and saving of results.

    Parameters
    ----------
    output_dir : str
        Directory where results will be saved.
    """
    def __init__(self, output_dir: Union[str, None], perm_run: int, n_folds: int, n_features: int, n_params: int = None,
                 is_inner_cv: bool = False):
        self.perm_run = perm_run
        self.is_inner_cv = is_inner_cv
        self.results_directory = self.update_results_directory(output_dir=output_dir)
        self.n_folds = n_folds
        self.n_features = n_features
        self.n_params = n_params

        self.cv_results = pd.DataFrame()
        self.cv_predictions = pd.DataFrame()
        self.cv_edges = self.initialize_edges(n_folds=self.n_folds, n_features=self.n_features,
                                              n_params=self.n_params)
        self.cv_network_strengths = pd.DataFrame()
        self.agg_results = None

    def update_results_directory(self, output_dir: Union[str, None]):
        """
        Determine the directory to save results.

        :param output_dir:
        :return: Results directory path.
        """
        if not self.is_inner_cv and self.perm_run > 0:
            perm_directory = os.path.join(output_dir, 'permutation', f'{self.perm_run}')
            if not os.path.exists(perm_directory):
                os.makedirs(perm_directory)
            return perm_directory

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        return output_dir

    @staticmethod
    def initialize_edges(n_folds, n_features, n_params=None):
        """
        Initialize a dictionary to store edges for cross-validation.

        :param n_folds: Number of outer folds.
        :param n_features: Number of features in the data.
        :return: Dictionary to store edges.
        """
        if n_params is None:
            return {'positive': np.zeros((n_folds, n_features)), 'negative': np.zeros((n_folds, n_features))}
        else:
            return {'positive': np.zeros((n_folds, n_features, n_params)),
                    'negative': np.zeros((n_folds, n_features, n_params))}

    def store_edges(self, edges: dict, fold: int, param_id: int = None):
        if param_id is None:
            self.cv_edges['positive'][fold, edges['positive']] = 1
            self.cv_edges['negative'][fold, edges['negative']] = 1
        else:
            self.cv_edges['positive'][fold, edges['positive'], param_id] = 1
            self.cv_edges['negative'][fold, edges['negative'], param_id] = 1

    def calculate_edge_stability(self, write: bool = True, best_param_id: int = None):
        """
        Calculate and save edge stability and overlap.

        :param cv_edges: Cross-validation edges.
        :param results_directory: Directory to save the results.
        """
        edge_stability = {}
        for sign, edges in self.cv_edges.items():
            if best_param_id is None:
                edge_stability[sign] = np.sum(edges, axis=0) / edges.shape[0]
            else:
                edge_stability[sign] = np.sum(edges[:, :, best_param_id], axis=0) / edges.shape[0]

            if write:
                np.save(os.path.join(self.results_directory, f'{sign}_edges.npy'),
                        vector_to_upper_triangular_matrix(edges[0]))
                np.save(os.path.join(self.results_directory, f'stability_{sign}_edges.npy'),
                        vector_to_upper_triangular_matrix(edge_stability[sign]))
        return edge_stability

    def calculate_model_increments(self):
        """
        Calculate model increments comparing full model to a baseline.

        :param cv_results: Cross-validation results.
        :param metrics: List of metrics to calculate.
        :return: Cross-validation results with increments.
        """
        increments = self.cv_results[regression_metrics].xs(key='full', level='model') - self.cv_results[regression_metrics].xs(key='covariates',
                                                                                                level='model')
        increments['params'] = self.cv_results.xs(key='full', level='model')['params']
        increments['model'] = 'increment'
        increments = increments.set_index('model', append=True)
        self.cv_results = pd.concat([self.cv_results, increments])
        self.cv_results.sort_index(inplace=True)
        return

    def store_metrics(self, metrics, params, fold, param_id):
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
            d['param_id'] = [param_id] * NetworkDict.n_networks()
            d['fold'] = [fold] * NetworkDict.n_networks()
            df = pd.concat([df, d], axis=0)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'network'}, inplace=True)

        self.cv_results = pd.concat([self.cv_results, df], axis=0)
        return

    def store_predictions(self, y_pred, y_true, params, fold, param_id, test_indices):
        """
        Update predictions DataFrame with new predictions and parameters.

        :param y_pred: Predicted values.
        :param y_true: True values.
        :param params: Best hyperparameters from inner cross-validation.
        :param fold: Current fold number.
        :return: Updated predictions DataFrame.
        """
        #preds = (pd.DataFrame.from_dict(y_pred).stack().explode().reset_index().rename(
        #    {'level_0': 'network', 'level_1': 'model', 0: 'y_pred'}, axis=1).set_index(['network', 'model']))
        preds = (
            pd.DataFrame.from_dict(y_pred)
            .stack()
            .explode()
            .reset_index()
            .rename({'level_0': 'network', 'level_1': 'model', 0: 'y_pred'}, axis=1)
            .set_index(['network', 'model'])
        )
        n_network_model = ModelDict.n_models() * NetworkDict.n_networks()
        preds['y_true'] = np.tile(y_true, n_network_model)
        preds['params'] = [params] * y_true.shape[0] * n_network_model
        preds['fold'] = [fold] * y_true.shape[0] * n_network_model
        preds['param_id'] = [param_id] * y_true.shape[0] * n_network_model
        preds['sample_index'] = np.tile(test_indices, n_network_model)  # include indices
        self.cv_predictions = pd.concat([self.cv_predictions, preds], axis=0)
        return

    def store_network_strengths(self, network_strengths, y_true, fold):
        dfs = list()
        models = ['connectome', 'residuals']
        networks = ['positive', 'negative']
        for model in models:
            for network in networks:
                df = pd.DataFrame()
                df['y_true'] = y_true
                df['network_strength'] = np.squeeze(network_strengths[model][network])
                df['model'] = [model] * network_strengths[model][network].shape[0]
                df['fold'] = [fold] * network_strengths[model][network].shape[0]
                df['network'] = [network] * network_strengths[model][network].shape[0]
                dfs.append(df)

        df = pd.concat(dfs, axis=0)
        self.cv_network_strengths = pd.concat([self.cv_network_strengths, df], axis=0)
        return

    @staticmethod
    def load_cv_results(folder):
        """
        Load cross-validation results from a CSV file.

        :param folder: Directory containing the results file.
        :return: DataFrame with the loaded results.
        """
        results = pd.read_csv(os.path.join(folder, 'cv_results_mean_std.csv'), header=[0, 1], index_col=[0, 1])
        results = results.loc[:, results.columns.get_level_values(1) == 'mean']
        results.columns = results.columns.droplevel(1)
        return results

    def save_predictions(self):  # update save function to sort by index prior to saving
        """
        Save predictions to CSV.
        """
        df = self.cv_predictions.copy()
        df.sort_values(by='sample_index', inplace=True)
        #df.drop(columns='sample_index', inplace=True)
        df.to_csv(os.path.join(self.results_directory, 'cv_predictions.csv'))
        # self.cv_predictions.to_csv(os.path.join(self.results_directory, 'cv_predictions.csv'))

    def save_network_strengths(self):
        """
        Save network strengths to CSV.
        """
        self.cv_network_strengths.to_csv(os.path.join(self.results_directory, 'cv_network_strengths.csv'))

    def calculate_final_cv_results(self):
        """
        Calculate mean and standard deviation of cross-validation results and save to CSV.

        :param cv_results: DataFrame with cross-validation results.
        :param results_directory: Directory to save the results.
        :return: Updated cross-validation results DataFrame.
        """
        self.cv_results.set_index(['fold', 'network', 'model'], inplace=True)
        self.calculate_model_increments()
        self.agg_results = self.cv_results.groupby(['network', 'model'])[regression_metrics].agg(['mean', 'std'])

        # Save results to CSV
        self.cv_results.to_csv(os.path.join(self.results_directory, 'cv_results.csv'))
        self.agg_results.to_csv(os.path.join(self.results_directory, 'cv_results_mean_std.csv'), float_format='%.4f')
        return

    def aggregate_inner_folds(self):
        self.cv_results.set_index(['fold', 'param_id', 'network', 'model'], inplace=True)
        self.cv_results.sort_index(inplace=True)
        self.calculate_model_increments()
        self.agg_results = self.cv_results.groupby(['network', 'param_id', 'model'])[regression_metrics].agg(['mean', 'std'])

        # save inner cv results to csv in case this is not a permutation run
        if self.perm_run == 0:
            self.cv_results.to_csv(os.path.join(self.results_directory, 'inner_cv_results.csv'))
            self.agg_results.to_csv(os.path.join(self.results_directory, 'inner_cv_results_mean_std.csv'))
        return

    def find_best_params(self):
        # find the best hyperparameter configuration (best edge selection)
        best_params_ids = self.agg_results['spearman_score'].groupby(['network', 'model'])['mean'].idxmax()
        best_params = self.cv_results.loc[(0, best_params_ids.loc[('both', 'full')][1], 'both', 'full'), 'params']
        best_param_id = best_params_ids.loc[('both', 'full')][1]
        return best_params, best_param_id

    @staticmethod
    def collect_results(fold_id, param_id, param, metrics):
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


class PermutationManager:
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
            p_value_series = PermutationManager._calculate_group_p_value(true_group, perms_group)
            p_values.append(pd.DataFrame(p_value_series).T.assign(network=name[0], model=name[1]))

        p_values_df = pd.concat(p_values).reset_index(drop=True)
        p_values_df = p_values_df.set_index(['network', 'model'])
        return p_values_df

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
    def calculate_permutation_results(results_directory, logger):
        """
        Calculate and save the permutation test results.

        :param results_directory: Directory where the results are saved.
        """
        true_results = ResultsManager.load_cv_results(results_directory)

        perm_dir = os.path.join(results_directory, 'permutation')
        valid_perms = glob(os.path.join(perm_dir, '*'))
        perm_results = list()
        stability_positive = list()
        stability_negative = list()
        for perm_run_folder in valid_perms:
            try:
                perm_res = ResultsManager.load_cv_results(perm_run_folder)
                perm_res['permutation'] = os.path.basename(perm_run_folder)
                perm_res = perm_res.set_index('permutation', append=True)
                perm_results.append(perm_res)

                # load edge stability
                stability_positive.append(np.load(os.path.join(perm_run_folder, 'stability_positive_edges.npy')))
                stability_negative.append(np.load(os.path.join(perm_run_folder, 'stability_negative_edges.npy')))

            except FileNotFoundError:
                print(f'No permutation results found for {perm_run_folder}')
        concatenated_df = pd.concat(perm_results)
        concatenated_df.to_csv(os.path.join(results_directory, 'permutation_results.csv'))
        p_values = PermutationManager.calculate_p_values(true_results, concatenated_df)
        p_values.to_csv(os.path.join(results_directory, 'p_values.csv'))

        # permutation stability
        stability_positive = np.stack(stability_positive)
        stability_negative = np.stack(stability_negative)

        # actual stability
        true_stability_positive = np.load(os.path.join(results_directory, 'stability_positive_edges.npy'))
        true_stability_negative = np.load(os.path.join(results_directory, 'stability_negative_edges.npy'))

        use_fdr = True
        if use_fdr:
            calculate_p_values_edges = PermutationManager.calculate_p_values_edges_fdr
        else:
            calculate_p_values_edges = PermutationManager.calculate_p_values_edges_max_value

        sig_stability_positive = calculate_p_values_edges(true_stability_positive, stability_positive)
        sig_stability_negative = calculate_p_values_edges(true_stability_negative, stability_negative)

        np.save(os.path.join(results_directory, 'sig_stability_positive_edges.npy'), sig_stability_positive)
        np.save(os.path.join(results_directory, 'sig_stability_negative_edges.npy'), sig_stability_negative)

        logger.debug("Saving significance of edge stability.")
        logger.info("Permutation test results")
        logger.info(p_values.round(4).to_string())
        return

    @staticmethod
    def calculate_p_values_edges_max_value(true_stability, permutation_stability):
        """
        Calculate empirical p-values for each edge in a connectivity matrix using the
        max-value method from permutation testing.

        For each permutation, the maximum value across all edges is taken to construct
        a max-null distribution. Each true edge value is then compared to this distribution
        to compute a p-value, which controls the family-wise error rate (FWER).

        Parameters
        ----------
        true_stability : ndarray of shape (n_regions, n_regions)
            Symmetric matrix containing the observed stability scores for each edge.

        permutation_stability : ndarray of shape (n_permutations, n_regions, n_regions)
            Array containing stability scores from each permutation run. Each entry is
            a symmetric matrix of the same shape as `true_stability`.

        Returns
        -------
        sig_stability : ndarray of shape (n_regions, n_regions)
            Symmetric matrix of empirical p-values for each edge, calculated by comparing
            the true stability values to the max null distribution. The p-values reflect
            the probability of observing a value as extreme or more extreme under the null.
            Family-wise error is controlled via the max-statistic method.
        """
        # n_permutations
        n_permutations = permutation_stability.shape[0]

        triu_indices = np.triu_indices_from(true_stability, k=1)

        # Extract only the upper triangle for each permutation (ignores symmetric redundancy and diagonal)
        max_null = np.max(permutation_stability[:, triu_indices[0], triu_indices[1]], axis=1)

        # Compute significance p-values per edge, comparing against max null distribution
        sig_stability = np.ones_like(true_stability)

        # For only upper triangle (to avoid redundant computation)
        for i, j in zip(*triu_indices):
            true_val = true_stability[i, j]
            p = (np.sum(max_null >= true_val) + 1) / (n_permutations + 1)

            sig_stability[i, j] = p
            sig_stability[j, i] = p  # symmetric
        return sig_stability

    @staticmethod
    def calculate_p_values_edges_fdr(true_stability, permutation_stability):
        """
        Calculate FDR-corrected p-values for each edge in a connectivity matrix using
        permutation-based empirical p-values and the Benjamini–Yekutieli procedure.

        For each edge, an empirical p-value is calculated by comparing the true
        stability score to the distribution of permuted scores at the same edge.
        The Benjamini–Yekutieli (BY) method is then applied to correct for multiple
        comparisons, controlling the false discovery rate (FDR).

        Parameters
        ----------
        true_stability : ndarray of shape (n_regions, n_regions)
            Symmetric matrix containing the observed stability scores for each edge.

        permutation_stability : ndarray of shape (n_permutations, n_regions, n_regions)
            Array containing stability scores from each permutation run. Each entry is
            a symmetric matrix of the same shape as `true_stability`.

        Returns
        -------
        sig_stability : ndarray of shape (n_regions, n_regions)
            Symmetric matrix of FDR-corrected p-values for each edge, calculated by first
            computing empirical p-values and then applying the Benjamini–Yekutieli correction
            to control the expected false discovery rate across all edges.
        """
        n_permutations = permutation_stability.shape[0]
        triu_indices = np.triu_indices_from(true_stability, k=1)

        # Flatten permutation values at upper triangle positions
        perm_values = permutation_stability[:, triu_indices[0], triu_indices[1]]  # shape: (n_permutations, n_edges)

        # Compute empirical p-values for each edge
        true_values = true_stability[triu_indices]
        p_vals = (np.sum(perm_values >= true_values[None, :], axis=0) + 1) / (n_permutations + 1)
        #p_vals = (np.sum(perm_values >= 1, axis=0) + 1) / 1000

        # Apply Benjamini-Yekutieli correction
        #_, p_vals_corrected, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_by')
        p_vals_corrected = p_vals
        # Fill into symmetric matrix
        sig_stability = np.ones_like(true_stability)
        for idx, (i, j) in enumerate(zip(*triu_indices)):
            p = p_vals_corrected[idx]
            sig_stability[i, j] = p
            sig_stability[j, i] = p

        return sig_stability
