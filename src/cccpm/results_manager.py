import os
import pandas as pd

from typing import Union

import numpy as np
import torch

from glob import glob

from cccpm.constants import Networks, Models, Metrics
from cccpm.utils import vector_to_upper_triangular_matrix, vector_to_matrix_tensor_version


class ResultsManager:
    """
    A class to handle the aggregation, formatting, and saving of results.

    Parameters
    ----------
    output_dir : str
        Directory where results will be saved.
    """
    def __init__(self,
                 output_dir: Union[str, None],
                 n_runs: int,
                 n_folds: int,
                 n_features: int,
                 n_params: int = None,
                 is_inner_cv: bool = False,
                 device: torch.device = torch.device('cpu')):
        self.results_directory = output_dir
        self.is_inner_cv = is_inner_cv
        self.device = device

        # 1. Define Dimensions based on Enums
        self.dims = {
            'models': len(Models),
            'networks': len(Networks),
            'params': n_params if n_params is not None else 1,
            'folds': n_folds,
            'metrics': len(Metrics),
            'runs': n_runs
        }

        # 2. Preallocate Metrics Tensor
        # Shape: [Metrics, Models, Networks, Params, Folds, Runs]
        self.results = torch.zeros(
            self.dims['metrics'],
            self.dims['models'],
            self.dims['networks'],
            self.dims['params'],
            self.dims['folds'],
            self.dims['runs'],
            device=self.device
        )

        # 3. Handle Edges (Features)
        # Shape: [N_Features, 2, Params, Folds, Runs]
        # Only store positive and negative edges (not "both")
        self.n_features = n_features
        self.cv_edges = torch.zeros(
            self.n_features,
            2,  # Only positive and negative networks
            self.dims['params'],
            self.dims['folds'],
            self.dims['runs'],
            dtype=torch.bool,
            device=self.device
        )

        # Placeholder for predictions if you need them later
        self.cv_predictions = []
        self.agg_results = None

    def store_edges(self, param_idx: int, fold_idx: int, edges_tensor):
        """
        Stores the edge masks for Positive and Negative networks.

        Args:
            param_idx: Index of current parameter.
            fold_idx: Index of current fold.
            edges_tensor: Boolean Tensor of shape [Features, 2, Runs].
                          Dimension 1 must correspond to [Positive, Negative].
        """
        # We assume edges_tensor comes in as [Features, 2, Runs]
        # This matches cv_edges shape directly

        # Target Slice: [:, :, param, fold, :]
        self.cv_edges[:, :, param_idx, fold_idx, :] = torch.Tensor(edges_tensor)


    def store_metrics(self, param_idx: int, fold_idx: int, metrics_tensor: torch.Tensor):
        """
        Stores a batch of metrics returned by FastCPMMetrics.

        Args:
            param_idx: Index of the current parameter configuration.
            fold_idx: Index of the current CV fold.
            metrics_tensor: 4D Tensor [Metrics, Models, Networks, Runs]
        """
        # We assign the entire 4D block into the 6D tensor at the specific param/fold slice.
        # This replaces the need for nested loops.

        # Destination slice: [:, :, :, param_idx, fold_idx, :]
        # Source shape:      [Metrics, Models, Networks, Runs]
        self.results[:, :, :, param_idx, fold_idx, :] = metrics_tensor.cpu()

    def calculate_edge_stability(self, write: bool = True, best_param_id: int = None):
        """
        Calculate and save edge stability and overlap.

        :param cv_edges: Cross-validation edges.
        :param results_directory: Directory to save the results.
        """
        if best_param_id is None:
            best_param_id = 0
            n_runs = 1
        else:
            n_runs = best_param_id.size(0)
        run_indices = torch.arange(n_runs, device=self.cv_edges.device)

        # 1. Advanced Indexing: Select the specific param for each run simultaneously
        # Input Shape:  [N_Features, 2, Params, Folds, Runs]
        # We index Dim 2 (Params) and Dim 4 (Runs) with paired vectors.
        # Result Shape: [N_Features, 2, Folds, Runs]
        selected_edges = self.cv_edges[:, :, best_param_id, :, run_indices]

        # 2. Calculate Stability
        # Average over Folds (Dim 2)
        # Shape: [N_Features, 2, Runs]
        edge_stability = selected_edges.float().mean(dim=2)

        if write:
            # Keep shape [Features, 2, Folds, Runs] for edges
            # Keep shape [Features, 2, Runs] for stability
            np.save(os.path.join(self.results_directory, f'edges.npy'),
                    vector_to_matrix_tensor_version(selected_edges, dim=0).float().cpu().numpy())
            np.save(os.path.join(self.results_directory, f'stability_edges.npy'),
                    vector_to_matrix_tensor_version(edge_stability, dim=0).cpu().numpy())
        return edge_stability

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

    def calculate_final_cv_results_old(self):
        """
        Calculate mean and standard deviation of cross-validation results and save to CSV.

        :param cv_results: DataFrame with cross-validation results.
        :param results_directory: Directory to save the results.
        :return: Updated cross-validation results DataFrame.
        """
        # calculate increments
        self.results[Models.increment] = self.results[Models.full] - self.results[Models.connectome]

        # 2. Calculate Means across Folds (Dimension 3)
        means = torch.mean(self.results, dim=3)
        std = torch.std(self.results, dim=3)


        self.agg_results = self.cv_results.groupby(['network', 'model'])[regression_metrics].agg(['mean', 'std'])

        # Save results to CSV
        self.cv_results.to_csv(os.path.join(self.results_directory, 'cv_results.csv'))
        self.agg_results.to_csv(os.path.join(self.results_directory, 'cv_results_mean_std.csv'), float_format='%.4f')
        return

    def calculate_final_cv_results(self):
        # Calculate increment: Full - Connectome
        self.results[:, Models.increment] = self.results[:, Models.full] - self.results[:, Models.connectome]

        # Move to CPU for processing
        # Shape: [Metrics, Models, Networks, Params, Folds, Runs]
        data = self.results.cpu()

        n_metrics, n_models, n_nets, _, n_folds, n_runs = data.shape

        # Get Lists of Names for Indices
        model_names = [m.name for m in Models]
        net_names = [n.name for n in Networks]
        metrics = [m.name for m in Metrics]
        fold_indices = range(n_folds)
        run_indices = range(n_runs)

        # Permute to [Models, Networks, Params, Folds, Runs, Metrics]
        raw_tensor = data.permute(1, 2, 3, 4, 5, 0)

        # 2. Reshape into 2D Matrix: [Rows, Metrics]
        raw_matrix = raw_tensor.reshape(-1, n_metrics).numpy()

        # 3. Create MultiIndex
        raw_index = pd.MultiIndex.from_product(
            [model_names, net_names, fold_indices, run_indices],
            names=['model', 'network', 'fold', 'run']
        )

        # 4. Create DataFrame
        df_raw = pd.DataFrame(raw_matrix, index=raw_index, columns=metrics)

        # Save Raw Results
        df_raw.to_csv(os.path.join(self.results_directory, 'cv_results_full.csv'))

        # 1. Calculate Stats over Folds (dim = 4)
        # Shape after mean/std: [Metrics, Models, Networks, Params, Runs]
        means = torch.mean(data, dim=4)
        stds = torch.std(data, dim=4)

        # Permute to [Models, Networks, Params, Runs, Metrics]
        means = means.permute(1, 2, 3, 4, 0)
        stds = stds.permute(1, 2, 3, 4, 0)

        # 3. Reshape
        means_flat = means.reshape(-1, n_metrics).numpy()
        stds_flat = stds.reshape(-1, n_metrics).numpy()

        # 4. Create Index (Model, Network, run)
        agg_index = pd.MultiIndex.from_product(
            [model_names, net_names, run_indices],
            names=['model', 'network', 'run']
        )

        # 5. Create DataFrame with MultiIndex Columns
        df_mean = pd.DataFrame(means_flat, index=agg_index, columns=metrics)
        df_std = pd.DataFrame(stds_flat, index=agg_index, columns=metrics)

        # Concatenate columns: Metric -> (Mean, Std)
        df_agg = pd.concat([df_mean, df_std], axis=1, keys=['mean', 'std'])
        df_agg = df_agg.swaplevel(0, 1, axis=1).sort_index(axis=1)

        self.agg_results = df_agg.copy()
        df_agg.to_csv(os.path.join(self.results_directory, 'cv_results_summary.csv'), float_format='%.4f')
        return df_agg

    def aggregate_inner_folds(self):
        """
        Calculates increments, aggregates across folds, and saves results.
        """
        # 1. Calculate Increments (Full - Connectome) inline
        # This operates on the entire tensor at once (all params, folds, metrics, perms)
        self.results[:, Models.increment] = self.results[:, Models.full] - self.results[:, Models.connectome]

        # 2. Calculate Means across Folds (Dimension 4)
        inner_means = torch.mean(self.results, dim=4)

        # (Optional) Calculate Stds if you want to save/inspect them
        # inner_stds = torch.std(self.results, dim=3)

        # 3. Save to CSV (assuming Permutation 0 is the real data)
        #if self.dims['perms'] > 0:
        #    self._save_inner_cv_to_csv(inner_means, perm_idx=0)

    def _save_inner_cv_to_csv(self, data_tensor, perm_idx=0):
        """
        Helper to flatten the tensor and save to CSV.
        """
        import pandas as pd

        # Create MultiIndex for rows: [Models, Networks, Params]
        iterables = [
            [m.name for m in Models],
            [n.name for n in Networks],
            range(self.dims['params'])
        ]
        index = pd.MultiIndex.from_product(iterables, names=['model', 'network', 'param_id'])

        # Flatten: Select perm -> Flatten to [Rows, Metrics]
        data_slice = data_tensor[..., perm_idx].reshape(-1, self.dims['metrics']).cpu().numpy()

        df = pd.DataFrame(data_slice, index=index, columns=[m.name for m in Metrics])

        path = f"{self.results_directory}/inner_cv_results_mean.csv"
        df.to_csv(path)
        print(f"Inner CV means saved to {path}")

    def find_best_params(self):
        # Slice Result Shape: [Params, Folds]
        scores_slice = self.results[
            Models.connectome,
            Networks.both,
            :,
            :,
            Metrics.pearson_score,
            :
        ]

        # 2. Calculate Means across Folds (now Dimension 1) -> Shape: [Params, Perms]
        mean_scores = torch.mean(scores_slice, dim=1)

        # 3. Find Index of Maximum -> Shape: [Perms]
        best_param_idx = torch.argmax(mean_scores, dim=0)
        return best_param_idx

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
