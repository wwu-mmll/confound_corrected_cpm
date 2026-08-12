import os
import json
import pandas as pd

from typing import Union

import numpy as np
import torch


import networkx as nx

from cccpm.constants import Networks, Models, Metrics, TaskType, get_metrics_for_task
from cccpm.utils import vector_to_matrix_tensor_version


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
                 device: torch.device = torch.device('cpu'),
                 store_fold_edges: bool = True):
        self.results_directory = output_dir
        self.is_inner_cv = is_inner_cv
        self.device = device
        # Edge bookkeeping is kept on the CPU regardless of the compute device: it
        # only aggregates stability and (optionally) writes per-fold edges, and a
        # [Features, 2, Folds, Runs] tensor would otherwise consume huge amounts of
        # VRAM for large parcellations x many folds x many permutations (this was
        # a real CUDA OOM source -- see the "develop" branch fix this ports).
        self.edge_device = torch.device('cpu')
        self.store_fold_edges = store_fold_edges

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

        # 3. Handle Edges (Features) -- kept on the CPU (self.edge_device), never
        # on the compute device; see the note on self.edge_device above.
        self.n_features = n_features
        # Running sum of selected-edge masks over folds -> stability = sum / n_folds.
        # This is all edge stability needs, and dropping the folds axis here (vs.
        # keeping every fold's mask) is what avoids the VRAM blowup for many
        # folds x many permutations.
        # Shape: [N_Features, 2, Params, Runs] (positive/negative only, not "both").
        self.cv_edge_sum = torch.zeros(
            self.n_features, 2, self.dims['params'], self.dims['runs'],
            dtype=torch.float32, device=self.edge_device
        )
        # Per-fold masks are retained only when we need to write edges.npy (the
        # real run). Permutation / inner-CV passes keep only the fold-sum above.
        # Shape: [N_Features, 2, Params, Folds, Runs].
        self.cv_edges = None
        if store_fold_edges:
            self.cv_edges = torch.zeros(
                self.n_features, 2, self.dims['params'],
                self.dims['folds'], self.dims['runs'],
                dtype=torch.bool, device=self.edge_device
            )

        # Placeholder for predictions if you need them later
        self.cv_predictions = []
        self.cv_network_strengths = pd.DataFrame()
        self.agg_results = None

    @staticmethod
    def estimate_slice_bytes(n_features: int, n_params: int, n_folds: int, n_runs: int) -> int:
        """
        Bytes needed, on the COMPUTE DEVICE, for a `[..., n_params, n_folds,
        n_runs]`-shaped slice of `self.results` (float32) -- using the exact
        same shape formula as the tensor allocated in `__init__`. Shared with
        `batch_planning.py` so the byte-size arithmetic is defined once.

        Edge bookkeeping (`cv_edges`/`cv_edge_sum`) is intentionally excluded:
        it always lives on the CPU (`self.edge_device`), never on the compute
        device, so it doesn't count against the GPU/CPU compute-device memory
        budget batch_planning checks.
        """
        return len(Metrics) * len(Models) * len(Networks) * n_params * n_folds * n_runs * 4

    def store_edges(self, param_idx, fold_idx, edges_tensor, run_idx=slice(None)):
        """
        Stores the edge masks for Positive and Negative networks: moves the
        mask to the CPU edge store (frees VRAM) and accumulates the fold-sum
        used for stability. The per-fold mask itself is retained only when
        this manager was built with store_fold_edges=True.

        Args:
            param_idx: Index (int or slice) of current parameter(s).
            fold_idx: Index (int or slice) of current fold(s).
            edges_tensor: Boolean Tensor of shape [Features, 2, (Params,) (Folds,) Runs],
                          matching whatever param_idx/fold_idx/run_idx slice out.
                          Dimension 1 must correspond to [Positive, Negative].
            run_idx: Index (int or slice) of current run(s)/permutation(s).
                     Defaults to every run (today's behaviour).
        """
        mask = torch.as_tensor(edges_tensor, dtype=torch.bool, device=self.edge_device)

        if self.cv_edges is not None:
            self.cv_edges[:, :, param_idx, fold_idx, run_idx] = mask

        # cv_edge_sum has no folds axis, so any folds axis present in `mask`
        # must be summed out before accumulating. `mask`'s axis layout mirrors
        # cv_edges[:, :, param_idx, fold_idx, run_idx] positionally: Features,
        # Network, then Params (present unless param_idx is an int, which
        # collapses it), then Folds (present unless fold_idx is an int).
        # fold_idx is a plain int only for the single-fold-at-a-time (non-batched)
        # call pattern, in which case there's no folds axis to sum at all.
        if isinstance(fold_idx, int):
            fold_sum = mask.float()
        else:
            fold_axis = 2 + (0 if isinstance(param_idx, int) else 1)
            fold_sum = mask.float().sum(dim=fold_axis)
        self.cv_edge_sum[:, :, param_idx, run_idx] += fold_sum


    def store_metrics(self, param_idx, fold_idx, metrics_tensor: torch.Tensor, run_idx=slice(None)):
        """
        Stores a batch of metrics returned by FastCPMMetrics.

        Args:
            param_idx: Index (int or slice) of the current parameter configuration(s).
            fold_idx: Index (int or slice) of the current CV fold(s).
            metrics_tensor: Tensor [Metrics, Models, Networks, (Params,) (Folds,) Runs],
                            matching whatever param_idx/fold_idx/run_idx slice out.
            run_idx: Index (int or slice) of the current run(s)/permutation(s).
                     Defaults to every run (today's behaviour).
        """
        # We assign the entire block into the 6D tensor at the specific param/fold/run slice.
        # This replaces the need for nested loops.

        # Destination slice: [:, :, :, param_idx, fold_idx, run_idx]
        # Source shape:      [Metrics, Models, Networks, (Params,) (Folds,) Runs]
        self.results[:, :, :, param_idx, fold_idx, run_idx] = metrics_tensor.to(self.results.device)

    def calculate_edge_stability(self, write: bool = True, best_param_id=None):
        """
        Calculate and save edge stability.

        Stability is the fraction of outer folds in which each edge was selected,
        for the chosen hyperparameter of each run -- read straight from the CPU
        fold-sum accumulator (`cv_edge_sum`), so this never touches the compute
        device. ``stability_edges.npy`` (``[Nodes, Nodes, 2, Runs]``) is always
        written; the per-fold ``edges.npy`` is written only when this manager
        retained the per-fold masks (the real run -- see ``store_fold_edges``).

        Args:
            write: whether to save the .npy files.
            best_param_id: None (assume param index 0 for every run), a single
                           int/0-d tensor (same param for every run), or an
                           array-like of length n_runs (one winning param per run).
        """
        n_runs = self.dims['runs']
        if best_param_id is None:
            best_param_id = torch.zeros(n_runs, dtype=torch.long)
        else:
            best_param_id = torch.as_tensor(best_param_id, dtype=torch.long).reshape(-1)
            if best_param_id.numel() == 1:
                best_param_id = best_param_id.expand(n_runs)
        best_param_id = best_param_id.to(self.edge_device)
        run_indices = torch.arange(n_runs, device=self.edge_device)

        # Fraction of folds selecting each edge, for the chosen param per run.
        # cv_edge_sum: [Features, 2, Params, Runs]; paired-index Params x Runs.
        # Result: [Features, 2, Runs].
        edge_stability = self.cv_edge_sum[:, :, best_param_id, run_indices] / self.dims['folds']

        if write:
            np.save(os.path.join(self.results_directory, 'stability_edges.npy'),
                    vector_to_matrix_tensor_version(edge_stability, dim=0).numpy())
            if self.cv_edges is not None:
                # Per-fold selected edges for the chosen param -> edges.npy.
                # [:, :, param, :, run] -> [Runs, Features, 2, Folds] (advanced
                # indices are non-adjacent, so they lead) -> [Features, 2, Folds, Runs].
                selected_edges = self.cv_edges[:, :, best_param_id, :, run_indices]
                selected_edges = selected_edges.permute(1, 2, 3, 0)
                np.save(os.path.join(self.results_directory, 'edges.npy'),
                        vector_to_matrix_tensor_version(selected_edges, dim=0).float().numpy())
        return edge_stability

    def store_predictions(self, y_pred, y_true, fold, test_indices):
        y_pred = y_pred.detach().cpu().numpy().squeeze(-1)
        y_true = y_true.reshape(-1)

        batch_size = y_pred.shape[0]
        n_models = len(Models)
        n_networks = len(Networks)
        n_combinations = n_models * n_networks
        network_names = [n.name for n in sorted(Networks, key=lambda x: x.value)]
        model_names = [m.name for m in sorted(Models, key=lambda x: x.value)]

        flat_true = np.repeat(y_true, n_combinations)

        multi_index = pd.MultiIndex.from_product(
            [test_indices, model_names, network_names],
            names=['sample_index', 'model', 'network']
        )
        flat_preds = y_pred.ravel()

        # 5. Build Mini-DataFrame
        batch_df = pd.DataFrame({
            'y_pred': flat_preds,
            'y_true': flat_true
        }, index=multi_index)

        # Add fold metadata
        batch_df['fold'] = fold

        # Reset index to turn MultiIndex levels into columns
        batch_df = batch_df.reset_index()

        self.cv_predictions.append(batch_df)



    def store_network_strengths(self, network_strengths, y_true, fold):
        # Use a list comprehension to build data more concisely
        data = [
            pd.DataFrame({
                'y_true': y_true.squeeze(),
                'network_strength': np.squeeze(network_strengths[m][n].cpu().numpy()),
                'model': m,
                'network': n,
                'fold': fold
            })
            for m in ['connectome', 'residuals']
            for n in ['positive', 'negative']
        ]

        self.cv_network_strengths = pd.concat([self.cv_network_strengths] + data, ignore_index=True)

    @staticmethod
    def load_cv_results(folder):
        """
        Load cross-validation results from a CSV file.

        :param folder: Directory containing the results file.
        :return: DataFrame with the loaded results.
        """
        results = pd.read_csv(os.path.join(folder, 'cv_results_summary.csv'), header=[0, 1], index_col=[0, 1, 2])
        results = results.loc[:, results.columns.get_level_values(1) == 'mean']
        results.columns = results.columns.droplevel(1)
        return results

    def save_predictions(self):
        """
        Save predictions to CSV, sorted by sample index.
        """
        if isinstance(self.cv_predictions, list):
            if not self.cv_predictions:
                return
            self.cv_predictions = pd.concat(self.cv_predictions, ignore_index=True)

        df = self.cv_predictions.sort_values(by='sample_index')
        df.to_csv(os.path.join(self.results_directory, 'cv_predictions.csv'), index=False)

    # def save_predictions(self):
    #     """
    #     Save predictions to CSV, sorted by sample index.
    #     """
    #     if isinstance(self.cv_predictions, list):
    #         if not self.cv_predictions:
    #             return
    #         self.cv_predictions = pd.concat(self.cv_predictions, ignore_index=True)
    #
    #     df = self.cv_predictions.sort_values(by='sample_index')
    #     df.to_csv(os.path.join(self.results_directory, 'cv_predictions.csv'), index=False)

    def save_network_strengths(self):
        """
        Save network strengths to CSV.
        """
        self.cv_network_strengths.to_csv(os.path.join(self.results_directory, 'cv_network_strengths.csv'))

    def calculate_final_cv_results(self, task_type: TaskType = TaskType.regression):
        # Calculate increment: Full - Covariates (added value of the connectome over confounds)
        self.results[:, Models.increment] = self.results[:, Models.full] - self.results[:, Models.covariates]

        # Move to CPU for processing
        # Shape: [Metrics, Models, Networks, Params, Folds, Runs]
        data = self.results.cpu()

        n_metrics, n_models, n_nets, _, n_folds, n_runs = data.shape

        # Get Lists of Names for Indices
        model_names = [m.name for m in Models]
        net_names = [n.name for n in Networks]
        all_metrics = [m.name for m in Metrics]
        relevant_metrics = [m.name for m in get_metrics_for_task(task_type)]
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

        # 4. Create DataFrame and filter to relevant metrics
        df_raw = pd.DataFrame(raw_matrix, index=raw_index, columns=all_metrics)
        df_raw = df_raw[relevant_metrics]

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

        # 5. Create DataFrame with MultiIndex Columns, filtered to relevant metrics
        df_mean = pd.DataFrame(means_flat, index=agg_index, columns=all_metrics)[relevant_metrics]
        df_std = pd.DataFrame(stds_flat, index=agg_index, columns=all_metrics)[relevant_metrics]

        # Concatenate columns: Metric -> (Mean, Std)
        df_agg = pd.concat([df_mean, df_std], axis=1, keys=['mean', 'std'])
        df_agg = df_agg.swaplevel(0, 1, axis=1).sort_index(axis=1)

        self.agg_results = df_agg.copy()
        df_agg.to_csv(os.path.join(self.results_directory, 'cv_results_summary.csv'), float_format='%.4f')

        if self.cv_predictions:
            self.cv_predictions = pd.concat(self.cv_predictions, ignore_index=True)
            self.cv_predictions.to_csv(os.path.join(self.results_directory, 'cv_predictions.csv'))
        return df_agg

    def aggregate_inner_folds(self):
        """
        Calculates increments, aggregates across folds, and saves results.
        """
        # 1. Calculate Increments (Full - Covariates) inline
        # This operates on the entire tensor at once (all params, folds, metrics, perms)
        self.results[:, Models.increment] = self.results[:, Models.full] - self.results[:, Models.covariates]

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

    def find_best_params(self, task_type=TaskType.regression):
        # Select appropriate metric based on task type
        if task_type == TaskType.classification:
            metric = Metrics.balanced_accuracy
        else:
            metric = Metrics.pearson_score

        # Slice Result Shape: [Params, Folds]
        scores_slice = self.results[
            metric,
            Models.connectome,
            Networks.both
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

    # Metrics where lower is better (p-value: true < perm)
    LOWER_IS_BETTER = {'mean_squared_error', 'mean_absolute_error'}

    @staticmethod
    def _is_lower_better(column_name):
        """Check if a metric is one where lower values are better."""
        if column_name in PermutationManager.LOWER_IS_BETTER:
            return True
        if column_name.endswith('error'):
            return True
        return False

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
            if PermutationManager._is_lower_better(column):
                condition_count = (true_group[column].values[0] > perms_group[column].astype(float)).sum()
            else:
                # Higher is better: score, accuracy, balanced_accuracy, f1_score, roc_auc, etc.
                condition_count = (true_group[column].values[0] < perms_group[column].astype(float)).sum()

            # Standard permutation p-value (Phipson & Smyth, 2010): the +1 in both
            # numerator and denominator counts the observed statistic itself and
            # guarantees a valid p-value in (0, 1].
            result_dict[column] = (condition_count + 1) / (len(perms_group[column]) + 1)

        return pd.Series(result_dict)

    @staticmethod
    def calculate_permutation_results(results_directory, logger, method="nbs",
                                      nbs_threshold=0.5, nbs_component_stat="extent"):
        """
        Calculate and save the permutation test results.

        Model-level metric p-values are always computed. Edge-stability
        significance is established at the *subnetwork* level via a
        Network-Based Statistic (``method='nbs'``) or, threshold-free, via
        network TFCE (``method='tfce'``); both control the family-wise error
        rate through a permutation max-statistic and write a per-edge p-value
        matrix (edges belonging to a significant subnetwork carry that
        subnetwork's p-value) to ``stability_edges_significance.npy``.

        :param results_directory: Directory where the results are saved.
        :param logger: Logger for progress messages.
        :param method: Edge-significance method, ``'nbs'`` (default) or ``'tfce'``.
        :param nbs_threshold: Stability threshold for NBS component forming.
        :param nbs_component_stat: NBS component statistic, ``'extent'`` or ``'intensity'``.
        """
        true_results = ResultsManager.load_cv_results(results_directory)

        perm_dir = os.path.join(results_directory, 'permutation')
        perm_results = ResultsManager.load_cv_results(perm_dir)

        true_edge_stability = np.load(os.path.join(results_directory, 'stability_edges.npy'))
        perm_edge_stability = np.load(os.path.join(perm_dir, 'stability_edges.npy'))

        p_values = PermutationManager.calculate_p_values(true_results, perm_results)
        p_values.to_csv(os.path.join(results_directory, 'p_values.csv'))

        if method == "nbs":
            stability_significance, sig_meta = PermutationManager.calculate_p_values_edges_nbs(
                true_edge_stability, perm_edge_stability,
                threshold=nbs_threshold, component_stat=nbs_component_stat,
                return_diagnostics=True)
        elif method == "tfce":
            stability_significance, sig_meta = PermutationManager.calculate_p_values_edges_tfce(
                true_edge_stability, perm_edge_stability, return_diagnostics=True)
        else:
            raise ValueError(
                f"Unknown edge-significance method '{method}'. Use 'nbs' or 'tfce'.")

        np.save(os.path.join(results_directory, 'stability_edges_significance.npy'), stability_significance)
        with open(os.path.join(results_directory, 'stability_edges_significance_meta.json'), 'w') as f:
            json.dump(sig_meta, f)

        logger.debug("Saving significance of edge stability.")
        logger.info("Permutation test results")
        logger.info(p_values.round(4).to_string())
        return

    @staticmethod
    def _layer_arrays(true_stability, permutation_stability, layer):
        """Extract the observed ``[n_nodes, n_nodes]`` matrix and the
        ``[n_nodes, n_nodes, n_perms]`` null stack for one network *layer*
        (``0`` = positive, ``1`` = negative) from the stored stability arrays,
        which have shape ``[n_nodes, n_nodes, 2, runs]``."""
        true_layer = true_stability[:, :, layer, 0]
        perm_layer = permutation_stability[:, :, layer, :]
        return true_layer, perm_layer

    @staticmethod
    def _connected_components(supra):
        """Yield the edge lists of the connected components (each with at least
        one edge) of a symmetric boolean adjacency matrix, using its upper
        triangle only. Isolated nodes are skipped."""
        graph = nx.from_numpy_array(np.triu(supra, k=1))
        for nodes in nx.connected_components(graph):
            if len(nodes) < 2:
                continue
            edges = list(graph.subgraph(nodes).edges())
            if edges:
                yield edges

    @staticmethod
    def calculate_p_values_edges_nbs(true_stability, permutation_stability,
                                     threshold=0.5, component_stat="extent",
                                     alpha=0.05, return_diagnostics=False):
        """
        Network-Based Statistic (Zalesky et al., 2010) for edge-stability
        significance.

        Edges whose stability meets ``threshold`` form a graph; its connected
        components are the candidate subnetworks. A permutation null of the
        **largest component statistic** controls the family-wise error rate, so
        an observed component is significant if it is larger/stronger than the
        biggest component seen in (almost) any permutation. Each component's
        p-value is broadcast onto all of its member edges; every other edge is
        assigned ``p = 1``.

        Inference is at the *subnetwork* level: a significant result licenses
        "this connected subnetwork is selected more consistently than chance",
        not per-edge claims.

        Note on discreteness: stability over ``K`` outer folds takes only the
        values ``{0, 1/K, ..., 1}``, so ``threshold=0.5`` keeps edges selected
        in a majority of folds and the effective thresholding is coarse for few
        folds. A continuous edge statistic (deferred) would sharpen this.

        Parameters
        ----------
        true_stability : ndarray of shape (n_nodes, n_nodes, 2, 1)
            Observed edge stability; dim 2 is the positive/negative network.
        permutation_stability : ndarray of shape (n_nodes, n_nodes, 2, n_perms)
            Edge stability from each permutation run.
        threshold : float, default=0.5
            Stability threshold (``>=``) for component forming.
        component_stat : {'extent', 'intensity'}, default='extent'
            ``'extent'`` = number of edges in the component (classic NBS);
            ``'intensity'`` = sum of ``(stability - threshold)`` over its edges.
        alpha : float, default=0.05
            Significance level recorded in the diagnostics.
        return_diagnostics : bool, default=False
            If ``True``, also return a JSON-serialisable diagnostics dict with
            the per-network max-component null distribution, the observed
            components (size / statistic / p-value) and the largest component.

        Returns
        -------
        sig_stability : ndarray of shape (n_nodes, n_nodes, 2)
            Per-edge p-values (member edges carry their component's p-value).
        diagnostics : dict, optional
            Returned only when ``return_diagnostics=True``.
        """
        true_stability = np.asarray(true_stability, dtype=float)
        permutation_stability = np.asarray(permutation_stability, dtype=float)
        n_nodes = true_stability.shape[0]
        n_perms = permutation_stability.shape[-1]
        sig = np.ones((n_nodes, n_nodes, 2))

        if component_stat not in ("extent", "intensity"):
            raise ValueError(
                f"Unknown component_stat '{component_stat}'. Use 'extent' or 'intensity'.")

        def components_with_stats(mat):
            out = []
            for edges in PermutationManager._connected_components(mat >= threshold):
                if component_stat == "extent":
                    stat = float(len(edges))
                else:  # intensity
                    stat = float(sum(mat[i, j] - threshold for i, j in edges))
                out.append((edges, stat))
            return out

        meta = {
            "method": "nbs",
            "threshold": float(threshold),
            "component_stat": component_stat,
            "n_permutations": int(n_perms),
            "alpha": float(alpha),
            "statistic_label": ("Component size (edges)" if component_stat == "extent"
                                else "Component intensity"),
            "networks": {},
        }

        for layer in (Networks.positive, Networks.negative):
            true_layer, perm_layer = PermutationManager._layer_arrays(
                true_stability, permutation_stability, layer)

            # Null distribution of the largest component statistic.
            max_null = np.zeros(n_perms)
            for p in range(n_perms):
                comps = components_with_stats(perm_layer[:, :, p])
                max_null[p] = max((s for _, s in comps), default=0.0)

            # Observed components -> component p-value on every member edge.
            components = []
            for edges, stat in components_with_stats(true_layer):
                p_value = (np.sum(max_null >= stat) + 1) / (n_perms + 1)
                nodes = set()
                for i, j in edges:
                    nodes.update((i, j))
                    sig[i, j, layer] = p_value
                    sig[j, i, layer] = p_value
                components.append({
                    "n_edges": int(len(edges)),
                    "n_nodes": int(len(nodes)),
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "significant": bool(p_value < alpha),
                })

            components.sort(key=lambda c: c["statistic"], reverse=True)
            name = "positive" if layer == Networks.positive else "negative"
            meta["networks"][name] = {
                "max_null": [float(v) for v in max_null],
                "critical_value": float(np.quantile(max_null, 1.0 - alpha)) if n_perms else 0.0,
                "components": components,
                "largest_component_edges": max((c["n_edges"] for c in components), default=0),
                "n_significant_components": int(sum(c["significant"] for c in components)),
            }

        if return_diagnostics:
            return sig, meta
        return sig

    @staticmethod
    def _tfce_map(layer_matrix, heights, E, H, dh):
        """Threshold-Free Cluster Enhancement score per edge for one network
        layer: integrate ``extent(component)^E * h^H * dh`` over the threshold
        sweep *heights* (extent = number of edges in the component the edge
        belongs to at height ``h``)."""
        n_nodes = layer_matrix.shape[0]
        tfce = np.zeros((n_nodes, n_nodes))
        # Tolerance so an edge whose stability lands exactly on a sweep height is
        # reliably included there. ``heights`` comes from ``np.arange``, whose
        # accumulated rounding can place a gridpoint a hair above the intended
        # value (e.g. 0.8 -> 0.8000000000000001); without this, an edge at that
        # value would non-deterministically drop its top contribution.
        tol = dh * 1e-6
        for h in heights:
            for edges in PermutationManager._connected_components(layer_matrix >= h - tol):
                contrib = (len(edges) ** E) * (h ** H) * dh
                for i, j in edges:
                    tfce[i, j] += contrib
                    tfce[j, i] += contrib
        return tfce

    @staticmethod
    def calculate_p_values_edges_tfce(true_stability, permutation_stability,
                                      E=0.5, H=2.0, dh=0.1, alpha=0.05,
                                      return_diagnostics=False):
        """
        Threshold-Free Cluster Enhancement (Smith & Nichols, 2009) adapted to
        networks, for per-edge stability significance without an arbitrary
        primary threshold.

        Each edge's TFCE score integrates the support of the components it
        belongs to across a sweep of stability thresholds. A permutation
        **max-TFCE** null across edges controls the family-wise error rate, so
        this yields genuine per-edge FWER-corrected p-values (unlike NBS, which
        is subnetwork-level).

        Parameters
        ----------
        true_stability : ndarray of shape (n_nodes, n_nodes, 2, 1)
            Observed edge stability; dim 2 is the positive/negative network.
        permutation_stability : ndarray of shape (n_nodes, n_nodes, 2, n_perms)
            Edge stability from each permutation run.
        E, H : float
            TFCE extent/height exponents (field-standard defaults 0.5 / 2.0).
        dh : float, default=0.1
            Step of the stability-threshold sweep over ``(0, 1]``.
        alpha : float, default=0.05
            Significance level recorded in the diagnostics.
        return_diagnostics : bool, default=False
            If ``True``, also return a JSON-serialisable diagnostics dict with
            the per-network max-TFCE null distribution and observed maximum.

        Returns
        -------
        sig_stability : ndarray of shape (n_nodes, n_nodes, 2)
            Per-edge FWER-corrected p-values.
        diagnostics : dict, optional
            Returned only when ``return_diagnostics=True``.
        """
        true_stability = np.asarray(true_stability, dtype=float)
        permutation_stability = np.asarray(permutation_stability, dtype=float)
        n_nodes = true_stability.shape[0]
        n_perms = permutation_stability.shape[-1]
        heights = np.arange(dh, 1.0 + dh / 2, dh)
        triu = np.triu_indices(n_nodes, k=1)
        sig = np.ones((n_nodes, n_nodes, 2))

        meta = {
            "method": "tfce",
            "E": float(E), "H": float(H), "dh": float(dh),
            "n_permutations": int(n_perms),
            "alpha": float(alpha),
            "statistic_label": "Max TFCE score",
            "networks": {},
        }

        for layer in (Networks.positive, Networks.negative):
            true_layer, perm_layer = PermutationManager._layer_arrays(
                true_stability, permutation_stability, layer)

            true_tfce = PermutationManager._tfce_map(true_layer, heights, E, H, dh)
            max_null = np.zeros(n_perms)
            for p in range(n_perms):
                perm_tfce = PermutationManager._tfce_map(perm_layer[:, :, p], heights, E, H, dh)
                max_null[p] = perm_tfce[triu].max(initial=0.0)

            for i, j in zip(*triu):
                score = true_tfce[i, j]
                if score > 0:
                    p_value = (np.sum(max_null >= score) + 1) / (n_perms + 1)
                    sig[i, j, layer] = p_value
                    sig[j, i, layer] = p_value

            name = "positive" if layer == Networks.positive else "negative"
            meta["networks"][name] = {
                "max_null": [float(v) for v in max_null],
                "critical_value": float(np.quantile(max_null, 1.0 - alpha)) if n_perms else 0.0,
                "observed_max": float(true_tfce[triu].max(initial=0.0)),
                "n_significant_edges": int(np.sum(sig[:, :, layer][triu] < alpha)),
            }

        if return_diagnostics:
            return sig, meta
        return sig
