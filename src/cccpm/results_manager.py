"""
Accumulation of cross-validation results.

`ResultsManager` is a preallocated store: the pipeline writes per-fold edges,
metrics, predictions and network strengths into fixed-shape tensors, then
aggregates them into the CSV/`npy` files the report reads.

It holds no statistics beyond means and argmax -- the permutation inference
that used to live in this file is now in `inference.py`.
"""
import os
import json
from typing import Union

import numpy as np
import pandas as pd
import torch

from cccpm.constants import Networks, Models, Metrics, TaskType, get_metrics_for_task
from cccpm.connectome import vector_to_matrix_tensor_version


class ResultsManager:
    """
    A class to handle the aggregation, formatting, and saving of results.

    Parameters
    ----------
    output_dir : str
        Directory where results will be saved.
    available_models : list of str, optional
        Names of the model variants this run defines. Persisted to
        ``available_models.json`` so the report can skip the rest; defaults to
        every entry of :class:`~cccpm.constants.Models`.
    """
    def __init__(self,
                 output_dir: Union[str, None],
                 n_runs: int,
                 n_folds: int,
                 n_features: int,
                 n_params: int = None,
                 is_inner_cv: bool = False,
                 device: torch.device = torch.device('cpu'),
                 store_fold_edges: bool = True,
                 available_models: Union[list, None] = None):
        self.results_directory = output_dir
        self.is_inner_cv = is_inner_cv
        # Which model variants this run actually defines. The results tensor is
        # always full-size -- undefined variants are NaN-filled rather than
        # reshaped away -- so the reporting layer needs to be told which rows
        # mean something. Defaults to all of them.
        self.available_models = (list(available_models) if available_models is not None
                                 else [m.name for m in Models])
        self.device = device
        # Edge bookkeeping is kept on the CPU regardless of the compute device: it
        # only aggregates stability and (optionally) writes per-fold edges, and a
        # [Features, 2, Folds, Runs] tensor would otherwise consume huge amounts of
        # VRAM for large parcellations × many folds × many permutations.
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

        # 3. Handle Edges (Features)
        self.n_features = n_features
        # Running sum of selected-edge masks over folds -> stability = sum / n_folds.
        # This is all edge stability needs, and it avoids keeping every fold's mask
        # (the folds axis is what blows up memory for many permutations).
        # Shape: [N_Features, 2, Params, Runs]  (positive/negative only, not "both").
        self.cv_edge_sum = torch.zeros(
            self.n_features, 2, self.dims['params'], self.dims['runs'],
            dtype=torch.float32, device=self.edge_device
        )
        # Per-fold masks are retained only when we need to write edges.npy (the real
        # run). Permutation / inner-CV passes keep only the fold-sum above.
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

    def store_edges(self, param_idx, fold_idx, edges_tensor, run_idx=slice(None)):
        """
        Store the edge masks for the positive and negative networks: move the
        mask to the CPU edge store (freeing VRAM) and accumulate the fold-sum
        used for stability. The per-fold mask itself is retained only when this
        manager was built with store_fold_edges=True.

        Args:
            param_idx: index (int or slice) of the current parameter(s).
            fold_idx: index (int) of the current fold.
            edges_tensor: boolean tensor shaped to match
                          ``cv_edges[:, :, param_idx, fold_idx, run_idx]``,
                          i.e. [Features, 2, (Params,) Runs]. Dimension 1 must
                          be [Positive, Negative].
            run_idx: index (int or slice) of the current run(s)/permutation(s).
        """
        mask = torch.as_tensor(edges_tensor, dtype=torch.bool, device=self.edge_device)

        if self.cv_edges is not None:
            self.cv_edges[:, :, param_idx, fold_idx, run_idx] = mask

        # cv_edge_sum has no folds axis; one fold contributes one mask.
        self.cv_edge_sum[:, :, param_idx, run_idx] += mask.float()

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

        for the chosen hyperparameter of each run — read straight from the CPU
        fold-sum accumulator (``cv_edge_sum``), so this never touches the compute
        device. ``stability_edges.npy`` (``[Nodes, Nodes, 2, Runs]``) is always
        written; the per-fold ``edges.npy`` is written only when this manager
        retained the per-fold masks (the real run — see ``store_fold_edges``).
        Both node×node arrays are built on the CPU.

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
        # cv_edge_sum: [Features, 2, Params, Runs]; paired-index Params × Runs.
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

    def store_predictions(self, y_pred, y_true, fold, test_indices, repeat=0):
        y_pred = y_pred.detach().cpu().numpy().squeeze(-1)
        y_true = torch.as_tensor(y_true).detach().cpu().numpy().reshape(-1)

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

        # Add fold/repeat metadata. ``repeat`` distinguishes the passes of a
        # RepeatedKFold so a subject's predictions can be averaged across repeats
        # before anything is plotted at the individual level (each subject is a
        # test case once per repeat).
        batch_df['fold'] = fold
        batch_df['repeat'] = repeat

        # Reset index to turn MultiIndex levels into columns
        batch_df = batch_df.reset_index()

        self.cv_predictions.append(batch_df)



    def store_network_strengths(self, network_strengths, y_true, fold, test_indices=None, repeat=0):
        y_true = torch.as_tensor(y_true).detach().cpu().numpy().squeeze()
        # Use a list comprehension to build data more concisely. ``sample_index``
        # and ``repeat`` let a subject's network strengths be averaged across the
        # repeats of a RepeatedKFold (see ``store_predictions``).
        data = [
            pd.DataFrame({
                'sample_index': test_indices,
                'y_true': y_true,
                'network_strength': np.squeeze(network_strengths[m][n].cpu().numpy()),
                'model': m,
                'network': n,
                'fold': fold,
                'repeat': repeat,
            })
            # Iterate what the model actually produced: without covariates
            # there is no 'residuals' entry to store.
            for m in network_strengths
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
        Save predictions to CSV, sorted by sample index. This is the single
        writer of ``cv_predictions.csv`` (``calculate_final_cv_results`` only
        concatenates the per-fold frames into ``self.cv_predictions``).
        """
        if isinstance(self.cv_predictions, list):
            if not self.cv_predictions:
                return
            self.cv_predictions = pd.concat(self.cv_predictions, ignore_index=True)

        df = self.cv_predictions.sort_values(by='sample_index')
        df.to_csv(os.path.join(self.results_directory, 'cv_predictions.csv'), index=False)

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

        # The report reads this to know which model rows carry a real number and
        # which are NaN placeholders for variants this run does not define.
        with open(os.path.join(self.results_directory, 'available_models.json'), 'w') as f:
            json.dump(self.available_models, f)

        # Concatenate the per-fold prediction frames into a single DataFrame;
        # save_predictions() (real runs only) is what writes it to disk.
        if self.cv_predictions:
            self.cv_predictions = pd.concat(self.cv_predictions, ignore_index=True)
        return df_agg

    def aggregate_inner_folds(self):
        """
        Compute the increment model (full - covariates) across all inner folds.

        The per-fold means are read directly by ``find_best_params``, so nothing
        else needs to be aggregated or saved here.
        """
        self.results[:, Models.increment] = self.results[:, Models.full] - self.results[:, Models.covariates]

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

