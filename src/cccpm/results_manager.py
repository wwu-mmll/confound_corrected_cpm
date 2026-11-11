"""`ResultsManager` is a preallocated store: writes per-fold edges/metrics/predictions into fixed-shape tensors, then aggregates to the CSV/npy files the report reads."""
import os
import json
from typing import Union

import numpy as np
import pandas as pd
import torch

from cccpm.constants import (INCREMENTABLE_METRICS, Metrics, Models, Networks,
                             TaskType, get_metrics_for_task)
from cccpm.connectome import vector_to_matrix_tensor_version


class ResultsManager:
    """Handles the aggregation, formatting, and saving of CV results."""

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
        """Preallocates the result tensors for one run.

        Args:
            store_fold_edges: Whether to retain each fold's edge mask (needed
                to write ``edges.npy`` for a real run).
            available_models: Model variants this run defines; undefined ones
                stay NaN-filled in the (always full-size) results tensor.
                Defaults to every :class:`~cccpm.constants.Models` entry.
        """
        self.results_directory = output_dir
        self.is_inner_cv = is_inner_cv
        self.available_models = (list(available_models) if available_models is not None
                                 else [m.name for m in Models])
        self.device = device
        # CPU regardless of compute device: a [Features, 2, Folds, Runs] edge
        # tensor would otherwise consume huge VRAM for large parcellations x
        # many folds x many permutations.
        self.edge_device = torch.device('cpu')
        self.store_fold_edges = store_fold_edges

        self.dims = {
            'models': len(Models),
            'networks': len(Networks),
            'params': n_params if n_params is not None else 1,
            'folds': n_folds,
            'metrics': len(Metrics),
            'runs': n_runs
        }

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

        self.n_features = n_features
        # Running sum of selected-edge masks over folds; stability = sum / n_folds.
        # Shape: [N_Features, 2, Params, Runs] (positive/negative only, not "both").
        self.cv_edge_sum = torch.zeros(
            self.n_features, 2, self.dims['params'], self.dims['runs'],
            dtype=torch.float32, device=self.edge_device
        )
        # Shape: [N_Features, 2, Params, Folds, Runs].
        self.cv_edges = None
        if store_fold_edges:
            self.cv_edges = torch.zeros(
                self.n_features, 2, self.dims['params'],
                self.dims['folds'], self.dims['runs'],
                dtype=torch.bool, device=self.edge_device
            )

        self.cv_predictions = []
        self.cv_network_strengths = pd.DataFrame()
        self.agg_results = None

    def store_edges(self, param_idx, fold_idx, edges_tensor, run_idx=slice(None)):
        """Moves the [Features, 2, (Params,) Runs] edge mask to the CPU store and accumulates the fold-sum for stability."""
        mask = torch.as_tensor(edges_tensor, dtype=torch.bool, device=self.edge_device)

        if self.cv_edges is not None:
            self.cv_edges[:, :, param_idx, fold_idx, run_idx] = mask

        self.cv_edge_sum[:, :, param_idx, run_idx] += mask.float()

    def store_metrics(self, param_idx, fold_idx, metrics_tensor: torch.Tensor, run_idx=slice(None)):
        """Stores a [Metrics, Models, Networks, (Params,) (Folds,) Runs] metrics batch at [:, :, :, param_idx, fold_idx, run_idx]."""
        self.results[:, :, :, param_idx, fold_idx, run_idx] = metrics_tensor.to(self.results.device)

    def calculate_edge_stability(self, write: bool = True, best_param_id=None):
        """Returns edge stability [Features, 2, Runs] for the chosen hyperparameter of each run; writes stability_edges.npy (and edges.npy if store_fold_edges) when `write`.

        Args:
            best_param_id: None (param index 0 for every run), a single
                int/0-d tensor (same param for every run), or an array-like
                of length n_runs (one winning param per run).
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

        # Fraction of folds selecting each edge, for the chosen param per run
        # (paired-index Params x Runs). Result: [Features, 2, Runs].
        edge_stability = self.cv_edge_sum[:, :, best_param_id, run_indices] / self.dims['folds']

        if write:
            np.save(os.path.join(self.results_directory, 'stability_edges.npy'),
                    vector_to_matrix_tensor_version(edge_stability, dim=0).numpy())
            if self.cv_edges is not None:
                # Advanced (non-adjacent) indices lead -> [Runs, Features, 2,
                # Folds]; permute back to [Features, 2, Folds, Runs].
                selected_edges = self.cv_edges[:, :, best_param_id, :, run_indices]
                selected_edges = selected_edges.permute(1, 2, 3, 0)
                np.save(os.path.join(self.results_directory, 'edges.npy'),
                        vector_to_matrix_tensor_version(selected_edges, dim=0).float().numpy())
        return edge_stability

    def store_predictions(self, y_pred, y_true, fold, test_indices, repeat=0):
        """Appends one fold's predictions, for every model/network combination, to the prediction buffer."""
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

        batch_df = pd.DataFrame({
            'y_pred': flat_preds,
            'y_true': flat_true
        }, index=multi_index)

        # `repeat` lets a subject's predictions be averaged across the
        # repeats of a RepeatedKFold before individual-level plotting.
        batch_df['fold'] = fold
        batch_df['repeat'] = repeat

        batch_df = batch_df.reset_index()

        self.cv_predictions.append(batch_df)



    def store_network_strengths(self, network_strengths, y_true, fold, test_indices=None, repeat=0):
        """Appends one fold's per-model, per-network strengths (mapping model name -> {'positive','negative'} tensors) to the strengths buffer."""
        y_true = torch.as_tensor(y_true).detach().cpu().numpy().squeeze()
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
            for m in network_strengths
            for n in ['positive', 'negative']
        ]

        self.cv_network_strengths = pd.concat([self.cv_network_strengths] + data, ignore_index=True)

    @staticmethod
    def load_cv_results(folder):
        """Loads the mean-metric columns of cv_results_summary.csv as a DataFrame."""
        results = pd.read_csv(os.path.join(folder, 'cv_results_summary.csv'), header=[0, 1], index_col=[0, 1, 2])
        results = results.loc[:, results.columns.get_level_values(1) == 'mean']
        results.columns = results.columns.droplevel(1)
        return results

    def save_predictions(self):
        """Saves predictions to cv_predictions.csv, sorted by sample index."""
        if isinstance(self.cv_predictions, list):
            if not self.cv_predictions:
                return
            self.cv_predictions = pd.concat(self.cv_predictions, ignore_index=True)

        df = self.cv_predictions.sort_values(by='sample_index')
        df.to_csv(os.path.join(self.results_directory, 'cv_predictions.csv'), index=False)

    def save_network_strengths(self):
        """Saves network strengths to CSV."""
        self.cv_network_strengths.to_csv(os.path.join(self.results_directory, 'cv_network_strengths.csv'))

    def calculate_final_cv_results(self, task_type: TaskType = TaskType.regression):
        """Computes the increment model, writes cv_results_full.csv, cv_results_summary.csv and available_models.json; returns the summary DataFrame."""
        self.results[:, Models.increment] = self.results[:, Models.full] - self.results[:, Models.covariates]
        self._suppress_uninterpretable_increments()

        # Shape: [Metrics, Models, Networks, Params, Folds, Runs]
        data = self.results.cpu()

        n_metrics, n_models, n_nets, _, n_folds, n_runs = data.shape

        model_names = [m.name for m in Models]
        net_names = [n.name for n in Networks]
        all_metrics = [m.name for m in Metrics]
        relevant_metrics = [m.name for m in get_metrics_for_task(task_type)]
        fold_indices = range(n_folds)
        run_indices = range(n_runs)

        # Permute to [Models, Networks, Params, Folds, Runs, Metrics]
        raw_tensor = data.permute(1, 2, 3, 4, 5, 0)
        raw_matrix = raw_tensor.reshape(-1, n_metrics).numpy()

        raw_index = pd.MultiIndex.from_product(
            [model_names, net_names, fold_indices, run_indices],
            names=['model', 'network', 'fold', 'run']
        )

        df_raw = pd.DataFrame(raw_matrix, index=raw_index, columns=all_metrics)
        df_raw = df_raw[relevant_metrics]
        df_raw.to_csv(os.path.join(self.results_directory, 'cv_results_full.csv'))

        # Shape after mean/std: [Metrics, Models, Networks, Params, Runs]
        means = torch.mean(data, dim=4)
        stds = torch.std(data, dim=4)

        # Permute to [Models, Networks, Params, Runs, Metrics]
        means = means.permute(1, 2, 3, 4, 0)
        stds = stds.permute(1, 2, 3, 4, 0)

        means_flat = means.reshape(-1, n_metrics).numpy()
        stds_flat = stds.reshape(-1, n_metrics).numpy()

        agg_index = pd.MultiIndex.from_product(
            [model_names, net_names, run_indices],
            names=['model', 'network', 'run']
        )

        df_mean = pd.DataFrame(means_flat, index=agg_index, columns=all_metrics)[relevant_metrics]
        df_std = pd.DataFrame(stds_flat, index=agg_index, columns=all_metrics)[relevant_metrics]

        # Columns become Metric -> (Mean, Std).
        df_agg = pd.concat([df_mean, df_std], axis=1, keys=['mean', 'std'])
        df_agg = df_agg.swaplevel(0, 1, axis=1).sort_index(axis=1)

        self.agg_results = df_agg.copy()
        df_agg.to_csv(os.path.join(self.results_directory, 'cv_results_summary.csv'), float_format='%.4f')

        with open(os.path.join(self.results_directory, 'available_models.json'), 'w', encoding='utf-8') as f:
            json.dump(self.available_models, f)

        if self.cv_predictions:
            self.cv_predictions = pd.concat(self.cv_predictions, ignore_index=True)
        return df_agg

    def _suppress_uninterpretable_increments(self):
        """NaNs out `increment` for metrics not in `constants.INCREMENTABLE_METRICS` (e.g. Pearson r needs Fisher z/Steiger's test, not subtraction)."""
        for metric in Metrics:
            if metric not in INCREMENTABLE_METRICS:
                self.results[metric, Models.increment] = float('nan')

    def aggregate_inner_folds(self):
        """Computes the increment model (full - covariates); `find_best_params` reads the per-fold means directly."""
        self.results[:, Models.increment] = self.results[:, Models.full] - self.results[:, Models.covariates]

    def find_best_params(self, task_type=TaskType.regression):
        """Returns, per run, the index of the hyperparameter with the best mean connectome-model score (Pearson r or balanced accuracy), shape [Runs]."""
        if task_type == TaskType.classification:
            metric = Metrics.balanced_accuracy
        else:
            metric = Metrics.pearson_score

        # Shape: [Params, Folds, Runs]
        scores_slice = self.results[
            metric,
            Models.connectome,
            Networks.both
        ]

        mean_scores = torch.mean(scores_slice, dim=1)
        best_param_idx = torch.argmax(mean_scores, dim=0)
        return best_param_idx

