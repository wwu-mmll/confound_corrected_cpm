import os
import pandas as pd
import numpy as np
import torch
from glob import glob
from typing import Union
from cccpm.models import NetworkDict, ModelDict
from cccpm.utils import vector_to_upper_triangular_matrix
from cccpm.scoring import regression_metrics


class ResultsManager:
    """
    Optimized ResultsManager.

    Improvements:
    1. Buffers metrics in memory and consolidates ONCE at the end (orders of magnitude faster).
    2. Accumulates edge stability iteratively to prevent RAM explosion with high n_perms.
    3. Saves numeric results to Parquet instead of CSV for efficiency.
    """

    def __init__(self, output_dir: Union[str, None], n_perms: int, n_folds: int, n_features: int, n_params: int = None,
                 is_inner_cv: bool = False):
        self.n_perms = n_perms
        self.is_inner_cv = is_inner_cv
        self.results_directory = output_dir
        self.n_folds = n_folds
        self.n_features = n_features
        self.n_params = n_params if n_params is not None else 1

        if self.results_directory:
            os.makedirs(self.results_directory, exist_ok=True)

        # -- OPTIMIZATION 1: Metric Buffer --
        # Instead of a growing DataFrame, we store lightweight dicts here.
        self.metrics_buffer = []

        # -- OPTIMIZATION 2: Iterative Edge Storage --
        # Storing binary masks for 1000 permutations x 5 folds x 35k edges = ~175GB RAM.
        # Instead, we just accumulate the SUM (stability) on the fly.
        # Shape: [n_folds, n_features, n_params] (Permutations are summed out)
        self.edge_counts = {
            'positive': np.zeros((n_folds, n_features, self.n_params), dtype=np.float32),
            'negative': np.zeros((n_folds, n_features, self.n_params), dtype=np.float32)
        }

        # Keep track of predictions separately if needed (optional, can be large)
        self.prediction_buffer = []

        self.agg_results = None

    def store_edges(self, edges: dict, fold: int, param_id: int = 0):
        """
        Accumulates edge counts (stability) instead of storing every raw mask.
        Input `edges` is expected to be [n_features, n_perms].
        """
        for sign in ['positive', 'negative']:
            # edges[sign] shape is [Features, Perms]
            # We sum across permutations (axis 1) to get stability for this fold
            # If edges is just [Features] (single perm), sum is just the value.

            edge_data = edges[sign]

            # Handle potential shape mismatch if n_perms=1 vs n_perms>1
            if hasattr(edge_data, 'shape') and len(edge_data.shape) > 1:
                # Sum across permutations (columns)
                # We normalize by n_perms later.
                perm_sum = np.sum(edge_data, axis=1)
            else:
                perm_sum = edge_data

            # Add to the accumulator
            self.edge_counts[sign][fold, :, param_id] += perm_sum

    def store_metrics(self, metrics, params, fold, param_id):
        """
        Fast storage: just appends data to a buffer.
        """
        # Detach tensors to CPU numpy to save GPU memory
        clean_metrics = self._detach_recursive(metrics)

        self.metrics_buffer.append({
            'data': clean_metrics,
            'params': params,
            'fold': fold,
            'param_id': param_id
        })

    def store_predictions(self, y_pred, y_true, params, fold, param_id, test_indices):
        """
        Buffer predictions efficiently.
        """
        # Convert y_pred dict to a more compact form if necessary, or just store raw
        clean_preds = self._detach_recursive(y_pred)

        # We store just the necessary data to reconstruct the DF later
        self.prediction_buffer.append({
            'y_pred': clean_preds,
            'y_true': y_true if isinstance(y_true, np.ndarray) else y_true.cpu().numpy(),
            'test_indices': test_indices,
            'fold': fold,
            'param_id': param_id,
            'params': params
        })

    def aggregate_inner_folds(self):
        """
        Consolidates all buffered metrics into a single DataFrame and saves it.
        """
        if not self.metrics_buffer:
            print("Warning: No metrics to aggregate.")
            return

        print(f"Aggregating results from {len(self.metrics_buffer)} folds/params...")

        # 1. CONSOLIDATE METRICS
        all_dfs = []
        for entry in self.metrics_buffer:
            # Convert the nested metric dict into a wide DataFrame
            # Index: 0..N_perms
            df_batch = self._metrics_to_df(entry['data'])

            # Add metadata columns
            df_batch['fold'] = entry['fold']
            df_batch['param_id'] = entry['param_id']
            # We can store the full params dict as a string or separate cols
            # For simplicity, let's store param_id which links to the config

            # Ensure index represents permutations
            df_batch.index.name = 'permutation'
            df_batch.reset_index(inplace=True)

            all_dfs.append(df_batch)

        # Concatenate once (Fast)
        self.cv_results = pd.concat(all_dfs, ignore_index=True)

        # 2. CALCULATE AGGREGATED STATS (Mean/Std across folds)
        # Group by: param_id, network, model, permutation
        # (Averaging over folds)

        # This creates the main results table
        # We use a compact list of grouping keys
        group_cols = ['param_id', 'network', 'model']

        # If n_perms > 1, we usually want mean across folds FOR EACH permutation first
        # But for 'agg_results' (summary), we usually average across permutations too?
        # The original code grouped by ['network', 'param_id', 'model']

        numeric_cols = self.cv_results.select_dtypes(include=np.number).columns
        self.agg_results = self.cv_results.groupby(group_cols)[numeric_cols].agg(['mean', 'std'])

        # 3. SAVE TO DISK (Parquet is best for this size)
        if self.results_directory:
            # Save raw detailed results (Very large) efficiently
            self.cv_results.to_parquet(os.path.join(self.results_directory, 'cv_results.parquet'))

            # Save summary (CSV is fine for summary)
            self.agg_results.to_csv(os.path.join(self.results_directory, 'cv_results_summary.csv'))

        return

    def find_best_params(self):
        """
        Finds best param_id based on mean spearman score across folds.
        """
        # agg_results has MultiIndex columns (metric, stat) -> ('spearman_score', 'mean')

        # Filter for the target metric
        target_score = self.agg_results[('spearman_score', 'mean')]

        # We specifically look at 'both' network and 'full' model as per your logic
        # target_score index is (param_id, network, model)

        # Slicing MultiIndex safely
        try:
            # Get the slice for all param_ids, specific network/model
            subset = target_score.xs(('both', 'full'), level=('network', 'model'))
            best_param_id = subset.idxmax()  # Returns the param_id

            # Retrieve the actual config dict (Need to find it in buffer)
            # Find first entry with this param_id
            best_params = next(item['params'] for item in self.metrics_buffer if item['param_id'] == best_param_id)

            return best_params, best_param_id

        except KeyError:
            print("Warning: Could not find ('both', 'full') model in results. Defaulting to first param.")
            return self.metrics_buffer[0]['params'], 0

    def calculate_edge_stability(self, write: bool = True, best_param_id: int = None):
        """
        Calculates stability from the accumulated edge counts.
        """
        stability_results = {}

        for sign in ['positive', 'negative']:
            # edge_counts is [n_folds, n_features, n_params]
            # It already contains the SUM across permutations.

            # 1. Select the params
            if best_param_id is not None:
                counts = self.edge_counts[sign][:, :, best_param_id]  # [n_folds, n_features]
            else:
                # Average across all params? Or just take index 0
                counts = self.edge_counts[sign][:, :, 0]

            # 2. Total Observations = n_folds * n_perms
            total_obs = self.n_folds * self.n_perms

            # 3. Sum across folds and divide by total
            # Result: [n_features] (Stability 0.0 to 1.0)
            stability_score = np.sum(counts, axis=0) / total_obs

            stability_results[sign] = stability_score

            if write and self.results_directory:
                # Save stability vector
                np.save(os.path.join(self.results_directory, f'stability_{sign}_edges.npy'),
                        vector_to_upper_triangular_matrix(stability_score))

        return stability_results

    # --- Helpers ---

    def _metrics_to_df(self, metrics_dict):
        """Flattens nested dictionary [model][net][metric] -> DataFrame"""
        data = []
        models = list(metrics_dict.keys())

        for model in models:
            nets = metrics_dict[model].keys()
            for net in nets:
                # This dict contains {metric_name: tensor_array}
                scores = metrics_dict[model][net]

                # Convert to DataFrame (Columns=Metrics, Index=Permutations)
                # We assume all metrics have same shape [n_perms]
                df_subset = pd.DataFrame(scores)
                df_subset['model'] = model
                df_subset['network'] = net
                data.append(df_subset)

        return pd.concat(data, ignore_index=False)

    def _detach_recursive(self, d):
        """Recursively converts Tensors to Numpy and moves to CPU."""
        if isinstance(d, dict):
            return {k: self._detach_recursive(v) for k, v in d.items()}
        elif isinstance(d, torch.Tensor):
            return d.detach().cpu().numpy()
        elif isinstance(d, list):
            return [self._detach_recursive(v) for v in d]
        return d