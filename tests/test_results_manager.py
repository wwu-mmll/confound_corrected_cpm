import os
import pytest
import numpy as np
import pandas as pd
import torch

from cccpm.results_manager import ResultsManager
from cccpm.constants import (INCREMENTABLE_METRICS, Metrics, Models, Networks,
                             TaskType)


class TestResultsManagerInit:

    def test_results_tensor_shape(self, tmp_path):
        mgr = ResultsManager(
            output_dir=str(tmp_path), n_runs=3, n_folds=5, n_features=10, n_params=2
        )
        expected = (len(Metrics), len(Models), len(Networks), 2, 5, 3)
        assert mgr.results.shape == expected

    def test_edges_tensor_shape_with_params(self, tmp_path):
        mgr = ResultsManager(
            output_dir=str(tmp_path), n_runs=1, n_folds=4, n_features=10, n_params=3
        )
        assert mgr.cv_edges.shape == (10, 2, 3, 4, 1)
        # bool, not float: this tensor is the largest allocation in a big run.
        assert mgr.cv_edges.dtype == torch.bool


class TestStoreAndRetrieve:
    def test_store_metrics(self, tmp_path):
        mgr = ResultsManager(
            output_dir=str(tmp_path), n_runs=1, n_folds=2, n_features=5
        )
        metrics = torch.randn(len(Metrics), len(Models), len(Networks), 1)
        mgr.store_metrics(param_idx=0, fold_idx=0, metrics_tensor=metrics)

        stored = mgr.results[:, :, :, 0, 0, :]
        assert torch.allclose(stored, metrics)

    def test_store_edges_and_calculate_stability(self, tmp_path):
        n_features = 6
        n_folds = 3
        mgr = ResultsManager(
            output_dir=str(tmp_path), n_runs=1, n_folds=n_folds, n_features=n_features
        )

        # 0,1 positive every fold; 2 positive only in fold 0; 3 negative every fold.
        for fold in range(n_folds):
            edges = torch.zeros(n_features, 2, 1, dtype=torch.bool)
            edges[0, Networks.positive, 0] = True
            edges[1, Networks.positive, 0] = True
            if fold == 0:
                edges[2, Networks.positive, 0] = True
            edges[3, Networks.negative, 0] = True
            mgr.store_edges(param_idx=0, fold_idx=fold, edges_tensor=edges)

        stability = mgr.calculate_edge_stability(write=True)
        assert stability.shape == (n_features, 2, 1)

        assert stability[0, Networks.positive, 0].item() == pytest.approx(1.0)
        assert stability[1, Networks.positive, 0].item() == pytest.approx(1.0)
        assert stability[2, Networks.positive, 0].item() == pytest.approx(1/3)
        assert stability[3, Networks.negative, 0].item() == pytest.approx(1.0)

        assert os.path.exists(os.path.join(str(tmp_path), 'edges.npy'))
        assert os.path.exists(os.path.join(str(tmp_path), 'stability_edges.npy'))

    def test_stability_without_fold_edges(self, tmp_path):
        """store_fold_edges=False (permutation/inner-CV): no per-fold masks kept
        -- a persistent [Features, 2, Folds, Runs] tensor was the CUDA-OOM
        source for large parcellations x many folds x many permutations."""
        n_features, n_folds = 6, 3
        mgr = ResultsManager(
            output_dir=str(tmp_path), n_runs=1, n_folds=n_folds,
            n_features=n_features, store_fold_edges=False,
        )
        assert mgr.cv_edges is None
        assert mgr.cv_edge_sum.device == torch.device('cpu')

        for fold in range(n_folds):
            edges = torch.zeros(n_features, 2, 1, dtype=torch.bool)
            edges[0, Networks.positive, 0] = True  # all folds
            if fold == 0:
                edges[2, Networks.positive, 0] = True  # one fold only
            mgr.store_edges(param_idx=0, fold_idx=fold, edges_tensor=edges)

        stability = mgr.calculate_edge_stability(write=True)
        assert stability[0, Networks.positive, 0].item() == pytest.approx(1.0)
        assert stability[2, Networks.positive, 0].item() == pytest.approx(1 / 3)

        assert os.path.exists(os.path.join(str(tmp_path), 'stability_edges.npy'))
        assert not os.path.exists(os.path.join(str(tmp_path), 'edges.npy'))

    def test_cv_edges_and_edge_sum_stay_on_cpu_regardless_of_compute_device(self, tmp_path):
        """Edge bookkeeping must never be on the compute device -- that
        persistent allocation was the actual CUDA OOM source."""
        mgr = ResultsManager(
            output_dir=str(tmp_path), n_runs=1, n_folds=2, n_features=6,
            device=torch.device('cpu'),  # can't assume CUDA is present in CI
        )
        assert mgr.edge_device == torch.device('cpu')
        assert mgr.cv_edge_sum.device == torch.device('cpu')
        assert mgr.cv_edges.device == torch.device('cpu')

class TestCalculateFinalCVResults:
    def test_saves_csv_files(self, tmp_path):
        mgr = ResultsManager(
            output_dir=str(tmp_path), n_runs=1, n_folds=5, n_features=3
        )
        for fold in range(5):
            metrics = torch.randn(len(Metrics), len(Models), len(Networks), 1)
            mgr.store_metrics(param_idx=0, fold_idx=fold, metrics_tensor=metrics)

        mgr.calculate_final_cv_results()

        assert os.path.exists(os.path.join(str(tmp_path), 'cv_results_full.csv'))
        assert os.path.exists(os.path.join(str(tmp_path), 'cv_results_summary.csv'))

    def test_regression_filters_metrics(self, tmp_path):
        mgr = ResultsManager(
            output_dir=str(tmp_path), n_runs=1, n_folds=3, n_features=3
        )
        for fold in range(3):
            metrics = torch.randn(len(Metrics), len(Models), len(Networks), 1)
            mgr.store_metrics(param_idx=0, fold_idx=fold, metrics_tensor=metrics)

        mgr.calculate_final_cv_results(task_type=TaskType.regression)

        df_full = pd.read_csv(os.path.join(str(tmp_path), 'cv_results_full.csv'))
        assert 'pearson_score' in df_full.columns
        assert 'mean_squared_error' in df_full.columns
        assert 'accuracy' not in df_full.columns
        assert 'roc_auc' not in df_full.columns

    def test_classification_filters_metrics(self, tmp_path):
        mgr = ResultsManager(
            output_dir=str(tmp_path), n_runs=1, n_folds=3, n_features=3
        )
        for fold in range(3):
            metrics = torch.randn(len(Metrics), len(Models), len(Networks), 1)
            mgr.store_metrics(param_idx=0, fold_idx=fold, metrics_tensor=metrics)

        mgr.calculate_final_cv_results(task_type=TaskType.classification)

        df_full = pd.read_csv(os.path.join(str(tmp_path), 'cv_results_full.csv'))
        assert 'accuracy' in df_full.columns
        assert 'roc_auc' in df_full.columns
        assert 'pearson_score' not in df_full.columns
        assert 'mean_squared_error' not in df_full.columns


class TestLoadCVResults:
    def test_load_cv_results_filters_mean(self, tmp_path):
        model_names = [m.name for m in Models]
        net_names = [n.name for n in Networks]
        metrics = ['pearson_score', 'mean_squared_error']

        agg_index = pd.MultiIndex.from_product(
            [model_names, net_names, [0]],
            names=['model', 'network', 'run']
        )
        df_mean = pd.DataFrame(
            np.random.rand(len(agg_index), len(metrics)),
            index=agg_index, columns=metrics
        )
        df_std = pd.DataFrame(
            np.random.rand(len(agg_index), len(metrics)),
            index=agg_index, columns=metrics
        )
        df_agg = pd.concat([df_mean, df_std], axis=1, keys=['mean', 'std'])
        df_agg = df_agg.swaplevel(0, 1, axis=1).sort_index(axis=1)

        df_agg.to_csv(os.path.join(str(tmp_path), 'cv_results_summary.csv'))

        loaded = ResultsManager.load_cv_results(str(tmp_path))

        assert set(loaded.columns) == set(metrics)
        assert loaded.shape[0] == len(agg_index)



class TestIncrementIsSuppressedWhereMeaningless:
    """increment = full - covariates is only a statistic for metrics where
    differences are standard; see constants.INCREMENTABLE_METRICS."""

    def _manager_with_metrics(self, tmp_path, task_type=TaskType.regression):
        mgr = ResultsManager(
            output_dir=str(tmp_path), n_runs=1, n_folds=3, n_features=3
        )
        for fold in range(3):
            metrics = torch.rand(len(Metrics), len(Models), len(Networks), 1) + 0.5
            mgr.store_metrics(param_idx=0, fold_idx=fold, metrics_tensor=metrics)
        mgr.calculate_final_cv_results(task_type=task_type)
        return mgr

    def test_pearson_increment_is_nan(self, tmp_path):
        """Comparing two correlations needs Fisher z or Steiger's test, not subtraction."""
        mgr = self._manager_with_metrics(tmp_path)
        assert torch.isnan(mgr.results[Metrics.pearson_score, Models.increment]).all()

    def test_f1_increment_is_nan(self, tmp_path):
        mgr = self._manager_with_metrics(tmp_path, TaskType.classification)
        assert torch.isnan(mgr.results[Metrics.f1_score, Models.increment]).all()

    def test_interpretable_increments_survive(self, tmp_path):
        """Iterates INCREMENTABLE_METRICS itself (not a hardcoded list) so a
        metric added to it is covered automatically; also pins the set's
        membership, since reading the constant can't catch a metric removed."""
        assert {m.name for m in INCREMENTABLE_METRICS} == {
            "explained_variance_score", "mean_squared_error",
            "mean_absolute_error", "accuracy", "balanced_accuracy", "roc_auc",
        }, "INCREMENTABLE_METRICS changed -- is the increment still a statistic?"

        mgr = self._manager_with_metrics(tmp_path)
        for metric in INCREMENTABLE_METRICS:
            expected = (mgr.results[metric, Models.full]
                        - mgr.results[metric, Models.covariates])
            assert torch.allclose(mgr.results[metric, Models.increment],
                                  expected), metric.name

    def test_suppression_does_not_touch_the_other_models(self, tmp_path):
        mgr = self._manager_with_metrics(tmp_path)
        for model in (Models.connectome, Models.covariates, Models.full):
            assert torch.isfinite(
                mgr.results[Metrics.pearson_score, model]).all(), model.name
