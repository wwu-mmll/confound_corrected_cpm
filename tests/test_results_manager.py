import os
import pytest
import numpy as np
import pandas as pd
import torch

from cccpm.results_manager import ResultsManager
from cccpm.constants import Networks, Models, Metrics, TaskType


class TestResultsManagerInit:
    def test_constructor(self, tmp_path):
        mgr = ResultsManager(
            output_dir=str(tmp_path), n_runs=1, n_folds=5, n_features=10
        )
        assert mgr.results_directory == str(tmp_path)
        assert mgr.dims['folds'] == 5
        assert mgr.dims['runs'] == 1
        assert mgr.dims['params'] == 1

    def test_results_tensor_shape(self, tmp_path):
        mgr = ResultsManager(
            output_dir=str(tmp_path), n_runs=3, n_folds=5, n_features=10, n_params=2
        )
        expected = (len(Metrics), len(Models), len(Networks), 2, 5, 3)
        assert mgr.results.shape == expected

    def test_edges_tensor_shape(self, tmp_path):
        mgr = ResultsManager(
            output_dir=str(tmp_path), n_runs=2, n_folds=3, n_features=6
        )
        # [N_features, 2(pos/neg), params, folds, runs]
        assert mgr.cv_edges.shape == (6, 2, 1, 3, 2)
        assert mgr.cv_edges.dtype == torch.bool

    def test_edges_tensor_shape_with_params(self, tmp_path):
        mgr = ResultsManager(
            output_dir=str(tmp_path), n_runs=1, n_folds=4, n_features=10, n_params=3
        )
        assert mgr.cv_edges.shape == (10, 2, 3, 4, 1)


class TestStoreAndRetrieve:
    def test_store_metrics(self, tmp_path):
        mgr = ResultsManager(
            output_dir=str(tmp_path), n_runs=1, n_folds=2, n_features=5
        )
        # Create a metrics tensor [Metrics, Models, Networks, Runs]
        metrics = torch.randn(len(Metrics), len(Models), len(Networks), 1)
        mgr.store_metrics(param_idx=0, fold_idx=0, metrics_tensor=metrics)

        # Verify it was stored in the right place
        stored = mgr.results[:, :, :, 0, 0, :]
        assert torch.allclose(stored, metrics)

    def test_store_edges(self, tmp_path):
        mgr = ResultsManager(
            output_dir=str(tmp_path), n_runs=1, n_folds=2, n_features=6
        )
        # Create edges tensor [Features, 2, Runs]
        edges = torch.zeros(6, 2, 1, dtype=torch.bool)
        edges[0, Networks.positive, 0] = True
        edges[2, Networks.positive, 0] = True
        edges[1, Networks.negative, 0] = True

        mgr.store_edges(param_idx=0, fold_idx=0, edges_tensor=edges)

        assert mgr.cv_edges[0, Networks.positive, 0, 0, 0] == True
        assert mgr.cv_edges[2, Networks.positive, 0, 0, 0] == True
        assert mgr.cv_edges[1, Networks.negative, 0, 0, 0] == True
        assert mgr.cv_edges[3, Networks.positive, 0, 0, 0] == False

    def test_store_edges_and_calculate_stability(self, tmp_path):
        n_features = 6
        n_folds = 3
        mgr = ResultsManager(
            output_dir=str(tmp_path), n_runs=1, n_folds=n_folds, n_features=n_features
        )

        # Store edges for each fold — some overlap, some don't
        for fold in range(n_folds):
            edges = torch.zeros(n_features, 2, 1, dtype=torch.bool)
            # Features 0 and 1 always positive
            edges[0, Networks.positive, 0] = True
            edges[1, Networks.positive, 0] = True
            # Feature 2 positive only in fold 0
            if fold == 0:
                edges[2, Networks.positive, 0] = True
            # Feature 3 always negative
            edges[3, Networks.negative, 0] = True
            mgr.store_edges(param_idx=0, fold_idx=fold, edges_tensor=edges)

        stability = mgr.calculate_edge_stability(write=True)

        # stability shape: [N_Features, 2, Runs]
        assert stability.shape == (n_features, 2, 1)

        # Features 0,1 should have stability 1.0 (selected in all 3 folds)
        assert stability[0, Networks.positive, 0].item() == pytest.approx(1.0)
        assert stability[1, Networks.positive, 0].item() == pytest.approx(1.0)

        # Feature 2 positive stability = 1/3
        assert stability[2, Networks.positive, 0].item() == pytest.approx(1/3)

        # Feature 3 negative stability = 1.0
        assert stability[3, Networks.negative, 0].item() == pytest.approx(1.0)

        # Verify files were written
        assert os.path.exists(os.path.join(str(tmp_path), 'edges.npy'))
        assert os.path.exists(os.path.join(str(tmp_path), 'stability_edges.npy'))

    def test_stability_without_fold_edges(self, tmp_path):
        """With store_fold_edges=False (permutation / inner-CV passes) stability
        is still computed correctly from the CPU fold-sum accumulator, but no
        per-fold masks are kept (this is the CUDA-OOM fix: a persistent
        [Features, 2, Folds, Runs] tensor on the compute device was the source
        of the OOM for large parcellations × many folds × many permutations)
        and edges.npy is not written (only stability_edges.npy)."""
        n_features, n_folds = 6, 3
        mgr = ResultsManager(
            output_dir=str(tmp_path), n_runs=1, n_folds=n_folds,
            n_features=n_features, store_fold_edges=False,
        )
        assert mgr.cv_edges is None  # per-fold masks not retained
        assert mgr.cv_edge_sum.device == torch.device('cpu')

        for fold in range(n_folds):
            edges = torch.zeros(n_features, 2, 1, dtype=torch.bool)
            edges[0, Networks.positive, 0] = True          # all folds
            if fold == 0:
                edges[2, Networks.positive, 0] = True      # one fold only
            mgr.store_edges(param_idx=0, fold_idx=fold, edges_tensor=edges)

        stability = mgr.calculate_edge_stability(write=True)
        assert stability[0, Networks.positive, 0].item() == pytest.approx(1.0)
        assert stability[2, Networks.positive, 0].item() == pytest.approx(1 / 3)

        assert os.path.exists(os.path.join(str(tmp_path), 'stability_edges.npy'))
        assert not os.path.exists(os.path.join(str(tmp_path), 'edges.npy'))

    def test_cv_edges_and_edge_sum_stay_on_cpu_regardless_of_compute_device(self, tmp_path):
        """Edge bookkeeping must never be allocated on the compute device --
        that persistent allocation was the actual CUDA OOM source."""
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
        # Store some random metrics
        for fold in range(5):
            metrics = torch.randn(len(Metrics), len(Models), len(Networks), 1)
            mgr.store_metrics(param_idx=0, fold_idx=fold, metrics_tensor=metrics)

        mgr.calculate_final_cv_results()

        assert os.path.exists(os.path.join(str(tmp_path), 'cv_results_full.csv'))
        assert os.path.exists(os.path.join(str(tmp_path), 'cv_results_summary.csv'))

    def test_agg_results_populated(self, tmp_path):
        mgr = ResultsManager(
            output_dir=str(tmp_path), n_runs=1, n_folds=3, n_features=3
        )
        for fold in range(3):
            metrics = torch.randn(len(Metrics), len(Models), len(Networks), 1)
            mgr.store_metrics(param_idx=0, fold_idx=fold, metrics_tensor=metrics)

        mgr.calculate_final_cv_results()
        assert mgr.agg_results is not None
        assert isinstance(mgr.agg_results, pd.DataFrame)

    def test_regression_filters_metrics(self, tmp_path):
        """Regression task should only output regression metrics."""
        mgr = ResultsManager(
            output_dir=str(tmp_path), n_runs=1, n_folds=3, n_features=3
        )
        for fold in range(3):
            metrics = torch.randn(len(Metrics), len(Models), len(Networks), 1)
            mgr.store_metrics(param_idx=0, fold_idx=fold, metrics_tensor=metrics)

        mgr.calculate_final_cv_results(task_type=TaskType.regression)

        df_full = pd.read_csv(os.path.join(str(tmp_path), 'cv_results_full.csv'))
        # Should contain regression metrics but not classification
        assert 'pearson_score' in df_full.columns
        assert 'mean_squared_error' in df_full.columns
        assert 'accuracy' not in df_full.columns
        assert 'roc_auc' not in df_full.columns

    def test_classification_filters_metrics(self, tmp_path):
        """Classification task should only output classification metrics."""
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

    def test_increment_computed(self, tmp_path):
        """Test that increment = full - covariates is computed correctly."""
        mgr = ResultsManager(
            output_dir=str(tmp_path), n_runs=1, n_folds=2, n_features=3
        )
        for fold in range(2):
            metrics = torch.randn(len(Metrics), len(Models), len(Networks), 1)
            mgr.store_metrics(param_idx=0, fold_idx=fold, metrics_tensor=metrics)

        mgr.calculate_final_cv_results()

        expected = mgr.results[:, Models.full] - mgr.results[:, Models.covariates]
        actual = mgr.results[:, Models.increment]
        assert torch.allclose(actual, expected)


class TestLoadCVResults:
    def test_load_cv_results_filters_mean(self, tmp_path):
        """Test loading cv_results_summary.csv and filtering to mean columns."""
        # Create a summary CSV with the expected structure
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

        # Only mean columns should remain
        assert set(loaded.columns) == set(metrics)
        assert loaded.shape[0] == len(agg_index)

