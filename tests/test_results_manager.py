import os
import pytest
import numpy as np
import pandas as pd
import torch

from cccpm.results_manager import ResultsManager, PermutationManager
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
        """Test that increment = full - connectome is computed correctly."""
        mgr = ResultsManager(
            output_dir=str(tmp_path), n_runs=1, n_folds=2, n_features=3
        )
        for fold in range(2):
            metrics = torch.randn(len(Metrics), len(Models), len(Networks), 1)
            mgr.store_metrics(param_idx=0, fold_idx=fold, metrics_tensor=metrics)

        mgr.calculate_final_cv_results()

        expected = mgr.results[:, Models.full] - mgr.results[:, Models.connectome]
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


class TestPermutationManager:
    def test_calculate_group_p_value_higher_is_better(self):
        """For metrics where higher is better, p = (count(true < perm) + 1) / (n_perms + 1)."""
        true = pd.DataFrame({'pearson_score': [0.5]})
        perms = pd.DataFrame({'pearson_score': [0.3, 0.6, 0.4, 0.7]})

        p = PermutationManager._calculate_group_p_value(true, perms)

        # true (0.5) < perm: 0.6, 0.7 → 2 out of 4. p = (2+1)/(4+1) = 0.6
        assert p['pearson_score'] == pytest.approx(3 / 5)

    def test_calculate_group_p_value_lower_is_better(self):
        """For error metrics (lower is better), p = (count(true > perm) + 1) / (n_perms + 1)."""
        true = pd.DataFrame({'mean_squared_error': [1.5]})
        perms = pd.DataFrame({'mean_squared_error': [1.4, 1.6, 1.5, 1.7]})

        p = PermutationManager._calculate_group_p_value(true, perms)

        # true (1.5) > perm: 1.4 → 1 out of 4. p = (1+1)/(4+1) = 0.4
        assert p['mean_squared_error'] == pytest.approx(2 / 5)

    def test_calculate_group_p_value_mixed_metrics(self):
        """Test with both higher-is-better and lower-is-better metrics."""
        true = pd.DataFrame({'pearson_score': [0.2], 'mean_squared_error': [1.5]})
        perms = pd.DataFrame({
            'pearson_score': [0.1, 0.3, 0.25],
            'mean_squared_error': [1.4, 1.6, 1.5]
        })

        p = PermutationManager._calculate_group_p_value(true, perms)

        # pearson: true 0.2 < perm → 0.3, 0.25 = 2 of 3. p = (2+1)/(3+1) = 0.75
        assert p['pearson_score'] == pytest.approx(3 / 4)
        # mse: true 1.5 > perm → 1.4 = 1 of 3. p = (1+1)/(3+1) = 0.5
        assert p['mean_squared_error'] == pytest.approx(2 / 4)

    def test_calculate_group_p_value_never_exceeds_one(self):
        """A valid p-value must be in (0, 1] even when every permutation beats the true value."""
        true = pd.DataFrame({'pearson_score': [0.0]})
        perms = pd.DataFrame({'pearson_score': [0.5, 0.6, 0.7]})  # all beat true

        p = PermutationManager._calculate_group_p_value(true, perms)

        # (3 + 1) / (3 + 1) = 1.0 — must not exceed 1
        assert p['pearson_score'] == pytest.approx(1.0)
        assert 0 < p['pearson_score'] <= 1

    def test_calculate_p_values_groups(self):
        """Test grouped p-value calculation across model/network combinations."""
        df_true = pd.DataFrame({
            'network': ['positive', 'negative'],
            'model': ['connectome', 'connectome'],
            'pearson_score': [0.5, 0.6],
            'mean_squared_error': [1.0, 0.9]
        }).set_index(['network', 'model'])

        perms_list = []
        for vals in ([0.4, 0.7], [0.6, 0.5]):
            df = pd.DataFrame({
                'network': ['positive', 'negative'],
                'model': ['connectome', 'connectome'],
                'pearson_score': vals,
                'mean_squared_error': [1.1, 0.8]
            }).set_index(['network', 'model'])
            perms_list.append(df)

        all_perms = pd.concat(perms_list)
        pvals = PermutationManager.calculate_p_values(df_true, all_perms)

        assert ('positive', 'connectome') in pvals.index
        assert ('negative', 'connectome') in pvals.index
        assert 'pearson_score' in pvals.columns
        assert 'mean_squared_error' in pvals.columns

    def test_is_lower_better(self):
        assert PermutationManager._is_lower_better('mean_squared_error') == True
        assert PermutationManager._is_lower_better('mean_absolute_error') == True
        assert PermutationManager._is_lower_better('pearson_score') == False
        assert PermutationManager._is_lower_better('accuracy') == False
        assert PermutationManager._is_lower_better('some_error') == True
