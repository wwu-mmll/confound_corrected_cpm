"""Synthetic data with known signal edges: verifies correct edges are selected,
predictive performance meets thresholds, and noise edges are not selected."""

import numpy as np
import pytest
import torch
import os

from sklearn.model_selection import KFold, StratifiedKFold

from cccpm.cpm_analysis import CPMAnalysis
from cccpm.edge_selection import UnivariateEdgeSelection, PThreshold
from cccpm.constants import TaskType, Models, Networks


def generate_regression_data_known_edges(
    n_samples=200, n_features=45, n_positive=3, n_negative=3,
    signal_strength=1.0, noise_level=0.3, seed=42
):
    """y = sum(positive_edges) - sum(negative_edges) + noise; covariates are
    uncorrelated noise. Returns (X, y, covariates, positive_idx, negative_idx)."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)

    pos_indices = list(range(n_positive))
    neg_indices = list(range(n_positive, n_positive + n_negative))

    y = (signal_strength * X[:, pos_indices].sum(axis=1)
         - signal_strength * X[:, neg_indices].sum(axis=1)
         + rng.randn(n_samples) * noise_level).astype(np.float32)

    covariates = rng.randn(n_samples, 2).astype(np.float32)

    return X, y, covariates, pos_indices, neg_indices


def generate_classification_data_known_edges(
    n_samples=200, n_features=45, n_positive=3, n_negative=3,
    signal_strength=1.0, seed=42
):
    """Like generate_regression_data_known_edges, with y binarized at the median."""
    X, y_cont, covariates, pos_indices, neg_indices = generate_regression_data_known_edges(
        n_samples=n_samples, n_features=n_features,
        n_positive=n_positive, n_negative=n_negative,
        signal_strength=signal_strength, noise_level=0.3, seed=seed
    )
    y_binary = (y_cont > np.median(y_cont)).astype(np.float64)
    return X, y_binary, covariates, pos_indices, neg_indices


class TestRegressionGroundTruth:
    @pytest.fixture
    def regression_data(self):
        return generate_regression_data_known_edges(
            n_samples=200, n_features=45, n_positive=3, n_negative=3,
            signal_strength=1.0, noise_level=0.3, seed=42
        )

    @pytest.fixture
    def cpm_result(self, tmp_path, regression_data):
        X, y, covariates, pos_idx, neg_idx = regression_data

        edge_selection = UnivariateEdgeSelection(
            selection_statistic='pearson',
            edge_selection=[PThreshold(threshold=[0.05], correction=[None])]
        )

        cpm = CPMAnalysis(
            results_directory=str(tmp_path),
            task_type='regression',
            cv=KFold(n_splits=10, shuffle=True, random_state=42),
            edge_selection=edge_selection,
            n_permutations=0,
            impute_missing_values=False
        )
        cpm.run(X, y, covariates)
        return cpm, pos_idx, neg_idx

    def test_positive_performance(self, cpm_result):
        cpm, _, _ = cpm_result
        agg = cpm.results_manager.agg_results

        pearson_col = ('pearson_score', 'mean')
        r = agg.loc[('connectome', 'both', 0), pearson_col]

        print(f"\nConnectome/both Pearson r: {r:.4f}")
        assert r > 0.5, f"Pearson r too low: {r:.4f}. Expected > 0.5 with strong signal."

    def test_connectome_outperforms_covariates(self, cpm_result):
        cpm, _, _ = cpm_result
        agg = cpm.results_manager.agg_results

        pearson_col = ('pearson_score', 'mean')
        r_conn = agg.loc[('connectome', 'both', 0), pearson_col]
        r_cov = agg.loc[('covariates', 'both', 0), pearson_col]

        print(f"\nConnectome r: {r_conn:.4f}, Covariates r: {r_cov:.4f}")
        assert r_conn > r_cov, (
            f"Connectome ({r_conn:.4f}) should outperform covariates ({r_cov:.4f})"
        )

    def test_informative_edges_have_high_stability(self, cpm_result):
        cpm, pos_idx, neg_idx = cpm_result

        # stability shape: [N_Features, 2, Runs]
        stability = cpm.results_manager.calculate_edge_stability(write=False)

        for idx in pos_idx:
            stab = stability[idx, Networks.positive, 0].item()
            print(f"Feature {idx} (positive signal): stability = {stab:.2f}")
            assert stab > 0.5, (
                f"Positive edge {idx} stability too low: {stab:.2f}"
            )

        for idx in neg_idx:
            stab = stability[idx, Networks.negative, 0].item()
            print(f"Feature {idx} (negative signal): stability = {stab:.2f}")
            assert stab > 0.5, (
                f"Negative edge {idx} stability too low: {stab:.2f}"
            )

    def test_noise_edges_have_low_stability(self, cpm_result):
        cpm, pos_idx, neg_idx = cpm_result
        informative_idx = set(pos_idx + neg_idx)
        n_features = cpm.results_manager.n_features

        stability = cpm.results_manager.calculate_edge_stability(write=False)

        noise_stabilities = []
        for idx in range(n_features):
            if idx not in informative_idx:
                stab = stability[idx, :, 0].sum().item()  # sum over pos/neg networks
                noise_stabilities.append(stab)

        mean_noise_stability = np.mean(noise_stabilities)
        print(f"\nMean noise edge stability: {mean_noise_stability:.4f}")

        # With p<0.05 and 45 features, ~2 noise edges may pass by chance, but
        # the average should stay well below 0.5.
        assert mean_noise_stability < 0.3, (
            f"Noise edge stability too high: {mean_noise_stability:.4f}"
        )

    def test_results_files_exist(self, cpm_result):
        cpm, _, _ = cpm_result
        results_dir = cpm.results_directory

        assert os.path.exists(os.path.join(results_dir, 'cv_results_full.csv'))
        assert os.path.exists(os.path.join(results_dir, 'cv_results_summary.csv'))
        assert os.path.exists(os.path.join(results_dir, 'edges.npy'))
        assert os.path.exists(os.path.join(results_dir, 'stability_edges.npy'))
        assert os.path.exists(os.path.join(results_dir, 'cv_predictions.csv'))
        assert os.path.exists(os.path.join(results_dir, 'task_type.txt'))

    def test_csv_contains_only_regression_metrics(self, cpm_result):
        import pandas as pd
        cpm, _, _ = cpm_result
        df = pd.read_csv(os.path.join(cpm.results_directory, 'cv_results_full.csv'))

        regression_metrics = {'explained_variance_score', 'pearson_score',
                              'mean_squared_error', 'mean_absolute_error'}
        classification_metrics = {'accuracy', 'balanced_accuracy', 'f1_score', 'roc_auc'}

        for m in regression_metrics:
            assert m in df.columns, f"Missing regression metric: {m}"
        for m in classification_metrics:
            assert m not in df.columns, f"Classification metric should not be present: {m}"


class TestClassificationGroundTruth:
    @pytest.fixture
    def classification_data(self):
        return generate_classification_data_known_edges(
            n_samples=200, n_features=45, n_positive=3, n_negative=3,
            signal_strength=1.0, seed=42
        )

    @pytest.fixture
    def cpm_result(self, tmp_path, classification_data):
        X, y, covariates, pos_idx, neg_idx = classification_data

        edge_selection = UnivariateEdgeSelection(
            selection_statistic='pearson',
            edge_selection=[PThreshold(threshold=[0.05], correction=[None])]
        )

        cpm = CPMAnalysis(
            results_directory=str(tmp_path),
            task_type='classification',
            cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
            edge_selection=edge_selection,
            n_permutations=0,
            impute_missing_values=False
        )
        cpm.run(X, y, covariates)
        return cpm, pos_idx, neg_idx

    def test_above_chance_accuracy(self, cpm_result):
        cpm, _, _ = cpm_result
        agg = cpm.results_manager.agg_results

        bal_acc_col = ('balanced_accuracy', 'mean')
        bal_acc = agg.loc[('connectome', 'both', 0), bal_acc_col]

        print(f"\nConnectome/both balanced accuracy: {bal_acc:.4f}")
        assert bal_acc > 0.6, (
            f"Balanced accuracy too low: {bal_acc:.4f}. Expected > 0.6 with signal."
        )

    def test_auc_above_chance(self, cpm_result):
        cpm, _, _ = cpm_result
        agg = cpm.results_manager.agg_results

        auc_col = ('roc_auc', 'mean')
        auc = agg.loc[('connectome', 'both', 0), auc_col]

        print(f"\nConnectome/both ROC AUC: {auc:.4f}")
        assert auc > 0.65, f"AUC too low: {auc:.4f}. Expected > 0.65 with signal."

    def test_informative_edges_selected(self, cpm_result):
        cpm, pos_idx, neg_idx = cpm_result
        stability = cpm.results_manager.calculate_edge_stability(write=False)

        for idx in pos_idx:
            stab = stability[idx, Networks.positive, 0].item()
            print(f"Feature {idx} (positive signal): stability = {stab:.2f}")
            assert stab > 0.5, f"Positive edge {idx} stability too low: {stab:.2f}"

        for idx in neg_idx:
            stab = stability[idx, Networks.negative, 0].item()
            print(f"Feature {idx} (negative signal): stability = {stab:.2f}")
            assert stab > 0.5, f"Negative edge {idx} stability too low: {stab:.2f}"

    def test_csv_contains_only_classification_metrics(self, cpm_result):
        import pandas as pd
        cpm, _, _ = cpm_result
        df = pd.read_csv(os.path.join(cpm.results_directory, 'cv_results_full.csv'))

        classification_metrics = {'accuracy', 'balanced_accuracy', 'f1_score', 'roc_auc'}
        regression_metrics = {'explained_variance_score', 'pearson_score',
                              'mean_squared_error', 'mean_absolute_error'}

        for m in classification_metrics:
            assert m in df.columns, f"Missing classification metric: {m}"
        for m in regression_metrics:
            assert m not in df.columns, f"Regression metric should not be present: {m}"


class TestModelFitting:
    def test_ols_with_positive_and_negative_edges(self):
        """The 'both' network (combining positive and negative strengths) should
        yield a strong prediction when the generative process matches it."""
        from cccpm.models.linear_model import LinearCPM

        rng = np.random.RandomState(42)
        n_samples = 100
        n_pos = 3
        n_neg = 2
        n_features = n_pos + n_neg

        X = rng.randn(n_samples, n_features).astype(np.float32)
        y = (X[:, :n_pos].sum(axis=1) - X[:, n_pos:].sum(axis=1)).astype(np.float32).reshape(-1, 1)

        edges = torch.zeros(n_features, 2, 1, dtype=torch.bool)
        edges[:n_pos, Networks.positive, 0] = True
        edges[n_pos:, Networks.negative, 0] = True

        model = LinearCPM(edges=edges, device='cpu', task_type=TaskType.regression)
        cov = np.zeros((n_samples, 1), dtype=np.float32)
        model.fit(X, y, cov)

        preds = model.predict(X, cov)
        y_pred = preds[:, Models.connectome, Networks.both, 0].numpy()  # uses both strengths

        from scipy.stats import pearsonr
        r, _ = pearsonr(y.ravel(), y_pred)
        print(f"\nPearson r with pos+neg edges: {r:.6f}")
        # Should be perfect: the generative process matches the model exactly.
        assert r > 0.999, f"Should be near-perfect fit: r={r:.6f}"

    def test_logistic_regression_recovers_separation(self):
        from cccpm.models.linear_model import LinearCPM

        rng = np.random.RandomState(42)
        n_samples = 100

        X = np.concatenate([
            rng.randn(50, 1) - 2,
            rng.randn(50, 1) + 2
        ]).astype(np.float32)
        y = np.concatenate([np.zeros(50), np.ones(50)]).reshape(-1, 1).astype(np.float32)

        edges = torch.ones(1, 2, 1, dtype=torch.bool)
        edges[:, Networks.negative, :] = False

        model = LinearCPM(edges=edges, device='cpu', task_type=TaskType.classification)
        cov = np.zeros((n_samples, 1), dtype=np.float32)
        model.fit(X, y, cov)

        preds = model.predict_class(X, cov)
        y_pred = preds[:, Models.connectome, Networks.positive, 0].numpy()

        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y.ravel(), y_pred)
        print(f"\nAccuracy on separable data: {acc:.4f}")
        assert acc > 0.95, f"Should achieve near-perfect accuracy: {acc:.4f}"
