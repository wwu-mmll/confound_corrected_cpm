"""
Ground-truth end-to-end tests for the CPM pipeline.

These tests use synthetic data with known structure so we can verify:
1. The correct edges are selected (high stability for informative edges)
2. Predictive performance meets expected thresholds
3. Non-informative edges are not selected
4. The full pipeline produces correct statistical results
"""

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
    """
    Generate regression data where we know exactly which edges are informative.

    The target y is constructed as:
        y = sum(positive_edges) - sum(negative_edges) + noise

    Args:
        n_samples: Number of samples
        n_features: Total number of features (edges)
        n_positive: Number of edges positively correlated with y
        n_negative: Number of edges negatively correlated with y
        signal_strength: Coefficient for informative edges
        noise_level: Standard deviation of noise added to y

    Returns:
        X, y, covariates, positive_edge_indices, negative_edge_indices
    """
    rng = np.random.RandomState(seed)

    # Features: all random noise initially
    X = rng.randn(n_samples, n_features).astype(np.float32)

    # Target: constructed from specific edges
    pos_indices = list(range(n_positive))
    neg_indices = list(range(n_positive, n_positive + n_negative))

    y = (signal_strength * X[:, pos_indices].sum(axis=1)
         - signal_strength * X[:, neg_indices].sum(axis=1)
         + rng.randn(n_samples) * noise_level).astype(np.float32)

    # Covariates: random noise (uncorrelated with y)
    covariates = rng.randn(n_samples, 2).astype(np.float32)

    return X, y, covariates, pos_indices, neg_indices


def generate_classification_data_known_edges(
    n_samples=200, n_features=45, n_positive=3, n_negative=3,
    signal_strength=1.0, seed=42
):
    """
    Generate binary classification data with known informative edges.

    The latent score is computed from specific edges, then binarized at the median.
    """
    X, y_cont, covariates, pos_indices, neg_indices = generate_regression_data_known_edges(
        n_samples=n_samples, n_features=n_features,
        n_positive=n_positive, n_negative=n_negative,
        signal_strength=signal_strength, noise_level=0.3, seed=seed
    )
    y_binary = (y_cont > np.median(y_cont)).astype(np.float64)
    return X, y_binary, covariates, pos_indices, neg_indices


class TestRegressionGroundTruth:
    """End-to-end test with known ground truth for regression."""

    @pytest.fixture
    def regression_data(self):
        return generate_regression_data_known_edges(
            n_samples=200, n_features=45, n_positive=3, n_negative=3,
            signal_strength=1.0, noise_level=0.3, seed=42
        )

    @pytest.fixture
    def cpm_result(self, tmp_path, regression_data):
        """Run the full CPM pipeline and return the instance."""
        X, y, covariates, pos_idx, neg_idx = regression_data

        edge_selection = UnivariateEdgeSelection(
            edge_statistic='pearson',
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

    def test_task_type_detected(self, cpm_result):
        cpm, _, _ = cpm_result
        assert cpm.task_type == TaskType.regression

    def test_positive_performance(self, cpm_result):
        """
        With strong signal edges, the connectome model should achieve
        meaningful predictive performance (Pearson r > 0.5).
        """
        cpm, _, _ = cpm_result
        agg = cpm.results_manager.agg_results

        # Get mean Pearson r for connectome/both model
        pearson_col = ('pearson_score', 'mean')
        r = agg.loc[('connectome', 'both', 0), pearson_col]

        print(f"\nConnectome/both Pearson r: {r:.4f}")
        assert r > 0.5, f"Pearson r too low: {r:.4f}. Expected > 0.5 with strong signal."

    def test_connectome_outperforms_covariates(self, cpm_result):
        """
        Since covariates are random noise, the connectome model should
        outperform the covariates-only model.
        """
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
        """
        Edges that are truly correlated with y should be selected in most folds
        (high stability), while noise edges should have low stability.
        """
        cpm, pos_idx, neg_idx = cpm_result
        informative_idx = pos_idx + neg_idx

        # stability shape: [N_Features, 2, Runs] after calculate_edge_stability
        stability = cpm.results_manager.calculate_edge_stability(write=False)

        # Check stability of positive edges (should be high)
        for idx in pos_idx:
            stab = stability[idx, Networks.positive, 0].item()
            print(f"Feature {idx} (positive signal): stability = {stab:.2f}")
            assert stab > 0.5, (
                f"Positive edge {idx} stability too low: {stab:.2f}"
            )

        # Check stability of negative edges (should be high)
        for idx in neg_idx:
            stab = stability[idx, Networks.negative, 0].item()
            print(f"Feature {idx} (negative signal): stability = {stab:.2f}")
            assert stab > 0.5, (
                f"Negative edge {idx} stability too low: {stab:.2f}"
            )

    def test_noise_edges_have_low_stability(self, cpm_result):
        """
        Noise edges (not in the signal set) should have low average stability.
        """
        cpm, pos_idx, neg_idx = cpm_result
        informative_idx = set(pos_idx + neg_idx)
        n_features = cpm.results_manager.n_features

        stability = cpm.results_manager.calculate_edge_stability(write=False)

        noise_stabilities = []
        for idx in range(n_features):
            if idx not in informative_idx:
                # Sum stability across pos and neg networks
                stab = stability[idx, :, 0].sum().item()
                noise_stabilities.append(stab)

        mean_noise_stability = np.mean(noise_stabilities)
        print(f"\nMean noise edge stability: {mean_noise_stability:.4f}")

        # Most noise edges should not be consistently selected
        # With p<0.05 threshold and 45 features, ~2 noise edges may pass by chance
        # But average stability should be well below 0.5
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
        """Verify CSV output only has regression metric columns."""
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
    """End-to-end test with known ground truth for classification."""

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
            edge_statistic='pearson',
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

    def test_task_type_detected(self, cpm_result):
        cpm, _, _ = cpm_result
        assert cpm.task_type == TaskType.classification

    def test_above_chance_accuracy(self, cpm_result):
        """With informative edges, accuracy should be well above chance (0.5)."""
        cpm, _, _ = cpm_result
        agg = cpm.results_manager.agg_results

        bal_acc_col = ('balanced_accuracy', 'mean')
        bal_acc = agg.loc[('connectome', 'both', 0), bal_acc_col]

        print(f"\nConnectome/both balanced accuracy: {bal_acc:.4f}")
        assert bal_acc > 0.6, (
            f"Balanced accuracy too low: {bal_acc:.4f}. Expected > 0.6 with signal."
        )

    def test_auc_above_chance(self, cpm_result):
        """ROC AUC should be well above 0.5 with informative edges."""
        cpm, _, _ = cpm_result
        agg = cpm.results_manager.agg_results

        auc_col = ('roc_auc', 'mean')
        auc = agg.loc[('connectome', 'both', 0), auc_col]

        print(f"\nConnectome/both ROC AUC: {auc:.4f}")
        assert auc > 0.65, f"AUC too low: {auc:.4f}. Expected > 0.65 with signal."

    def test_informative_edges_selected(self, cpm_result):
        """Signal edges should be frequently selected across folds."""
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


class TestEdgeSelectionStatistics:
    """
    Unit-level tests verifying that the edge selection step produces
    statistically correct results on data with known correlation structure.
    """

    def test_pearson_selects_correlated_edges(self):
        """
        Verify that pearson-based edge selection identifies edges
        that are correlated with y and rejects uncorrelated ones.
        """
        rng = np.random.RandomState(42)
        n_samples = 200
        n_features = 20

        X = rng.randn(n_samples, n_features).astype(np.float32)
        # Feature 0 is strongly positively correlated with y
        y = (X[:, 0] * 2.0 + rng.randn(n_samples) * 0.3).astype(np.float32).reshape(-1, 1)

        edge_sel = UnivariateEdgeSelection(
            edge_statistic='pearson',
            edge_selection=[PThreshold(threshold=[0.01], correction=[None])]
        )
        # set_params to configure the edge_selection as a single selector (as the pipeline does)
        edge_sel.set_params(**list(edge_sel.param_grid)[0])

        result = edge_sel.fit_transform(X=X, y=y, covariates=rng.randn(n_samples, 1))
        edges = result.return_selected_edges()  # [n_features, 2, 1]

        # Feature 0 should be selected as positive
        assert edges[0, Networks.positive, 0] == True, "Feature 0 should be selected as positive"

        # Count how many noise features are selected
        noise_selected = edges[1:, :, 0].sum().item()
        print(f"Noise features selected: {noise_selected} out of {n_features - 1}")
        # At p<0.01 with 19 noise features, expected false positives ~0.19
        assert noise_selected <= 5, f"Too many noise features selected: {noise_selected}"

    def test_negative_correlation_detected(self):
        """Edges negatively correlated with y should be selected as negative."""
        rng = np.random.RandomState(42)
        n_samples = 200
        n_features = 10

        X = rng.randn(n_samples, n_features).astype(np.float32)
        # Feature 0 is negatively correlated with y
        y = (-X[:, 0] * 2.0 + rng.randn(n_samples) * 0.3).astype(np.float32).reshape(-1, 1)

        edge_sel = UnivariateEdgeSelection(
            edge_statistic='pearson',
            edge_selection=[PThreshold(threshold=[0.01], correction=[None])]
        )
        edge_sel.set_params(**list(edge_sel.param_grid)[0])

        result = edge_sel.fit_transform(X=X, y=y, covariates=rng.randn(n_samples, 1))
        edges = result.return_selected_edges()

        # Feature 0 should be selected as negative
        assert edges[0, Networks.negative, 0] == True, "Feature 0 should be selected as negative"
        # And NOT selected as positive
        assert edges[0, Networks.positive, 0] == False, "Feature 0 should not be selected as positive"


class TestModelFitting:
    """Test that the linear model produces correct fits on simple data."""

    def test_ols_with_positive_and_negative_edges(self):
        """
        CPM aggregates features into positive/negative network strengths.
        When positive edges contribute positively and negative edges contribute
        negatively, the 'both' network (using both strengths) should yield
        a strong prediction.
        """
        from cccpm.models.linear_model import LinearCPM

        rng = np.random.RandomState(42)
        n_samples = 100
        n_pos = 3
        n_neg = 2
        n_features = n_pos + n_neg

        X = rng.randn(n_samples, n_features).astype(np.float32)
        # y = sum(positive features) - sum(negative features)
        y = (X[:, :n_pos].sum(axis=1) - X[:, n_pos:].sum(axis=1)).astype(np.float32).reshape(-1, 1)

        # Set up edges: first n_pos are positive, rest are negative
        edges = torch.zeros(n_features, 2, 1, dtype=torch.bool)
        edges[:n_pos, Networks.positive, 0] = True
        edges[n_pos:, Networks.negative, 0] = True

        model = LinearCPM(edges=edges, device='cpu', task_type=TaskType.regression)
        cov = np.zeros((n_samples, 1), dtype=np.float32)
        model.fit(X, y, cov)

        preds = model.predict(X, cov)
        # The "both" network uses both positive and negative strengths
        y_pred = preds[:, Models.connectome, Networks.both, 0].numpy()

        from scipy.stats import pearsonr
        r, _ = pearsonr(y.ravel(), y_pred)
        print(f"\nPearson r with pos+neg edges: {r:.6f}")
        # Should be perfect since the generative process matches the model
        assert r > 0.999, f"Should be near-perfect fit: r={r:.6f}"

    def test_ols_with_uniform_coefficients(self):
        """
        When all true coefficients are equal, the sum-based aggregation
        in CPM should recover the signal perfectly.
        """
        from cccpm.models.linear_model import LinearCPM

        rng = np.random.RandomState(42)
        n_samples = 100
        n_features = 5

        X = rng.randn(n_samples, n_features).astype(np.float32)
        # y = sum of all features (uniform weights of 1.0)
        y = X.sum(axis=1).astype(np.float32).reshape(-1, 1)

        edges = torch.ones(n_features, 2, 1, dtype=torch.bool)
        edges[:, Networks.negative, :] = False

        model = LinearCPM(edges=edges, device='cpu', task_type=TaskType.regression)
        cov = np.zeros((n_samples, 1), dtype=np.float32)
        model.fit(X, y, cov)

        preds = model.predict(X, cov)
        y_pred = preds[:, Models.connectome, Networks.positive, 0].numpy()

        from scipy.stats import pearsonr
        r, _ = pearsonr(y.ravel(), y_pred)
        print(f"\nPearson r with uniform coefficients: {r:.6f}")
        # Sum aggregation matches the true generative process exactly
        assert r > 0.999, f"Should be perfect fit with uniform coefficients: r={r:.6f}"

    def test_logistic_regression_recovers_separation(self):
        """
        With linearly separable data, logistic regression should achieve
        near-perfect classification.
        """
        from cccpm.models.linear_model import LinearCPM

        rng = np.random.RandomState(42)
        n_samples = 100

        # 1D feature with clear separation
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
