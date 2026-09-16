import pytest
import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score as sklearn_f1_score,
    roc_auc_score
)
from cccpm.scoring import FastCPMMetrics, FastCPMClassificationMetrics, score_models
from cccpm.constants import Networks, Models, Metrics, TaskType


N_MODELS = len(Models)
N_NETWORKS = len(Networks)
N_METRICS = len(Metrics)


@pytest.fixture
def device():
    """Returns 'cuda' if available, else 'cpu'."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def simple_data():
    """Creates simple synthetic data for testing.
    Returns (y_true, y_pred) as tensors with the current API shapes.
    """
    np.random.seed(42)
    N_samples = 50
    N_runs = 3

    y_true = np.random.randn(N_samples, N_runs).astype(np.float32)
    # y_pred: [N_samples, N_models, N_networks, N_runs]
    y_pred = np.random.randn(N_samples, N_MODELS, N_NETWORKS, N_runs).astype(np.float32)

    return y_true, y_pred


@pytest.fixture
def perfect_data():
    """Creates data where predictions perfectly match ground truth."""
    np.random.seed(42)
    N_samples = 30
    N_runs = 2

    y_true = np.random.randn(N_samples, N_runs).astype(np.float32)
    # Expand y_true to [N_samples, N_models, N_networks, N_runs]
    y_pred = np.broadcast_to(
        y_true[:, np.newaxis, np.newaxis, :],
        (N_samples, N_MODELS, N_NETWORKS, N_runs)
    ).copy()

    return y_true, y_pred


class TestFastCPMMetrics:

    def test_initialization(self, device):
        """Test that FastCPMMetrics initializes correctly."""
        evaluator = FastCPMMetrics(device=device)
        assert evaluator.device == device

    def test_score_output_shape(self, simple_data, device):
        """Test that score returns correct output shape."""
        y_true, y_pred = simple_data
        evaluator = FastCPMMetrics(device=device)
        scores = evaluator.score(y_true, y_pred)

        N_runs = y_true.shape[1]
        expected_shape = (N_METRICS, N_MODELS, N_NETWORKS, N_runs)
        assert scores.shape == expected_shape, f"Expected {expected_shape}, got {scores.shape}"

    def test_score_returns_tensor(self, simple_data, device):
        """Test that score returns a torch.Tensor."""
        y_true, y_pred = simple_data
        evaluator = FastCPMMetrics(device=device)
        scores = evaluator.score(y_true, y_pred)
        assert isinstance(scores, torch.Tensor)

    def test_perfect_predictions_pearson(self, perfect_data, device):
        """Test that perfect predictions yield Pearson correlation of 1."""
        y_true, y_pred = perfect_data
        evaluator = FastCPMMetrics(device=device)
        scores = evaluator.score(y_true, y_pred)

        pearson_scores = scores[Metrics.pearson_score, :, :, :]
        assert torch.allclose(pearson_scores, torch.ones_like(pearson_scores), atol=1e-5)

    def test_perfect_predictions_mse(self, perfect_data, device):
        """Test that perfect predictions yield MSE of 0."""
        y_true, y_pred = perfect_data
        evaluator = FastCPMMetrics(device=device)
        scores = evaluator.score(y_true, y_pred)

        mse_scores = scores[Metrics.mean_squared_error, :, :, :]
        assert torch.allclose(mse_scores, torch.zeros_like(mse_scores), atol=1e-5)

    def test_perfect_predictions_mae(self, perfect_data, device):
        """Test that perfect predictions yield MAE of 0."""
        y_true, y_pred = perfect_data
        evaluator = FastCPMMetrics(device=device)
        scores = evaluator.score(y_true, y_pred)

        mae_scores = scores[Metrics.mean_absolute_error, :, :, :]
        assert torch.allclose(mae_scores, torch.zeros_like(mae_scores), atol=1e-5)

    def test_perfect_predictions_explained_variance(self, perfect_data, device):
        """Test that perfect predictions yield explained variance of 1."""
        y_true, y_pred = perfect_data
        evaluator = FastCPMMetrics(device=device)
        scores = evaluator.score(y_true, y_pred)

        ev_scores = scores[Metrics.explained_variance_score, :, :, :]
        assert torch.allclose(ev_scores, torch.ones_like(ev_scores), atol=1e-5)

    def test_mse_calculation(self, device):
        """Test MSE calculation with known values."""
        N, P = 10, 1
        y_true = np.full((N, P), 2.0, dtype=np.float32)
        y_pred = np.full((N, N_MODELS, N_NETWORKS, P), 4.0, dtype=np.float32)

        evaluator = FastCPMMetrics(device=device)
        scores = evaluator.score(y_true, y_pred)

        mse_scores = scores[Metrics.mean_squared_error, :, :, :]
        expected = torch.full_like(mse_scores, 4.0)  # (4-2)^2 = 4
        assert torch.allclose(mse_scores, expected, atol=1e-5)

    def test_mae_calculation(self, device):
        """Test MAE calculation with known values."""
        N, P = 10, 1
        y_true = np.full((N, P), 2.0, dtype=np.float32)
        y_pred = np.full((N, N_MODELS, N_NETWORKS, P), 5.0, dtype=np.float32)

        evaluator = FastCPMMetrics(device=device)
        scores = evaluator.score(y_true, y_pred)

        mae_scores = scores[Metrics.mean_absolute_error, :, :, :]
        expected = torch.full_like(mae_scores, 3.0)  # |5-2| = 3
        assert torch.allclose(mae_scores, expected, atol=1e-5)

    def test_covariates_shared_across_networks(self, device):
        """Test that when covariates preds are identical across networks, scores match."""
        np.random.seed(42)
        N, P = 20, 2
        y_true = np.random.randn(N, P).astype(np.float32)

        # Build y_pred where covariates model has same predictions across all networks
        y_pred = np.random.randn(N, N_MODELS, N_NETWORKS, P).astype(np.float32)
        cov_pred = np.random.randn(N, P).astype(np.float32)
        for net in Networks:
            y_pred[:, Models.covariates, net, :] = cov_pred

        evaluator = FastCPMMetrics(device=device)
        scores = evaluator.score(y_true, y_pred)

        cov_scores = scores[:, Models.covariates, :, :]
        for net in Networks:
            assert torch.allclose(
                cov_scores[:, Networks.positive, :],
                cov_scores[:, net, :],
                atol=1e-5
            )

    def test_handles_torch_and_numpy_input(self, device):
        """Test that the method handles both torch.Tensor and numpy.ndarray inputs."""
        np.random.seed(42)
        N, P = 15, 2
        y_true_np = np.random.randn(N, P).astype(np.float32)
        y_pred_np = np.random.randn(N, N_MODELS, N_NETWORKS, P).astype(np.float32)

        y_true_torch = torch.from_numpy(y_true_np)
        y_pred_torch = torch.from_numpy(y_pred_np)

        evaluator = FastCPMMetrics(device=device)
        scores_np = evaluator.score(y_true_np, y_pred_np)
        scores_torch = evaluator.score(y_true_torch, y_pred_torch)

        assert torch.allclose(scores_np, scores_torch, atol=1e-5)

    def test_pearson_negative_correlation(self, device):
        """Test Pearson correlation with negatively correlated predictions."""
        N, P = 50, 1
        y_true = np.linspace(0, 10, N).reshape(-1, 1).astype(np.float32)
        y_pred_vals = -y_true  # Perfect negative correlation
        y_pred = np.broadcast_to(
            y_pred_vals[:, np.newaxis, np.newaxis, :],
            (N, N_MODELS, N_NETWORKS, P)
        ).copy()

        evaluator = FastCPMMetrics(device=device)
        scores = evaluator.score(y_true, y_pred)

        pearson_scores = scores[Metrics.pearson_score, :, :, :]
        expected = torch.full_like(pearson_scores, -1.0)
        assert torch.allclose(pearson_scores, expected, atol=1e-5)

    def test_multiple_permutations(self, device):
        """Test that multiple runs are handled correctly."""
        np.random.seed(42)
        N, P = 30, 5
        y_true = np.random.randn(N, P).astype(np.float32)
        y_pred = np.random.randn(N, N_MODELS, N_NETWORKS, P).astype(np.float32)

        evaluator = FastCPMMetrics(device=device)
        scores = evaluator.score(y_true, y_pred)

        assert scores.shape[-1] == P

        # Each run should have different scores (extremely unlikely to be identical)
        for model in Models:
            for network in Networks:
                for metric in [Metrics.pearson_score, Metrics.mean_squared_error]:
                    run_scores = scores[metric, model, network, :]
                    assert not torch.allclose(run_scores, run_scores[0].expand_as(run_scores))

    def test_score_models_wrapper(self, simple_data, device):
        """Test the score_models wrapper function for regression."""
        y_true, y_pred = simple_data
        scores = score_models(y_true, y_pred, task_type=TaskType.regression, device=device)

        N_runs = y_true.shape[1]
        expected_shape = (N_METRICS, N_MODELS, N_NETWORKS, N_runs)
        assert scores.shape == expected_shape
        assert isinstance(scores, torch.Tensor)

    def test_metrics_ordering(self, simple_data, device):
        """Test that metrics are correctly ordered according to constants.Metrics."""
        y_true, y_pred = simple_data
        evaluator = FastCPMMetrics(device=device)
        scores = evaluator.score(y_true, y_pred)

        assert scores.shape[0] == N_METRICS

        for metric in [Metrics.explained_variance_score, Metrics.pearson_score,
                       Metrics.mean_squared_error, Metrics.mean_absolute_error]:
            metric_scores = scores[metric, :, :, :]
            assert metric_scores.shape == (N_MODELS, N_NETWORKS, y_true.shape[1])


class TestFastCPMMetricsVsSklearn:
    """Test that FastCPMMetrics produces the same results as scikit-learn."""

    def _check_metric_vs_reference(self, scores, y_true, y_pred, metric_idx,
                                    reference_fn, label):
        """Helper to compare our metric against a reference for all model/network/run combos."""
        N_runs = y_true.shape[1]
        metric_scores = scores[metric_idx, :, :, :].cpu().numpy()

        for model in Models:
            for network in Networks:
                for run_idx in range(N_runs):
                    yt = y_true[:, run_idx]
                    yp = y_pred[:, model, network, run_idx]
                    expected = reference_fn(yt, yp)
                    actual = metric_scores[model, network, run_idx]
                    assert np.isclose(actual, expected, rtol=1e-4), (
                        f"{label} mismatch for {model.name}/{network.name}/run{run_idx}: "
                        f"ours={actual:.6f}, reference={expected:.6f}"
                    )

    def test_mse_vs_sklearn(self, simple_data, device):
        y_true, y_pred = simple_data
        evaluator = FastCPMMetrics(device=device)
        scores = evaluator.score(y_true, y_pred)
        self._check_metric_vs_reference(scores, y_true, y_pred,
            Metrics.mean_squared_error, mean_squared_error, "MSE")

    def test_mae_vs_sklearn(self, simple_data, device):
        y_true, y_pred = simple_data
        evaluator = FastCPMMetrics(device=device)
        scores = evaluator.score(y_true, y_pred)
        self._check_metric_vs_reference(scores, y_true, y_pred,
            Metrics.mean_absolute_error, mean_absolute_error, "MAE")

    def test_explained_variance_vs_sklearn(self, simple_data, device):
        y_true, y_pred = simple_data
        evaluator = FastCPMMetrics(device=device)
        scores = evaluator.score(y_true, y_pred)
        self._check_metric_vs_reference(scores, y_true, y_pred,
            Metrics.explained_variance_score, explained_variance_score, "EV")

    def test_pearson_vs_scipy(self, simple_data, device):
        y_true, y_pred = simple_data
        evaluator = FastCPMMetrics(device=device)
        scores = evaluator.score(y_true, y_pred)

        def pearson_ref(yt, yp):
            r, _ = pearsonr(yt, yp)
            return r

        self._check_metric_vs_reference(scores, y_true, y_pred,
            Metrics.pearson_score, pearson_ref, "Pearson")

    def test_all_metrics_vs_sklearn_random_data(self, device):
        """Comprehensive test with varying prediction quality."""
        np.random.seed(42)
        N, P = 100, 4
        y_true = np.random.randn(N, P).astype(np.float32)

        # Create predictions with varying noise levels per model/network
        y_pred = np.zeros((N, N_MODELS, N_NETWORKS, P), dtype=np.float32)
        for model in Models:
            for network in Networks:
                noise = np.random.uniform(0.1, 2.0)
                y_pred[:, model, network, :] = y_true + np.random.randn(N, P).astype(np.float32) * noise

        evaluator = FastCPMMetrics(device=device)
        scores = evaluator.score(y_true, y_pred)

        for run_idx in range(P):
            for model in Models:
                for network in Networks:
                    yt = y_true[:, run_idx]
                    yp = y_pred[:, model, network, run_idx]

                    assert np.isclose(
                        scores[Metrics.mean_squared_error, model, network, run_idx].cpu().item(),
                        mean_squared_error(yt, yp), rtol=1e-4)
                    assert np.isclose(
                        scores[Metrics.mean_absolute_error, model, network, run_idx].cpu().item(),
                        mean_absolute_error(yt, yp), rtol=1e-4)
                    assert np.isclose(
                        scores[Metrics.explained_variance_score, model, network, run_idx].cpu().item(),
                        explained_variance_score(yt, yp), rtol=1e-4)

                    r, _ = pearsonr(yt, yp)
                    assert np.isclose(
                        scores[Metrics.pearson_score, model, network, run_idx].cpu().item(),
                        r, rtol=1e-4)

    def test_edge_cases_vs_sklearn(self, device):
        """Test edge case: constant predictions."""
        np.random.seed(42)
        N, P = 50, 2
        y_true = np.random.randn(N, P).astype(np.float32)
        y_pred = np.full((N, N_MODELS, N_NETWORKS, P), 5.0, dtype=np.float32)

        evaluator = FastCPMMetrics(device=device)
        scores = evaluator.score(y_true, y_pred)

        for run_idx in range(P):
            yt = y_true[:, run_idx]
            yp = y_pred[:, 0, 0, run_idx]

            assert np.isclose(
                scores[Metrics.mean_squared_error, 0, 0, run_idx].cpu().item(),
                mean_squared_error(yt, yp), rtol=1e-4)
            assert np.isclose(
                scores[Metrics.mean_absolute_error, 0, 0, run_idx].cpu().item(),
                mean_absolute_error(yt, yp), rtol=1e-4)


class TestFastCPMClassificationMetricsVsSklearn:
    """Test that FastCPMClassificationMetrics matches sklearn."""

    def test_perfect_classification_vs_sklearn(self):
        N = 50
        y_true = np.concatenate([np.zeros(25), np.ones(25)]).reshape(-1, 1).astype(np.float32)
        y_pred_proba = y_true.copy()
        y_pred = np.broadcast_to(
            y_pred_proba[:, np.newaxis, np.newaxis, :],
            (N, N_MODELS, N_NETWORKS, 1)
        ).copy()

        evaluator = FastCPMClassificationMetrics(device='cpu')
        scores = evaluator.score(y_true, y_pred)

        # All classification metrics should be 1.0 for perfect predictions
        assert torch.allclose(scores[Metrics.accuracy], torch.ones(N_MODELS, N_NETWORKS, 1), atol=1e-5)
        assert torch.allclose(scores[Metrics.balanced_accuracy], torch.ones(N_MODELS, N_NETWORKS, 1), atol=1e-5)
        assert torch.allclose(scores[Metrics.f1_score], torch.ones(N_MODELS, N_NETWORKS, 1), atol=1e-5)
        assert torch.allclose(scores[Metrics.roc_auc], torch.ones(N_MODELS, N_NETWORKS, 1), atol=1e-5)

    def test_classification_metrics_vs_sklearn(self):
        """Compare classification metrics against sklearn for random predictions."""
        np.random.seed(42)
        N = 100
        y_true = np.random.randint(0, 2, (N, 1)).astype(np.float32)
        y_pred_proba = np.random.rand(N, N_MODELS, N_NETWORKS, 1).astype(np.float32)

        evaluator = FastCPMClassificationMetrics(device='cpu')
        scores = evaluator.score(y_true, y_pred_proba)

        for model in Models:
            for network in Networks:
                proba = y_pred_proba[:, model, network, 0]
                preds = (proba > 0.5).astype(int)
                yt = y_true[:, 0].astype(int)

                sk_acc = accuracy_score(yt, preds)
                sk_bal_acc = balanced_accuracy_score(yt, preds)
                sk_f1 = sklearn_f1_score(yt, preds)
                sk_auc = roc_auc_score(yt, proba)

                assert np.isclose(scores[Metrics.accuracy, model, network, 0].item(), sk_acc, atol=1e-4), \
                    f"Accuracy mismatch for {model.name}/{network.name}"
                assert np.isclose(scores[Metrics.balanced_accuracy, model, network, 0].item(), sk_bal_acc, atol=1e-4), \
                    f"Balanced accuracy mismatch for {model.name}/{network.name}"
                assert np.isclose(scores[Metrics.f1_score, model, network, 0].item(), sk_f1, atol=1e-4), \
                    f"F1 mismatch for {model.name}/{network.name}"
                assert np.isclose(scores[Metrics.roc_auc, model, network, 0].item(), sk_auc, atol=1e-3), \
                    f"ROC AUC mismatch for {model.name}/{network.name}"

    def test_score_models_routes_classification(self):
        """Test that score_models routes correctly for classification."""
        np.random.seed(42)
        N = 40
        y_true = np.random.randint(0, 2, (N, 1)).astype(np.float32)
        y_pred = np.random.rand(N, N_MODELS, N_NETWORKS, 1).astype(np.float32)

        scores = score_models(y_true, y_pred, task_type=TaskType.classification, device='cpu')
        assert scores.shape == (N_METRICS, N_MODELS, N_NETWORKS, 1)

        # Classification metrics should be in [0, 1]
        for metric in [Metrics.accuracy, Metrics.balanced_accuracy, Metrics.f1_score, Metrics.roc_auc]:
            vals = scores[metric]
            assert (vals >= 0).all() and (vals <= 1).all()
