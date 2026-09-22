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


class TestFastCPMMetrics:

    def test_score_output_shape(self, simple_data, device):
        """Test that score returns correct output shape."""
        y_true, y_pred = simple_data
        evaluator = FastCPMMetrics(device=device)
        scores = evaluator.score(y_true, y_pred)

        N_runs = y_true.shape[1]
        expected_shape = (N_METRICS, N_MODELS, N_NETWORKS, N_runs)
        assert scores.shape == expected_shape, f"Expected {expected_shape}, got {scores.shape}"

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

    def test_score_models_wrapper(self, simple_data, device):
        """Test the score_models wrapper function for regression."""
        y_true, y_pred = simple_data
        scores = score_models(y_true, y_pred, task_type=TaskType.regression, device=device)

        N_runs = y_true.shape[1]
        expected_shape = (N_METRICS, N_MODELS, N_NETWORKS, N_runs)
        assert scores.shape == expected_shape
        assert isinstance(scores, torch.Tensor)

    def test_metrics_ordering(self, device):
        """Each row of the score tensor is the metric its `Metrics` member names.

        The previous version of this test asserted `scores[metric].shape`, which
        holds for any index in range and so said nothing about ordering. If the
        enum and the tensor's rows ever disagreed, every reported number would be
        silently mislabelled -- so compare against an independent reference per
        row, with predictions good enough that the metrics take distinct values.
        """
        rng = np.random.RandomState(0)
        N = 200
        y_true = rng.randn(N, 1).astype(np.float32)
        y_pred = np.empty((N, N_MODELS, N_NETWORKS, 1), dtype=np.float32)
        y_pred[:] = (y_true + rng.randn(N, 1).astype(np.float32) * 0.5)[
            :, None, None, :]

        scores = FastCPMMetrics(device=device).score(y_true, y_pred)

        yt, yp = y_true[:, 0], y_pred[:, 0, 0, 0]
        expected = {
            Metrics.explained_variance_score: explained_variance_score(yt, yp),
            Metrics.mean_squared_error: mean_squared_error(yt, yp),
            Metrics.mean_absolute_error: mean_absolute_error(yt, yp),
            Metrics.pearson_score: pearsonr(yt, yp)[0],
        }
        # The guard only means something if the rows are distinguishable.
        assert len(set(round(v, 4) for v in expected.values())) == len(expected)

        for metric, reference in expected.items():
            assert np.isclose(scores[metric, 0, 0, 0].cpu().item(), reference,
                              rtol=1e-4), (
                f"row {int(metric)} is not {metric.name}: got "
                f"{scores[metric, 0, 0, 0].cpu().item():.6f}, "
                f"expected {reference:.6f}")


class TestFastCPMMetricsVsSklearn:
    """Test that FastCPMMetrics produces the same results as scikit-learn."""

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
