import pytest
import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score
)
from cccpm.scoring import FastCPMMetrics, score_regression_models
from cccpm.constants import Networks, Models, Metrics


@pytest.fixture
def device():
    """Returns 'cuda' if available, else 'cpu'."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def simple_data():
    """Creates simple synthetic data for testing."""
    N_samples = 50
    N_perms = 3

    # Ground truth
    y_true = np.random.randn(N_samples, N_perms).astype(np.float32)

    # Predictions for different model/network combinations
    y_pred_dict = {
        'connectome_positive': np.random.randn(N_samples, N_perms).astype(np.float32),
        'connectome_negative': np.random.randn(N_samples, N_perms).astype(np.float32),
        'connectome_both': np.random.randn(N_samples, N_perms).astype(np.float32),
        'covariates': np.random.randn(N_samples, N_perms).astype(np.float32),
        'full_positive': np.random.randn(N_samples, N_perms).astype(np.float32),
        'full_negative': np.random.randn(N_samples, N_perms).astype(np.float32),
        'full_both': np.random.randn(N_samples, N_perms).astype(np.float32),
        'residuals_positive': np.random.randn(N_samples, N_perms).astype(np.float32),
        'residuals_negative': np.random.randn(N_samples, N_perms).astype(np.float32),
        'residuals_both': np.random.randn(N_samples, N_perms).astype(np.float32),
        'increment_positive': np.random.randn(N_samples, N_perms).astype(np.float32),
        'increment_negative': np.random.randn(N_samples, N_perms).astype(np.float32),
        'increment_both': np.random.randn(N_samples, N_perms).astype(np.float32),
    }

    return y_true, y_pred_dict


@pytest.fixture
def perfect_predictions():
    """Creates data where predictions perfectly match ground truth."""
    N_samples = 30
    N_perms = 2

    y_true = np.random.randn(N_samples, N_perms).astype(np.float32)

    # All predictions are identical to ground truth
    y_pred_dict = {
        'connectome_positive': y_true.copy(),
        'connectome_negative': y_true.copy(),
        'connectome_both': y_true.copy(),
        'covariates': y_true.copy(),
        'full_positive': y_true.copy(),
        'full_negative': y_true.copy(),
        'full_both': y_true.copy(),
        'residuals_positive': y_true.copy(),
        'residuals_negative': y_true.copy(),
        'residuals_both': y_true.copy(),
        'increment_positive': y_true.copy(),
        'increment_negative': y_true.copy(),
        'increment_both': y_true.copy(),
    }

    return y_true, y_pred_dict


class TestFastCPMMetrics:

    def test_initialization(self, device):
        """Test that FastCPMMetrics initializes correctly."""
        evaluator = FastCPMMetrics(device=device)
        assert evaluator.device == device

    def test_score_output_shape(self, simple_data, device):
        """Test that score returns correct output shape."""
        y_true, y_pred_dict = simple_data
        evaluator = FastCPMMetrics(device=device)

        scores = evaluator.score(y_true, y_pred_dict)

        # Output shape should be [N_models, N_networks, N_metrics, N_perms]
        expected_shape = (len(Models), len(Networks), len(Metrics), y_true.shape[1])
        assert scores.shape == expected_shape, f"Expected {expected_shape}, got {scores.shape}"

    def test_score_returns_tensor(self, simple_data, device):
        """Test that score returns a torch.Tensor."""
        y_true, y_pred_dict = simple_data
        evaluator = FastCPMMetrics(device=device)

        scores = evaluator.score(y_true, y_pred_dict)

        assert isinstance(scores, torch.Tensor)

    def test_perfect_predictions_pearson(self, perfect_predictions, device):
        """Test that perfect predictions yield Pearson correlation of 1."""
        y_true, y_pred_dict = perfect_predictions
        evaluator = FastCPMMetrics(device=device)

        scores = evaluator.score(y_true, y_pred_dict)

        # Extract Pearson scores: scores[model_idx, network_idx, Metrics.pearson_score, perm_idx]
        pearson_scores = scores[:, :, Metrics.pearson_score, :]

        # All Pearson correlations should be very close to 1
        assert torch.allclose(pearson_scores, torch.ones_like(pearson_scores), atol=1e-5)

    def test_perfect_predictions_mse(self, perfect_predictions, device):
        """Test that perfect predictions yield MSE of 0."""
        y_true, y_pred_dict = perfect_predictions
        evaluator = FastCPMMetrics(device=device)

        scores = evaluator.score(y_true, y_pred_dict)

        # Extract MSE scores
        mse_scores = scores[:, :, Metrics.mean_squared_error, :]

        # All MSE should be very close to 0
        assert torch.allclose(mse_scores, torch.zeros_like(mse_scores), atol=1e-5)

    def test_perfect_predictions_mae(self, perfect_predictions, device):
        """Test that perfect predictions yield MAE of 0."""
        y_true, y_pred_dict = perfect_predictions
        evaluator = FastCPMMetrics(device=device)

        scores = evaluator.score(y_true, y_pred_dict)

        # Extract MAE scores
        mae_scores = scores[:, :, Metrics.mean_absolute_error, :]

        # All MAE should be very close to 0
        assert torch.allclose(mae_scores, torch.zeros_like(mae_scores), atol=1e-5)

    def test_perfect_predictions_explained_variance(self, perfect_predictions, device):
        """Test that perfect predictions yield explained variance of 1."""
        y_true, y_pred_dict = perfect_predictions
        evaluator = FastCPMMetrics(device=device)

        scores = evaluator.score(y_true, y_pred_dict)

        # Extract explained variance scores
        ev_scores = scores[:, :, Metrics.explained_variance_score, :]

        # All explained variance should be very close to 1
        assert torch.allclose(ev_scores, torch.ones_like(ev_scores), atol=1e-5)

    def test_mse_calculation(self, device):
        """Test MSE calculation with known values."""
        N = 10
        P = 1

        y_true = torch.ones(N, P) * 2.0
        y_pred = torch.ones(N, P) * 4.0  # Difference of 2.0

        y_pred_dict = {
            'connectome_positive': y_pred.cpu().numpy(),
            'connectome_negative': y_pred.cpu().numpy(),
            'connectome_both': y_pred.cpu().numpy(),
            'covariates': y_pred.cpu().numpy(),
            'full_positive': y_pred.cpu().numpy(),
            'full_negative': y_pred.cpu().numpy(),
            'full_both': y_pred.cpu().numpy(),
            'residuals_positive': y_pred.cpu().numpy(),
            'residuals_negative': y_pred.cpu().numpy(),
            'residuals_both': y_pred.cpu().numpy(),
            'increment_positive': y_pred.cpu().numpy(),
            'increment_negative': y_pred.cpu().numpy(),
            'increment_both': y_pred.cpu().numpy(),
        }

        evaluator = FastCPMMetrics(device=device)
        scores = evaluator.score(y_true.cpu().numpy(), y_pred_dict)

        # MSE should be (2.0)^2 = 4.0
        mse_scores = scores[:, :, Metrics.mean_squared_error, :]
        expected_mse = torch.full_like(mse_scores, 4.0)

        assert torch.allclose(mse_scores, expected_mse, atol=1e-5)

    def test_mae_calculation(self, device):
        """Test MAE calculation with known values."""
        N = 10
        P = 1

        y_true = torch.ones(N, P) * 2.0
        y_pred = torch.ones(N, P) * 5.0  # Difference of 3.0

        y_pred_dict = {
            'connectome_positive': y_pred.cpu().numpy(),
            'connectome_negative': y_pred.cpu().numpy(),
            'connectome_both': y_pred.cpu().numpy(),
            'covariates': y_pred.cpu().numpy(),
            'full_positive': y_pred.cpu().numpy(),
            'full_negative': y_pred.cpu().numpy(),
            'full_both': y_pred.cpu().numpy(),
            'residuals_positive': y_pred.cpu().numpy(),
            'residuals_negative': y_pred.cpu().numpy(),
            'residuals_both': y_pred.cpu().numpy(),
            'increment_positive': y_pred.cpu().numpy(),
            'increment_negative': y_pred.cpu().numpy(),
            'increment_both': y_pred.cpu().numpy(),
        }

        evaluator = FastCPMMetrics(device=device)
        scores = evaluator.score(y_true.cpu().numpy(), y_pred_dict)

        # MAE should be 3.0
        mae_scores = scores[:, :, Metrics.mean_absolute_error, :]
        expected_mae = torch.full_like(mae_scores, 3.0)

        assert torch.allclose(mae_scores, expected_mae, atol=1e-5)

    def test_covariates_shared_across_networks(self, device):
        """Test that covariates predictions are shared across all networks."""
        N = 20
        P = 2

        y_true = np.random.randn(N, P).astype(np.float32)
        covariates_pred = np.random.randn(N, P).astype(np.float32)

        y_pred_dict = {
            'connectome_positive': np.random.randn(N, P).astype(np.float32),
            'connectome_negative': np.random.randn(N, P).astype(np.float32),
            'connectome_both': np.random.randn(N, P).astype(np.float32),
            'covariates': covariates_pred,
            'full_positive': np.random.randn(N, P).astype(np.float32),
            'full_negative': np.random.randn(N, P).astype(np.float32),
            'full_both': np.random.randn(N, P).astype(np.float32),
            'residuals_positive': np.random.randn(N, P).astype(np.float32),
            'residuals_negative': np.random.randn(N, P).astype(np.float32),
            'residuals_both': np.random.randn(N, P).astype(np.float32),
            'increment_positive': np.random.randn(N, P).astype(np.float32),
            'increment_negative': np.random.randn(N, P).astype(np.float32),
            'increment_both': np.random.randn(N, P).astype(np.float32),
        }

        evaluator = FastCPMMetrics(device=device)
        scores = evaluator.score(y_true, y_pred_dict)

        # Get scores for covariates model across all networks
        covariates_scores = scores[Models.covariates, :, :, :]

        # All networks should have identical scores for the covariates model
        positive_scores = covariates_scores[Networks.positive, :, :]
        negative_scores = covariates_scores[Networks.negative, :, :]
        both_scores = covariates_scores[Networks.both, :, :]

        assert torch.allclose(positive_scores, negative_scores, atol=1e-5)
        assert torch.allclose(positive_scores, both_scores, atol=1e-5)

    def test_handles_torch_and_numpy_input(self, device):
        """Test that the method handles both torch.Tensor and numpy.ndarray inputs."""
        N = 15
        P = 2

        # NumPy inputs
        y_true_np = np.random.randn(N, P).astype(np.float32)
        y_pred_np = np.random.randn(N, P).astype(np.float32)

        y_pred_dict_np = {
            'connectome_positive': y_pred_np,
            'connectome_negative': y_pred_np,
            'connectome_both': y_pred_np,
            'covariates': y_pred_np,
            'full_positive': y_pred_np,
            'full_negative': y_pred_np,
            'full_both': y_pred_np,
            'residuals_positive': y_pred_np,
            'residuals_negative': y_pred_np,
            'residuals_both': y_pred_np,
            'increment_positive': y_pred_np,
            'increment_negative': y_pred_np,
            'increment_both': y_pred_np,
        }

        # Torch inputs
        y_true_torch = torch.from_numpy(y_true_np)
        y_pred_torch = torch.from_numpy(y_pred_np)

        y_pred_dict_torch = {
            'connectome_positive': y_pred_torch,
            'connectome_negative': y_pred_torch,
            'connectome_both': y_pred_torch,
            'covariates': y_pred_torch,
            'full_positive': y_pred_torch,
            'full_negative': y_pred_torch,
            'full_both': y_pred_torch,
            'residuals_positive': y_pred_torch,
            'residuals_negative': y_pred_torch,
            'residuals_both': y_pred_torch,
            'increment_positive': y_pred_torch,
            'increment_negative': y_pred_torch,
            'increment_both': y_pred_torch,
        }

        evaluator = FastCPMMetrics(device=device)

        scores_np = evaluator.score(y_true_np, y_pred_dict_np)
        scores_torch = evaluator.score(y_true_torch, y_pred_dict_torch)

        # Both should yield the same results
        assert torch.allclose(scores_np, scores_torch, atol=1e-5)

    def test_pearson_negative_correlation(self, device):
        """Test Pearson correlation with negatively correlated predictions."""
        N = 50
        P = 1

        y_true = torch.linspace(0, 10, N).unsqueeze(1)
        y_pred = -y_true  # Perfect negative correlation

        y_pred_dict = {
            'connectome_positive': y_pred.cpu().numpy(),
            'connectome_negative': y_pred.cpu().numpy(),
            'connectome_both': y_pred.cpu().numpy(),
            'covariates': y_pred.cpu().numpy(),
            'full_positive': y_pred.cpu().numpy(),
            'full_negative': y_pred.cpu().numpy(),
            'full_both': y_pred.cpu().numpy(),
            'residuals_positive': y_pred.cpu().numpy(),
            'residuals_negative': y_pred.cpu().numpy(),
            'residuals_both': y_pred.cpu().numpy(),
            'increment_positive': y_pred.cpu().numpy(),
            'increment_negative': y_pred.cpu().numpy(),
            'increment_both': y_pred.cpu().numpy(),
        }

        evaluator = FastCPMMetrics(device=device)
        scores = evaluator.score(y_true.cpu().numpy(), y_pred_dict)

        pearson_scores = scores[:, :, Metrics.pearson_score, :]
        expected_pearson = torch.full_like(pearson_scores, -1.0)

        assert torch.allclose(pearson_scores, expected_pearson, atol=1e-5)

    def test_multiple_permutations(self, device):
        """Test that multiple permutations are handled correctly."""
        N = 30
        P = 5  # Multiple permutations

        y_true = np.random.randn(N, P).astype(np.float32)
        y_pred_dict = {
            'connectome_positive': np.random.randn(N, P).astype(np.float32),
            'connectome_negative': np.random.randn(N, P).astype(np.float32),
            'connectome_both': np.random.randn(N, P).astype(np.float32),
            'covariates': np.random.randn(N, P).astype(np.float32),
            'full_positive': np.random.randn(N, P).astype(np.float32),
            'full_negative': np.random.randn(N, P).astype(np.float32),
            'full_both': np.random.randn(N, P).astype(np.float32),
            'residuals_positive': np.random.randn(N, P).astype(np.float32),
            'residuals_negative': np.random.randn(N, P).astype(np.float32),
            'residuals_both': np.random.randn(N, P).astype(np.float32),
            'increment_positive': np.random.randn(N, P).astype(np.float32),
            'increment_negative': np.random.randn(N, P).astype(np.float32),
            'increment_both': np.random.randn(N, P).astype(np.float32),
        }

        evaluator = FastCPMMetrics(device=device)
        scores = evaluator.score(y_true, y_pred_dict)

        # Verify shape accounts for all permutations
        assert scores.shape[3] == P

        # Verify each permutation has different scores (very unlikely to be identical)
        for model in Models:
            for network in Networks:
                for metric in Metrics:
                    perm_scores = scores[model, network, metric, :]
                    # Check that not all permutation scores are identical
                    if P > 1:
                        assert not torch.allclose(perm_scores, perm_scores[0].expand_as(perm_scores))

    def test_score_regression_models_wrapper(self, simple_data, device):
        """Test the score_regression_models wrapper function."""
        y_true, y_pred_dict = simple_data

        scores = score_regression_models(y_true, y_pred_dict, device=device)

        # Should return the same shape as FastCPMMetrics.score
        expected_shape = (len(Models), len(Networks), len(Metrics), y_true.shape[1])
        assert scores.shape == expected_shape
        assert isinstance(scores, torch.Tensor)

    def test_metrics_ordering(self, simple_data, device):
        """Test that metrics are correctly ordered according to constants.Metrics."""
        y_true, y_pred_dict = simple_data
        evaluator = FastCPMMetrics(device=device)

        scores = evaluator.score(y_true, y_pred_dict)

        # Verify we have scores for all metrics
        assert scores.shape[2] == len(Metrics)

        # Verify each metric index corresponds correctly
        # (We can't check values without knowing them, but we can verify the structure)
        for metric in Metrics:
            metric_scores = scores[:, :, metric, :]
            assert metric_scores.shape == (len(Models), len(Networks), y_true.shape[1])


class TestFastCPMMetricsVsSklearn:
    """Test that FastCPMMetrics produces the same results as scikit-learn."""

    def test_mse_vs_sklearn(self, simple_data, device):
        """Verify MSE matches sklearn.metrics.mean_squared_error."""
        y_true, y_pred_dict = simple_data
        evaluator = FastCPMMetrics(device=device)

        scores = evaluator.score(y_true, y_pred_dict)
        mse_scores = scores[:, :, Metrics.mean_squared_error, :].cpu().numpy()

        # Check each model/network/permutation combination
        for model in Models:
            for network in Networks:
                for perm_idx in range(y_true.shape[1]):
                    # Get predictions for this specific combination
                    if model.name == 'covariates':
                        key = 'covariates'
                    else:
                        key = f"{model.name}_{network.name}"

                    y_pred = y_pred_dict[key][:, perm_idx]
                    y_true_perm = y_true[:, perm_idx]

                    # Calculate using sklearn
                    sklearn_mse = mean_squared_error(y_true_perm, y_pred)

                    # Get our calculated MSE
                    our_mse = mse_scores[model, network, perm_idx]

                    assert np.isclose(our_mse, sklearn_mse, rtol=1e-5), \
                        f"MSE mismatch for {model.name}_{network.name} perm {perm_idx}: " \
                        f"ours={our_mse}, sklearn={sklearn_mse}"

    def test_mae_vs_sklearn(self, simple_data, device):
        """Verify MAE matches sklearn.metrics.mean_absolute_error."""
        y_true, y_pred_dict = simple_data
        evaluator = FastCPMMetrics(device=device)

        scores = evaluator.score(y_true, y_pred_dict)
        mae_scores = scores[:, :, Metrics.mean_absolute_error, :].cpu().numpy()

        # Check each model/network/permutation combination
        for model in Models:
            for network in Networks:
                for perm_idx in range(y_true.shape[1]):
                    # Get predictions for this specific combination
                    if model.name == 'covariates':
                        key = 'covariates'
                    else:
                        key = f"{model.name}_{network.name}"

                    y_pred = y_pred_dict[key][:, perm_idx]
                    y_true_perm = y_true[:, perm_idx]

                    # Calculate using sklearn
                    sklearn_mae = mean_absolute_error(y_true_perm, y_pred)

                    # Get our calculated MAE
                    our_mae = mae_scores[model, network, perm_idx]

                    assert np.isclose(our_mae, sklearn_mae, rtol=1e-5), \
                        f"MAE mismatch for {model.name}_{network.name} perm {perm_idx}: " \
                        f"ours={our_mae}, sklearn={sklearn_mae}"

    def test_explained_variance_vs_sklearn(self, simple_data, device):
        """Verify explained variance matches sklearn.metrics.explained_variance_score."""
        y_true, y_pred_dict = simple_data
        evaluator = FastCPMMetrics(device=device)

        scores = evaluator.score(y_true, y_pred_dict)
        ev_scores = scores[:, :, Metrics.explained_variance_score, :].cpu().numpy()

        # Check each model/network/permutation combination
        for model in Models:
            for network in Networks:
                for perm_idx in range(y_true.shape[1]):
                    # Get predictions for this specific combination
                    if model.name == 'covariates':
                        key = 'covariates'
                    else:
                        key = f"{model.name}_{network.name}"

                    y_pred = y_pred_dict[key][:, perm_idx]
                    y_true_perm = y_true[:, perm_idx]

                    # Calculate using sklearn
                    sklearn_ev = explained_variance_score(y_true_perm, y_pred)

                    # Get our calculated explained variance
                    our_ev = ev_scores[model, network, perm_idx]

                    assert np.isclose(our_ev, sklearn_ev, rtol=1e-5), \
                        f"Explained variance mismatch for {model.name}_{network.name} perm {perm_idx}: " \
                        f"ours={our_ev}, sklearn={sklearn_ev}"

    def test_pearson_vs_scipy(self, simple_data, device):
        """Verify Pearson correlation matches scipy.stats.pearsonr."""
        y_true, y_pred_dict = simple_data
        evaluator = FastCPMMetrics(device=device)

        scores = evaluator.score(y_true, y_pred_dict)
        pearson_scores = scores[:, :, Metrics.pearson_score, :].cpu().numpy()

        # Check each model/network/permutation combination
        for model in Models:
            for network in Networks:
                for perm_idx in range(y_true.shape[1]):
                    # Get predictions for this specific combination
                    if model.name == 'covariates':
                        key = 'covariates'
                    else:
                        key = f"{model.name}_{network.name}"

                    y_pred = y_pred_dict[key][:, perm_idx]
                    y_true_perm = y_true[:, perm_idx]

                    # Calculate using scipy
                    scipy_pearson, _ = pearsonr(y_true_perm, y_pred)

                    # Get our calculated Pearson
                    our_pearson = pearson_scores[model, network, perm_idx]

                    assert np.isclose(our_pearson, scipy_pearson, rtol=1e-5), \
                        f"Pearson mismatch for {model.name}_{network.name} perm {perm_idx}: " \
                        f"ours={our_pearson}, scipy={scipy_pearson}"

    def test_all_metrics_vs_sklearn_random_data(self, device):
        """Comprehensive test with random data comparing all metrics against sklearn/scipy."""
        np.random.seed(42)
        N = 100
        P = 4

        y_true = np.random.randn(N, P).astype(np.float32)

        # Create predictions with varying quality
        y_pred_dict = {}
        for model in Models:
            for network in Networks:
                if model.name == 'covariates':
                    key = 'covariates'
                    # Only create once
                    if key not in y_pred_dict:
                        y_pred_dict[key] = y_true + np.random.randn(N, P).astype(np.float32) * 0.5
                else:
                    key = f"{model.name}_{network.name}"
                    # Add varying levels of noise
                    noise_level = np.random.uniform(0.1, 2.0)
                    y_pred_dict[key] = y_true + np.random.randn(N, P).astype(np.float32) * noise_level

        evaluator = FastCPMMetrics(device=device)
        scores = evaluator.score(y_true, y_pred_dict)

        # Extract all metric scores
        mse_scores = scores[:, :, Metrics.mean_squared_error, :].cpu().numpy()
        mae_scores = scores[:, :, Metrics.mean_absolute_error, :].cpu().numpy()
        ev_scores = scores[:, :, Metrics.explained_variance_score, :].cpu().numpy()
        pearson_scores = scores[:, :, Metrics.pearson_score, :].cpu().numpy()

        # Verify all against sklearn/scipy
        for model in Models:
            for network in Networks:
                for perm_idx in range(P):
                    if model.name == 'covariates':
                        key = 'covariates'
                    else:
                        key = f"{model.name}_{network.name}"

                    y_pred = y_pred_dict[key][:, perm_idx]
                    y_true_perm = y_true[:, perm_idx]

                    # MSE
                    sklearn_mse = mean_squared_error(y_true_perm, y_pred)
                    assert np.isclose(mse_scores[model, network, perm_idx], sklearn_mse, rtol=1e-5)

                    # MAE
                    sklearn_mae = mean_absolute_error(y_true_perm, y_pred)
                    assert np.isclose(mae_scores[model, network, perm_idx], sklearn_mae, rtol=1e-5)

                    # Explained Variance
                    sklearn_ev = explained_variance_score(y_true_perm, y_pred)
                    assert np.isclose(ev_scores[model, network, perm_idx], sklearn_ev, rtol=1e-5)

                    # Pearson
                    scipy_pearson, _ = pearsonr(y_true_perm, y_pred)
                    assert np.isclose(pearson_scores[model, network, perm_idx], scipy_pearson, rtol=1e-5)

    def test_edge_cases_vs_sklearn(self, device):
        """Test edge cases like constant predictions against sklearn."""
        N = 50
        P = 2

        # Edge case 1: Constant predictions
        y_true = np.random.randn(N, P).astype(np.float32)
        constant_pred = np.ones((N, P), dtype=np.float32) * 5.0

        y_pred_dict = {
            'connectome_positive': constant_pred,
            'connectome_negative': constant_pred,
            'connectome_both': constant_pred,
            'covariates': constant_pred,
            'full_positive': constant_pred,
            'full_negative': constant_pred,
            'full_both': constant_pred,
            'residuals_positive': constant_pred,
            'residuals_negative': constant_pred,
            'residuals_both': constant_pred,
            'increment_positive': constant_pred,
            'increment_negative': constant_pred,
            'increment_both': constant_pred,
        }

        evaluator = FastCPMMetrics(device=device)
        scores = evaluator.score(y_true, y_pred_dict)

        # Check MSE and MAE (should be computable)
        for perm_idx in range(P):
            y_true_perm = y_true[:, perm_idx]
            y_pred_perm = constant_pred[:, perm_idx]

            sklearn_mse = mean_squared_error(y_true_perm, y_pred_perm)
            sklearn_mae = mean_absolute_error(y_true_perm, y_pred_perm)

            # Check one model/network combination (all should be same due to constant pred)
            our_mse = scores[0, 0, Metrics.mean_squared_error, perm_idx].cpu().numpy()
            our_mae = scores[0, 0, Metrics.mean_absolute_error, perm_idx].cpu().numpy()

            assert np.isclose(our_mse, sklearn_mse, rtol=1e-5)
            assert np.isclose(our_mae, sklearn_mae, rtol=1e-5)
