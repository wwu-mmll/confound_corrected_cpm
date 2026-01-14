import torch
from cccpm.constants import Networks, Models, Metrics


class FastCPMMetrics:

    def __init__(self, device='cuda'):
        self.device = device

    def score(self, y_true, y_pred):
        """
        Calculates all metrics efficiently and returns a 4D Tensor.

        Args:
            y_true: [N_samples, N_runs]
            y_pred: Tensor of predictions with shape [N_samples, N_models, N_networks, N_runs]

        Returns:
            scores: Tensor of shape [N_metrics, N_models, N_networks, N_runs]
        """
        # 1. Setup Data
        y_true = torch.as_tensor(y_true, device=self.device, dtype=torch.float32)
        y_pred = torch.as_tensor(y_pred, device=self.device, dtype=torch.float32)

        # Reshape Truth: [N_samples, N_runs, 1, 1] for broadcasting against Models/Networks
        truth_expanded = y_true.unsqueeze(2).unsqueeze(2)

        # 2. Vectorized Calculations
        # Result of each calc is shape: [N_models, N_networks, N_runs] (Reduced over Samples dim=0)

        # MSE
        mse = torch.mean((truth_expanded - y_pred) ** 2, dim=0)

        # MAE
        mae = torch.mean(torch.abs(truth_expanded - y_pred), dim=0)

        # Explained Variance
        y_diff = truth_expanded - y_pred
        var_true = torch.var(truth_expanded, dim=0, unbiased=False)
        var_diff = torch.var(y_diff, dim=0, unbiased=False)
        expl_var = 1 - (var_diff / (var_true + 1e-8))

        # Pearson
        pearson = self._pearson_vectorized(truth_expanded, y_pred)

        # 3. Stack Metrics into a single Tensor
        # CRITICAL: The order here MUST match the integer values in constants.Metrics
        # Metrics.explained_variance_score = 0
        # Metrics.pearson_score = 1
        # Metrics.mean_squared_error = 2
        # Metrics.mean_absolute_error = 3

        # We create a list of length len(Metrics) to ensure correct slotting
        metrics_list = [None] * len(Metrics)
        metrics_list[Metrics.explained_variance_score] = expl_var
        metrics_list[Metrics.pearson_score] = pearson
        metrics_list[Metrics.mean_squared_error] = mse
        metrics_list[Metrics.mean_absolute_error] = mae

        # Stack dim=0 -> Shape: [N_metrics, N_models, N_networks, N_runs]
        stacked_metrics = torch.stack(metrics_list, dim=0)

        return stacked_metrics

    def _pearson_vectorized(self, x, y):
        # Center
        x_c = x - x.mean(dim=0, keepdim=True)
        y_c = y - y.mean(dim=0, keepdim=True)
        # Correlation
        cov = torch.sum(x_c * y_c, dim=0)
        std_x = torch.sqrt(torch.sum(x_c ** 2, dim=0))
        std_y = torch.sqrt(torch.sum(y_c ** 2, dim=0))
        return cov / (std_x * std_y + 1e-8)


def score_regression_models(y_true, y_pred, device='cuda', **kwargs):
    evaluator = FastCPMMetrics(device=device)
    return evaluator.score(y_true, y_pred)