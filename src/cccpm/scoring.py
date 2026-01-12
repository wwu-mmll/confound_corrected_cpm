import torch
from cccpm.constants import Networks, Models, Metrics


class FastCPMMetrics:

    def __init__(self, device='cuda'):
        self.device = device

    def score(self, y_true, y_pred_dict):
        """
        Calculates all metrics efficiently and returns a 4D Tensor.

        Args:
            y_true: [N_samples, N_perms]
            y_pred_dict: Dictionary of predictions. Keys e.g., 'connectome_positive'.

        Returns:
            scores: Tensor of shape [N_models, N_networks, N_metrics, N_perms]
        """
        # 1. Setup Data
        y_true = torch.as_tensor(y_true, device=self.device, dtype=torch.float32)

        # Stack into 4D Tensor: [Samples, Perms, Models, Networks]
        preds_stack = self._stack_predictions(y_pred_dict, y_true.shape)

        # Reshape Truth: [N, P, 1, 1] for broadcasting against Models/Networks
        truth_expanded = y_true.unsqueeze(-1).unsqueeze(-1)

        # 2. Vectorized Calculations
        # Result of each calc is shape: [Perms, Models, Networks] (Reduced over Samples dim=0)

        # MSE
        mse = torch.mean((truth_expanded - preds_stack) ** 2, dim=0)

        # MAE
        mae = torch.mean(torch.abs(truth_expanded - preds_stack), dim=0)

        # Explained Variance
        y_diff = truth_expanded - preds_stack
        var_true = torch.var(truth_expanded, dim=0, unbiased=False)
        var_diff = torch.var(y_diff, dim=0, unbiased=False)
        expl_var = 1 - (var_diff / (var_true + 1e-8))

        # Pearson
        pearson = self._pearson_vectorized(truth_expanded, preds_stack)

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

        # Stack dim=0 -> Shape: [Metrics, Perms, Models, Networks]
        stacked_metrics = torch.stack(metrics_list, dim=0)

        # 4. Permute to desired output shape: [Models, Networks, Metrics, Perms]
        # Current indices: 0:Metrics, 1:Perms, 2:Models, 3:Networks
        # Target indices:  2:Models,  3:Networks, 0:Metrics, 1:Perms
        final_tensor = stacked_metrics.permute(2, 3, 0, 1)

        return final_tensor

    def _stack_predictions(self, preds, shape):
        """
        Converts input dict to [N, P, M, Net] tensor.
        Duplicates 'covariates' predictions across network dimension.
        """
        N, P = shape
        # Pre-allocate tensor based on Enum lengths
        stack = torch.empty((N, P, len(Models), len(Networks)), device=self.device)

        # Iterate over the Enums directly
        for model in Models:
            for net in Networks:
                m_idx = model.value
                n_idx = net.value

                # Construct the key as expected in the dictionary
                # E.g., Models.connectome (0) -> "connectome"
                # E.g., Networks.positive (0) -> "positive"

                if model.name == 'covariates':
                    # Special Case: Covariates model usually has one key, shared across networks
                    key = 'covariates'
                else:
                    key = f"{model.name}_{net.name}"

                # Retrieve and assign
                if key in preds:
                    tensor = preds[key]

                    # Ensure tensor is on correct device
                    if not isinstance(tensor, torch.Tensor):
                        tensor = torch.as_tensor(tensor, device=self.device)

                    stack[:, :, m_idx, n_idx] = tensor
                else:
                    # Fallback if key is missing (optional safety)
                    # print(f"Warning: Key {key} not found in predictions.")
                    pass

        return stack

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