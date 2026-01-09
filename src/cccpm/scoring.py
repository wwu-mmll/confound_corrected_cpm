import torch


class FastCPMMetrics:
    # Define order to map indices back to strings later
    MODELS = ['connectome', 'residuals', 'full', 'covariates']
    NETWORKS = ['positive', 'negative', 'both']

    def __init__(self, device='cuda'):
        self.device = device

    def score(self, y_true, y_pred_dict):
        """
        Calculates all metrics efficiently, then returns them in the original nested format.
        Structure: scores[model][network][metric] -> Tensor[N_perms]
        """
        # 1. Setup Data
        y_true = torch.as_tensor(y_true, device=self.device, dtype=torch.float32)

        # Stack into 4D Tensor: [Samples, Perms, Models, Networks]
        preds_stack = self._stack_predictions(y_pred_dict, y_true.shape)

        # Reshape Truth: [N, P, 1, 1] for broadcasting
        truth_expanded = y_true.unsqueeze(-1).unsqueeze(-1)

        # 2. Vectorized Calculations (All metrics at once)
        # Result shapes will be [Perms, Models, Networks]
        metrics_tensors = {}

        # MSE
        metrics_tensors['mean_squared_error'] = torch.mean((truth_expanded - preds_stack) ** 2, dim=0)

        # MAE
        metrics_tensors['mean_absolute_error'] = torch.mean(torch.abs(truth_expanded - preds_stack), dim=0)

        # Explained Variance
        y_diff = truth_expanded - preds_stack
        var_true = torch.var(truth_expanded, dim=0, unbiased=False)
        var_diff = torch.var(y_diff, dim=0, unbiased=False)
        metrics_tensors['explained_variance_score'] = 1 - (var_diff / (var_true + 1e-8))

        # Pearson
        metrics_tensors['pearson_score'] = self._pearson_vectorized(truth_expanded, preds_stack)

        # Spearman (Rank -> Pearson)
        rank_true = truth_expanded.argsort(dim=0).argsort(dim=0).float()
        rank_pred = preds_stack.argsort(dim=0).argsort(dim=0).float()
        metrics_tensors['spearman_score'] = self._pearson_vectorized(rank_true, rank_pred)

        # 3. Unpack into Nested Dictionary Structure
        # Target: scores[model][network][metric]
        final_scores = {}

        for m_idx, model in enumerate(self.MODELS):
            final_scores[model] = {}
            for n_idx, net in enumerate(self.NETWORKS):
                final_scores[model][net] = {}

                for metric_name, tensor_data in metrics_tensors.items():
                    # Extract specific slice: [Perms]
                    final_scores[model][net][metric_name] = tensor_data[:, m_idx, n_idx]

        return final_scores

    def _stack_predictions(self, preds, shape):
        """
        Converts input dict to [N, P, M, Net] tensor.
        Handles 'covariates' by duplicating it across network dimension.
        """
        N, P = shape
        stack = torch.empty((N, P, len(self.MODELS), len(self.NETWORKS)), device=self.device)

        for m_idx, model in enumerate(self.MODELS):
            for n_idx, net in enumerate(self.NETWORKS):
                if model == 'covariates':
                    # Covariates model is usually network-agnostic, but your structure 
                    # requires keys for pos/neg/both. We duplicate the prediction.
                    tensor = preds['covariates']
                else:
                    tensor = preds[f"{model}_{net}"]

                # Ensure tensor is on correct device
                if not isinstance(tensor, torch.Tensor):
                    tensor = torch.as_tensor(tensor, device=self.device)

                stack[:, :, m_idx, n_idx] = tensor
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


# --- Wrapper to replace your original function ---

def score_regression_models(y_true, y_pred, device='cuda', **kwargs):
    """
    Drop-in replacement for your original function.
    Returns: dict[model][network][metric] -> Tensor of shape [N_perms]
    """
    evaluator = FastCPMMetrics(device=device)
    return evaluator.score(y_true, y_pred)


regression_metrics_functions = {
    'mean_squared_error': None,
    'mean_absolute_error': None,
    'explained_variance_score': None,
    'pearson_score': None,
    'spearman_score': None}

regression_metrics = list(regression_metrics_functions.keys())
