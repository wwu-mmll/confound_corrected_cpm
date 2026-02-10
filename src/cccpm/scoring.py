import torch
import numpy as np
from cccpm.constants import Networks, Models, Metrics, TaskType


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

        # Reshape Truth: [N_samples, 1, 1, N_Runs] for broadcasting against Models/Networks/Runs
        truth_expanded = y_true.unsqueeze(1).unsqueeze(1)

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

        # We create a list of length len(Metrics), filling unused slots with zeros
        zero_placeholder = torch.zeros_like(mse)
        metrics_list = [zero_placeholder] * len(Metrics)
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


class FastCPMClassificationMetrics:
    """
    Fast GPU-accelerated computation of classification metrics for CPM.

    Computes accuracy, balanced accuracy, F1 score, and ROC AUC
    efficiently across all models, networks, and runs simultaneously.
    """

    def __init__(self, device='cuda'):
        self.device = device

    def score(self, y_true, y_pred_proba):
        """
        Calculates all classification metrics efficiently and returns a 4D Tensor.

        Args:
            y_true: [N_samples, N_runs] - Binary labels (0 or 1)
            y_pred_proba: Tensor of predicted probabilities with shape [N_samples, N_models, N_networks, N_runs]

        Returns:
            scores: Tensor of shape [N_metrics, N_models, N_networks, N_runs]
        """
        # 1. Setup Data
        y_true = torch.as_tensor(y_true, device=self.device, dtype=torch.float32)
        y_pred_proba = torch.as_tensor(y_pred_proba, device=self.device, dtype=torch.float32)

        # Reshape Truth: [N_samples, 1, 1, N_Runs] for broadcasting
        truth_expanded = y_true.unsqueeze(1).unsqueeze(1)

        # 2. Convert probabilities to binary predictions (threshold = 0.5)
        y_pred_binary = (y_pred_proba > 0.5).float()

        # 3. Calculate confusion matrix components
        # All shapes: [N_samples, N_models, N_networks, N_runs] after broadcasting
        tp = ((truth_expanded == 1) & (y_pred_binary == 1)).float()
        tn = ((truth_expanded == 0) & (y_pred_binary == 0)).float()
        fp = ((truth_expanded == 0) & (y_pred_binary == 1)).float()
        fn = ((truth_expanded == 1) & (y_pred_binary == 0)).float()

        # Sum over samples (dim=0) to get counts
        # Result shape: [N_models, N_networks, N_runs]
        tp_sum = tp.sum(dim=0)
        tn_sum = tn.sum(dim=0)
        fp_sum = fp.sum(dim=0)
        fn_sum = fn.sum(dim=0)

        # 4. Calculate metrics
        # Accuracy: (TP + TN) / (TP + TN + FP + FN)
        accuracy = (tp_sum + tn_sum) / (tp_sum + tn_sum + fp_sum + fn_sum + 1e-8)

        # Balanced Accuracy: (TPR + TNR) / 2
        tpr = tp_sum / (tp_sum + fn_sum + 1e-8)  # Sensitivity/Recall
        tnr = tn_sum / (tn_sum + fp_sum + 1e-8)  # Specificity
        balanced_accuracy = (tpr + tnr) / 2

        # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
        precision = tp_sum / (tp_sum + fp_sum + 1e-8)
        recall = tpr  # Same as TPR
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

        # ROC AUC: Approximation using trapezoidal rule
        # For exact AUC, we'd need to sort by predicted probabilities
        # Here we use a fast approximation based on the Mann-Whitney U statistic
        roc_auc = self._fast_roc_auc(truth_expanded, y_pred_proba)

        # 5. Stack Metrics into a single Tensor
        # Order must match constants.Metrics indices for classification metrics
        # Metrics.accuracy = 4
        # Metrics.balanced_accuracy = 5
        # Metrics.f1_score = 6
        # Metrics.roc_auc = 7

        # Fill unused regression metric slots with zeros
        zero_placeholder = torch.zeros_like(accuracy)
        metrics_list = [zero_placeholder] * len(Metrics)
        metrics_list[Metrics.accuracy] = accuracy
        metrics_list[Metrics.balanced_accuracy] = balanced_accuracy
        metrics_list[Metrics.f1_score] = f1_score
        metrics_list[Metrics.roc_auc] = roc_auc

        # Stack dim=0 -> Shape: [N_metrics, N_models, N_networks, N_runs]
        stacked_metrics = torch.stack(metrics_list, dim=0)

        return stacked_metrics

    def _fast_roc_auc(self, y_true, y_pred_proba):
        """
        Fast approximation of ROC AUC using Mann-Whitney U statistic.

        Args:
            y_true: [N_samples, 1, 1, N_runs]
            y_pred_proba: [N_samples, N_models, N_networks, N_runs]

        Returns:
            ROC AUC scores [N_models, N_networks, N_runs]
        """
        # Mask for positive and negative samples
        # Shape after broadcasting: [N_samples, N_models, N_networks, N_runs]
        pos_mask = (y_true == 1)
        neg_mask = (y_true == 0)

        # Count positives and negatives per run
        n_pos = pos_mask.sum(dim=0)  # [N_models, N_networks, N_runs]
        n_neg = neg_mask.sum(dim=0)  # [N_models, N_networks, N_runs]

        # For each negative sample, count how many positive samples have higher scores
        # This is the Mann-Whitney U statistic
        # Expand dimensions for broadcasting
        y_pred_expanded_pos = y_pred_proba.unsqueeze(1)  # [N_samples, 1, N_models, N_networks, N_runs]
        y_pred_expanded_neg = y_pred_proba.unsqueeze(0)  # [1, N_samples, N_models, N_networks, N_runs]

        pos_mask_expanded = pos_mask.unsqueeze(1)  # [N_samples, 1, N_models, N_networks, N_runs]
        neg_mask_expanded = neg_mask.unsqueeze(0)  # [1, N_samples, N_models, N_networks, N_runs]

        # Count pairs where positive > negative
        comparisons = (y_pred_expanded_pos > y_pred_expanded_neg).float()
        # Add 0.5 for ties
        ties = (y_pred_expanded_pos == y_pred_expanded_neg).float() * 0.5

        # Only count valid pairs (pos vs neg)
        valid_pairs = pos_mask_expanded & neg_mask_expanded
        weighted_sum = ((comparisons + ties) * valid_pairs).sum(dim=(0, 1))  # [N_models, N_networks, N_runs]

        # AUC = sum of ranks / (n_pos * n_neg)
        auc = weighted_sum / (n_pos * n_neg + 1e-8)

        return auc


def score_regression_models(y_true, y_pred, device='cuda', **kwargs):
    evaluator = FastCPMMetrics(device=device)
    return evaluator.score(y_true, y_pred)


def score_classification_models(y_true, y_pred_proba, device='cuda', **kwargs):
    """
    Score classification models using predicted probabilities.

    Args:
        y_true: True binary labels [N_samples, N_runs]
        y_pred_proba: Predicted probabilities [N_samples, N_models, N_networks, N_runs]
        device: Device for computation

    Returns:
        Tensor of metrics [N_metrics, N_models, N_networks, N_runs]
    """
    evaluator = FastCPMClassificationMetrics(device=device)
    return evaluator.score(y_true, y_pred_proba)


def score_models(y_true, y_pred, task_type, device='cuda', **kwargs):
    """
    Score models based on task type (regression or classification).

    Args:
        y_true: True labels [N_samples, N_runs]
        y_pred: Predictions [N_samples, N_models, N_networks, N_runs]
                For classification, should be probabilities
        task_type: TaskType.regression or TaskType.classification
        device: Device for computation

    Returns:
        Tensor of metrics [N_metrics, N_models, N_networks, N_runs]
    """
    if task_type == TaskType.regression:
        return score_regression_models(y_true, y_pred, device=device, **kwargs)
    elif task_type == TaskType.classification:
        return score_classification_models(y_true, y_pred, device=device, **kwargs)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")