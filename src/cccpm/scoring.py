"""Vectorised CPM metrics, one implementation per task type, shape-generic over the batch axes between samples and runs.

``y_pred`` is [N_samples, *batch, N_runs]; ``y_true`` is always [N_samples,
N_runs] and broadcasts across the batch axes. A fold-batched variant was
removed: measured (scripts/benchmark_batching.py) at 1.09x GPU speed for 7.2x
VRAM, and 1.85x *slower* on CPU.
"""
import torch

from cccpm.constants import Metrics, TaskType


def _broadcast_truth(y_true, y_pred):
    """Reshape ``y_true`` ``[N, R]`` to broadcast against ``y_pred`` ``[N, *batch, R]``."""
    n_batch_axes = y_pred.dim() - 2
    return y_true.view(y_true.shape[0], *([1] * n_batch_axes), y_true.shape[-1])


def _average_ranks(x, dim=0):
    """1-based ranks along `dim`, tied values sharing their mean rank (matches scipy.stats.rankdata's 'average' method).

    Avoids materialising a pairwise tensor: a cummax/cummin over sorted
    positions gives each tie run's first/last index; their midpoint is the
    average rank.
    """
    n = x.shape[dim]
    sorted_x, sort_idx = torch.sort(x, dim=dim)

    shape = [1] * x.dim()
    shape[dim] = n
    pos = torch.arange(n, device=x.device, dtype=x.dtype).view(shape).expand_as(x)

    # First index of each tie run: mark run starts, carry forward via cummax.
    starts_run = torch.ones_like(sorted_x, dtype=torch.bool)
    starts_run.narrow(dim, 1, n - 1).copy_(
        sorted_x.narrow(dim, 1, n - 1) != sorted_x.narrow(dim, 0, n - 1))
    first_src = torch.where(starts_run, pos, torch.full_like(pos, -1.0))
    first = torch.cummax(first_src, dim=dim)[0]

    # Last index: same, scanning backwards.
    ends_run = torch.ones_like(sorted_x, dtype=torch.bool)
    ends_run.narrow(dim, 0, n - 1).copy_(
        sorted_x.narrow(dim, 0, n - 1) != sorted_x.narrow(dim, 1, n - 1))
    last_src = torch.where(ends_run, pos, torch.full_like(pos, float(n + 1)))
    last = torch.flip(torch.cummin(torch.flip(last_src, [dim]), dim=dim)[0], [dim])

    avg = (first + last) / 2.0 + 1.0
    ranks = torch.empty_like(avg)
    ranks.scatter_(dim, sort_idx, avg)
    return ranks


class FastCPMMetrics:
    """Regression metrics: explained variance, Pearson r, MSE, MAE."""

    def __init__(self, device='cpu'):
        self.device = device

    def score(self, y_true, y_pred):
        """Scores every regression model variant at once; y_true [N_samples, N_runs], y_pred [N_samples, *batch, N_runs] -> [N_metrics, *batch, N_runs]."""
        y_true = torch.as_tensor(y_true, device=self.device, dtype=torch.float32)
        y_pred = torch.as_tensor(y_pred, device=self.device, dtype=torch.float32)
        truth = _broadcast_truth(y_true, y_pred)

        mse = torch.mean((truth - y_pred) ** 2, dim=0)
        mae = torch.mean(torch.abs(truth - y_pred), dim=0)

        y_diff = truth - y_pred
        var_true = torch.var(truth, dim=0, unbiased=False)
        var_diff = torch.var(y_diff, dim=0, unbiased=False)
        expl_var = 1 - (var_diff / (var_true + 1e-8))

        pearson = self._pearson_vectorized(truth, y_pred)

        # Order matches constants.Metrics; slots for the other task type stay zero.
        zero = torch.zeros_like(mse)
        metrics_list = [zero] * len(Metrics)
        metrics_list[Metrics.explained_variance_score] = expl_var
        metrics_list[Metrics.pearson_score] = pearson
        metrics_list[Metrics.mean_squared_error] = mse
        metrics_list[Metrics.mean_absolute_error] = mae

        return torch.stack(metrics_list, dim=0)

    def _pearson_vectorized(self, x, y):
        x_c = x - x.mean(dim=0, keepdim=True)
        y_c = y - y.mean(dim=0, keepdim=True)
        cov = torch.sum(x_c * y_c, dim=0)
        std_x = torch.sqrt(torch.sum(x_c ** 2, dim=0))
        std_y = torch.sqrt(torch.sum(y_c ** 2, dim=0))
        return cov / (std_x * std_y + 1e-8)


class FastCPMClassificationMetrics:
    """Binary-classification metrics: accuracy, balanced accuracy, F1, ROC AUC."""

    def __init__(self, device='cpu'):
        self.device = device

    def score(self, y_true, y_pred_proba):
        """Scores every classification model variant at once; y_true [N_samples, N_runs] (0/1), y_pred_proba [N_samples, *batch, N_runs] -> [N_metrics, *batch, N_runs]."""
        y_true = torch.as_tensor(y_true, device=self.device, dtype=torch.float32)
        y_pred_proba = torch.as_tensor(y_pred_proba, device=self.device, dtype=torch.float32)
        truth = _broadcast_truth(y_true, y_pred_proba)

        y_pred_binary = (y_pred_proba > 0.5).float()

        tp = ((truth == 1) & (y_pred_binary == 1)).float().sum(dim=0)
        tn = ((truth == 0) & (y_pred_binary == 0)).float().sum(dim=0)
        fp = ((truth == 0) & (y_pred_binary == 1)).float().sum(dim=0)
        fn = ((truth == 1) & (y_pred_binary == 0)).float().sum(dim=0)

        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        tpr = tp / (tp + fn + 1e-8)          # sensitivity / recall
        tnr = tn / (tn + fp + 1e-8)          # specificity
        balanced_accuracy = (tpr + tnr) / 2
        precision = tp / (tp + fp + 1e-8)
        f1_score = 2 * (precision * tpr) / (precision + tpr + 1e-8)
        roc_auc = self._fast_roc_auc(truth, y_pred_proba)

        # Comparisons against NaN are silently False, so an undefined column
        # would otherwise score as "predicted class 0 for everyone" -- a
        # plausible number instead of a flag. (Regression needs no such guard;
        # means/variances propagate NaN on their own.)
        undefined = torch.isnan(y_pred_proba).any(dim=0)
        nan = torch.full_like(accuracy, float('nan'))
        accuracy = torch.where(undefined, nan, accuracy)
        balanced_accuracy = torch.where(undefined, nan, balanced_accuracy)
        f1_score = torch.where(undefined, nan, f1_score)
        roc_auc = torch.where(undefined, nan, roc_auc)

        zero = torch.zeros_like(accuracy)
        metrics_list = [zero] * len(Metrics)
        metrics_list[Metrics.accuracy] = accuracy
        metrics_list[Metrics.balanced_accuracy] = balanced_accuracy
        metrics_list[Metrics.f1_score] = f1_score
        metrics_list[Metrics.roc_auc] = roc_auc

        return torch.stack(metrics_list, dim=0)

    def _fast_roc_auc(self, truth, y_pred_proba):
        """ROC AUC via the Mann-Whitney U statistic from rank sums: AUC = (sum of ranks of positives - n_pos(n_pos+1)/2) / (n_pos * n_neg).

        Average ranks make ties contribute exactly 0.5 each, matching
        sklearn.metrics.roc_auc_score. Replaces an earlier pairwise [N, N,
        *batch] formulation that measured 9.7 GB for 200 samples x 1000
        permutations and OOMed on large folds; ranking is O(N log N) and
        linear in memory. Returns shape [*batch, N_runs].
        """
        ranks = _average_ranks(y_pred_proba, dim=0)          # [N, *batch, R]

        is_pos = (truth == 1).to(y_pred_proba.dtype)
        is_neg = (truth == 0).to(y_pred_proba.dtype)
        n_pos = is_pos.sum(dim=0)
        n_neg = is_neg.sum(dim=0)

        sum_ranks_pos = (ranks * is_pos).sum(dim=0)
        u_statistic = sum_ranks_pos - n_pos * (n_pos + 1) / 2
        return u_statistic / (n_pos * n_neg + 1e-8)


def score_models(y_true, y_pred, task_type, device='cpu', **kwargs):
    """Scores every CPM model variant at once, dispatching on task type (y_pred must be probabilities for classification)."""
    if task_type == TaskType.regression:
        evaluator = FastCPMMetrics(device=device)
    elif task_type == TaskType.classification:
        evaluator = FastCPMClassificationMetrics(device=device)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    return evaluator.score(y_true, y_pred)
