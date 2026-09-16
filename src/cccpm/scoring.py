"""
Vectorised CPM metrics.

One implementation per task type, shape-generic over whatever batch axes sit
between the samples axis and the runs axis. `y_pred` is
``[N_samples, *batch, N_runs]`` -- in practice ``[N, Models, Networks, N_runs]``
or, when an inner CV evaluates several hyperparameter configurations at once,
``[N, Models, Networks, N_params, N_runs]``. `y_true` is always
``[N_samples, N_runs]`` and is broadcast across the batch axes.

There used to be a second, `*_batched` copy of every function here, carrying an
extra folds axis and a per-fold validity mask for zero-padded ragged fold
batches. Fold batching was measured (scripts/benchmark_batching.py) to buy 1.09x
on GPU at 7.2x the VRAM and to be 1.85x *slower* on CPU, so it was removed along
with the padding it required -- and with it the need for two implementations.
"""
import torch

from cccpm.constants import Metrics, TaskType


def _broadcast_truth(y_true, y_pred):
    """Reshape ``y_true`` ``[N, R]`` to broadcast against ``y_pred`` ``[N, *batch, R]``."""
    n_batch_axes = y_pred.dim() - 2
    return y_true.view(y_true.shape[0], *([1] * n_batch_axes), y_true.shape[-1])


def _average_ranks(x, dim=0):
    """
    1-based ranks along ``dim``, with tied values sharing their mean rank
    (scipy.stats.rankdata's 'average' method), computed without materialising
    any pairwise tensor.

    Tie groups are located by comparing each sorted element with its
    neighbours and running a cummax/cummin over the positions, which gives the
    first and last index of the run each element belongs to; their midpoint is
    the average rank.
    """
    n = x.shape[dim]
    sorted_x, sort_idx = torch.sort(x, dim=dim)

    shape = [1] * x.dim()
    shape[dim] = n
    pos = torch.arange(n, device=x.device, dtype=x.dtype).view(shape).expand_as(x)

    # First index of each tie run: mark run starts, then carry forward.
    starts_run = torch.ones_like(sorted_x, dtype=torch.bool)
    starts_run.narrow(dim, 1, n - 1).copy_(
        sorted_x.narrow(dim, 1, n - 1) != sorted_x.narrow(dim, 0, n - 1))
    first_src = torch.where(starts_run, pos, torch.full_like(pos, -1.0))
    first = torch.cummax(first_src, dim=dim)[0]

    # Last index of each tie run: same idea scanning backwards.
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
        """
        Args:
            y_true: [N_samples, N_runs]
            y_pred: [N_samples, *batch, N_runs]

        Returns:
            scores: [N_metrics, *batch, N_runs]
        """
        y_true = torch.as_tensor(y_true, device=self.device, dtype=torch.float32)
        y_pred = torch.as_tensor(y_pred, device=self.device, dtype=torch.float32)
        truth = _broadcast_truth(y_true, y_pred)

        # Each reduction is over the samples axis (dim 0).
        mse = torch.mean((truth - y_pred) ** 2, dim=0)
        mae = torch.mean(torch.abs(truth - y_pred), dim=0)

        y_diff = truth - y_pred
        var_true = torch.var(truth, dim=0, unbiased=False)
        var_diff = torch.var(y_diff, dim=0, unbiased=False)
        expl_var = 1 - (var_diff / (var_true + 1e-8))

        pearson = self._pearson_vectorized(truth, y_pred)

        # Order must match the integer values in constants.Metrics; slots
        # belonging to the other task type stay zero.
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
        """
        Args:
            y_true: [N_samples, N_runs] -- binary labels (0 or 1)
            y_pred_proba: [N_samples, *batch, N_runs] -- predicted probabilities

        Returns:
            scores: [N_metrics, *batch, N_runs]
        """
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

        zero = torch.zeros_like(accuracy)
        metrics_list = [zero] * len(Metrics)
        metrics_list[Metrics.accuracy] = accuracy
        metrics_list[Metrics.balanced_accuracy] = balanced_accuracy
        metrics_list[Metrics.f1_score] = f1_score
        metrics_list[Metrics.roc_auc] = roc_auc

        return torch.stack(metrics_list, dim=0)

    def _fast_roc_auc(self, truth, y_pred_proba):
        """
        ROC AUC via the Mann-Whitney U statistic, computed from rank sums:

            AUC = (sum of ranks of positives - n_pos(n_pos+1)/2) / (n_pos * n_neg)

        Average ranks make ties contribute exactly 0.5 each, matching
        sklearn.metrics.roc_auc_score.

        This replaces an earlier pairwise formulation that built an
        [N, N, *batch] comparison tensor -- quadratic in the test-set size, and
        measured at 9.7 GB for 200 test samples x 1000 permutations, which is
        what made classification runs OOM on large folds. Ranking is O(N log N)
        in time and linear in memory, so scoring now scales like the regression
        path.

        Args:
            truth: [N, *ones, N_runs] -- broadcastable against y_pred_proba.
            y_pred_proba: [N, *batch, N_runs]

        Returns:
            [*batch, N_runs]
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
    """
    Score every CPM model variant at once.

    Args:
        y_true: [N_samples, N_runs]
        y_pred: [N_samples, *batch, N_runs]. For classification these must be
                probabilities, not class labels.
        task_type: TaskType.regression or TaskType.classification
        device: device for the computation

    Returns:
        [N_metrics, *batch, N_runs]
    """
    if task_type == TaskType.regression:
        evaluator = FastCPMMetrics(device=device)
    elif task_type == TaskType.classification:
        evaluator = FastCPMClassificationMetrics(device=device)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    return evaluator.score(y_true, y_pred)
