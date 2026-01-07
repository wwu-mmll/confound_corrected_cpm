import torch
import numpy as np
from torchmetrics.functional import (
    mean_squared_error,
    mean_absolute_error,
    explained_variance,
    pearson_corrcoef,
    spearman_corrcoef,
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

regression_metrics_functions = {
    'mean_squared_error': mean_squared_error,
    'mean_absolute_error': mean_absolute_error,
    'explained_variance_score': explained_variance,
    'pearson_score': pearson_corrcoef,
    'spearman_score': spearman_corrcoef,
}

regression_metrics = list(regression_metrics_functions.keys())


def score_regression(y_true: torch.Tensor, y_pred: torch.Tensor):
    scores = {}
    for name, metric_fn in regression_metrics_functions.items():
        scores[name] = metric_fn(y_pred, y_true).item()
    return scores


def apply_metrics(y_true, y_pred, primary_metric_only: bool = False):
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true).float()
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred).float()

    y_true = y_true.to(device)
    y_pred = y_pred.to(device)

    result = {}
    result['spearman_score'] = regression_metrics_functions['spearman_score'](y_pred, y_true).item()

    if not primary_metric_only:
        for name, func in regression_metrics_functions.items():
            if name != 'spearman_score':
                result[name] = func(y_pred, y_true).item()
    return result


def score_regression_models(y_true, y_pred_dict, primary_metric_only: bool = False):
    scores = {}
    for model in ['full', 'covariates', 'connectome', 'residuals']:
        scores[model] = {}
        for network in ['positive', 'negative', 'both']:
            preds = y_pred_dict[model][network]
            scores[model][network] = apply_metrics(y_true, preds, primary_metric_only)
    return scores
