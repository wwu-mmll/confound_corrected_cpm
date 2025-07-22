import torch
from torchmetrics.functional import (
    mean_squared_error,
    mean_absolute_error,
    explained_variance,
    pearson_corrcoef,
    spearman_corrcoef,
)

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
    result = {}
    # Spearman immer drin
    result['spearman_score'] = regression_metrics_functions['spearman_score'](y_pred, y_true).item()

    if not primary_metric_only:
        for name, func in regression_metrics_functions.items():
            if name != 'spearman_score':
                result[name] = func(y_pred, y_true).item()
    return result


# def apply_metrics(y_true, y_pred, primary_metric_only: bool = False):
#     result = {}
#     result['spearman_score'] = regression_metrics_functions['spearman_score'](y_true, y_pred)
#     for metric_name, metric_func in regression_metrics_functions.items():
#         if metric_name == 'spearman_score':
#             pass
#         if not primary_metric_only:
#             result[metric_name] = regression_metrics_functions[metric_name](y_true, y_pred)
#     return result


def score_regression_models(y_true, y_pred_dict, primary_metric_only: bool = False):
    scores = {}
    for model in ['full', 'covariates', 'connectome', 'residuals']:
        scores[model] = {}
        for network in ['positive', 'negative', 'both']:
            preds = y_pred_dict[model][network]
            scores[model][network] = apply_metrics(y_true, preds, primary_metric_only)
    return scores


# def score_regression_models(y_true, y_pred, primary_metric_only: bool = False):
#     scores = {}
#     for model in ['full', 'covariates', 'connectome', 'residuals']:
#         scores[model] = {}
#         for network in ['positive', 'negative', 'both']:
#             scores[model][network] = apply_metrics(y_true, y_pred[model][network], primary_metric_only = primary_metric_only)
#     return scores

