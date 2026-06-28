from sklearn.metrics import (mean_squared_error, mean_absolute_error, explained_variance_score)
from scipy.stats import pearsonr, spearmanr


regression_metrics_functions = {
    'mean_squared_error': mean_squared_error,
    'mean_absolute_error': mean_absolute_error,
    'explained_variance_score': explained_variance_score,
    'pearson_score': lambda y_true, y_pred: pearsonr(y_true, y_pred)[0],
    'spearman_score': lambda y_true, y_pred: spearmanr(y_true, y_pred)[0]}

regression_metrics = list(regression_metrics_functions.keys())


def score_regression(y_true, y_pred):
    scores = {}
    for metric_name, metric_func in regression_metrics_functions.items():
        scores[metric_name] = metric_func(y_true, y_pred)
    return scores


def apply_metrics(y_true, y_pred, primary_metric_only: bool = False):
    result = {}
    result['spearman_score'] = regression_metrics_functions['spearman_score'](y_true, y_pred)
    for metric_name, metric_func in regression_metrics_functions.items():
        if metric_name == 'spearman_score':
            pass
        if not primary_metric_only:
            result[metric_name] = regression_metrics_functions[metric_name](y_true, y_pred)
    return result


def score_regression_models(y_true, y_pred, primary_metric_only: bool = False):
    scores = {}
    for model in ['full', 'covariates', 'connectome', 'residuals']:
        scores[model] = {}
        for network in ['positive', 'negative', 'both']:
            scores[model][network] = apply_metrics(y_true, y_pred[model][network], primary_metric_only = primary_metric_only)
    return scores

