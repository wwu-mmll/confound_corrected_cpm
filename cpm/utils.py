from sklearn.metrics import (mean_squared_error, mean_absolute_error, explained_variance_score, accuracy_score,
                             balanced_accuracy_score, roc_auc_score)
from scipy.stats import pearsonr, spearmanr


def score_regression(y_true, y_pred):
    return {'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'exp': explained_variance_score(y_true, y_pred),
            'pearson': pearsonr(y_true, y_pred)[0],
            'spearman': spearmanr(y_true, y_pred)[0]}


def score_classification(y_true, y_pred):
    return {'accuracy': accuracy_score(y_true, y_pred),
            'balanced accuracy': balanced_accuracy_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred)}