from sklearn.metrics import (mean_squared_error, mean_absolute_error, explained_variance_score, accuracy_score,
                             balanced_accuracy_score, roc_auc_score)
from scipy.stats import pearsonr, spearmanr


def score_regression(y_true, y_pred):
    return {'mean_squared_error': mean_squared_error(y_true, y_pred),
            'mean_absolute_error': mean_absolute_error(y_true, y_pred),
            'explained_variance_score': explained_variance_score(y_true, y_pred),
            'pearson_score': pearsonr(y_true, y_pred)[0],
            'spearman_score': spearmanr(y_true, y_pred)[0]}


def score_classification(y_true, y_pred):
    return {'accuracy_score': accuracy_score(y_true, y_pred),
            'balanced accuracy_score': balanced_accuracy_score(y_true, y_pred),
            'roc_auc_score': roc_auc_score(y_true, y_pred)}