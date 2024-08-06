import numpy as np

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


def apply_metrics(y_true, y_pred):
    result = {}
    for metric_name, metric_func in regression_metrics_functions.items():
        result[metric_name] = metric_func(y_true, y_pred)
    return result


def score_regression_models(y_true, y_pred):
    scores = {}
    for model in ['full', 'covariates', 'connectome', 'residuals']:
        scores[model] = {}
        for network in ['positive', 'negative', 'both']:
            scores[model][network] = apply_metrics(y_true, y_pred[model][network])
    return scores


def train_test_split(train, test, X, y, covariates):
    return X[train], X[test], y[train], y[test], covariates[train], covariates[test]


def matrix_to_upper_triangular_vector(matrix):
    """
    Convert a 2D square matrix to a vector containing only the elements
    of the strictly upper triangular part (excluding the diagonal).

    Parameters:
    matrix (np.ndarray): Input 2D square matrix of shape (n, n).

    Returns:
    np.ndarray: A vector containing the strictly upper triangular elements.
    """
    if not (matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]):
        raise ValueError("Input must be a 2D square matrix.")

    n = matrix.shape[0]
    # Get the indices of the strictly upper triangular part
    row_indices, col_indices = np.triu_indices(n, k=1)
    # Extract the elements at these indices
    upper_triangular_elements = matrix[row_indices, col_indices]

    return upper_triangular_elements


def vector_to_upper_triangular_matrix(vector):
    """
    Convert a vector containing strictly upper triangular elements back
    to a 2D square matrix.

    Parameters:
    vector (np.ndarray): A vector containing the strictly upper triangular elements.

    Returns:
    np.ndarray: The reconstructed 2D square matrix.
    """
    # Calculate the size of the matrix from the vector length
    size = int((np.sqrt(8 * vector.size + 1) - 1) / 2) + 1
    if size * (size - 1) // 2 != vector.size:
        raise ValueError("Vector size does not match the number of elements for a valid square matrix.")

    matrix = np.zeros((size, size))
    # Get the indices of the strictly upper triangular part
    row_indices, col_indices = np.triu_indices(size, k=1)
    # Place the elements into the matrix
    matrix[row_indices, col_indices] = vector
    matrix[col_indices, row_indices] = vector
    return matrix
