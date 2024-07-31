import numpy as np

from sklearn.metrics import (mean_squared_error, mean_absolute_error, explained_variance_score, accuracy_score,
                             balanced_accuracy_score, roc_auc_score)
from scipy.stats import pearsonr, spearmanr


regression_metrics = ['mean_squared_error', 'mean_absolute_error', 'explained_variance_score', 'pearson_score',
                      'spearman_score']


def score_regression(y_true, y_pred):
    return {'mean_squared_error': mean_squared_error(y_true, y_pred),
            'mean_absolute_error': mean_absolute_error(y_true, y_pred),
            'explained_variance_score': explained_variance_score(y_true, y_pred),
            'pearson_score': pearsonr(y_true, y_pred)[0],
            'spearman_score': spearmanr(y_true, y_pred)[0]}


def score_regression_models(y_true, y_pred):
    scores = {}
    for model in ['full', 'covariates', 'connectome']:
        scores[model] = {}
        for network in ['positive', 'negative', 'both']:
            scores[model][network] = {'mean_squared_error': mean_squared_error(y_true, y_pred[model][network]),
            'mean_absolute_error': mean_absolute_error(y_true, y_pred[model][network]),
            'explained_variance_score': explained_variance_score(y_true, y_pred[model][network]),
            'pearson_score': pearsonr(y_true, y_pred[model][network])[0],
            'spearman_score': spearmanr(y_true, y_pred[model][network])[0]}
    return scores


def score_classification(y_true, y_pred):
    return {'accuracy_score': accuracy_score(y_true, y_pred),
            'balanced accuracy_score': balanced_accuracy_score(y_true, y_pred),
            'roc_auc_score': roc_auc_score(y_true, y_pred)}


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

    return matrix
