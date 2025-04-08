import numpy as np

from sklearn.utils import check_X_y
from sklearn.impute import SimpleImputer

from scipy.stats import ConstantInputWarning, NearConstantInputWarning

import matplotlib.pyplot as plt
import warnings

import logging


logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=ConstantInputWarning)
warnings.filterwarnings("ignore", category=NearConstantInputWarning)

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


def get_colors_from_colormap(n_colors, colormap_name='tab10'):
    """
    Get a set of distinct colors from a specified colormap.

    Parameters:
    n_colors (int): Number of distinct colors needed.
    colormap_name (str): Name of the colormap to use (e.g., 'tab10').

    Returns:
    list: A list of color strings.
    """
    cmap = plt.get_cmap(colormap_name)
    colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]
    return colors


def check_data(X, y, covariates, impute_missings: bool = False):
    logger.info("Checking data...")
    if impute_missings:
        try:
            X, y = check_X_y(X, y, force_all_finite='allow-nan', allow_nd=True, y_numeric=True)
        except ValueError as e:
            logger.info("y contains NaN values. Only missing values in X and covariates can be imputed.")
            raise e
    else:
        try:
            X, y = check_X_y(X, y, force_all_finite=True, allow_nd=True, y_numeric=True)
        except ValueError as e:
            logger.info("Your input contains NaN values. Fix NaNs or use impute_missing_values=True.")
            raise e
    return X, y, covariates

def impute_missing_values(X_train, X_test, cov_train, cov_test):
    # Initialize imputers with chosen strategy (e.g., mean, median, most_frequent)
    x_imputer = SimpleImputer(strategy='mean')
    cov_imputer = SimpleImputer(strategy='mean')

    # Fit on training data and transform both training and test data
    X_train = x_imputer.fit_transform(X_train)
    X_test = x_imputer.transform(X_test)
    cov_train = cov_imputer.fit_transform(cov_train)
    cov_test = cov_imputer.transform(cov_test)
    return X_train, X_test, cov_train, cov_test

def select_stable_edges(stability_edges, stability_threshold):
    return {'positive': np.where(stability_edges['positive'] > stability_threshold)[0],
            'negative': np.where(stability_edges['negative'] > stability_threshold)[0]}
