import os
import numpy as np
import pandas as pd

import torch
from scipy.stats import ConstantInputWarning, NearConstantInputWarning

import seaborn as sns
import matplotlib.pyplot as plt

import warnings

import logging

from cccpm.reporting.plots.plots import pairplot_flexible

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=ConstantInputWarning)
warnings.filterwarnings("ignore", category=NearConstantInputWarning)


def train_test_split(train_idx, test_idx, X, y, covariates):
    return (
        X[train_idx], X[test_idx],
        y[train_idx], y[test_idx],
        covariates[train_idx], covariates[test_idx]
    )


def matrix_to_upper_triangular_vector(matrix):
    # matrix: (n, n)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be a 2D square matrix.")
    n = matrix.shape[0]
    row_idx, col_idx = torch.triu_indices(n, n, offset=1, device=matrix.device)
    return matrix[row_idx, col_idx]


def vector_to_upper_triangular_matrix(vector, include_diagonal=True):
    import numpy as np
    import torch

    if isinstance(vector, torch.Tensor):
        num_elements = vector.numel()
        device = vector.device
    elif isinstance(vector, np.ndarray):
        num_elements = vector.size
        device = torch.device("cpu")
        vector = torch.from_numpy(vector).float()
    else:
        raise TypeError("Unsupported type")

    # Calculate the size of the matrix from the vector length
    size = int((np.sqrt(8 * vector.size + 1) - 1) / 2) + 1
    if size * (size - 1) // 2 != vector.size:
        raise ValueError("Vector size does not match the number of elements for a valid square matrix.")

    matrix = torch.zeros((size, size), device=device)
    offset = 0 if include_diagonal else 1
    # Get the indices of the strictly upper triangular part
    row_idx, col_idx = torch.triu_indices(size, size, offset=offset, device=device)
    matrix[row_idx, col_idx] = vector.flatten() if vector.ndim > 1 else vector
    matrix[col_idx, row_idx] = matrix[row_idx, col_idx]
    return matrix


def matrix_to_vector_3d(matrix_3d):
    """
    Convert a 3D connectivity matrix to a 2D array of upper-triangular vectors.

    Parameters
    ----------
    matrix_3d: np.ndarray
        Input 3D array of shape (n_samples, n, n), where each 2D matrix is square.

    Returns
    -------
    upper: np.ndarray
        2D array of shape (n_samples, n*(n - 1)/2) containing strictly upper-triangular elements of each matrix.
    """
    n_samples, n, _ = matrix_3d.shape
    row_idx, col_idx = torch.triu_indices(n, n, offset=1, device=matrix_3d.device)
    return matrix_3d[:, row_idx, col_idx]  # shape: (n_samples, upper_tri_size)


def vector_to_matrix_3d(vector_2d, shape):
    """
    Convert a vector containing strictly upper triangular parts back to a 3D matrix.

    Parameters:
    vector_2d (np.ndarray): A 2D array where each row is a vector of the strictly upper triangular part of a 2D matrix.
    shape (tuple): The shape of the original 3D matrix, (n_samples, n, n).

    Returns:
    np.ndarray: The reconstructed 3D matrix of shape (n_samples, n, n).
    """
    n_samples, n, _ = shape
    matrix_3d = torch.zeros((n_samples, n, n), device=vector_2d.device)
    row_idx, col_idx = torch.triu_indices(n, n, offset=1, device=vector_2d.device)
    matrix_3d[:, row_idx, col_idx] = vector_2d
    matrix_3d[:, col_idx, row_idx] = vector_2d  # symmetric
    return matrix_3d


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


def check_data(X, y, covariates, impute_missings: bool = False, device='cuda'):
    """
    Validate and format input tensors for modeling.

    Parameters
    ----------
    X: torch.Tensor or array-like
        Feature data of shape (n_samples, n_features) or
        connectivity matrices of shape (n_samples, n, n). 3D matrices are vectorized.
    y: torch.Tensor or array-like
        Target values; 1D tensor or 2D with shape (n_samples, 1)
    covariates: torch.Tensor, array-like, pd.Series or pd.DataFrame
        Covariates; Series are converted to 2D; DataFrames are one-hot encoded.
    impute_missings: bool, default=False
        If True, allows NaNs in X for imputation. NaNs in y raise an error.
    device: str
        Device to move tensors to (e.g. "cuda", "cpu")

    Returns
    -------
    X_checked, y_checked, covariates_checked : torch.Tensor
        Validated and reshaped tensors on the specified device.
    """
    X = torch.as_tensor(X)
    if X.ndim == 3:
        X = matrix_to_vector_3d(X)
    elif X.ndim != 2:
        raise ValueError(f"X must be 2D or 3D, got shape {X.shape}")

    y = torch.as_tensor(y)
    if y.ndim == 2 and 1 in y.shape:
        y = y.flatten()
    elif y.ndim != 1:
        raise ValueError(f"y must be 1D or shape (n, 1), got shape {y.shape}")

    if torch.isnan(y).any():
        raise ValueError("y contains NaNs, which are not allowed.")
    if not impute_missings and torch.isnan(X).any():
        raise ValueError("X contains NaNs, and impute_missings=False.")

    if isinstance(covariates, pd.Series):
        cov_df = covariates.to_frame()
    elif isinstance(covariates, pd.DataFrame):
        cov_df = pd.get_dummies(covariates, drop_first=True)
    else:
        cov_df = covariates

    if isinstance(cov_df, (pd.Series, pd.DataFrame)):
        cov_arr = torch.as_tensor(cov_df.to_numpy(), dtype=torch.float32)
    else:
        cov_arr = torch.as_tensor(cov_df, dtype=torch.float32)

    if cov_arr.ndim == 1:
        cov_arr = cov_arr.reshape(-1, 1)
    elif cov_arr.ndim != 2:
        raise ValueError(f"covariates must be 1D or 2D, got shape {cov_arr.shape}")

    return X.to(device), y.to(device), cov_arr.to(device)


def impute_missing_values(X_train, X_test, cov_train, cov_test):
    def impute(train, test):
        means = torch.nanmean(train, dim=0)
        train_filled = torch.where(torch.isnan(train), means, train)
        test_filled = torch.where(torch.isnan(test), means, test)
        return train_filled, test_filled

    X_train_filled, X_test_filled = impute(X_train, X_test)
    cov_train_filled, cov_test_filled = impute(cov_train, cov_test)

    return X_train_filled, X_test_filled, cov_train_filled, cov_test_filled


def select_stable_edges(stability_edges: dict, stability_threshold: float):
    return {
        'positive': torch.nonzero(stability_edges['positive'] > stability_threshold, as_tuple=True)[0],
        'negative': torch.nonzero(stability_edges['negative'] > stability_threshold, as_tuple=True)[0],
    }


def generate_data_insights(X, y, covariates, results_directory):
    """
    Generate summary statistics and diagnostic plots about the input data.
    Saves outputs to a subfolder in results_directory.

    Handles both pandas DataFrames and NumPy arrays.
    """
    output_dir = os.path.join(results_directory, "data_insights")
    os.makedirs(output_dir, exist_ok=True)

    X_names, y_name, covariates_names = get_variable_names(X, y, covariates)
    pd.Series(X_names).to_csv(os.path.join(output_dir, "X_names.csv"), index=False, header=False)
    pd.Series([y_name]).to_csv(os.path.join(output_dir, "y_name.csv"), index=False, header=False)
    pd.Series(covariates_names).to_csv(os.path.join(output_dir, "covariate_names.csv"), index=False, header=False)

    # Convert X to DataFrame if needed
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"feature {i + 1}" for i in range(X.shape[1])])

    if isinstance(X, torch.Tensor):
        X = pd.DataFrame(X.numpy(), columns=[f"feature {i + 1}" for i in range(X.numpy().shape[1])])

    # Convert y to Series
    if isinstance(y, np.ndarray):
        y = pd.Series(np.squeeze(y), name="target")
    elif isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
        y.name = y.name or "target"
    elif isinstance(y, pd.Series):
        y.name = y.name or "target"
    elif isinstance(y, torch.Tensor):
        y = pd.Series(torch.squeeze(y).numpy(), name="target")

    # Convert covariates to DataFrame
    if covariates is not None:
        if isinstance(covariates, np.ndarray):
            covariates = pd.DataFrame(covariates, columns=[f"covariate {i + 1}" for i in range(covariates.shape[1])])
        elif isinstance(covariates, torch.Tensor):
            covariates = pd.DataFrame(covariates.numpy(), columns=[f"covariate {i + 1}" for i in range(covariates.numpy().shape[1])])
        elif isinstance(covariates, pd.Series):
            covariates = covariates.to_frame()
            if covariates.columns[0] is None:
                covariates.columns = ["covariate 1"]

    full_data = pd.concat([X, y.rename("target"), covariates], axis=1)
    missing_total = full_data.isnull().sum().sum()

    summary = {
        "Number of samples": len(X),
        "Number of features (connectivity values)": X.shape[1],
        "Number of covariates": covariates.shape[1] if covariates is not None else 0,
        "Total missing values": missing_total
    }
    summary_df = pd.DataFrame.from_dict(summary, orient="index", columns=["Value"])
    summary_df.to_csv(os.path.join(output_dir, "summary.csv"))

    plt.figure(figsize=(4, 3))
    sns.histplot(y, bins=30, color="gray", edgecolor="white")
    plt.title("Distribution of Target Variable")
    plt.xlabel(y.name)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "target_distribution.png"), dpi=300)
    plt.close()

    if covariates is not None and not covariates.empty:
        cov_y = pd.concat([covariates, y.rename("target")], axis=1)
        pairplot_flexible(cov_y, os.path.join(output_dir, "scatter_matrix.png"))

    if missing_total > 0:
        plt.figure(figsize=(10, 6))
        sns.heatmap(full_data.isnull(), cbar=False, yticklabels=False)
        plt.title("Missing Values Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "missing_values_heatmap.png"), dpi=300)
        plt.close()
    return


def get_variable_names(X, y, covariates):
    """
    Generate names for features, target, and covariates based on input types.

    Parameters
    ----------
    X : array-like or pandas.DataFrame
        Feature data. If DataFrame, column names are returned; otherwise,
        generic names "feature_{i}" are generated for each feature (i from
        0 to n_features - 1).
    y : array-like, pandas.Series, or pandas.DataFrame
        Target vector. If Series, its name is used; if DataFrame, the first
        column name is used; otherwise, the default name "target" is returned.
    covariates : array-like, pandas.Series, or pandas.DataFrame
        Covariate data. If Series, its name is returned as a single-element
        list; if DataFrame, its column names are returned; otherwise, generic
        names "covariate_{i}" are generated for each covariate column.

    Returns
    -------
    X_names : list of str
        Names for each feature column.
    y_name : str
        Name for the target variable.
    covar_names : list of str
        Names for each covariate column.
    """
    # Features
    X_names = list(X.columns) if isinstance(X, pd.DataFrame) else [
        f"feature_{i}" for i in range(X.shape[1])
    ]

    # Target
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y_name = y.name if isinstance(y, pd.Series) else y.columns[0]
    else:
        y_name = "target"

    # Covariates
    if isinstance(covariates, (pd.Series, pd.DataFrame)):
        covar_names = (
            [covariates.name]
            if isinstance(covariates, pd.Series)
            else list(covariates.columns)
        )
    else:
        covar_names = [
            f"covariate_{i}" for i in range(covariates.shape[1])
        ]

    return X_names, y_name, covar_names
