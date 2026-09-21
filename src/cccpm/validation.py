"""
Input validation and naming.

Everything that inspects user-supplied ``X`` / ``y`` / ``covariates`` before
the pipeline touches them: task-type detection, the connectome-shape contract,
and the variable names carried through to the report.
"""
import math
import logging

import numpy as np
import pandas as pd
import torch

from sklearn.utils import check_X_y

from cccpm.connectome import matrix_to_vector_3d
from cccpm.constants import TaskType


logger = logging.getLogger(__name__)


def detect_task_type(y):
    """
    Automatically detect whether the task is regression or classification.

    Args:
        y: Target variable (array-like)

    Returns:
        TaskType.regression or TaskType.classification

    Raises:
        ValueError: If y is not suitable for either regression or binary classification
    """
    y_arr = np.asarray(y).ravel()

    # Get unique values
    unique_vals = np.unique(y_arr[~np.isnan(y_arr)])  # Exclude NaNs

    # Check if binary (exactly 2 unique values)
    if len(unique_vals) == 2:
        # Check if values are 0/1 or -1/1
        if set(unique_vals) == {0, 1} or set(unique_vals) == {-1, 1}:
            logger.info(f"Detected binary classification task (unique values: {unique_vals})")
            return TaskType.classification
        else:
            # Two unique values but not standard binary encoding
            logger.warning(
                f"Target has only 2 unique values {unique_vals} but not in {0,1} or {-1,1} format. "
                f"Treating as regression. For classification, please encode as 0/1."
            )
            return TaskType.regression

    # More than 2 unique values -> regression
    elif len(unique_vals) > 2:
        logger.info(f"Detected regression task ({len(unique_vals)} unique values)")
        return TaskType.regression

    # Less than 2 unique values (constant target)
    else:
        raise ValueError(
            f"Target variable has only {len(unique_vals)} unique value(s): {unique_vals}. "
            f"Cannot perform prediction with constant target."
        )


def validate_task_type(y, task_type):
    """
    Validate that the specified task type matches the target variable.

    Args:
        y: Target variable (array-like)
        task_type: Specified TaskType

    Raises:
        ValueError: If task_type doesn't match the data
    """
    detected_task = detect_task_type(y)

    if task_type != detected_task:
        y_arr = np.asarray(y).ravel()
        unique_vals = np.unique(y_arr[~np.isnan(y_arr)])
        raise ValueError(
            f"Specified task_type='{task_type}' but detected '{detected_task}' "
            f"from target variable (unique values: {unique_vals}). "
            f"Please check your data or task_type specification."
        )


def infer_n_nodes(n_features: int):
    """
    Return the number of nodes ``n`` for which ``n * (n - 1) / 2 == n_features``,
    i.e. the connectome size whose upper triangle has exactly ``n_features`` edges.

    Returns ``None`` if ``n_features`` is not a valid upper-triangular edge count.
    """
    if n_features is None or n_features < 1:
        return None
    discriminant = 1 + 8 * n_features
    root = math.isqrt(discriminant)
    if root * root != discriminant or (1 + root) % 2 != 0:
        return None
    return (1 + root) // 2


def check_data(X, y, covariates, impute_missings: bool = False):
    """
    Validate and format input data for modeling.

    Parameters
    ----------
    X: array-like
        Feature data of shape (n_samples, n_features) or
        connectivity matrices of shape (n_samples, n, n). 3D matrices are vectorized.
    y: array-like
        Target values; 1D array of shape (n_samples,) or
        2D array of shape (n_samples, 1) to be squeezed.
    covariates: array-like or pandas.Series or pandas.DataFrame
        Covariate data. Series are converted to 2D; DataFrames are one-hot encoded.
    impute_missings: bool, default=False
        If True, allow NaNs in X for imputation; NaNs in y always raise an error.

    Returns
    -------
    X_checked: np.ndarray
        2D array of validated (and vectorized) feature data.
    y_checked: np.ndarray
        1D array of target values.
    cov_arr: np.ndarray
        2D array of covariates.
    """
    # Convert to numpy for dimension checks
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    if isinstance(covariates, torch.Tensor):
        covariates = covariates.detach().cpu().numpy()
    X_arr = np.asarray(X)
    # Handle 3D connectivity matrices
    if X_arr.ndim == 3:
        X_arr = matrix_to_vector_3d(X_arr)
    elif X_arr.ndim != 2:
        raise ValueError(f"X must be 2D or 3D, got shape {X_arr.shape}")

    # Connectome features must be the upper triangle of a symmetric node-by-node
    # matrix, i.e. n_features == n_nodes * (n_nodes - 1) / 2. Otherwise edge
    # selection/stability cannot map edges back to a connectome and the run would
    # later fail with a cryptic shape-mismatch error. Fail fast with a clear message.
    n_features = X_arr.shape[1]
    if infer_n_nodes(n_features) is None:
        n_lower = int((1 + (1 + 8 * n_features) ** 0.5) / 2)
        lower = n_lower * (n_lower - 1) // 2
        upper = (n_lower + 1) * n_lower // 2
        raise ValueError(
            f"X has {n_features} features, which is not a valid connectome size. "
            f"CCCPM expects the upper-triangular edges of a symmetric node-by-node "
            f"connectome, i.e. n_features = n_nodes * (n_nodes - 1) / 2. "
            f"The nearest valid sizes are {lower} ({n_lower} nodes) and "
            f"{upper} ({n_lower + 1} nodes). Alternatively, pass connectivity "
            f"matrices of shape (n_samples, n_nodes, n_nodes) and CCCPM will "
            f"vectorize them for you."
        )

    # Ensure y is 1D vector
    y_arr = np.asarray(y)
    if y_arr.ndim == 2:
        if 1 in y_arr.shape:
            y_arr = y_arr.ravel()
        else:
            raise ValueError(f"y must be a vector, got shape {y_arr.shape}")
    elif y_arr.ndim != 1:
        raise ValueError(f"y must be 1D array, got shape {y_arr.shape}")

    # Validate X and y with sklearn
    if impute_missings:
        try:
            X_checked, y_checked = check_X_y(
                X_arr, y_arr,
                ensure_all_finite='allow-nan',
                allow_nd=True,
                y_numeric=True
            )
        except ValueError:
            logger.info(
                "y contains NaN values. Only missing values in X and covariates can be imputed."
            )
            raise
    else:
        try:
            X_checked, y_checked = check_X_y(
                X_arr, y_arr,
                ensure_all_finite=True,
                allow_nd=True,
                y_numeric=True
            )
        except ValueError:
            logger.info(
                "Your input contains NaN values. Fix NaNs or use impute_missing_values=True."
            )
            raise

    # Process covariates
    if isinstance(covariates, pd.Series):
        cov_df = covariates.to_frame()
    elif isinstance(covariates, pd.DataFrame):
        cov_df = pd.get_dummies(covariates, drop_first=True)
    else:
        cov_df = covariates

    if isinstance(cov_df, (pd.Series, pd.DataFrame)):
        cov_arr = cov_df.to_numpy()
    else:
        cov_arr = np.asarray(cov_df)

    # Ensure covariates are 2D
    if cov_arr.ndim == 1:
        cov_arr = cov_arr.reshape(-1, 1)
    elif cov_arr.ndim != 2:
        raise ValueError(f"covariates must be 1D or 2D, got shape {cov_arr.shape}")

    return X_checked, y_checked, cov_arr


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
        if len(covariates.shape) == 1:
            covar_names = ["covariate_1"]
        else:
            covar_names = [
                f"covariate_{i}" for i in range(covariates.shape[1])
            ]

    return X_names, y_name, covar_names
