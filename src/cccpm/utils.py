import os
import math
import numpy as np
import pandas as pd
import torch

from sklearn.utils import check_X_y

from scipy.stats import ConstantInputWarning, NearConstantInputWarning

import seaborn as sns
import matplotlib.pyplot as plt

import warnings

import logging

from cccpm.reporting.plots.plots import pairplot_flexible
from cccpm.constants import TaskType


logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=ConstantInputWarning)
warnings.filterwarnings("ignore", category=NearConstantInputWarning)


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


def torch_train_test_split(train, test, X, y, covariates):
    """Wie utils.train_test_split, aber X/y/covariates sind bereits GPU-Tensoren."""
    train_idx = torch.as_tensor(train, device=X.device)
    test_idx = torch.as_tensor(test, device=X.device)
    return (X[train_idx], X[test_idx], y[train_idx], y[test_idx],
            covariates[train_idx], covariates[test_idx])


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
    row_idx, col_idx = np.triu_indices(n, k=1)
    flat = matrix_3d.reshape(n_samples, n * n)
    upper = flat[:, np.ravel_multi_index((row_idx, col_idx), (n, n))]
    return upper


def vector_to_matrix_tensor_version(tensor, dim):
    """
    Expands a specific dimension containing vectorised upper-triangular edges
    into a symmetric square matrix at that same location.

    Args:
        tensor: Arbitrary shape, e.g. [Networks, Folds, Features, Perms]
        dim: The index of the dimension to expand (e.g., 2 for Features)

    Returns:
        Tensor with 'dim' replaced by two dimensions (Nodes, Nodes).
        Example: [Net, Fold, Feat, Perm] -> [Net, Fold, Nodes, Nodes, Perm]
    """
    # 1. Normalize dim to positive index (handles -1, etc.)
    ndim = tensor.ndim
    dim = dim % ndim

    # 2. Calculate Number of Nodes
    # Formula: F = N(N-1)/2  =>  N = (1 + sqrt(1 + 8F)) / 2
    n_features = tensor.shape[dim]
    n_nodes = int((1 + (1 + 8 * n_features) ** 0.5) / 2)

    # 3. Move the target dimension to the end for easier broadcasting
    # shape: [..., Features]
    temp_tensor = tensor.movedim(dim, -1)

    # 4. Create the Output Placeholder
    # shape: [..., Nodes, Nodes]
    out_shape = temp_tensor.shape[:-1] + (n_nodes, n_nodes)
    out = torch.zeros(out_shape, device=tensor.device, dtype=tensor.dtype)

    # 5. Get Upper Triangle Indices
    rows, cols = torch.triu_indices(n_nodes, n_nodes, offset=1, device=tensor.device)

    # 6. Assign Values (Vectorized)
    # The '...' ellipses handle any number of preceding dimensions automatically
    out[..., rows, cols] = temp_tensor

    # 7. Make Symmetric
    out[..., cols, rows] = temp_tensor

    # 8. Move the Matrix dimensions back to the original location
    # We moved 'dim' to the end. Now we have two dims at the end (-2, -1).
    # We want to put them back at 'dim' and 'dim+1'.
    return out.movedim((-2, -1), (dim, dim + 1))


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


def torch_impute_missing_values(X_train, X_test, cov_train, cov_test):
    def _impute(train, test):
        train = torch.as_tensor(train, dtype=torch.float32)
        test = torch.as_tensor(test, dtype=torch.float32)
        col_mean = torch.nanmean(train, dim=0)
        col_mean = torch.nan_to_num(col_mean, nan=0.0)
        train_filled = torch.where(torch.isnan(train), col_mean.unsqueeze(0), train)
        test_filled = torch.where(torch.isnan(test), col_mean.unsqueeze(0), test)
        return train_filled, test_filled

    X_train, X_test = _impute(X_train, X_test)
    cov_train, cov_test = _impute(cov_train, cov_test)
    return X_train, X_test, cov_train, cov_test


def available_memory_bytes(device) -> int:
    """Free memory on `device`: CUDA via mem_get_info, CPU via psutil."""
    device = torch.device(device)
    if device.type == 'cuda':
        free_bytes, _total = torch.cuda.mem_get_info(device)
        return int(free_bytes)
    import psutil
    return int(psutil.virtual_memory().available)


def plan_permutation_chunk(n_features, n_samples, n_runs, device,
                           safety_factor: float = 0.4) -> int:
    """
    How many permutation columns can be processed at once without running out
    of memory.

    Permutations are always *vectorised* -- y is [N_samples, N_runs] and the
    edge statistic is one matrix product over all columns, which is where the
    GPU speedup comes from (measured 433x versus looping over columns). This
    function does not change that; it only caps how many columns are in flight,
    so a large parcellation crossed with many permutations degrades in speed
    instead of crashing.

    The per-column cost is dominated by the ~10 [N_features, N_runs] float32
    temporaries inside `correlations_and_pvalues` (residualised X and y, cross
    products, r, t, z, p, ...). Measured directly at ~40 bytes per
    (run x feature), stable across n_features from 4,950 to 35,778; the
    n_samples term covers the prediction and rank tensors, which are much
    smaller.

    The estimate is deliberately conservative (`safety_factor` keeps well clear
    of the free-memory figure). The predecessor of this function, the
    `batch_planning.estimate_bytes` cost model, under-counted -- it modelled
    neither the imputation intermediates nor the O(N^2) ROC-AUC term -- and
    planned batches that then OOMed. Under-counting is worse than not
    estimating at all, so this errs low and accepts extra chunks.

    Returns a chunk size in [1, n_runs].
    """
    bytes_per_run = 40 * n_features + 240 * n_samples
    budget = available_memory_bytes(device) * safety_factor
    chunk = int(budget // max(bytes_per_run, 1))
    return max(1, min(chunk, n_runs))


def residualize_train_test(X_train, X_test, confounds_train, confounds_test):
    """
    Regress X ~ intercept + confounds via OLS on the training set only, then
    subtract the fitted values from both train and test (residualizing test
    with the train-fit model, never fitting on test data). Pure torch,
    dtype/device-agnostic (works equally on CPU or GPU tensors) -- same
    closed-form pseudo-inverse approach as edge_selection.get_residuals,
    generalized to the fit-on-train/apply-to-both split CPMAnalysis needs
    for its `calculate_residuals` option.

    Args:
        X_train, X_test: [N_train, F], [N_test, F].
        confounds_train, confounds_test: [N_train, C], [N_test, C].

    Returns:
        X_train_resid, X_test_resid: same shapes as X_train, X_test.
    """
    dtype = X_train.dtype
    device = X_train.device
    confounds_train = torch.as_tensor(confounds_train, dtype=dtype, device=device)
    confounds_test = torch.as_tensor(confounds_test, dtype=dtype, device=device)

    ones_train = torch.ones(confounds_train.shape[0], 1, dtype=dtype, device=device)
    ones_test = torch.ones(confounds_test.shape[0], 1, dtype=dtype, device=device)
    Z_train = torch.cat([ones_train, confounds_train], dim=1)
    Z_test = torch.cat([ones_test, confounds_test], dim=1)

    Z_pinv = torch.linalg.pinv(Z_train)
    beta = torch.matmul(Z_pinv, X_train)

    X_train_resid = X_train - torch.matmul(Z_train, beta)
    X_test_resid = X_test - torch.matmul(Z_test, beta)
    return X_train_resid, X_test_resid


def select_stable_edges(stability_edges, stability_threshold):
    """
    Threshold per-edge selection stability into a boolean edge mask.

    Args:
        stability_edges: Tensor [N_features, 2, N_runs] (fraction of folds each
                          edge was selected in; dim 1 = [Positive, Negative]),
                          as returned by ResultsManager.calculate_edge_stability.
        stability_threshold: float; edges selected in more than this fraction
                              of folds are kept.

    Returns:
        Boolean tensor [N_features, 2, N_runs], same convention as every other
        edge mask in the pipeline (e.g. PThreshold.select's return value).
    """
    return stability_edges > stability_threshold


def generate_data_insights(X, y, covariates, results_directory):
    """
    Generate summary statistics and diagnostic plots about the input data.
    Saves outputs to a subfolder in results_directory.

    Handles both pandas DataFrames and NumPy arrays.
    """
    # Create output folder
    output_dir = os.path.join(results_directory, "data_insights")
    os.makedirs(output_dir, exist_ok=True)

    X_names, y_name, covariates_names = get_variable_names(X, y, covariates)
    pd.Series(X_names).to_csv(os.path.join(output_dir, "X_names.csv"), index=False, header=False)
    pd.Series([y_name]).to_csv(os.path.join(output_dir, "y_name.csv"), index=False, header=False)
    pd.Series(covariates_names).to_csv(os.path.join(output_dir, "covariate_names.csv"), index=False, header=False)

    # Convert X to DataFrame if needed
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"feature {i + 1}" for i in range(X.shape[1])])

    # Convert y to Series
    if isinstance(y, np.ndarray):
        y = pd.Series(np.squeeze(y), name="target")
    elif isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
        y.name = y.name or "target"
    elif isinstance(y, pd.Series):
        y.name = y.name or "target"

    # Convert covariates to DataFrame
    if covariates is not None:
        if isinstance(covariates, np.ndarray):
            if len(covariates.shape) == 1:
                covariates = pd.DataFrame(covariates, columns=["covariate 1"])
            else:
                covariates = pd.DataFrame(covariates, columns=[f"covariate {i + 1}" for i in range(covariates.shape[1])])
        elif isinstance(covariates, pd.Series):
            covariates = covariates.to_frame()
            if covariates.columns[0] is None:
                covariates.columns = ["covariate 1"]

    # --- Combine all data to check for missing values ---
    if isinstance(X, torch.Tensor):
        X = pd.DataFrame(X.detach().cpu().numpy())
    if isinstance(y, torch.Tensor):
        y = pd.Series(y.detach().cpu().numpy(), name="target")
    if isinstance(covariates, torch.Tensor):
        covariates = pd.DataFrame(covariates.detach().cpu().numpy())
    full_data = pd.concat([X, y.rename("target"), covariates], axis=1)
    missing_total = full_data.isnull().sum().sum()

    # --- Summary ---
    summary = {
        "Number of samples": len(X),
        "Number of features (connectivity values)": X.shape[1],
        "Number of covariates": covariates.shape[1] if covariates is not None else 0,
        "Total missing values": missing_total
    }
    summary_df = pd.DataFrame.from_dict(summary, orient="index", columns=["Value"])
    summary_df.to_csv(os.path.join(output_dir, "summary.csv"))

    # --- Target Histogram ---
    plt.figure(figsize=(4, 3))
    sns.histplot(y, bins=30, color="gray", edgecolor="white")
    plt.title("Distribution of Target Variable")
    plt.xlabel(y.name)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "target_distribution.png"), dpi=300)
    plt.close()

    # --- Scatter Matrix: Covariates and Target ---
    if covariates is not None and not covariates.empty:
        cov_y = pd.concat([covariates, y.rename("target")], axis=1)
        pairplot_flexible(cov_y, os.path.join(output_dir, "scatter_matrix.png"))

    # --- Optional: Missing Values Heatmap ---
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
        if len(covariates.shape) == 1:
            covar_names = ["covariate_1"]
        else:
            covar_names = [
                f"covariate_{i}" for i in range(covariates.shape[1])
            ]

    return X_names, y_name, covar_names
