import os
import numpy as np
import pandas as pd

from sklearn.utils import check_X_y
from sklearn.impute import SimpleImputer

from scipy.stats import ConstantInputWarning, NearConstantInputWarning

import seaborn as sns
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

    if isinstance(covariates, pd.Series):
        covariates = covariates.to_frame().to_numpy()
    if isinstance(covariates, pd.DataFrame):
        # Convert categorical variables to one-hot encoding
        covariates = pd.get_dummies(covariates, drop_first=True).to_numpy()
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
        y = pd.Series(y, name="target")
    elif isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
        y.name = y.name or "target"
    elif isinstance(y, pd.Series):
        y.name = y.name or "target"

    # Convert covariates to DataFrame
    if covariates is not None:
        if isinstance(covariates, np.ndarray):
            covariates = pd.DataFrame(covariates, columns=[f"covariate {i + 1}" for i in range(covariates.shape[1])])
        elif isinstance(covariates, pd.Series):
            covariates = covariates.to_frame()
            if covariates.columns[0] is None:
                covariates.columns = ["covariate 1"]

    # --- Combine all data to check for missing values ---
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
    # Features
    X_names = list(X.columns) if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X.shape[1])]

    # Target
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y_name = y.name if isinstance(y, pd.Series) else y.columns[0]
    else:
        y_name = "target"

    # Covariates
    if isinstance(covariates, (pd.Series, pd.DataFrame)):
        covar_names = [covariates.name] if isinstance(covariates, pd.Series) else list(covariates.columns)
    else:
        covar_names = [f"covariate_{i}" for i in range(covariates.shape[1])]

    return X_names, y_name, covar_names

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype


def pairplot_flexible(df: pd.DataFrame, output_path: str) -> str:
    sns.set_theme(style="white")
    variables = df.columns
    n = len(variables)
    fig, axes = plt.subplots(n, n, figsize=(2.5 * n, 2.5 * n))

    for i, row_var in enumerate(variables):
        for j, col_var in enumerate(variables):
            ax = axes[i, j]
            ax.set_xlabel("")
            ax.set_ylabel("")

            x = df[col_var]
            y = df[row_var]

            is_x_cont = is_numeric_dtype(x)
            is_y_cont = is_numeric_dtype(y)

            if i == j:
                if is_x_cont:
                    sns.histplot(x, bins=20, ax=ax, color="gray", edgecolor="white")
                else:
                    counts = x.value_counts().sort_index()
                    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, palette="pastel")
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                ax.set_title(row_var, fontsize=9)
                sns.despine(ax=ax)
                continue

            if is_x_cont and is_y_cont:
                sns.scatterplot(x=x, y=y, ax=ax, s=15, alpha=0.6, edgecolor="white", linewidth=0.3)
            elif is_x_cont and not is_y_cont:
                sns.histplot(data=df, x=col_var, hue=row_var, ax=ax, element="step", stat="count",
                             common_norm=False, bins=20, palette="Set2")
            elif not is_x_cont and is_y_cont:
                sns.histplot(data=df, x=row_var, hue=col_var, ax=ax, element="step", stat="count",
                             common_norm=False, bins=20, palette="Set2")
            else:
                ctab = pd.crosstab(y, x)
                sns.heatmap(ctab, annot=True, fmt='d', cmap="Blues", cbar=False, ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va='center')

            ax.tick_params(axis='both', labelsize=6)
            sns.despine(ax=ax)

    plt.tight_layout()
    fig.savefig(output_path, dpi=600)
    plt.close(fig)
    return output_path

