"""
Diagnostic summary of the input data, written before the analysis runs.

This lives in `reporting/` rather than next to the validation helpers because
it draws figures. Keeping it here is what allows the numeric core to be
imported without matplotlib and seaborn.
"""
import os

import numpy as np
import pandas as pd
import torch

import seaborn as sns
import matplotlib.pyplot as plt

from cccpm.reporting.plots.plots import pairplot_flexible
from cccpm.validation import get_variable_names


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
    parts = [X, y.rename("target")]
    if covariates is not None:
        parts.append(covariates)
    full_data = pd.concat(parts, axis=1)
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
