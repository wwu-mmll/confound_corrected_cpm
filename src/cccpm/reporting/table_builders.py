"""
Table building utilities for HTML report generation.

This module contains functions for creating and formatting various
tables used in the CPM HTML report.
"""

import pandas as pd
import numpy as np
from typing import Dict


def create_edge_stability_table(
    edge_matrices: Dict[str, np.ndarray],
    atlas_labels: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Create a table from edge stability matrices.

    Args:
        edge_matrices: Dict with 'stability' and 'stability_significance' arrays
        atlas_labels: DataFrame with region labels (optional)

    Returns:
        DataFrame with edge stability information sorted by significance
    """
    stability_matrix = edge_matrices['stability']
    significance_matrix = edge_matrices['stability_significance']

    n = stability_matrix.shape[0]
    stability_values = []
    significance_values = []
    region_a_names = []
    region_b_names = []

    # Extract upper triangle values (excluding diagonal)
    for i in range(1, n):
        for j in range(i):
            # Skip edges with zero stability
            if stability_matrix[i, j] == 0:
                continue

            # Get region names from atlas or use default naming
            if atlas_labels is not None:
                region_a_names.append(atlas_labels['region'][i])
                region_b_names.append(atlas_labels['region'][j])
            else:
                region_a_names.append(f"Region {i}")
                region_b_names.append(f"Region {j}")

            stability_values.append(stability_matrix[i, j])
            significance_values.append(significance_matrix[i, j])

    # Create DataFrame
    df = pd.DataFrame({
        'Region A': region_a_names,
        'Region B': region_b_names,
        'Stability': stability_values,
        'Stability Significance': significance_values
    })

    # Round numeric columns
    df[['Stability', 'Stability Significance']] = \
        df[['Stability', 'Stability Significance']].round(5)

    # Sort by significance (ascending) and stability (descending)
    df.sort_values(
        by=['Stability Significance', 'Stability'],
        inplace=True,
        ascending=[True, False]
    )

    # Set multi-index
    df.set_index(['Region A', 'Region B'], inplace=True)

    # Handle empty dataframe
    if df.empty:
        first_col = df.columns[0]
        row = {
            col: ("No significantly stable edges." if col == first_col else np.nan)
            for col in df.columns
        }
        df = pd.DataFrame([row])

    return df


def create_hyperparameter_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a formatted hyperparameter table from CV results.

    Args:
        df: DataFrame containing fold-wise hyperparameters

    Returns:
        Formatted DataFrame with hyperparameters by fold
    """
    if 'params' not in df.columns:
        return pd.DataFrame()

    hyper_df = df[['fold', 'model', 'params']].copy()

    # Keep only first row per fold (hyperparameters are the same across models)
    hyper_df = (
        hyper_df
        .drop(columns='model')
        .drop_duplicates(subset=['fold'])
        .reset_index(drop=True)
    )

    # Convert dict params to formatted string
    if isinstance(hyper_df['params'].iloc[0], dict):
        hyper_df['params'] = hyper_df['params'].apply(
            lambda x: "\n".join(f"{k}: {v}" for k, v in x.items())
        )

    return hyper_df


def create_hyperparameter_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary statistics for hyperparameters across folds.

    Args:
        df: DataFrame containing hyperparameters

    Returns:
        DataFrame with hyperparameter summary statistics
    """
    if 'params' not in df.columns:
        return pd.DataFrame()

    if not isinstance(df['params'].iloc[0], dict):
        return pd.DataFrame()

    # Normalize nested dict structure
    all_params = pd.json_normalize(df['params'])

    # Calculate summary statistics
    param_summary = all_params.describe().T

    return param_summary


def combine_results_with_pvalues(
    df_mean: pd.DataFrame,
    df_p_values: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine mean results with p-values into a single formatted table.

    Args:
        df_mean: DataFrame with mean and std of metrics
        df_p_values: DataFrame with p-values

    Returns:
        Combined DataFrame with interleaved mean, std, and p columns
    """
    # Prepare p-values DataFrame
    df_p_values.set_index(['model', 'network'], inplace=True)
    df_p_values.columns = pd.MultiIndex.from_tuples(
        [(col, 'p') for col in df_p_values.columns]
    )

    # Concatenate along columns
    df_combined = pd.concat([df_mean, df_p_values], axis=1)

    # Sort columns: first by metric name, then by statistic type
    df_combined = df_combined.sort_index(axis=1, level=0)

    # Define desired order of statistics
    desired_order = ["mean", "std", "p"]

    # Sort columns with custom order
    df_combined = df_combined.loc[
        :,
        sorted(
            df_combined.columns,
            key=lambda x: (x[0], desired_order.index(x[1]))
        )
    ]

    return df_combined
