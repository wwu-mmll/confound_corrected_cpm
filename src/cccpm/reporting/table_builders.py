"""Table building utilities for HTML report generation."""

import pandas as pd
import numpy as np
from typing import Dict


def create_edge_stability_table(
    edge_matrices: Dict[str, np.ndarray],
    atlas_labels: pd.DataFrame = None
) -> pd.DataFrame:
    """Build an edge-stability table from 'stability'/'stability_significance' matrices, sorted by significance."""
    stability_matrix = edge_matrices['stability']
    significance_matrix = edge_matrices['stability_significance']

    n = stability_matrix.shape[0]
    stability_values = []
    significance_values = []
    region_a_names = []
    region_b_names = []

    # Upper triangle, excluding the diagonal.
    for i in range(1, n):
        for j in range(i):
            if stability_matrix[i, j] == 0:
                continue

            if atlas_labels is not None:
                region_a_names.append(atlas_labels['region'][i])
                region_b_names.append(atlas_labels['region'][j])
            else:
                region_a_names.append(f"Region {i}")
                region_b_names.append(f"Region {j}")

            stability_values.append(stability_matrix[i, j])
            significance_values.append(significance_matrix[i, j])

    df = pd.DataFrame({
        'Region A': region_a_names,
        'Region B': region_b_names,
        'Stability': stability_values,
        'Stability Significance': significance_values
    })

    df[['Stability', 'Stability Significance']] = \
        df[['Stability', 'Stability Significance']].round(5)

    # Most significant and most stable first.
    df.sort_values(
        by=['Stability Significance', 'Stability'],
        inplace=True,
        ascending=[True, False]
    )

    df.set_index(['Region A', 'Region B'], inplace=True)

    if df.empty:
        first_col = df.columns[0]
        row = {
            col: ("No significantly stable edges." if col == first_col else np.nan)
            for col in df.columns
        }
        df = pd.DataFrame([row])

    return df


def create_hyperparameter_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a formatted hyperparameter table from CV results, one row per fold."""
    if 'params' not in df.columns:
        return pd.DataFrame()

    hyper_df = df[['fold', 'model', 'params']].copy()

    # Hyperparameters are the same across models, so one row per fold is enough.
    hyper_df = (
        hyper_df
        .drop(columns='model')
        .drop_duplicates(subset=['fold'])
        .reset_index(drop=True)
    )

    if isinstance(hyper_df['params'].iloc[0], dict):
        hyper_df['params'] = hyper_df['params'].apply(
            lambda x: "\n".join(f"{k}: {v}" for k, v in x.items())
        )

    return hyper_df


def create_hyperparameter_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics for hyperparameters across folds."""
    if 'params' not in df.columns:
        return pd.DataFrame()

    if not isinstance(df['params'].iloc[0], dict):
        return pd.DataFrame()

    all_params = pd.json_normalize(df['params'])
    param_summary = all_params.describe().T

    return param_summary


def combine_results_with_pvalues(
    df_mean: pd.DataFrame,
    df_p_values: pd.DataFrame
) -> pd.DataFrame:
    """Combine mean/std results with p-values into one table with interleaved mean, std, p columns per metric."""
    df_p_values.set_index(['model', 'network'], inplace=True)
    df_p_values.columns = pd.MultiIndex.from_tuples(
        [(col, 'p') for col in df_p_values.columns]
    )

    df_combined = pd.concat([df_mean, df_p_values], axis=1)
    df_combined = df_combined.sort_index(axis=1, level=0)

    # mean, std, p in that order within each metric.
    desired_order = ["mean", "std", "p"]
    df_combined = df_combined.loc[
        :,
        sorted(
            df_combined.columns,
            key=lambda x: (x[0], desired_order.index(x[1]))
        )
    ]

    return df_combined
