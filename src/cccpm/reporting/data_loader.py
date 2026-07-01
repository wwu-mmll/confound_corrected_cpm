"""
Data loading utilities for HTML report generation.

This module handles all data loading operations for the CPM HTML reporter,
including results files, predictions, network strengths, and metadata.
"""

import os
import pandas as pd
from typing import Tuple, Optional


class ReportDataLoader:
    """
    Loads and prepares all data required for HTML report generation.

    Attributes:
        results_directory: Path to the directory containing CPM results
        atlas_labels: DataFrame containing atlas region labels (optional)
    """

    def __init__(self, results_directory: str, atlas_labels: Optional[str] = None):
        """
        Initialize the data loader.

        Args:
            results_directory: Path to directory containing CPM analysis results
            atlas_labels: Path to CSV file containing atlas region labels (optional)
        """
        self.results_directory = results_directory
        self.atlas_labels = self._load_atlas_labels(atlas_labels)

    def _load_atlas_labels(self, atlas_labels_path: Optional[str]) -> Optional[pd.DataFrame]:
        """Load atlas labels from CSV file if provided."""
        if atlas_labels_path is not None:
            return pd.read_csv(atlas_labels_path)
        return None

    def load_variable_names(self) -> Tuple[list, str, list]:
        """
        Load variable names from saved CSV files.

        Returns:
            Tuple of (feature_names, target_name, covariate_names)
        """
        data_insights_dir = os.path.join(self.results_directory, 'data_insights')

        X_names = pd.read_csv(
            os.path.join(data_insights_dir, "X_names.csv"),
            header=None
        )[0].tolist()

        y_name = pd.read_csv(
            os.path.join(data_insights_dir, "y_name.csv"),
            header=None
        ).iloc[0, 0]

        covar_names = pd.read_csv(
            os.path.join(data_insights_dir, "covariate_names.csv"),
            header=None
        )[0].tolist()

        return X_names, y_name, covar_names

    def load_cv_results(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load cross-validation results (full and summary).

        Returns:
            Tuple of (full_results_df, summary_results_df)
        """
        from cccpm.reporting.reporting_utils import load_results_from_folder

        # Load full CV results
        df_full = pd.read_csv(
            os.path.join(self.results_directory, 'cv_results_full.csv')
        ).drop("run", axis=1)

        # Load summary results
        df_summary = load_results_from_folder(
            self.results_directory,
            'cv_results_summary.csv'
        )

        # Reorder and sort the multi-index
        df_summary = self._sort_summary_results(df_summary)

        return df_full, df_summary

    def _sort_summary_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort summary results by model and network type in standard order.

        Args:
            df: DataFrame with multi-index (model, network, run)

        Returns:
            Sorted DataFrame with run level dropped
        """
        df = df.reorder_levels(["model", "network", "run"])

        model_order = ["covariates", "connectome", "full", "residuals", "increment"]
        network_order = ["positive", "negative", "both"]

        # Create categorical index for proper sorting
        df.index = pd.MultiIndex.from_frame(
            df.index.to_frame().assign(
                model=pd.Categorical(
                    df.index.get_level_values("model"),
                    categories=model_order,
                    ordered=True
                ),
                network=pd.Categorical(
                    df.index.get_level_values("network"),
                    categories=network_order,
                    ordered=True
                )
            )
        )

        return df.sort_index().droplevel("run")

    def load_predictions(self) -> pd.DataFrame:
        """Load cross-validation predictions."""
        from cccpm.reporting.reporting_utils import load_data_from_folder
        return load_data_from_folder(self.results_directory, 'cv_predictions.csv')

    def load_p_values(self) -> Optional[pd.DataFrame]:
        """Load permutation test p-values."""
        csv_path = os.path.join(self.results_directory, 'p_values.csv')
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        return None

    def load_permutations(self) -> Optional[pd.DataFrame]:
        """Load permutation test results."""
        perm_dir = os.path.join(self.results_directory, 'permutation')
        csv_path = os.path.join(perm_dir, 'cv_results_summary.csv')
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        return None

    def load_network_strengths(self) -> pd.DataFrame:
        """Load network strength values."""
        from cccpm.reporting.reporting_utils import load_data_from_folder
        return load_data_from_folder(self.results_directory, 'cv_network_strengths.csv')

    def load_edge_stability(self) -> Tuple:
        """
        Load edge stability matrices.

        Returns:
            Tuple of (edge_stability, edge_stability_significance)
        """
        import numpy as np

        stability_path = os.path.join(self.results_directory, "stability_edges.npy")
        significance_path = os.path.join(self.results_directory, "stability_edges_significance.npy")

        edge_stability = np.load(stability_path) if os.path.exists(stability_path) else None
        edge_stability_significance = np.load(significance_path) if os.path.exists(significance_path) else None

        return edge_stability, edge_stability_significance

    def load_edge_significance_meta(self) -> Optional[dict]:
        """Load the edge-significance diagnostics (method, null distributions,
        components) written by the permutation step, or ``None`` if absent."""
        import json

        path = os.path.join(self.results_directory, "stability_edges_significance_meta.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    def load_data_insights(self) -> Tuple[Optional[pd.DataFrame], str, str]:
        """
        Load data insight files (summary stats and scatter matrix).

        Returns:
            Tuple of (summary_df, scatter_matrix_path, exists_flag)
        """
        insights_dir = os.path.join(self.results_directory, "data_insights")
        summary_path = os.path.join(insights_dir, "summary.csv")
        scatter_path = os.path.join(insights_dir, "scatter_matrix.png")

        summary_df = None
        if os.path.exists(summary_path):
            summary_df = pd.read_csv(summary_path, index_col=0)

        return summary_df, scatter_path, summary_path

    def load_task_type(self) -> str:
        """
        Load task type from results directory.

        Returns:
            Task type string ('regression' or 'classification')
        """
        task_type_path = os.path.join(self.results_directory, 'task_type.txt')
        if os.path.exists(task_type_path):
            with open(task_type_path, 'r') as f:
                return f.read().strip()
        return 'regression'

    def load_all(self) -> dict:
        """
        Load all data needed for report generation.

        Returns:
            Dictionary containing all loaded data
        """
        X_names, y_name, covariates_names = self.load_variable_names()
        df_full, df_summary = self.load_cv_results()

        return {
            'X_names': X_names,
            'y_name': y_name,
            'covariates_names': covariates_names,
            'df_full': df_full,
            'df_summary': df_summary,
            'df_predictions': self.load_predictions(),
            'df_p_values': self.load_p_values(),
            'df_permutations': self.load_permutations(),
            'df_network_strengths': self.load_network_strengths(),
            'atlas_labels': self.atlas_labels
        }
