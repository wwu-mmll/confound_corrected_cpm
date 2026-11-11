"""Data loading utilities for HTML report generation."""

import os
import json
import pandas as pd
from typing import Tuple, Optional

from cccpm.reporting.plots.plots import MODEL_ORDER


class ReportDataLoader:
    """Loads and prepares all data required for HTML report generation."""

    def __init__(self, results_directory: str, atlas_labels: Optional[str] = None):
        self.results_directory = results_directory
        self.atlas_labels = self._load_atlas_labels(atlas_labels)

    def _load_atlas_labels(self, atlas_labels_path: Optional[str]) -> Optional[pd.DataFrame]:
        """Load atlas labels from CSV file if provided."""
        if atlas_labels_path is not None:
            return pd.read_csv(atlas_labels_path)
        return None

    def load_variable_names(self) -> Tuple[list, str, list]:
        """Returns (feature_names, target_name, covariate_names) from saved CSV files."""
        data_insights_dir = os.path.join(self.results_directory, 'data_insights')

        X_names = pd.read_csv(
            os.path.join(data_insights_dir, "X_names.csv"),
            header=None
        )[0].tolist()

        y_name = pd.read_csv(
            os.path.join(data_insights_dir, "y_name.csv"),
            header=None
        ).iloc[0, 0]

        # Written as an empty file when the analysis ran without covariates;
        # pandas cannot parse a file with no columns.
        covar_path = os.path.join(data_insights_dir, "covariate_names.csv")
        if os.path.getsize(covar_path) == 0:
            covar_names = []
        else:
            covar_names = pd.read_csv(covar_path, header=None)[0].tolist()

        return X_names, y_name, covar_names

    def load_available_models(self) -> list:
        """Model variants this run defines, in report order; falls back to the full set if absent (old result dirs)."""
        path = os.path.join(self.results_directory, 'available_models.json')
        if not os.path.exists(path):
            return list(MODEL_ORDER)
        with open(path, encoding='utf-8') as f:
            available = set(json.load(f))
        return [m for m in MODEL_ORDER if m in available]

    def load_run_config(self) -> dict:
        """The confound configuration this run used; {} for old result dirs (treat missing keys as unknown, not default)."""
        path = os.path.join(self.results_directory, 'run_config.json')
        if not os.path.exists(path):
            return {}
        with open(path, encoding='utf-8') as f:
            return json.load(f)

    def load_cv_results(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Returns (full_results_df, summary_results_df)."""
        from cccpm.reporting.reporting_utils import load_results_from_folder

        df_full = pd.read_csv(
            os.path.join(self.results_directory, 'cv_results_full.csv')
        ).drop("run", axis=1)
        df_full = self._keep_available_models(df_full)

        df_summary = load_results_from_folder(
            self.results_directory,
            'cv_results_summary.csv'
        )
        df_summary = self._sort_summary_results(df_summary)

        return df_full, df_summary

    def _sort_summary_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort by model/network in standard order and drop the run level."""
        df = df.reorder_levels(["model", "network", "run"])

        # Drop NaN placeholder rows for variants this run does not define,
        # rather than printing a table of blanks.
        model_order = self.load_available_models()
        df = df[df.index.get_level_values("model").isin(model_order)]
        network_order = ["positive", "negative", "both"]

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

    def _keep_available_models(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows for model variants this run does not define."""
        if 'model' not in df.columns:
            return df
        return df[df['model'].isin(self.load_available_models())]

    def load_predictions(self) -> pd.DataFrame:
        """Load cross-validation predictions."""
        from cccpm.reporting.reporting_utils import load_data_from_folder
        return self._keep_available_models(
            load_data_from_folder(self.results_directory, 'cv_predictions.csv'))

    def load_p_values(self) -> Optional[pd.DataFrame]:
        """Load permutation test p-values."""
        csv_path = os.path.join(self.results_directory, 'p_values.csv')
        if os.path.exists(csv_path):
            return self._keep_available_models(pd.read_csv(csv_path))
        return None

    def load_permutations(self) -> Optional[pd.DataFrame]:
        """Load permutation test results."""
        perm_dir = os.path.join(self.results_directory, 'permutation')
        csv_path = os.path.join(perm_dir, 'cv_results_summary.csv')
        if os.path.exists(csv_path):
            return self._keep_available_models(pd.read_csv(csv_path))
        return None

    def load_network_strengths(self) -> pd.DataFrame:
        """Load network strength values."""
        from cccpm.reporting.reporting_utils import load_data_from_folder
        return self._keep_available_models(
            load_data_from_folder(self.results_directory, 'cv_network_strengths.csv'))

    def load_edge_stability(self) -> Tuple:
        """Returns (edge_stability, edge_stability_significance) matrices."""
        import numpy as np

        stability_path = os.path.join(self.results_directory, "stability_edges.npy")
        significance_path = os.path.join(self.results_directory, "stability_edges_significance.npy")

        edge_stability = np.load(stability_path) if os.path.exists(stability_path) else None
        edge_stability_significance = np.load(significance_path) if os.path.exists(significance_path) else None

        return edge_stability, edge_stability_significance

    def load_edge_significance_meta(self) -> Optional[dict]:
        """Load edge-significance diagnostics written by the permutation step, or None if absent."""
        import json

        path = os.path.join(self.results_directory, "stability_edges_significance_meta.json")
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                return json.load(f)
        return None

    def load_data_insights(self) -> Tuple[Optional[pd.DataFrame], str, str]:
        """Returns (summary_df, scatter_matrix_path, summary_csv_path)."""
        insights_dir = os.path.join(self.results_directory, "data_insights")
        summary_path = os.path.join(insights_dir, "summary.csv")
        scatter_path = os.path.join(insights_dir, "scatter_matrix.png")

        summary_df = None
        if os.path.exists(summary_path):
            summary_df = pd.read_csv(summary_path, index_col=0)

        return summary_df, scatter_path, summary_path

    def load_task_type(self) -> str:
        """Returns 'regression' or 'classification'."""
        task_type_path = os.path.join(self.results_directory, 'task_type.txt')
        if os.path.exists(task_type_path):
            with open(task_type_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        return 'regression'

    def load_all(self) -> dict:
        """Load all data needed for report generation."""
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
