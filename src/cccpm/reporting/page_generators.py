"""
Page generation utilities for HTML report.

This module contains classes and functions for generating individual
pages/sections of the CPM HTML report.
"""

import os
import pandas as pd
import numpy as np
import arakawa as ar
from pathlib import Path
from typing import Optional

from cccpm.reporting.plots.plots import (
    boxplot_model_performance,
    scatter_plot,
    scatter_plot_covariates_model,
    scatter_plot_network_strengths,
    histograms_network_strengths,
    boxplot_models
)
from cccpm.reporting.plots.cpm_chord_plot import plot_netplotbrain
from cccpm.reporting.reporting_utils import format_results_table, extract_log_block
from cccpm.reporting.table_builders import (
    create_edge_stability_table,
    create_hyperparameter_table,
    create_hyperparameter_summary,
    combine_results_with_pvalues
)


class PageGenerator:
    """
    Base class for generating individual report pages.

    Attributes:
        results_directory: Path to CPM results directory
        plots_dir: Path to plots subdirectory
    """

    def __init__(self, results_directory: str, plots_dir: str):
        """
        Initialize page generator.

        Args:
            results_directory: Path to results directory
            plots_dir: Path to plots output directory
        """
        self.results_directory = results_directory
        self.plots_dir = plots_dir


class InfoPageGenerator(PageGenerator):
    """Generates the information/welcome page."""

    def generate(self) -> ar.Blocks:
        """
        Create the info page with package description and analysis setup.

        Returns:
            Arakawa Blocks object for the info page
        """
        log_text = extract_log_block(
            os.path.join(self.results_directory, "cpm_log.txt")
        )

        # Package information section
        info_text = ar.Group(
            ar.Text("""
        # Confound-Corrected Connectome-Based Predictive Modeling
        ## Python Toolbox
        **Author**: Nils R. Winter  
        **GitHub**: https://github.com/wwu-mmll/cpm_python

        **Confound-Corrected Connectome-Based Predictive Modelling** is a Python package for performing connectome-based
        predictive modeling (**CPM**). This toolbox is designed for researchers in neuroscience and psychiatry, providing
        robust methods for building **predictive models** based on structural or functional **connectome** data. It emphasizes
        replicability, interpretability, and flexibility, making it a valuable tool for analyzing brain connectivity
        and its relationship to behavior or clinical outcomes.
        """),
            ar.Text("**Version: 0.1.0**"),
            widths=[7, 1],
            columns=2
        )

        # Analysis setup section
        header = ar.Text("## Analysis Setup")
        log_block = ar.Text(f"<pre>{log_text}</pre>")
        log_group = ar.Group(
            ar.Blocks(blocks=[header, log_block]),
            ar.Text("Current Analysis"),
            columns=2,
            widths=[7, 1]
        )

        return ar.Blocks(blocks=[info_text, log_group], label="Info")


class DataDescriptionPageGenerator(PageGenerator):
    """Generates the data description page."""

    def generate(
        self,
        df_predictions: pd.DataFrame,
        atlas_labels: Optional[pd.DataFrame]
    ) -> ar.Blocks:
        """
        Create data description page with target variable statistics.

        Args:
            df_predictions: DataFrame with predictions and true values
            atlas_labels: DataFrame with atlas labels (optional)

        Returns:
            Arakawa Blocks object for the data description page
        """
        target_column_name = 'y_true'
        target_series = df_predictions[target_column_name]

        # Target variable description
        target_desc = f"""
        ## Target Variable Description

        - Number of observations: {len(df_predictions)}
        - Target variable name: {target_column_name}
        - Range: {target_series.min():.2f} to {target_series.max():.2f}
        - Mean: {target_series.mean():.2f}
        - Standard deviation: {target_series.std():.2f}
        """

        # Feature description
        feature_desc = """
        ## Dummy Feature Description

        Connectivity features were derived from:
        """

        if atlas_labels is not None:
            feature_desc += f"\n- Atlas labels provided: {len(atlas_labels)} regions"

        return ar.Blocks(
            blocks=[ar.Text(target_desc), ar.Text(feature_desc)],
            label='Data Description'
        )


class DataInsightsPageGenerator(PageGenerator):
    """Generates the data insights page with summary stats and scatter matrix."""

    def generate(
        self,
        summary_df: Optional[pd.DataFrame],
        scatter_matrix_path: str
    ) -> ar.Blocks:
        """
        Create page with input data summary and scatter matrix.

        Args:
            summary_df: DataFrame with summary statistics
            scatter_matrix_path: Path to scatter matrix image

        Returns:
            Arakawa Blocks object for the data insights page
        """
        # Summary table
        if summary_df is not None:
            summary_block = ar.DataTable(df=summary_df, label="Input Data Summary")
        else:
            summary_block = ar.Text("Summary file not found.")

        # Scatter matrix image
        if os.path.exists(scatter_matrix_path):
            scatter_block = ar.Media(
                file=scatter_matrix_path,
                name="ScatterMatrix",
                caption="Scatter matrix of covariates and target."
            )
        else:
            scatter_block = ar.Text("Scatter matrix image not found.")

        row = ar.Group(
            name='data_overview',
            blocks=[summary_block, scatter_block],
            columns=2,
            widths=[1, 1]
        )

        return ar.Blocks(blocks=[row], label='Target, Covariates & Features')


class HyperparametersPageGenerator(PageGenerator):
    """Generates the hyperparameters page."""

    def generate(self, df_full: pd.DataFrame) -> ar.Blocks:
        """
        Create page showing hyperparameters used in each fold.

        Args:
            df_full: Full CV results DataFrame

        Returns:
            Arakawa Blocks object for the hyperparameters page
        """
        try:
            if 'params' not in df_full.columns:
                return ar.Blocks(
                    blocks=[ar.Text("No hyperparameters found in results.")],
                    label='Hyperparameters'
                )

            # Create fold-wise hyperparameter table
            hyper_df = create_hyperparameter_table(df_full)
            hyper_table = ar.DataTable(
                df=hyper_df,
                label='Hyperparameters by Fold and Model'
            )

            # Try to create summary statistics table
            try:
                if isinstance(df_full['params'].iloc[0], dict):
                    param_summary = create_hyperparameter_summary(df_full)

                    if not param_summary.empty:
                        summary_table = ar.DataTable(
                            df=param_summary,
                            label='Hyperparameter Summary Statistics'
                        )

                        return ar.Blocks(
                            blocks=[
                                ar.Text("## Model Hyperparameters"),
                                ar.Text("### Parameters Used in Each Fold"),
                                hyper_table,
                                ar.Text("### Parameter Summary Across Folds"),
                                summary_table
                            ],
                            label='Hyperparameters'
                        )
            except Exception:
                pass

            # Return basic version if summary creation fails
            return ar.Blocks(
                blocks=[
                    ar.Text("## Model Hyperparameters"),
                    ar.Text("### Parameters Used in Each Fold"),
                    hyper_table
                ],
                label='Hyperparameters'
            )

        except Exception as e:
            return ar.Blocks(
                blocks=[ar.Text(f"Could not generate hyperparameters page: {str(e)}")],
                label='Hyperparameters'
            )


class MainResultsPageGenerator(PageGenerator):
    """Generates the main results page with performance metrics."""

    def generate(
        self,
        df_full: pd.DataFrame,
        df_mean: pd.DataFrame,
        df_p_values: pd.DataFrame,
        df_predictions: pd.DataFrame,
        y_name: str
    ) -> ar.Blocks:
        """
        Create main results page with performance plots and tables.

        Args:
            df_full: Full CV results
            df_mean: Mean CV results
            df_p_values: P-values from permutation tests
            df_predictions: Predictions DataFrame
            y_name: Name of target variable

        Returns:
            Arakawa Blocks object for the main results page
        """
        # Combine results with p-values
        df_combined = combine_results_with_pvalues(df_mean.copy(), df_p_values.copy())

        # Format and create results table
        styled_df = format_results_table(df_combined)
        table = ar.HTML(styled_df.to_html(escape=False), label='Predictive Performance')

        # Generate boxplots for each metric
        bar_plot_blocks = self._create_metric_boxplots(df_full)

        # Generate prediction scatter plots
        scatter_block = self._create_prediction_scatter(df_predictions, y_name)
        scatter_block_covariates = self._create_covariates_scatter(df_predictions, y_name)

        # Arrange in layout
        first_row = ar.Group(
            name='main_results',
            blocks=[ar.Select(blocks=bar_plot_blocks), scatter_block, scatter_block_covariates],
            columns=3,
            widths=[2, 1, 1]
        )

        second_row = ar.Group(
            name='perms_and_predictions',
            blocks=[table],
            columns=2,
            widths=[2, 1]
        )

        return ar.Blocks(blocks=[first_row, second_row], label='Predictive Performance')

    def _create_metric_boxplots(self, df_full: pd.DataFrame) -> list:
        """Generate boxplots for all metrics."""
        bar_plot_blocks = []

        # Get metric columns (skip fold, model, network, params)
        for metric in list(df_full.columns)[3:-1]:
            if metric == 'params':
                continue

            # Plot main models
            plot_name = boxplot_model_performance(
                df_full,
                metric,
                self.plots_dir,
                models=["covariates", "connectome", "full", "residuals"]
            )

            # Plot increment model separately
            plot_name_increment = boxplot_model_performance(
                df_full,
                metric,
                self.plots_dir,
                models=["increment"],
                filename_suffix="increment"
            )

            # Create combined block
            plot_block = ar.Blocks(
                blocks=[
                    ar.Media(file=plot_name, name=f"Image1_{metric}"),
                    ar.Media(file=plot_name_increment, name=f"Image1_increment_{metric}")
                ],
                label=f'{metric}'
            )

            bar_plot_blocks.append(plot_block)

        return bar_plot_blocks

    def _create_prediction_scatter(self, df_predictions: pd.DataFrame, y_name: str) -> ar.Media:
        """Generate scatter plot of predictions."""
        scatter_plot_name = scatter_plot(df_predictions, self.plots_dir, y_name)
        return ar.Media(
            file=scatter_plot_name,
            name="Predictions",
            caption="Scatter plot of true versus predicted scores.",
            label='predictions'
        )

    def _create_covariates_scatter(self, df_predictions: pd.DataFrame, y_name: str) -> ar.Media:
        """Generate scatter plot for covariates model."""
        scatter_covariates_name = scatter_plot_covariates_model(
            df_predictions,
            self.plots_dir,
            y_name
        )
        return ar.Media(
            file=scatter_covariates_name,
            name="PredictionsCovariatesModel",
            caption="Scatter plot of true versus predicted scores for the covariates model.",
            label='predictions_covariates'
        )


class MainResultsPageGenerator2(PageGenerator):
    """Generates the main results page with performance metrics."""

    def generate(
        self,
        df_full: pd.DataFrame,
        df_mean: pd.DataFrame,
        df_p_values: pd.DataFrame,
        df_predictions: pd.DataFrame,
        y_name: str,
        task_type: str = 'regression'
    ) -> ar.Blocks:
        """
        Create main results page with performance plots and tables.

        Args:
            df_full: Full CV results
            df_mean: Mean CV results
            df_p_values: P-values from permutation tests
            df_predictions: Predictions DataFrame
            y_name: Name of target variable
            task_type: 'regression' or 'classification'

        Returns:
            Arakawa Blocks object for the main results page
        """

        # Generate boxplots for each metric
        bar_plot_blocks = self._create_boxplots(df_full)

        # Generate prediction scatter plots
        scatter_block = self._create_prediction_scatter(df_predictions, y_name, task_type)
        scatter_block_covariates = self._create_covariates_scatter(df_predictions, y_name)

        # Select metric choices based on task type
        if task_type == 'classification':
            metric_choices = ['accuracy', 'balanced_accuracy', 'f1_score', 'roc_auc']
        else:
            metric_choices = ['explained_variance_score', 'pearson_score', 'mean_squared_error']

        # Arrange in layout
        first_row = ar.Group(
            name='header',
            blocks=[ar.Text("## Brain Connectome"), ar.ChoiceField('Choosemetric', metric_choices)],
            columns=2,
            widths=[4, 1]
        )

        second_row = ar.Group(
            name='main_results',
            blocks=[ar.Select(blocks=bar_plot_blocks, type=ar.SelectType.DROPDOWN), scatter_block, scatter_block_covariates],
            columns=3,
            widths=[1, 1, 1]
        )


        return ar.Blocks(blocks=[first_row, second_row], label='Predictive Performance')

    def _create_boxplots(self, df_full: pd.DataFrame) -> list:
        """Generate boxplots for all metrics."""
        bar_plot_blocks = []

        # Get metric columns (skip fold, model, network, params)
        for metric in list(df_full.columns)[3:-1]:
            if metric == 'params':
                continue

            # Plot main models
            plot_name = boxplot_models(
                df_full,
                metric,
                self.plots_dir,
                models=["connectome"]
            )

            # Create combined block
            plot_block = ar.Blocks(
                blocks=[
                    ar.Media(file=plot_name, name=f"Image1_{metric}", caption=f"Model performance for {metric}."),
                ],
                label=f'{metric}'
            )

            bar_plot_blocks.append(plot_block)

        return bar_plot_blocks

    def _create_prediction_scatter(self, df_predictions: pd.DataFrame, y_name: str,
                                    task_type: str = 'regression') -> ar.Media:
        """Generate scatter plot of predictions."""
        if task_type == 'classification':
            from cccpm.reporting.plots.plots import classification_scatter_plot
            plot_name = classification_scatter_plot(df_predictions, self.plots_dir, y_name)
            return ar.Media(
                file=plot_name,
                name="Predictions",
                caption="Predicted probability by true class.",
                label='predictions'
            )
        else:
            scatter_plot_name = scatter_plot(df_predictions, self.plots_dir, y_name)
            return ar.Media(
                file=scatter_plot_name,
                name="Predictions",
                caption="Scatter plot of true versus predicted scores.",
                label='predictions'
            )

    def _create_covariates_scatter(self, df_predictions: pd.DataFrame, y_name: str) -> ar.Media:
        """Generate scatter plot for covariates model."""
        scatter_covariates_name = scatter_plot_covariates_model(
            df_predictions,
            self.plots_dir,
            y_name
        )
        return ar.Media(
            file=scatter_covariates_name,
            name="PredictionsCovariatesModel",
            caption="Scatter plot of true versus predicted scores for the covariates model.",
            label='predictions_covariates'
        )

class NetworkStrengthsPageGenerator(PageGenerator):
    """Generates the network strengths page."""

    def generate(self, df_network_strengths: pd.DataFrame, y_name: str) -> ar.Blocks:
        """
        Create page showing network strength analyses.

        Args:
            df_network_strengths: DataFrame with network strength values
            y_name: Name of target variable

        Returns:
            Arakawa Blocks object for the network strengths page
        """
        # Scatter plot of network strengths vs target
        scatter_network_strengths = scatter_plot_network_strengths(
            df_network_strengths,
            self.plots_dir,
            y_name
        )
        scatter_block = ar.Media(
            file=scatter_network_strengths,
            name="NetworkStrengths",
            caption="Scatter plot of target versus network strength scores.",
            label='Network Strengths'
        )

        # Histograms of network strengths
        hist = histograms_network_strengths(df_network_strengths, self.plots_dir, y_name)
        hist_block = ar.Media(
            file=hist,
            name="NetworkStrengthsHist",
            caption="Histograms of network strength scores.",
            label='Distribution of Network Strengths'
        )

        row = ar.Group(
            name='network_strengths',
            blocks=[scatter_block, hist_block],
            columns=4
        )

        return ar.Blocks(blocks=[row], label='Network Strengths')


class BrainPlotsPageGenerator(PageGenerator):
    """Generates the brain visualization page."""

    def generate(self, atlas_labels: Optional[pd.DataFrame]) -> ar.Blocks:
        """
        Create page with brain network visualizations.

        Args:
            atlas_labels: DataFrame with atlas region labels

        Returns:
            Arakawa Blocks object for the brain plots page
        """
        if atlas_labels is None:
            return ar.Blocks(
                blocks=[ar.Group(blocks=[ar.Text("Provide atlas labels as csv file.")], columns=1)],
                label='Brain Plots'
            )

        plots = []
        edges = []

        # Generate plots for positive and negative edges
        for metric in ["sig_stability_positive_edges", "sig_stability_negative_edges"]:
            plot_brainplot, edge_list = plot_netplotbrain(
                results_folder=self.results_directory,
                selected_metric=metric,
                atlas_labels=atlas_labels
            )
            plots.append(plot_brainplot)
            edges.append(edge_list)

        # Create layout
        third_header = ar.Group(
            blocks=[
                ar.Text("Significantly Stable Positive Edges"),
                ar.Text("Significantly Stable Negative Edges")
            ],
            columns=2
        )
        third_row = ar.Group(
            blocks=[ar.Media(file=plots[0]), ar.Media(file=plots[1])],
            columns=2
        )

        return ar.Blocks(blocks=[third_header, third_row], label='Brain Plots')


class EdgeTablePageGenerator(PageGenerator):
    """Generates the edge stability table page."""

    def generate(
        self,
        edge_stability: Optional[np.ndarray],
        edge_stability_significance: Optional[np.ndarray],
        atlas_labels: Optional[pd.DataFrame]
    ) -> ar.Blocks:
        """
        Create page with edge stability tables.

        Args:
            edge_stability: Array with edge stability values
            edge_stability_significance: Array with significance values
            atlas_labels: DataFrame with atlas region labels

        Returns:
            Arakawa Blocks object for the edge table page
        """
        if edge_stability is None or edge_stability_significance is None:
            return ar.Blocks(
                blocks=[ar.Text("No edge stability data available. Run with permutations to generate.")],
                label='Stable Edges'
            )

        dfs = {}

        # Create tables for positive and negative networks
        for i, network in enumerate(['positive', 'negative']):
            edges = {
                'stability': edge_stability[:, :, i, :].squeeze(),
                'stability_significance': edge_stability_significance[:, :, i]
            }
            dfs[network] = create_edge_stability_table(edges, atlas_labels)

        # Create layout
        first_header = ar.Group(
            blocks=[ar.Text("## Positive Edges"), ar.Text("## Negative Edges")],
            columns=2
        )
        first_row = ar.Group(
            blocks=[ar.DataTable(df=dfs['positive']), ar.DataTable(df=dfs['negative'])],
            columns=2
        )

        return ar.Blocks(blocks=[first_header, first_row], label='Stable Edges')
