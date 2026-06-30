"""
HTML Report Generator for Confound-Corrected CPM Analysis.

This module provides the main HTMLReporter class that orchestrates the generation
of comprehensive HTML reports for CPM analysis results.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import arakawa as ar

from cccpm.reporting.data_loader import ReportDataLoader
from cccpm.reporting.page_generators import (
    InfoPageGenerator,
    DataDescriptionPageGenerator,
    DataInsightsPageGenerator,
    HyperparametersPageGenerator,
    MainResultsPageGenerator2,
    NetworkStrengthsPageGenerator,
    BrainPlotsPageGenerator,
    EdgeTablePageGenerator
)


class HTMLReporter:
    """
    Main class for generating HTML reports from CPM analysis results.

    This class orchestrates the entire report generation process by:
    1. Loading all necessary data via ReportDataLoader
    2. Creating individual report pages via specialized page generators
    3. Assembling pages into a complete HTML report

    Attributes:
        results_directory: Path to the directory containing CPM results
        plots_dir: Path to the plots subdirectory
        data_loader: Instance of ReportDataLoader for loading data
        X_names: List of feature names
        y_name: Name of the target variable
        covariates_names: List of covariate names
        atlas_labels: DataFrame with atlas region labels (optional)
    """

    def __init__(self, results_directory: str, atlas_labels: str = None):
        """
        Initialize the HTML reporter.

        Args:
            results_directory: Path to directory containing CPM analysis results
            atlas_labels: Path to CSV file with atlas region labels (optional)
        """
        self.results_directory = results_directory
        self.plots_dir = os.path.join(results_directory, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

        # Initialize data loader
        self.data_loader = ReportDataLoader(results_directory, atlas_labels)

        # Load variable names
        self.X_names, self.y_name, self.covariates_names = \
            self.data_loader.load_variable_names()

        # Store atlas labels reference
        self.atlas_labels = self.data_loader.atlas_labels

        # Load task type
        self.task_type = self.data_loader.load_task_type()

        # Load all data needed for report generation
        self._load_data()

    def _load_data(self):
        """Load all required data for report generation."""
        self.df, self.df_mean = self.data_loader.load_cv_results()
        self.df_predictions = self.data_loader.load_predictions()
        self.df_p_values = self.data_loader.load_p_values()
        self.df_permutations = self.data_loader.load_permutations()
        self.df_network_strengths = self.data_loader.load_network_strengths()

    def generate_html_report(self):
        """
        Generate the complete HTML report.

        This method orchestrates the generation of all report pages and
        assembles them into a single HTML file saved in the results directory.
        """
        # Load additional data needed for specific pages
        summary_df, scatter_path, _ = self.data_loader.load_data_insights()
        edge_stability, edge_stability_significance = self.data_loader.load_edge_stability()

        # Generate all report pages using specialized generators
        report_blocks = [
            InfoPageGenerator(self.results_directory, self.plots_dir).generate(),

            DataInsightsPageGenerator(self.results_directory, self.plots_dir).generate(
                summary_df, scatter_path
            ),

            DataDescriptionPageGenerator(self.results_directory, self.plots_dir).generate(
                self.df_predictions, self.atlas_labels
            ),

            MainResultsPageGenerator2(self.results_directory, self.plots_dir).generate(
                self.df, self.df_mean, self.df_p_values, self.df_predictions, self.y_name,
                self.task_type
            ),

            HyperparametersPageGenerator(self.results_directory, self.plots_dir).generate(
                self.df
            ),

            NetworkStrengthsPageGenerator(self.results_directory, self.plots_dir).generate(
                self.df_network_strengths, self.y_name
            ),

            BrainPlotsPageGenerator(self.results_directory, self.plots_dir).generate(
                self.atlas_labels
            ),

            EdgeTablePageGenerator(self.results_directory, self.plots_dir).generate(
                edge_stability, edge_stability_significance, self.atlas_labels
            )
        ]

        # Create tabbed interface
        main_tabs = ar.Select(blocks=report_blocks)

        # Add logo
        script_dir = Path(__file__).parent
        image_path = (script_dir / 'assets/CCCPM.png').resolve()

        main_page = ar.Group(
            ar.Media(file=image_path, name="Logo"),
            main_tabs,
            widths=[1, 10],
            columns=2
        )

        # Generate and save report
        report = ar.Report(blocks=[main_page])
        report.save(
            os.path.join(self.results_directory, 'report.html'),
            open=False,
            formatting=ar.Formatting(width=ar.Width.FULL, accent_color="orange")
        )

        # Clean up matplotlib figures
        plt.close('all')

if __name__=="__main__":
    reporter = HTMLReporter(results_directory='/spm-data/vault-data3/mmll/projects/cpm_python/results_new/hcp_SSAGA_TB_Yrs_Smoked_spearman_partial_p=0.01')
    reporter.generate_html_report()
