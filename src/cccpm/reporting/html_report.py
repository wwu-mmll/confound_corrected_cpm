import os

from pathlib import Path

import pandas as pd
import numpy as np
import arakawa as ar
import matplotlib.pyplot as plt

from cccpm.reporting.plots.plots import boxplot_model_performance
from cccpm.reporting.plots.plots import (scatter_plot, scatter_plot_covariates_model, scatter_plot_network_strengths,
                                             histograms_network_strengths)
from cccpm.reporting.plots.cpm_chord_plot import plot_netplotbrain
from cccpm.reporting.reporting_utils import format_results_table, extract_log_block, load_results_from_folder, load_data_from_folder


class HTMLReporter:
    def __init__(self, results_directory: str, atlas_labels: str = None):
        self.results_directory = results_directory
        self.plots_dir = os.path.join(results_directory, "plots")
        self.X_names, self.y_name, self.covariates_names = self.load_variable_names()
        os.makedirs(self.plots_dir, exist_ok=True)

        # copy atlas labels file to plotting directory
        if atlas_labels is not None:
            self.atlas_labels = pd.read_csv(atlas_labels)
        else:
            self.atlas_labels = None

        # Load results
        self.df = pd.read_csv(os.path.join(results_directory, 'cv_results.csv'))
        self.df_mean = load_results_from_folder(results_directory, 'cv_results_mean_std.csv')
        self.df_mean = self.df_mean.reorder_levels(["model", "network"])
        model_order = ["covariates", "connectome", "full", "residuals", "increment"]
        network_order = ["positive", "negative", "both"]

        self.df_mean.index = pd.MultiIndex.from_frame(
            self.df_mean.index.to_frame().assign(
                model=pd.Categorical(self.df_mean.index.get_level_values("model"), categories=model_order, ordered=True),
                network=pd.Categorical(self.df_mean.index.get_level_values("network"), categories=network_order, ordered=True)
            )
        )
        # Now sort
        self.df_mean = self.df_mean.sort_index()
        self.df_predictions = load_data_from_folder(results_directory, 'cv_predictions.csv')
        self.df_p_values = load_data_from_folder(results_directory, 'p_values.csv')
        self.df_permutations = load_data_from_folder(results_directory, 'permutation_results.csv')
        self.df_network_strengths = load_data_from_folder(results_directory, 'cv_network_strengths.csv')

    def generate_html_report(self):

        info_page = self.generate_info_page()
        main_results_page = self.generate_main_results_page()
        edges_page = self.generate_brain_plot_page()
        edges_table_page = self.generate_edge_page()
        network_strength_page = self.generate_network_strengths_page()
        data_description_page = self.generate_data_description_page()
        hyperparameters_page = self.generate_hyperparameters_page()
        data_insight_page = self.generate_target_cov_features_insights_page()  # <-- add this

        report_blocks = [
            info_page,
            data_insight_page,
            data_description_page,
            main_results_page,
            hyperparameters_page,
            network_strength_page,
            edges_page,
            edges_table_page
        ]

        main_tabs = ar.Select(blocks=report_blocks)
        script_dir = Path(__file__).parent
        image_path = script_dir / 'assets/CCCPM.png'
        image_path = image_path.resolve()  # Optional: resolve to absolute path

        main_page = ar.Group(ar.Media(file=image_path, name="Logo"),
                        main_tabs,
                        widths=[1, 10], columns=2)
        report = ar.Report(blocks=[main_page])
        report.save(os.path.join(self.results_directory, 'report.html'),
                    open=False, formatting=ar.Formatting(width=ar.Width.FULL, accent_color="orange"))
        plt.close('all')
        return

    def generate_hyperparameters_page(self):
        try:
            if 'params' not in self.df.columns:
                return ar.Blocks(blocks=[ar.Text("No hyperparameters found in results.")],
                                 label='Hyperparameters')

            hyper_df = self.df[['fold', 'model', 'params']].copy()
            hyper_df = (
                hyper_df
                .drop(columns='model')  # we don’t need the model any more
                .drop_duplicates(subset=['fold'])  # keep first row per fold
                .reset_index(drop=True)
            )

            if isinstance(hyper_df['params'].iloc[0], dict):
                hyper_df['params'] = hyper_df['params'].apply(
                    lambda x: "\n".join(f"{k}: {v}" for k, v in x.items())
                )

            hyper_table = ar.DataTable(
                df=hyper_df,
                label='Hyperparameters by Fold and Model'
            )

            try:
                if isinstance(self.df['params'].iloc[0], dict):
                    all_params = pd.json_normalize(self.df['params'])
                    param_summary = all_params.describe().T
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

    def generate_data_description_page(self):
        target_column_name = 'y_true'
        target_series = self.df_predictions[target_column_name]

        target_desc = f"""
        ## Target Variable Description

        - Number of observations: {len(self.df_predictions)}
        - Target variable name: {target_column_name}
        - Range: {target_series.min():.2f} to {target_series.max():.2f}
        - Mean: {target_series.mean():.2f}
        - Standard deviation: {target_series.std():.2f}
        """

        feature_desc = """
        ## Dummy Feature Description

        Connectivity features were derived from:
        """

        if self.atlas_labels is not None:
            feature_desc += f"\n- Atlas labels provided: {len(self.atlas_labels)} regions"

        return ar.Blocks(blocks=[
            ar.Text(target_desc),
            ar.Text(feature_desc)
        ], label='Data Description')

    def generate_info_page(self):
        log_text = extract_log_block(os.path.join(self.results_directory, "cpm_log.txt"))
        # --- Page: Info ---
        info_text = ar.Group(ar.Text("""
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
                             widths=[7, 1], columns=2)


        header = ar.Text("## Analysis Setup")
        log_block = ar.Text(f"<pre>{log_text}</pre>")
        log_group = ar.Group(ar.Blocks(blocks=[header, log_block]), ar.Text("Current Analysis"), columns=2, widths=[7, 1])

        blocks = ar.Blocks(blocks=[info_text, log_group], label="Info")
        return blocks

    def generate_main_results_page(self):
        self.df_p_values.set_index(['model', 'network'], inplace=True)
        self.df_p_values.columns = pd.MultiIndex.from_tuples([(col, 'p') for col in self.df_p_values.columns])
        df_combined = pd.concat([self.df_mean, self.df_p_values], axis=1)
        df_combined = df_combined.sort_index(axis=1, level=0)
        desired_order = ["mean", "std", "p"]
        df_combined = df_combined.loc[:,
                      sorted(df_combined.columns, key=lambda x: (x[0], desired_order.index(x[1])))
                      ]

        # Style with smaller font
        styled_df = format_results_table(df_combined)
        table = ar.HTML(styled_df.to_html(escape=False), label='Predictive Performance')

        bar_plot_blocks = []
        for metric in list(self.df.columns)[3:-1]:
            if metric == 'params':
                continue
            plot_name = boxplot_model_performance(self.df, metric, self.plots_dir, models=["covariates", "connectome", "full", "residuals"])
            plot_name_increment = boxplot_model_performance(self.df, metric, self.plots_dir, models=["increment"], filename_suffix="increment")
            #plot_block = ar.Media(file=plot_name, name=f"Image1_{metric}", caption="Boxplot of main predictive performance",
            #                      label=f'{metric}')
            plot_block = ar.Blocks(blocks=[ar.Media(file=plot_name, name=f"Image1_{metric}"),
                                           #ar.HTML(plot_name_increment, name=f"Image1_increment_{metric}"),
                                           ar.Media(file=plot_name_increment, name=f"Image1_increment_{metric}")
                                           ],
                                   label=f'{metric}')
            #plot_block = ar.HTML(plot_name, name=f"Image1_{metric}")

            bar_plot_blocks.append(plot_block)

        # predictions scatter plot
        scatter_plot_name = scatter_plot(self.df_predictions, self.plots_dir, self.y_name)
        scatter_covariates_name = scatter_plot_covariates_model(self.df_predictions, self.plots_dir, self.y_name)

        scatter_block = ar.Media(file=scatter_plot_name, name=f"Predictions", caption="Scatter plot of true versus predicted scores.",
                              label='predictions')
        scatter_block_covariates = ar.Media(file=scatter_covariates_name, name=f"PredictionsCovariatesModel",
                                 caption="Scatter plot of true versus predicted scores for the covariates model.",
                                 label='predictions_covariates')

        first_row = ar.Group(name='main_results', blocks=[ar.Select(blocks=bar_plot_blocks), scatter_block, scatter_block_covariates], columns=3,
                             widths=[2, 1, 1])

        second_row = ar.Group(name='perms_and_predictions', blocks=[table], columns=2, widths=[2, 1])

        return ar.Blocks(blocks=[first_row, second_row], label='Predictive Performance')

    def generate_network_strengths_page(self):
        scatter_network_strengths = scatter_plot_network_strengths(self.df_network_strengths, self.plots_dir, self.y_name)
        scatter_block_network_strength = ar.Media(file=scatter_network_strengths, name=f"NetworkStrengths",
                                 caption="Scatter plot of target versus network strength scores.",
                                 label='Network Strengths')
        hist = histograms_network_strengths(self.df_network_strengths, self.plots_dir, self.y_name)
        hist_block_network_strength = ar.Media(file=hist, name=f"NetworkStrengthsHist",
                                                  caption="Histograms of network strength scores.",
                                                  label='Distribution of Network Strengths')
        row = ar.Group(name='network_strengths', blocks=[scatter_block_network_strength, hist_block_network_strength], columns=4)
        return ar.Blocks(blocks=[row], label='Network Strengths')

    def generate_brain_plot_page(self):
        if self.atlas_labels is None:
            return ar.Blocks(blocks=[ar.Group(blocks=[ar.Text("Provide atlas labels as csv file.")], columns=1)],
                             label='Brain Plots')
        plots = list()
        edges = list()
        for metric in ["sig_stability_positive_edges", "sig_stability_negative_edges"]:
            plot_brainplot, edge_list = plot_netplotbrain(results_folder=self.results_directory,
                                                          selected_metric=metric,
                                                          atlas_labels=self.atlas_labels)
            plots.append(plot_brainplot)
            edges.append(edge_list)

        third_header = ar.Group(blocks=[ar.Text("Significantly Stable Positive Edges"), ar.Text("Significantly Stable Negative Edges")], columns=2)
        third_row = ar.Group(blocks=[ar.Media(file=plots[0]), ar.Media(file=plots[1])], columns=2)
        blocks = ar.Blocks(blocks=[third_header, third_row], label='Brain Plots')
        return blocks

    def generate_edge_page(self):
        import numpy as np

        dfs = dict()
        for network in ['positive', 'negative']:
            edges = {'stability': np.load(os.path.join(self.results_directory, f"stability_{network}_edges.npy")),
                     'stability_significance': np.load(os.path.join(self.results_directory, f"sig_stability_{network}_edges.npy"))}
            dfs[network] = self.create_edge_table(edges, self.atlas_labels)


        first_header = ar.Group(blocks=[ar.Text("## Positive Edges"), ar.Text("## Negative Edges")], columns=2)
        first_row = ar.Group(blocks=[ar.DataTable(df=dfs['positive']), ar.DataTable(df=dfs['negative'])], columns=2)

        blocks = ar.Blocks(blocks=[first_header, first_row], label='Stable Edges')
        return blocks

    @staticmethod
    def create_edge_table(matrix, atlas):
        n = matrix['stability'].shape[0]
        stability = []
        significance = []
        region_a = []
        region_b = []

        for i in range(1, n):
            for j in range(i):
                if matrix['stability'][i, j] == 0:
                    continue
                if atlas is not None:
                    region_a.append(atlas['region'][i])
                    region_b.append(atlas['region'][j])
                else:
                    region_a.append(f"Region {i}")
                    region_b.append(f"Region {j}")
                stability.append(matrix['stability'][i, j])
                significance.append(matrix['stability_significance'][i, j])

        df = pd.DataFrame({'Region A': region_a, 'Region B': region_b,
                           'Stability': stability, 'Stability Significance': significance})
        df[['Stability', 'Stability Significance']] = df[['Stability', 'Stability Significance']].round(5)

        df.sort_values(by=['Stability Significance', 'Stability'], inplace=True, ascending=[True, False])
        df.set_index(['Region A', 'Region B'], inplace=True)
        if df.empty:
            first_col = df.columns[0]
            row = {col: ("No significantly stable edges." if col == first_col else np.nan) for col in df.columns}
            df = pd.DataFrame([row])
        return df

    def generate_target_cov_features_insights_page(self):
        """
           Generate a page summarizing the input data:
           - summary statistics from summary.csv
           - scatter matrix image
           """
        summary_path = os.path.join(self.results_directory, "data_insights", "summary.csv")
        scatter_matrix_path = os.path.join(self.results_directory, "data_insights", "scatter_matrix.png")

        # Load summary table
        if os.path.exists(summary_path):
            summary_df = pd.read_csv(summary_path, index_col=0)
            summary_block = ar.DataTable(df=summary_df, label="Input Data Summary")
        else:
            summary_block = ar.Text("Summary file not found.")

        # Load scatter matrix image
        if os.path.exists(scatter_matrix_path):
            scatter_block = ar.Media(file=scatter_matrix_path, name="ScatterMatrix",
                                     caption="Scatter matrix of covariates and target.")
        else:
            scatter_block = ar.Text("Scatter matrix image not found.")

        # Combine both into a single report block
        row = ar.Group(name='data_overview', blocks=[summary_block, scatter_block], columns=2, widths=[1, 1])
        return ar.Blocks(blocks=[row], label='Target, Covariates & Features')

    def load_variable_names(self):
        X_names = pd.read_csv(os.path.join(self.results_directory, 'data_insights', "X_names.csv"), header=None)[0].tolist()
        y_name = pd.read_csv(os.path.join(self.results_directory, 'data_insights', "y_name.csv"), header=None).iloc[0, 0]
        covar_names = pd.read_csv(os.path.join(self.results_directory, 'data_insights', "covariate_names.csv"), header=None)[0].tolist()
        return X_names, y_name, covar_names

if __name__=="__main__":
    reporter = HTMLReporter(results_directory='/spm-data/vault-data3/mmll/projects/cpm_python/results_new/hcp_SSAGA_TB_Yrs_Smoked_spearman_partial_p=0.01')
    reporter.generate_html_report()
