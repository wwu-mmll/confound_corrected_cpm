import os
import pandas as pd
import arakawa as ar

from plots.plots import bar_plot
from plots.plots import scatter_plot
from plots.cpm_chord_plot import plot_netplotbrain
from streamlit_utils import load_results_from_folder, load_data_from_folder, style_apa


def generate_html_report(results_directory):
    plots_dir = os.path.join(results_directory, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(os.path.join(results_directory, 'cv_results.csv'))
    df_main = load_results_from_folder(results_directory, 'cv_results_mean_std.csv')
    df_predictions = load_data_from_folder(results_directory, 'cv_predictions.csv')
    df_p_values = load_data_from_folder(results_directory, 'p_values.csv')
    df_permutations = load_data_from_folder(results_directory, 'permutation_results.csv')

    info_page = generate_info_page()
    main_results_page = generate_main_results_page(df, df_main)
    report_blocks = [
        info_page,
        main_results_page
    ]
    report = ar.Report(blocks=report_blocks)
    report.save('report.html', open=True)
    return


def generate_info_page():
    # --- Page: Info ---
    info_text = ar.Text("""
    # Confound-Corrected Connectome-Based Predictive Modeling
    ## Python Toolbox  
    **Software Version**: 0.1.0  
    **Author**: Nils R. Winter  
    **GitHub**: [cpm_python](https://github.com/wwu-mmll/cpm_python)
    """)
    return ar.Page(title='Toolbox Info', blocks=[info_text])


def generate_main_results_page(df, df_main):
    table = ar.HTML(style_apa(df_main).to_html())
    plot_name, fig = bar_plot(df, 'pearson_score', '')
    return ar.Page(title='Results', blocks=[table,
                                            ar.Plot(fig)])


"""
# --- Page: Main Results ---
selected_metric = df.columns[4]  # Example metric for default plot
plot_name, _ = bar_plot(df, selected_metric, RESULTS_DIR)
main_plot = ar.Plot(plot_name)
main_table = ar.HTML(style_apa(df_main).to_html())
main_results = ar.Select(
    main_plot,
    main_table
)

# --- Page: Selected Edges ---
plots = []
edges = []
labels = [
    ("Positive Edges", "positive_edges"),
    ("Negative Edges", "negative_edges"),
    ("Stability Positive Edges", "stability_positive_edges"),
    ("Stability Negative Edges", "stability_negative_edges"),
    ("Significant Stability Positive Edges", "sig_stability_positive_edges"),
    ("Significant Stability Negative Edges", "sig_stability_negative_edges")
]

for _, metric in labels:
    plot_img, edge_df = plot_netplotbrain(RESULTS_DIR, metric)
    plots.append(ar.Plot(plot_img))
    edges.append(ar.DataTable(edge_df))

edges_blocks = [
    ar.Group(p, e) for (l, _), p, e in zip(labels, plots, edges)
]
selected_edges = ar.Select(*edges_blocks)

# --- Page: Predictions ---
pred_img = scatter_plot(df_predictions, RESULTS_DIR)
predictions_block = ar.Plot(pred_img, label="Model Predictions")

# --- Page: Permutations ---
p_values_table = ar.HTML(style_apa(df_p_values.set_index(['network', 'model'])).to_html())
perm_img_path = os.path.join(PLOTS_DIR, 'permutations.png')
if os.path.exists(perm_img_path):
    permutations_plot = ar.Plot(perm_img_path)
else:
    permutations_plot = ar.Text("No permutation plot available.")
permutations_block = ar.Select(
    permutations_plot,
    p_values_table
)

# --- Final Report Assembly ---
report = ar.Blocks(
    ar.Page(info_block, title="Info"),
    ar.Page(main_results, title="Main Results"),
    ar.Page(selected_edges, title="Selected Edges"),
    ar.Page(predictions_block, title="Model Predictions"),
    ar.Page(permutations_block, title="Permutations")
)

# Save to HTML
ar.save_report(report, path="connectome_report.html", open=True)

"""

if __name__=="__main__":
    generate_html_report(results_directory='/spm-data/vault-data3/mmll/projects/cpm_python/results_new/hcp_SSAGA_TB_Yrs_Smoked_spearman_partial_p=0.01')
