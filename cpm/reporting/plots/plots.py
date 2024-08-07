from matplotlib import pyplot as plt
import seaborn as sns
import os
import numpy as np


def bar_plot(df, selected_metric, results_folder):
    cmap = sns.color_palette("muted")

    # Dropdown for metric selection
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left plot for connectome, covariates, and full
    sns.pointplot(
        data=df[df['model'].isin(['connectome', 'covariates', 'residuals', 'full'])],
        x="network", y=selected_metric, hue="model",
        dodge=0.4, linestyle="none", errorbar="sd", capsize=.1,
        ax=axes[0], palette=cmap, linewidth=1.5,
    )
    sns.move_legend(axes[0], "upper left", bbox_to_anchor=(1, 1))
    if selected_metric == 'pearson_score' or selected_metric == 'spearman_score':
        axes[0].set_ylim(-0.5, 1)

    # Right plot for increment
    sns.pointplot(
        data=df[df['model'].isin(['increment'])],
        x="network", y=selected_metric, hue="model",
        linestyle="none", errorbar="sd", capsize=.1,
        ax=axes[1], color='k', linewidth=1.5,
    )
    sns.move_legend(axes[1], "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    if selected_metric == 'pearson_score' or selected_metric == 'spearman_score':
        axes[1].set_ylim(-0.5, 1)

    plot_name = os.path.join(results_folder, "plots", 'point_plot.png')
    sns.despine(offset=1, trim=True)
    fig.savefig(plot_name, dpi=300)
    return plot_name, fig


def scatter_plot(df, results_folder):
    g = sns.FacetGrid(df, row="network", col="model", margin_titles=True)
    g.map(sns.regplot, "y_true", "y_pred", scatter_kws={"alpha": 0.7, "s": 10, "edgecolor": "white"})
    g.set_titles(col_template="{col_name}", row_template="{row_name}", size=14)
    sns.despine(trim=True)
    plot_name = os.path.join(results_folder, "plots", 'predictions.png')
    g.fig.savefig(plot_name)
    return plot_name


def corr_plot(results_folder, selected_metric):
    corr = np.load(os.path.join(results_folder, f"{selected_metric}.npy"))
    if (selected_metric == "sig_stability_positive_edges") or (selected_metric == "sig_stability_negative_edges"):
        threshold = 0.05
        corr_transformed = np.where(corr > threshold, 0, corr)
        corr_transformed = np.where(corr <= threshold, 1, corr_transformed)
        corr = corr_transformed

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr.T, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plot_name = os.path.join(results_folder, "plots", 'corr.png')
    fig.savefig(plot_name)
    return plot_name


# Define a plotting function that handles each subset
def plot_histogram_with_line(data, **kwargs):
    sns.histplot(data['permuted_value'], kde=False, **kwargs)
    for true_value in data['true_value'].unique():
        plt.axvline(true_value, color='red', linestyle='dashed', linewidth=1)
