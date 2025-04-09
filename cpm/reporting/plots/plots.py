from matplotlib import pyplot as plt
import seaborn as sns
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
import pandas as pd

#from cpm.reporting.edges import metric


def bar_plot(df, selected_metric, results_folder):
    #cmap = sns.color_palette("muted", n_colors=3)
    flatui = ["#d6d6d6", "#6C88C4","#FF5768"]
    cmap = sns.color_palette(flatui)

    # Dropdown for metric selection

    fig = sns.FacetGrid(df[df['model'].isin(['connectome', 'covariates', 'residuals', 'full'])],
                      row='model', col='network', hue='network',
                      aspect=3.5, height=2 / 2.54,
                      col_order = ["positive", "negative", "both"],
                      row_order=['covariates', 'connectome','full', 'residuals'],
                      sharex='col', sharey=False, palette=cmap)

    # then we add the densities kdeplots
    fig.map(sns.boxplot, selected_metric, "model")

    fig.set_titles(template='')
    fig.set(ylabel=None)
    for ax,m in zip(fig.axes[0,:], ["positive", "negative", "both"]):
        ax.set_title(m, fontweight='bold', fontsize=8)
    for ax,l in zip(fig.axes[:,0], ['covariates', 'connectome','full', 'residuals']):
        ax.set_ylabel(l, fontweight='bold', fontsize=6, rotation=0, ha='right', va='center')
    fig.set(yticks=[])

    fig.fig.subplots_adjust(hspace=0.2, wspace=0.15)
    if selected_metric == 'pearson_score' or selected_metric == 'spearman_score':
        for ax in fig.axes.ravel():
            ax.set_xlim(-0.5, 1)
        fig.map(plt.axvline, x=0, color='black', linewidth=0.5, zorder=-1)

    plot_name = os.path.join(results_folder, f'point_plot_{selected_metric}.png')
    #sns.despine(offset=1, trim=True)
    fig.savefig(plot_name, dpi=300)
    return plot_name, fig


def bar_plot_plotly(df, selected_metric, results_folder):
    # Set color palette
    color_mapping = {
        'connectome': 'rgb(99, 110, 250)',
        'covariates': 'rgb(239, 85, 59)',
        'residuals': 'rgb(0, 204, 150)',
        'full': 'rgb(171, 99, 250)',
        'increment': 'rgb(50, 50, 50)'
    }

    # Helper function to get mean and standard deviation
    def get_mean_std(df, metric, groupby_cols):
        agg_df = df.groupby(groupby_cols)[metric].agg(['mean', 'std']).reset_index()
        return agg_df

    # Prepare data
    data_left = get_mean_std(df[df['model'].isin(['connectome', 'covariates', 'residuals', 'full'])], selected_metric,
                             ['network', 'model'])
    data_right = get_mean_std(df[df['model'].isin(['increment'])], selected_metric, ['network', 'model'])

    # Create a subplot with 1 row and 2 columns
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Connectome, Covariates, Residuals, Full", "Increment"))

    # Left plot
    for model in data_left['model'].unique():
        filtered_data = data_left[data_left['model'] == model]
        fig.add_trace(go.Scatter(
            x=filtered_data['network'],
            y=filtered_data['mean'],
            mode='markers',
            name=model,
            error_y=dict(
                type='data',
                array=filtered_data['std'],
                visible=True
            ),
            marker=dict(color=color_mapping[model])
        ), row=1, col=1)

    # Right plot
    filtered_data_right = data_right[data_right['model'] == 'increment']
    fig.add_trace(go.Scatter(
        x=filtered_data_right['network'],
        y=filtered_data_right['mean'],
        mode='markers',
        name='increment',
        error_y=dict(
            type='data',
            array=filtered_data_right['std'],
            visible=True
        ),
        marker=dict(color=color_mapping['increment'])
    ), row=1, col=2)

    # Update layout
    fig.update_layout(height=400, width=1000, title_text="Interactive Error Plot")
    fig.update_xaxes(title_text="Network", row=1, col=1)
    fig.update_xaxes(title_text="Network", row=1, col=2)
    fig.update_yaxes(title_text=selected_metric, row=1, col=1)
    fig.update_yaxes(title_text=selected_metric, row=1, col=2)
    if selected_metric in ['pearson_score', 'spearman_score']:
        fig.update_yaxes(range=[-0.5, 1], col=1)
        fig.update_yaxes(range=[-0.5, 1], col=2)

    return fig


def scatter_plot(df, results_folder):
    g = sns.FacetGrid(df, row="model", col="network", margin_titles=True)
    g.map(sns.regplot, "y_true", "y_pred", scatter_kws={"alpha": 0.7, "s": 14, "edgecolor": "white"})
    g.set_titles(col_template="{col_name}", row_template="{row_name}", size=14)
    sns.despine(trim=True)
    plot_name = os.path.join(results_folder, 'predictions.png')
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
    sns.histplot(data['permuted_value'], kde=True, bins=20, **kwargs)
    for true_value in data['true_value'].unique():
        plt.axvline(true_value, color='red', linestyle='dashed', linewidth=1)
