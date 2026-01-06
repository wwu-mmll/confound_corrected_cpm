import os

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

from pandas.api.types import is_numeric_dtype


# Shared plotting settings
COLOR_MAP = {
    "positive": "#FF5768",
    "negative": "#6C88C4",
    "both": "#d6d6d6"
}

MODEL_ORDER = ["covariates", "connectome", "full", "residuals", "increment"]

def apply_nature_style():
    sns.set_theme(style="white")
    mpl.rcParams.update({
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "lines.linewidth": 0.75,
        "axes.linewidth": 0.5,
        "legend.fontsize": 6
    })


def scatter_plot(df: pd.DataFrame, results_folder: str, y_name) -> str:
    apply_nature_style()

    df = df[df['model'].isin(['connectome', 'residuals', 'full'])]

    def regplot_colored(data, **kwargs):
        color = COLOR_MAP.get(data['network'].iloc[0], "#000000")
        sns.regplot(
            data=data,
            x="y_true", y="y_pred",
            scatter_kws={"alpha": 0.7, "s": 14, "edgecolor": "white", "color": color},
            line_kws={"color": color, "linewidth": 0.75},
            **kwargs
        )

    g = sns.FacetGrid(df, row="model", col="network", margin_titles=True, height=1.5, aspect=1)
    g.map_dataframe(regplot_colored)
    g.set_titles(col_template="{col_name}", row_template="{row_name}", size=7)
    g.set_xlabels(y_name)
    g.set_ylabels(f"predicted {y_name}")
    sns.despine(trim=True)
    g.fig.tight_layout(pad=0.5)

    png_path = os.path.join(results_folder, "predictions.png")
    pdf_path = os.path.join(results_folder, "predictions.pdf")
    g.fig.savefig(png_path, dpi=600, bbox_inches="tight")
    g.fig.savefig(pdf_path, bbox_inches="tight")

    return png_path


def scatter_plot_covariates_model(df: pd.DataFrame, results_folder: str, y_name) -> str:
    """
    Generate a single scatter plot with regression line for the 'covariates' model.
    """
    apply_nature_style()

    df = df[df["model"] == "covariates"]

    # Create a figure with GridSpec
    fig = plt.figure(figsize=(6, 2))
    gs = gridspec.GridSpec(1, 3, figure=fig)

    # Create one subplot in the center cell
    ax = fig.add_subplot(gs[0, 1])

    sns.regplot(
        data=df,
        x="y_true",
        y="y_pred",
        scatter_kws={"alpha": 0.7, "s": 14, "edgecolor": "white", "color": "black"},
        line_kws={"color": "black", "linewidth": 0.75},
        ax=ax
    )

    sns.despine(trim=True)
    ax.set_xlabel(y_name)
    ax.set_ylabel(f"predicted {y_name}")
    ax.set_title('covariates')
    png_path = os.path.join(results_folder, "scatter_covariates.png")
    pdf_path = os.path.join(results_folder, "scatter_covariates.pdf")
    plt.tight_layout(pad=0.5)
    # This makes the figure 10x10 inches
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path)

    return png_path


def histograms_network_strengths(df: pd.DataFrame, results_folder: str, y_name) -> str:
    """
    Create a 2x2 grid of histograms showing the distribution of network_strength
    for two models ('connectome', 'residuals') and two networks ('positive', 'negative').
    """
    apply_nature_style()

    # Filter relevant data
    df = df[df["model"].isin(["connectome", "residuals"])]
    df = df[df["network"].isin(["positive", "negative"])]

    # Color mapping
    color_map = {
        "positive": "#FF5768",  # red
        "negative": "#6C88C4"   # blue
    }

    def histplot_colored(data, color=None, **kwargs):
        # Override color based on 'network' value
        network = data["network"].iloc[0]
        color = {"positive": "#FF5768", "negative": "#6C88C4"}[network]
        sns.histplot(
            data=data,
            x="network_strength",
            bins=30,
            edgecolor="white",
            linewidth=0.3,
            color=color,  # This now safely overrides the one passed by FacetGrid
            **kwargs
        )

    # Create 2x2 facet grid
    g = sns.FacetGrid(
        df,
        row="model",
        col="network",
        margin_titles=True,
        height=1.5,
        aspect=1
    )
    g.map_dataframe(histplot_colored)
    g.set_titles(col_template="{col_name}", row_template="{row_name}", size=7)
    g.set_axis_labels("network strength", y_name)
    sns.despine(trim=True)
    g.fig.tight_layout(pad=0.5)

    # Save
    png_path = os.path.join(results_folder, "histograms_network_strengths.png")
    pdf_path = os.path.join(results_folder, "histograms_network_strengths.pdf")
    g.fig.savefig(png_path, dpi=600, bbox_inches="tight")
    g.fig.savefig(pdf_path, bbox_inches="tight")

    return png_path


def scatter_plot_network_strengths(df: pd.DataFrame, results_folder: str, y_name) -> str:
    """
    Create a 2x2 scatter plot of y_true vs network_strength
    for two models ('connectome', 'residuals') and two networks ('positive', 'negative').
    """
    apply_nature_style()

    # Define color mapping
    color_map = {
        "positive": "#FF5768",  # red
        "negative": "#6C88C4"   # blue
    }

    # Plotting function with custom color per network
    def regplot_colored(data, **kwargs):
        network = data["network"].iloc[0]
        color = color_map.get(network, "black")
        sns.regplot(
            data=data,
            x="network_strength",
            y="y_true",
            scatter_kws={"alpha": 0.7, "s": 14, "edgecolor": "white", "color": color},
            line_kws={"color": color, "linewidth": 0.75},
            **kwargs
        )

    # Create 2x2 facet grid: rows = model, cols = network
    g = sns.FacetGrid(
        df,
        row="model",
        col="network",
        margin_titles=True,
        height=1.5,
        aspect=1
    )
    g.map_dataframe(regplot_colored)
    g.set_titles(col_template="{col_name}", row_template="{row_name}", size=7)
    g.set_axis_labels("network strength", y_name)
    sns.despine(trim=True)
    g.fig.tight_layout(pad=0.5)

    # Save
    png_path = os.path.join(results_folder, "scatter_network_strengths.png")
    pdf_path = os.path.join(results_folder, "scatter_network_strengths.pdf")
    g.fig.savefig(png_path, dpi=600, bbox_inches="tight")
    g.fig.savefig(pdf_path, bbox_inches="tight")

    return png_path

def boxplot_model_performance(
    df: pd.DataFrame,
    metric: str,
    results_folder: str,
    models: list[str],
    filename_suffix: str = ""
) -> str:
    """
    Creates a horizontal boxplot comparing models across network types.

    Parameters:
        df: Input dataframe.
        metric: Name of the column to be plotted on the x-axis.
        results_folder: Output folder path.
        models: List of model names to include (e.g. ['increment'] or others).
        filename_suffix: Optional string to append to the output filename.
    """
    apply_nature_style()

    df = df[df["model"].isin(models)]

    # Adjust figure size based on model count
    height = 0.75 if len(models) == 1 else 2
    fig, ax = plt.subplots(figsize=(7, height))

    sns.boxplot(
        data=df,
        x=metric,
        y="model",
        hue="network",
        order=models,
        hue_order=["both", "negative", "positive"],
        palette=COLOR_MAP,
        orient="h",
        fliersize=2,
        linewidth=0.5,
        width=0.5,
        ax=ax
    )

    if metric in ["pearson_score", "spearman_score", "explained_variance_score"]:
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.set_xlim(-0.5, 1)

    sns.despine(trim=True)
    ax.set_xlabel(metric.replace("_", " "))
    ax.set_ylabel("")
    # Move legend outside the plot
    ax.legend(
        title="",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        handletextpad=0.5
    )
    # Save plot
    suffix = f"_{filename_suffix}" if filename_suffix else ""
    png_path = os.path.join(results_folder, f"boxplot_{metric}{suffix}.png")
    pdf_path = os.path.join(results_folder, f"boxplot_{metric}{suffix}.pdf")
    svg_path = os.path.join(results_folder, f"boxplot_{metric}{suffix}.svg")
    fig.tight_layout(pad=0.2)
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")

    return png_path


def pairplot_flexible(df: pd.DataFrame, output_path: str) -> str:
    sns.set_theme(style="white")
    variables = df.columns
    n = len(variables)
    fig, axes = plt.subplots(n, n, figsize=(2.5 * n, 2.5 * n))

    for i, row_var in enumerate(variables):
        for j, col_var in enumerate(variables):
            ax = axes[i, j]
            ax.set_xlabel("")
            ax.set_ylabel("")

            x = df[col_var]
            y = df[row_var]

            is_x_cont = is_numeric_dtype(x)
            is_y_cont = is_numeric_dtype(y)

            if i == j:
                if is_x_cont:
                    sns.histplot(x, bins=20, ax=ax, color="gray", edgecolor="white")
                else:
                    counts = x.value_counts().sort_index()

                    sns.barplot(
                        x=counts.index.astype(str),
                        y=counts.values,
                        hue=counts.index.astype(str),  # ← now we have a hue
                        palette="pastel",
                        legend=False,
                        ax=ax
                    )

                    # rotate labels (you can also use tick_params as shown earlier)
                    ax.set_xticks(range(len(counts)))
                    ax.set_xticklabels(counts.index.astype(str), rotation=45, ha="right")
                ax.set_title(row_var, fontsize=9)
                sns.despine(ax=ax)
                continue

            if is_x_cont and is_y_cont:
                sns.scatterplot(x=x, y=y, ax=ax, s=15, alpha=0.6, edgecolor="white", linewidth=0.3)
            elif is_x_cont and not is_y_cont:
                sns.histplot(data=df, x=col_var, hue=row_var, ax=ax, element="step", stat="count",
                             common_norm=False, bins=20, palette="Set2")
            elif not is_x_cont and is_y_cont:
                sns.histplot(data=df, x=row_var, hue=col_var, ax=ax, element="step", stat="count",
                             common_norm=False, bins=20, palette="Set2")
            else:
                ctab = pd.crosstab(y, x)
                sns.heatmap(ctab, annot=True, fmt='d', cmap="Blues", cbar=False, ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va='center')

            ax.tick_params(axis='both', labelsize=6)
            sns.despine(ax=ax)

    #plt.tight_layout()
    fig.savefig(output_path, dpi=600)
    plt.close(fig)
    return output_path