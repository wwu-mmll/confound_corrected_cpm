import os

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

from pandas.api.types import is_numeric_dtype

from cccpm.reporting.plots.figure_style import (
    COLOR_MAP,
    PANEL,
    SQUARE,
    WIDE,
    WIDE_TALL,
    apply_nature_style,
    save_figure,
)

MODEL_ORDER = ["covariates", "connectome", "full", "residuals", "increment"]


def scatter_plot_main(
    df: pd.DataFrame,
    results_folder: str,
    y_name,
    task_type: str = "regression",
    model: str = "connectome",
    network: str = "both",
) -> str:
    """
    Single prominent predicted-vs-observed panel for the hero section.

    Regression: scatter + identity line + annotated Pearson r.
    Classification: predicted probability stripped by true class + 0.5 line.
    """
    apply_nature_style()

    sub = df[(df["model"] == model) & (df["network"] == network)]
    if sub.empty:
        sub = df[df["model"] == model]

    color = COLOR_MAP.get(network, "#000000")
    fig, ax = plt.subplots(figsize=SQUARE)

    if task_type == "classification":
        sns.stripplot(
            data=sub, x="y_true", y="y_pred",
            jitter=0.2, alpha=0.7, size=4, edgecolor="white", linewidth=0.3,
            color=color, ax=ax,
        )
        ax.axhline(y=0.5, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlabel(f"true {y_name}")
        ax.set_ylabel("predicted probability")
    else:
        sns.regplot(
            data=sub, x="y_true", y="y_pred",
            scatter_kws={"alpha": 0.7, "s": 16, "edgecolor": "white", "color": color},
            line_kws={"color": color, "linewidth": 1.0},
            ax=ax,
        )
        # Identity line (perfect prediction reference)
        lims = [
            min(sub["y_true"].min(), sub["y_pred"].min()),
            max(sub["y_true"].max(), sub["y_pred"].max()),
        ]
        ax.plot(lims, lims, color="gray", linewidth=0.5, linestyle="--", zorder=0)
        # Annotate Pearson r (pooled across folds)
        r = sub["y_true"].corr(sub["y_pred"])
        ax.annotate(
            f"r = {r:.2f}", xy=(0.05, 0.92), xycoords="axes fraction",
            fontsize=8, fontweight="bold", color=color,
        )
        ax.set_xlabel(y_name)
        ax.set_ylabel(f"predicted {y_name}")

    sns.despine(ax=ax, trim=True)
    fig.tight_layout(pad=0.4)
    return save_figure(fig, os.path.join(results_folder, "hero_scatter"))


def scatter_plot(df: pd.DataFrame, results_folder: str, y_name) -> str:
    apply_nature_style()

    #df = df[df['model'].isin(['connectome', 'residuals', 'full'])]
    df = df[df['model'].isin(['connectome'])]

    def regplot_colored(data, **kwargs):
        color = COLOR_MAP.get(data['network'].iloc[0], "#000000")
        sns.regplot(
            data=data,
            x="y_true", y="y_pred",
            scatter_kws={"alpha": 0.7, "s": 14, "edgecolor": "white", "color": color},
            line_kws={"color": color, "linewidth": 0.75},
            **kwargs
        )

    g = sns.FacetGrid(df, row="network", col="model", margin_titles=True, height=1.5, aspect=1)
    g.map_dataframe(regplot_colored)
    g.set_titles(col_template="{col_name}", row_template="{row_name}", size=7)
    g.set_xlabels(y_name)
    g.set_ylabels(f"predicted {y_name}")
    sns.despine(trim=True)
    g.fig.tight_layout(pad=0.5)

    return save_figure(g.fig, os.path.join(results_folder, "predictions"))


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
    plt.tight_layout(pad=0.5)

    return save_figure(fig, os.path.join(results_folder, "scatter_covariates"))


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

    return save_figure(g.fig, os.path.join(results_folder, "histograms_network_strengths"))


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

    return save_figure(g.fig, os.path.join(results_folder, "scatter_network_strengths"))

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
    fig.tight_layout(pad=0.2)
    return save_figure(fig, os.path.join(results_folder, f"boxplot_{metric}{suffix}"))


def boxplot_models(
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
    fig, ax = plt.subplots(figsize=(3, 2))

    sns.boxplot(
        data=df,
        y=metric,
        x="network",
        orient="v",
        fliersize=2,
        linewidth=0.5,
        width=0.3,
        ax=ax
    )

    if metric in ["pearson_score", "spearman_score", "explained_variance_score"]:
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_ylim(-0.3, 1)

    ax.set_ylabel(metric.replace("_", " "))
    ax.set_xlabel("network")
    sns.despine(ax=ax)

    # Save plot
    suffix = f"_{filename_suffix}" if filename_suffix else ""
    fig.tight_layout(pad=0.2)
    return save_figure(fig, os.path.join(results_folder, f"boxplot_{metric}{suffix}"))


def performance_grid(
    df: pd.DataFrame,
    metrics: list[str],
    results_folder: str,
    models: list[str] | None = None,
) -> str:
    """
    One faceted small-multiples figure summarising model performance.

    Metrics are columns; models are on the y-axis; network is the hue. This
    replaces the previous one-figure-per-metric stack with a single panel that
    makes the model x network x metric comparison legible at a glance.
    """
    apply_nature_style()

    if models is None:
        models = [m for m in MODEL_ORDER if m in df["model"].unique()]

    long = df.melt(
        id_vars=[c for c in ["model", "network", "fold", "run"] if c in df.columns],
        value_vars=[m for m in metrics if m in df.columns],
        var_name="metric",
        value_name="value",
    )
    long = long[long["model"].isin(models)]
    long["metric"] = long["metric"].str.replace("_", " ")

    g = sns.catplot(
        data=long,
        kind="box",
        y="model", x="value", hue="network",
        col="metric",
        order=models,
        hue_order=["positive", "negative", "both"],
        palette=COLOR_MAP,
        orient="h",
        fliersize=1.5,
        linewidth=0.5,
        width=0.6,
        height=2.6,
        aspect=0.7,
        sharex=False,
        legend_out=True,
    )
    g.set_titles(col_template="{col_name}", size=7)
    g.set_axis_labels("", "")
    for ax in g.axes.flat:
        sns.despine(ax=ax)
    g.fig.tight_layout(pad=0.3)

    return save_figure(g.fig, os.path.join(results_folder, "performance_grid"))


def classification_scatter_plot(df: pd.DataFrame, results_folder: str, y_name) -> str:
    """
    Generate a strip/swarm plot of predicted probabilities by true class for classification.
    """
    apply_nature_style()

    df = df[df['model'].isin(['connectome'])]

    def stripplot_colored(data, **kwargs):
        color = COLOR_MAP.get(data['network'].iloc[0], "#000000")
        kwargs.pop('color', None)
        sns.stripplot(
            data=data,
            x="y_true", y="y_pred",
            jitter=0.2, alpha=0.7, size=3, edgecolor="white", linewidth=0.3,
            color=color,
            **kwargs
        )
        ax = plt.gca()
        ax.axhline(y=0.5, color="gray", linewidth=0.5, linestyle="--")

    g = sns.FacetGrid(df, row="network", col="model", margin_titles=True, height=1.5, aspect=1)
    g.map_dataframe(stripplot_colored)
    g.set_titles(col_template="{col_name}", row_template="{row_name}", size=7)
    g.set_xlabels(f"true {y_name}")
    g.set_ylabels("predicted probability")
    sns.despine(trim=True)
    g.fig.tight_layout(pad=0.5)

    return save_figure(g.fig, os.path.join(results_folder, "predictions_classification"))


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

    # Reduce label collisions: only show tick labels on the outer edge
    # (bottom row for x, left column for y), like a conventional pairplot.
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i != n - 1:
                ax.set_xticklabels([])
                ax.set_xlabel("")
            if j != 0:
                ax.set_yticklabels([])
                ax.set_ylabel("")

    fig.tight_layout(pad=0.4)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path