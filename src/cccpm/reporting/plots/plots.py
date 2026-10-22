import os

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from pandas.api.types import is_numeric_dtype

from cccpm.constants import INCREMENTABLE_METRICS
from cccpm.reporting.plots.figure_style import (
    COLOR_MAP,
    SQUARE,
    apply_nature_style,
    save_figure,
)

MODEL_ORDER = ["covariates", "connectome", "full", "increment"]

# Metrics with a fixed, meaningful reference value, drawn as a dashed line.
# Classification: chance level. Regression: "no relationship".
_REFERENCE_LINE = {
    "pearson_score": 0.0,
    "explained_variance_score": 0.0,
    "accuracy": 0.5,
    "balanced_accuracy": 0.5,
    "roc_auc": 0.5,
}

# df column name -> bool, is `increment` (full - covariates) defined for this metric.
_INCREMENTABLE_NAMES = {m.name for m in INCREMENTABLE_METRICS}


def _shade(color, amount: float):
    """Blend *color* toward white (amount > 0) or black (amount < 0)."""
    r, g, b = mcolors.to_rgb(color)
    if amount >= 0:
        return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)
    amount = -amount
    return (r * (1 - amount), g * (1 - amount), b * (1 - amount))


def scatter_plot_main(
    df: pd.DataFrame,
    results_folder: str,
    y_name,
    task_type: str = "regression",
    model: str = "connectome",
    network: str = "both",
    annotate_value: float | None = None,
    annotate_label: str = "r",
    annotate_p: str | None = None,
    name: str = "hero_scatter",
) -> str:
    """
    Single square predicted-vs-observed panel for the summary section.

    Regression: scatter + identity line + annotated effect size (and p).
    Classification: predicted-probability distribution per true class
    (overlaid histograms) + 0.5 decision line. A scatter/strip plot puts the
    binary true label on an axis as if it were continuous; the two classes'
    probability distributions are the thing that is actually informative
    here, so they are the plot, with true class as colour instead of an axis.

    If *annotate_value* is given it is shown verbatim (e.g. the cross-validated
    mean r reported in the hero verdict, so the two agree); otherwise the pooled
    correlation across all plotted points is computed and shown. The data area is
    forced square (equal x/y) via ``set_box_aspect``.
    """
    apply_nature_style()

    sub = df[(df["model"] == model) & (df["network"] == network)]
    if sub.empty:
        sub = df[df["model"] == model]

    color = COLOR_MAP.get(network, "#000000")
    fig, ax = plt.subplots(figsize=SQUARE)

    if task_type == "classification":
        classes = sorted(sub["y_true"].dropna().unique())
        light, dark = _shade(color, 0.7), _shade(color, -0.25)
        palette = {classes[0]: light, classes[-1]: dark} if len(classes) == 2 else None
        sns.histplot(
            data=sub, x="y_pred", hue="y_true", hue_order=classes or None,
            palette=palette, element="step", stat="count", common_norm=False,
            alpha=0.65, bins=12, binrange=(0, 1), linewidth=0.8, ax=ax,
        )
        ax.axvline(x=0.5, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("count")
        ax.set_xlim(-0.02, 1.02)
        # Annotations (effect size / p) anchor top-left; keep the legend clear of them.
        if ax.get_legend() is not None:
            sns.move_legend(
                ax, "upper right", title=y_name,
                fontsize=6, title_fontsize=6, frameon=False,
            )
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
        ax.set_xlabel(y_name)
        ax.set_ylabel(f"predicted {y_name}")

    # Effect-size annotation (r, with p underneath when provided).
    if annotate_value is not None:
        eff = f"{annotate_label} = {annotate_value:.2f}"
    elif task_type != "classification":
        eff = f"{annotate_label} = {sub['y_true'].corr(sub['y_pred']):.2f}"
    else:
        eff = ""
    if eff:
        ax.annotate(eff, xy=(0.05, 0.93), xycoords="axes fraction",
                    fontsize=8, fontweight="bold", color=color)
    if annotate_p:
        ax.annotate(annotate_p, xy=(0.05, 0.84), xycoords="axes fraction",
                    fontsize=7, color=color)

    # Force a square data area so x and y get equal space.
    ax.set_box_aspect(1)
    sns.despine(ax=ax)
    fig.tight_layout(pad=0.4)
    return save_figure(fig, os.path.join(results_folder, name))


def scatter_plot(df: pd.DataFrame, results_folder: str, y_name) -> str:
    apply_nature_style()

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
    for the connectome model and two networks ('positive', 'negative').
    """
    apply_nature_style()

    # Filter relevant data
    df = df[df["model"] == "connectome"]
    df = df[df["network"].isin(["positive", "negative"])]

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
    for the connectome model and two networks ('positive', 'negative').
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
    One small-multiples figure summarising model performance.

    Metrics are columns; models are on the y-axis; network is the hue. When
    ``increment`` (full - covariates) is one of the models, it gets its own
    x-axis per metric column, separate from covariates/connectome/full: the
    increment is a *difference* of the metric and can be negative or exceed
    the metric's own natural range (e.g. a balanced-accuracy increment can run
    from -0.2 to 1), while chance-level for the other three models already
    sits at 0.5 (accuracy/balanced accuracy/ROC AUC) or 0 (Pearson r/explained
    variance) — sharing one x-axis would force one of the two onto a scale
    that hides its structure.
    """
    apply_nature_style()

    if models is None:
        models = [m for m in MODEL_ORDER if m in df["model"].unique()]

    has_increment = "increment" in models
    main_models = [m for m in models if m != "increment"]
    hue_order = ["positive", "negative", "both"]
    n_cols = len(metrics)

    fig = plt.figure(figsize=(1.75 * n_cols + 0.6, 3.6 if has_increment else 2.4))
    if has_increment:
        gs = gridspec.GridSpec(
            2, n_cols, figure=fig,
            height_ratios=[max(len(main_models), 1), 1],
            hspace=0.9,
        )
    else:
        gs = gridspec.GridSpec(1, n_cols, figure=fig)

    for col, metric in enumerate(metrics):
        ax_main = fig.add_subplot(gs[0, col])
        sns.boxplot(
            data=df[df["model"].isin(main_models)],
            y="model", x=metric, hue="network",
            order=main_models,
            hue_order=hue_order,
            palette=COLOR_MAP,
            orient="h",
            fliersize=1.5,
            linewidth=0.5,
            width=0.6,
            legend=False,
            ax=ax_main,
        )
        ax_main.set_title(metric.replace("_", " "), fontsize=7)
        ax_main.set_xlabel("")
        ax_main.set_ylabel("")
        if metric in _REFERENCE_LINE:
            ax_main.axvline(
                _REFERENCE_LINE[metric], color="black", linewidth=0.5,
                linestyle="--", zorder=0,
            )
        if col > 0:
            ax_main.set_yticklabels([])
        sns.despine(ax=ax_main)

        if has_increment:
            ax_inc = fig.add_subplot(gs[1, col])
            if metric in _INCREMENTABLE_NAMES:
                sns.boxplot(
                    data=df[df["model"] == "increment"],
                    y="model", x=metric, hue="network",
                    order=["increment"],
                    hue_order=hue_order,
                    palette=COLOR_MAP,
                    orient="h",
                    fliersize=1.5,
                    linewidth=0.5,
                    width=0.6,
                    legend=False,
                    ax=ax_inc,
                )
                ax_inc.axvline(0, color="black", linewidth=0.5, linestyle="--", zorder=0)
            else:
                ax_inc.set_xticks([])
                ax_inc.set_yticks([0])
                ax_inc.set_yticklabels(["increment"])
                ax_inc.text(
                    0.5, 0.5, "not defined", ha="center", va="center",
                    fontsize=6, color="#888888", transform=ax_inc.transAxes,
                )
            ax_inc.set_xlabel("")
            ax_inc.set_ylabel("")
            if col > 0:
                ax_inc.set_yticklabels([])
            sns.despine(ax=ax_inc)

    handles = [
        mpatches.Patch(facecolor=COLOR_MAP[net], edgecolor="#404040", linewidth=0.5, label=net)
        for net in hue_order
    ]
    fig.tight_layout(pad=0.3)
    fig.legend(
        handles=handles, labels=hue_order, title="",
        loc="center left", bbox_to_anchor=(1.0, 0.5),
        frameon=False, handletextpad=0.5,
    )
    fig.subplots_adjust(right=0.93)

    return save_figure(fig, os.path.join(results_folder, "performance_grid"))


def roc_curve_grid(
    df_predictions: pd.DataFrame,
    results_folder: str,
    models: list[str] | None = None,
) -> str:
    """
    ROC curve per model (columns), one curve per network (colour), from the
    pooled out-of-sample predictions.

    ``performance_grid`` already shows ROC AUC as a boxplot across folds; this
    shows the curve that number comes from, pooled across all folds (as the
    hero scatters do for their effect size) rather than per-fold, since a
    single fold's 10-20 held-out cases makes for a visually noisy curve. Two
    models can share the same AUC with different curve shapes (e.g. one
    trades sensitivity for specificity differently) -- that trade-off is the
    one thing the boxplot cannot show.
    """
    from sklearn.metrics import auc, roc_curve

    apply_nature_style()

    if models is None:
        models = [m for m in MODEL_ORDER if m != "increment" and m in df_predictions["model"].unique()]
    hue_order = ["positive", "negative", "both"]
    n_cols = max(len(models), 1)

    fig, axes = plt.subplots(1, n_cols, figsize=(1.9 * n_cols + 0.4, 2.3), sharex=True, sharey=True)
    axes = [axes] if n_cols == 1 else list(axes)

    for ax, model in zip(axes, models):
        # The covariates model doesn't touch the connectome, so its prediction
        # is identical across "networks" -- one curve, not three overlapping ones.
        networks = ["both"] if model == "covariates" else hue_order
        for network in networks:
            sub = df_predictions[
                (df_predictions["model"] == model) & (df_predictions["network"] == network)
            ]
            if sub.empty or sub["y_true"].nunique() < 2:
                continue
            fpr, tpr, _ = roc_curve(sub["y_true"], sub["y_pred"])
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr, tpr, color=COLOR_MAP[network], linewidth=1.1,
                label=f"{network} ({roc_auc:.2f})",
            )
        ax.plot([0, 1], [0, 1], color="gray", linewidth=0.5, linestyle="--", zorder=0)
        ax.set_title(model, fontsize=7)
        ax.set_xlabel("false positive rate")
        ax.set_box_aspect(1)
        ax.legend(
            fontsize=5, frameon=False, loc="lower right",
            handlelength=1.2, labelspacing=0.25, borderaxespad=0.2,
        )
        sns.despine(ax=ax)

    axes[0].set_ylabel("true positive rate")
    fig.tight_layout(pad=0.4)
    return save_figure(fig, os.path.join(results_folder, "roc_curves"))


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
    """
    Pairplot where each cell picks scatter / histogram-by-group / crosstab
    based on whether the two variables are continuous or categorical.

    A numeric column with only 2 distinct values (e.g. a 0/1 classification
    target, or a binary covariate like sex) is treated as categorical, not
    continuous: plotting it on a scatter axis produces two overlapping strips
    of points with no readable structure, so it is routed to the
    histogram-by-group / crosstab branches instead, matching how this
    codebase defines "binary" elsewhere (``validation.detect_task_type``).
    """
    sns.set_theme(style="white")
    variables = df.columns
    n = len(variables)
    fig, axes = plt.subplots(n, n, figsize=(2.5 * n, 2.5 * n))

    def is_continuous(series: pd.Series) -> bool:
        return is_numeric_dtype(series) and series.dropna().nunique() > 2

    for i, row_var in enumerate(variables):
        for j, col_var in enumerate(variables):
            ax = axes[i, j]
            ax.set_xlabel("")
            ax.set_ylabel("")

            x = df[col_var]
            y = df[row_var]

            is_x_cont = is_continuous(x)
            is_y_cont = is_continuous(y)

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