import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


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


def scatter_plot(df: pd.DataFrame, results_folder: str) -> str:
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
    g.set_xlabels("true")
    g.set_ylabels("predicted")
    sns.despine(trim=True)
    g.fig.tight_layout(pad=0.5)

    png_path = os.path.join(results_folder, "predictions.png")
    pdf_path = os.path.join(results_folder, "predictions.pdf")
    g.fig.savefig(png_path, dpi=600, bbox_inches="tight")
    g.fig.savefig(pdf_path, bbox_inches="tight")

    return png_path


def scatter_plot_covariates_model(df: pd.DataFrame, results_folder: str) -> str:
    """
    Generate a single scatter plot with regression line for the 'covariates' model.
    """
    apply_nature_style()

    df = df[df["model"] == "covariates"]

    fig, ax = plt.subplots(figsize=(2.5, 2.5))

    sns.regplot(
        data=df,
        x="y_true",
        y="y_pred",
        scatter_kws={"alpha": 0.7, "s": 14, "edgecolor": "white", "color": "black"},
        line_kws={"color": "black", "linewidth": 0.75},
        ax=ax
    )

    sns.despine(trim=True)
    ax.set_xlabel("true")
    ax.set_ylabel("predicted")
    ax.set_title('covariates')
    png_path = os.path.join(results_folder, "scatter_covariates.png")
    pdf_path = os.path.join(results_folder, "scatter_covariates.pdf")
    fig.tight_layout(pad=0.2)
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

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
    height = 0.75 if models == ["increment"] else 2
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
    fig.tight_layout(pad=0.2)
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    return png_path
