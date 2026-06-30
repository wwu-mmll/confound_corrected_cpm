"""
Section builders for the Jinja2-based HTML report.

Each function returns an HTML fragment string (or a simple value) that is
inserted directly into the Jinja2 template context.  The data layer
(ReportDataLoader) and figure-generation code (plots/) are reused unchanged.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cccpm.reporting.reporting_utils import (
    embed_image_base64,
    embed_svg,
    format_results_table,
    styler_to_html,
)
from cccpm.reporting.table_builders import (
    combine_results_with_pvalues,
    create_edge_stability_table,
    create_hyperparameter_table,
)
from cccpm.reporting.plots.plots import (
    boxplot_models,
    classification_scatter_plot,
    histograms_network_strengths,
    scatter_plot,
    scatter_plot_covariates_model,
    scatter_plot_network_strengths,
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _png_to_html(png_path: str) -> str:
    """Embed a PNG as an <img> data-URI or return a placeholder."""
    return embed_image_base64(png_path)


def _svg_to_html(svg_path: str) -> str:
    """Inline an SVG file or fall back to embedding as base64 PNG."""
    p = Path(svg_path)
    if p.exists():
        return embed_svg(str(p))
    # try the corresponding PNG
    png_path = p.with_suffix(".png")
    if png_path.exists():
        return embed_image_base64(str(png_path))
    return f'<p class="missing-asset">Figure not found: {p.name}</p>'


# ---------------------------------------------------------------------------
# 1. Overview section
# ---------------------------------------------------------------------------

def build_overview_context(
    results_directory: str,
    version: str,
    run_date: str,
) -> dict:
    """
    Build the context variables for the Overview section.

    Returns a dict with keys: version, run_date, config_items, headline.
    """
    from cccpm.reporting.reporting_utils import parse_config_block

    log_path = os.path.join(results_directory, "cpm_log.txt")
    config_items: list[tuple[str, str]] = []
    if os.path.exists(log_path):
        config_items = parse_config_block(log_path)

    return {
        "version": version,
        "run_date": run_date,
        "config_items": config_items,
        "headline": "",
    }


# ---------------------------------------------------------------------------
# 2. Data section
# ---------------------------------------------------------------------------

def build_data_context(
    summary_df: Optional[pd.DataFrame],
    scatter_matrix_path: str,
) -> dict:
    """Build context for the Data section."""
    summary_table_html = ""
    if summary_df is not None:
        summary_table_html = summary_df.to_html(classes="data-table", border=0)

    scatter_html = ""
    if os.path.exists(scatter_matrix_path):
        scatter_html = embed_image_base64(scatter_matrix_path)

    return {
        "summary_table": summary_table_html,
        "scatter_matrix": scatter_html,
    }


# ---------------------------------------------------------------------------
# 3. Predictive Performance section
# ---------------------------------------------------------------------------

def build_performance_context(
    df_full: pd.DataFrame,
    df_mean: pd.DataFrame,
    df_p_values: Optional[pd.DataFrame],
    df_predictions: pd.DataFrame,
    y_name: str,
    task_type: str,
    plots_dir: str,
) -> dict:
    """Build context for the Predictive Performance section."""
    # Results table
    results_table_html = ""
    if df_p_values is not None:
        try:
            df_combined = combine_results_with_pvalues(df_mean.copy(), df_p_values.copy())
            styled = format_results_table(df_combined)
            results_table_html = styler_to_html(styled)
        except Exception:
            pass

    # Boxplots (one per metric)
    boxplots: list[tuple[str, str]] = []
    metric_cols = [c for c in df_full.columns if c not in ("fold", "model", "network", "params", "run")]
    for metric in metric_cols:
        try:
            png_path = boxplot_models(
                df_full, metric, plots_dir, models=["connectome", "covariates", "full", "residuals"]
            )
            boxplots.append((metric.replace("_", " "), _png_to_html(png_path)))
        except Exception:
            pass

    # Prediction scatter
    scatter_pred_html = ""
    scatter_cov_html = ""
    try:
        if task_type == "classification":
            png = classification_scatter_plot(df_predictions, plots_dir, y_name)
        else:
            png = scatter_plot(df_predictions, plots_dir, y_name)
        scatter_pred_html = _png_to_html(png)
    except Exception:
        pass

    try:
        png = scatter_plot_covariates_model(df_predictions, plots_dir, y_name)
        scatter_cov_html = _png_to_html(png)
    except Exception:
        pass

    return {
        "results_table": results_table_html,
        "boxplots": boxplots,
        "scatter_predictions": scatter_pred_html,
        "scatter_covariates": scatter_cov_html,
    }


# ---------------------------------------------------------------------------
# 4. Network Strengths section
# ---------------------------------------------------------------------------

def build_network_strengths_context(
    df_network_strengths: pd.DataFrame,
    y_name: str,
    plots_dir: str,
) -> dict:
    """Build context for the Network Strengths section."""
    ns_scatter_html = ""
    ns_hist_html = ""
    try:
        png = scatter_plot_network_strengths(df_network_strengths, plots_dir, y_name)
        ns_scatter_html = _png_to_html(png)
    except Exception:
        pass

    try:
        png = histograms_network_strengths(df_network_strengths, plots_dir, y_name)
        ns_hist_html = _png_to_html(png)
    except Exception:
        pass

    return {
        "ns_scatter": ns_scatter_html,
        "ns_hist": ns_hist_html,
    }


# ---------------------------------------------------------------------------
# 5. Brain Plots section
# ---------------------------------------------------------------------------

def build_brain_plots_context(
    results_directory: str,
    atlas_labels: Optional[pd.DataFrame],
) -> dict:
    """Build context for the Brain Plots section."""
    if atlas_labels is None:
        return {"has_atlas": False, "brain_positive": "", "brain_negative": ""}

    from cccpm.reporting.plots.cpm_chord_plot import plot_netplotbrain

    brain_positive = ""
    brain_negative = ""
    try:
        pos_path, _ = plot_netplotbrain(
            results_folder=results_directory,
            selected_metric="sig_stability_positive_edges",
            atlas_labels=atlas_labels,
        )
        brain_positive = _png_to_html(pos_path)
    except Exception:
        pass

    try:
        neg_path, _ = plot_netplotbrain(
            results_folder=results_directory,
            selected_metric="sig_stability_negative_edges",
            atlas_labels=atlas_labels,
        )
        brain_negative = _png_to_html(neg_path)
    except Exception:
        pass

    return {
        "has_atlas": True,
        "brain_positive": brain_positive,
        "brain_negative": brain_negative,
    }


# ---------------------------------------------------------------------------
# 6. Stable Edges section
# ---------------------------------------------------------------------------

def build_stable_edges_context(
    edge_stability: Optional[np.ndarray],
    edge_stability_significance: Optional[np.ndarray],
    atlas_labels: Optional[pd.DataFrame],
) -> dict:
    """Build context for the Stable Edges section."""
    if edge_stability is None or edge_stability_significance is None:
        return {"has_edge_data": False, "edge_table_positive": "", "edge_table_negative": ""}

    edge_table_positive = ""
    edge_table_negative = ""
    try:
        for i, (network, attr) in enumerate(
            [("positive", "edge_table_positive"), ("negative", "edge_table_negative")]
        ):
            edges = {
                "stability": edge_stability[:, :, i, :].squeeze(),
                "stability_significance": edge_stability_significance[:, :, i],
            }
            df = create_edge_stability_table(edges, atlas_labels)
            html = df.to_html(classes="data-table", border=0)
            if network == "positive":
                edge_table_positive = html
            else:
                edge_table_negative = html
    except Exception:
        pass

    return {
        "has_edge_data": True,
        "edge_table_positive": edge_table_positive,
        "edge_table_negative": edge_table_negative,
    }
