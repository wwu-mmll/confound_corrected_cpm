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
    parse_config_block,
    styler_to_html,
)
from cccpm.reporting.table_builders import (
    combine_results_with_pvalues,
    create_edge_stability_table,
    create_hyperparameter_summary,
    create_hyperparameter_table,
)
from cccpm.reporting.plots.plots import (
    classification_scatter_plot,
    histograms_network_strengths,
    performance_grid,
    scatter_plot,
    scatter_plot_covariates_model,
    scatter_plot_main,
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
# Config / headline helpers
# ---------------------------------------------------------------------------

def _config_items(results_directory: str) -> list[tuple[str, str]]:
    log_path = os.path.join(results_directory, "cpm_log.txt")
    if os.path.exists(log_path):
        return parse_config_block(log_path)
    return []


def _config_dict(results_directory: str) -> dict[str, str]:
    return {k: v for k, v in _config_items(results_directory)}


def _format_p(p: float) -> str:
    """APA-style p-value string."""
    if p is None or pd.isna(p):
        return "p = n/a"
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".")


def _summary_value(summary_df: Optional[pd.DataFrame], label: str) -> Optional[str]:
    if summary_df is None or label not in summary_df.index:
        return None
    try:
        return str(summary_df.loc[label].iloc[0])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 1. Hero / executive summary
# ---------------------------------------------------------------------------

def build_hero_context(
    df_mean: pd.DataFrame,
    df_p_values: Optional[pd.DataFrame],
    df_predictions: pd.DataFrame,
    summary_df: Optional[pd.DataFrame],
    edge_stability,
    results_directory: str,
    plots_dir: str,
    y_name: str,
    task_type: str,
    version: str,
    run_date: str,
) -> dict:
    """
    Build the hero: a one-sentence verdict, key-stat chips, and the prominent
    predicted-vs-observed scatter for the headline (connectome, both) model.
    """
    import re

    cfg = _config_dict(results_directory)
    raw = "\n".join(f"{k}: {v}" for k, v in _config_items(results_directory))

    # Headline metric per task type, with graceful fallback.
    if task_type == "classification":
        metric, metric_label = "roc_auc", "AUC"
    else:
        metric, metric_label = "pearson_score", "r"

    value = None
    try:
        if (metric, "mean") in df_mean.columns and ("connectome", "both") in df_mean.index:
            value = float(df_mean.loc[("connectome", "both"), (metric, "mean")])
    except Exception:
        value = None

    pval = None
    if df_p_values is not None:
        try:
            row = df_p_values[
                (df_p_values["model"] == "connectome") & (df_p_values["network"] == "both")
            ]
            if not row.empty and metric in row.columns:
                pval = float(row[metric].iloc[0])
        except Exception:
            pval = None

    # CV folds + permutations from the config block.
    n_folds = None
    m = re.search(r"n_splits\s*=\s*(\d+)", raw)
    if m:
        n_folds = m.group(1)
    n_perm = cfg.get("Number of Permutations")

    verb = "classified" if task_type == "classification" else "predicted"
    if value is not None:
        cv_part = f"{n_folds}-fold CV" if n_folds else "cross-validation"
        perm_part = f", {n_perm} permutations" if n_perm else ""
        p_part = f", {_format_p(pval)}" if pval is not None else ""
        headline = (
            f"The connectome model {verb} {y_name}: "
            f"{metric_label} = {value:.2f}{p_part} ({cv_part}{perm_part})."
        )
    else:
        headline = ""

    # Stat chips.
    n_features = _summary_value(summary_df, "Number of features (connectivity values)")
    n_nodes = None
    if n_features is not None:
        try:
            from cccpm.utils import infer_n_nodes
            n_nodes = infer_n_nodes(int(float(n_features)))
        except Exception:
            n_nodes = None
    n_edges = None
    if edge_stability is not None:
        try:
            from cccpm.reporting.plots.connectome_utils import signed_stability_matrix
            sm = signed_stability_matrix(edge_stability)
            n_edges = int((np.abs(sm) > 1e-9).sum() // 2)
        except Exception:
            n_edges = None
    p_thresh = None
    m = re.search(r"threshold\s*=\s*\[([^\]]+)\]", raw)
    if m:
        p_thresh = m.group(1).strip()

    chips: list[tuple[str, str]] = []
    n_samples = _summary_value(summary_df, "Number of samples")
    if n_samples:
        chips.append(("Samples", n_samples))
    if n_nodes:
        chips.append(("Nodes", str(n_nodes)))
    if n_features:
        chips.append(("Edges (total)", str(int(float(n_features)))))
    if n_edges is not None:
        chips.append(("Stable edges", str(n_edges)))
    ncov = _summary_value(summary_df, "Number of covariates")
    if ncov:
        chips.append(("Covariates", ncov))
    if p_thresh:
        chips.append(("Edge p-threshold", p_thresh))
    if n_perm:
        chips.append(("Permutations", n_perm))

    # Hero figure — annotate with the same cross-validated r as the verdict
    # so the two never disagree (the pooled-points r would differ).
    hero_scatter = ""
    try:
        hero_scatter = _svg_to_html(
            scatter_plot_main(
                df_predictions, plots_dir, y_name, task_type=task_type,
                annotate_value=value if task_type != "classification" else None,
                annotate_label="mean CV r",
            )
        )
    except Exception:
        pass

    return {
        "version": version,
        "run_date": run_date,
        "headline": headline,
        "stat_chips": chips,
        "hero_scatter": hero_scatter,
        "config_items": _config_items(results_directory),
    }


# ---------------------------------------------------------------------------
# 2. Data section
# ---------------------------------------------------------------------------

def build_data_context(
    summary_df: Optional[pd.DataFrame],
    scatter_matrix_path: str,
    results_directory: str = "",
) -> dict:
    """Build context for the Data & Methods appendix."""
    summary_table_html = ""
    if summary_df is not None:
        summary_table_html = summary_df.to_html(classes="data-table", border=0)

    scatter_html = ""
    if os.path.exists(scatter_matrix_path):
        scatter_html = embed_image_base64(scatter_matrix_path)

    target_dist_html = ""
    if results_directory:
        target_path = os.path.join(
            results_directory, "data_insights", "target_distribution.png"
        )
        if os.path.exists(target_path):
            target_dist_html = embed_image_base64(target_path)

    return {
        "summary_table": summary_table_html,
        "scatter_matrix": scatter_html,
        "target_distribution": target_dist_html,
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
    """Build context for the Model Comparison + Predictions sections."""
    # APA results table (mean[sd] + permutation p)
    results_table_html = ""
    if df_p_values is not None:
        try:
            df_combined = combine_results_with_pvalues(df_mean.copy(), df_p_values.copy())
            styled = format_results_table(df_combined)
            results_table_html = styler_to_html(styled)
        except Exception:
            pass

    # One faceted small-multiples figure (replaces 4 stacked boxplots)
    if task_type == "classification":
        metrics = ["accuracy", "balanced_accuracy", "f1_score", "roc_auc"]
    else:
        metrics = ["pearson_score", "explained_variance_score",
                   "mean_squared_error", "mean_absolute_error"]
    metrics = [m for m in metrics if m in df_full.columns]

    perf_grid_html = ""
    try:
        perf_grid_html = _svg_to_html(performance_grid(df_full, metrics, plots_dir))
    except Exception:
        pass

    # Predicted-vs-observed scatters
    scatter_pred_html = ""
    scatter_cov_html = ""
    try:
        if task_type == "classification":
            fig_path = classification_scatter_plot(df_predictions, plots_dir, y_name)
        else:
            fig_path = scatter_plot(df_predictions, plots_dir, y_name)
        scatter_pred_html = _svg_to_html(fig_path)
    except Exception:
        pass

    try:
        fig_path = scatter_plot_covariates_model(df_predictions, plots_dir, y_name)
        scatter_cov_html = _svg_to_html(fig_path)
    except Exception:
        pass

    return {
        "results_table": results_table_html,
        "performance_grid": perf_grid_html,
        "scatter_predictions": scatter_pred_html,
        "scatter_covariates": scatter_cov_html,
    }


# ---------------------------------------------------------------------------
# Hyperparameters (appendix)
# ---------------------------------------------------------------------------

def build_hyperparameters_context(df_full: pd.DataFrame) -> dict:
    """Restore the hyperparameters tables (per-fold + summary) for the appendix."""
    if "params" not in df_full.columns:
        return {"has_hyperparameters": False, "hyper_table": "", "hyper_summary": ""}

    hyper_table_html = ""
    hyper_summary_html = ""
    try:
        hyper_df = create_hyperparameter_table(df_full)
        if not hyper_df.empty:
            hyper_table_html = hyper_df.to_html(classes="data-table", border=0, index=False)
    except Exception:
        pass

    try:
        if isinstance(df_full["params"].iloc[0], dict):
            summary = create_hyperparameter_summary(df_full)
            if not summary.empty:
                hyper_summary_html = summary.to_html(classes="data-table", border=0)
    except Exception:
        pass

    return {
        "has_hyperparameters": bool(hyper_table_html or hyper_summary_html),
        "hyper_table": hyper_table_html,
        "hyper_summary": hyper_summary_html,
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
        fig_path = scatter_plot_network_strengths(df_network_strengths, plots_dir, y_name)
        ns_scatter_html = _svg_to_html(fig_path)
    except Exception:
        pass

    try:
        fig_path = histograms_network_strengths(df_network_strengths, plots_dir, y_name)
        ns_hist_html = _svg_to_html(fig_path)
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
    plots_dir: str,
    atlas_labels: Optional[pd.DataFrame],
    edge_stability: Optional[np.ndarray] = None,
    edge_significance: Optional[np.ndarray] = None,
) -> dict:
    """
    Build context for the Brain & Edges section.

    Connectivity matrix and node-degree work without an atlas. The
    network-summary matrix and chord diagram need a ``network`` column. The
    glass-brain render additionally needs node coordinates (x, y, z).
    """
    from cccpm.reporting.plots import brain_figures as bf
    from cccpm.reporting.plots.connectome_utils import (
        signed_stability_matrix,
        significant_edge_matrices,
    )
    from cccpm.reporting.plots.figure_style import NEG, POS

    ctx = {
        "has_atlas": atlas_labels is not None,
        "has_network_labels": atlas_labels is not None and "network" in atlas_labels.columns,
        "brain_positive": "",
        "brain_negative": "",
        "conn_matrix": "",
        "network_summary": "",
        "chord": "",
        "node_degree": "",
    }

    # ── Matrix-based figures (built from the stability matrices) ──
    if edge_stability is not None:
        try:
            signed = signed_stability_matrix(edge_stability)
        except Exception:
            signed = None

        if signed is not None:
            try:
                ctx["conn_matrix"] = _svg_to_html(
                    bf.connectivity_matrix(signed, plots_dir, atlas=atlas_labels)
                )
            except Exception:
                pass
            try:
                ctx["node_degree"] = _svg_to_html(
                    bf.node_degree_plot(signed, plots_dir, atlas=atlas_labels)
                )
            except Exception:
                pass
            if ctx["has_network_labels"]:
                try:
                    p = bf.network_summary_matrix(signed, atlas_labels, plots_dir)
                    if p:
                        ctx["network_summary"] = _svg_to_html(p)
                except Exception:
                    pass
                try:
                    p = bf.chord_diagram(signed, atlas_labels, plots_dir)
                    if p:
                        ctx["chord"] = _svg_to_html(p)
                except Exception:
                    pass

    # ── Glass-brain renders (netplotbrain; need atlas with x/y/z) ──
    # Built directly from the significant-edge matrices, not from the
    # `sig_stability_*` .npy files (which the pipeline no longer writes).
    if atlas_labels is not None and edge_stability is not None:
        try:
            pos_mat, neg_mat = significant_edge_matrices(edge_stability, edge_significance)
        except Exception:
            pos_mat = neg_mat = None
        for mat, color, fname, key in [
            (pos_mat, POS, "netplotbrain_positive", "brain_positive"),
            (neg_mat, NEG, "netplotbrain_negative", "brain_negative"),
        ]:
            if mat is None:
                continue
            try:
                path = bf.glass_brain(mat, atlas_labels, plots_dir,
                                      edge_color=color, name=fname)
                if path:
                    ctx[key] = _png_to_html(path)
            except Exception:
                pass

    return ctx


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
