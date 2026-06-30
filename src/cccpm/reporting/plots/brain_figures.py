"""
Neuroscience figures for the CCCPM HTML report.

Each function takes a node-level connectome matrix (and, where needed, an atlas
DataFrame) and returns the path to a saved SVG figure. The functions are kept
self-contained — input = numpy matrix / atlas, output = matplotlib Figure — so
they can later be moved into the shared ``wwu-mmll/brainplots`` toolbox.

Atlas convention (matches brainchord): one row per node, columns include
``region``, ``network``, ``hemisphere``, ``x``, ``y``, ``z``. Only ``network``
is required for the network-summary and chord figures.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cccpm.reporting.plots.connectome_utils import (
    aggregate_matrix_by_group,
    get_network_color,
)
from cccpm.reporting.plots.figure_style import (
    DIVERGING_CMAP,
    NEG,
    POS,
    SQUARE,
    apply_nature_style,
    save_figure,
)


# ── 1. Connectivity matrix ─────────────────────────────────────────────────────

def connectivity_matrix(
    matrix: np.ndarray,
    results_folder: str,
    atlas: pd.DataFrame | None = None,
    *,
    name: str = "connectivity_matrix",
) -> str:
    """
    Node×node heatmap of (signed) edge stability.

    Red = positive-network edges, blue = negative-network edges. If *atlas* has
    a ``network`` column, nodes are ordered by network and dividers are drawn,
    so within/between-network structure is visible (the classic CPM matrix view).
    """
    apply_nature_style()
    mat = np.asarray(matrix, dtype=float)
    n = mat.shape[0]

    order = np.arange(n)
    boundaries: list[int] = []
    tick_pos: list[float] = []
    tick_lab: list[str] = []
    if atlas is not None and "network" in atlas.columns and len(atlas) == n:
        nets = atlas["network"].to_numpy()
        order = np.argsort(nets, kind="stable")
        nets_sorted = nets[order]
        mat = mat[np.ix_(order, order)]
        # network block boundaries + centered labels
        start = 0
        for i in range(1, n + 1):
            if i == n or nets_sorted[i] != nets_sorted[start]:
                boundaries.append(i)
                tick_pos.append((start + i) / 2)
                tick_lab.append(str(nets_sorted[start]))
                start = i

    vmax = np.abs(mat).max() or 1.0
    fig, ax = plt.subplots(figsize=SQUARE)
    im = ax.imshow(mat, cmap=DIVERGING_CMAP, vmin=-vmax, vmax=vmax, interpolation="nearest")

    if boundaries:
        for b in boundaries[:-1]:
            ax.axhline(b - 0.5, color="#444", linewidth=0.4)
            ax.axvline(b - 0.5, color="#444", linewidth=0.4)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab, rotation=90, fontsize=5)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_lab, fontsize=5)
    else:
        ax.set_xlabel("node")
        ax.set_ylabel("node")
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label("stability (signed)", fontsize=6)
    fig.tight_layout(pad=0.3)
    return save_figure(fig, os.path.join(results_folder, name))


# ── 2. Network-summary matrix ──────────────────────────────────────────────────

def network_summary_matrix(
    matrix: np.ndarray,
    atlas: pd.DataFrame,
    results_folder: str,
    *,
    groupby: str = "network",
    name: str = "network_summary_matrix",
) -> str | None:
    """
    Region×region matrix: edges aggregated by canonical network.

    Returns ``None`` if the atlas lacks the *groupby* column (caller shows a
    graceful "needs network labels" message).
    """
    if atlas is None or groupby not in atlas.columns or len(atlas) != matrix.shape[0]:
        return None

    apply_nature_style()
    groups = atlas[groupby].astype(str).tolist()
    region_df = aggregate_matrix_by_group(np.asarray(matrix, dtype=float), groups, signed=True)

    vmax = np.abs(region_df.values).max() or 1.0
    fig, ax = plt.subplots(figsize=SQUARE)
    im = ax.imshow(region_df.values, cmap=DIVERGING_CMAP, vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(region_df.columns)))
    ax.set_xticklabels(region_df.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(region_df.index)))
    ax.set_yticklabels(region_df.index, fontsize=6)

    # annotate cells
    for i in range(region_df.shape[0]):
        for j in range(region_df.shape[1]):
            v = region_df.values[i, j]
            if abs(v) > 1e-9:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5, color="#222")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label("mean stability (signed)", fontsize=6)
    fig.tight_layout(pad=0.3)
    return save_figure(fig, os.path.join(results_folder, name))


# ── 3. Chord diagram ───────────────────────────────────────────────────────────

def chord_diagram(
    matrix: np.ndarray,
    atlas: pd.DataFrame,
    results_folder: str,
    *,
    groupby: str = "network",
    name: str = "chord_diagram",
    link_alpha: float = 0.55,
) -> str | None:
    """
    Network-level chord diagram (pycirclize). Sectors = canonical networks,
    link colour = sign (red positive, blue negative), arc width = |stability|.

    Returns ``None`` if the atlas lacks the *groupby* column or pycirclize is
    unavailable.
    """
    if atlas is None or groupby not in atlas.columns or len(atlas) != matrix.shape[0]:
        return None
    try:
        from pycirclize import Circos
    except Exception:
        return None

    groups = atlas[groupby].astype(str).tolist()
    region_df = aggregate_matrix_by_group(np.asarray(matrix, dtype=float), groups, signed=True)

    # drop empty sectors
    active = region_df.abs().sum(axis=1) > 1e-9
    region_df = region_df.loc[active, active]
    if region_df.empty or region_df.shape[0] < 2:
        return None

    abs_df = region_df.abs()
    sector_colors = {r: get_network_color(r) for r in region_df.index}

    def _link_kws_handler(r1: str, r2: str) -> dict:
        try:
            val = region_df.loc[r1, r2]
        except KeyError:
            val = 0.0
        color = POS if val >= 0 else NEG
        return dict(fc=color, alpha=link_alpha, zorder=1.0)

    circos = Circos.chord_diagram(
        abs_df,
        space=min(3.0, 360 / max(len(region_df), 1) * 0.8),
        cmap=sector_colors,
        label_kws=dict(size=7, r=110, orientation="vertical"),
        link_kws=dict(alpha=link_alpha, zorder=1.0),
        link_kws_handler=_link_kws_handler,
    )
    fig = circos.plotfig(figsize=SQUARE, dpi=150)
    fig.patch.set_facecolor("white")
    return save_figure(fig, os.path.join(results_folder, name), formats=("svg",))


# ── 4. Node-degree / hub plot ──────────────────────────────────────────────────

def node_degree_plot(
    matrix: np.ndarray,
    results_folder: str,
    atlas: pd.DataFrame | None = None,
    *,
    top_k: int = 15,
    name: str = "node_degree",
) -> str:
    """
    Horizontal bar of the most-connected nodes (hubs): number of stable edges
    per node, split into positive (red) and negative (blue) contributions.
    """
    apply_nature_style()
    mat = np.asarray(matrix, dtype=float)
    n = mat.shape[0]

    pos_deg = (mat > 1e-9).sum(axis=1)
    neg_deg = (mat < -1e-9).sum(axis=1)
    total = pos_deg + neg_deg

    if atlas is not None and "region" in atlas.columns and len(atlas) == n:
        labels = atlas["region"].astype(str).to_numpy()
    else:
        labels = np.array([f"node {i}" for i in range(n)])

    order = np.argsort(total)[::-1][:top_k]
    order = order[total[order] > 0]
    if len(order) == 0:
        order = np.argsort(total)[::-1][:top_k]

    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(SQUARE[0], SQUARE[1]))
    ax.barh(y, pos_deg[order], color=POS, label="positive")
    ax.barh(y, neg_deg[order], left=pos_deg[order], color=NEG, label="negative")
    ax.set_yticks(y)
    ax.set_yticklabels(labels[order], fontsize=5)
    ax.set_xlabel("number of stable edges")
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    sns_despine(ax)
    fig.tight_layout(pad=0.3)
    return save_figure(fig, os.path.join(results_folder, name))


def sns_despine(ax) -> None:
    """Local despine to avoid importing seaborn just for this."""
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
