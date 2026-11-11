"""Neuroscience figures for the CCCPM HTML report.

Atlas convention (matches brainchord): one row per node, columns include
``region``, ``network``, ``hemisphere``, ``x``, ``y``, ``z``.
"""

from __future__ import annotations

import os

import matplotlib as mpl
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


def connectivity_matrix(
    matrix: np.ndarray,
    results_folder: str,
    atlas: pd.DataFrame | None = None,
    *,
    name: str = "connectivity_matrix",
) -> str:
    """Node x node heatmap of (signed) edge stability, ordered by network if available."""
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
        start = 0
        for i in range(1, n + 1):
            if i == n or nets_sorted[i] != nets_sorted[start]:
                boundaries.append(i)
                tick_pos.append((start + i) / 2)
                tick_lab.append(str(nets_sorted[start]))
                start = i

    vmax = np.abs(mat).max() or 1.0
    fig, ax = plt.subplots(figsize=SQUARE)
    # rasterize the cell grid so the SVG stays small for large atlases
    # (200-1000 ROIs) while axes/labels remain crisp vectors.
    im = ax.imshow(mat, cmap=DIVERGING_CMAP, vmin=-vmax, vmax=vmax,
                   interpolation="nearest", rasterized=True)

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


def network_summary_matrix(
    matrix: np.ndarray,
    atlas: pd.DataFrame,
    results_folder: str,
    *,
    groupby: str = "network",
    name: str = "network_summary_matrix",
) -> str | None:
    """Region x region matrix of edges aggregated by canonical network; ``None`` if atlas lacks *groupby*."""
    if atlas is None or groupby not in atlas.columns or len(atlas) != matrix.shape[0]:
        return None

    apply_nature_style()
    groups = atlas[groupby].astype(str).tolist()
    region_df = aggregate_matrix_by_group(np.asarray(matrix, dtype=float), groups, signed=True)

    vmax = np.abs(region_df.values).max() or 1.0
    fig, ax = plt.subplots(figsize=SQUARE)
    im = ax.imshow(region_df.values, cmap=DIVERGING_CMAP, vmin=-vmax, vmax=vmax,
                   rasterized=True)

    ax.set_xticks(range(len(region_df.columns)))
    ax.set_xticklabels(region_df.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(region_df.index)))
    ax.set_yticklabels(region_df.index, fontsize=6)

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


def chord_diagram(
    matrix: np.ndarray,
    atlas: pd.DataFrame,
    results_folder: str,
    *,
    groupby: str = "network",
    name: str = "chord_diagram",
    link_alpha: float = 0.55,
) -> str | None:
    """Network-level chord diagram (pycirclize); ``None`` if atlas lacks *groupby* or pycirclize is missing."""
    if atlas is None or groupby not in atlas.columns or len(atlas) != matrix.shape[0]:
        return None
    try:
        from pycirclize import Circos
    except Exception:
        return None

    groups = atlas[groupby].astype(str).tolist()
    region_df = aggregate_matrix_by_group(np.asarray(matrix, dtype=float), groups, signed=True)

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


def node_degree_plot(
    matrix: np.ndarray,
    results_folder: str,
    atlas: pd.DataFrame | None = None,
    *,
    top_k: int = 15,
    name: str = "node_degree",
) -> str:
    """Horizontal bar of the most-connected nodes (hubs), split by positive/negative edges."""
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
    # Legend above the plot so it never covers bars.
    ax.legend(fontsize=6, frameon=False, ncol=2,
              loc="lower right", bbox_to_anchor=(1.0, 1.0))
    sns_despine(ax)
    fig.tight_layout(pad=0.3)
    return save_figure(fig, os.path.join(results_folder, name))


def sns_despine(ax) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def glass_brain(
    matrix: np.ndarray,
    atlas: pd.DataFrame,
    results_folder: str,
    *,
    edge_color: str = POS,
    name: str = "glass_brain",
    top_n: int | None = 10,
) -> str | None:
    """Anatomical brain-surface rendering of non-zero *matrix* edges via netplotbrain.

    Keeps only the *top_n* edges by |weight| (``None`` keeps all) -- with a few
    hundred to a few thousand stable edges, drawing everything collapses into an
    unreadable hairball with no anatomical information left. ``None`` if
    netplotbrain or the required atlas columns (x, y, z) are unavailable.
    """
    if atlas is None or not {"x", "y", "z"}.issubset(atlas.columns):
        return None
    mat = np.asarray(matrix, dtype=float)
    if mat.shape[0] != len(atlas):
        return None

    rows, cols = np.triu_indices(mat.shape[0], k=1)
    mask = np.abs(mat[rows, cols]) > 1e-9
    i_idx, j_idx, weights = rows[mask], cols[mask], mat[rows, cols][mask]
    if len(i_idx) == 0:
        return None

    if top_n is not None and len(weights) > top_n:
        keep = np.argsort(np.abs(weights))[::-1][:top_n]
        i_idx, j_idx, weights = i_idx[keep], j_idx[keep], weights[keep]

    try:
        import netplotbrain
    except Exception:
        return None

    edges_df = pd.DataFrame({"i": i_idx, "j": j_idx, "weights": weights})
    try:
        # The report's global nature-style rcParams (apply_nature_style, applied
        # before every other figure) make netplotbrain's 3D surface mesh render
        # blank -- some seaborn "white"-theme rcParam leaves the Poly3DCollection
        # invisible. Reset to matplotlib defaults for just this call.
        with mpl.rc_context(rc=mpl.rcParamsDefault):
            fig, _ = netplotbrain.plot(
                template="MNI152NLin2009cAsym",
                template_style="surface",
                nodes=atlas,
                edges=edges_df,
                view=["LSR"],
                highlight_edges=True,
                highlight_nodes=None,
                node_type="circles",
                edge_color=edge_color,
                node_color="#332f2c",
            )
            out = os.path.join(results_folder, f"{name}.png")
            fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    except Exception:
        return None

    plt.close(fig)
    return out
