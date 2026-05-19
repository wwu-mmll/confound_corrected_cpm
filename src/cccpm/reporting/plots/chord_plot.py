import os
import numpy as np
import pandas as pd
from pycirclize import Circos
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ── Region colour palette (extendable) ──────────────────────────────────────
REGION_COLORS = {
    "Prefrontal":   "#d32f2f",
    "Frontal":      "#e53935",
    "Motor":        "#c62828",
    "Premotor":     "#b71c1c",
    "Insula":       "#f9a825",
    "Parietal":     "#388e3c",
    "Temporal":     "#43a047",
    "Occipital":    "#00897b",
    "Limbic":       "#0288d1",
    "Cingulate":    "#0277bd",
    "Cerebellum":   "#1565c0",
    "Subcortical":  "#7b1fa2",
    "Subcortex":    "#7b1fa2",
    "Thalamus":     "#6a1b9a",
    "Brainstem":    "#c2185b",
    "Default":      "#78909c",
}

# Fallback color cycle for unknown regions
_FALLBACK_COLORS = list(plt.cm.tab20.colors)


def _get_region_color(region_name: str, fallback_map: dict) -> str:
    """Return a hex color for a region name, with fuzzy matching."""
    for key, color in REGION_COLORS.items():
        if key.lower() in region_name.lower():
            return color
    # assign a deterministic fallback color
    if region_name not in fallback_map:
        idx = len(fallback_map) % len(_FALLBACK_COLORS)
        fallback_map[region_name] = mcolors.to_hex(_FALLBACK_COLORS[idx])
    return fallback_map[region_name]


def _upper_tri_to_matrix(n_regions: int, edge_values: np.ndarray) -> np.ndarray:
    """
    Convert an upper-triangle edge vector back to a symmetric (n x n) matrix.
    Edge ordering matches np.triu_indices(n_regions, k=1).
    """
    mat = np.zeros((n_regions, n_regions))
    rows, cols = np.triu_indices(n_regions, k=1)
    mat[rows, cols] = edge_values
    mat[cols, rows] = edge_values
    return mat


def build_connectivity_matrix(
    X: np.ndarray,
    atlas_labels: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate the raw edge matrix X (n_subjects × n_edges) into a
    (n_regions × n_regions) mean-absolute-correlation matrix grouped by region.

    Parameters
    ----------
    X : np.ndarray, shape (n_subjects, n_edges)
        Raw connectivity features. Edges are assumed to be the upper triangle
        of a (n_nodes × n_nodes) matrix, ordered by np.triu_indices(n_nodes, k=1).
    atlas_labels : pd.DataFrame
        Must contain a 'region' column (and optionally 'x','y','z').
        Each row corresponds to one node/ROI.

    Returns
    -------
    pd.DataFrame  (n_regions × n_regions) symmetric matrix of mean |connectivity|.
    """
    n_nodes = len(atlas_labels)
    n_edges_expected = n_nodes * (n_nodes - 1) // 2

    if X.shape[1] != n_edges_expected:
        raise ValueError(
            f"X has {X.shape[1]} edges but atlas has {n_nodes} nodes "
            f"→ expected {n_edges_expected} edges. "
            "Make sure atlas_labels rows match the number of nodes."
        )

    regions = atlas_labels["network"].values
    unique_regions = list(dict.fromkeys(regions))   # preserves order, unique
    n_regions = len(unique_regions)
    region_idx = {r: i for i, r in enumerate(unique_regions)}

    # Accumulate mean |edge value| between region pairs
    sum_mat = np.zeros((n_regions, n_regions))
    count_mat = np.zeros((n_regions, n_regions), dtype=int)

    node_rows, node_cols = np.triu_indices(n_nodes, k=1)
    # Mean over subjects for each edge
    mean_edge_values = np.mean(np.abs(X), axis=0)

    for edge_i, (ni, nj) in enumerate(zip(node_rows, node_cols)):
        ri = region_idx[regions[ni]]
        rj = region_idx[regions[nj]]
        if ri == rj:
            continue                # skip within-region edges for chord diagram
        val = mean_edge_values[edge_i]
        sum_mat[ri, rj]   += val
        sum_mat[rj, ri]   += val
        count_mat[ri, rj] += 1
        count_mat[rj, ri] += 1

    # Avoid division by zero
    with np.errstate(invalid="ignore"):
        mean_mat = np.where(count_mat > 0, sum_mat / count_mat, 0.0)

    return pd.DataFrame(mean_mat, index=unique_regions, columns=unique_regions)


def plot_chord_diagram(
    X: np.ndarray,
    atlas_labels: pd.DataFrame,
    output_path: str,
    title: str = "Brain Connectivity Chord Diagram",
    figsize: tuple = (4, 4),
    dpi: int = 150,
    min_link_value: float = 0.7,
    link_alpha: float = 0.45,
    space: float = 3.0,
    label_r: float = 118,
    label_size: int = 12,
) -> str:
    """
    Generate and save a chord diagram from raw CPM edge data.

    Parameters
    ----------
    X               : np.ndarray (n_subjects × n_edges)  — raw connectivity matrix
    atlas_labels    : pd.DataFrame with at least a 'region' column
    output_path     : full path for the saved PNG
    title           : figure title
    figsize         : matplotlib figure size
    dpi             : resolution
    min_link_value  : links below this threshold are zeroed out (reduces clutter)
    link_alpha      : transparency of chord links
    space           : degrees of space between sectors
    label_r         : radial position of sector labels
    label_size      : font size of sector labels

    Returns
    -------
    str  — path of the saved file
    """

    # ── 1. Build region-level connectivity matrix ────────────────────────────
    conn_df = build_connectivity_matrix(X, atlas_labels)

    n_regions = len(conn_df)
    space = min(3.0, 360 / n_regions * 0.8)

    # Optional: suppress very weak links
    if min_link_value > 0:
        conn_df[conn_df < min_link_value] = 0.0

    # Drop regions that have zero connectivity with everything
    active = conn_df.sum(axis=1) > 0
    conn_df = conn_df.loc[active, active]

    if conn_df.empty:
        raise ValueError("No non-zero connections found. Lower min_link_value or check your data.")

    # ── 2. Assign colours ────────────────────────────────────────────────────
    fallback_map: dict = {}
    region_colors = {
        r: _get_region_color(r, fallback_map)
        for r in conn_df.index
    }

    # ── 3. Draw chord diagram ────────────────────────────────────────────────
    circos = Circos.chord_diagram(
        conn_df,
        space=space,
        cmap=region_colors,
        label_kws=dict(size=label_size, r=label_r, orientation="vertical"),
        link_kws=dict(alpha=link_alpha, zorder=1.0),
    )

    fig = circos.plotfig(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=8, y=1, fontweight="normal", color="#333333")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path