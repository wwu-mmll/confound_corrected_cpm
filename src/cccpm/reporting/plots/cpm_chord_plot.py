"""Legacy netplotbrain chord/glass-brain helpers; superseded by ``brain_figures.glass_brain``, kept for external callers."""

import os
import numpy as np
from typing import Union, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import netplotbrain


def get_colors_from_colormap(n_colors, colormap_name='tab10'):
    """Return n_colors distinct colors sampled from a matplotlib colormap."""
    cmap = plt.get_cmap(colormap_name)
    colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]
    return colors


def convert_matrix(adj: Union[list, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert an adjacency matrix into non-zero ``(i, j)`` node pairs and their weights."""
    if isinstance(adj, list):
        adj = np.array(adj)
    idxs = np.triu_indices(adj.shape[0], k=1)
    weights = adj[idxs]
    idxs = np.array(idxs).T
    smol = 1e-6
    idxs = idxs[(weights > smol) | (weights < -smol)]
    weights = weights[(weights > smol) | (weights < -smol)]
    return idxs, weights


def extract_edges(matrix, keep_only_non_zero_edges: bool = False):
    """Convert a square adjacency matrix into an ``[N, 2]`` array of node-id pairs and their weights."""
    if isinstance(matrix, np.ndarray) and matrix.shape[0] == matrix.shape[1]:
        n = matrix.shape[0]
        edges = []
        weights = []

        for i in range(1, n):
            for j in range(i):
                if keep_only_non_zero_edges:
                    if matrix[i, j] != 0:
                        edges.append([i, j])
                        weights.append(matrix[i, j])
                else:
                    edges.append([i, j])
                    weights.append(matrix[i, j])
        edges = np.array(edges, dtype=int)
        weights = np.array(weights)
        return edges, weights
    else:
        raise ValueError("Input must be a square matrix (2D NumPy array).")


def plot_netplotbrain(results_folder, selected_metric, atlas_labels):
    """Render a glass-brain edge plot from a saved ``{selected_metric}.npy`` file (e.g. ``"stability_positive_edges"``)."""
    edges = np.load(os.path.join(results_folder, f"{selected_metric}.npy"))
    if (selected_metric == "sig_stability_positive_edges") or (selected_metric == "sig_stability_negative_edges"):
        threshold = 0.01
        corr_transformed = np.where(np.abs(edges) > threshold, 0, edges)
        corr_transformed = np.where(np.abs(edges) <= threshold, 1, corr_transformed)
        edges = corr_transformed
    elif (selected_metric == "stability_positive_edges") or (selected_metric == "stability_negative_edges"):
        threshold = 1
        corr_transformed = np.where(np.abs(edges) < threshold, 0, edges)
        corr_transformed = np.where(np.abs(edges) >= threshold, 1, corr_transformed)
        edges = corr_transformed

    if 'positive' in selected_metric:
        edge_color = "#b22222"
    else:
        edge_color = "#317199"

    edges_plot, edge_weights = extract_edges(edges, keep_only_non_zero_edges=True)

    if atlas_labels is not None and edges_plot.any():
        aparc = atlas_labels

        edges_netplot = pd.DataFrame({'i': edges_plot[:, 0], 'j': edges_plot[:, 1],
                                      'weights': edge_weights})

        fig, ax = netplotbrain.plot(template='MNI152NLin2009cAsym',
                                    template_style='glass',
                                    nodes=aparc,
                                    edges=edges_netplot,
                                    view=['LSR'],
                                    highlight_edges=True,
                                    highlight_nodes=None,
                                    node_type='circles',
                                    edge_color=edge_color,
                                    node_color='#332f2c'
                                    )
    else:
        fig = plt.figure()
        edges_netplot = None
    fig.savefig(os.path.join(results_folder, "plots", f"netplotbrain_{selected_metric}.png"))
    return os.path.join(results_folder, "plots", f"netplotbrain_{selected_metric}.png"), edges_netplot
