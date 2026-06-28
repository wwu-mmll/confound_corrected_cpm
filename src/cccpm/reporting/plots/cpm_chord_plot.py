import os
import numpy as np
from typing import Union, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import netplotbrain


def vector_to_upper_triangular_matrix(vector):
    """
    Convert a vector containing strictly upper triangular elements back
    to a 2D square matrix.

    Parameters:
    vector (np.ndarray): A vector containing the strictly upper triangular elements.

    Returns:
    np.ndarray: The reconstructed 2D square matrix.
    """
    # Calculate the size of the matrix from the vector length
    size = int((np.sqrt(8 * vector.size + 1) - 1) / 2) + 1
    if size * (size - 1) // 2 != vector.size:
        raise ValueError("Vector size does not match the number of elements for a valid square matrix.")

    matrix = np.zeros((size, size))
    # Get the indices of the strictly upper triangular part
    row_indices, col_indices = np.triu_indices(size, k=1)
    # Place the elements into the matrix
    matrix[row_indices, col_indices] = vector
    matrix[col_indices, row_indices] = vector
    return matrix


def get_colors_from_colormap(n_colors, colormap_name='tab10'):
    """
    Get a set of distinct colors from a specified colormap.

    Parameters:
    n_colors (int): Number of distinct colors needed.
    colormap_name (str): Name of the colormap to use (e.g., 'tab10').

    Returns:
    list: A list of color strings.
    """
    cmap = plt.get_cmap(colormap_name)
    colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]
    return colors


def convert_matrix(adj: Union[list, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts your adjacency (connectivity) matrix into a list of edges (i, j)
        and their weights
    :param adj: the matrix
    """
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
    """
    Given a square matrix (graph), this function returns:
    1. A NumPy array with two columns containing the ids of the two nodes connected by an edge.
    2. A NumPy array containing the weights of the edges.

    Args:
    matrix (2D numpy array): A square matrix representing a graph.

    Returns:
    edges (2D numpy array): Array of edges.
    weights (1D numpy array): Array of weights corresponding to the edges.
    """
    if isinstance(matrix, np.ndarray) and matrix.shape[0] == matrix.shape[1]:
        n = matrix.shape[0]
        edges = []
        weights = []

        for i in range(1, n):
            for j in range(i):
                if keep_only_non_zero_edges:
                    if matrix[i, j] != 0:  # Only include non-zero edges
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


if __name__ == "__main__":
    results_directory = '/spm-data/vault-data3/mmll/projects/cpm_python/results/hcp_SSAGA_TB_Yrs_Smoked_spearman_partial_p=0.001/'
    selected_metric = "sig_stability_negative_edges"
    #plot_cpm_chord_plot(results_directory, selected_metric)
    plot_netplotbrain(results_directory, selected_metric)
