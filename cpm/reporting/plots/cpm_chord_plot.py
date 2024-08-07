import os
import numpy as np
from plots.chord_v2 import plot_chord
from typing import Union, Tuple
import matplotlib.pyplot as plt


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

def plot_cpm_chord_plot(results_folder, selected_metric):
    edges = np.load(os.path.join(results_folder, f"{selected_metric}.npy"))
    edges += 0.01
    if (selected_metric == "sig_stability_positive_edges") or (selected_metric == "sig_stability_negative_edges"):
        threshold = 0.05
        corr_transformed = np.where(edges > threshold, 0, edges)
        corr_transformed = np.where(edges <= threshold, 1, corr_transformed)
        edges = corr_transformed

    edges_plot, edge_weights = convert_matrix(edges)
    n_regions = edges.shape[1]
    networks = ["Network 1"] * n_regions
    regions = [f"Region {r}" for r in range(n_regions)]

    colors = get_colors_from_colormap(len(set(networks)) + 1, 'tab10')
    network_colors_dict = {}
    for i, network in enumerate(set(networks)):
        network_colors_dict[network] = colors[i]
    idx_to_label = {}
    network_colors = {}
    for idx, region in enumerate(regions):
        idx_to_label[idx] = region
        network_colors[region] = network_colors_dict[networks[idx]]

    filename = os.path.join(results_folder, "plots", "edge_chord.png")
    plot_chord(idx_to_label, edges_plot, edge_weights=edge_weights, fp_chord=filename,
               edge_threshold=0, arc_setting=False, network_colors=network_colors,
               linewidths=3, alphas=0.6, label_fontsize=12)
    return filename
