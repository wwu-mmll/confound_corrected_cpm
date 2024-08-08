import os
import numpy as np
from plots.chord_v2 import plot_chord
from typing import Union, Tuple
import matplotlib.pyplot as plt
from nichord.glassbrain import plot_glassbrain
import pandas as pd


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


def extract_edges(matrix):
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
                edges.append([i, j])
                weights.append(matrix[i, j])

        edges = np.array(edges, dtype=int)
        weights = np.array(weights)
        return edges, weights
    else:
        raise ValueError("Input must be a square matrix (2D NumPy array).")


def plot_cpm_chord_plot(results_folder, selected_metric):
    edges = np.load(os.path.join(results_folder, f"{selected_metric}.npy"))
    if (selected_metric == "sig_stability_positive_edges") or (selected_metric == "sig_stability_negative_edges"):
        threshold = 0.05
        corr_transformed = np.where(np.abs(edges) > threshold, 0, edges)
        corr_transformed = np.where(np.abs(edges) <= threshold, 1, corr_transformed)
        edges = corr_transformed

    try:
        aparc = pd.read_csv(os.path.join(results_folder, "atlas_labels.csv"), names=['x', 'y', 'z', 'regions'])
    except FileNotFoundError:
        aparc = pd.DataFrame({'x': np.ones(edges.shape[1]), 'y': np.ones(edges.shape[1]),
                              'z': np.ones(edges.shape[1]), 'regions': [f"Region {i}" for i in range(edges.shape[1])]})
    edges_plot, edge_weights = extract_edges(edges)
    n_regions = edges.shape[1]
    networks = ["Network 1"] * n_regions
    regions = aparc['regions'].to_list()[:n_regions]
    coords = aparc.iloc[:, :3].to_numpy()
    coords_list = list()
    for i, region in enumerate(regions):
        coords_list.append((coords[i, 0], coords[i, 1], coords[i, 2]))
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
               network_order=regions,
               edge_threshold=0, arc_setting=False, network_colors=network_colors,
               linewidths=3, alphas=0.6, label_fontsize=12)

    fp_glass = os.path.join(results_folder, "plots", "glass_brain.png")
    plot_glassbrain(idx_to_label, edges_plot, edge_weights, fp_glass,
                    coords_list, linewidths=7, node_size=13, network_order=regions,
                    network_colors=network_colors)
    return filename, fp_glass


if __name__ == "__main__":
    results_directory = '/home/nwinter/PycharmProjects/cpm_python/examples/tmp/example_simulated_data2/'
    selected_metric = "sig_stability_positive_edges"
    plot_cpm_chord_plot(results_directory, selected_metric)
