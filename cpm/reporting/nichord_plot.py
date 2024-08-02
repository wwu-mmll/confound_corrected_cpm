from nichord.coord_labeler import get_idx_to_label

from nichord.convert import convert_matrix
from nichord.chord import plot_chord
from nichord.combine import plot_and_combine
from cpm.simulate_data import simulate_regression_data
from nilearn.datasets import fetch_atlas_schaefer_2018
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


def matrix_to_vector_3d(matrix_3d):
    """
    Convert a 3D matrix to a vector containing only the elements of the strictly upper triangular parts
    of each 2D matrix (excluding the diagonal elements).

    Parameters:
    matrix_3d (np.ndarray): Input 3D matrix of shape (n_samples, n, n), where each 2D matrix is square.

    Returns:
    np.ndarray: A 2D array where each row is a vector of the strictly upper triangular part of the corresponding 2D matrix.
    """
    n_samples, n, _ = matrix_3d.shape
    # Create an index matrix for the strictly upper triangular indices
    row_indices, col_indices = np.triu_indices(n, k=1)  # k=1 excludes the diagonal
    upper_tri_indices = np.ravel_multi_index((row_indices, col_indices), (n, n))

    # Flatten the 3D matrix along the last two dimensions
    flat_matrix = matrix_3d.reshape(-1, n * n)

    # Extract the strictly upper triangular elements for each 2D matrix
    upper_tri_vectors = flat_matrix[:, upper_tri_indices]

    return upper_tri_vectors


def vector_to_matrix_3d(vector_2d, shape):
    """
    Convert a vector containing strictly upper triangular parts back to a 3D matrix.

    Parameters:
    vector_2d (np.ndarray): A 2D array where each row is a vector of the strictly upper triangular part of a 2D matrix.
    shape (tuple): The shape of the original 3D matrix, (n_samples, n, n).

    Returns:
    np.ndarray: The reconstructed 3D matrix of shape (n_samples, n, n).
    """
    n_samples, n, _ = shape
    # Create an empty 3D matrix to fill
    matrix_3d = np.zeros((n_samples, n, n))

    # Create an index matrix for the strictly upper triangular indices
    row_indices, col_indices = np.tril_indices(n, k=-1)  # k=1 excludes the diagonal
    upper_tri_indices = np.ravel_multi_index((row_indices, col_indices), (n, n))

    # Flatten the 3D matrix along the last two dimensions
    flat_matrix = matrix_3d.reshape(n_samples, -1)

    # Place the strictly upper triangular elements into the corresponding positions
    np.put_along_axis(flat_matrix, upper_tri_indices[None, :], vector_2d, axis=1)

    return matrix_3d


X, y, covariates = simulate_regression_data(n_features=4950, n_informative_features=100)
atlas = fetch_atlas_schaefer_2018(n_rois=100)
networks = [str(roi).split("_")[-2] for roi in list(atlas["labels"])]
regions = [str(roi).split("b'7Networks_")[1] for roi in list(atlas["labels"])]
mat = vector_to_matrix_3d(X, (X.shape[0], 100, 100))
edges, edge_weights = convert_matrix(mat[0])

idx_to_label = {}
for idx, network in enumerate(networks):
    idx_to_label[idx] = network


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


# Example usage
unique_networks = set(idx_to_label.keys())
n_colors = len(unique_networks)
colors = get_colors_from_colormap(n_colors, 'tab10')
network_colors = {}
for idx, network in enumerate(unique_networks):
    network_colors[network] = colors[idx]


# If the filepath is left None, the chord diagram can be opened in a matplotlib with plt.show()
fp_chord = 'ex0_chord.png'
plot_chord(idx_to_label, edges, edge_weights=edge_weights, fp_chord=fp_chord,
           linewidths=5, alphas=0.9, do_ROI_circles=True, label_fontsize=70,
           # July 2023 update allows changing label fontsize
           do_ROI_circles_specific=True, ROI_circle_radius=0.02)

"""
dir_out = '.'
fn = 'ex1.png'
network_order = ['FPCN', 'DMN', 'DAN', 'Visual', 'SM', 'Limbic',
                 'Uncertain', 'VAN']
chord_kwargs = {'alphas': 0.7, 'linewidths': 10,
                'plot_count': False,
                'do_ROI_circles': False,
                'do_ROI_circles_specific': False,
                'arc_setting': False}
glass_kwargs = {'linewidths': 8,
                'node_size': 20}
plot_and_combine(dir_out, fn, idx_to_label, edges,
                 edge_weights=edge_weights, coords=coords, network_colors=network_colors,
                 glass_kwargs=glass_kwargs,
                 chord_kwargs=chord_kwargs, network_order=network_order, cmap='coolwarm')
"""
