import matplotlib.pyplot as plt
import numpy as np


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
