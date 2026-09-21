"""
Connectome geometry: converting between square node-by-node matrices and the
vectorised upper triangle CCCPM works on internally.

Kept free of any plotting or validation dependency -- these two functions are
the only place the triangle indexing convention is defined, and both the
numeric core and the reporting layer build on them.
"""
import numpy as np
import torch


def matrix_to_vector_3d(matrix_3d):
    """
    Convert a 3D connectivity matrix to a 2D array of upper-triangular vectors.

    Parameters
    ----------
    matrix_3d: np.ndarray
        Input 3D array of shape (n_samples, n, n), where each 2D matrix is square.

    Returns
    -------
    upper: np.ndarray
        2D array of shape (n_samples, n*(n - 1)/2) containing strictly upper-triangular elements of each matrix.
    """
    n_samples, n, _ = matrix_3d.shape
    row_idx, col_idx = np.triu_indices(n, k=1)
    flat = matrix_3d.reshape(n_samples, n * n)
    upper = flat[:, np.ravel_multi_index((row_idx, col_idx), (n, n))]
    return upper


def vector_to_matrix_tensor_version(tensor, dim):
    """
    Expands a specific dimension containing vectorised upper-triangular edges
    into a symmetric square matrix at that same location.

    Args:
        tensor: Arbitrary shape, e.g. [Networks, Folds, Features, Perms]
        dim: The index of the dimension to expand (e.g., 2 for Features)

    Returns:
        Tensor with 'dim' replaced by two dimensions (Nodes, Nodes).
        Example: [Net, Fold, Feat, Perm] -> [Net, Fold, Nodes, Nodes, Perm]
    """
    # 1. Normalize dim to positive index (handles -1, etc.)
    ndim = tensor.ndim
    dim = dim % ndim

    # 2. Calculate Number of Nodes
    # Formula: F = N(N-1)/2  =>  N = (1 + sqrt(1 + 8F)) / 2
    n_features = tensor.shape[dim]
    n_nodes = int((1 + (1 + 8 * n_features) ** 0.5) / 2)

    # 3. Move the target dimension to the end for easier broadcasting
    # shape: [..., Features]
    temp_tensor = tensor.movedim(dim, -1)

    # 4. Create the Output Placeholder
    # shape: [..., Nodes, Nodes]
    out_shape = temp_tensor.shape[:-1] + (n_nodes, n_nodes)
    out = torch.zeros(out_shape, device=tensor.device, dtype=tensor.dtype)

    # 5. Get Upper Triangle Indices
    rows, cols = torch.triu_indices(n_nodes, n_nodes, offset=1, device=tensor.device)

    # 6. Assign Values (Vectorized)
    # The '...' ellipses handle any number of preceding dimensions automatically
    out[..., rows, cols] = temp_tensor

    # 7. Make Symmetric
    out[..., cols, rows] = temp_tensor

    # 8. Move the Matrix dimensions back to the original location
    # We moved 'dim' to the end. Now we have two dims at the end (-2, -1).
    # We want to put them back at 'dim' and 'dim+1'.
    return out.movedim((-2, -1), (dim, dim + 1))
