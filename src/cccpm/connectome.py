"""Converts between square node-by-node matrices and the vectorised upper triangle CCCPM works on internally.

The only place the triangle indexing convention is defined; kept free of any
plotting/validation dependency since both the numeric core and reporting build on it.
"""
import numpy as np
import torch


def matrix_to_vector_3d(matrix_3d):
    """Converts (n_samples, n, n) connectivity matrices to (n_samples, n*(n-1)/2) strict-upper-triangle vectors."""
    n_samples, n, _ = matrix_3d.shape
    row_idx, col_idx = np.triu_indices(n, k=1)
    flat = matrix_3d.reshape(n_samples, n * n)
    upper = flat[:, np.ravel_multi_index((row_idx, col_idx), (n, n))]
    return upper


def vector_to_matrix_tensor_version(tensor, dim):
    """Expands `dim` (vectorised upper-triangular edges) into two symmetric (Nodes, Nodes) dims at that location."""
    ndim = tensor.ndim
    dim = dim % ndim  # normalise negative indices (e.g. -1)

    # F = N(N-1)/2  =>  N = (1 + sqrt(1 + 8F)) / 2
    n_features = tensor.shape[dim]
    n_nodes = int((1 + (1 + 8 * n_features) ** 0.5) / 2)

    # Move the target dimension to the end for easier broadcasting.
    temp_tensor = tensor.movedim(dim, -1)

    out_shape = temp_tensor.shape[:-1] + (n_nodes, n_nodes)
    out = torch.zeros(out_shape, device=tensor.device, dtype=tensor.dtype)

    rows, cols = torch.triu_indices(n_nodes, n_nodes, offset=1, device=tensor.device)

    # '...' covers any number of leading dimensions.
    out[..., rows, cols] = temp_tensor
    out[..., cols, rows] = temp_tensor  # mirror to make symmetric

    # Move the two matrix dims from the end back to (dim, dim + 1).
    return out.movedim((-2, -1), (dim, dim + 1))
