import numpy as np

import torch

from cccpm.connectome import (
    matrix_to_vector_3d,
    vector_to_matrix_tensor_version,
)


# ============================================================
# Reference implementations for the connectome <-> vector conversions.
#
# These plain NumPy converters used to live in the old cccpm.utils (deleted as
# dead production code -- nothing in src/, examples/, or scripts/ called them; the
# only production converter still in use is the tensor-based
# vector_to_matrix_tensor_version, imported above). They are kept here
# verbatim as a second implementation that the production converter must agree
# with -- a consistency check, NOT an independent reference: both descend from
# the same original code, so they would not catch a shared conceptual error.
# The genuinely independent checks live in test_sklearn_equivalence.py.
# ============================================================

def vector_to_matrix_numpy(array, dim):
    """
    Expands a dimension containing vectorized upper-triangular edges
    into a symmetric square matrix.
    """
    # 1. Normalize dim
    ndim = array.ndim
    dim = dim % ndim

    # 2. Calculate Number of Nodes
    n_features = array.shape[dim]
    n_nodes = int((1 + np.sqrt(1 + 8 * n_features)) / 2)

    # 3. Move target dimension to the end
    temp_array = np.moveaxis(array, dim, -1)

    # 4. Create Output Placeholder
    out_shape = temp_array.shape[:-1] + (n_nodes, n_nodes)
    out = np.zeros(out_shape, dtype=array.dtype)

    # 5. Get Upper Triangle Indices (k=1 excludes diagonal)
    rows, cols = np.triu_indices(n_nodes, k=1)

    # 6. Assign Values
    # NumPy advanced indexing allows assigning to the last two dims at once
    out[..., rows, cols] = temp_array

    # 7. Make Symmetric
    out[..., cols, rows] = temp_array

    # 8. Move the Matrix dimensions back to the original location
    return np.moveaxis(out, (-2, -1), (dim, dim + 1))


def matrix_to_vector_numpy(array, dim):
    """
    Collapses two adjacent dimensions of a NumPy array into a
    single dimension of upper-triangular elements.
    """
    ndim = array.ndim
    dim = dim % ndim

    n_nodes = array.shape[dim]
    if n_nodes != array.shape[dim + 1]:
        raise ValueError(f"Dimensions at {dim} and {dim + 1} must be square.")

    # 1. Move target dimensions to the end
    temp_array = np.moveaxis(array, (dim, dim + 1), (-2, -1))

    # 2. Get Upper Triangle Indices
    rows, cols = np.triu_indices(n_nodes, k=1)

    # 3. Extract Values
    # In NumPy, trailing indices work slightly differently with '...'
    # We slice the last two dimensions using the coordinate pairs
    out = temp_array[..., rows, cols]

    # 4. Move the new vector dimension back to the original 'dim'
    return np.moveaxis(out, -1, dim)


def matrix_to_vector_tensor_version(tensor, dim):
    """
    Collapses two adjacent dimensions (representing a symmetric matrix)
    into a single dimension containing the upper-triangular edges.

    Args:
        tensor: Arbitrary shape, e.g. [Net, Fold, Nodes, Nodes, Perm]
        dim: The index of the first of the two matrix dimensions.

    Returns:
        Tensor with [dim, dim+1] replaced by a single dimension of size Features.
        Example: [Net, Fold, Nodes, Nodes, Perm] -> [Net, Fold, Features, Perm]
    """
    # 1. Normalize dim to positive index
    ndim = tensor.ndim
    dim = dim % ndim

    # 2. Identify Matrix Size (N)
    n_nodes = tensor.shape[dim]
    if n_nodes != tensor.shape[dim + 1]:
        raise ValueError(f"Dimensions at {dim} and {dim + 1} must be square.")

    # 3. Move the target dimensions to the end for indexing
    # Current: [..., Nodes, Nodes, ...] -> [..., Nodes, Nodes]
    # We move dim and dim+1 to the last two positions
    temp_tensor = tensor.movedim((dim, dim + 1), (-2, -1))

    # 4. Get Upper Triangle Indices (offset=1 excludes the diagonal)
    rows, cols = torch.triu_indices(n_nodes, n_nodes, offset=1, device=tensor.device)

    # 5. Extract Values
    # Indexing with [..., rows, cols] returns a tensor where the last
    # two dimensions are flattened into the length of the indices.
    out = temp_tensor[..., rows, cols]

    # 6. Move the collapsed dimension back to the original 'dim' position
    # After step 5, the new "Features" dimension is at the very end (-1).
    return out.movedim(-1, dim)


def test_matrix_to_vector_3d():
    n = 4
    n_samples = 10
    mat = np.random.randn(n_samples, n, n)
    vec = matrix_to_vector_3d(mat)
    expected_dim = n * (n - 1) // 2
    assert vec.shape == (n_samples, expected_dim)



# ============================================================
# Connectome <-> vector conversions (foundation of edge stability
# and mapping p-values back to a connectome). These must round-trip
# exactly, and the numpy and tensor implementations must agree.
# ============================================================

def _symmetric_zero_diag(n, rng):
    m = rng.randn(n, n).astype(np.float32)
    m = (m + m.T) / 2
    np.fill_diagonal(m, 0.0)
    return m


def test_vector_to_matrix_numpy_roundtrip_1d():
    rng = np.random.RandomState(0)
    n_nodes = 6
    n_edges = n_nodes * (n_nodes - 1) // 2  # 15
    vec = rng.randn(n_edges).astype(np.float32)

    mat = vector_to_matrix_numpy(vec, dim=0)
    assert mat.shape == (n_nodes, n_nodes)
    # symmetric with zero diagonal
    assert np.allclose(mat, mat.T)
    assert np.allclose(np.diag(mat), 0.0)
    # vector -> matrix -> vector is the identity
    back = matrix_to_vector_numpy(mat, dim=0)
    assert np.allclose(back, vec)


def test_matrix_to_vector_numpy_roundtrip_symmetric():
    rng = np.random.RandomState(1)
    n_nodes = 7
    mat = _symmetric_zero_diag(n_nodes, rng)
    vec = matrix_to_vector_numpy(mat, dim=0)
    assert vec.shape == (n_nodes * (n_nodes - 1) // 2,)
    # matrix -> vector -> matrix recovers a symmetric, zero-diagonal matrix
    recon = vector_to_matrix_numpy(vec, dim=0)
    assert np.allclose(recon, mat)


def test_vector_to_matrix_numpy_batched():
    rng = np.random.RandomState(2)
    n_nodes, n_runs = 5, 3
    n_edges = n_nodes * (n_nodes - 1) // 2  # 10
    arr = rng.randn(n_edges, n_runs).astype(np.float32)
    mat = vector_to_matrix_numpy(arr, dim=0)
    assert mat.shape == (n_nodes, n_nodes, n_runs)
    back = matrix_to_vector_numpy(mat, dim=0)
    assert np.allclose(back, arr)


def test_tensor_conversion_roundtrip():
    torch.manual_seed(0)
    n_nodes = 6
    n_edges = n_nodes * (n_nodes - 1) // 2
    vec = torch.randn(n_edges)
    mat = vector_to_matrix_tensor_version(vec, dim=0)
    assert tuple(mat.shape) == (n_nodes, n_nodes)
    assert torch.allclose(mat, mat.T)
    back = matrix_to_vector_tensor_version(mat, dim=0)
    assert torch.allclose(back, vec)


def test_numpy_and_tensor_versions_agree():
    rng = np.random.RandomState(3)
    n_nodes = 8
    n_edges = n_nodes * (n_nodes - 1) // 2
    vec = rng.randn(n_edges).astype(np.float32)

    mat_np = vector_to_matrix_numpy(vec, dim=0)
    mat_t = vector_to_matrix_tensor_version(torch.from_numpy(vec), dim=0).numpy()
    assert np.allclose(mat_np, mat_t)


def test_matrix_to_vector_3d_extracts_upper_triangle():
    # Build a batch of matrices with known upper-triangular values
    n_samples, n = 4, 4
    rng = np.random.RandomState(4)
    mats = np.zeros((n_samples, n, n), dtype=np.float32)
    rows, cols = np.triu_indices(n, k=1)
    for s in range(n_samples):
        mats[s, rows, cols] = rng.randn(len(rows))
    vec = matrix_to_vector_3d(mats)
    assert vec.shape == (n_samples, n * (n - 1) // 2)
    for s in range(n_samples):
        assert np.allclose(vec[s], mats[s, rows, cols])
