"""Tests for the connectome <-> upper-triangular-vector conversions."""

import numpy as np

import torch

from cccpm.connectome import (
    matrix_to_vector_3d,
    vector_to_matrix_tensor_version,
)


# Reference NumPy converters, kept only as a consistency check against the
# production tensor converter (same origin, not an independent reference --
# see test_sklearn_equivalence.py for that).

def vector_to_matrix_numpy(array, dim):
    ndim = array.ndim
    dim = dim % ndim

    n_features = array.shape[dim]
    n_nodes = int((1 + np.sqrt(1 + 8 * n_features)) / 2)

    temp_array = np.moveaxis(array, dim, -1)

    out_shape = temp_array.shape[:-1] + (n_nodes, n_nodes)
    out = np.zeros(out_shape, dtype=array.dtype)

    rows, cols = np.triu_indices(n_nodes, k=1)

    out[..., rows, cols] = temp_array
    out[..., cols, rows] = temp_array

    return np.moveaxis(out, (-2, -1), (dim, dim + 1))


def matrix_to_vector_tensor_version(tensor, dim):
    """Collapse adjacent dims [dim, dim+1] (a symmetric matrix) into one upper-triangular edge dim."""
    ndim = tensor.ndim
    dim = dim % ndim

    n_nodes = tensor.shape[dim]
    if n_nodes != tensor.shape[dim + 1]:
        raise ValueError(f"Dimensions at {dim} and {dim + 1} must be square.")

    temp_tensor = tensor.movedim((dim, dim + 1), (-2, -1))
    rows, cols = torch.triu_indices(n_nodes, n_nodes, offset=1, device=tensor.device)
    out = temp_tensor[..., rows, cols]
    return out.movedim(-1, dim)


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
