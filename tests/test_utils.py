import pytest
import numpy as np
import pandas as pd

import torch

from cccpm.utils import (
    check_data,
    matrix_to_vector_3d,
    get_variable_names,
    infer_n_nodes,
    vector_to_matrix_tensor_version,
    build_fold_batch,
    torch_impute_missing_values,
    torch_impute_missing_values_batched,
)


# ============================================================
# Reference implementations for the connectome <-> vector conversions.
#
# These plain NumPy converters used to live in cccpm.utils (deleted as dead
# production code -- nothing in src/, examples/, or scripts/ called them; the
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


def test_infer_n_nodes():
    # Valid upper-triangular edge counts
    assert infer_n_nodes(1) == 2      # 2 nodes -> 1 edge
    assert infer_n_nodes(6) == 4      # 4 nodes -> 6 edges
    assert infer_n_nodes(45) == 10    # 10 nodes -> 45 edges
    assert infer_n_nodes(1225) == 50  # 50 nodes -> 1225 edges
    # Invalid counts return None
    assert infer_n_nodes(5) is None
    assert infer_n_nodes(400) is None
    assert infer_n_nodes(0) is None


def test_check_data_rejects_non_triangular_n_features(small_data_setup):
    """A non-connectome number of features should fail fast with a clear error."""
    _, y1d, cov2d, _, _, _ = small_data_setup
    n_samples = 30
    X_bad = np.random.randn(n_samples, 400)  # 400 is not n*(n-1)/2 for any n
    with pytest.raises(ValueError, match="not a valid connectome size"):
        check_data(X_bad, y1d[:n_samples], cov2d[:n_samples])

# We create a local fixture for the specific small dataset used in these tests
@pytest.fixture
def small_data_setup(simulated_data):
    """
    Creates the specific data shapes used in the original TestUtils.setUp
    """
    # n_features must be a valid connectome size (n_nodes*(n_nodes-1)/2);
    # 6 corresponds to a 4-node connectome.
    X, y, cov = simulated_data
    n_samples, n_features = 50, 6

    X2d = X[:n_samples, :n_features]
    y1d = y[:n_samples]
    cov2d = cov[:n_samples]

    cov_series = pd.Series(np.arange(n_samples), name='covser')
    cov_1d_array = np.arange(n_samples)
    cov_2d_array = np.vstack([np.arange(n_samples), np.arange(n_samples)]).T

    return X2d, y1d, cov2d, cov_series, cov_1d_array, cov_2d_array


def test_matrix_to_vector_3d():
    n = 4
    n_samples = 10
    mat = np.random.randn(n_samples, n, n)
    vec = matrix_to_vector_3d(mat)
    expected_dim = n * (n - 1) // 2
    assert vec.shape == (n_samples, expected_dim)


def test_accepts_3d_X(small_data_setup):
    X2d, y1d, cov2d, _, _, _ = small_data_setup
    n = 6
    n_samples = 20
    X3d = np.random.randn(n_samples, n, n)

    X_out, y_out, cov_out = check_data(
        X3d, y1d[:n_samples], cov2d[:n_samples]
    )
    expected_dim = n * (n - 1) // 2
    assert X_out.shape == (n_samples, expected_dim)


def test_error_on_invalid_X_dim(small_data_setup):
    _, y1d, cov2d, _, _, _ = small_data_setup
    X1d = np.arange(10)
    with pytest.raises(ValueError):
        check_data(X1d, y1d[:10], cov2d[:10])


def test_squeeze_y(small_data_setup):
    X2d, y1d, cov2d, _, _, _ = small_data_setup
    y2d = y1d.reshape(-1, 1)
    _, y_out, _ = check_data(X2d, y2d, cov2d)
    assert y_out.ndim == 1
    assert y_out.shape == (len(y1d),)


def test_error_on_invalid_y_dim(small_data_setup):
    X2d, _, cov2d, _, _, _ = small_data_setup
    y2d_bad = np.random.randn(10, 2)
    with pytest.raises(ValueError):
        check_data(X2d[:10], y2d_bad, cov2d[:10])


def test_covariates_series(small_data_setup):
    X2d, y1d, _, cov_series, _, _ = small_data_setup
    _, _, cov_out = check_data(X2d, y1d, cov_series)
    assert cov_out.ndim == 2
    assert cov_out.shape == (50, 1)


def test_covariates_1d_array(small_data_setup):
    X2d, y1d, _, _, cov_1d_array, _ = small_data_setup
    _, _, cov_out = check_data(X2d, y1d, cov_1d_array)
    assert cov_out.shape == (50, 1)


def test_covariates_2d_array(small_data_setup):
    X2d, y1d, _, _, _, cov_2d_array = small_data_setup
    _, _, cov_out = check_data(X2d, y1d, cov_2d_array)
    assert cov_out.shape == cov_2d_array.shape


def test_accepts_dataframe_covariates(small_data_setup):
    X2d, y1d, _, _, _, _ = small_data_setup
    cov_df = pd.DataFrame({
        'cat': np.random.choice(['A', 'B', 'C'], size=50),
        'num': np.random.randn(50)
    })
    _, _, cov_out = check_data(X2d, y1d, cov_df)
    # 'cat' -> 2 one-hot columns + 'num' = 3 features
    assert cov_out.shape == (50, 3)


def test_missing_values_behavior(small_data_setup):
    X2d, y1d, cov2d, _, _, _ = small_data_setup

    # NaN in X -> error without impute
    X_nan = X2d.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError):
        check_data(X_nan, y1d, cov2d, impute_missings=False)

    # NaN in X allowed with impute
    X_out, _, _ = check_data(X_nan, y1d, cov2d, impute_missings=True)
    assert np.isnan(X_out[0, 0])

    # NaN in y -> always error
    y_nan = y1d.copy()
    y_nan[0] = np.nan
    with pytest.raises(ValueError):
        check_data(X2d, y_nan, cov2d, impute_missings=True)


def test_error_on_invalid_covariate_dim(small_data_setup):
    X2d, y1d, _, _, _, _ = small_data_setup
    cov_3d = np.zeros((50, 2, 2))
    with pytest.raises(ValueError):
        check_data(X2d, y1d, cov_3d)


# Tests for get_variable_names
def test_get_variable_names_with_dataframe_inputs():
    X_df = pd.DataFrame(np.random.randn(10, 3), columns=['f1', 'f2', 'f3'])
    y_df = pd.DataFrame({'target_col': np.arange(10)})
    cov_df = pd.DataFrame({'c1': np.arange(10), 'c2': np.arange(10) * 2})

    X_names, y_name, cov_names = get_variable_names(X_df, y_df, cov_df)

    assert X_names == ['f1', 'f2', 'f3']
    assert y_name == 'target_col'
    assert cov_names == ['c1', 'c2']


def test_get_variable_names_with_series_and_numpy():
    X_arr = np.zeros((5, 2))
    y_ser = pd.Series(np.arange(5), name='yser')
    cov_arr = np.zeros((5, 4))

    X_names, y_name, cov_names = get_variable_names(X_arr, y_ser, cov_arr)

    assert X_names == ['feature_0', 'feature_1']
    assert y_name == 'yser'
    assert cov_names == ['covariate_0', 'covariate_1', 'covariate_2', 'covariate_3']


def test_get_variable_names_with_array_y_and_df_covariates():
    X_arr = np.zeros((7, 1))
    y_arr = np.arange(7)
    cov_ser = pd.Series(np.arange(7), name='cov_only')

    X_names, y_name, cov_names = get_variable_names(X_arr, y_arr, cov_ser)

    assert X_names == ['feature_0']
    assert y_name == 'target'
    assert cov_names == ['cov_only']


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

# ============================================================
# build_fold_batch / torch_impute_missing_values_batched
# ============================================================

def _uneven_splits(n_total, n_folds, seed):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n_total)
    edges = sorted(rng.choice(range(5, n_total - 5), size=n_folds - 1, replace=False))
    boundaries = [0] + list(edges) + [n_total]
    splits = []
    for i in range(n_folds):
        test_idx = idx[boundaries[i]:boundaries[i + 1]]
        train_idx = np.setdiff1d(idx, test_idx)
        splits.append((train_idx, test_idx))
    return splits


def test_build_fold_batch_matches_manual_indexing_with_padding():
    torch.manual_seed(0)
    N, F, P, C = 27, 5, 2, 3
    X = torch.randn(N, F)
    y = torch.randn(N, P)
    cov = torch.randn(N, C)

    splits = _uneven_splits(N, n_folds=4, seed=5)
    fb = build_fold_batch(X, y, cov, splits)
    assert fb.n_train.unique().numel() > 1  # genuinely uneven

    for b, (tr, te) in enumerate(splits):
        n_tr, n_te = len(tr), len(te)
        assert fb.n_train[b].item() == n_tr
        assert fb.n_test[b].item() == n_te
        assert torch.allclose(fb.X_train[b, :n_tr], X[torch.as_tensor(tr)])
        assert torch.allclose(fb.y_train[b, :n_tr], y[torch.as_tensor(tr)])
        assert torch.allclose(fb.cov_train[b, :n_tr], cov[torch.as_tensor(tr)])
        assert torch.allclose(fb.X_test[b, :n_te], X[torch.as_tensor(te)])
        # padded rows are exactly zero
        assert torch.all(fb.X_train[b, n_tr:] == 0)
        assert torch.all(fb.X_test[b, n_te:] == 0)
        assert fb.train_valid[b, :n_tr].all() and not fb.train_valid[b, n_tr:].any()


def test_build_fold_batch_handles_nan_in_pad_source_row():
    """Padded rows must be exactly 0 even if row 0 (the pad-index source)
    contains NaN -- NaN * 0 == NaN, so a naive multiply-by-mask would leave
    NaN in padded positions instead of zeroing them."""
    torch.manual_seed(1)
    N, F = 20, 4
    X = torch.randn(N, F)
    X[0, 1] = float('nan')
    y = torch.randn(N, 1)
    cov = torch.randn(N, 2)

    splits = _uneven_splits(N, n_folds=3, seed=6)
    fb = build_fold_batch(X, y, cov, splits)
    for b in range(len(splits)):
        n = fb.n_train[b].item()
        pad_rows = fb.X_train[b, n:]
        assert torch.all(pad_rows == 0)
        assert not torch.isnan(pad_rows).any()


def test_torch_impute_missing_values_batched_matches_unbatched():
    torch.manual_seed(2)
    N, F, C = 24, 5, 2
    X = torch.randn(N, F)
    X[3, 2] = float('nan')
    X[15, 0] = float('nan')
    y = torch.randn(N, 1)
    cov = torch.randn(N, C)
    cov[7, 1] = float('nan')

    splits = _uneven_splits(N, n_folds=3, seed=7)
    fb = build_fold_batch(X, y, cov, splits)
    Xtr_b, Xte_b, covtr_b, covte_b = torch_impute_missing_values_batched(
        fb.X_train, fb.X_test, fb.cov_train, fb.cov_test, fb.train_valid)

    assert not torch.isnan(Xtr_b).any()
    assert not torch.isnan(Xte_b).any()

    for b, (tr, te) in enumerate(splits):
        n_tr, n_te = len(tr), len(te)
        Xtr_ref, Xte_ref, covtr_ref, covte_ref = torch_impute_missing_values(
            X[torch.as_tensor(tr)].clone(), X[torch.as_tensor(te)].clone(),
            cov[torch.as_tensor(tr)].clone(), cov[torch.as_tensor(te)].clone())
        assert torch.allclose(Xtr_b[b, :n_tr], Xtr_ref, atol=1e-5)
        assert torch.allclose(Xte_b[b, :n_te], Xte_ref, atol=1e-5)
        assert torch.allclose(covtr_b[b, :n_tr], covtr_ref, atol=1e-5)
        assert torch.allclose(covte_b[b, :n_te], covte_ref, atol=1e-5)
