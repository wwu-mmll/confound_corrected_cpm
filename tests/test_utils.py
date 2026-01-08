import pytest
import numpy as np
import pandas as pd

from cccpm.utils import (
    check_data,
    matrix_to_vector_3d,
    get_variable_names,
    matrix_to_upper_triangular_vector,
    vector_to_upper_triangular_matrix,
    matrix_to_vector_3d,
    vector_to_matrix_3d,
    get_colors_from_colormap,
    impute_missing_values,
    select_stable_edges
)

# We create a local fixture for the specific small dataset used in these tests
@pytest.fixture
def small_data_setup(simulated_data):
    """
    Creates the specific data shapes used in the original TestUtils.setUp
    """
    # Original code used n_samples=50, n_features=5
    # We can just slice the larger simulated_data fixture
    X, y, cov = simulated_data
    n_samples, n_features = 50, 5

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


def test_matrix_to_upper_triangular_vector_basic():
    mat = np.array([
        [0, 1, 2],
        [1, 0, 3],
        [2, 3, 0]
    ])
    vec = matrix_to_upper_triangular_vector(mat)
    assert np.allclose(vec, np.array([1, 2, 3]))


def test_matrix_to_upper_triangular_vector_invalid_shape():
    with pytest.raises(ValueError):
        matrix_to_upper_triangular_vector(np.zeros((3, 4)))


def test_vector_to_upper_triangular_matrix_roundtrip():
    mat = np.array([
        [0, 1, 2],
        [1, 0, 3],
        [2, 3, 0]
    ])
    vec = matrix_to_upper_triangular_vector(mat)
    mat_rec = vector_to_upper_triangular_matrix(vec)
    assert np.allclose(mat, mat_rec)


def test_vector_to_upper_triangular_matrix_invalid_length():
    vec = np.arange(5)  # 5 fits no triangle number
    with pytest.raises(ValueError):
        vector_to_upper_triangular_matrix(vec)


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


def test_impute_missing_values_basic():
    X_train = np.array([[1.0, np.nan], [3.0, 4.0]])
    X_test = np.array([[np.nan, 2.0]])
    cov_train = np.array([[np.nan], [2.0]])
    cov_test = np.array([[1.0]])

    Xt, Xv, Ct, Cv = impute_missing_values(
        X_train, X_test, cov_train, cov_test
    )

    assert not np.isnan(Xt).any()
    assert not np.isnan(Xv).any()
    assert not np.isnan(Ct).any()
    assert not np.isnan(Cv).any()


def test_select_stable_edges_basic():
    stability = {
        'positive': np.array([0.1, 0.8, 0.6]),
        'negative': np.array([0.9, 0.2, 0.7])
    }

    selected = select_stable_edges(stability, stability_threshold=0.65)

    assert np.array_equal(selected['positive'], np.array([1]))
    assert np.array_equal(selected['negative'], np.array([0, 2]))


def test_select_stable_edges_empty():
    stability = {
        'positive': np.array([0.1, 0.2]),
        'negative': np.array([0.3, 0.4])
    }

    selected = select_stable_edges(stability, stability_threshold=0.9)

    assert selected['positive'].size == 0
    assert selected['negative'].size == 0
