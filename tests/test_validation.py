import pytest
import numpy as np
import pandas as pd

from cccpm.validation import (
    check_data,
    get_variable_names,
    infer_n_nodes,
)


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


@pytest.mark.parametrize("kind,expected_shape", [
    ("series", (50, 1)),          # a pandas Series -> one column
    ("array_1d", (50, 1)),        # a 1-D array -> one column
    ("array_2d", (50, 2)),        # a 2-D array passes through
    ("dataframe", (50, 3)),       # 'cat' one-hot encodes to 2 + 'num' = 3
])
def test_covariates_are_coerced_to_two_dimensions(small_data_setup, kind,
                                                  expected_shape):
    """Every accepted covariate spelling ends up as [n_samples, n_covariates].

    One behaviour, one test: these were four near-identical test functions
    differing only in the input type and the expected width.
    """
    X2d, y1d, _, cov_series, cov_1d_array, cov_2d_array = small_data_setup
    covariates = {
        "series": cov_series,
        "array_1d": cov_1d_array,
        "array_2d": cov_2d_array,
        "dataframe": pd.DataFrame({
            'cat': np.random.choice(['A', 'B', 'C'], size=50),
            'num': np.random.randn(50),
        }),
    }[kind]

    _, _, cov_out = check_data(X2d, y1d, covariates)

    assert cov_out.ndim == 2
    assert cov_out.shape == expected_shape


def test_error_on_invalid_covariate_dim(small_data_setup):
    X2d, y1d, _, _, _, _ = small_data_setup
    with pytest.raises(ValueError):
        check_data(X2d, y1d, np.zeros((50, 2, 2)))


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


@pytest.mark.parametrize("X,y,covariates,expected", [
    # Labelled inputs keep their own names...
    (pd.DataFrame(np.zeros((10, 3)), columns=['f1', 'f2', 'f3']),
     pd.DataFrame({'target_col': np.arange(10)}),
     pd.DataFrame({'c1': np.arange(10), 'c2': np.arange(10) * 2}),
     (['f1', 'f2', 'f3'], 'target_col', ['c1', 'c2'])),
    # ...unlabelled arrays fall back to positional names...
    (np.zeros((5, 2)), pd.Series(np.arange(5), name='yser'), np.zeros((5, 4)),
     (['feature_0', 'feature_1'], 'yser',
      ['covariate_0', 'covariate_1', 'covariate_2', 'covariate_3'])),
    # ...and the two can be mixed in one call.
    (np.zeros((7, 1)), np.arange(7), pd.Series(np.arange(7), name='cov_only'),
     (['feature_0'], 'target', ['cov_only'])),
])
def test_get_variable_names_falls_back_per_input(X, y, covariates, expected):
    """Names come from the data where it carries them, positional otherwise."""
    assert get_variable_names(X, y, covariates) == expected
