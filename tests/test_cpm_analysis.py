"""
Tests for the CPMAnalysis pipeline: input handling, data validation, and permutation generation.

Full pipeline correctness (regression + classification) is tested in test_ground_truth.py.
"""

import numpy as np
import pandas as pd
import pytest

from sklearn.model_selection import KFold

from cccpm import CPMAnalysis, UnivariateEdgeSelection, PThreshold
from cccpm.utils import check_data


# --- Input handling ---

def test_input_is_dataframe(cpm_instance, simulated_data):
    X, y, covariates = simulated_data
    cpm_instance.run(
        pd.DataFrame(X),
        pd.DataFrame(y),
        pd.DataFrame(covariates)
    )


# --- Missing value handling ---

def test_nan_in_X(simulated_data):
    X, y, covariates = simulated_data
    X_nan = X.copy()
    X_nan[0, 0] = np.nan

    with pytest.raises(ValueError):
        check_data(X_nan, y, covariates, impute_missings=False)

    # Should not raise
    check_data(X_nan, y, covariates, impute_missings=True)


def test_nan_in_y(simulated_data):
    X, y, covariates = simulated_data
    y_nan = y.copy()
    y_nan[0] = np.nan

    # raise error if y contains nan and impute_missings is False
    with pytest.raises(ValueError):
        check_data(X, y_nan, covariates, impute_missings=False)

    # but also raise an error if y contains nan and impute_missings is True
    # values in y should never be missing
    with pytest.raises(ValueError):
        check_data(X, y_nan, covariates, impute_missings=True)


# --- Reproducibility ---

def test_pipeline_is_reproducible(tmp_path, simulated_data):
    """
    Running the same analysis twice with the same configuration must produce
    identical results — a basic requirement for reproducible research.
    """
    X, y, covariates = simulated_data

    def run_once(subdir):
        edge_selection = UnivariateEdgeSelection(
            edge_statistic="pearson",
            edge_selection=[PThreshold(threshold=[0.05], correction=[None])],
        )
        cpm = CPMAnalysis(
            results_directory=str(tmp_path / subdir),
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
            edge_selection=edge_selection,
            n_permutations=0,
            impute_missing_values=True,
        )
        cpm.run(X=X, y=y, covariates=covariates)
        return cpm

    run_once("run_a")
    run_once("run_b")

    # Aggregated metrics (loaded from disk) must match exactly across runs
    res_a = pd.read_csv(tmp_path / "run_a" / "cv_results_summary.csv")
    res_b = pd.read_csv(tmp_path / "run_b" / "cv_results_summary.csv")
    pd.testing.assert_frame_equal(res_a, res_b)

    # The same edges must be selected
    edges_a = np.load(tmp_path / "run_a" / "stability_edges.npy")
    edges_b = np.load(tmp_path / "run_b" / "stability_edges.npy")
    np.testing.assert_array_equal(edges_a, edges_b)


# --- Permutation generation ---

def test_create_permuted_y_structure(cpm_instance):
    """Test structural integrity: shape and value conservation."""
    y = np.arange(20)

    if cpm_instance.n_permutations < 2:
        pytest.skip("Need at least 2 permutations to test variance")

    original_y_copy = y.copy()
    permuted_y = cpm_instance._create_permuted_y(y)

    # Shape Check
    assert permuted_y.shape == (len(y), cpm_instance.n_permutations)

    # Conservation Check (content is preserved)
    for i in range(min(cpm_instance.n_permutations, 5)):
        assert sorted(permuted_y[:, i]) == list(y)

    # Immutability Check (Original y must not change)
    np.testing.assert_array_equal(y, original_y_copy,
                                  err_msg="The function modified the original input array in place!")


def test_permutations_are_shuffled(cpm_instance):
    """Test that the output is actually randomized and not just repeated."""
    y = np.arange(50)
    permuted_y = cpm_instance._create_permuted_y(y)

    # Ensure column 0 is not identical to column 1
    assert not np.array_equal(permuted_y[:, 0], permuted_y[:, 1]), \
        "Columns are identical! The shuffle likely failed (Repeat Bug)."

    # Ensure the first permutation is not identical to the original input
    assert not np.array_equal(permuted_y[:, 0], y), \
        "The permuted vector is identical to the input! (No shuffle occurred)"
