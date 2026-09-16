"""
Tests for the CPMAnalysis pipeline: input handling, data validation, and permutation generation.

Full pipeline correctness (regression + classification) is tested in test_ground_truth.py.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from sklearn.model_selection import KFold, RepeatedKFold

from cccpm import CPMAnalysis, UnivariateEdgeSelection, PThreshold
from cccpm.utils import check_data
from cccpm.reporting.reporting_utils import average_over_repeats
from cccpm.simulation.simulate_simple import simulate_confounded_data_chyzhyk


def _make_cpm(results_directory, **kwargs):
    edge_selection = UnivariateEdgeSelection(
        edge_statistic="pearson",
        edge_selection=[PThreshold(threshold=[0.05], correction=[None])],
    )
    return CPMAnalysis(
        results_directory=str(results_directory),
        edge_selection=edge_selection,
        **kwargs,
    )


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


# --- Repeated k-fold ---

def test_average_over_repeats_collapses_per_subject():
    """The helper averages a subject's value over repeats, keeping y_true."""
    df = pd.DataFrame({
        "sample_index": [0, 0, 1, 1],
        "model": "connectome",
        "network": "both",
        "y_pred": [1.0, 3.0, 10.0, 20.0],
        "y_true": [2.0, 2.0, 5.0, 5.0],
        "repeat": [0, 1, 0, 1],
    })
    out = average_over_repeats(df, ["y_pred"])
    assert len(out) == 2  # one row per subject
    by_subject = out.set_index("sample_index")
    assert by_subject.loc[0, "y_pred"] == 2.0   # mean(1, 3)
    assert by_subject.loc[1, "y_pred"] == 15.0  # mean(10, 20)
    assert by_subject.loc[0, "y_true"] == 2.0   # constant, preserved


def test_repeated_kfold_averages_individual_outputs(tmp_path, simulated_data):
    """RepeatedKFold must tag predictions with a repeat id and, after the
    report's per-subject averaging, plot each subject exactly once."""
    X, y, covariates = simulated_data
    n_samples = X.shape[0]
    n_splits, n_repeats = 5, 3

    edge_selection = UnivariateEdgeSelection(
        edge_statistic="pearson",
        edge_selection=[PThreshold(threshold=[0.05], correction=[None])],
    )
    cpm = CPMAnalysis(
        results_directory=str(tmp_path),
        task_type="regression",
        cv=RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0),
        edge_selection=edge_selection,
        n_permutations=0,
        impute_missing_values=False,
    )
    cpm.run(X, y, covariates)

    # Raw CSV keeps every repeat: a repeat id spanning 0..n_repeats-1, and each
    # subject appears once per repeat for a given model/network.
    preds = pd.read_csv(tmp_path / "cv_predictions.csv")
    assert "repeat" in preds.columns
    assert set(preds["repeat"].unique()) == set(range(n_repeats))
    conn_both = preds[(preds["model"] == "connectome") & (preds["network"] == "both")]
    assert (conn_both["sample_index"].value_counts() == n_repeats).all()

    # After the report's averaging, every subject is represented once.
    averaged = average_over_repeats(conn_both, ["y_pred"])
    assert len(averaged) == n_samples
    assert averaged["sample_index"].nunique() == n_samples

    # Edge stability still pools all folds+repeats (folds axis == n_splits*n_repeats).
    assert cpm.results_manager.dims["folds"] == n_splits * n_repeats


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


# --- RNG hygiene ---

def test_construction_does_not_touch_global_rng(tmp_path):
    """Constructing CPMAnalysis must not mutate the user's global NumPy/torch RNG."""
    np.random.seed(123)
    torch.manual_seed(123)
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()

    _make_cpm(tmp_path / "rng")

    np_after = np.random.get_state()
    assert np_state[0] == np_after[0]
    assert np.array_equal(np_state[1], np_after[1])
    assert np_state[2] == np_after[2]
    assert torch.equal(torch_state, torch.get_rng_state())


def test_permutations_reproducible_across_instances(tmp_path):
    """Same random_state -> identical permutations, without relying on global RNG."""
    y = np.arange(30)
    perm_a = _make_cpm(tmp_path / "a", n_permutations=5)._create_permuted_y(y)
    perm_b = _make_cpm(tmp_path / "b", n_permutations=5)._create_permuted_y(y)
    np.testing.assert_array_equal(perm_a, perm_b)

    # A different random_state should give a different permutation
    perm_c = _make_cpm(tmp_path / "c", n_permutations=5,
                       random_state=7)._create_permuted_y(y)
    assert not np.array_equal(perm_a, perm_c)


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


# --- Permutation chunking (memory valve) --------------------------------------

def test_permutation_chunking_does_not_change_results(tmp_path, monkeypatch):
    """
    Chunking permutations is a memory valve, not a change of method: splitting
    the runs axis must not alter any result.

    Permutations are always vectorised (y is [N_samples, N_runs] and the edge
    statistic is one matmul over all columns). The chunk size only caps how many
    columns are in flight so a large parcellation crossed with many permutations
    degrades in speed rather than running out of memory -- so results computed in
    chunks must match results computed in one pass.

    Edge selection must agree exactly. Metrics are compared with a tolerance
    because reducing over a differently-shaped tensor changes float32
    accumulation order; the observed drift is ~6e-7 on values of order 1.
    """
    import cccpm.cpm_analysis as cpm_analysis_module
    from cccpm.results_manager import ResultsManager
    from cccpm.constants import TaskType

    X, y, covariates = simulate_confounded_data_chyzhyk(n_samples=120, n_features=45)
    rng = np.random.default_rng(0)
    y_perms = np.stack([rng.permutation(np.asarray(y).ravel()) for _ in range(17)], axis=1)

    def run(chunk_size):
        captured = {}
        original_init = ResultsManager.__init__

        def remember(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            captured['manager'] = self

        monkeypatch.setattr(ResultsManager, '__init__', remember)
        if chunk_size is not None:
            monkeypatch.setattr(cpm_analysis_module, 'plan_permutation_chunk',
                                lambda **kwargs: chunk_size)
        cpm = CPMAnalysis(
            results_directory=str(tmp_path / f"chunk_{chunk_size}"),
            cv=KFold(n_splits=3, shuffle=True, random_state=1),
            edge_selection=UnivariateEdgeSelection(
                edge_statistic='pearson',
                edge_selection=[PThreshold(threshold=[0.05], correction=[None])]),
            inner_cv=None, n_permutations=0, device='cpu')
        cpm.task_type = TaskType.regression
        cpm._single_run(X=X, y=y_perms, covariates=covariates, perm_run=True)
        monkeypatch.undo()
        manager = captured['manager']
        return manager.results.clone(), manager.cv_edge_sum.clone()

    reference_metrics, reference_edges = run(None)          # single pass, 17 runs
    for chunk_size in (1, 3, 5, 16):
        metrics, edges = run(chunk_size)
        assert torch.equal(edges, reference_edges), (
            f"chunk={chunk_size} selected different edges")
        torch.testing.assert_close(metrics, reference_metrics, rtol=0, atol=1e-5)
