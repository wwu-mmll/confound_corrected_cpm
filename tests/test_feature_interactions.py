"""
End-to-end coverage of every combination of the optional pipeline features.

Why this file exists: twice now, a feature has been silently inert in one code
path while working in another, with no crash and no failing test.

  - `presence_filter` was a no-op under the batched pipeline, because the
    feature was added to the unbatched edge-selection path only.
  - `connected_components` was skipped inside the inner CV whenever the model
    supported batching, so hyperparameters were tuned on UNFILTERED edges while
    the outer loop fitted on filtered ones.

Both were invisible to a suite that tested each feature on its own. The features
interact -- an option can be honoured alone and dropped in combination -- so the
combinations are what need covering.

The old batched/unbatched split that produced both bugs is gone, and with it
`can_batch_outer_folds`, the gate that decided which features were allowed on
the fast path. These tests exist so that a future optimisation cannot quietly
reintroduce a path where a requested option stops taking effect.

Each case asserts the run completes, produces well-formed output, and -- where
the option has an observable signature -- that the option actually did something.
Data is deliberately tiny; this is about wiring, not statistics.
"""
import itertools

import numpy as np
import pytest
import torch
from sklearn.model_selection import KFold, ShuffleSplit

from cccpm import CPMAnalysis, UnivariateEdgeSelection, PThreshold
from cccpm.constants import TaskType
from cccpm.simulation.simulate_simple import simulate_confounded_data_chyzhyk


@pytest.fixture(scope="module")
def tiny_data():
    """Small connectome with a real signal: 45 features == 10 nodes."""
    X, y, covariates = simulate_confounded_data_chyzhyk(n_samples=60, n_features=45)
    return X, np.asarray(y).ravel(), covariates


def _build(results_dir, *, inner_cv, calculate_residuals, connected_components,
           presence_filter, n_permutations=0):
    thresholds = [0.05, 0.1] if inner_cv else [0.05]
    return CPMAnalysis(
        results_directory=str(results_dir),
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        inner_cv=ShuffleSplit(n_splits=2, test_size=0.3, random_state=42) if inner_cv else None,
        edge_selection=UnivariateEdgeSelection(
            edge_statistic='pearson',
            presence_filter=presence_filter,
            connected_components=connected_components,
            edge_selection=[PThreshold(threshold=thresholds, correction=[None])]),
        calculate_residuals=calculate_residuals,
        n_permutations=n_permutations,
        impute_missing_values=True,
        device='cpu',
    )


# Every combination of the four optional features: 2^4 = 16 runs.
@pytest.mark.parametrize(
    "inner_cv,calculate_residuals,connected_components,presence_filter",
    list(itertools.product([False, True], repeat=4)),
)
def test_feature_combinations_run_and_produce_results(
        tmp_path, tiny_data, inner_cv, calculate_residuals,
        connected_components, presence_filter):
    X, y, covariates = tiny_data
    cpm = _build(tmp_path, inner_cv=inner_cv, calculate_residuals=calculate_residuals,
                 connected_components=connected_components,
                 presence_filter=presence_filter)
    cpm.run(X=X, y=y, covariates=covariates)

    results = cpm.results_manager
    assert results is not None

    # Aggregated metrics must be finite -- a silently broken combination tends to
    # surface as NaN rather than as an exception.
    agg = results.agg_results
    assert agg.notna().all().all(), "aggregated results contain NaN"

    # Edge stability is a fraction of folds, so it must lie in [0, 1].
    stability = results.calculate_edge_stability(write=False)
    assert torch.isfinite(stability).all()
    assert (stability >= 0).all() and (stability <= 1).all()

    for name in ("cv_results_summary.csv", "cv_predictions.csv", "stability_edges.npy"):
        assert (tmp_path / name).exists(), f"{name} was not written"


def test_connected_components_survives_stable_edge_selection(tmp_path, tiny_data):
    """
    Regression guard for the inner-CV bug, pinned to its observable consequence.

    `run_inner_folds` used to have two branches, and only the unbatched one
    applied the component filter -- the batched branch called the selector
    directly. Because `select_stable_edges=True` builds the FINAL edge mask from
    the inner loop's accumulated stability, `connected_components` was silently
    ignored for the fitted model whenever both options were combined.

    Verified against the pre-refactor tree: with connected_components=999, which
    must drop every edge, the old code reported total stability 45.0 (filter
    ignored) and the current code reports 0.0 (filter applied). An earlier
    version of this test read the OUTER loop's stability instead, which applied
    the filter in both versions and so passed on the buggy code -- it proved
    nothing. This one discriminates.
    """
    X, y, covariates = tiny_data

    def total_stability(connected_components, subdir):
        cpm = CPMAnalysis(
            results_directory=str(tmp_path / subdir),
            cv=KFold(n_splits=3, shuffle=True, random_state=42),
            inner_cv=ShuffleSplit(n_splits=2, test_size=0.3, random_state=42),
            edge_selection=UnivariateEdgeSelection(
                edge_statistic='pearson',
                connected_components=connected_components,
                edge_selection=[PThreshold(threshold=[0.05, 0.1], correction=[None])]),
            select_stable_edges=True, stability_threshold=0.0,
            n_permutations=0, device='cpu')
        cpm.run(X=X, y=y, covariates=covariates)
        return cpm.results_manager.calculate_edge_stability(write=False).sum().item()

    unfiltered = total_stability(False, "cc_off")
    filtered = total_stability(999, "cc_on")

    assert unfiltered > 0, "fixture selected no edges; the test would be vacuous"
    assert filtered == 0, (
        f"connected_components=999 must drop every edge, but total stability is "
        f"{filtered} (unfiltered: {unfiltered}). The component filter is not "
        f"reaching the stable-edge selection path.")


def test_presence_filter_is_applied_end_to_end(tmp_path):
    """
    Regression guard for the presence_filter no-op.

    That bug was fixed before this refactor (the filter had been added to the
    unbatched path only), so unlike the connected_components guard above this
    one does not fail on the pre-refactor tree -- it exists to keep the fix from
    being lost again, not to catch a live defect.

    Half the edges are structurally zero in most subjects. With
    presence_filter=0.9 they must never be selected, in any fold.
    """
    rng = np.random.default_rng(0)
    n_samples, n_features = 60, 45
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    sparse_edges = np.arange(n_features // 2)
    mask = rng.random((n_samples, len(sparse_edges))) < 0.8
    X[:, sparse_edges] = np.where(mask, 0.0, X[:, sparse_edges])
    y = X[:, n_features // 2:].sum(axis=1) + rng.normal(size=n_samples) * 0.1
    covariates = rng.normal(size=(n_samples, 2)).astype(np.float32)

    cpm = _build(tmp_path, inner_cv=False, calculate_residuals=False,
                 connected_components=False, presence_filter=0.9)
    cpm.run(X=X, y=y, covariates=covariates)

    stability = cpm.results_manager.calculate_edge_stability(write=False)
    assert stability.sum() > 0, "no edges selected at all; test is vacuous"
    assert stability[sparse_edges].sum() == 0, (
        "edges present in only ~20% of subjects were selected despite "
        "presence_filter=0.9 -- the filter is not being applied")
