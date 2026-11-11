"""End-to-end coverage of every combination of the optional pipeline features.

Why: twice now a feature was silently inert on one code path while working on
another, invisible to a suite that tested each feature alone -- presence_filter
was a no-op under the batched pipeline (added to the unbatched path only), and
connected_components was skipped inside batched inner CV, tuning hyperparameters
on unfiltered edges while the outer loop fit on filtered ones. Data is
deliberately tiny; this is about wiring, not statistics."""
import itertools

import numpy as np
import pytest
import torch
from sklearn.model_selection import KFold, ShuffleSplit

from cccpm import CPMAnalysis, UnivariateEdgeSelection, PThreshold
from cccpm.constants import INCREMENTABLE_METRICS, Metrics
from cccpm.simulation.simulate_simple import simulate_confounded_data_chyzhyk


@pytest.fixture(scope="module")
def tiny_data():
    X, y, covariates = simulate_confounded_data_chyzhyk(n_samples=60, n_features=45)
    return X, np.asarray(y).ravel(), covariates


def _build(results_dir, *, inner_cv, model_input, connected_components,
           presence_filter, n_permutations=0):
    thresholds = [0.05, 0.1] if inner_cv else [0.05]
    return CPMAnalysis(
        results_directory=str(results_dir),
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        inner_cv=ShuffleSplit(n_splits=2, test_size=0.3, random_state=42) if inner_cv else None,
        edge_selection=UnivariateEdgeSelection(
            selection_statistic='pearson',
            presence_filter=presence_filter,
            connected_components=connected_components,
            edge_selection=[PThreshold(threshold=thresholds, correction=[None])]),
        model_input=model_input,
        n_permutations=n_permutations,
        impute_missing_values=True,
        device='cpu',
    )


# Every combination of the four optional features: 2^4 = 16 runs.
@pytest.mark.parametrize(
    "inner_cv,model_input,connected_components,presence_filter",
    list(itertools.product([False, True], ['raw', 'residualized'],
                           [False, True], [False, True])),
)
def test_feature_combinations_run_and_produce_results(
        tmp_path, tiny_data, inner_cv, model_input,
        connected_components, presence_filter):
    X, y, covariates = tiny_data
    cpm = _build(tmp_path, inner_cv=inner_cv, model_input=model_input,
                 connected_components=connected_components,
                 presence_filter=presence_filter)
    cpm.run(X=X, y=y, covariates=covariates)

    results = cpm.results_manager
    assert results is not None

    # Aggregated metrics must be finite -- a broken combination tends to surface as
    # NaN, not an exception. Exception: increment is NaN by design for metrics
    # whose difference is not a statistic (constants.INCREMENTABLE_METRICS).
    agg = results.agg_results
    suppressed = [m.name for m in Metrics if m not in INCREMENTABLE_METRICS]
    checked = agg.drop(index="increment", level="model")
    assert checked.notna().all().all(), "aggregated results contain NaN"

    increment = agg.xs("increment", level="model")
    kept = increment.loc[:, [c for c in increment.columns if c[0] not in suppressed]]
    assert kept.notna().all().all(), "increment contains unexpected NaN"
    dropped = increment.loc[:, [c for c in increment.columns if c[0] in suppressed]]
    assert dropped.isna().all().all(), "increment should be NaN for these metrics"

    stability = results.calculate_edge_stability(write=False)  # fraction of folds
    assert torch.isfinite(stability).all()
    assert (stability >= 0).all() and (stability <= 1).all()

    for name in ("cv_results_summary.csv", "cv_predictions.csv", "stability_edges.npy"):
        assert (tmp_path / name).exists(), f"{name} was not written"


def test_connected_components_survives_stable_edge_selection(tmp_path, tiny_data):
    """Regression: run_inner_folds' batched branch called the selector directly,
    skipping the component filter, so connected_components was silently ignored
    whenever select_stable_edges built the final mask from inner-loop stability.
    Verified against the pre-refactor tree: connected_components=999 (drops every
    edge) gave total stability 45.0 there vs 0.0 now. Must read the INNER loop's
    stability -- the outer loop's passed on the buggy code too."""
    X, y, covariates = tiny_data

    def total_stability(connected_components, subdir):
        cpm = CPMAnalysis(
            results_directory=str(tmp_path / subdir),
            cv=KFold(n_splits=3, shuffle=True, random_state=42),
            inner_cv=ShuffleSplit(n_splits=2, test_size=0.3, random_state=42),
            edge_selection=UnivariateEdgeSelection(
                selection_statistic='pearson',
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
    """Keeps the presence_filter no-op fix (already made before this refactor)
    from being lost again -- does not fail on the pre-refactor tree. Half the
    edges are structurally zero in most subjects and must never be selected."""
    rng = np.random.default_rng(0)
    n_samples, n_features = 60, 45
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    sparse_edges = np.arange(n_features // 2)
    mask = rng.random((n_samples, len(sparse_edges))) < 0.8
    X[:, sparse_edges] = np.where(mask, 0.0, X[:, sparse_edges])
    y = X[:, n_features // 2:].sum(axis=1) + rng.normal(size=n_samples) * 0.1
    covariates = rng.normal(size=(n_samples, 2)).astype(np.float32)

    cpm = _build(tmp_path, inner_cv=False, model_input='raw',
                 connected_components=False, presence_filter=0.9)
    cpm.run(X=X, y=y, covariates=covariates)

    stability = cpm.results_manager.calculate_edge_stability(write=False)
    assert stability.sum() > 0, "no edges selected at all; test is vacuous"
    assert stability[sparse_edges].sum() == 0, (
        "edges present in only ~20% of subjects were selected despite "
        "presence_filter=0.9 -- the filter is not being applied")
