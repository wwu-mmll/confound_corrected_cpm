"""
Vanilla CPM: running without covariates.

`covariates` used to be a required argument, and omitting it crashed deep inside
`check_data`. It is now optional, which makes exactly one model meaningful --
`connectome`. The others (`covariates`, `full`, `increment`) all
need a confound design and are reported as NaN rather than as a number that
would read as a real, terrible model score.

The results tensor keeps its full shape: undefined variants are NaN-filled, not
reshaped away, and `available_models.json` tells the report which rows mean
something.
"""
import json
import os
import re

import numpy as np
import pytest
import torch
from sklearn.model_selection import KFold

from conftest import report_text

from cccpm import CPMAnalysis, PThreshold, UnivariateEdgeSelection
from cccpm.constants import Models, Networks
from cccpm.models.linear_model import LinearCPM
from cccpm.validation import check_data, get_variable_names


COVARIATE_DEPENDENT = (Models.covariates, Models.full, Models.increment)


def _says_nan(results_directory):
    """Does the reader actually see the word "nan" anywhere in the report?

    Anchored, and over flattened text rather than raw HTML -- see
    ``conftest.report_text`` for why both halves of that matter.
    """
    return re.search(r"\bnan\b", report_text(results_directory), re.I) is not None


def _data(seed=0, n=120, n_nodes=10, binary=False):
    rng = np.random.RandomState(seed)
    n_features = n_nodes * (n_nodes - 1) // 2
    X = rng.randn(n, n_features)
    Z = rng.randn(n, 2)
    y = X[:, :5].sum(1) * 0.5 + Z[:, 0] + rng.randn(n)
    if binary:
        y = (y > np.median(y)).astype(float)
    return X, y, Z


def _analysis(tmp_path, selection_input="raw", **kwargs):
    ue = UnivariateEdgeSelection(
        selection_statistic="pearson", selection_input=selection_input,
        edge_selection=[PThreshold(threshold=0.05, correction=[None])])
    return CPMAnalysis(
        results_directory=str(tmp_path),
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
        edge_selection=ue, n_permutations=0, **kwargs)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_check_data_returns_none_for_absent_covariates():
    X, y, _ = _data()
    X_checked, y_checked, cov = check_data(X, y, None)
    assert cov is None
    assert X_checked.shape == X.shape
    assert y_checked.shape == (len(y),)


def test_get_variable_names_without_covariates():
    X, y, _ = _data()
    _, _, covar_names = get_variable_names(X, y, None)
    assert covar_names == []


# ---------------------------------------------------------------------------
# Options that presuppose covariates must fail up front, not degrade silently
# ---------------------------------------------------------------------------

def test_residualized_selection_without_covariates_raises(tmp_path):
    X, y, _ = _data()
    cpm = _analysis(tmp_path, selection_input="residualized", task_type="regression")
    with pytest.raises(ValueError, match="require covariates"):
        cpm.run(X=X, y=y)


def test_error_names_the_offending_parameter(tmp_path):
    """A user has to be able to tell which argument to change."""
    X, y, _ = _data()
    cpm = _analysis(tmp_path, selection_input="residualized", task_type="regression")
    with pytest.raises(ValueError) as excinfo:
        cpm.run(X=X, y=y)
    assert "selection_input='residualized'" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The model itself
# ---------------------------------------------------------------------------

def test_model_nan_fills_covariate_dependent_variants():
    X, y, _ = _data()
    n_features = X.shape[1]
    edges = torch.zeros(n_features, 2, 1, dtype=torch.bool)
    edges[:5, Networks.positive, 0] = True
    edges[5:10, Networks.negative, 0] = True

    model = LinearCPM(edges=edges, device='cpu').fit(X, y.reshape(-1, 1), None)
    pred = model.predict(X).numpy()

    assert model.available_models == ['connectome']
    assert np.isfinite(pred[:, Models.connectome]).all()
    for model_idx in COVARIATE_DEPENDENT:
        assert np.isnan(pred[:, model_idx]).all(), f"{model_idx.name} should be NaN"


def test_connectome_model_is_unaffected_by_dropping_covariates():
    """Given the same edges, the connectome model never touches the covariate
    design, so its predictions must be identical either way."""
    X, y, Z = _data()
    n_features = X.shape[1]
    edges = torch.zeros(n_features, 2, 1, dtype=torch.bool)
    edges[:5, Networks.positive, 0] = True
    edges[5:10, Networks.negative, 0] = True

    with_cov = LinearCPM(edges=edges, device='cpu').fit(X, y.reshape(-1, 1), Z)
    without = LinearCPM(edges=edges, device='cpu').fit(X, y.reshape(-1, 1), None)

    np.testing.assert_allclose(
        with_cov.predict(X, Z).numpy()[:, Models.connectome],
        without.predict(X).numpy()[:, Models.connectome],
        rtol=1e-6, atol=1e-6)


def test_classification_metrics_are_nan_not_a_plausible_score():
    """`nan > 0.5` is False, so an undefined prediction column would otherwise
    score as "class 0 for everyone" -- an accuracy near the base rate, with
    nothing marking it as meaningless."""
    from cccpm.scoring import score_models
    from cccpm.constants import Metrics, TaskType

    rng = np.random.RandomState(0)
    n = 80
    y_true = (rng.rand(n, 1) > 0.5).astype(np.float32)
    y_pred = rng.rand(n, len(Models), len(Networks), 1).astype(np.float32)
    y_pred[:, Models.covariates] = np.nan

    scores = score_models(y_true=y_true, y_pred=y_pred,
                          task_type=TaskType.classification, device='cpu')

    for metric in (Metrics.accuracy, Metrics.balanced_accuracy,
                   Metrics.f1_score, Metrics.roc_auc):
        assert torch.isnan(scores[metric, Models.covariates]).all(), metric.name
        assert torch.isfinite(scores[metric, Models.connectome]).all(), metric.name


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("binary", [False, True])
def test_run_without_covariates_end_to_end(tmp_path, binary):
    X, y, _ = _data(binary=binary)
    task = "classification" if binary else "regression"

    cpm = _analysis(tmp_path, task_type=task)
    cpm.run(X=X, y=y)

    with open(os.path.join(str(tmp_path), 'available_models.json'), encoding='utf-8') as f:
        assert json.load(f) == ['connectome']

    metric = "accuracy" if binary else "pearson_score"
    ag = cpm.results_manager.agg_results
    connectome = ag.loc[("connectome", "both"), (metric, "mean")]
    assert np.isfinite(np.ravel(connectome)).all()

    for name in ("covariates", "full", "increment"):
        assert np.isnan(np.ravel(ag.loc[(name, "both"), (metric, "mean")])).all(), name

    # NaN means "not applicable" and must never reach the reader as "nan":
    # the same formatting path also carries the suppressed Pearson increment.
    # The report renders, and says nothing about models that do not exist.
    assert os.path.exists(os.path.join(str(tmp_path), 'report.html'))
    text = report_text(tmp_path)
    assert "Covariates only" not in text
    assert not _says_nan(tmp_path)


def test_permutation_p_values_are_nan_for_undefined_models(tmp_path):
    """The permutation test must not report the non-existent models as the most
    significant result in the table.

    p_values.csv is produced from cv_results_summary.csv, where the undefined
    variants are NaN placeholders. Without a guard every comparison against NaN
    is False, the count is 0, and the +1 correction reports the permutation
    floor -- 1/(n_perms+1) -- which reads as "highly significant".
    """
    import pandas as pd

    X, y, _ = _data()
    ue = UnivariateEdgeSelection(
        selection_statistic="pearson",
        edge_selection=[PThreshold(threshold=0.05, correction=[None])])
    cpm = CPMAnalysis(
        results_directory=str(tmp_path),
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
        edge_selection=ue, n_permutations=20, task_type="regression")
    cpm.run(X=X, y=y)

    p_values = pd.read_csv(os.path.join(str(tmp_path), 'p_values.csv'))
    p_values = p_values.set_index(['network', 'model'])

    metric = 'pearson_score'
    assert np.isfinite(p_values.loc[('both', 'connectome'), metric])
    for name in ('covariates', 'full', 'increment'):
        value = p_values.loc[('both', name), metric]
        assert np.isnan(value), f"{name} got p={value}, but the model does not exist"

    # And the report shows none of them.
    assert not _says_nan(tmp_path)


def test_network_strengths_hold_only_the_connectome(tmp_path):
    X, y, _ = _data()
    cpm = _analysis(tmp_path, task_type="regression")
    cpm.run(X=X, y=y)

    strengths = cpm.results_manager.cv_network_strengths
    assert set(strengths['model'].unique()) == {'connectome'}
