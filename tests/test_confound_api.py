"""
The confound-control API: selection_statistic x selection_input.

Four apparent confound levers in CCCPM reduce to two independent choices:

  S  does edge *selection* control for the confounds?   -> selection_input
  F  are the *features* deconfounded before the model?  -> which model you read

F costs nothing: `connectome_residualized` is computed on every run, so a single
run reports both of its cells. Only S is a run-level choice.

These tests pin that structure, the exact numeric equivalence of the deprecated
spellings, and the property that motivated choosing the coefficient test over a
plain correlation on residualised edges: a p-threshold means what it says
regardless of how confounded the data are.
"""
import numpy as np
import pytest
import torch
from sklearn.model_selection import KFold

from cccpm import CPMAnalysis, PThreshold, UnivariateEdgeSelection
from cccpm.edge_selection import EdgeStatistic
from cccpm.statistics import correlations_and_pvalues, get_residuals


def _data(seed=0, n=300, n_features=300, n_confounds=3, confound_beta=0.8):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features).astype(np.float32)
    Z = rng.randn(n, n_confounds).astype(np.float32)
    y = (X[:, :20].sum(1) * 0.1 + Z[:, 0] * confound_beta
         + rng.randn(n)).astype(np.float32)
    return X, y.reshape(-1, 1), Z


def _stat(**kwargs):
    X, y, Z = _data()
    st = EdgeStatistic(**kwargs)
    r, p = st.fit_transform(X=X, y=y, covariates=Z, device=torch.device('cpu'))
    return r.numpy(), p.numpy()


# ---------------------------------------------------------------------------
# Deprecated spellings must be exactly the new ones, not approximately
# ---------------------------------------------------------------------------

LEGACY = [
    ('pearson',                'pearson',  'raw'),
    ('spearman',               'spearman', 'raw'),
    ('point_biserial',         'pearson',  'raw'),
    ('pearson_partial',        'pearson',  'residualized'),
    ('spearman_partial',       'spearman', 'residualized'),
    ('point_biserial_partial', 'pearson',  'residualized'),
]


@pytest.mark.parametrize("legacy,statistic,selection_input", LEGACY)
def test_deprecated_edge_statistic_is_numerically_identical(
        legacy, statistic, selection_input):
    with pytest.deprecated_call():
        r_old, p_old = _stat(edge_statistic=legacy)
    r_new, p_new = _stat(selection_statistic=statistic,
                         selection_input=selection_input)
    np.testing.assert_array_equal(r_old, r_new)
    np.testing.assert_array_equal(p_old, p_new)


def test_deprecation_warning_names_the_replacement():
    with pytest.warns(DeprecationWarning) as record:
        EdgeStatistic(edge_statistic='pearson_partial')
    message = str(record[0].message)
    assert "selection_statistic='pearson'" in message
    assert "selection_input='residualized'" in message


def test_unknown_values_are_rejected():
    with pytest.raises(ValueError, match="selection_statistic must be one of"):
        EdgeStatistic(selection_statistic='kendall')
    with pytest.raises(ValueError, match="selection_input must be one of"):
        EdgeStatistic(selection_input='deconfounded')
    with pytest.raises(ValueError, match="Unknown edge_statistic"):
        EdgeStatistic(edge_statistic='pearson_semipartial')


def test_calculate_residuals_is_deprecated_onto_selection_input(tmp_path):
    ue = UnivariateEdgeSelection(
        selection_statistic="pearson",
        edge_selection=[PThreshold(threshold=0.05, correction=[None])])
    with pytest.warns(DeprecationWarning, match="connectome_residualized"):
        CPMAnalysis(results_directory=str(tmp_path),
                    cv=KFold(n_splits=3), edge_selection=ue,
                    calculate_residuals=True, n_permutations=0)
    assert ue.statistic._input == 'residualized'


# ---------------------------------------------------------------------------
# What selection_input actually changes
# ---------------------------------------------------------------------------

def test_residualized_selection_is_the_edge_coefficient_test():
    """One regression per edge, y ~ 1 + Z + edge: the reported effect size is
    the semipartial correlation and the p-value is the coefficient's."""
    X, y, Z = _data()
    r, p = _stat(selection_statistic='pearson', selection_input='residualized')
    r_ref, p_ref = correlations_and_pvalues(
        torch.as_tensor(X), torch.as_tensor(y),
        confounds=torch.as_tensor(Z), correlation_type='pearson')
    np.testing.assert_array_equal(r, r_ref.numpy())
    np.testing.assert_array_equal(p, p_ref.numpy())


def test_raw_selection_ignores_the_covariates():
    X, y, Z = _data()
    r_raw, p_raw = _stat(selection_statistic='pearson', selection_input='raw')
    r_ref, p_ref = correlations_and_pvalues(
        torch.as_tensor(X), torch.as_tensor(y), correlation_type='pearson')
    np.testing.assert_array_equal(r_raw, r_ref.numpy())
    np.testing.assert_array_equal(p_raw, p_ref.numpy())


def test_semipartial_and_partial_rank_edges_identically():
    """They are the same effect on two denominators, related by a constant
    within a fold. Which one is reported never changes the edge *ordering* --
    this is why reporting the semipartial costs nothing statistically."""
    from scipy.stats import spearmanr

    X, y, Z = _data(confound_beta=1.2)
    Xt, yt, Zt = (torch.as_tensor(a).double() for a in (X, y, Z))
    Xr = torch.as_tensor(get_residuals(Xt, Zt)).double()
    yr = torch.as_tensor(get_residuals(yt, Zt)).double()

    semipartial, _ = correlations_and_pvalues(Xr, yt, correlation_type='pearson')
    partial, _ = correlations_and_pvalues(Xr, yr, correlation_type='pearson')
    semipartial, partial = semipartial.numpy().ravel(), partial.numpy().ravel()

    assert spearmanr(semipartial, partial).statistic == pytest.approx(1.0)
    ratio = partial / semipartial
    assert np.abs(ratio - ratio.mean()).max() < 1e-9


@pytest.mark.parametrize("confound_beta", [0.0, 0.8, 1.6])
def test_threshold_means_what_it_says_whatever_the_confounding(confound_beta):
    """The reason for the coefficient test over a plain correlation on
    residualised edges: under the latter the effective alpha shrinks as
    confounding grows (measured 5.1% -> 3.3% -> 2.1%), so `p < 0.05` silently
    becomes a stricter threshold set by the data rather than by the user.
    """
    rng = np.random.RandomState(0)
    n, n_features, n_confounds = 400, 4000, 3
    Z = rng.randn(n, n_confounds)
    X = rng.randn(n, n_features)                      # pure noise edges
    y = Z[:, 0] * confound_beta + rng.randn(n)        # no real edge signal

    Xt, yt, Zt = (torch.as_tensor(a) for a in (X, y.reshape(-1, 1), Z))
    _, p = correlations_and_pvalues(Xt, yt, confounds=Zt, correlation_type='pearson')

    false_positive_rate = float((p.numpy() < 0.05).mean())
    assert 0.035 < false_positive_rate < 0.065, (
        f"nominal alpha 0.05, got {false_positive_rate:.4f} "
        f"at confound_beta={confound_beta}")


# ---------------------------------------------------------------------------
# The 2x2, end to end
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("selection_input", ["raw", "residualized"])
def test_one_run_reports_both_feature_cells(tmp_path, selection_input):
    """connectome and connectome_residualized are both defined on every run, so
    the F axis of the 2x2 needs no second run and no knob."""
    X, y, Z = _data(n=200)
    ue = UnivariateEdgeSelection(
        selection_statistic="pearson", selection_input=selection_input,
        edge_selection=[PThreshold(threshold=0.05, correction=[None])])
    cpm = CPMAnalysis(
        results_directory=str(tmp_path),
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
        edge_selection=ue, n_permutations=0, task_type="regression")
    cpm.run(X=X, y=y.ravel(), covariates=Z)

    ag = cpm.results_manager.agg_results
    for model in ("connectome", "connectome_residualized", "covariates",
                  "full", "increment"):
        value = np.ravel(ag.loc[(model, "both"), ("pearson_score", "mean")])
        assert np.isfinite(value).all(), f"{model} should be defined"

    # They are different models, not a relabelling of the same numbers.
    plain = np.ravel(ag.loc[("connectome", "both"), ("pearson_score", "mean")])[0]
    adjusted = np.ravel(
        ag.loc[("connectome_residualized", "both"), ("pearson_score", "mean")])[0]
    assert plain != pytest.approx(adjusted, abs=1e-9)


def test_selection_input_is_not_a_tuned_hyperparameter():
    """It is a design decision reported in the methods, not something the inner
    CV picks per fold, so it must not enter the parameter grid."""
    ue = UnivariateEdgeSelection(
        selection_statistic="pearson", selection_input="residualized",
        edge_selection=[PThreshold(threshold=[0.01, 0.05], correction=[None])])
    for params in ue.param_grid:
        assert 'selection_input' not in params
        assert not any('selection_input' in str(k) for k in params)
