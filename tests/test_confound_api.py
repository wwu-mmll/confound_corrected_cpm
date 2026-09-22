"""
The confound-control API: selection_statistic x selection_input.

Four apparent confound levers in CCCPM reduce to two independent choices:

  S  does edge *selection* control for the confounds?   -> selection_input
  F  is the *connectome* deconfounded before the model? -> model_input

Both are run-level choices, and F has to be: OLS is invariant to it once the
covariates are in the design, but a tree, forest or GAM is not, so it cannot be
a model name without meaning something different for each backend.

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
# The removed 0.6.x spellings must fail loudly, with the migration in the message
# ---------------------------------------------------------------------------

# Only the values that no longer mean anything as a `selection_statistic`.
# 'pearson' and 'spearman' were `edge_statistic` values too and carry over
# unchanged, so they must keep working rather than raise.
REMOVED = [
    ('point_biserial',         'pearson',  'raw'),
    ('pearson_partial',        'pearson',  'residualized'),
    ('spearman_partial',       'spearman', 'residualized'),
    ('point_biserial_partial', 'pearson',  'residualized'),
]


@pytest.mark.parametrize("removed,statistic,selection_input", REMOVED)
def test_removed_edge_statistic_values_name_their_replacement(
        removed, statistic, selection_input):
    """`edge_statistic` is gone rather than deprecated: 'pearson' etc. were
    numerically identical to the new spelling, but `calculate_residuals` and the
    `*_partial` values changed *meaning*, and code that keeps running while
    quietly producing different numbers is worse than code that stops."""
    with pytest.raises(ValueError) as excinfo:
        EdgeStatistic(selection_statistic=removed)
    message = str(excinfo.value)
    assert f"selection_statistic={statistic!r}" in message
    assert f"selection_input={selection_input!r}" in message
    if selection_input == 'residualized':
        assert "coefficient test" in message, (
            "migrating a *_partial value must warn that edge sets differ")


@pytest.mark.parametrize("carried_over", ["pearson", "spearman"])
def test_values_that_carry_over_still_work(carried_over):
    """'pearson'/'spearman' meant the same thing under the old name; they are
    valid selection_statistic values and must not be mistaken for removals."""
    EdgeStatistic(selection_statistic=carried_over)


def test_removed_keyword_arguments_are_gone():
    """`edge_statistic` is not accepted at all -- no silent pass-through."""
    with pytest.raises(TypeError, match="edge_statistic"):
        EdgeStatistic(edge_statistic='pearson')
    with pytest.raises(TypeError, match="edge_statistic"):
        UnivariateEdgeSelection(edge_statistic='pearson')


def test_calculate_residuals_is_gone(tmp_path):
    ue = UnivariateEdgeSelection(
        selection_statistic="pearson",
        edge_selection=[PThreshold(threshold=0.05, correction=[None])])
    with pytest.raises(TypeError):
        CPMAnalysis(results_directory=str(tmp_path), cv=KFold(n_splits=3),
                    edge_selection=ue, calculate_residuals=True, n_permutations=0)


def test_unknown_values_are_rejected():
    with pytest.raises(ValueError, match="selection_statistic must be one of"):
        EdgeStatistic(selection_statistic='kendall')
    with pytest.raises(ValueError, match="selection_input must be one of"):
        EdgeStatistic(selection_input='deconfounded')


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

def _run(tmp_path, selection_input, model_input, n=200):
    X, y, Z = _data(n=n)
    ue = UnivariateEdgeSelection(
        selection_statistic="pearson", selection_input=selection_input,
        edge_selection=[PThreshold(threshold=0.05, correction=[None])])
    cpm = CPMAnalysis(
        results_directory=str(tmp_path),
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
        edge_selection=ue, model_input=model_input,
        n_permutations=0, task_type="regression")
    cpm.run(X=X, y=y.ravel(), covariates=Z)
    ag = cpm.results_manager.agg_results
    # explained_variance rather than pearson: `increment` is deliberately NaN
    # for Pearson r, since a difference of correlations is not a statistic.
    return {m: float(np.ravel(
                ag.loc[(m, "both"), ("explained_variance_score", "mean")])[0])
            for m in ("connectome", "covariates", "full", "increment")}


@pytest.mark.parametrize("selection_input", ["raw", "residualized"])
@pytest.mark.parametrize("model_input", ["raw", "residualized"])
def test_every_cell_of_the_2x2_runs(tmp_path, selection_input, model_input):
    """Each cell is one run reporting one of each model -- no ambiguity about
    which connectome produced `full`."""
    out = _run(tmp_path, selection_input, model_input)
    for model, value in out.items():
        assert np.isfinite(value), f"{model} should be defined"


def test_model_input_changes_the_connectome_model(tmp_path):
    raw = _run(tmp_path, "residualized", "raw")
    res = _run(tmp_path, "residualized", "residualized")
    assert raw["connectome"] != pytest.approx(res["connectome"], abs=1e-6)


def test_model_input_leaves_full_and_covariates_alone_for_the_linear_model(tmp_path):
    """OLS is invariant once the covariates are in the design: residualising the
    connectome moves variance from the strength column into the Z columns, which
    are already there, so the span -- and the fit -- is unchanged.

    This is a property of the linear solver, NOT of CPM. The non-linear backends
    are not invariant to it, which is exactly why model_input is a run-level
    choice rather than a model name.
    """
    raw = _run(tmp_path, "residualized", "raw")
    res = _run(tmp_path, "residualized", "residualized")
    assert raw["full"] == pytest.approx(res["full"], abs=1e-4)
    assert raw["covariates"] == pytest.approx(res["covariates"], abs=1e-4)
    assert raw["increment"] == pytest.approx(res["increment"], abs=1e-4)


def test_selection_input_is_not_a_tuned_hyperparameter():
    """It is a design decision reported in the methods, not something the inner
    CV picks per fold, so it must not enter the parameter grid."""
    ue = UnivariateEdgeSelection(
        selection_statistic="pearson", selection_input="residualized",
        edge_selection=[PThreshold(threshold=[0.01, 0.05], correction=[None])])
    for params in ue.param_grid:
        assert 'selection_input' not in params
        assert not any('selection_input' in str(k) for k in params)


def test_nonlinear_models_are_not_invariant_to_model_input():
    """The reason model_input cannot be a model name.

    OLS absorbs a shift of the connectome column into the covariate columns, so
    `full` is unchanged. A tree splits on `s <= t`, and `s_resid <= t` is a
    different partition -- there is no coefficient to absorb the shift. Measured
    below: the same edges, the same data, `full` moving by a large fraction of
    sd(y) for every non-linear backend and by nothing for the linear one.

    If this ever passed for the non-linear models, `connectome_residualized`
    could go back to being a model variant. It does not.
    """
    from cccpm.models.linear_model import LinearCPM
    from cccpm.models.nonlinear_models import (DecisionTreeCPM, GAMCPM,
                                               RandomForestCPM)
    from cccpm.constants import Models, Networks
    from cccpm.preprocessing import residualize_train_test

    rng = np.random.RandomState(2)
    n, n_features, n_confounds, ntr = 400, 200, 3, 260
    X = rng.randn(n, n_features)
    Z = rng.randn(n, n_confounds)
    y = X[:, :20].sum(1) * 0.1 + Z[:, 0] * 0.9 + rng.randn(n)

    Xtr, Xte = torch.as_tensor(X[:ntr]).float(), torch.as_tensor(X[ntr:]).float()
    Ztr, Zte = torch.as_tensor(Z[:ntr]).float(), torch.as_tensor(Z[ntr:]).float()
    ytr = torch.as_tensor(y[:ntr].reshape(-1, 1)).float()

    edges = torch.zeros(n_features, 2, 1, dtype=torch.bool)
    edges[:20, Networks.positive, 0] = True
    edges[20:40, Networks.negative, 0] = True
    Xtr_r, Xte_r = residualize_train_test(Xtr, Xte, Ztr, Zte)

    def full_gap(cls):
        a = cls(edges=edges, device='cpu').fit(Xtr, ytr, Ztr).predict(Xte, Zte)
        b = cls(edges=edges, device='cpu').fit(Xtr_r, ytr, Ztr).predict(Xte_r, Zte)
        a, b = np.asarray(a)[:, Models.full], np.asarray(b)[:, Models.full]
        return np.abs(a - b).max() / y.std()

    assert full_gap(LinearCPM) < 1e-4, "OLS must be invariant"
    for cls in (DecisionTreeCPM, RandomForestCPM, GAMCPM):
        gap = full_gap(cls)
        assert gap > 0.05, (
            f"{cls.__name__} looks invariant to model_input (gap {gap:.1%} of "
            f"sd(y)); if that is real, this design decision should be revisited")
