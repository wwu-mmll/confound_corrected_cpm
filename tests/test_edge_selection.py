import numpy as np
import pandas as pd
import pytest
import torch
import statsmodels.api as sm
from scipy.stats import pointbiserialr
from cccpm.edge_selection import (
    correlations_and_pvalues,
    get_residuals,
    EdgeStatistic,
    UnivariateEdgeSelection,
    PThreshold,
)
from cccpm.constants import Networks


def test_partial_path_matches_glm_coefficient(simulated_data):
    """The confound-controlled edge selection must reproduce the OLS GLM.

    For every edge, the p-value returned by ``correlations_and_pvalues`` with
    confounds must equal the p-value of the edge coefficient in the regression
    ``y ~ intercept + confounds + edge`` (statsmodels OLS). The reported r must
    be the *semi-partial* correlation (confound removed from the edge only),
    and its sign must match the regression coefficient. P-values use a normal
    approximation to the t-tail (Open decision #6), so they are compared at a
    looser tolerance than the (exact) statsmodels values.
    """
    X, y, covariates = simulated_data
    Xt = torch.as_tensor(X, dtype=torch.float64)
    yt = torch.as_tensor(y, dtype=torch.float64).reshape(-1, 1)
    cov = covariates.astype(np.float64)

    r, p = correlations_and_pvalues(Xt, yt, correlation_type='pearson',
                                    confounds=torch.as_tensor(cov))
    r = r.numpy().ravel()
    p = p.numpy().ravel()

    n = X.shape[0]
    Z = sm.add_constant(cov)
    Pz = Z @ np.linalg.pinv(Z)          # confound hat matrix
    for i in range(X.shape[1]):
        m = sm.OLS(y, sm.add_constant(np.column_stack([cov, X[:, i]]))).fit()
        # GLM coefficient p-value (exact). Our p uses the normal-tail approx.
        np.testing.assert_allclose(p[i], m.pvalues[-1], atol=2e-3)
        # reported r is the semi-partial correlation: corr(raw y, residualised edge)
        x_res = X[:, i] - Pz @ X[:, i]
        yc = y - y.mean()
        sr = np.dot(x_res, yc) / (np.linalg.norm(x_res) * np.linalg.norm(yc))
        np.testing.assert_allclose(r[i], sr, atol=1e-6)
        assert np.sign(r[i]) == np.sign(m.params[-1])


def test_get_residuals_matches_ols():
    """get_residuals must reproduce OLS-with-intercept residuals for both
    data orientations: [N_samples, features] and [batch, N_samples]."""
    rng = np.random.RandomState(7)
    n, k = 100, 3
    Z = rng.randn(n, k)
    Zi = np.column_stack([np.ones(n), Z])               # intercept + confounds

    # Orientation A: data is [N_samples, features]
    data_A = rng.randn(n, 8)
    beta = np.linalg.lstsq(Zi, data_A, rcond=None)[0]
    expected_A = data_A - Zi @ beta
    got_A = get_residuals(data_A, Z)
    np.testing.assert_allclose(got_A, expected_A, atol=1e-9)

    # Orientation B: data is [batch, N_samples] (e.g. permuted targets)
    data_B = rng.randn(5, n)
    beta_B = np.linalg.lstsq(Zi, data_B.T, rcond=None)[0]
    expected_B = (data_B.T - Zi @ beta_B).T
    got_B = get_residuals(data_B, Z)
    np.testing.assert_allclose(got_B, expected_B, atol=1e-9)

    # Residuals must be orthogonal to the confound space (incl. the intercept).
    np.testing.assert_allclose(Zi.T @ got_A, 0.0, atol=1e-8)


@pytest.mark.parametrize("seed,n,n1", [
    (1, 80, 20),    # imbalanced groups
    (2, 60, 30),    # balanced
    (3, 200, 40),   # larger, imbalanced
    (4, 40, 8),     # small n, strong imbalance
])
def test_point_biserial_matches_scipy(seed, n, n1):
    """A binary 0/1 target through the unified OLS path must equal point-biserial.

    Point-biserial correlation is Pearson against a 0/1 target, which is exactly
    what the OLS path computes for a binary outcome (no special-casing).
    Regression test for a bug where the old group-mean formula used the *pooled
    within-group* SD as the denominator (instead of the total SD of X), which
    inflated |r| — with imbalanced groups and strong separation it drove r to
    the clamp (1.0) where scipy reports ~0.78.
    """
    rng = np.random.RandomState(seed)
    y = np.array([1] * n1 + [0] * (n - n1)).astype(np.float64)
    rng.shuffle(y)
    # features with varying (incl. strong) association with the binary target
    X = (y[:, None] * rng.uniform(0, 3, 6) + rng.randn(n, 6)).astype(np.float64)

    r, _ = correlations_and_pvalues(torch.as_tensor(X), torch.as_tensor(y).reshape(-1, 1),
                                    correlation_type='pearson')
    r = r.numpy().ravel()

    scipy_r = np.array([pointbiserialr(y, X[:, i])[0] for i in range(X.shape[1])])
    np.testing.assert_allclose(r, scipy_r, atol=1e-6)


@pytest.mark.parametrize("statistic", [
    "pearson", "spearman", "pearson_partial", "spearman_partial",
])
def test_batched_edge_statistics_match_per_column(statistic):
    """Computing (r, p) for all target columns at once must equal computing
    each column separately (up to float32 rounding).

    The pipeline batches edge-statistic computation over all runs/permutations
    in a single call (cpm_analysis._select_edges), relying on the fact that the
    statistic for one target column does not depend on the others. This guards
    that assumption: a regression here would change which edges get selected
    across permutations. The two paths are not bit-identical because a batched
    matrix-matrix product accumulates in a different order than the per-column
    matrix-vector product, but they agree to float32 precision (which does not
    change the downstream p<threshold edge selection in practice — this is also
    exactly what the inner-CV path has always computed).
    """
    rng = np.random.RandomState(0)
    n_samples, n_features, n_runs = 90, 15, 8
    X = rng.randn(n_samples, n_features).astype(np.float32)
    Y = rng.randn(n_samples, n_runs).astype(np.float32)
    covariates = rng.randn(n_samples, 2).astype(np.float32)

    stat = EdgeStatistic(edge_statistic=statistic)
    device = torch.device('cpu')

    # Batched: all columns at once.
    r_all, p_all = stat.fit_transform(X=X, y=Y, covariates=covariates, device=device)

    # Per column, exactly as the old loop did.
    for run_id in range(n_runs):
        r_col, p_col = stat.fit_transform(
            X=X, y=Y[:, [run_id]], covariates=covariates, device=device)
        torch.testing.assert_close(r_all[:, [run_id]], r_col, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(p_all[:, [run_id]], p_col, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("statistic,binary_target", [
    ("pearson", False),
    ("spearman", False),
    ("pearson_partial", False),
    ("spearman_partial", False),
    ("point_biserial", True),
    ("point_biserial_partial", True),
])
def test_edge_selection_recovers_signed_edges(statistic, binary_target):
    """
    End-to-end test of the production edge-selection path
    (UnivariateEdgeSelection -> EdgeStatistic dispatch -> PThreshold.select)
    for every supported edge statistic: a strongly positive edge must be
    selected as positive, a strongly negative edge as negative, and the sign
    must not be confused.
    """
    rng = np.random.RandomState(42)
    n_samples, n_features = 200, 10
    X = rng.randn(n_samples, n_features).astype(np.float32)
    # feature 0 drives y up, feature 1 drives y down
    signal = X[:, 0] * 2.0 - X[:, 1] * 2.0 + rng.randn(n_samples) * 0.3
    if binary_target:
        y = (signal > np.median(signal)).astype(np.float32).reshape(-1, 1)
    else:
        y = signal.astype(np.float32).reshape(-1, 1)
    covariates = rng.randn(n_samples, 1).astype(np.float32)

    sel = UnivariateEdgeSelection(
        edge_statistic=statistic,
        edge_selection=[PThreshold(threshold=[0.01], correction=[None])],
    )
    # Configure as a single selector, exactly as the pipeline does
    sel.set_params(**list(sel.param_grid)[0])

    edges = sel.fit_transform(X=X, y=y, covariates=covariates).return_selected_edges()
    # edges shape: [n_features, 2, n_runs]
    assert bool(edges[0, Networks.positive, 0]), f"{statistic}: feature 0 should be positive"
    assert bool(edges[1, Networks.negative, 0]), f"{statistic}: feature 1 should be negative"
    assert not bool(edges[0, Networks.negative, 0]), f"{statistic}: feature 0 should not be negative"
    assert not bool(edges[1, Networks.positive, 0]), f"{statistic}: feature 1 should not be positive"
