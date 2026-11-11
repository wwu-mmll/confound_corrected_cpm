"""Edge statistics checked against external references: statsmodels' GLM
(partial path), scipy's pointbiserialr (binary path), OLS residuals from
the definition, scipy's t distribution. Policy layer: test_edge_selection.py."""
import numpy as np
import pytest
import torch
import statsmodels.api as sm
from scipy.stats import pointbiserialr
from scipy.stats import t as t_dist

from cccpm.statistics import (
    correlations_and_pvalues,
    get_residuals,
    critical_t,
    t_pvalues,
)


def test_partial_path_matches_glm_coefficient(simulated_data):
    """Per edge, the p-value from `correlations_and_pvalues` with confounds
    must equal the p-value of the edge coefficient in `y ~ 1 + confounds +
    edge` (statsmodels OLS); the reported r is the semi-partial correlation
    (confound removed from the edge only) and its sign must match."""
    X, y, covariates = simulated_data
    Xt = torch.as_tensor(X, dtype=torch.float64)
    yt = torch.as_tensor(y, dtype=torch.float64).reshape(-1, 1)
    cov = covariates.astype(np.float64)

    r, p = correlations_and_pvalues(Xt, yt, correlation_type='pearson',
                                    confounds=torch.as_tensor(cov))
    r = r.numpy().ravel()
    p = p.numpy().ravel()

    Z = sm.add_constant(cov)
    Pz = Z @ np.linalg.pinv(Z)          # confound hat matrix
    for i in range(X.shape[1]):
        m = sm.OLS(y, sm.add_constant(np.column_stack([cov, X[:, i]]))).fit()
        np.testing.assert_allclose(p[i], m.pvalues[-1], rtol=1e-7, atol=1e-12)
        x_res = X[:, i] - Pz @ X[:, i]
        yc = y - y.mean()
        sr = np.dot(x_res, yc) / (np.linalg.norm(x_res) * np.linalg.norm(yc))
        np.testing.assert_allclose(r[i], sr, atol=1e-6)
        assert np.sign(r[i]) == np.sign(m.params[-1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_partial_edge_selection_runs_on_gpu():
    """Regression test: get_residuals built the intercept column on the CPU,
    so torch.cat(ones, confounds) crashed with confounds on cuda."""
    dev = torch.device('cuda')
    rng = np.random.RandomState(0)
    X = torch.as_tensor(rng.randn(80, 30), dtype=torch.float32, device=dev)
    y = torch.as_tensor(rng.randn(80, 1), dtype=torch.float32, device=dev)
    Z = torch.as_tensor(rng.randn(80, 2), dtype=torch.float32, device=dev)

    r, p = correlations_and_pvalues(X, y, correlation_type='pearson', confounds=Z)
    assert r.device.type == 'cuda' and p.device.type == 'cuda'

    r_cpu, p_cpu = correlations_and_pvalues(
        X.cpu(), y.cpu(), correlation_type='pearson', confounds=Z.cpu())
    torch.testing.assert_close(r.cpu(), r_cpu, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(p.cpu(), p_cpu, rtol=1e-4, atol=1e-5)


def test_get_residuals_matches_ols():
    """get_residuals must reproduce OLS-with-intercept residuals for both
    data orientations: [N_samples, features] and [batch, N_samples]."""
    rng = np.random.RandomState(7)
    n, k = 100, 3
    Z = rng.randn(n, k)
    Zi = np.column_stack([np.ones(n), Z])

    # [N_samples, features]
    data_A = rng.randn(n, 8)
    beta = np.linalg.lstsq(Zi, data_A, rcond=None)[0]
    expected_A = data_A - Zi @ beta
    got_A = get_residuals(data_A, Z)
    np.testing.assert_allclose(got_A, expected_A, atol=1e-9)

    # [batch, N_samples] (e.g. permuted targets)
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
    """Point-biserial = Pearson vs a 0/1 target, which is what the unified OLS
    path computes with no special-casing. Regression test: the old group-mean
    formula used the pooled within-group SD (not total SD of X) as the
    denominator, inflating |r| -- with imbalance + strong separation this hit
    the clamp (1.0) where scipy reports ~0.78."""
    rng = np.random.RandomState(seed)
    y = np.array([1] * n1 + [0] * (n - n1)).astype(np.float64)
    rng.shuffle(y)
    X = (y[:, None] * rng.uniform(0, 3, 6) + rng.randn(n, 6)).astype(np.float64)

    r, _ = correlations_and_pvalues(torch.as_tensor(X), torch.as_tensor(y).reshape(-1, 1),
                                    correlation_type='pearson')
    r = r.numpy().ravel()

    scipy_r = np.array([pointbiserialr(y, X[:, i])[0] for i in range(X.shape[1])])
    np.testing.assert_allclose(r, scipy_r, atol=1e-6)


@pytest.mark.parametrize("df", [5, 28, 120, 998])
def test_critical_t_selects_exactly_what_the_exact_p_value_does(df):
    """`|t| > critical_t(alpha, df)` must equal `p < alpha` for the exact p --
    this is what lets the pipeline threshold on-device without a per-edge
    p-value. Reference is scipy's t tail, independent of both."""
    rng = np.random.RandomState(df)
    t = torch.as_tensor(rng.standard_t(df, size=5000) * 2.0)
    p_exact = 2 * t_dist.sf(np.abs(t.numpy()), df)
    np.testing.assert_allclose(t_pvalues(t, df).numpy(), p_exact, rtol=1e-10, atol=0)

    for alpha in (0.1, 0.05, 0.01, 0.001, 1e-6):
        selected = (t.abs() > critical_t(alpha, df)).numpy()
        np.testing.assert_array_equal(selected, p_exact < alpha)
