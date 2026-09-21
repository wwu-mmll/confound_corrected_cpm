"""
Tests for the edge statistics in cccpm.statistics.

Every check here is against an external reference -- statsmodels' GLM for the
partial path, scipy's pointbiserialr for the binary path, OLS residuals computed
from the definition, statsmodels' multipletests for the correction. The
policy layer built on top of these lives in test_edge_selection.py.
"""
import numpy as np
import pytest
import torch
import statsmodels.api as sm
from scipy.stats import pointbiserialr

from cccpm.statistics import (
    correlations_and_pvalues,
    get_residuals,
    torch_bonferroni,
)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_partial_edge_selection_runs_on_gpu():
    """The confound-controlled path must run when X/y/confounds live on the GPU.

    Regression test for a device mismatch: get_residuals built the intercept
    column on the CPU, so torch.cat(ones, confounds) crashed with confounds on
    cuda. GPU results must also match the CPU computation.
    """
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


def test_torch_bonferroni_matches_statsmodels():
    """torch_bonferroni must reproduce statsmodels' bonferroni-corrected p-values
    and reject mask exactly (it's just min(p * n, 1))."""
    from statsmodels.stats import multitest

    rng = np.random.RandomState(3)
    p = rng.uniform(0, 1, size=(50, 4)).astype(np.float64)

    reject_sm, p_corrected_sm, _, _ = multitest.multipletests(
        p.flatten(), alpha=0.05, method='bonferroni')

    reject, p_corrected = torch_bonferroni(torch.as_tensor(p), alpha=0.05)

    np.testing.assert_allclose(p_corrected.numpy().flatten(), p_corrected_sm, atol=1e-12)
    np.testing.assert_array_equal(reject.numpy().flatten(), reject_sm)
