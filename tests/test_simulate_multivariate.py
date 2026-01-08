import numpy as np
import pytest

from cccpm.simulation.simulate_multivariate import (
    SimulationResult,
    ar1_cov,
    simulate_confounders,
    fit_ols,
    residualize_against_Z,
)


def test_ar1_cov_identity_when_rho_zero():
    p = 5
    sigma2 = 2.0
    cov = ar1_cov(p, rho=0.0, sigma2=sigma2)

    assert cov.shape == (p, p)
    assert np.allclose(cov, np.eye(p) * sigma2)


def test_ar1_cov_structure():
    p = 4
    rho = 0.5
    sigma2 = 1.0
    cov = ar1_cov(p, rho, sigma2)

    assert np.isclose(cov[0, 1], rho * sigma2)
    assert np.isclose(cov[0, 2], rho**2 * sigma2)
    assert np.allclose(cov, cov.T)


def test_simulate_confounders_shapes():
    res = simulate_confounders(
        n_samples=200,
        n_features=20,
        n_confounds=3,
        n_features_with_signal=5,
        n_confounded_features=7,
        random_state=0,
    )

    assert isinstance(res, SimulationResult)
    assert res.X.shape == (200, 20)
    assert res.Z.shape == (200, 3)
    assert res.y.shape == (200,)
    assert res.beta.shape == (20,)
    assert res.gamma.shape == (3,)
    assert res.A.shape == (3, 20)


def test_simulate_confounders_reproducible():
    res1 = simulate_confounders(random_state=123)
    res2 = simulate_confounders(random_state=123)

    assert np.allclose(res1.X, res2.X)
    assert np.allclose(res1.y, res2.y)
    assert np.allclose(res1.beta, res2.beta)


def test_signal_features_have_nonzero_beta():
    res = simulate_confounders(
        n_features=20,
        n_features_with_signal=6,
        random_state=1,
    )

    signal_idx = res.info["signal_idx"]
    assert np.all(res.beta[signal_idx] != 0.0)
    assert np.all(res.beta[len(signal_idx):] == 0.0)


def test_confounded_features_correlate_with_Z():
    res = simulate_confounders(
        n_samples=2000,
        n_features=30,
        n_confounded_features=10,
        gamma_scale=0.0,   # Z -> y off, but Z -> X on
        random_state=2,
    )

    Z = res.Z[:, 0]  # first confound
    X = res.X

    corrs = np.corrcoef(Z, X.T)[0, 1:]
    assert np.mean(np.abs(corrs[:10])) > np.mean(np.abs(corrs[10:]))


def test_residualization_removes_yz_dependence():
    res = simulate_confounders(
        n_samples=5000,
        n_confounds=3,
        gamma_scale=2.0,
        random_state=3,
    )

    y_res = residualize_against_Z(res.y, res.Z)

    corr_before = np.corrcoef(res.y, res.Z[:, 0])[0, 1]
    corr_after = np.corrcoef(y_res, res.Z[:, 0])[0, 1]

    assert abs(corr_after) < abs(corr_before)
    assert abs(corr_after) < 0.05


def test_gamma_zero_removes_direct_effect():
    res = simulate_confounders(
        n_samples=4000,
        gamma_scale=0.0,
        random_state=4,
    )

    y_res = residualize_against_Z(res.y, res.Z)
    corr = np.corrcoef(y_res, res.Z[:, 0])[0, 1]

    assert abs(corr) < 0.05


def test_ar1_structure_in_X_noise():
    res = simulate_confounders(
        n_samples=3000,
        rho_x=0.7,
        n_confounded_features=0,  # isolate noise structure
        gamma_scale=0.0,
        random_state=5,
    )

    corr = np.corrcoef(res.X.T)
    assert corr[0, 1] > corr[0, 2] > corr[0, 3]


def test_fit_ols_recovers_simple_linear_model():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 2))
    beta = np.array([1.0, -2.0])
    y = X @ beta + rng.normal(scale=0.1, size=500)

    coef = fit_ols(y, X)

    assert np.allclose(coef[1:], beta, atol=0.1)


def test_residualize_against_Z_removes_correlation():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(500, 2))
    y = 3 * Z[:, 0] + rng.normal(size=500)

    y_res = residualize_against_Z(y, Z)

    corr = np.corrcoef(y_res, Z[:, 0])[0, 1]
    assert abs(corr) < 0.05


def test_residualize_invalid_dim():
    Z = np.random.randn(10, 2)
    M = np.random.randn(10, 2, 2)

    with pytest.raises(ValueError):
        residualize_against_Z(M, Z)
