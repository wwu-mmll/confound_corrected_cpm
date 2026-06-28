"""
Simulate multivariate predictors X, confounders Z, and a continuous outcome y.

Data-generating process (pure confounding, no mediation):
    Z ~ N(0, I_q)
    X = Z A + E
    y = X beta + Z gamma + eps

- A (q x p) encodes how Z loads into columns of X (Z -> X).
- beta (p,) encodes the direct effect of X on y (X -> y). Only a subset is nonzero.
- gamma (q,) encodes the confounding path Z -> y.
- E is column-correlated noise for X with optional AR(1) structure across columns.
- eps is outcome noise for y.

Author: (your name)
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
from numpy.random import Generator
from numpy.linalg import lstsq
from scipy.linalg import toeplitz


# ---------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------

@dataclass
class SimulationResult:
    """
    Container for simulated data and the generating parameters.

    Attributes
    ----------
    X : np.ndarray, shape (n, p)
        Predictor matrix.
    Z : np.ndarray, shape (n, q)
        Confounder matrix.
    y : np.ndarray, shape (n,)
        Continuous outcome.
    beta : np.ndarray, shape (p,)
        True direct effects of X on y (nonzero on a subset).
    gamma : np.ndarray, shape (q,)
        Effects of Z on y (confounding strength).
    A : np.ndarray, shape (q, p)
        Loadings from Z into X (confounding footprint inside X).
    info : dict
        Metadata: indices for signal features, which X columns are confounded,
        and all simulation hyperparameters used.
    """
    X: np.ndarray
    Z: np.ndarray
    y: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    A: np.ndarray
    info: Dict[str, Any]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def ar1_cov(p: int, rho: float, sigma2: float) -> np.ndarray:
    """
    Construct an AR(1) covariance matrix across p variables.

    Var(X_j) = sigma2
    Corr(X_j, X_{j+k}) = rho^k

    Parameters
    ----------
    p : int
        Number of variables (columns).
    rho : float
        AR(1) correlation parameter in [-1, 1].
    sigma2 : float
        Marginal variance for each variable.

    Returns
    -------
    np.ndarray, shape (p, p)
        AR(1) covariance matrix.
    """
    if abs(rho) < 1e-12:
        return np.eye(p) * sigma2
    first_col = (rho ** np.arange(p)) * sigma2
    return toeplitz(first_col)


def _rng(seed: Optional[int]) -> Generator:
    """Create a NumPy Generator with a fixed seed (if provided)."""
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------
# Core simulator
# ---------------------------------------------------------------------

def simulate_confounders(
    n_samples: int = 1000,           # number of samples
    n_features: int = 50,            # number of predictors (columns of X)
    n_confounds: int = 3,                      # number of confounders (columns of Z)
    n_features_with_signal: int = 10,            # number of X columns with nonzero beta
    n_confounded_features: int = 10,      # number of X columns influenced by Z (nonzero column in A)
    beta_scale: float = 1.0,         # SD for nonzero beta entries
    gamma_scale: float = 1.0,        # SD for entries of gamma (Z -> y strength)
    z_to_x_strength: float = 0.7,    # typical magnitude of Z -> X loadings (A)
    sigma_x: float = 1.0,            # SD for X-noise E
    sigma_y: float = 1.0,            # SD for outcome noise eps
    rho_x: float = 0.0,              # AR(1) correlation among columns of X-noise E
    random_state: Optional[int] = 42 # RNG seed for reproducibility
) -> SimulationResult:
    """
    Simulate (X, Z, y) with pure confounding (Z affects both X and y).

    Model
    -----
    Z ~ N(0, I_q)
    X = Z A + E,   E ~ N(0, Sigma_E) with AR(1) columns
    y = X beta + Z gamma + eps,   eps ~ N(0, sigma_y^2 I_n)

    Notes
    -----
    - Confounding arises when BOTH A != 0 (Z -> X) and gamma != 0 (Z -> y).
    - If gamma_scale=0, there is no confounding even if Z strongly influences X.
    - Only a subset of beta entries is nonzero (true signal features).

    Returns
    -------
    SimulationResult
        Data and parameters used to generate it, including indices of signal
        features and which X columns are confounded.
    """
    rng = _rng(random_state)

    # 1) Confounders: Z ~ N(0, I_q)
    Z = rng.normal(0.0, 1.0, size=(n_samples, n_confounds))

    # 2) Choose signal features and betas
    if not (1 <= n_features_with_signal <= n_features):
        raise ValueError(f"`num_signal` must be in [1, p]; got {n_features_with_signal} (p={n_features}).")
    k = n_features_with_signal

    signal_idx = np.arange(k)  # 0..k-1
    beta = np.zeros(n_features)
    beta[signal_idx] = rng.normal(0.0, beta_scale, size=k)

    # 3) Choose confounded features
    confounded_mask = np.zeros(n_features, dtype=bool)
    confounded_mask[:n_confounded_features] = True  # first m columns are confounded

    A = np.zeros((n_confounds, n_features))
    if confounded_mask.any():
        # Use scale / sqrt(q) so variance contribution is stable w.r.t. number of confounders
        A[:, confounded_mask] = rng.normal(
            loc=0.0,
            scale=z_to_x_strength / np.sqrt(n_confounds),
            size=(n_confounds, int(confounded_mask.sum()))
        )

    # 4) Column-correlated noise for X (optional AR(1) across columns)
    Sigma_E = ar1_cov(p=n_features, rho=rho_x, sigma2=sigma_x ** 2)
    E = rng.multivariate_normal(mean=np.zeros(n_features), cov=Sigma_E, size=n_samples)

    # 5) Build X
    #    Each column X_j is a linear combination of Z (via A) plus noise E_j.
    X = Z @ A + E

    # 6) Outcome y: direct X->y via beta, plus confounding Z->y via gamma
    gamma = rng.normal(loc=0.0, scale=gamma_scale, size=n_confounds)
    eps_y = rng.normal(loc=0.0, scale=sigma_y, size=n_samples)
    y = X @ beta + Z @ gamma + eps_y

    # 7) Metadata for inspection and downstream benchmarking
    info: Dict[str, Any] = {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_confounds": n_confounds,
        "n_features_with_signal": n_features_with_signal,
        "signal_idx": signal_idx,                # indices where beta != 0
        "confounded_mask": confounded_mask,      # columns of X with nonzero A
        "n_confounded_features": n_confounded_features,
        "beta_scale": beta_scale,
        "gamma_scale": gamma_scale,
        "z_to_x_strength": z_to_x_strength,
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
        "rho_x": rho_x,
        "random_state": random_state,
    }

    return SimulationResult(X=X, Z=Z, y=y, beta=beta, gamma=gamma, A=A, info=info)


# ---------------------------------------------------------------------
# (Optional) tiny extras that are often handy while testing
# ---------------------------------------------------------------------

def fit_ols(y: np.ndarray, X: np.ndarray, add_intercept: bool = True) -> np.ndarray:
    """
    Minimal OLS via least squares (useful sanity check; no regularization).

    Returns
    -------
    np.ndarray
        If add_intercept=True: [intercept, betas...]
        Else: betas
    """
    if add_intercept:
        X1 = np.column_stack([np.ones(X.shape[0]), X])
        coef, *_ = lstsq(X1, y, rcond=None)
        return coef
    coef, *_ = lstsq(X, y, rcond=None)
    return coef


def residualize_against_Z(M: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Remove linear effects of Z from M (works for vector y or matrix X).

    Computes residuals from OLS projection:
        M_res = M - Proj_{[1,Z]}(M)

    Parameters
    ----------
    M : np.ndarray
        y (1D) or X (2D) to be residualized.
    Z : np.ndarray, shape (n, q)
        Confounders to regress out (intercept is added internally).

    Returns
    -------
    np.ndarray
        Residuals with the same shape as M.
    """
    n = Z.shape[0]
    Z1 = np.column_stack([np.ones(n), Z])

    if M.ndim == 1:
        coef, *_ = lstsq(Z1, M, rcond=None)
        return M - Z1 @ coef

    if M.ndim == 2:
        coef, *_ = lstsq(Z1, M, rcond=None)  # solves for all columns at once
        return M - Z1 @ coef

    raise ValueError("M must be a 1D vector (y) or a 2D matrix (X).")

