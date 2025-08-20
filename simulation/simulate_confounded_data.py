import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr



def partial_correlation(x, y, z):
    x_resid = x - LinearRegression().fit(z.reshape(-1, 1), x).predict(z.reshape(-1, 1))
    y_resid = y - LinearRegression().fit(z.reshape(-1, 1), y).predict(z.reshape(-1, 1))
    return pearsonr(x_resid, y_resid)[0], x_resid, y_resid


def simulate_x_with_target_corr(n, alpha, beta, gamma, sigma_z=1.0, rho_target=0.5):
    y = np.random.normal(0, 1, n)
    eps_z = np.random.normal(0, sigma_z, n)
    z = gamma * y + eps_z

    numerator = alpha + beta * gamma
    denom_squared = (numerator / rho_target) ** 2
    signal_variance = alpha**2 + beta**2 * (gamma**2 + sigma_z**2)
    sigma_x_squared = denom_squared - signal_variance

    if sigma_x_squared < 0:
        raise ValueError("Cannot reach target correlation with given parameters.")

    sigma_x = np.sqrt(sigma_x_squared)
    eps_x = np.random.normal(0, sigma_x, n)

    signal = alpha * y + beta * z
    x = signal + eps_x
    x = (x - np.mean(x)) / np.std(x)

    return x, y, z, signal


def build_multivariate_features_from_vector(vector, n_features=105, n_signal_features=15, z=None,
                                            n_additional_confounded_features=0):
    n = len(vector)
    X = np.zeros((n, n_features))

    # Normalize scalar to unit variance
    vector = (vector - np.mean(vector)) / np.std(vector)

    for i in range(n_signal_features):
        noise = np.random.normal(0, 1, n)
        signal_component = np.sqrt(0.5) * vector
        noise_component = np.sqrt(0.5) * noise
        if i % 2 == 0:
            X[:, i] = signal_component + noise_component
        else:
            X[:, i] = -signal_component + noise_component

    if z is not None:
        z = (z - np.mean(z)) / np.std(z)

        for i in range(n_additional_confounded_features):
            noise = np.random.normal(0, 1, n)
            z_signal_component = np.sqrt(0.5) * z
            noise_component = np.sqrt(0.5) * noise
            if i % 2 == 0:
                X[:, i + n_signal_features] = z_signal_component + noise_component
            else:
                X[:, i + n_signal_features] = -z_signal_component + noise_component

    for i in range(n_signal_features+n_additional_confounded_features, n_features):
        X[:, i] = np.random.normal(0, 1, n)

    return X


def residualize(u, V):
    """Project u to be orthogonal to the columns of V (both should be 1D/2D arrays)."""
    if V is None:
        return u
    V = np.atleast_2d(V)
    if V.shape[0] != u.shape[0]:
        V = V.T  # ensure shape (n, k)
    # Gram-Schmidt via normal equations: u - V(V^+ u)
    # Use lstsq for numerical stability.
    coef, *_ = np.linalg.lstsq(V, u, rcond=None)
    return u - V @ coef


def build_multivariate_features_from_vector2(
    vector, n_features=105, n_signal_features=15, z=None,
    n_additional_confounded_features=0, signal_var=0.5, rng=None,
    orth_to=('signal',),  # options: 'signal', 'z', 'both'
):
    rng = np.random.default_rng() if rng is None else rng
    n = len(vector)
    X = np.zeros((n, n_features))

    v = (vector - np.mean(vector)) / np.std(vector)
    sv = np.sqrt(signal_var)
    nv = np.sqrt(1 - signal_var)

    # X-block (features carrying v)
    for i in range(n_signal_features):
        e = rng.normal(0, 1, n)
        if 'signal' in orth_to or 'both' in orth_to:
            e = residualize(e, v)
        if 'z' in orth_to or 'both' in orth_to:
            if z is not None:
                z0 = (z - np.mean(z)) / np.std(z)
                e = residualize(e, z0)
        e = e / np.std(e)
        sgn = 1 if (i % 2 == 0) else -1
        X[:, i] = sgn * sv * v + nv * e

    # Z-block (features carrying z)
    if z is not None and n_additional_confounded_features > 0:
        z0 = (z - np.mean(z)) / np.std(z)
        start = n_signal_features
        for j in range(n_additional_confounded_features):
            e = rng.normal(0, 1, n)
            e = residualize(e, z0)  # key line: noise ⟂ z
            e = e / np.std(e)
            sgn = 1 if (j % 2 == 0) else -1
            X[:, start + j] = sgn * sv * z0 + nv * e

    # Fill remaining with pure noise
    filled = n_signal_features + (n_additional_confounded_features if z is not None else 0)
    for i in range(filled, n_features):
        X[:, i] = rng.normal(0, 1, n)

    return X


import numpy as np
from typing import Tuple, Sequence, Union

Number = Union[int, float, np.floating]

def gen_latent_sem(
    n: int,
    p: int,
    q: int,
    rho: float = 0.6,         # target latent corr(x, y) = a*b + c'
    m: float = 0.7,           # mediated share = (a*b) / (a*b + c')
    a: float = 0.5,           # path z <- a*x  (keep |a| < 1)
    load_x: Union[Number, Sequence[float]] = 0.8,  # X-indicator loadings
    load_z: Union[Number, Sequence[float]] = 0.9,  # Z-indicator loadings
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate multivariate indicators for latent x (p indicators) and z (q indicators),
    and a single observed outcome y, such that:
        z = a * x + e_z
        y = b * z + c' * x + e_y
    with Var(x) = Var(z) = Var(y) = 1 in expectation, and
        rho_target = corr(x, y) = a*b + c'    and    m = (a*b) / (a*b + c').

    Returns
    -------
    X_obs : (n, p) ndarray
        Observed indicators of latent x.
    y_obs : (n,) ndarray
        Univariate outcome.
    Z_obs : (n, q) ndarray
        Observed indicators of latent z.
    x_latent : (n,) ndarray
        True latent x used to generate X_obs and Z_obs.
    z_latent : (n,) ndarray
        True latent z used to generate Z_obs and y_obs.
    """
    rng = np.random.default_rng(seed)

    # ---- derive structural parameters from targets ----
    if a == 0:
        raise ValueError("Parameter 'a' must be nonzero (b = (m*rho)/a).")
    b = (m * rho) / a
    cprime = (1 - m) * rho

    # residual variances to standardize z and y
    theta_z = 1.0 - a**2
    theta_y = 1.0 - (b**2 + cprime**2 + 2.0 * a * b * cprime)
    if theta_z <= 0:
        raise ValueError("Choose |a| < 1 so Var(z) > 0.")
    if theta_y <= 0:
        raise ValueError("Var(y) would be <= 0. Increase |a| or lower rho/m.")

    # ---- prepare loadings (accept scalar or per-indicator array) ----
    def to_vec(val, k, name):
        if np.isscalar(val):
            vec = np.full(k, float(val))
        else:
            vec = np.asarray(val, dtype=float)
            if vec.shape != (k,):
                raise ValueError(f"'{name}' must be length {k} or a scalar.")
        if np.any(np.abs(vec) > 1):
            raise ValueError(f"All entries of '{name}' must satisfy |loading| <= 1.")
        return vec

    lam_x = to_vec(load_x, p, "load_x")
    lam_z = to_vec(load_z, q, "load_z")

    # indicator residual SDs so each indicator has Var ≈ 1
    sd_eps_x = np.sqrt(1.0 - lam_x**2)
    sd_eps_z = np.sqrt(1.0 - lam_z**2)

    # ---- simulate latents ----
    x_latent = rng.standard_normal(n)                                  # Var ≈ 1
    z_latent = a * x_latent + rng.normal(0.0, np.sqrt(theta_z), size=n)  # Var ≈ 1
    y_obs = b * z_latent + cprime * x_latent + rng.normal(0.0, np.sqrt(theta_y), size=n)  # Var ≈ 1

    # ---- simulate indicators ----
    X_obs = np.column_stack([lam_x[i] * x_latent + rng.normal(0.0, sd_eps_x[i], size=n) for i in range(p)])
    Z_obs = np.column_stack([lam_z[j] * z_latent + rng.normal(0.0, sd_eps_z[j], size=n) for j in range(q)])

    return X_obs, y_obs, Z_obs, x_latent, z_latent


def plot_scalar_X(X_scalar, y, z, title):
    r_xy = pearsonr(X_scalar, y)[0]
    r_xz = pearsonr(X_scalar, z)[0]
    r_yz = pearsonr(y, z)[0]
    r_partial, x_resid, y_resid = partial_correlation(X_scalar, y, z)

    print(f"\n=== {title} ===")
    print(f"corr(X_scalar, y):        {r_xy:.3f}")
    print(f"corr(X_scalar, z):        {r_xz:.3f}")
    print(f"corr(y, z):               {r_yz:.3f}")
    print(f"partial corr(X | z, y):   {r_partial:.3f}")

    df = pd.DataFrame({'X_scalar': X_scalar, 'y': y, 'z': z, 'X_resid': x_resid, 'y_resid': y_resid})
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title)

    sns.regplot(x='y', y='X_scalar', data=df, ax=axs[0, 0], scatter_kws={'alpha': 0.5})
    axs[0, 0].set_title(f'X vs y (r = {r_xy:.2f})')

    sns.regplot(x='z', y='X_scalar', data=df, ax=axs[0, 1], scatter_kws={'alpha': 0.5})
    axs[0, 1].set_title(f'X vs z (r = {r_xz:.2f})')

    sns.regplot(x='z', y='y', data=df, ax=axs[1, 0], scatter_kws={'alpha': 0.5})
    axs[1, 0].set_title(f'y vs z (r = {r_yz:.2f})')

    sns.regplot(x='y_resid', y='X_resid', data=df, ax=axs[1, 1], scatter_kws={'alpha': 0.5})
    axs[1, 1].set_title(f'X vs y | z (partial r = {r_partial:.2f})')

    plt.tight_layout()
    plt.show()


def simulate_independent_sources_predicting_y(n=1000, theta_x=1.0, theta_z=1.0, noise_y=1.0, noise_x=1.0, noise_z=1.0):
    # Independent latent factors
    f1 = np.random.normal(0, 1, n)  # drives X and part of y
    f2 = np.random.normal(0, 1, n)  # drives z and part of y

    # Observed predictors
    X = f1 + np.random.normal(0, noise_x, n)
    z = f2 + np.random.normal(0, noise_z, n)

    # Outcome
    y = theta_x * f1 + theta_z * f2 + np.random.normal(0, noise_y, n)

    return X, y, z
