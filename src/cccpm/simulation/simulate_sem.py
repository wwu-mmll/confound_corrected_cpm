"""SEM simulations of brain-behavior data with a known signal/confound ground truth."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def _solve_rho_for_R2(r2_X_y: float, r2_X_y_given_Z: float, r2_Z_y: float,
                      tol: float = 1e-4) -> float:
    """Solve corr(X, Z) for the target R2_X_y_given_Z; warns (does not raise) if no exact match exists."""
    r_x = np.sqrt(r2_X_y)
    r_z = np.sqrt(r2_Z_y)

    if r2_Z_y < tol:
        if abs(r2_X_y_given_Z - r2_X_y) > tol:
            print(
                f"Warning: R2_Z_y≈0, so R2_X_y_given_Z must equal R2_X_y. "
                f"Using R2_X_y_given_Z = {r2_X_y:.4f} instead of {r2_X_y_given_Z:.4f}."
            )
        return 0.0

    rhos = np.linspace(-0.99, 0.99, 20001)
    num = (r_x - rhos * r_z) ** 2
    den = 1 - rhos**2
    vals = num / den

    idx = np.argmin(np.abs(vals - r2_X_y_given_Z))
    rho_best = float(rhos[idx])
    val_best = float(vals[idx])

    if abs(val_best - r2_X_y_given_Z) > tol:
        print(
            f"Warning: Could not achieve R2_X_y_given_Z={r2_X_y_given_Z:.4f}. "
            f"Closest possible value is {val_best:.4f}. Using that instead."
        )

    return rho_best



def _build_dataset_from_corr(
    corr: np.ndarray,
    *,
    n_features: int,
    n_features_informative: int,
    n_confound_only_features: int,
    n_pure_signal_features: int,
    n_confounds: int,
    n_samples: int,
    rho_informative: float,
    rho_confound_only: float,
    rho_pure_signal: float,
    randomize_signs: bool,
    rng: np.random.Generator,
    info_extra: dict | None = None,
) -> dict:
    """Sample the latent triple (true_X, Z_scalar, y) and build X/Z/y for the four edge classes.

    ``s = (true_X - rho*u) / sqrt(1 - rho^2)`` (``u = Z_scalar``, ``rho = corr(true_X, u)``)
    is the confound-orthogonalised signal: uncorrelated with the confound, with
    ``corr(s, y) = sqrt(R2_X_y_given_Z)``. Columns of X, in order: ``mixed`` (copies of
    true_X — signal + confound), ``pure_signal`` (copies of s — signal only, ⟂
    confound), ``confound_only`` (copies of u), then pure-noise. This lets partial-
    correlation edge selection drop ``confound_only`` edges while keeping ``mixed``
    edges, whose raw values still leak confound variance into a network sum score.
    With ``randomize_signs``, ~half of each non-noise class has its loading sign
    flipped so both networks are populated.

    Returns:
        dict with ``X``, ``Z``, ``y``, ``true_X`` and ``info``.
    """
    L = np.linalg.cholesky(corr + 1e-8 * np.eye(3))  # jitter for numeric safety
    base = rng.normal(size=(n_samples, 3))
    XYZ = base @ L.T
    true_X = XYZ[:, 0]
    Z_scalar = XYZ[:, 1]
    y = XYZ[:, 2]

    rho = float(corr[0, 1])
    pure_signal = (true_X - rho * Z_scalar) / np.sqrt(max(1.0 - rho ** 2, 1e-12))

    # Only conf1 (Z_scalar) carries the confound; the rest are noise confounds.
    conf_cols = [Z_scalar] + [
        rng.normal(0.0, 1.0, size=n_samples) for _ in range(1, n_confounds)
    ]
    Z = np.column_stack(conf_cols)

    n_mix = n_features_informative
    n_ps = n_pure_signal_features
    n_co = n_confound_only_features
    mixed_idx = np.arange(0, n_mix)
    pure_signal_idx = np.arange(n_mix, n_mix + n_ps)
    confound_only_idx = np.arange(n_mix + n_ps, n_mix + n_ps + n_co)
    noise_idx = np.arange(n_mix + n_ps + n_co, n_features)

    X = np.empty((n_samples, n_features))

    def _signs(k: int) -> np.ndarray:
        if not randomize_signs or k == 0:
            return np.ones(k)
        s = np.ones(k)
        s[: k // 2] = -1.0
        return rng.permutation(s)

    def _fill(idx, latent, communality):
        """X_j = sign * sqrt(communality) * latent + sqrt(1 - communality) * noise."""
        load = np.sqrt(communality)
        resid = np.sqrt(1.0 - communality)
        for col, sign in zip(idx, _signs(len(idx))):
            X[:, col] = sign * load * latent + rng.normal(0.0, resid, size=n_samples)

    _fill(mixed_idx, true_X, rho_informative)
    _fill(pure_signal_idx, pure_signal, rho_pure_signal)
    _fill(confound_only_idx, Z_scalar, rho_confound_only)

    for col in noise_idx:
        X[:, col] = rng.normal(0.0, 1.0, size=n_samples)

    info = {
        "mixed_idx": mixed_idx,
        "pure_signal_idx": pure_signal_idx,
        "confound_only_idx": confound_only_idx,
        "noise_idx": noise_idx,
        # Backwards-compatible alias: "signal" used to mean the true_X-loading edges.
        "signal_idx": mixed_idx,
        "corr_true_X_Z": rho,
        "corr_true_X_y": float(corr[0, 2]),
        "corr_Z_y": float(corr[1, 2]),
        "n_features": n_features,
        "n_features_informative": n_features_informative,
        "n_pure_signal_features": n_pure_signal_features,
        "n_confound_only_features": n_confound_only_features,
        "n_confounds": n_confounds,
        "n_samples": n_samples,
        "rho_informative": rho_informative,
        "rho_pure_signal": rho_pure_signal,
        "rho_confound_only": rho_confound_only,
        "randomize_signs": randomize_signs,
    }
    if info_extra:
        info.update(info_extra)

    return {
        "X": X,
        "Z": Z,
        "y": y.reshape(-1, 1),
        "true_X": true_X.reshape(-1, 1),
        "info": info,
    }


def _validate_feature_args(n_features, n_features_informative,
                           n_confound_only_features, n_pure_signal_features,
                           n_confounds, rho_informative, rho_confound_only,
                           rho_pure_signal):
    total = (n_features_informative + n_confound_only_features
             + n_pure_signal_features)
    if total > n_features:
        raise ValueError(
            "n_features_informative + n_confound_only_features + "
            "n_pure_signal_features cannot be greater than n_features."
        )
    for name, val in [("rho_informative", rho_informative),
                      ("rho_confound_only", rho_confound_only),
                      ("rho_pure_signal", rho_pure_signal)]:
        if not (0.0 <= val < 1.0):
            raise ValueError(f"{name} must be in [0, 1).")
    if n_confounds < 1:
        raise ValueError("n_confounds must be at least 1.")


def simulate_data_given_R2(
    R2_X_y: float,
    R2_X_y_given_Z: float,
    R2_Z_y: float,
    n_features: int = 100,
    n_features_informative: int = 10,
    n_confound_only_features: int = 0,
    n_pure_signal_features: int = 0,
    n_confounds: int = 2,
    n_samples: int = 10_000,
    rho_informative: float = 0.5,
    rho_confound_only: float = 0.5,
    rho_pure_signal: float = 0.5,
    randomize_signs: bool = True,
    random_state: int | None = None,
) -> dict:
    """Simulate data matching target R² relationships between latent X, confounds Z, and y.

    Naive R²(y~X) ≈ R2_X_y, naive R²(y~Z) ≈ R2_Z_y, and the unique R² of X given Z
    (``R2(y~X+Z) - R2(y~Z)``) ≈ R2_X_y_given_Z, so R2_full ≈ R2_Z_y + R2_X_y_given_Z.
    See ``_build_dataset_from_corr`` for the edge-class/feature layout.
    """
    rng = np.random.default_rng(random_state)

    for name, val in [
        ("R2_X_y", R2_X_y),
        ("R2_X_y_given_Z", R2_X_y_given_Z),
        ("R2_Z_y", R2_Z_y),
    ]:
        if not (0.0 <= val < 1.0):
            raise ValueError(f"{name} must be in [0,1), got {val}.")

    if R2_X_y_given_Z > R2_X_y + 1e-8:
        raise ValueError(
            "R2_X_y_given_Z cannot exceed R2_X_y (unique effect cannot be "
            "larger than naive effect)."
        )
    if R2_Z_y + R2_X_y_given_Z >= 1.0:
        raise ValueError(
            "R2_Z_y + R2_X_y_given_Z must be < 1 (total R² cannot exceed 1)."
        )
    _validate_feature_args(n_features, n_features_informative,
                           n_confound_only_features, n_pure_signal_features,
                           n_confounds, rho_informative, rho_confound_only,
                           rho_pure_signal)

    r_x = np.sqrt(R2_X_y)
    r_z = np.sqrt(R2_Z_y)
    rho = _solve_rho_for_R2(R2_X_y, R2_X_y_given_Z, R2_Z_y)

    corr = np.array(
        [
            [1.0,  rho,  r_x],
            [rho,  1.0,  r_z],
            [r_x,  r_z,  1.0],
        ]
    )

    eigvals = np.linalg.eigvalsh(corr)
    if np.min(eigvals) < -1e-6:
        raise ValueError(
            "Requested R² combination leads to an invalid correlation matrix."
        )

    return _build_dataset_from_corr(
        corr,
        n_features=n_features,
        n_features_informative=n_features_informative,
        n_confound_only_features=n_confound_only_features,
        n_pure_signal_features=n_pure_signal_features,
        n_confounds=n_confounds,
        n_samples=n_samples,
        rho_informative=rho_informative,
        rho_confound_only=rho_confound_only,
        rho_pure_signal=rho_pure_signal,
        randomize_signs=randomize_signs,
        rng=rng,
        info_extra={
            "R2_X_y": R2_X_y,
            "R2_X_y_given_Z": R2_X_y_given_Z,
            "R2_Z_y": R2_Z_y,
            "random_state": random_state,
        },
    )


def simulate_data_given_kappa(
    R2_X_y: float,
    kappa: float,
    n_features: int = 100,
    n_features_informative: int = 10,
    n_confound_only_features: int = 10,
    n_pure_signal_features: int = 10,
    n_confounds: int = 2,
    n_samples: int = 10_000,
    rho_informative: float = 0.5,
    rho_confound_only: float = 0.5,
    rho_pure_signal: float = 0.5,
    randomize_signs: bool = True,
    random_state: int | None = None,
) -> dict:
    """Simulate data by sweeping confound strength via the spurious fraction ``kappa``.

    ``kappa`` is the fraction of naive R²(y~X) that is confound-driven: the confound
    "steals" ``R2_Z_y = kappa*R2_X_y``, leaving ``R2_X_y_given_Z = (1-kappa)*R2_X_y``
    as the true, deconfounded value. The required ``rho = corr(confound, X)`` is
    solved internally from ``kappa`` alone (not ``R2_X_y``), so — unlike pinning
    ``corr(Z,y) = corr(X,y)`` — this is feasible for any ``R2_X_y`` across the whole
    sweep. ``kappa=0``: no confounding; ``kappa=1``: full confounding (``rho``
    clamped near 0.99).
    """
    if not (0.0 <= R2_X_y < 1.0):
        raise ValueError(f"R2_X_y must be in [0,1), got {R2_X_y}.")
    if not (0.0 <= kappa <= 1.0):
        raise ValueError(f"kappa must be in [0, 1], got {kappa}.")

    R2_X_y_given_Z = (1.0 - kappa) * R2_X_y
    R2_Z_y = kappa * R2_X_y

    sim = simulate_data_given_R2(
        R2_X_y=R2_X_y,
        R2_X_y_given_Z=R2_X_y_given_Z,
        R2_Z_y=R2_Z_y,
        n_features=n_features,
        n_features_informative=n_features_informative,
        n_confound_only_features=n_confound_only_features,
        n_pure_signal_features=n_pure_signal_features,
        n_confounds=n_confounds,
        n_samples=n_samples,
        rho_informative=rho_informative,
        rho_confound_only=rho_confound_only,
        rho_pure_signal=rho_pure_signal,
        randomize_signs=randomize_signs,
        random_state=random_state,
    )
    sim["info"]["spurious_fraction"] = kappa
    return sim


def compute_r2s(sim: dict) -> dict:
    """Empirically estimate r2_naive, r2_conf_only, r2_full and r2_unique_X from simulated data."""
    y = sim["y"]
    X_naive = sim["true_X"]
    X_conf = sim["Z"]

    mdl_naive = LinearRegression().fit(X_naive, y)
    r2_naive = r2_score(y, mdl_naive.predict(X_naive))

    # Confounds-only: y ~ conf1..confK
    mdl_conf = LinearRegression().fit(X_conf, y)
    r2_conf_only = r2_score(y, mdl_conf.predict(X_conf))

    # Full: y ~ true_X + conf1..confK
    X_full = np.column_stack([X_naive, X_conf])
    mdl_full = LinearRegression().fit(X_full, y)
    r2_full = r2_score(y, mdl_full.predict(X_full))

    r2_unique_X = r2_full - r2_conf_only

    return {
        "r2_naive": r2_naive,
        "r2_conf_only": r2_conf_only,
        "r2_full": r2_full,
        "r2_unique_X": r2_unique_X,
    }


def generate_four_scenarios(
    n_features: int = 100,
    n_features_informative: int = 10,
    n_confounds: int = 2,
    n_samples: int = 10_000,
    rho_informative: float = 0.5,
    random_state: int | None = 123,
) -> dict[str, pd.DataFrame]:
    """Generate four preset confounding scenarios, keyed by scenario name."""
    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, 2**32 - 1, size=4)

    scenarios = {}

    scenarios["No Confounding Effect"] = simulate_data_given_R2(
        R2_X_y=0.25,
        R2_X_y_given_Z=0.25,
        R2_Z_y=0.0,
        n_features=n_features,
        n_features_informative=n_features_informative,
        n_confounds=n_confounds,
        n_samples=n_samples,
        rho_informative=rho_informative,
        random_state=int(seeds[0]),
    )

    scenarios["Moderate Confounding Effect"] = simulate_data_given_R2(
        R2_X_y=0.25,
        R2_X_y_given_Z=0.15,
        R2_Z_y=0.10,
        n_features=n_features,
        n_features_informative=n_features_informative,
        n_confounds=n_confounds,
        n_samples=n_samples,
        rho_informative=rho_informative,
        random_state=int(seeds[1]),
    )

    scenarios["Strong Confounding Effect"] = simulate_data_given_R2(
        R2_X_y=0.25,
        R2_X_y_given_Z=0.05,
        R2_Z_y=0.20,
        n_features=n_features,
        n_features_informative=n_features_informative,
        n_confounds=n_confounds,
        n_samples=n_samples,
        rho_informative=rho_informative,
        random_state=int(seeds[2]),
    )

    scenarios["No Confounding Effect But Useful Confounds"] = simulate_data_given_R2(
        R2_X_y=0.25,
        R2_X_y_given_Z=0.25,
        R2_Z_y=0.25,
        n_features=n_features,
        n_features_informative=n_features_informative,
        n_confounds=n_confounds,
        n_samples=n_samples,
        rho_informative=rho_informative,
        random_state=int(seeds[3]),
    )
    return scenarios


# Default grid axes: rows sweep R2_X_y, columns sweep kappa.
GRID_R2_X_Y = (0.09, 0.36, 0.81)
GRID_KAPPA = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def generate_confound_grid(
    r2_x_y_values: tuple = GRID_R2_X_Y,
    kappa_values: tuple = GRID_KAPPA,
    n_features: int = 105,  # 15-node connectome (valid CCCPM edge count)
    n_features_informative: int = 10,
    n_confound_only_features: int = 10,
    n_pure_signal_features: int = 10,
    n_confounds: int = 2,
    n_samples: int = 5000,
    rho_informative: float = 0.5,
    rho_confound_only: float = 0.5,
    rho_pure_signal: float = 0.5,
    randomize_signs: bool = True,
    random_state: int | None = 123,
) -> dict[tuple, dict]:
    """Generate the full (R2_X_y, kappa) matrix of confound scenarios via ``simulate_data_given_kappa``.

    Each cell gets an independent seed derived from ``random_state``.
    """
    rng = np.random.default_rng(random_state)
    grid = {}
    for r2 in r2_x_y_values:
        for kappa in kappa_values:
            seed = int(rng.integers(0, 2**32 - 1))
            grid[(r2, kappa)] = simulate_data_given_kappa(
                R2_X_y=r2,
                kappa=kappa,
                n_features=n_features,
                n_features_informative=n_features_informative,
                n_confound_only_features=n_confound_only_features,
                n_pure_signal_features=n_pure_signal_features,
                n_confounds=n_confounds,
                n_samples=n_samples,
                rho_informative=rho_informative,
                rho_confound_only=rho_confound_only,
                rho_pure_signal=rho_pure_signal,
                randomize_signs=randomize_signs,
                random_state=seed,
            )
    return grid


if __name__ == "__main__":
    grid = generate_confound_grid(n_samples=5000, random_state=43)

    print(f"{'R2_X_y':>8} {'kappa':>6} {'rho':>6} {'true R2':>9} "
          f"{'r2_naive':>9} {'r2_unique_X':>12}")
    for (r2, kappa), sim in grid.items():
        info = sim["info"]
        r2s = compute_r2s(sim)
        print(f"{r2:>8.2f} {kappa:>6.1f} {info['corr_true_X_Z']:>6.2f} "
              f"{info['R2_X_y_given_Z']:>9.3f} {r2s['r2_naive']:>9.3f} "
              f"{r2s['r2_unique_X']:>12.3f}")
