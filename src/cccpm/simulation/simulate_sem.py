import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def _solve_rho_for_R2(r2_X_y: float, r2_X_y_given_Z: float, r2_Z_y: float,
                      tol: float = 1e-4) -> float:
    """
    Find corr(X, Z) = rho that best matches the desired R2_X_y_given_Z.
    If an exact match is impossible, return the closest achievable rho
    and issue a warning instead of raising an error.
    """
    r_x = np.sqrt(r2_X_y)
    r_z = np.sqrt(r2_Z_y)

    # Special case: Z has no effect → cannot adjust X→y
    if r2_Z_y < tol:
        if abs(r2_X_y_given_Z - r2_X_y) > tol:
            print(
                f"Warning: R2_Z_y≈0, so R2_X_y_given_Z must equal R2_X_y. "
                f"Using R2_X_y_given_Z = {r2_X_y:.4f} instead of {r2_X_y_given_Z:.4f}."
            )
        return 0.0

    # Search over allowable rho values
    rhos = np.linspace(-0.99, 0.99, 20001)
    num = (r_x - rhos * r_z) ** 2
    den = 1 - rhos**2
    vals = num / den

    # Find closest match
    idx = np.argmin(np.abs(vals - r2_X_y_given_Z))
    rho_best = float(rhos[idx])
    val_best = float(vals[idx])

    # If we cannot achieve exact target, warn and continue
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
    """
    Sample the latent triple [true_X, Z_scalar, y] from a 3x3 correlation matrix
    and build the observed features, confounds, and outcome.

    Three interpretable edge classes are constructed from two orthogonal latent
    sources: the confound ``u = Z_scalar`` and the **confound-orthogonalised
    signal** ``s = (true_X - rho*u) / sqrt(1 - rho^2)`` (rho = corr(true_X, u)).
    ``s`` is uncorrelated with ``u`` and has ``corr(s, y) = sqrt(R2_X_y_given_Z)``;
    the full latent ``true_X = rho*u + sqrt(1-rho^2)*s`` mixes the two.

    Feature layout (columns of X), in order:
      * ``mixed`` edges (``n_features_informative``) — noisy copies of ``true_X``
        (loading ``sqrt(rho_informative)``). Carry **both** genuine signal and, when
        rho>0, confound variance (``corr(X_j, u) = sqrt(rho_informative)*rho``).
      * ``pure_signal`` edges (``n_pure_signal_features``) — noisy copies of ``s``
        (loading ``sqrt(rho_pure_signal)``). Related to ``y`` but **uncorrelated with
        the confound** (``corr(X_j, u) = 0``).
      * ``confound_only`` edges (``n_confound_only_features``) — noisy copies of ``u``
        (loading ``sqrt(rho_confound_only)``). Correlated with ``y`` **only through
        the confound**.
      * remaining columns — pure N(0, 1) noise edges.

    This lets an analysis separate what partial-correlation edge selection does:
    it drops ``confound_only`` edges (their partial association with y is null) but
    keeps ``mixed`` edges (which retain a genuine partial association) — whose raw
    values still leak confound variance into the network sum score.

    With ``randomize_signs`` the loading sign of ~half of each non-noise class is
    flipped, so both the positive and negative networks are populated downstream.

    Returns a dict with ``X``, ``Z``, ``y``, ``true_X`` (numpy arrays) and ``info``
    (ground-truth indices and metadata).
    """
    # --- sample (true_X, Z_scalar, y) as multivariate normal ---
    L = np.linalg.cholesky(corr + 1e-8 * np.eye(3))  # jitter for numeric safety
    base = rng.normal(size=(n_samples, 3))
    XYZ = base @ L.T
    true_X = XYZ[:, 0]
    Z_scalar = XYZ[:, 1]
    y = XYZ[:, 2]

    # Confound-orthogonalised (pure) signal: s ⟂ u, unit variance,
    # corr(s, y) = sqrt(R2_X_y_given_Z).
    rho = float(corr[0, 1])
    pure_signal = (true_X - rho * Z_scalar) / np.sqrt(max(1.0 - rho ** 2, 1e-12))

    # --- expand scalar Z into n_confounds (only conf1 carries the confound) ---
    conf_cols = [Z_scalar] + [
        rng.normal(0.0, 1.0, size=n_samples) for _ in range(1, n_confounds)
    ]
    Z = np.column_stack(conf_cols)

    # --- feature-block index layout ---
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

    _fill(mixed_idx, true_X, rho_informative)          # signal + confound
    _fill(pure_signal_idx, pure_signal, rho_pure_signal)  # signal only (⟂ confound)
    _fill(confound_only_idx, Z_scalar, rho_confound_only)  # confound only

    for col in noise_idx:                              # pure noise
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
    """
    Simulate data that matches specified R² relationships between a latent
    predictor X, confounds Z, and outcome y.

    Inputs (targets)
    ----------------
    R2_X_y : float
        Naive R² from regressing y ~ X.
    R2_X_y_given_Z : float
        Unique / incremental R² from X after adjusting for all confounds Z:
            R2_X_y_given_Z = R2(y ~ X + Z) - R2(y ~ Z).
    R2_Z_y : float
        Naive R² from regressing y ~ Z (all confounds jointly).

    Data structure
    --------------
    - latent variable: true_X (standardized)
    - confounds: conf1..confK (K = n_confounds); only conf1 carries the confound
    - features: up to four edge classes — ``mixed`` (copies of true_X; signal +
      confound), optional ``pure_signal`` (copies of the confound-orthogonalised
      signal), optional ``confound_only`` (copies of the confound), and pure-noise
      edges — see ``_build_dataset_from_corr``.
    - outcome: y (standardized)

    R² logic
    --------
    - Naive R²(y ~ X) ≈ R2_X_y
    - Naive R²(y ~ Z) ≈ R2_Z_y
    - Unique R² of X given Z ≈ R2_X_y_given_Z, so R2_full ≈ R2_Z_y + R2_X_y_given_Z

    Returns
    -------
    dict with keys ``X``, ``Z``, ``y``, ``true_X`` (numpy arrays) and ``info``.
    """
    rng = np.random.default_rng(random_state)

    # --- basic validity checks ---
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

    # --- build correlation matrix for [true_X, Z, y] ---
    r_x = np.sqrt(R2_X_y)
    r_z = np.sqrt(R2_Z_y)

    # Solve Corr(X, Z) = rho to match desired R2_X_y_given_Z
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
    """
    Simulate data by sweeping the confound strength as the *spurious fraction*
    ``kappa`` — the fraction of the naive R²(y~X) that is confound-driven.

    Convention (the confound "steals" a fraction ``kappa`` of the brain's
    explained variance, adding no net variance):

        R2_X_y_given_Z = (1 - kappa) * R2_X_y     (true, deconfounded value)
        R2_Z_y         =      kappa  * R2_X_y

    From these the required brain–confound coupling ``rho = corr(confound, X)`` is
    solved internally; it depends only on ``kappa`` (not on ``R2_X_y``):
    ``rho ≈ {0, 0.45, 0.63, 0.77, 0.89, 0.99}`` for ``kappa = {0, .2, .4, .6, .8, 1}``.

    Unlike pinning ``corr(Z,y) = corr(X,y)``, this parameterisation is feasible for
    *all* ``R2_X_y`` (including 0.81) across the whole ``kappa`` sweep.

    ``kappa = 0`` → no confounding (true == naive); ``kappa = 1`` → full confounding
    (true → 0; ``rho`` approaches its clamp at 0.99).

    Parameters
    ----------
    R2_X_y : float
        Naive R²(y ~ X).
    kappa : float
        Spurious fraction in [0, 1].

    Returns
    -------
    dict with keys ``X``, ``Z``, ``y``, ``true_X`` and ``info`` (the info dict also
    records ``R2_X_y``, ``R2_X_y_given_Z``, ``R2_Z_y`` and ``spurious_fraction``).
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
    """
    Convenience function to empirically estimate the R² components from
    the simulated data:

        - r2_naive:      R²(y ~ true_X)
        - r2_conf_only:  R²(y ~ all confounds)
        - r2_full:       R²(y ~ true_X + all confounds)
        - r2_unique_X:   r2_full - r2_conf_only
    """
    y = sim["y"]
    X_naive = sim["true_X"]
    X_conf = sim["Z"]

    # Naive: y ~ true_X
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

    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, 2**32 - 1, size=4)

    scenarios = {}

    # 1. No confounding
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

    # 2. Weak/partial confounding
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

    # 3. Full confounding
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

    # 4. No confounding but Z explains part of y
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


# Default axes for the confound-inflation demonstration.
GRID_R2_X_Y = (0.09, 0.36, 0.81)              # brain–outcome strength (rows)
GRID_KAPPA = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)   # spurious fraction (columns)


def generate_confound_grid(
    r2_x_y_values: tuple = GRID_R2_X_Y,
    kappa_values: tuple = GRID_KAPPA,
    n_features: int = 105,   # 15-node connectome (valid CCCPM edge count)
    n_features_informative: int = 10,   # mixed edges (signal + confound)
    n_confound_only_features: int = 10,  # confound-only edges
    n_pure_signal_features: int = 10,    # pure-signal edges (⟂ confound)
    n_confounds: int = 2,
    n_samples: int = 5000,
    rho_informative: float = 0.5,
    rho_confound_only: float = 0.5,
    rho_pure_signal: float = 0.5,
    randomize_signs: bool = True,
    random_state: int | None = 123,
) -> dict[tuple, dict]:
    """
    Generate the full matrix of confound scenarios for the demonstration.

    Rows sweep the brain–outcome strength ``R2_X_y`` (default {0.09, 0.36, 0.81});
    columns sweep the spurious fraction ``kappa`` (default {0, 0.2, ..., 1.0}) — the
    fraction of the naive R² that is confound-driven (see
    :func:`simulate_data_given_kappa`). The true, deconfounded value in each cell is
    ``R2_X_y_given_Z = (1 - kappa) * R2_X_y``.

    Each cell contains three interpretable edge classes (default 10 each): **mixed**
    (signal + confound), **pure-signal** (⟂ confound) and **confound-only**.

    Returns
    -------
    dict keyed by ``(R2_X_y, kappa)`` → simulated-data dict (``X``, ``Z``, ``y``,
    ``true_X``, ``info``). Each cell gets an independent seed derived from
    ``random_state`` so cells are reproducible and mutually independent.
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


# -------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------
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
