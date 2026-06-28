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



def simulate_data_given_R2(
    R2_X_y: float,
    R2_X_y_given_Z: float,
    R2_Z_y: float,
    n_features: int = 100,
    n_features_informative: int = 10,
    n_confounds: int = 2,
    n_samples: int = 10_000,
    rho_informative: float = 0.5,
    random_state: int | None = None,
) -> pd.DataFrame:
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
    - confounds: conf1..confK (K = n_confounds)
      * only conf1 is used to build the desired R² (others are noise)
    - features: X1..Xn_features
      * first n_features_informative are informative: correlated with true_X
      * remaining are pure noise
    - outcome: y (standardized)

    R² logic
    --------
    - Naive R²(y ~ X) ≈ R2_X_y
    - Naive R²(y ~ Z) ≈ R2_Z_y
    - Unique R² of X given Z ≈ R2_X_y_given_Z
      so:
          R2_full ≈ R2_Z_y + R2_X_y_given_Z

    Returns
    -------
    df : pandas.DataFrame
        Columns:
            conf1..confK
            X1..Xn_features
            true_X
            y
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
    if n_features_informative > n_features:
        raise ValueError("n_features_informative cannot be greater than n_features.")
    if not (0.0 <= rho_informative < 1.0):
        raise ValueError("rho_informative must be in [0, 1).")
    if n_confounds < 1:
        raise ValueError("n_confounds must be at least 1.")

    # --- step 1: build correlation matrix for (X, Z, y) ---
    r_x = np.sqrt(R2_X_y)
    r_z = np.sqrt(R2_Z_y)

    # Solve Corr(X, Z) = rho to match desired R2_X_y_given_Z
    rho = _solve_rho_for_R2(R2_X_y, R2_X_y_given_Z, R2_Z_y)

    # correlation matrix for [X, Z, y]
    # order: [X, Z, y]
    corr = np.array(
        [
            [1.0,  rho,  r_x],
            [rho,  1.0,  r_z],
            [r_x,  r_z,  1.0],
        ]
    )

    # sanity check: positive semi-definite
    eigvals = np.linalg.eigvalsh(corr)
    if np.min(eigvals) < -1e-6:
        raise ValueError(
            "Requested R² combination leads to an invalid correlation matrix."
        )

    # --- step 2: sample (X, Z, y) as multivariate normal with this corr matrix ---
    L = np.linalg.cholesky(corr + 1e-8 * np.eye(3))  # jitter for numeric safety
    Z_samples = rng.normal(size=(n_samples, 3))
    XYZ = Z_samples @ L.T
    true_X = XYZ[:, 0]
    Z_scalar = XYZ[:, 1]
    y = XYZ[:, 2]

    # Now:
    #   Corr(true_X, y)^2 ≈ R2_X_y
    #   Corr(Z_scalar, y)^2 ≈ R2_Z_y
    # and the full regression y ~ X + Z_scalar has unique R² for X ≈ R2_X_y_given_Z.

    # --- step 3: expand scalar Z into n_confounds (only conf1 matters) ---
    conf_data = {}
    conf_data["conf1"] = Z_scalar
    for i in range(1, n_confounds):
        conf_data[f"conf{i+1}"] = rng.normal(0.0, 1.0, size=n_samples)
    conf_df = pd.DataFrame(conf_data)

    # --- step 4: generate X features ---
    # First n_features_informative are equicorrelated measures of true_X
    # For informative features with Var=1 and pairwise corr=rho_informative:
    #   Xj = sqrt(rho_inf)*true_X + sqrt(1-rho_inf)*N(0,1)
    X_data = {}
    loading = np.sqrt(rho_informative)
    resid_sd = np.sqrt(1.0 - rho_informative)

    for j in range(n_features_informative):
        eps = rng.normal(0.0, resid_sd, size=n_samples)
        X_data[f"X{j+1}"] = loading * true_X + eps

    # Remaining features are pure noise
    for j in range(n_features_informative, n_features):
        X_data[f"X{j+1}"] = rng.normal(0.0, 1.0, size=n_samples)

    X_df = pd.DataFrame(X_data)

    # --- step 5: assemble final DataFrame ---
    df = pd.concat([conf_df, X_df], axis=1)
    df["true_X"] = true_X
    df["y"] = y

    return {'X': X_df.to_numpy(), 'Z': conf_df.to_numpy(), 'y': y.reshape(-1, 1), 'true_X': true_X.reshape(-1, 1)}


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


# -------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    scenarios = generate_four_scenarios(
        n_features=100,
        n_features_informative=10,
        n_confounds=3,
        n_samples=1000,
        rho_informative=0.5,
        random_state=43,
    )

    for name, df in scenarios.items():
        print(f"\nScenario: {name}")
        r2s = compute_r2s(df)
        for k, v in r2s.items():
            print(f"  {k}: {v:.3f}")
