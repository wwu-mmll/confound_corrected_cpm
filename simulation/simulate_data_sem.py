import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

np.random.seed(123)
N = 10000  # large n for stable estimates
RHO_AM = 0.3  # correlation between age and motion


# ---------------------------------------------------------
# Helper: simulate age & motion with correlation RHO_AM
# ---------------------------------------------------------
def simulate_age_motion(n=N, rho=RHO_AM):
    cov = np.array([[1.0, rho],
                    [rho, 1.0]])
    mean = np.array([0.0, 0.0])
    age, motion = np.random.multivariate_normal(mean, cov, size=n).T
    return age, motion


# ---------------------------------------------------------
# Helper: generate 10 indicators x1..x10 from latent X
# (optional, just to keep the SEM flavor)
# ---------------------------------------------------------
def generate_indicators(X, loading=0.8, n_indicators=10):
    n = len(X)
    xs = {}
    resid_sd = np.sqrt(1.0 - loading**2)  # so each indicator has Var ≈ 1
    for j in range(1, n_indicators + 1):
        eps = np.random.normal(0.0, resid_sd, size=n)
        xs[f"x{j}"] = loading * X + eps
    return xs  # dict name -> values


# ---------------------------------------------------------
# Scenario 1: No confounding
# ---------------------------------------------------------
def simulate_scenario1(n=N):
    """
    X ~ N(0,1) independent of age & motion
    y = 0.5*X + e, Var(e)=0.75 -> R²(y|X) ≈ 0.25
    age, motion do not influence X or y.
    """
    age, motion = simulate_age_motion(n)
    X = np.random.normal(0.0, 1.0, size=n)

    e_y = np.random.normal(0.0, np.sqrt(0.75), size=n)
    y = 0.5 * X + e_y

    xs = generate_indicators(X)
    data = pd.DataFrame(xs)
    data["true_X"] = X
    data["age"] = age
    data["motion"] = motion
    data["y"] = y
    return data


# ---------------------------------------------------------
# Scenario 2: Partial confounding
# ---------------------------------------------------------
def simulate_scenario2(n=N):
    """
    age & motion -> X
    y depends only on X.
    Parameters chosen so that:
      - Naive R²(y ~ X) ≈ 0.25
      - R²(y ~ age + motion) ≈ 0.125
      - Unique R² of X given confounds ≈ 0.125
    """
    age, motion = simulate_age_motion(n)

    # X = 0.44*age + 0.44*motion + u_X, Var(u_X)≈0.49664 => Var(X)≈1
    u_X = np.random.normal(0.0, np.sqrt(0.49664), size=n)
    X = 0.44 * age + 0.44 * motion + u_X

    # y = 0.5*X + e_y, Var(e_y)=0.75 => R²(y|X)≈0.25
    e_y = np.random.normal(0.0, np.sqrt(0.75), size=n)
    y = 0.5 * X + e_y

    xs = generate_indicators(X)
    data = pd.DataFrame(xs)
    data["true_X"] = X
    data["age"] = age
    data["motion"] = motion
    data["y"] = y
    return data


# ---------------------------------------------------------
# Scenario 3: Full confounding
# ---------------------------------------------------------
def simulate_scenario3(n=N):
    """
    age & motion -> X
    age & motion -> y
    No direct X -> y effect.

    Tuned so that:
      - Naive R²(y ~ X) ≈ 0.25 (spurious)
      - R²(y ~ age + motion) ≈ 0.25
      - Unique R² of X given confounds ≈ 0
    """
    age, motion = simulate_age_motion(n)

    # X fully determined by age & motion: Var(X)≈1
    X = 0.6202 * age + 0.6202 * motion

    # y = 0.3101*age + 0.3101*motion + e_y, Var(e_y)=0.75 -> R²≈0.25
    e_y = np.random.normal(0.0, np.sqrt(0.75), size=n)
    y = 0.3101 * age + 0.3101 * motion + e_y

    xs = generate_indicators(X)
    data = pd.DataFrame(xs)
    data["true_X"] = X
    data["age"] = age
    data["motion"] = motion
    data["y"] = y
    return data


# ---------------------------------------------------------
# Helper: compute the four R²s using plain regressions
# ---------------------------------------------------------
def compute_r2s(df: pd.DataFrame):
    y = df["y"].values.reshape(-1, 1)

    # Naive: y ~ X
    X_naive = df[["true_X"]].values
    mdl = LinearRegression().fit(X_naive, y)
    y_pred_naive = mdl.predict(X_naive)
    r2_naive = r2_score(y, y_pred_naive)

    # Confounds-only: y ~ age + motion
    X_conf = df[["age", "motion"]].values
    mdl = LinearRegression().fit(X_conf, y)
    y_pred_conf = mdl.predict(X_conf)
    r2_conf_only = r2_score(y, y_pred_conf)

    # Full: y ~ X + age + motion
    X_full = df[["true_X", "age", "motion"]].values
    mdl = LinearRegression().fit(X_full, y)
    y_pred_full = mdl.predict(X_full)
    r2_full = r2_score(y, y_pred_full)

    # Unique contribution of X given confounds
    r2_unique_X = r2_full - r2_conf_only

    return {
        "r2_naive": r2_naive,
        "r2_conf_only": r2_conf_only,
        "r2_full": r2_full,
        "r2_unique_X": r2_unique_X,
    }


# ---------------------------------------------------------
# Run all scenarios and print results
# ---------------------------------------------------------
if __name__ == "__main__":
    dat_s1 = simulate_scenario1()
    dat_s2 = simulate_scenario2()
    dat_s3 = simulate_scenario3()

    r2_s1 = compute_r2s(dat_s1)
    r2_s2 = compute_r2s(dat_s2)
    r2_s3 = compute_r2s(dat_s3)

    results = pd.DataFrame(
        [r2_s1, r2_s2, r2_s3],
        index=[
            "scenario1_no_confounding",
            "scenario2_partial_confounding",
            "scenario3_full_confounding",
        ],
    )

    print(results.round(3))
