import numpy as np
import pandas as pd
import statsmodels.api as sm


def simulate_sem_data(n_samples, target_total_r2, target_unique_r2, loadings=0.8, seed=None):
    """
    Simulates multivariate X, Z, and univariate y based on R-squared targets.

    Parameters:
    - target_total_r2: The R2 observed in a naive regression (y ~ X)
    - target_unique_r2: The true R2 coming from the causal path (X -> y)
    """
    if seed is not None:
        np.random.seed(seed)

    # --- 1. Derive Path Coefficients ---
    # r_xy_observed = sqrt(Total R2)
    r_xy = np.sqrt(target_total_r2)

    # True beta_yx = sqrt(Unique R2)
    # If unique R2 is 0, beta_yx is 0.
    beta_yx = np.sqrt(target_unique_r2)

    # Calculate Spurious Correlation needed
    # r_xy = beta_yx + bias
    bias = r_xy - beta_yx

    # We split bias symmetrically between Z->X and Z->Y
    # bias = beta_zx * beta_yz
    # If bias is effectively 0, set paths to 0
    if bias < 1e-9:
        beta_zx = 0.0
        beta_yz = 0.0
    else:
        beta_zx = np.sqrt(bias)
        beta_yz = np.sqrt(bias)

    # --- 2. Generate Latent Variables ---
    # Z ~ Normal(0, 1)
    Z_lat = np.random.normal(0, 1, n_samples)

    # X = beta_zx * Z + error
    sd_ex = np.sqrt(1 - beta_zx ** 2)
    X_lat = beta_zx * Z_lat + np.random.normal(0, sd_ex, n_samples)

    # y = beta_yx * X + beta_yz * Z + error
    y_structural = beta_yx * X_lat + beta_yz * Z_lat

    # Scale y error to ensure Var(y) = 1
    var_struct = np.var(y_structural)
    # Safety check if structural variance > 1 (should not happen with these inputs)
    if var_struct >= 1:
        sd_ey = 0
    else:
        sd_ey = np.sqrt(1 - var_struct)

    y = y_structural + np.random.normal(0, sd_ey, n_samples)

    # --- 3. Generate Observed Indicators (Measurement Model) ---
    def make_items(latent, loading, n_items=3):
        items = np.zeros((len(latent), n_items))
        sd_err = np.sqrt(1 - loading ** 2)
        for i in range(n_items):
            items[:, i] = loading * latent + np.random.normal(0, sd_err, len(latent))
        return items

    X_obs = make_items(X_lat, loadings)
    Z_obs = make_items(Z_lat, loadings)

    # Package into DataFrame
    cols = ['y'] + [f'x{i + 1}' for i in range(3)] + [f'z{i + 1}' for i in range(3)]
    data = np.column_stack([y, X_obs, Z_obs])
    df = pd.DataFrame(data, columns=cols)

    # Return df and the true latent scores for verification
    return df, X_lat, Z_lat, beta_yx, beta_zx


def run_scenarios():
    scenarios = [
        {"name": "1. No Confounding", "total_r2": 0.25, "unique_r2": 0.25},
        {"name": "2. Moderate Confounding", "total_r2": 0.25, "unique_r2": 0.125},
        {"name": "3. Strong Confounding", "total_r2": 0.25, "unique_r2": 0.00}
    ]

    results = []

    for sc in scenarios:
        print(f"--- Running: {sc['name']} ---")

        # Simulate Data
        df, X_lat, Z_lat, true_beta, confound_path = simulate_sem_data(
            n_samples=2000,
            target_total_r2=sc['total_r2'],
            target_unique_r2=sc['unique_r2'],
            seed=42
        )

        # Check Naive Regression (y ~ X)
        # Should match Total R2
        X_naive = sm.add_constant(X_lat)
        model_naive = sm.OLS(df['y'], X_naive).fit()
        obs_r2 = model_naive.rsquared
        obs_beta = model_naive.params[1]

        # Check True Regression (y ~ X + Z)
        # Should match Unique R2
        X_true = sm.add_constant(np.column_stack([X_lat, Z_lat]))
        model_true = sm.OLS(df['y'], X_true).fit()
        true_r2_est = model_true.params[1] ** 2  # approx unique variance contribution
        true_beta_est = model_true.params[1]

        print(f"  > Target Total R2: {sc['total_r2']} | Observed Naive R2: {obs_r2:.3f}")
        print(f"  > Target Unique R2:{sc['unique_r2']} | Observed Beta^2:   {true_beta_est ** 2:.3f}")
        print(f"  > Confounder Path (Z->X): {confound_path:.3f}")
        print("-" * 50 + "\n")

        results.append(df)

    return results


# Execute
dfs = run_scenarios()