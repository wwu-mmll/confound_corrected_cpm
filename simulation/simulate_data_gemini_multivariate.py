import numpy as np


def analytical_neuro_simulation(n_features=100,
                                r2_true=0.125,  # True Signal (T -> y)
                                r2_confound=0.125,  # Confound (Z -> y)
                                signal_sparsity=0.20):  # 20% of voxels have signal

    # --- 1. DEFINE WEIGHTS (THE "GEN" PART) ---
    # We define how T and Z map onto X (Factor Loadings)

    # A. Signal Weights (Lambda_T)
    # Only 20 voxels have signal
    lambda_t = np.zeros(n_features)
    n_signal = int(n_features * signal_sparsity)
    # Let's give them varying weights (e.g., 0.5 to 0.8)
    lambda_t[:n_signal] = np.linspace(0.5, 0.8, n_signal)

    # B. Confound Weights (Lambda_Z)
    # All voxels have confound (e.g., varying 0.3 to 0.6)
    np.random.seed(42)
    lambda_z = np.random.uniform(0.3, 0.6, n_features)

    # --- 2. CONSTRUCT THEORETICAL COVARIANCE MATRICES ---
    # We don't need to generate data (N=1000). We just use algebra.

    # Variances of Latents are 1.0 (Standardized)
    # Var(T)=1, Var(Z)=1. They are orthogonal (Cov=0).

    # MATRIX 1: Covariance of X (Sigma_XX)
    # X = L_t*T + L_z*Z + E
    # Var(X) = L_t*L_t' + L_z*L_z' + Var(E)

    # Outer products
    cov_signal = np.outer(lambda_t, lambda_t)
    cov_confound = np.outer(lambda_z, lambda_z)

    # Diagonal Noise (Identity matrix for unique voxel noise)
    # We assume standardized voxels, so diagonal must sum to 1.
    # But for simplicity, let's just add independent noise variance = 0.5
    cov_noise = np.eye(n_features) * 0.5

    Sigma_XX = cov_signal + cov_confound + cov_noise

    # MATRIX 2: Covariance of X and y (Sigma_Xy)
    # y = beta_t*T + beta_z*Z
    beta_t = np.sqrt(r2_true)
    beta_z = np.sqrt(r2_confound)

    # Cov(X_i, y) = Cov(lambda_t_i*T + ..., beta_t*T + ...)
    #             = lambda_t_i * beta_t  +  lambda_z_i * beta_z

    Sigma_Xy = (lambda_t * beta_t) + (lambda_z * beta_z)

    # --- 3. CALCULATE NAIVE R2 ---
    # The formula for R2 in OLS regression is: (Sigma_Xy' * Sigma_XX_inv * Sigma_Xy) / Var(y)

    # Invert Sigma_XX
    Sigma_XX_inv = np.linalg.inv(Sigma_XX)

    # Calculate numerator
    explained_variance = Sigma_Xy.T @ Sigma_XX_inv @ Sigma_Xy

    # Total Var(y) is 1.0 (if we assume standardized y)
    naive_r2 = explained_variance / 1.0

    return naive_r2, r2_true


# --- RUN THE CALCULATION ---
naive, true = analytical_neuro_simulation(r2_true=0.125, r2_confound=0.125)

print(f"--- Analytical Results (No Simulation Needed) ---")
print(f"True Target R2 (T -> y): {true}")
print(f"Naive GLM R2   (X -> y): {naive:.4f}")
print(f"Bias due to Confound:    {naive - true:.4f}")

if naive > true:
    print("\nResult: If you run a GLM on this setup, it WILL overestimate the effect.")
    print("This confirms the 'Confounding Trap' mathematically.")