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

def simulate_scalar_X_with_target_corr(n, alpha, beta, gamma, sigma_z=1.0, rho_target=0.5):
    np.random.seed(42)
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
    X_scalar = signal + eps_x
    X_scalar = (X_scalar - np.mean(X_scalar)) / np.std(X_scalar)

    return X_scalar, y, z, signal

def build_multivariate_X_from_scalar(X_scalar, n_features=105, n_signal_features=15):
    n = len(X_scalar)
    X = np.zeros((n, n_features))

    # Normalize scalar to unit variance
    X_scalar = (X_scalar - np.mean(X_scalar)) / np.std(X_scalar)

    for i in range(n_signal_features):
        noise = np.random.normal(0, 1, n)
        signal_component = np.sqrt(0.5) * X_scalar
        noise_component = np.sqrt(0.5) * noise
        if i % 2 == 0:
            X[:, i] = signal_component + noise_component
        else:
            X[:, i] = -signal_component + noise_component

    for i in range(n_signal_features, n_features):
        X[:, i] = np.random.normal(0, 1, n)

    return X

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
