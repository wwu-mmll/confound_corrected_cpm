import os

import numpy as np

from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler


def simulate_data(n_samples: int = 500,
                  n_features: int = 4950,
                  n_informative_features: int = 200,
                  n_covariates: int = 5,
                  noise_level: float = 0.1,
                  random_state: int = 42):
    """
    Simulates data with specified parameters for regression problems.

    This function generates a dataset consisting of features (with some informative),
    covariates, and a target variable. The generated dataset has correlations between
    the features, covariates, and the target variable as per input arguments.

    Args:
        n_samples (int): Number of samples in the dataset. Default is 500.
        n_features (int): Total number of features including irrelevant ones. Default is 4950.
        n_informative_features (int): Number of informative features affecting the target. Default is 200.
        n_covariates (int): Number of covariates which have influence from both features and target. Default is 5.
        noise_level (float): Level of Gaussian noise added to the target variable. Default is 0.1.
        random_state (int): Seed value used for reproducibility. Default is 42.

    Returns:
        tuple: Tuple containing three NumPy arrays - X, y, and covariates representing features, target, and additional variables respectively.
             - X: Array-like shape=(n_samples, n_features).
             - y: Array-like shape=(n_samples, ).
             - covariates: Array-like shape=(n_samples, n_covariates).
    """

    # Parameters
    np.random.seed(random_state)
    # Generate informative features and random features
    X, y = make_regression(n_samples=n_samples, n_features=n_features,
                           n_informative=n_informative_features, noise=noise_level,
                           random_state=random_state, shuffle=False)

    # Generate covariates that are influenced by both features and target
    covariates = np.random.rand(n_samples, n_covariates)  # Initial random covariates

    # Define coefficients for covariates based on features to introduce correlation
    feature_to_covariate_coeff = np.random.rand(n_features, n_covariates)
    covariates += np.dot(X, feature_to_covariate_coeff) * 0.1  # Introducing correlation with features

    # Standardize the covariates after introducing correlation
    scaler = StandardScaler()
    covariates = scaler.fit_transform(covariates)

    # Define coefficients for features and covariates for target association
    feature_coeff = np.random.rand(n_features)
    covariate_coeff = np.random.rand(n_covariates)

    # Create the target associated with features and covariates
    y += np.dot(X, feature_coeff) * 0.5  # Features association
    y += np.dot(covariates, covariate_coeff) * 1.0  # Covariates association

    return X, y, covariates
