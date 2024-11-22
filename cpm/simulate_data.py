import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler


def simulate_regression_data(n_samples: int = 500,
                             n_features: int = 4950,
                             n_informative_features: int = 200,
                             n_covariates: int = 5,
                             noise_level: float = 0.1,
                             feature_effect_size: float = 0.2,  # Reduced effect size for features
                             covariate_effect_size: float = 5.0,
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
        feature_effect_size (float): Strength of the association between features and the target variable. Default is 0.2.
        covariate_effect_size (float): Strength of the association between covariates and the target variable. Default is 5.0.
        random_state (int): Seed value used for reproducibility. Default is 42.

    Returns:
        tuple: Tuple containing three NumPy arrays - X, y, and covariates representing features, target, and additional variables respectively.
             - X: Array-like shape=(n_samples, n_features).
             - y: Array-like shape=(n_samples, ).
             - covariates: Array-like shape=(n_samples, n_covariates).
    """

    # Set seed for reproducibility
    np.random.seed(random_state)

    # Generate informative features and random features
    X, y = make_regression(n_samples=n_samples, n_features=n_features,
                           n_informative=n_informative_features, noise=noise_level,
                           random_state=random_state, shuffle=False)

    # Introduce some variability in features
    #X[:, np.arange(0, X.shape[1], 2)] *= -1

    # Generate initial random covariates
    covariates = np.random.rand(n_samples, n_covariates)

    # Define coefficients for covariates based on features to introduce correlation
    feature_to_covariate_coeff = np.random.rand(n_features, n_covariates)
    covariates += np.dot(X, feature_to_covariate_coeff) * 0.2  # Introducing correlation with features

    # Standardize the covariates after introducing correlation
    scaler_cov = StandardScaler()
    covariates = scaler_cov.fit_transform(covariates)

    # Adjust the contribution of features and covariates to the target y
    y += np.dot(X, np.random.rand(n_features)) * feature_effect_size  # Smaller effect size for features
    y += np.dot(covariates, np.random.rand(n_covariates)) * covariate_effect_size  # Covariate effect on y

    # Standardize features X
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)

    # Standardize target y
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    return X, y, covariates


def simulate_regression_data_2(n_samples: int = 500,
                               n_features: int = 4950,
                               n_informative_features: int = 200,
                               link_type: str = 'no_link',
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
        random_state (int): Seed value used for reproducibility. Default is 42.

    Returns:
        tuple: Tuple containing three NumPy arrays - X, y, and z representing features, target, and additional variables respectively.
             - X: Array-like shape=(n_samples, n_features).
             - y: Array-like shape=(n_samples, ).
             - z: Array-like shape=(n_samples, ).
    """

    # Set seed for reproducibility
    np.random.seed(random_state)

    mu, sigma = 0, 1.0  # mean and standard deviation
    x_rand = np.random.normal(mu, sigma, [n_samples, n_features])
    y_rand = np.random.normal(mu, sigma, n_samples).reshape(-1, 1)
    z_rand = np.random.normal(mu, sigma, n_samples).reshape(-1, 1)
    noise = np.random.normal(mu, sigma, n_samples).reshape(-1, 1)
    X = np.copy(x_rand)
    y = np.copy(y_rand)

    if link_type == 'no_link':
        z = 0.4 * y_rand + z_rand
        X[:, :n_informative_features] = x_rand[:, :n_informative_features] + z
    elif link_type == 'no_no_link':
        z = 0.4 * y_rand + z_rand
    elif link_type == 'direct_link':
        z = 0.4 * y_rand + z_rand
        X[:, :n_informative_features] = x_rand[:, :n_informative_features] + (0.2 * (noise * y_rand) + y_rand) + z
    elif link_type == 'weak_link':
        z = 0.4 * y_rand + z_rand
        X[:, :n_informative_features] = x_rand[:, :n_informative_features] + (0.8 * (noise * y_rand) + y_rand) + z
    else:
        raise NotImplemented(f'Link type {link_type} not implemented')

    X[:, :n_informative_features:2] = X[:, :n_informative_features:2] * (-1)

    return X, y, z
