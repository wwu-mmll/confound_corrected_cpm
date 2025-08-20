import numpy as np


def simulate_confounded_data_chyzhyk(link_type='direct_link',
                                     n_samples=100, n_features=100):
    """
    This is code by Darya Chyzhyk et al. 2022
    https://github.com/darya-chyzhyk/confound_prediction/blob/master/confound_prediction/data_simulation.py
    :param link_type: str,
        Type of the links between target and confound. Options: "no_link",
        "direct_link", "weak_link"
    :param n_samples: int,
        number of samples
    :param n_features: int,
        number of features
    :return:
    """
    np.random.seed(42)

    mu, sigma = 0, 1.0  # mean and standard deviation
    x_rand = np.random.normal(mu, sigma, [n_samples, n_features])
    y_rand = np.random.normal(mu, sigma, n_samples)
    z_rand = np.random.normal(mu, sigma, n_samples)

    if link_type == 'no_link':
        y = np.copy(y_rand)
        z = 1 * y_rand + z_rand
        X = x_rand + z.reshape(-1, 1)
    elif link_type == 'direct_link':
        y = np.copy(y_rand)
        z = y_rand + z_rand
        X = x_rand + y_rand.reshape(-1, 1) + z.reshape(-1, 1)
    elif link_type == 'weak_link':
        y = np.copy(y_rand)
        z = 0.5 * y_rand + z_rand
        X = x_rand + y_rand.reshape(-1, 1) + z.reshape(-1, 1)
    return X, y, z


def simulate_regression_data_scenarios(n_samples: int = 500,
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
