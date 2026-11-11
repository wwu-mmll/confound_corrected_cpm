"""Simple confound-outcome-link simulations for testing deconfounding."""

import numpy as np


def simulate_confounded_data_chyzhyk(link_type='direct_link',
                                     n_samples=100, n_features=100):
    """Simulate confounded data following Chyzhyk et al. (2022).

    Adapted from
    https://github.com/darya-chyzhyk/confound_prediction/blob/master/confound_prediction/data_simulation.py

    Args:
        link_type: One of "no_link", "direct_link", "weak_link".

    Returns:
        ``(X, y, z)``, with ``z`` shaped ``(n_samples, 1)``.
    """
    np.random.seed(42)

    mu, sigma = 0, 1.0
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
    else:
        raise ValueError(
            f"Unknown link_type {link_type!r}. Use 'no_link', 'direct_link' "
            f"or 'weak_link'."
        )
    return X, y, z.reshape(-1, 1)


def simulate_confounded_binary_data_chyzhyk(link_type='direct_link',
                                            n_samples=100, n_features=100):
    """Like ``simulate_confounded_data_chyzhyk``, but ``y`` is binarised at its median."""
    X, y, covariates = simulate_confounded_data_chyzhyk(
        link_type=link_type, n_samples=n_samples, n_features=n_features
    )
    y_binary = (y > np.median(y)).astype(float)
    return X, y_binary, covariates
