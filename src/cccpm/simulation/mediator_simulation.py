from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
from numpy.random import Generator
from numpy.linalg import lstsq
from scipy.linalg import toeplitz


def generate_confound_simulation(n_samples=1000, n_features=100,
                                 sparsity=0.8, x_collinearity=0.6,
                                 target_r2=0.36, confounder_rho=0.4,
                                 random_state=None):
    """
    Simulate X, y, and a correlate ``c`` of the outcome.

    .. warning::
        Here ``c = confounder_rho * y + noise`` is generated *from* ``y`` and is
        **not** injected into ``X``. It is therefore a downstream **proxy of the
        outcome**, not a common-cause confounder (there is no ``c -> X`` path).
        Partialling ``c`` out of the X–y relationship removes genuine signal
        (over-correction / collider-like behaviour), so this generator does **not**
        demonstrate confound inflation. For true common-cause confounding with
        analytically known R² use
        :func:`cccpm.simulation.simulate_sem.simulate_data_given_kappa` /
        :func:`~cccpm.simulation.simulate_sem.generate_confound_grid`, or the
        mechanistic :func:`cccpm.simulation.simulate_multivariate.simulate_confounders`.
    """
    if random_state is not None:
        np.random.seed(random_state)
    cov_row = x_collinearity ** np.arange(n_features)
    cov_matrix = toeplitz(cov_row)
    X = np.random.multivariate_normal(np.zeros(n_features), cov_matrix, size=n_samples)
    weights = np.random.randn(n_features)
    n_zero_weights = int(n_features * sparsity)
    zero_indices = np.random.choice(n_features, n_zero_weights, replace=False)
    weights[zero_indices] = 0.0
    signal = X @ weights
    var_signal = np.var(signal)
    if target_r2 == 0:
        y = np.random.randn(n_samples)
    else:
        var_noise = var_signal * ((1.0 - target_r2) / target_r2)
        noise = np.random.normal(0, np.sqrt(var_noise), size=n_samples)
        y = signal + noise
    y = (y - np.mean(y)) / np.std(y)
    noise_c = np.random.normal(0, 1, size=n_samples)
    c = confounder_rho * y + np.sqrt(1 - confounder_rho**2) * noise_c
    return X, y, c
