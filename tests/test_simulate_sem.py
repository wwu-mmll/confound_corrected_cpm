import re

import numpy as np
import pytest

from cccpm.simulation.simulate_sem import (
    _solve_rho_for_R2,
    simulate_data_given_R2,
    compute_r2s,
    generate_four_scenarios,
)


def test_solve_rho_zero_confounds():
    rho = _solve_rho_for_R2(
        r2_X_y=0.30,
        r2_X_y_given_Z=0.30,
        r2_Z_y=0.0,
    )
    assert rho == 0.0


def test_simulate_data_deterministic_with_seed():
    sim1 = simulate_data_given_R2(
        0.25, 0.15, 0.10, random_state=123, n_samples=200
    )
    sim2 = simulate_data_given_R2(
        0.25, 0.15, 0.10, random_state=123, n_samples=200
    )

    assert np.allclose(sim1["X"], sim2["X"])
    assert np.allclose(sim1["y"], sim2["y"])


def test_simulated_r2s_match_targets():
    targets = {
        "R2_X_y": 0.25,
        "R2_X_y_given_Z": 0.15,
        "R2_Z_y": 0.10,
    }

    sim = simulate_data_given_R2(
        **targets,
        n_samples=10_000,
        random_state=0,
    )

    r2s = compute_r2s(sim)

    assert np.isclose(r2s["r2_naive"], targets["R2_X_y"], atol=0.02)
    assert np.isclose(r2s["r2_conf_only"], targets["R2_Z_y"], atol=0.02)
    assert np.isclose(
        r2s["r2_unique_X"], targets["R2_X_y_given_Z"], atol=0.02
    )


def test_rejects_invalid_arguments():
    """Every rejected argument combination, and *which* error each produces.

    These were seven test functions asserting only that some ValueError came
    out -- which an unrelated typo raising ValueError would also satisfy. Each
    case now pins the message, so the test fails if the wrong guard fires.
    """
    cases = [
        # (kwargs, expected message fragment)
        (dict(R2_X_y=-0.1, R2_X_y_given_Z=0.0, R2_Z_y=0.0),
         "R2_X_y must be in"),
        (dict(R2_X_y=1.1, R2_X_y_given_Z=0.0, R2_Z_y=0.0),
         "R2_X_y must be in"),
        (dict(R2_X_y=0.2, R2_X_y_given_Z=0.3, R2_Z_y=0.0),
         "cannot exceed R2_X_y"),
        (dict(R2_X_y=0.4, R2_X_y_given_Z=0.3, R2_Z_y=0.8),
         "must be < 1"),
        (dict(R2_X_y=0.2, R2_X_y_given_Z=0.1, R2_Z_y=0.1,
              n_features=5, n_features_informative=10),
         "cannot be greater than n_features"),
        (dict(R2_X_y=0.2, R2_X_y_given_Z=0.1, R2_Z_y=0.1, rho_informative=1.5),
         "rho_informative must be in"),
        (dict(R2_X_y=0.2, R2_X_y_given_Z=0.1, R2_Z_y=0.1, n_confounds=0),
         "n_confounds must be at least 1"),
    ]
    for kwargs, expected in cases:
        with pytest.raises(ValueError, match=re.escape(expected)):
            simulate_data_given_R2(**kwargs)


def test_generate_four_scenarios_r2_ordering():
    scenarios = generate_four_scenarios(n_samples=8000)

    r2s = {
        name: compute_r2s(sim)
        for name, sim in scenarios.items()
    }

    assert (
            r2s["No Confounding Effect"]["r2_unique_X"]
            > r2s["Strong Confounding Effect"]["r2_unique_X"]
    )
