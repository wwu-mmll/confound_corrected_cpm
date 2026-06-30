import numpy as np
import pytest

from cccpm.simulation.simulate_sem import (
    _solve_rho_for_R2,
    simulate_data_given_R2,
    compute_r2s,
    generate_four_scenarios,
)


def test_solve_rho_basic_case():
    rho = _solve_rho_for_R2(
        r2_X_y=0.25,
        r2_X_y_given_Z=0.15,
        r2_Z_y=0.10,
    )
    assert -1.0 < rho < 1.0


def test_solve_rho_zero_confounds():
    rho = _solve_rho_for_R2(
        r2_X_y=0.30,
        r2_X_y_given_Z=0.30,
        r2_Z_y=0.0,
    )
    assert rho == 0.0


def test_solve_rho_returns_float():
    rho = _solve_rho_for_R2(0.2, 0.1, 0.1)
    assert isinstance(rho, float)


def test_simulate_data_shapes():
    sim = simulate_data_given_R2(
        R2_X_y=0.25,
        R2_X_y_given_Z=0.15,
        R2_Z_y=0.10,
        n_features=20,
        n_features_informative=5,
        n_confounds=3,
        n_samples=500,
        random_state=42,
    )

    assert sim["X"].shape == (500, 20)
    assert sim["Z"].shape == (500, 3)
    assert sim["y"].shape == (500, 1)
    assert sim["true_X"].shape == (500, 1)


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


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(R2_X_y=-0.1, R2_X_y_given_Z=0.0, R2_Z_y=0.0),
        dict(R2_X_y=1.1, R2_X_y_given_Z=0.0, R2_Z_y=0.0),
        dict(R2_X_y=0.2, R2_X_y_given_Z=0.3, R2_Z_y=0.0),
        dict(R2_X_y=0.4, R2_X_y_given_Z=0.3, R2_Z_y=0.8),
    ],
)
def test_simulate_data_invalid_r2_raises(kwargs):
    with pytest.raises(ValueError):
        simulate_data_given_R2(**kwargs)


def test_informative_features_gt_total_features():
    with pytest.raises(ValueError):
        simulate_data_given_R2(
            0.2, 0.1, 0.1,
            n_features=5,
            n_features_informative=10,
        )


def test_invalid_rho_informative():
    with pytest.raises(ValueError):
        simulate_data_given_R2(
            0.2, 0.1, 0.1,
            rho_informative=1.5,
        )


def test_invalid_n_confounds():
    with pytest.raises(ValueError):
        simulate_data_given_R2(
            0.2, 0.1, 0.1,
            n_confounds=0,
        )


def test_compute_r2s_keys():
    sim = simulate_data_given_R2(
        0.25, 0.15, 0.10, n_samples=500
    )
    r2s = compute_r2s(sim)

    assert set(r2s.keys()) == {
        "r2_naive",
        "r2_conf_only",
        "r2_full",
        "r2_unique_X",
    }


def test_r2_relationships_hold():
    sim = simulate_data_given_R2(
        0.25, 0.15, 0.10, n_samples=5000
    )
    r2s = compute_r2s(sim)

    assert r2s["r2_full"] >= r2s["r2_conf_only"]
    assert r2s["r2_unique_X"] >= 0.0


def test_generate_four_scenarios_keys():
    scenarios = generate_four_scenarios(n_samples=500)

    assert len(scenarios) == 4
    assert "No Confounding Effect" in scenarios
    assert "Strong Confounding Effect" in scenarios


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
