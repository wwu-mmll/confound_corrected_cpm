import numpy as np
import pytest

from cccpm.simulation.simulate_simple import simulate_confounded_data_chyzhyk


def test_shapes():
    X, y, z = simulate_confounded_data_chyzhyk(
        link_type="direct_link",
        n_samples=200,
        n_features=50,
    )

    assert X.shape == (200, 50)
    assert y.shape == (200,)
    assert z.shape == (200, 1)


def test_reproducibility():
    X1, y1, z1 = simulate_confounded_data_chyzhyk()
    X2, y2, z2 = simulate_confounded_data_chyzhyk()

    assert np.allclose(X1, X2)
    assert np.allclose(y1, y2)
    assert np.allclose(z1, z2)


@pytest.mark.parametrize("link_type", ["no_link", "direct_link", "weak_link"])
def test_valid_link_types(link_type):
    X, y, z = simulate_confounded_data_chyzhyk(link_type=link_type)

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(z, np.ndarray)


def test_no_link_structure():
    """
    no_link:
    y → z → X
    no direct y → X effect
    """
    X, y, z = simulate_confounded_data_chyzhyk(
        link_type="no_link",
        n_samples=2000,
    )

    # Marginal dependence exists (mediated)
    corr_yX = np.mean(np.abs(np.corrcoef(y, X.T)[0, 1:]))
    assert corr_yX > 0.4

    # Conditional independence: y ⟂ X | z
    y_res = y - np.linalg.lstsq(z, y, rcond=None)[0] * z[:, 0]
    X_res = X - z @ np.linalg.lstsq(z, X, rcond=None)[0]

    corr_res = np.mean(np.abs(np.corrcoef(y_res, X_res.T)[0, 1:]))
    assert corr_res < 0.05


def test_direct_link_structure():
    """
    direct_link:
    y → z
    y → X
    z → X
    """
    X, y, z = simulate_confounded_data_chyzhyk(
        link_type="direct_link",
        n_samples=2000,
    )

    corr_yz = np.corrcoef(y, z[:, 0])[0, 1]
    corr_yX = np.mean(np.abs(np.corrcoef(y, X.T)[0, 1:]))

    assert abs(corr_yz) > 0.5
    assert corr_yX > 0.5


def test_weak_link_has_weaker_yz_than_direct():
    X_d, y_d, z_d = simulate_confounded_data_chyzhyk(
        link_type="direct_link",
        n_samples=2000,
    )

    X_w, y_w, z_w = simulate_confounded_data_chyzhyk(
        link_type="weak_link",
        n_samples=2000,
    )

    corr_direct = abs(np.corrcoef(y_d, z_d[:, 0])[0, 1])
    corr_weak = abs(np.corrcoef(y_w, z_w[:, 0])[0, 1])

    assert corr_direct > corr_weak


def test_features_are_exchangeable():
    """
    No feature should behave differently from others
    """
    X, y, z = simulate_confounded_data_chyzhyk(
        link_type="direct_link",
        n_samples=1000,
        n_features=50,
    )

    corrs = np.abs(np.corrcoef(z[:, 0], X.T)[0, 1:])
    assert np.std(corrs) < 0.05


def test_invalid_link_type_raises():
    with pytest.raises(UnboundLocalError):
        simulate_confounded_data_chyzhyk(link_type="invalid")
