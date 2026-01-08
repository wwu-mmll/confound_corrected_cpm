import numpy as np
import pytest

from cccpm.models import LinearCPMModel, NetworkDict, ModelDict


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)

    n_samples = 100
    n_features = 20
    n_covariates = 3

    X = rng.normal(size=(n_samples, n_features))
    covariates = rng.normal(size=(n_samples, n_covariates))

    # create target with signal in first 5 features
    y = X[:, :5].sum(axis=1) + 0.5 * covariates[:, 0] + rng.normal(scale=0.1, size=n_samples)

    edges = {
        "positive": np.arange(0, 5),
        "negative": np.arange(5, 10),
    }

    return X, y, covariates, edges


def test_network_dict_structure():
    nd = NetworkDict()
    assert set(nd.keys()) == {"positive", "negative", "both"}
    assert NetworkDict.n_networks() == 3


def test_model_dict_structure():
    md = ModelDict()
    assert set(md.keys()) == {"connectome", "covariates", "full", "residuals"}
    assert ModelDict.n_models() == 4


def test_model_initialization(toy_data):
    _, _, _, edges = toy_data
    model = LinearCPMModel(edges)

    assert isinstance(model.models, ModelDict)
    assert model.edges == edges
    assert model.models_residuals == {}


def test_fit_runs_and_returns_self(toy_data):
    X, y, covariates, edges = toy_data
    model = LinearCPMModel(edges)

    out = model.fit(X, y, covariates)

    assert out is model


def test_fit_creates_models_for_all_networks(toy_data):
    X, y, covariates, edges = toy_data
    model = LinearCPMModel(edges)
    model.fit(X, y, covariates)

    for model_type in ModelDict().keys():
        for network in NetworkDict().keys():
            assert network in model.models[model_type]


def test_predict_shapes(toy_data):
    X, y, covariates, edges = toy_data
    model = LinearCPMModel(edges).fit(X, y, covariates)

    preds = model.predict(X, covariates)

    for model_type in ModelDict().keys():
        for network in NetworkDict().keys():
            assert preds[model_type][network].shape == (X.shape[0],)


def test_predict_returns_modeldict(toy_data):
    X, y, covariates, edges = toy_data
    model = LinearCPMModel(edges).fit(X, y, covariates)

    preds = model.predict(X, covariates)

    assert isinstance(preds, ModelDict)


def test_residuals_uncorrelated_with_covariates(toy_data):
    X, y, covariates, edges = toy_data
    model = LinearCPMModel(edges).fit(X, y, covariates)

    strengths = model.get_network_strengths(X, covariates)
    residuals = strengths["residuals"]["positive"]

    corr = np.corrcoef(residuals[:, 0], covariates[:, 0])[0, 1]
    assert abs(corr) < 0.05


def test_full_model_performs_at_least_as_well_as_connectome(toy_data):
    X, y, covariates, edges = toy_data
    model = LinearCPMModel(edges).fit(X, y, covariates)

    preds = model.predict(X, covariates)

    y_hat_connectome = preds["connectome"]["both"]
    y_hat_full = preds["full"]["both"]

    r2_connectome = np.corrcoef(y, y_hat_connectome)[0, 1] ** 2
    r2_full = np.corrcoef(y, y_hat_full)[0, 1] ** 2

    assert r2_full >= r2_connectome


def test_get_network_strengths_shapes(toy_data):
    X, _, covariates, edges = toy_data
    model = LinearCPMModel(edges).fit(X, np.random.randn(X.shape[0]), covariates)

    strengths = model.get_network_strengths(X, covariates)

    assert "connectome" in strengths
    assert "residuals" in strengths

    for network in ["positive", "negative"]:
        assert strengths["connectome"][network].shape == (X.shape[0], 1)
        assert strengths["residuals"][network].shape == (X.shape[0], 1)
