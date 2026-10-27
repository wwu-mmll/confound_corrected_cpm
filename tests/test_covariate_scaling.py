"""
Covariates on very different scales must give the same answer as standardized ones.

OLS residualization is invariant to an affine rescaling of the covariates, so a
CPM run with intracranial volume in mm^3 (~1e6) next to 0/1 indicators must
produce exactly the results of the same run with z-scored covariates. In 0.7.0
it did not: the confound regressions (``statistics.get_residuals`` and
``preprocessing.residualize_train_test``) use a float32 pseudo-inverse, whose
default tolerance truncated the small singular directions of the badly
conditioned design -- the intercept and the indicators were never regressed
out, in edge selection or in the residualized connectome, with no error and no
warning. Found on real data (HCP: TIV next to 12 acquisition-period dummies).

``CPMAnalysis(standardize_covariates=True)`` (the default since 0.7.1) z-scores
the covariates inside every outer training fold. The end-to-end test below fails
on 0.7.0.
"""

import json
import os

import numpy as np
import pytest
import torch
from sklearn.model_selection import KFold

from cccpm import CPMAnalysis, UnivariateEdgeSelection, PThreshold
from cccpm.preprocessing import standardize_train_test


def _badly_scaled_data(n=300, n_nodes=10, seed=0):
    """Covariates like HCP's: age, sex, TIV in mm^3, six acquisition dummies.
    Every covariate shifts the edges and the target, so leaving any of them in
    changes which edges are selected and what the model predicts."""
    rng = np.random.default_rng(seed)
    p = n_nodes * (n_nodes - 1) // 2
    age = rng.normal(30, 4, n)
    sex = rng.integers(0, 2, n).astype(float)
    tiv = rng.normal(1.5e6, 1.5e5, n)
    period = rng.integers(0, 7, n)
    dummies = np.eye(7)[period][:, 1:]                       # drop the reference level
    Z = np.column_stack([age, sex, tiv, dummies])
    Z_std = (Z - Z.mean(0)) / Z.std(0)
    loadings = rng.normal(0, 0.6, (Z.shape[1], p))
    signal = rng.normal(size=n)
    X = Z_std @ loadings + 0.5 * signal[:, None] * rng.normal(1, 0.3, p) + rng.normal(size=(n, p))
    y = signal + Z_std @ rng.normal(0, 0.5, Z.shape[1]) + rng.normal(size=n)
    return X, y, Z


def _run(tmp_path, name, X, y, Z, **kwargs):
    cpm = CPMAnalysis(
        results_directory=str(tmp_path / name),
        cv=KFold(n_splits=5, shuffle=True, random_state=0),
        edge_selection=UnivariateEdgeSelection(
            selection_statistic='pearson', selection_input='residualized',
            edge_selection=[PThreshold(threshold=[0.05], correction=[None])]),
        model_input='residualized',
        n_permutations=0,
        **kwargs,
    )
    cpm.run(X=X, y=y, covariates=Z)
    return cpm


def test_covariate_scale_does_not_change_the_results(tmp_path):
    X, y, Z = _badly_scaled_data()
    Z_std = (Z - Z.mean(0)) / Z.std(0)
    raw = _run(tmp_path, "raw_scale", X, y, Z)
    std = _run(tmp_path, "standardized", X, y, Z_std)

    edges_raw = np.load(os.path.join(raw.results_directory, "edges.npy"))
    edges_std = np.load(os.path.join(std.results_directory, "edges.npy"))
    np.testing.assert_array_equal(edges_raw, edges_std)

    pred_raw = raw.results_manager.cv_predictions.sort_values(
        ["sample_index", "model", "network"])["y_pred"].to_numpy()
    pred_std = std.results_manager.cv_predictions.sort_values(
        ["sample_index", "model", "network"])["y_pred"].to_numpy()
    np.testing.assert_allclose(pred_raw, pred_std, atol=1e-3)


def test_residualized_connectome_is_orthogonal_to_raw_scale_covariates(tmp_path):
    """The symptom on real data: with raw-scale covariates the residualized
    connectome's out-of-fold predictions still correlated with what the
    covariates alone predict (.55 in HCP). Deconfounded, they must not."""
    X, y, Z = _badly_scaled_data(n=600)
    cpm = _run(tmp_path, "orthogonal", X, y, Z)
    p = cpm.results_manager.cv_predictions
    p = p[p["network"] == "both"].pivot_table(index="sample_index", columns="model",
                                              values="y_pred")
    r = np.corrcoef(p["connectome"], p["covariates"])[0, 1]
    assert abs(r) < 0.15, r


def test_standardize_train_test_uses_training_statistics_only():
    rng = np.random.default_rng(1)
    train = torch.tensor(np.column_stack([rng.normal(5, 2, 50), np.full(50, 3.0)]),
                         dtype=torch.float32)
    test = torch.tensor(np.column_stack([rng.normal(5, 2, 20), np.full(20, 3.0)]),
                        dtype=torch.float32)
    tr, te = standardize_train_test(train, test)

    torch.testing.assert_close(tr[:, 0].mean(), torch.tensor(0.0), atol=1e-5, rtol=0)
    torch.testing.assert_close(tr[:, 0].std(unbiased=False), torch.tensor(1.0), atol=1e-5, rtol=0)
    mu, sd = train[:, 0].mean(), train[:, 0].std(unbiased=False)
    torch.testing.assert_close(te[:, 0], (test[:, 0] - mu) / sd)
    # A constant column is only centred, never divided by zero.
    assert torch.all(tr[:, 1] == 0) and torch.all(te[:, 1] == 0)


def test_standardize_train_test_accepts_an_empty_design():
    tr, te = standardize_train_test(torch.empty(10, 0), torch.empty(4, 0))
    assert tr.shape == (10, 0) and te.shape == (4, 0)


@pytest.mark.parametrize("flag", [True, False])
def test_the_choice_is_recorded(tmp_path, flag):
    X, y, Z = _badly_scaled_data(n=120)
    cpm = _run(tmp_path, f"flag_{flag}", X, y, Z, standardize_covariates=flag)
    with open(os.path.join(cpm.results_directory, "run_config.json"), encoding="utf-8") as f:
        assert json.load(f)["standardize_covariates"] is flag


def test_standardization_is_the_default():
    import inspect
    default = inspect.signature(CPMAnalysis.__init__).parameters["standardize_covariates"].default
    assert default is True
