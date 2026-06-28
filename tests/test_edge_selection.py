import numpy as np
import pandas as pd
import pytest
import pingouin as pg
from scipy.stats import pearsonr, spearmanr
from cccpm.edge_selection import (
    pearson_correlation_with_pvalues,
    spearman_correlation_with_pvalues,
    semi_partial_correlation_pearson,
    UnivariateEdgeSelection,
    PThreshold,
)
from cccpm.constants import Networks


def test_cpm_pearson(simulated_data):
    """Test CPM implementation of Pearson correlation with p-values"""
    X, y, _ = simulated_data

    cpm_r, cpm_p = pearson_correlation_with_pvalues(y, X)

    scipy_r = []
    scipy_p = []
    for feature in range(X.shape[1]):
        c, p = pearsonr(X[:, feature], y)
        scipy_r.append(c)
        scipy_p.append(p)

    scipy_r = np.array(scipy_r)
    scipy_p = np.array(scipy_p)

    np.testing.assert_almost_equal(scipy_r, cpm_r, decimal=10)
    np.testing.assert_almost_equal(scipy_p, cpm_p, decimal=10)


def test_cpm_spearman(simulated_data):
    """Test CPM implementation of Spearman correlation with p-values"""
    X, y, _ = simulated_data

    cpm_r, cpm_p = spearman_correlation_with_pvalues(y, X)

    scipy_r = []
    scipy_p = []
    for feature in range(X.shape[1]):
        c, p = spearmanr(X[:, feature], y)
        scipy_r.append(c)
        scipy_p.append(p)

    scipy_r = np.array(scipy_r)
    scipy_p = np.array(scipy_p)

    np.testing.assert_almost_equal(scipy_r, cpm_r, decimal=10)
    np.testing.assert_almost_equal(scipy_p, cpm_p, decimal=10)


def test_semi_partial_correlation_pearson(simulated_data):
    X, y, covariates = simulated_data

    # Calculate partial correlation using the provided function
    partial_corr, p_values = semi_partial_correlation_pearson(y, X, covariates)

    # Calculate partial correlation using pingouin
    # We construct the DF strictly for verification
    cols_y = ["y"]
    cols_x = [f"x{i}" for i in range(X.shape[1])]
    cols_cov = [f"cov{i}" for i in range(covariates.shape[1])]

    df = pd.DataFrame(
        np.column_stack([y, X, covariates]),
        columns=cols_y + cols_x + cols_cov
    )

    pcorr_pingouin = []
    pval_pingouin = []
    for i in range(X.shape[1]):
        result = pg.partial_corr(
            data=df, x="y", y=f"x{i}", covar=cols_cov, method='pearson'
        )
        pcorr_pingouin.append(result['r'].values[0])
        pval_pingouin.append(result['p-val'].values[0])

    pcorr_pingouin = np.array(pcorr_pingouin)
    pval_pingouin = np.array(pval_pingouin)

    np.testing.assert_almost_equal(partial_corr, pcorr_pingouin, decimal=10)
    np.testing.assert_almost_equal(p_values, pval_pingouin, decimal=10)


@pytest.mark.parametrize("statistic,binary_target", [
    ("pearson", False),
    ("spearman", False),
    ("pearson_partial", False),
    ("spearman_partial", False),
    ("point_biserial", True),
    ("point_biserial_partial", True),
])
def test_edge_selection_recovers_signed_edges(statistic, binary_target):
    """
    End-to-end test of the production edge-selection path
    (UnivariateEdgeSelection -> EdgeStatistic dispatch -> PThreshold.select)
    for every supported edge statistic: a strongly positive edge must be
    selected as positive, a strongly negative edge as negative, and the sign
    must not be confused.
    """
    rng = np.random.RandomState(42)
    n_samples, n_features = 200, 10
    X = rng.randn(n_samples, n_features).astype(np.float32)
    # feature 0 drives y up, feature 1 drives y down
    signal = X[:, 0] * 2.0 - X[:, 1] * 2.0 + rng.randn(n_samples) * 0.3
    if binary_target:
        y = (signal > np.median(signal)).astype(np.float32).reshape(-1, 1)
    else:
        y = signal.astype(np.float32).reshape(-1, 1)
    covariates = rng.randn(n_samples, 1).astype(np.float32)

    sel = UnivariateEdgeSelection(
        edge_statistic=statistic,
        edge_selection=[PThreshold(threshold=[0.01], correction=[None])],
    )
    # Configure as a single selector, exactly as the pipeline does
    sel.set_params(**list(sel.param_grid)[0])

    edges = sel.fit_transform(X=X, y=y, covariates=covariates).return_selected_edges()
    # edges shape: [n_features, 2, n_runs]
    assert bool(edges[0, Networks.positive, 0]), f"{statistic}: feature 0 should be positive"
    assert bool(edges[1, Networks.negative, 0]), f"{statistic}: feature 1 should be negative"
    assert not bool(edges[0, Networks.negative, 0]), f"{statistic}: feature 0 should not be negative"
    assert not bool(edges[1, Networks.positive, 0]), f"{statistic}: feature 1 should not be positive"
