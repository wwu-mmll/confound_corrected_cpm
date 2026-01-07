import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import pearsonr, spearmanr
from cccpm.edge_selection import (
    pearson_correlation_with_pvalues,
    spearman_correlation_with_pvalues,
    semi_partial_correlation_pearson
)


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
