import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp, t, rankdata
from typing import Union

from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid
import statsmodels.stats.multitest as multitest
from warnings import filterwarnings


def one_sample_t_test(matrix, population_mean):
    # Calculate the mean and standard deviation along the rows
    sample_means = np.mean(matrix, axis=0)
    sample_stds = np.std(matrix, axis=0, ddof=1)
    n = matrix.shape[1]  # Number of samples in each row

    # Calculate the t-statistics
    filterwarnings('ignore', category=RuntimeWarning)
    t_stats = (sample_means - population_mean) / (sample_stds / np.sqrt(n))

    # Calculate the p-values using the t-distribution survival function
    p_values = 2 * t.sf(np.abs(t_stats), df=n - 1)

    return t_stats, p_values


def compute_t_and_p_values(correlations, df):
    # Calculate t-statistics
    t_stats = correlations * np.sqrt(df / (1 - correlations ** 2))
    # Calculate p-values
    p_values = 2 * t.sf(np.abs(t_stats), df=df)
    return t_stats, p_values


def compute_correlation_and_pvalues(x, Y, rank=False):
    n = len(x)

    if rank:
        x = rankdata(x)
        Y = np.apply_along_axis(rankdata, axis=0, arr=Y)

    # Mean-centering
    x_centered = x - np.mean(x)
    Y_centered = Y - np.mean(Y, axis=0)

    # Correlation calculation
    corr_numerator = np.dot(Y_centered.T, x_centered)
    corr_denominator = (np.sqrt(np.sum(Y_centered ** 2, axis=0)) * np.sqrt(np.sum(x_centered ** 2)))

    correlations = corr_numerator / corr_denominator

    # Calculate t-statistics and p-values
    _, p_values = compute_t_and_p_values(correlations, n - 2)

    return correlations, p_values


def get_residuals(X, Z):
    # Add a column of ones to Z for the intercept
    Z = np.hstack([Z, np.ones((Z.shape[0], 1))])

    # Compute the coefficients using the normal equation
    B = np.linalg.lstsq(Z, X, rcond=None)[0]

    # Predict X from Z
    X_hat = Z.dot(B)

    # Compute residuals
    residuals = X - X_hat

    return residuals


def semi_partial_correlation(x, Y, Z, rank=False):
    if rank:
        x = rankdata(x)
        Y = np.apply_along_axis(rankdata, 0, Y)
        Z = np.apply_along_axis(rankdata, 0, Z)

    # Calculate residuals for x and each column in Y
    x_residuals = get_residuals(x.reshape(-1, 1), Z).ravel()
    Y_residuals = get_residuals(Y, Z)

    # Mean-centering the residuals
    x_centered = x_residuals - np.mean(x_residuals)
    Y_centered = Y_residuals - np.mean(Y_residuals, axis=0)

    # Correlation calculation
    corr_numerator = np.dot(Y_centered.T, x_centered)
    corr_denominator = (np.sqrt(np.sum(Y_centered ** 2, axis=0)) * np.sqrt(np.sum(x_centered ** 2)))
    partial_corr = corr_numerator / corr_denominator

    # Calculate t-statistics and p-values
    n = len(x)
    k = Z.shape[1]
    _, p_values = compute_t_and_p_values(partial_corr, n - k - 2)

    return partial_corr, p_values


def pearson_correlation_with_pvalues(x, Y):
    return compute_correlation_and_pvalues(x, Y, rank=False)


def spearman_correlation_with_pvalues(x, Y):
    return compute_correlation_and_pvalues(x, Y, rank=True)


def semi_partial_correlation_pearson(x, Y, Z):
    return semi_partial_correlation(x, Y, Z, rank=False)


def semi_partial_correlation_spearman(x, Y, Z):
    return semi_partial_correlation(x, Y, Z, rank=True)


class BaseEdgeSelector(BaseEstimator):
    def select(self, r, p):
        pass


class PThreshold(BaseEdgeSelector):
    def __init__(self, threshold: Union[float, list] = 0.05, correction: Union[str, list] = None):
        """

        :param threshold:
        :param correction: can be one of statsmodels methods
                            bonferroni : one-step correction
                            sidak : one-step correction
                            holm-sidak : step down method using Sidak adjustments
                            holm : step-down method using Bonferroni adjustments
                            simes-hochberg : step-up method (independent)
                            hommel : closed method based on Simes tests (non-negative)
                            fdr_bh : Benjamini/Hochberg (non-negative)
                            fdr_by : Benjamini/Yekutieli (negative)
                            fdr_tsbh : two stage fdr correction (non-negative)
                            fdr_tsbky : two stage fdr correction (non-negative)
        """
        self.threshold = threshold
        self.correction = correction

    def select(self, r, p):
        if self.correction is not None:
            _, p, _, _ = multitest.multipletests(p, alpha=0.05, method=self.correction)
        pos_edges = np.where((p < self.threshold) & (r > 0))[0]
        neg_edges = np.where((p < self.threshold) & (r < 0))[0]
        return {'positive': pos_edges, 'negative': neg_edges}


class SelectPercentile(BaseEdgeSelector):
    def __init__(self, percentile: Union[float, list] = 0.05):
        self.percentile = percentile


class SelectKBest(BaseEdgeSelector):
    def __init__(self, k: Union[int, list] = None):
        self.k = k


class UnivariateEdgeSelection(BaseEstimator):
    def __init__(self,
                 edge_statistic: Union[str, list] = 'pearson',
                 edge_selection: list = None,
                 t_test_filter: bool = True):
        self.edge_statistic = edge_statistic
        self.edge_selection = edge_selection
        self.t_test_filter = t_test_filter
        self.param_grid = self._generate_config_grid()

    def _generate_config_grid(self):
        grid_elements = []
        for selector in self.edge_selection:
            params = {}
            params['edge_statistic'] = self.edge_statistic
            params['edge_selection'] = [selector]
            for key, value in selector.get_params().items():
                params['edge_selection__' + key] = value
            grid_elements.append(params)
        return ParameterGrid(grid_elements)

    def fit_transform(self, X, y=None, covariates=None):
        if self.t_test_filter:
            _, p_values = one_sample_t_test(X, 0)
            valid_edges = p_values < 0.05
        else:
            valid_edges = np.bool(np.ones(X.shape[1]))

        r_edges, p_edges = np.zeros(X.shape[1]), np.ones(X.shape[1])
        r_edges_masked, p_edges_masked = self.compute_edge_statistics(X=X[:, valid_edges], y=y, covariates=covariates)
        r_edges[valid_edges] = r_edges_masked
        p_edges[valid_edges] = p_edges_masked

        #r_edges, p_edges = self.compute_edge_statistics(X=X, y=y, covariates=covariates)

        edges = self.edge_selection.select(r=r_edges, p=p_edges)
        return edges

    def compute_edge_statistics(self,
                                X: Union[pd.DataFrame, np.ndarray],
                                y: Union[pd.Series, pd.DataFrame, np.ndarray],
                                covariates: Union[pd.Series, pd.DataFrame, np.ndarray]):
        if self.edge_statistic == 'pearson':
            r_edges, p_edges = pearson_correlation_with_pvalues(y, X)
        elif self.edge_statistic == 'spearman':
            r_edges, p_edges = spearman_correlation_with_pvalues(y, X)
        elif self.edge_statistic == 'pearson_partial':
            r_edges, p_edges = semi_partial_correlation_pearson(y, X, covariates)
        elif self.edge_statistic == 'spearman_partial':
            r_edges, p_edges = semi_partial_correlation_spearman(y, X, covariates)
        else:
            raise NotImplemented("Unsupported edge selection method")

        return r_edges, p_edges
