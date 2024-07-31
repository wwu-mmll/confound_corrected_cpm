import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp, pearsonr, t, rankdata
from pingouin import partial_corr

from typing import Union
from sklearn.base import BaseEstimator
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LinearRegression
import statsmodels.stats.multitest as multitest


class BaseEdgeSelector(BaseEstimator):
    def transform(self):
        pass


def one_sample_t_test(x):
    # use two-sided for correlations (functional connectome) or one-sided positive for NOS etc (structural connectome)
    _, p_value = ttest_1samp(x, popmean=0, nan_policy='omit', alternative='two-sided')
    return p_value


def partial_correlation(X, y, covariates, method: str = 'pearson'):
    p_values = list()

    df = pd.concat([pd.DataFrame(X, columns=[f"x{s}" for s in range(X.shape[1])]), pd.DataFrame({'y': y})], axis=1)
    cov_names = list()
    for c in range(covariates.shape[1]):
        df[f'cov{c}'] = covariates[:, c]
        cov_names.append(f'cov{c}')

    for xi in range(X.shape[1]):
        res = partial_corr(data=df, x=f'x{xi}', y='y', covar=cov_names, method=method)['p-val'].iloc[0]
        p_values.append(res)
    return np.asarray(p_values)


def pearson_correlation(x, Y):
    correlations = np.apply_along_axis(lambda col: pearsonr(x, col), 0, Y)
    return correlations[0, :], correlations[1, :]


def pearson_correlation_with_pvalues(x, Y):
    n = len(x)

    # Ensure x is a 1D array
    x = np.ravel(x)

    # Mean-centering x and Y
    x_centered = x - np.mean(x)
    Y_centered = Y - np.mean(Y, axis=0)

    # Correlation calculation
    corr_numerator = np.dot(Y_centered.T, x_centered)
    corr_denominator = (np.sqrt(np.sum(Y_centered ** 2, axis=0)) * np.sqrt(np.sum(x_centered ** 2)))

    correlations = corr_numerator / corr_denominator

    # Calculate t-statistics
    t_stats = correlations * np.sqrt((n - 2) / (1 - correlations ** 2))

    # Calculate p-values
    p_values = 2 * t.sf(np.abs(t_stats), df=n - 2)

    return correlations, p_values


def spearman_correlation_with_pvalues(x, Y):
    n = len(x)

    # Rank the input data
    x_ranked = rankdata(x)
    Y_ranked = np.apply_along_axis(rankdata, axis=0, arr=Y)

    # Mean-centering the ranked data
    x_ranked_centered = x_ranked - np.mean(x_ranked)
    Y_ranked_centered = Y_ranked - np.mean(Y_ranked, axis=0)

    # Correlation calculation
    corr_numerator = np.dot(Y_ranked_centered.T, x_ranked_centered)
    corr_denominator = (np.sqrt(np.sum(Y_ranked_centered ** 2, axis=0)) * np.sqrt(np.sum(x_ranked_centered ** 2)))

    spearman_corr = corr_numerator / corr_denominator

    # Calculate t-statistics
    t_stats = spearman_corr * np.sqrt((n - 2) / (1 - spearman_corr ** 2))

    # Calculate p-values
    p_values = 2 * t.sf(np.abs(t_stats), df=n - 2)

    return spearman_corr, p_values


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


def semi_partial_correlation_pearson(x, Y, Z):
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

    # Calculate t-statistics
    n = len(x)
    t_stats = partial_corr * np.sqrt((n - 2) / (1 - partial_corr ** 2))

    # Calculate p-values
    p_values = 2 * t.sf(np.abs(t_stats), df=n - 2)

    return partial_corr, p_values


def semi_partial_correlation_spearman(x, Y, Z):
    # Calculate residuals for x and each column in Y
    x_residuals = get_residuals(x.reshape(-1, 1), Z).ravel()
    Y_residuals = get_residuals(Y, Z)

    # Rank the residuals
    x_ranked = rankdata(x_residuals)
    Y_ranked = np.apply_along_axis(rankdata, axis=0, arr=Y_residuals)

    # Mean-centering the ranked residuals
    x_centered = x_ranked - np.mean(x_ranked)
    Y_centered = Y_ranked - np.mean(Y_ranked, axis=0)

    # Correlation calculation
    corr_numerator = np.dot(Y_centered.T, x_centered)
    corr_denominator = (np.sqrt(np.sum(Y_centered ** 2, axis=0)) * np.sqrt(np.sum(x_centered ** 2)))

    partial_corr = corr_numerator / corr_denominator

    # Calculate t-statistics
    n = len(x)
    t_stats = partial_corr * np.sqrt((n - 2) / (1 - partial_corr ** 2))

    # Calculate p-values
    p_values = 2 * t.sf(np.abs(t_stats), df=n - 2)

    return partial_corr, p_values


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
        return pos_edges, neg_edges


class SelectPercentile(BaseEdgeSelector):
    def __init__(self, percentile: Union[float, list] = 0.05):
        self.percentile = percentile


class SelectKBest(BaseEdgeSelector):
    def __init__(self, k: Union[int, list] = None):
        self.k = k


class CPMPipeline(_BaseComposition):
    def __init__(self, elements):
        self.elements = elements
        self.current_config = None

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self._get_params('elements', deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of this estimator.
        Valid parameter keys can be listed with ``get_params()``.
        Returns
        -------
        self
        """
        if self.current_config is not None and len(self.current_config) > 0:
            if kwargs is not None and len(kwargs) == 0:
                raise ValueError("Pipeline cannot set parameters to elements with an emtpy dictionary. Old values persist")
        self.current_config = kwargs
        self._set_params('elements', **kwargs)

        return self

    @property
    def named_steps(self):
        return dict(self.elements)


class UnivariateEdgeSelection(BaseEstimator):
    def __init__(self,
                 edge_statistic: Union[str, list] = 'pearson',
                 edge_selection: list = None):
        self.edge_statistic = edge_statistic
        self.edge_selection = edge_selection
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
        r_edges, p_edges = self._correlation(X=X, y=y, covariates=covariates)
        #pos_edges, neg_edges = self._edge_selection(r=r_edges, p=p_edges, threshold=0.01)
        pos_edges, neg_edges = self.edge_selection.select(r=r_edges, p=p_edges)

        return pos_edges, neg_edges

    def _correlation(self, X: Union[pd.DataFrame, np.ndarray],
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

    @staticmethod
    def _edge_selection(r: np.ndarray,
                        p: np.ndarray,
                        threshold: float):
        pos_edges = np.where((p < threshold) & (r > 0))[0]
        neg_edges = np.where((p < threshold) & (r < 0))[0]
        return pos_edges, neg_edges
