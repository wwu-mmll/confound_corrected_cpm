# import pandas as pd
# import numpy as np
# from scipy.stats import ttest_1samp, t, rankdata
# from typing import Union
#
# from sklearn.base import BaseEstimator
# from sklearn.model_selection import ParameterGrid
# import statsmodels.stats.multitest as multitest
# from warnings import filterwarnings
# import torch
#
# cuda = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#
# def beta_func(a, b):
#     a = torch.as_tensor(a)
#     b = torch.as_tensor(b)
#     dev = a.device
#     if not b.device == a.device:
#         b.to(dev)
#     return torch.exp(torch.lgamma(a.to(dev)) + torch.lgamma(b.to(dev)) - torch.lgamma((a + b).to(dev)))
#
#
# def incomplete_beta(x, a, b, n=1000) -> torch.Tensor:
#     x = x.clamp(min=1e-8, max=1 - 1e-8)
#     device = x.device
#     a = torch.as_tensor(a, dtype=torch.float32, device=device)
#     b = torch.as_tensor(b, dtype=torch.float32, device=device)
#
#     t = torch.linspace(0, 1, n, device=device).unsqueeze(1)  # [n, 1]
#     # Broadcasting fix
#     t = t.view(-1, 1, 1)  # [n, 1, 1]
#     x = x.unsqueeze(0)  # [1, B, p]
#     t_scaled = t * x  # [n, B, p]
#     t_scaled = t_scaled.permute(1, 2, 0)  # [B, p, n]
#     #t_scaled = t * x.unsqueeze(0)  # [n, d]
#
#     integrand = (t_scaled ** (a - 1)) * ((1 - t_scaled) ** (b - 1))
#     dx = x / (n - 1)
#     integral = torch.sum(integrand, dim=2) * dx
#     return integral
#
#
# def regularized_incomplete_beta(x, a, b, n=1000):
#     B = beta_func(a, b)
#     inc_B = incomplete_beta(x, a, b, n)
#     return inc_B / B
#
#
# def student_t_cdf(t, df, n=1000):
#     device = t.device
#     x = df / (df + t ** 2)
#     a = df / 2
#     b = 0.5
#
#     I = regularized_incomplete_beta(x, a, b, n=n)
#     cdf = torch.where(t >= 0, 1 - 0.5 * I, 0.5 * I)
#     return cdf
#
#
# def one_sample_t_test(matrix, population_mean):
#     matrix = torch.tensor(matrix, dtype=torch.float32, device=cuda)
#     population_mean = torch.tensor(population_mean, dtype=torch.float32, device=cuda)
#
#     sample_means = matrix.mean(dim=0)
#     sample_stds = matrix.std(dim=0, unbiased=True)
#     n = matrix.shape[0]
#
#     t_stats = (sample_means - population_mean) / (
#                 sample_stds / torch.sqrt(torch.tensor(n, dtype=torch.float32, device=cuda)))
#     df = n - 1
#     cdf_vals = student_t_cdf(torch.abs(t_stats.to(cuda)), df)
#     p_values = 2 * (1 - cdf_vals)
#     return t_stats, p_values
#
#
# # def one_sample_t_test(matrix, population_mean):
# #     # Calculate the mean and standard deviation along the rows
# #     sample_means = np.mean(matrix, axis=0)
# #     sample_stds = np.std(matrix, axis=0, ddof=1)
# #     n = matrix.shape[1]  # Number of samples in each row
# #
# #     # Calculate the t-statistics
# #     filterwarnings('ignore', category=RuntimeWarning)
# #     t_stats = (sample_means - population_mean) / (sample_stds / np.sqrt(n))
# #
# #     # Calculate the p-values using the t-distribution survival function
# #     p_values = 2 * t.sf(np.abs(t_stats), df=n - 1)
# #
# #     return t_stats, p_values
#
#
# def compute_t_and_p_values(correlations, df) -> tuple[torch.Tensor, torch.Tensor]:
#     correlations = correlations
#     df = torch.tensor(df, dtype=torch.float32, device=cuda)
#
#     t_stats = correlations * torch.sqrt(df / (1 - correlations ** 2))
#     cdf_vals = student_t_cdf(torch.abs(t_stats.to(cuda)), df)
#     p_values = 2 * (1 - cdf_vals)
#     return t_stats, p_values
#
#
# def torch_rankdata(x: torch.Tensor, axis=None):
#     def rank1d(a):
#         argsort = torch.argsort(a)
#         ranks = torch.zeros_like(a, dtype=torch.float)
#         sorted_a = a[argsort]
#         unique, inverse, counts = torch.unique(sorted_a, return_inverse=True, return_counts=True)
#         cumsum = counts.cumsum(0)
#         ends = cumsum
#         starts = cumsum - counts
#         avg_ranks = (starts + ends - 1).float() / 2 + 1  # 1-based rank
#         for idx, r in enumerate(avg_ranks):
#             ranks[argsort[inverse == idx]] = r
#         return ranks
#
#     if axis is None:
#         return rank1d(x.flatten()).reshape(x.shape)
#     elif axis == 0:
#         return torch.stack([rank1d(col) for col in x.T], dim=1)
#     elif axis == 1:
#         return torch.stack([rank1d(row) for row in x], dim=0)
#     else:
#         raise ValueError("axis must be None, 0, or 1")
#
#
# def compute_correlation_and_pvalues(x, Y, rank=False, valid_edges=None):
#     B, n, p = Y.shape
#
#     if rank:
#         x = x.argsort(dim=1).argsort(dim=1).float()
#         Y = Y.argsort(dim=1).argsort(dim=1).float()
#
#     # Mean-centering
#     x_centered = x - x.mean(dim=1, keepdim=True)  # [B, n]
#     Y_centered = Y - Y.mean(dim=1, keepdim=True)  # [B, n, p]
#
#     # Numerator
#     corr_numerator = torch.einsum('bnp,bn->bp', Y_centered, x_centered)  # [B, p]
#
#     # Denominator
#     x_sq = x_centered.pow(2).sum(dim=1).unsqueeze(1)  # [B, 1]
#     Y_sq = Y_centered.pow(2).sum(dim=1)  # [B, p]
#     corr_denominator = torch.sqrt(x_sq * Y_sq + 1e-8)  # [B, p]
#
#     correlations = corr_numerator / corr_denominator  # [B, p]
#
#     # p-value calculation (delegated)
#     dof = n - 2
#     _, p_values = compute_t_and_p_values(correlations, dof)
#
#     return correlations, p_values
#
#
# def get_residuals(X, Z):
#     # Add a column of ones to Z for the intercept
#     Z = torch.hstack([Z, torch.ones((Z.shape[0], 1))])
#
#     # Compute the coefficients using the normal equation
#     B = torch.linalg.lstsq(Z, X, rcond=None)[0]
#
#     # Predict X from Z
#     X_hat = Z.dot(B)
#
#     # Compute residuals
#     residuals = X - X_hat
#
#     return residuals
#
#
# def semi_partial_correlation(x, Y, Z, rank=False):
#     if rank:
#         x = torch_rankdata(x)
#         Y = torch_rankdata(Y, axis=0)
#         Z = torch_rankdata(Z, axis=0)
#
#         #Y = np.apply_along_axis(rankdata, 0, Y)
#         #Z = np.apply_along_axis(rankdata, 0, Z)
#
#     # Calculate residuals for x and each column in Y
#     x_residuals = get_residuals(x.reshape(-1, 1), Z).ravel()
#     Y_residuals = get_residuals(Y, Z)
#
#     # Mean-centering the residuals
#     x_centered = x_residuals - torch.mean(x_residuals)
#     Y_centered = Y_residuals - torch.mean(Y_residuals, axis=0)
#
#     # Correlation calculation
#     corr_numerator = torch.dot(Y_centered.T, x_centered)
#     corr_denominator = (torch.sqrt(torch.sum(Y_centered ** 2, axis=0)) * torch.sqrt(torch.sum(x_centered ** 2)))
#     partial_corr = corr_numerator / corr_denominator
#
#     # Calculate t-statistics and p-values
#     n = len(x)
#     k = Z.shape[1]
#     _, p_values = compute_t_and_p_values(partial_corr, n - k - 2)
#
#     return partial_corr, p_values
#
#
# def pearson_correlation_with_pvalues(x, Y, valid_edges):
#     return compute_correlation_and_pvalues(x, Y, rank=False, valid_edges=valid_edges)
#
#
# def spearman_correlation_with_pvalues(x, Y, valid_edges):
#     return compute_correlation_and_pvalues(x, Y, rank=True, valid_edges=valid_edges)
#
#
# def semi_partial_correlation_pearson(x, Y, Z):
#     return semi_partial_correlation(x, Y, Z, rank=False)
#
#
# def semi_partial_correlation_spearman(x, Y, Z):
#     return semi_partial_correlation(x, Y, Z, rank=True)
#
#
# class BaseEdgeSelector(BaseEstimator):
#     def select(self, r, p):
#         pass
#
#
# class PThreshold(BaseEdgeSelector):
#     def __init__(self, threshold: Union[float, list] = 0.05, correction: Union[str, list] = None):
#         """
#
#         :param threshold:
#         :param correction: can be one of statsmodels methods
#                             bonferroni : one-step correction
#                             sidak : one-step correction
#                             holm-sidak : step down method using Sidak adjustments
#                             holm : step-down method using Bonferroni adjustments
#                             simes-hochberg : step-up method (independent)
#                             hommel : closed method based on Simes tests (non-negative)
#                             fdr_bh : Benjamini/Hochberg (non-negative)
#                             fdr_by : Benjamini/Yekutieli (negative)
#                             fdr_tsbh : two stage fdr correction (non-negative)
#                             fdr_tsbky : two stage fdr correction (non-negative)
#         """
#         self._threshold = None
#         self._correction = None
#         self.threshold = threshold
#         self.correction = correction
#
#     @property
#     def threshold(self):
#         if isinstance(self._threshold, (int, float)):
#             return [float(self._threshold)]
#         return self._threshold or [0.05]
#
#     @threshold.setter
#     def threshold(self, value):
#         if isinstance(value, (int, float)):
#             self._threshold = float(value)
#         elif isinstance(value, list):
#             self._threshold = value
#         else:
#             raise ValueError("threshold must be float or list")
#
#     @property
#     def correction(self):
#         if self._correction is None:
#             return [None]
#         if isinstance(self._correction, str):
#             return [self._correction]
#         return self._correction
#
#     @correction.setter
#     def correction(self, value):
#         if value is None:
#             self._correction = None
#         elif isinstance(value, str):
#             self._correction = value
#         elif isinstance(value, list):
#             self._correction = value
#         else:
#             raise ValueError("correction must be None, str, or list")
#
#     def select(self, r, p):
#         if self._correction is not None:
#             _, p, _, _ = multitest.multipletests(p, alpha=0.05, method=self._correction)
#         pos_edges = torch.where((p < self.threshold) & (r > 0))[0]
#         neg_edges = torch.where((p < self.threshold) & (r < 0))[0]
#         return {'positive': pos_edges, 'negative': neg_edges}
#
#
# class SelectPercentile(BaseEdgeSelector):
#     def __init__(self, percentile: Union[float, list] = 0.05):
#         self.percentile = percentile
#
#
# class SelectKBest(BaseEdgeSelector):
#     def __init__(self, k: Union[int, list] = None):
#         self.k = k
#
#
# class EdgeStatistic(BaseEstimator):
#     def __init__(self, edge_statistic: str = 'spearman', t_test_filter: bool = False):
#         self.edge_statistic = edge_statistic
#         self.t_test_filter = t_test_filter
#
#     def fit_transform(self,
#                       X: torch.Tensor,  # [B, n, p]
#                       y: torch.Tensor,  # [B, n]
#                       covariates: torch.Tensor  # [B, n, c]
#                       ):
#         B, n, p = X.shape
#
#         if self.t_test_filter:
#             _, p_values = one_sample_t_test(X, 0)  # [B, p]
#             valid_edges = p_values < 0.05  # [B, p]
#         else:
#             valid_edges = torch.ones(B, p, dtype=torch.bool, device=X.device)
#
#         if self.edge_statistic == 'pearson':
#             r_masked, p_masked = pearson_correlation_with_pvalues(y, X, valid_edges=valid_edges)  # [B, p]
#         elif self.edge_statistic == 'spearman':
#             r_masked, p_masked = spearman_correlation_with_pvalues(y, X, valid_edges=valid_edges)
#         elif self.edge_statistic == 'pearson_partial':
#             r_masked, p_masked = semi_partial_correlation_pearson(y, X, covariates)
#         elif self.edge_statistic == 'spearman_partial':
#             r_masked, p_masked = semi_partial_correlation_spearman(y, X, covariates)
#         else:
#             raise NotImplementedError("Unsupported edge selection method")
#
#         # Store masked correlations directly
#         r_edges = r_masked
#         p_edges = p_masked
#
#         return r_edges, p_edges  # shape: [B, p]
#
#
# class UnivariateEdgeSelection(BaseEstimator):
#     def __init__(self,
#                  edge_statistic: str = 'spearman',
#                  t_test_filter: bool = False,
#                  edge_selection: list = None,
#                  ):
#         self.r_edges = None
#         self.p_edges = None
#         self.t_test_filter = t_test_filter
#         self.edge_statistic = EdgeStatistic(edge_statistic=edge_statistic, t_test_filter=t_test_filter)
#         self.edge_selection = edge_selection
#         self.param_grid = self._generate_config_grid()
#
#     def _generate_config_grid(self):
#         grid_elements = []
#         for selector in self.edge_selection:
#             params = {}
#             params['edge_selection'] = [selector]
#             for key, value in selector.get_params().items():
#                 params['edge_selection__' + key] = value
#             grid_elements.append(params)
#         return ParameterGrid(grid_elements)
#
#     def fit_transform(self, X, y=None, covariates=None):
#         self.r_edges, self.p_edges = self.edge_statistic.fit_transform(X=X, y=y, covariates=covariates)
#         return self.r_edges, self.p_edges
#
#     def return_selected_edges(self):
#         selected_edges = self.edge_selection.select(r=self.r_edges, p=self.p_edges)
#         return selected_edges

from typing import Union

from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid
import statsmodels.stats.multitest as multitest
from warnings import filterwarnings

import torch
from torch.distributions import StudentT
from torch import erf, sqrt

cuda = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def beta_func(a, b):
    a = torch.as_tensor(a)
    b = torch.as_tensor(b)
    dev = cuda
    if not b.device == cuda or not a.device == cuda:
        b.to(dev)
        a.to(dev)
    return torch.exp(torch.lgamma(a) + torch.lgamma(b) - torch.lgamma((a + b)))


def incomplete_beta(x, a, b, n=1000) -> torch.Tensor:
    x = x.clamp(min=1e-8, max=1 - 1e-8)
    device = cuda
    a = torch.as_tensor(a, dtype=torch.float32, device=device)
    b = torch.as_tensor(b, dtype=torch.float32, device=device)

    t = torch.linspace(0, 1, n, device=device).unsqueeze(1)  # [n, 1]
    # Broadcasting fix
    t = t.view(-1, 1, 1)  # [n, 1, 1]
    x = x.unsqueeze(0)  # [1, B, p]
    t_scaled = t * x  # [n, B, p]
    t_scaled = t_scaled.permute(1, 2, 0)  # [B, p, n]
    # t_scaled = t * x.unsqueeze(0)  # [n, d]

    integrand = (t_scaled ** (a - 1)) * ((1 - t_scaled) ** (b - 1))
    dx = x / (n - 1)
    integral = torch.sum(integrand, dim=2) * dx
    return integral


def regularized_incomplete_beta(x, a, b, n=1000):
    B = beta_func(a, b)
    inc_B = incomplete_beta(x, a, b, n)
    return inc_B / B


def student_t_cdf(t, df, n=1000):
    x = df / (df + t ** 2)
    a = df / 2
    b = 0.5

    I = regularized_incomplete_beta(x, a, b, n=n)
    cdf = torch.where(t >= 0, 1 - 0.5 * I, 0.5 * I)
    return cdf


def one_sample_t_test(matrix, population_mean):
    matrix = torch.tensor(matrix, dtype=torch.float32, device=cuda)
    population_mean = torch.tensor(population_mean, dtype=torch.float32, device=cuda)

    sample_means = matrix.mean(dim=0)
    sample_stds = matrix.std(dim=0, unbiased=True)
    n = matrix.shape[0]

    t_stats = (sample_means - population_mean) / (
            sample_stds / torch.sqrt(torch.tensor(n, dtype=torch.float32, device=cuda)))
    df = n - 1
    cdf_vals = student_t_cdf(torch.abs(t_stats), df)
    p_values = 2 * (1 - cdf_vals)
    return t_stats, p_values


# def one_sample_t_test(matrix, population_mean):
#     # Calculate the mean and standard deviation along the rows
#     sample_means = np.mean(matrix, axis=0)
#     sample_stds = np.std(matrix, axis=0, ddof=1)
#     n = matrix.shape[1]  # Number of samples in each row
#
#     # Calculate the t-statistics
#     filterwarnings('ignore', category=RuntimeWarning)
#     t_stats = (sample_means - population_mean) / (sample_stds / np.sqrt(n))
#
#     # Calculate the p-values using the t-distribution survival function
#     p_values = 2 * t.sf(np.abs(t_stats), df=n - 1)
#
#     return t_stats, p_values


# def compute_t_and_p_values(correlations, df) -> tuple[torch.Tensor, torch.Tensor]:
#     correlations = correlations.to(cuda)
#     df = torch.tensor(df, dtype=torch.float32, device=cuda)
#
#     t_stats = correlations * torch.sqrt(df / (1 - correlations ** 2))
#     cdf_vals = student_t_cdf(torch.abs(t_stats), df)
#     p_values = 2 * (1 - cdf_vals)
#     return t_stats, p_values


def compute_t_and_p_values(correlations, df):
    # Ensure df is a tensor on the same device as correlations
    correlations = correlations.to(cuda)
    print(correlations.device)
    df = torch.tensor(df, dtype=correlations.dtype, device=cuda)

    # Compute t-statistics
    t_stats = correlations * torch.sqrt(df / (1 - correlations ** 2))

    # Crude normal approximation for p-values
    z = t_stats / torch.sqrt(df / (df + 1))
    p_values = 2 * (0.5 * (1 + torch.erf(-torch.abs(z) / torch.sqrt(torch.tensor(2.0, device=cuda)))))

    return t_stats, p_values


def torch_rankdata(x: torch.Tensor, axis=None):
    def rank1d(a):
        argsort = torch.argsort(a)
        ranks = torch.zeros_like(a, dtype=torch.float)
        sorted_a = a[argsort]
        unique, inverse, counts = torch.unique(sorted_a, return_inverse=True, return_counts=True)
        cumsum = counts.cumsum(0)
        ends = cumsum
        starts = cumsum - counts
        avg_ranks = (starts + ends - 1).float() / 2 + 1  # 1-based rank
        for idx, r in enumerate(avg_ranks):
            ranks[argsort[inverse == idx]] = r
        return ranks

    if axis is None:
        return rank1d(x.flatten()).reshape(x.shape)
    elif axis == 0:
        return torch.stack([rank1d(col) for col in x.T], dim=1)
    elif axis == 1:
        return torch.stack([rank1d(row) for row in x], dim=0)
    else:
        raise ValueError("axis must be None, 0, or 1")


def compute_correlation_and_pvalues(x, Y, rank=False, valid_edges=None):
    B, n, p = Y.shape
    x = x.to(cuda)
    Y = Y.to(cuda)

    if rank:
        x = x.argsort(dim=1).argsort(dim=1).float().to(cuda)
        Y = Y.argsort(dim=1).argsort(dim=1).float().to(cuda)

    # Mean-centering
    x_centered = x - x.mean(dim=1, keepdim=True)  # [B, n]
    Y_centered = Y - Y.mean(dim=1, keepdim=True)  # [B, n, p]

    # Numerator
    corr_numerator = torch.einsum('bnp,bn->bp', Y_centered, x_centered)  # [B, p]
    #corr_numerator = (Y_centered * x_centered.unsqueeze(2)).sum(dim=1)  # same but slower

    # Denominator
    x_sq = x_centered.pow(2).sum(dim=1).unsqueeze(1)  # [B, 1]
    Y_sq = Y_centered.pow(2).sum(dim=1)  # [B, p]
    corr_denominator = torch.sqrt(x_sq * Y_sq + 1e-8)  # [B, p]

    correlations = corr_numerator / corr_denominator  # [B, p]

    # p-value calculation (delegated)
    dof = n - 2
    _, p_values = compute_t_and_p_values(correlations, dof)

    return correlations, p_values


def get_residuals(X, Z):
    # Add a column of ones to Z for the intercept
    Z = torch.hstack([Z, torch.ones((Z.shape[0], 1))])

    # Compute the coefficients using the normal equation
    B = torch.linalg.lstsq(Z, X, rcond=None)[0]

    # Predict X from Z
    X_hat = Z.dot(B)

    # Compute residuals
    residuals = X - X_hat

    return residuals


def semi_partial_correlation(x, Y, Z, rank=False):
    if rank:
        x = torch_rankdata(x)
        Y = torch_rankdata(Y, axis=0)
        Z = torch_rankdata(Z, axis=0)

        # Y = np.apply_along_axis(rankdata, 0, Y)
        # Z = np.apply_along_axis(rankdata, 0, Z)

    # Calculate residuals for x and each column in Y
    x_residuals = get_residuals(x.reshape(-1, 1), Z).ravel()
    Y_residuals = get_residuals(Y, Z)

    # Mean-centering the residuals
    x_centered = x_residuals - torch.mean(x_residuals)
    Y_centered = Y_residuals - torch.mean(Y_residuals, axis=0)

    # Correlation calculation
    corr_numerator = torch.dot(Y_centered.T, x_centered)
    corr_denominator = (torch.sqrt(torch.sum(Y_centered ** 2, axis=0)) * torch.sqrt(torch.sum(x_centered ** 2)))
    partial_corr = corr_numerator / corr_denominator

    # Calculate t-statistics and p-values
    n = len(x)
    k = Z.shape[1]
    _, p_values = compute_t_and_p_values(partial_corr, n - k - 2)

    return partial_corr, p_values


def pearson_correlation_with_pvalues(x, Y, valid_edges):
    return compute_correlation_and_pvalues(x, Y, rank=False, valid_edges=valid_edges)


def spearman_correlation_with_pvalues(x, Y, valid_edges):
    return compute_correlation_and_pvalues(x, Y, rank=True, valid_edges=valid_edges)


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
        self._threshold = None
        self._correction = None
        self.threshold = threshold
        self.correction = correction

    @property
    def threshold(self):
        if isinstance(self._threshold, (int, float)):
            return [float(self._threshold)]
        return self._threshold or [0.05]

    @threshold.setter
    def threshold(self, value):
        if isinstance(value, (int, float)):
            self._threshold = float(value)
        elif isinstance(value, list):
            self._threshold = value
        else:
            raise ValueError("threshold must be float or list")

    @property
    def correction(self):
        if self._correction is None:
            return [None]
        if isinstance(self._correction, str):
            return [self._correction]
        return self._correction

    @correction.setter
    def correction(self, value):
        if value is None:
            self._correction = None
        elif isinstance(value, str):
            self._correction = value
        elif isinstance(value, list):
            self._correction = value
        else:
            raise ValueError("correction must be None, str, or list")

    def select(self, r, p):
        if self._correction is not None:
            _, p, _, _ = multitest.multipletests(p, alpha=0.05, method=self._correction)
        pos_edges = torch.where((p < self.threshold) & (r > 0))[0]
        neg_edges = torch.where((p < self.threshold) & (r < 0))[0]
        return {'positive': pos_edges, 'negative': neg_edges}


class SelectPercentile(BaseEdgeSelector):
    def __init__(self, percentile: Union[float, list] = 0.05):
        self.percentile = percentile


class SelectKBest(BaseEdgeSelector):
    def __init__(self, k: Union[int, list] = None):
        self.k = k


class EdgeStatistic(BaseEstimator):
    def __init__(self, edge_statistic: str = 'spearman', t_test_filter: bool = False):
        self.edge_statistic = edge_statistic
        self.t_test_filter = t_test_filter

    def fit_transform(self,
                      X: torch.Tensor,  # [B, n, p]
                      y: torch.Tensor,  # [B, n]
                      covariates: torch.Tensor  # [B, n, c]
                      ):
        B, n, p = X.shape

        # r_edges = torch.zeros(B, p, device=X.device)
        # p_edges = torch.ones(B, p, device=X.device)

        if self.t_test_filter:
            _, p_values = one_sample_t_test(X, 0)  # [B, p]
            valid_edges = p_values < 0.05  # [B, p]
        else:
            valid_edges = torch.ones(B, p, dtype=torch.bool, device=cuda)

        if self.edge_statistic == 'pearson':
            r_masked, p_masked = pearson_correlation_with_pvalues(y, X, valid_edges=valid_edges)  # [B, p]
        elif self.edge_statistic == 'spearman':
            r_masked, p_masked = spearman_correlation_with_pvalues(y, X, valid_edges=valid_edges)
        elif self.edge_statistic == 'pearson_partial':
            r_masked, p_masked = semi_partial_correlation_pearson(y, X, covariates)
        elif self.edge_statistic == 'spearman_partial':
            r_masked, p_masked = semi_partial_correlation_spearman(y, X, covariates)
        else:
            raise NotImplementedError("Unsupported edge selection method")

        # Store masked correlations directly
        r_edges = r_masked
        p_edges = p_masked

        return r_edges, p_edges  # shape: [B, p]

    @staticmethod
    def edge_statistic_fn(Xb, yb, covb, edge_statistic: str = 'spearman', t_test_filter: bool = False):
        n, p = Xb.shape
        _, c = covb.shape

        if t_test_filter:
            _, p_values = one_sample_t_test(Xb.unsqueeze(0), 0)  # Output: [1, p]
            valid_edges = p_values[0] < 0.05  # [p]
        else:
            valid_edges = torch.ones(p, dtype=torch.bool, device=cuda)

        if edge_statistic == 'pearson':
            r, pval = pearson_correlation_with_pvalues(yb.unsqueeze(0), Xb.unsqueeze(0), valid_edges.unsqueeze(0))
        elif edge_statistic == 'spearman':
            r, pval = spearman_correlation_with_pvalues(yb.unsqueeze(0), Xb.unsqueeze(0), valid_edges.unsqueeze(0))
        elif edge_statistic == 'pearson_partial':
            r, pval = semi_partial_correlation_pearson(yb.unsqueeze(0), Xb.unsqueeze(0), covb.unsqueeze(0))
        elif edge_statistic == 'spearman_partial':
            r, pval = semi_partial_correlation_spearman(yb.unsqueeze(0), Xb.unsqueeze(0), covb.unsqueeze(0))
        else:
            raise NotImplementedError("Unsupported edge selection method")

        return r.squeeze(0), pval.squeeze(0)  # [p], [p]


class UnivariateEdgeSelection(BaseEstimator):
    def __init__(self,
                 edge_statistic: str = 'spearman',
                 t_test_filter: bool = False,
                 edge_selection: list = None,
                 ):
        self.r_edges = None
        self.p_edges = None
        self.t_test_filter = t_test_filter
        self.edge_statistic = EdgeStatistic(edge_statistic=edge_statistic, t_test_filter=t_test_filter)
        self.edge_selection = edge_selection
        self.param_grid = self._generate_config_grid()

    def _generate_config_grid(self):
        grid_elements = []
        for selector in self.edge_selection:
            params = {}
            params['edge_selection'] = [selector]
            for key, value in selector.get_params().items():
                params['edge_selection__' + key] = value
            grid_elements.append(params)
        return ParameterGrid(grid_elements)

    def fit_transform(self, X, y=None, covariates=None):
        self.r_edges, self.p_edges = self.edge_statistic.fit_transform(X=X, y=y, covariates=covariates)
        return self.r_edges, self.p_edges

    def return_selected_edges(self):
        selected_edges = self.edge_selection.select(r=self.r_edges, p=self.p_edges)
        return selected_edges
