from typing import Union

from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid
import statsmodels.stats.multitest as multitest

import numpy as np
import torch

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


def compute_t_and_p_values(correlations, df):
    correlations = correlations.to(cuda)
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


def compute_correlation_and_pvalues(x, Y, rank=False):
    """
    x: shape (N, F)  or (samples, features)
    Y: shape (N,)      or (samples,)
    Returns:
        correlations: shape (F,)
        p_values: shape (F,)
    """
    n = x.shape[0]  # number of samples

    if rank:
        x = torch_rankdata(x, axis=0)
        Y = torch_rankdata(Y)

    # Mean-centering
    x_centered = x - x.mean()        # center features
    Y_centered = Y - Y.mean()             # center target

    # Compute correlation
    corr_numerator = torch.sum(x_centered * Y_centered.unsqueeze(1), dim=0)
    corr_denominator = (torch.sqrt(torch.sum(Y_centered ** 2, dim=0))
                        * torch.sqrt(torch.sum(x_centered ** 2)))

    correlations = corr_numerator / corr_denominator

    _, p_values = compute_t_and_p_values(correlations, n - 2)

    return correlations, p_values


    # B, n, p = Y.shape
    # x = x.to(cuda)
    # Y = Y.to(cuda)
    #
    # if rank:
    #     x = x.argsort(dim=1).argsort(dim=1).float().to(cuda)
    #     Y = Y.argsort(dim=1).argsort(dim=1).float().to(cuda)
    #
    # # Mean-centering
    # x_centered = x - x.mean(dim=1, keepdim=True)  # [B, n]
    # Y_centered = Y - Y.mean(dim=1, keepdim=True)  # [B, n, p]
    #
    # # Numerator
    # corr_numerator = torch.einsum('bnp,bn->bp', Y_centered, x_centered)  # [B, p]
    # #corr_numerator = (Y_centered * x_centered.unsqueeze(2)).sum(dim=1)  # same but slower
    #
    # # Denominator
    # x_sq = x_centered.pow(2).sum(dim=1).unsqueeze(1)  # [B, 1]
    # Y_sq = Y_centered.pow(2).sum(dim=1)  # [B, p]
    # corr_denominator = torch.sqrt(x_sq * Y_sq + 1e-8)  # [B, p]
    #
    # correlations = corr_numerator / corr_denominator  # [B, p]
    #
    # # p-value calculation (delegated)
    # dof = n - 2
    # _, p_values = compute_t_and_p_values(correlations, dof)
    #
    #return correlations, p_values


def get_residuals(X, Z):
    Z = torch.hstack([Z, torch.ones((Z.shape[0], 1))])
    B = torch.linalg.lstsq(Z, X, rcond=None)[0]
    X_hat = Z.dot(B)
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
            _, p_corrected, _, _ = multitest.multipletests(
                p.cpu().numpy(), alpha=0.05, method=self._correction
            )
            p = torch.tensor(p_corrected, device=r.device, dtype=r.dtype)
        #p = p.view(-1)
        #r = r.view(-1)

        threshold = float(
            self.threshold[0] if isinstance(self.threshold, (list, np.ndarray))
            else self.threshold
        )
        pos_mask = ((p < threshold) & (r > 0)).to(torch.bool)
        neg_mask = ((p < threshold) & (r < 0)).to(torch.bool)
        return {
            "positive": pos_mask,  # shape: (n_edges,) boolean mask
            "negative": neg_mask,  # shape: (n_edges,)
        }


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
                      X: torch.Tensor,  # [B, n, p] or [n, p]
                      y: torch.Tensor,  # [B, n] or [n]
                      covariates: torch.Tensor  # [B, n, c] or [n, c]
                      ):
        # Ensure tensors
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        if covariates is not None and isinstance(covariates, np.ndarray):
            covariates = torch.from_numpy(covariates).float()

        if self.t_test_filter:
            _, p_values = one_sample_t_test(X, 0)
            valid_edges = p_values < 0.05
        else:
            valid_edges = torch.ones(X.shape[1], dtype=torch.bool, device=X.device)

        if self.edge_statistic == 'pearson':
            r_masked, p_masked = pearson_correlation_with_pvalues(X[:, valid_edges], y)
        elif self.edge_statistic == 'spearman':
            r_masked, p_masked = spearman_correlation_with_pvalues(X[:, valid_edges], y)
        elif self.edge_statistic == 'pearson_partial':
            r_masked, p_masked = semi_partial_correlation_pearson(X[:, valid_edges], y, covariates)
        elif self.edge_statistic == 'spearman_partial':
            r_masked, p_masked = semi_partial_correlation_spearman(X[:, valid_edges], y, covariates)
        else:
            raise NotImplementedError("Unsupported edge selection method")

        return r_masked, p_masked, valid_edges


class UnivariateEdgeSelection(BaseEstimator):
    def __init__(self,
                 edge_statistic: str = 'spearman',
                 t_test_filter: bool = False,
                 edge_selection: list = None,
                 ):
        self.r_edges = None
        self.p_edges = None
        self.valid_edges = None
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
                if isinstance(value, list):
                    params['edge_selection__' + key] = value
                else:
                    params['edge_selection__' + key] = [value]
            grid_elements.append(params)
        return ParameterGrid(grid_elements)

    def fit_transform(self, X, y=None, covariates=None):
        r_masked, p_masked, valid_edges = self.edge_statistic.fit_transform(
            X=X, y=y, covariates=covariates
        )

        self.r_edges = r_masked
        self.p_edges = p_masked
        self.valid_edges = valid_edges

        return r_masked, p_masked, valid_edges

    def return_selected_edges(self, r=None, p=None):
        if r is None: r = self.r_edges
        if p is None: p = self.p_edges
        p = p.to(r.device)
        r = r.to(r.device)
        selected_edges = self.edge_selection.select(r=r, p=p)
        return selected_edges
